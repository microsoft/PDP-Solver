# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# pdp_predict.py : Defines various predictors and scorers for the PDP framework.

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pytorch.PDP import util

###############################################################
### The Predictor Classes
###############################################################


class NeuralPredictor(nn.Module):
    "Implements the neural predictor."

    def __init__(self, device, decimator_dimension, prediction_dimension, 
        edge_dimension, meta_data_dimension, mem_hidden_dimension, agg_hidden_dimension, mem_agg_hidden_dimension,
        variable_classifier=None, function_classifier=None):

        super(NeuralPredictor, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()

        self._variable_classifier = variable_classifier
        self._function_classifier = function_classifier

        if variable_classifier is not None:
            self._variable_aggregator = util.MessageAggregator(device, decimator_dimension + edge_dimension + meta_data_dimension, 
                decimator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, 0, include_self_message=True)

            self._module_list.append(self._variable_aggregator)
            self._module_list.append(self._variable_classifier)

        if function_classifier is not None:
            self._function_aggregator = util.MessageAggregator(device, decimator_dimension + edge_dimension + meta_data_dimension, 
                decimator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, 0, include_self_message=True)

            self._module_list.append(self._function_aggregator)
            self._module_list.append(self._function_classifier)

    def forward(self, decimator_state, sat_problem, last_call=False):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = sat_problem._batch_mask_tuple

        variable_prediction = None
        function_prediction = None

        if sat_problem._meta_data is not None:
            graph_feat = torch.mm(b_variable_mask, sat_problem._meta_data)
            graph_feat = torch.mm(variable_mask_transpose, graph_feat)

        if len(decimator_state) == 3:
            decimator_variable_state, decimator_function_state, edge_mask = decimator_state
        else:
            decimator_variable_state, decimator_function_state = decimator_state
            edge_mask = None

        if self._variable_classifier is not None:

            aggregated_variable_state = torch.cat((decimator_variable_state, sat_problem._edge_feature), 1)

            if sat_problem._meta_data is not None:
                aggregated_variable_state = torch.cat((aggregated_variable_state, graph_feat), 1)

            aggregated_variable_state = self._variable_aggregator(
                aggregated_variable_state, None, variable_mask, variable_mask_transpose, edge_mask)

            variable_prediction = self._variable_classifier(aggregated_variable_state)

        if self._function_classifier is not None:

            aggregated_function_state = torch.cat((decimator_function_state, sat_problem._edge_feature), 1)

            if sat_problem._meta_data is not None:
                aggregated_function_state = torch.cat((aggregated_function_state, graph_feat), 1)

            aggregated_function_state = self._function_aggregator(
                aggregated_function_state, None, function_mask, function_mask_transpose, edge_mask)

            function_prediction = self._function_classifier(aggregated_function_state)

        return variable_prediction, function_prediction


###############################################################


class IdentityPredictor(nn.Module):
    "Implements the Identity predictor (prediction based on the assignments to the solution property of the SAT problem)."
    
    def __init__(self, device, random_fill=False):
        super(IdentityPredictor, self).__init__()
        self._random_fill = random_fill
        self._device = device

    def forward(self, decimator_state, sat_problem, last_call=False):
        pred = sat_problem._solution.unsqueeze(1)

        if self._random_fill and last_call:
            active_var_num = (sat_problem._active_variables[:, 0] > 0).long().sum()

            if active_var_num > 0:
                pred[sat_problem._active_variables[:, 0] > 0, 0] = \
                    torch.rand(active_var_num.item(), device=self._device)

        return pred, None


###############################################################


class SurveyScorer(nn.Module):
    "Implements the varaible scoring mechanism for SP-guided decimation."

    def __init__(self, device, message_dimension, include_adaptors=False, pi=0.0):
        super(SurveyScorer, self).__init__()
        self._device = device
        self._include_adaptors = include_adaptors
        self._eps = torch.tensor([1e-10], device=self._device)
        self._max_logit = torch.tensor([30.0], device=self._device)
        self._pi = torch.tensor([pi], dtype=torch.float32, device=device)

        if self._include_adaptors:
            self._projector = nn.Linear(message_dimension, 2, bias=False)
            self._module_list = nn.ModuleList([self._projector])

    def safe_log(self, x):
        return torch.max(x, self._eps).log()

    def safe_exp(self, x):
        return torch.min(x, self._max_logit).exp()

    def forward(self, message_state, sat_problem, last_call=False):
        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, _, _, _ = sat_problem._batch_mask_tuple
        p_variable_mask, _, _, _ = sat_problem._pos_mask_tuple
        n_variable_mask, _, _, _ = sat_problem._neg_mask_tuple

        if self._include_adaptors:
            function_message = self._projector(message_state[1])
            function_message[:, 0] = F.sigmoid(function_message[:, 0])
            function_message[:, 1] = torch.sign(function_message[:, 1])
        else:
            function_message = message_state[1]

        external_force = torch.sign(torch.mm(variable_mask, function_message[:, 1].unsqueeze(1)))
        function_message = self.safe_log(1 - function_message[:, 0]).unsqueeze(1)

        edge_mask = torch.mm(function_mask_transpose, sat_problem._active_functions)
        function_message = function_message * edge_mask

        pos = torch.mm(p_variable_mask, function_message) + self.safe_log(1.0 - self._pi * (external_force == 1).float())
        neg = torch.mm(n_variable_mask, function_message) + self.safe_log(1.0 - self._pi * (external_force == -1).float())

        pos_neg_sum = pos + neg

        dont_care = torch.mm(variable_mask, function_message) + self.safe_log(1.0 - self._pi)

        bias = (2 * pos_neg_sum + dont_care) / 4.0
        pos = pos - bias
        neg = neg - bias
        pos_neg_sum = pos_neg_sum - bias
        dont_care = self.safe_exp(dont_care - bias)

        q_0 = self.safe_exp(pos) - self.safe_exp(pos_neg_sum)
        q_1 = self.safe_exp(neg) - self.safe_exp(pos_neg_sum)

        total = self.safe_log(q_0 + q_1 + dont_care)

        return self.safe_exp(self.safe_log(q_1) - total) - self.safe_exp(self.safe_log(q_0) - total), None

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication):

        edge_num = graph_map.size(1) * batch_replication

        if randomized:
            variable_state = torch.rand(edge_num, 3, dtype=torch.float32, device=self._device)
            # variable_state = variable_state / torch.sum(variable_state, 1).unsqueeze(1)
            function_state = torch.rand(edge_num, 2, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0
        else:
            variable_state = torch.ones(edge_num, 3, dtype=torch.float32, device=self._device) / 3.0
            function_state = 0.5 * torch.ones(edge_num, 2, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0

        return (variable_state, function_state)


###############################################################


class ReinforcePredictor(nn.Module):
    "Implements the prediction mechanism for the Reinforce Algorithm."
    
    def __init__(self, device):
        super(ReinforcePredictor, self).__init__()
        self._device = device

    def forward(self, decimator_state, sat_problem, last_call=False):

        pred = decimator_state[1][:, 1].unsqueeze(1)
        pred = (torch.mm(sat_problem._graph_mask_tuple[0], pred) > 0).float()

        return pred, None
