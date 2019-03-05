# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# pdp_propagate.py : Defines various propagators for the PDP framework.

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdp.nn import util


###############################################################
### The Propagator Classes
###############################################################


class NeuralMessagePasser(nn.Module):
    "Implements the neural propagator."

    def __init__(self, device, edge_dimension, decimator_dimension, meta_data_dimension, hidden_dimension, mem_hidden_dimension,
                 mem_agg_hidden_dimension, agg_hidden_dimension, dropout):

        super(NeuralMessagePasser, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        self._drop_out = dropout

        self._variable_aggregator = util.MessageAggregator(device, decimator_dimension + edge_dimension + meta_data_dimension, 
            hidden_dimension, mem_hidden_dimension,
            mem_agg_hidden_dimension, agg_hidden_dimension, edge_dimension, include_self_message=False)
        self._function_aggregator = util.MessageAggregator(device, decimator_dimension + edge_dimension + meta_data_dimension, 
            hidden_dimension, mem_hidden_dimension,
            mem_agg_hidden_dimension, agg_hidden_dimension, edge_dimension, include_self_message=False)

        self._module_list.append(self._variable_aggregator)
        self._module_list.append(self._function_aggregator)

        self._hidden_dimension = hidden_dimension
        self._mem_hidden_dimension = mem_hidden_dimension
        self._agg_hidden_dimension = agg_hidden_dimension
        self._mem_agg_hidden_dimension = mem_agg_hidden_dimension

    def forward(self, init_state, decimator_state, sat_problem, is_training, active_mask=None):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, _, _, _ = sat_problem._batch_mask_tuple

        if active_mask is not None:
            mask = torch.mm(b_variable_mask, active_mask.float())
            mask = torch.mm(variable_mask_transpose, mask)
        else:
            edge_num = init_state[0].size(0)
            mask = torch.ones(edge_num, 1, device=self._device)

        if sat_problem._meta_data is not None:
            graph_feat = torch.mm(b_variable_mask, sat_problem._meta_data)
            graph_feat = torch.mm(variable_mask_transpose, graph_feat)

        if len(decimator_state) == 3:
            decimator_variable_state, decimator_function_state, edge_mask = decimator_state
        else:
            decimator_variable_state, decimator_function_state = decimator_state
            edge_mask = None

        variable_state, function_state = init_state

        ## variables --> functions
        decimator_variable_state = torch.cat((decimator_variable_state, sat_problem._edge_feature), 1)

        if sat_problem._meta_data is not None:
            decimator_variable_state = torch.cat((decimator_variable_state, graph_feat), 1)

        function_state = mask * self._variable_aggregator(
            decimator_variable_state, sat_problem._edge_feature, variable_mask, variable_mask_transpose, edge_mask) + (1 - mask) * function_state

        function_state = F.dropout(function_state, p=self._drop_out, training=is_training)

        ## functions --> variables
        decimator_function_state = torch.cat((decimator_function_state, sat_problem._edge_feature), 1)

        if sat_problem._meta_data is not None:
            decimator_function_state = torch.cat((decimator_function_state, graph_feat), 1)

        variable_state = mask * self._function_aggregator(
            decimator_function_state, sat_problem._edge_feature, function_mask, function_mask_transpose, edge_mask) + (1 - mask) * variable_state

        variable_state = F.dropout(variable_state, p=self._drop_out, training=is_training)

        del mask

        return variable_state, function_state

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication):

        edge_num = graph_map.size(1) * batch_replication

        if randomized:
            variable_state = 2.0*torch.rand(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device) - 1.0
            function_state = 2.0*torch.rand(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device) - 1.0
        else:
            variable_state = torch.zeros(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device)
            function_state = torch.zeros(edge_num, self._hidden_dimension, dtype=torch.float32, device=self._device)

        return (variable_state, function_state)


###############################################################


class SurveyPropagator(nn.Module):
    "Implements the Survey Propagator (SP)."

    def __init__(self, device, decimator_dimension, include_adaptors=False, pi=0.0):

        super(SurveyPropagator, self).__init__()
        self._device = device
        self._function_message_dim = 3
        self._variable_message_dim = 2
        self._include_adaptors = include_adaptors
        self._eps = torch.tensor([1e-40], device=self._device)
        self._max_logit = torch.tensor([30.0], device=self._device)
        self._pi = torch.tensor([pi], dtype=torch.float32, device=device)

        if self._include_adaptors:
            self._variable_input_projector = nn.Linear(decimator_dimension, self._variable_message_dim, bias=False)
            self._function_input_projector = nn.Linear(decimator_dimension, 1, bias=False)
            self._module_list = nn.ModuleList([self._variable_input_projector, self._function_input_projector])

    def safe_log(self, x):
        return torch.max(x, self._eps).log()

    def safe_exp(self, x):
        return torch.min(x, self._max_logit).exp()

    def forward(self, init_state, decimator_state, sat_problem, is_training, active_mask=None):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, _, _, _ = sat_problem._batch_mask_tuple
        p_variable_mask, _, _, _ = sat_problem._pos_mask_tuple
        n_variable_mask, _, _, _ = sat_problem._neg_mask_tuple

        if active_mask is not None:
            mask = torch.mm(b_variable_mask, active_mask.float())
            mask = torch.mm(variable_mask_transpose, mask)
        else:
            edge_num = init_state[0].size(0)
            mask = torch.ones(edge_num, 1, device=self._device)

        if len(decimator_state) == 3:
            decimator_variable_state, decimator_function_state, edge_mask = decimator_state
        else:
            decimator_variable_state, decimator_function_state = decimator_state
            edge_mask = None

        variable_state, function_state = init_state

        ## functions --> variables

        if self._include_adaptors:
            decimator_variable_state = F.logsigmoid(self._function_input_projector(decimator_variable_state))
        else:
            decimator_variable_state = self.safe_log(decimator_variable_state[:, 0]).unsqueeze(1)

        if edge_mask is not None:
            decimator_variable_state = decimator_variable_state * edge_mask

        aggregated_variable_state = torch.mm(function_mask, decimator_variable_state)
        aggregated_variable_state = torch.mm(function_mask_transpose, aggregated_variable_state)
        aggregated_variable_state = aggregated_variable_state - decimator_variable_state

        function_state = mask * self.safe_exp(aggregated_variable_state) + (1 - mask) * function_state[:, 0].unsqueeze(1)

        ## functions --> variables

        if self._include_adaptors:
            decimator_function_state = self._variable_input_projector(decimator_function_state)
            decimator_function_state[:, 0] = F.sigmoid(decimator_function_state[:, 0])
            decimator_function_state[:, 1] = torch.sign(decimator_function_state[:, 1])

        external_force = decimator_function_state[:, 1].unsqueeze(1)
        decimator_function_state = self.safe_log(1 - decimator_function_state[:, 0]).unsqueeze(1)

        if edge_mask is not None:
            decimator_function_state = decimator_function_state * edge_mask

        pos = torch.mm(p_variable_mask, decimator_function_state)
        pos = torch.mm(variable_mask_transpose, pos)
        neg = torch.mm(n_variable_mask, decimator_function_state)
        neg = torch.mm(variable_mask_transpose, neg)

        same_sign = 0.5 * (1 + sat_problem._edge_feature) * pos + 0.5 * (1 - sat_problem._edge_feature) * neg
        same_sign = same_sign - decimator_function_state
        same_sign += self.safe_log(1.0 - self._pi * (external_force == sat_problem._edge_feature).float())

        opposite_sign = 0.5 * (1 - sat_problem._edge_feature) * pos + 0.5 * (1 + sat_problem._edge_feature) * neg
        # The opposite sign edge aggregation does not include the current edge by definition, therefore no need for subtraction.
        opposite_sign += self.safe_log(1.0 - self._pi * (external_force == -sat_problem._edge_feature).float())

        dont_care = same_sign + opposite_sign

        bias = 0 #(2 * dont_care) / 3.0
        same_sign = same_sign - bias
        opposite_sign = opposite_sign - bias
        dont_care = self.safe_exp(dont_care - bias)

        same_sign = self.safe_exp(same_sign)
        opposite_sign = self.safe_exp(opposite_sign)
        q_u = same_sign * (1 - opposite_sign)
        q_s = opposite_sign * (1 - same_sign)

        total = q_u + q_s + dont_care
        temp = torch.cat((q_u, q_s, dont_care), 1) / total

        variable_state = mask * temp + (1 - mask) * variable_state

        del mask
        return variable_state, torch.cat((function_state, external_force), 1)

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication):

        edge_num = graph_map.size(1) * batch_replication

        if randomized:
            variable_state = torch.rand(edge_num, self._function_message_dim, dtype=torch.float32, device=self._device)
            variable_state = variable_state / torch.sum(variable_state, 1).unsqueeze(1)
            function_state = torch.rand(edge_num, self._variable_message_dim, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0
        else:
            variable_state = torch.ones(edge_num, self._function_message_dim, dtype=torch.float32, device=self._device) / self._function_message_dim
            function_state = 0.5 * torch.ones(edge_num, self._variable_message_dim, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0

        return (variable_state, function_state)


###############################################################
