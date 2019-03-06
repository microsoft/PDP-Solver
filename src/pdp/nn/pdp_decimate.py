"""
Define various decimators for the PDP framework.
"""

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdp.nn import util


###############################################################
### The Decimator Classes
###############################################################


class NeuralDecimator(nn.Module):
    "Implements a neural decimator."

    def __init__(self, device, message_dimension, meta_data_dimension, hidden_dimension, mem_hidden_dimension,
                 mem_agg_hidden_dimension, agg_hidden_dimension, edge_dimension, dropout):

        super(NeuralDecimator, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        self._drop_out = dropout

        if isinstance(message_dimension, tuple):
            variable_message_dim, function_message_dim = message_dimension
        else:
            variable_message_dim = message_dimension
            function_message_dim = message_dimension

        self._variable_rnn_cell = nn.GRUCell(
            variable_message_dim + edge_dimension + meta_data_dimension, hidden_dimension, bias=True)
        self._function_rnn_cell = nn.GRUCell(
            function_message_dim + edge_dimension + meta_data_dimension, hidden_dimension, bias=True)

        self._module_list.append(self._variable_rnn_cell)
        self._module_list.append(self._function_rnn_cell)

        self._hidden_dimension = hidden_dimension
        self._mem_hidden_dimension = mem_hidden_dimension
        self._agg_hidden_dimension = agg_hidden_dimension
        self._mem_agg_hidden_dimension = mem_agg_hidden_dimension

    def forward(self, init_state, message_state, sat_problem, is_training, active_mask=None):

        variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = sat_problem._batch_mask_tuple

        if active_mask is not None:
            mask = torch.mm(b_variable_mask, active_mask.float())
            mask = torch.mm(variable_mask_transpose, mask)
        else:
            edge_num = init_state[0].size(0)
            mask = torch.ones(edge_num, 1, device=self._device)

        if sat_problem._meta_data is not None:
            graph_feat = torch.mm(b_variable_mask, sat_problem._meta_data)
            graph_feat = torch.mm(variable_mask_transpose, graph_feat)

        variable_state, function_state = message_state

        # Variable states
        variable_state = torch.cat((variable_state, sat_problem._edge_feature), 1)

        if sat_problem._meta_data is not None:
            variable_state = torch.cat((variable_state, graph_feat), 1)

        variable_state = mask * self._variable_rnn_cell(variable_state, init_state[0]) + (1 - mask) * init_state[0]

        # Function states
        function_state = torch.cat((function_state, sat_problem._edge_feature), 1)

        if sat_problem._meta_data is not None:
            function_state = torch.cat((function_state, graph_feat), 1)

        function_state = mask * self._function_rnn_cell(function_state, init_state[1]) + (1 - mask) * init_state[1]

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


class SequentialDecimator(nn.Module):
    "Implements the general (greedy) sequential decimator."

    def __init__(self, device, message_dimension, scorer, tolerance, t_max):
        super(SequentialDecimator, self).__init__()

        self._device = device
        self._tolerance = tolerance
        self._scorer = scorer
        self._previous_function_state = None
        self._message_dimension = message_dimension
        self._constant = torch.ones(1, 1, device=self._device)
        self._t_max = t_max
        self._counters = None
        self._module_list = nn.ModuleList([self._scorer])

    def forward(self, init_state, message_state, sat_problem, is_training, active_mask=None):

        if self._counters is None:
            self._counters = torch.zeros(sat_problem._batch_size, 1, device=self._device)

        if active_mask is not None:
            survey = message_state[1][:, 0].unsqueeze(1)
            survey = util.sparse_smooth_max(survey, sat_problem._graph_mask_tuple[0], self._device)
            survey = survey * sat_problem._active_variables
            survey = util.sparse_max(survey.squeeze(1), sat_problem._batch_mask_tuple[0], self._device).unsqueeze(1)

            active_mask[survey <= 1e-10] = 0

        if self._previous_function_state is not None and sat_problem._active_variables.sum() > 0:
            function_diff = (self._previous_function_state - message_state[1][:, 0]).abs().unsqueeze(1)

            if sat_problem._edge_mask is not None:
                function_diff = function_diff * sat_problem._edge_mask

            sum_diff = util.sparse_smooth_max(function_diff, sat_problem._graph_mask_tuple[0], self._device)
            sum_diff = sum_diff * sat_problem._active_variables
            sum_diff = util.sparse_max(sum_diff.squeeze(1), sat_problem._batch_mask_tuple[0], self._device).unsqueeze(1)

            self._counters[sum_diff[:, 0] < self._tolerance, 0] = 0
            sum_diff = (sum_diff < self._tolerance).float()
            sum_diff[self._counters[:, 0] >= self._t_max, 0] = 1
            self._counters[self._counters[:, 0] >= self._t_max, 0] = 0

            sum_diff = torch.mm(sat_problem._batch_mask_tuple[0], sum_diff)

            if sum_diff.sum() > 0:
                score, _ = self._scorer(message_state, sat_problem)

                # Find the variable index with max score for each instance in the batch
                coeff = score.abs() * sat_problem._active_variables * sum_diff

                if coeff.sum() > 0:
                    max_ind = util.sparse_argmax(coeff.squeeze(1), sat_problem._batch_mask_tuple[0], self._device)
                    norm = torch.mm(sat_problem._batch_mask_tuple[1], coeff)

                    if active_mask is not None:
                        max_ind = max_ind[(active_mask * (norm != 0)).squeeze(1)]
                    else:
                        max_ind = max_ind[norm.squeeze(1) != 0]

                    if max_ind.size()[0] > 0:
                        assignment = torch.zeros(sat_problem._variable_num, 1, device=self._device)
                        assignment[max_ind, 0] = score.sign()[max_ind, 0]

                        sat_problem.set_variables(assignment)

            self._counters = self._counters + 1

        self._previous_function_state = message_state[1][:, 0]

        return message_state

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication):
        self._previous_function_state = None
        self._counters = None

        return self._scorer.get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication)


###############################################################


class ReinforceDecimator(nn.Module):
    "Implements the (distributed) Reinforce decimator."

    def __init__(self, device, scorer, decimation_probability=0.5):
        super(ReinforceDecimator, self).__init__()

        self._device = device
        self._scorer = scorer
        self._decimation_probability = decimation_probability
        self._function_message_dim = 3
        self._variable_message_dim = 2
        self._previous_function_state = None

    def forward(self, init_state, message_state, sat_problem, is_training, active_mask=None):
        variable_state, function_state = message_state
        
        if active_mask is not None and self._previous_function_state is not None and sat_problem._active_variables.sum() > 0:
            function_diff = (self._previous_function_state - message_state[1][:, 0]).abs().unsqueeze(1)

            if sat_problem._edge_mask is not None:
                function_diff = function_diff * sat_problem._edge_mask

            sum_diff = util.sparse_smooth_max(function_diff, sat_problem._graph_mask_tuple[0], self._device)
            sum_diff = sum_diff * sat_problem._active_variables
            sum_diff = util.sparse_max(sum_diff.squeeze(1), sat_problem._batch_mask_tuple[0], self._device)
            active_mask[sum_diff <= 0.01, 0] = 0

        self._previous_function_state = message_state[1][:, 0]

        if torch.rand(1, device=self._device) < self._decimation_probability:
            variable_mask, variable_mask_transpose, function_mask, function_mask_transpose = sat_problem._graph_mask_tuple
            b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = sat_problem._batch_mask_tuple

            if active_mask is not None:
                mask = torch.mm(b_variable_mask, active_mask.float())
                mask = torch.mm(variable_mask_transpose, mask)
            else:
                mask = torch.ones(sat_problem._edge_num, 1, device=self._device)

            mask = mask.squeeze(1)
            score, _ = self._scorer(message_state, sat_problem)
            score = torch.mm(variable_mask_transpose, torch.sign(score)).squeeze(1)

            function_state[:, 1] = mask * score + (1 - mask) * function_state[:, 1]

        return variable_state, function_state

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication):

        edge_num = graph_map.size(1) * batch_replication
        self._previous_function_state = None

        if randomized:
            variable_state = torch.rand(edge_num, self._function_message_dim, dtype=torch.float32, device=self._device)
            function_state = torch.rand(edge_num, self._variable_message_dim, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0
        else:
            variable_state = torch.ones(edge_num, self._function_message_dim, dtype=torch.float32, device=self._device) / self._function_message_dim
            function_state = 0.5 * torch.ones(edge_num, self._variable_message_dim, dtype=torch.float32, device=self._device)
            function_state[:, 1] = 0

        return (variable_state, function_state)
