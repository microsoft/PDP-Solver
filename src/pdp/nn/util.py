# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# util.py : Defines the utility functionalities for the PDP framework.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageAggregator(nn.Module):
    "Implements a deep set function for message aggregation at variable and function nodes."

    def __init__(self, device, input_dimension, output_dimension, mem_hidden_dimension,
                 mem_agg_hidden_dimension, agg_hidden_dimension, feature_dimension, include_self_message):

        super(MessageAggregator, self).__init__()
        self._device = device
        self._include_self_message = include_self_message
        self._module_list = nn.ModuleList()

        if mem_hidden_dimension > 0 and mem_agg_hidden_dimension > 0:

            self._W1_m = nn.Linear(
                input_dimension, mem_hidden_dimension, bias=True)  # .to(self._device)

            self._W2_m = nn.Linear(
                mem_hidden_dimension, mem_agg_hidden_dimension, bias=False)  # .to(self._device)

            self._module_list.append(self._W1_m)
            self._module_list.append(self._W2_m)

        if agg_hidden_dimension > 0 and mem_agg_hidden_dimension > 0:

            if mem_hidden_dimension <= 0:
                mem_agg_hidden_dimension = input_dimension

            self._W1_a = nn.Linear(
                mem_agg_hidden_dimension + feature_dimension, agg_hidden_dimension, bias=True)  # .to(self._device)

            self._W2_a = nn.Linear(
                agg_hidden_dimension, output_dimension, bias=False)  # .to(self._device)

            self._module_list.append(self._W1_a)
            self._module_list.append(self._W2_a)

        self._agg_hidden_dimension = agg_hidden_dimension
        self._mem_hidden_dimension = mem_hidden_dimension
        self._mem_agg_hidden_dimension = mem_agg_hidden_dimension

    def forward(self, state, feature, mask, mask_transpose, edge_mask=None):

        # Apply the pre-aggregation transform
        if self._mem_hidden_dimension > 0 and self._mem_agg_hidden_dimension > 0:
            state = F.logsigmoid(self._W2_m(F.logsigmoid(self._W1_m(state))))

        if edge_mask is not None:
            state = state * edge_mask

        aggregated_state = torch.mm(mask, state)

        if not self._include_self_message:
            aggregated_state = torch.mm(mask_transpose, aggregated_state)

            if edge_mask is not None:
                aggregated_state = aggregated_state - state * edge_mask
            else:
                aggregated_state = aggregated_state - state

        if feature is not None:
            aggregated_state = torch.cat((aggregated_state, feature), 1)

        # Apply the post-aggregation transform
        if self._agg_hidden_dimension > 0 and self._mem_agg_hidden_dimension > 0:
            aggregated_state = F.logsigmoid(self._W2_a(F.logsigmoid(self._W1_a(aggregated_state))))

        return aggregated_state


###############################################################


class MultiLayerPerceptron(nn.Module):
    "Implements a standard fully-connected, multi-layer perceptron."

    def __init__(self, device, layer_dims):

        super(MultiLayerPerceptron, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        self._layer_num = len(layer_dims) - 1

        self._inner_layers = []
        for i in range(self._layer_num - 1):
            self._inner_layers += [nn.Linear(layer_dims[i], layer_dims[i + 1])]
            self._module_list.append(self._inner_layers[i])

        self._output_layer = nn.Linear(layer_dims[self._layer_num - 1], layer_dims[self._layer_num], bias=False)
        self._module_list.append(self._output_layer)

    def forward(self, inp):
        x = inp

        for layer in self._inner_layers:
            x = F.relu(layer(x))

        return F.sigmoid(self._output_layer(x))


##########################################################################################################################


class SatLossEvaluator(nn.Module):
    "Implements a module to calculate the energy (i.e. the loss) for the current prediction."

    def __init__(self, alpha, device):
        super(SatLossEvaluator, self).__init__()
        self._alpha = alpha
        self._device = device

    @staticmethod
    def safe_log(x, eps):
        return torch.max(x, eps).log()

    @staticmethod
    def compute_masks(graph_map, batch_variable_map, batch_function_map, edge_feature, device):
        edge_num = graph_map.size(1)
        variable_num = batch_variable_map.size(0)
        function_num = batch_function_map.size(0)
        all_ones = torch.ones(edge_num, device=device)
        edge_num_range = torch.arange(edge_num, dtype=torch.int64, device=device)

        variable_sparse_ind = torch.stack([edge_num_range, graph_map[0, :].long()])
        function_sparse_ind = torch.stack([graph_map[1, :].long(), edge_num_range])

        if device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1), 
                torch.Size([edge_num, variable_num]), device=device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, all_ones, 
                torch.Size([function_num, edge_num]), device=device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1), 
                torch.Size([edge_num, variable_num]), device=device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, all_ones, 
                torch.Size([function_num, edge_num]), device=device)

        return variable_mask, function_mask

    @staticmethod
    def compute_batch_mask(batch_variable_map, batch_function_map, device):
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        variable_all_ones = torch.ones(variable_num, device=device)
        function_all_ones = torch.ones(function_num, device=device)
        variable_range = torch.arange(variable_num, dtype=torch.int64, device=device)
        function_range = torch.arange(function_num, dtype=torch.int64, device=device)
        batch_size = (batch_variable_map.max() + 1).long().item()

        variable_sparse_ind = torch.stack([variable_range, batch_variable_map.long()])
        function_sparse_ind = torch.stack([function_range, batch_function_map.long()])

        if device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, variable_all_ones, 
                torch.Size([variable_num, batch_size]), device=device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, function_all_ones, 
                torch.Size([function_num, batch_size]), device=device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, variable_all_ones, 
                torch.Size([variable_num, batch_size]), device=device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, function_all_ones, 
                torch.Size([function_num, batch_size]), device=device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return (variable_mask, variable_mask_transpose, function_mask, function_mask_transpose)

    def forward(self, variable_prediction, label, graph_map, batch_variable_map, 
        batch_function_map, edge_feature, meta_data, global_step, eps, max_coeff, loss_sharpness):

        coeff = torch.min(global_step.pow(self._alpha), torch.tensor([max_coeff], device=self._device))

        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map,
            edge_feature, self._device)

        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2

        weights = (coeff * edge_values).exp()

        nominator = torch.mm(function_mask, weights * edge_values)
        denominator = torch.mm(function_mask, weights)

        clause_value = denominator / torch.max(nominator, eps)
        clause_value = 1 + (clause_value - 1).pow(loss_sharpness)
        return torch.mean(SatLossEvaluator.safe_log(clause_value, eps))


##########################################################################################################################


class SatCNFEvaluator(nn.Module):
    "Implements a module to evaluate the current prediction."

    def __init__(self, device):
        super(SatCNFEvaluator, self).__init__()
        self._device = device

    def forward(self, variable_prediction, graph_map, batch_variable_map, 
        batch_function_map, edge_feature, meta_data):

        variable_num = batch_variable_map.size(0)
        function_num = batch_function_map.size(0)
        batch_size = (batch_variable_map.max() + 1).item()
        all_ones = torch.ones(function_num, 1, device=self._device)

        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map, 
            edge_feature, self._device)

        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = \
            SatLossEvaluator.compute_batch_mask(
            batch_variable_map, batch_function_map, self._device)

        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2
        edge_values = (edge_values > 0.5).float()

        clause_values = torch.mm(function_mask, edge_values)
        clause_values = (clause_values > 0).float()

        max_sat = torch.mm(b_function_mask_transpose, all_ones)
        batch_values = torch.mm(b_function_mask_transpose, clause_values)

        return (max_sat == batch_values).float(), max_sat - batch_values


##########################################################################################################################


class PerceptronTanh(nn.Module):
    "Implements a 1-layer perceptron with Tanh activaton."

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(PerceptronTanh, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False)

    def forward(self, inp):
        return F.tanh(self._layer2(F.relu(self._layer1(inp))))


##########################################################################################################################


def sparse_argmax(x, mask, device):
    "Implements the exact, memory-inefficient argmax operation for a row vector input."

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()

    return torch.argmax(dense_mat, 0)

def sparse_max(x, mask, device):
    "Implements the exact, memory-inefficient max operation for a row vector input."

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()

    return torch.max(dense_mat, 0)[0] + x.min() - 1

def safe_exp(x, device):
    "Implements safe exp operation."

    return torch.min(x, torch.tensor([30.0], device=device)).exp()

def sparse_smooth_max(x, mask, device, alpha=30):
    "Implements the approximate, memory-efficient max operation for a row vector input."

    coeff = safe_exp(alpha * x, device)
    return torch.mm(mask, x * coeff) / torch.max(torch.mm(mask, coeff), torch.ones(1, device=device))
