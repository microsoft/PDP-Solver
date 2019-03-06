# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# solver.py : Defines the base class for all PDP Solvers as well as the various inherited solvers.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pytorch.PDP import pdp_propagate, pdp_decimate, pdp_predict, util

###############################################################
### The Problem Class
###############################################################

class SATProblem(object):
    "The class that encapsulates a batch of CNF problem instances."

    def __init__(self, data_batch, device, batch_replication=1):
        self._device = device
        self._batch_replication = batch_replication
        self.setup_problem(data_batch, batch_replication)

    def setup_problem(self, data_batch, batch_replication):
        "Setup the problem properties as well as the relevant sparse matrices."

        if batch_replication > 1:
            self._replication_mask_tuple = self._compute_batch_replication_map(data_batch[1], batch_replication)
            self._graph_map, self._batch_variable_map, self._batch_function_map, self._edge_feature, self._meta_data, _ = self._replicate_batch(data_batch, batch_replication)
        else:
            self._graph_map, self._batch_variable_map, self._batch_function_map, self._edge_feature, self._meta_data, _ = data_batch
        
        self._variable_num = self._batch_variable_map.size()[0]
        self._function_num = self._batch_function_map.size()[0]
        self._edge_num = self._graph_map.size()[1]

        self._vf_mask_tuple = self._compute_variable_function_map(self._graph_map, self._batch_variable_map,
                self._batch_function_map, self._edge_feature)
        self._batch_mask_tuple = self._compute_batch_map(self._batch_variable_map, self._batch_function_map)
        self._graph_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map, self._batch_function_map)
        self._pos_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map, self._batch_function_map, (self._edge_feature == 1).squeeze(1).float())
        self._neg_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map, self._batch_function_map, (self._edge_feature == -1).squeeze(1).float())
        self._signed_mask_tuple = self._compute_graph_mask(self._graph_map, self._batch_variable_map, self._batch_function_map, self._edge_feature.squeeze(1))

        self._active_variables = torch.ones(self._variable_num, 1, device=self._device)
        self._active_functions = torch.ones(self._function_num, 1, device=self._device)
        self._solution = 0.5 * torch.ones(self._variable_num, device=self._device)

        self._batch_size = (self._batch_variable_map.max() + 1).long().item()
        self._is_sat = 0.5 * torch.ones(self._batch_size, device=self._device)

    def _replicate_batch(self, data_batch, batch_replication):
        "Implements the batch replication."

        graph_map, batch_variable_map, batch_function_map, edge_feature, meta_data, label = data_batch
        edge_num = graph_map.size()[1]
        batch_size = (batch_variable_map.max() + 1).long().item()
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]

        ind = torch.arange(batch_replication, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, edge_num).view(1, -1)
        graph_map = graph_map.repeat(1, batch_replication) + ind.repeat(2, 1) * torch.tensor([[variable_num], [function_num]], dtype=torch.int32, device=self._device)

        ind = torch.arange(batch_replication, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, variable_num).view(1, -1)
        batch_variable_map = batch_variable_map.repeat(batch_replication) + ind * batch_size

        ind = torch.arange(batch_replication, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, function_num).view(1, -1)
        batch_function_map = batch_function_map.repeat(batch_replication) + ind * batch_size

        edge_feature = edge_feature.repeat(batch_replication, 1)

        if meta_data is not None:
            meta_data = meta_data.repeat(batch_replication, 1)

        if label is not None:
            label = label.repeat(batch_replication, 1)

        return graph_map, batch_variable_map.squeeze(0), batch_function_map.squeeze(0), edge_feature, meta_data, label

    def _compute_batch_replication_map(self, batch_variable_map, batch_replication):
        batch_size = (batch_variable_map.max() + 1).long().item()
        x_ind = torch.arange(batch_size * batch_replication, dtype=torch.int64, device=self._device)
        y_ind = torch.arange(batch_size, dtype=torch.int64, device=self._device).repeat(batch_replication)
        ind = torch.stack([x_ind, y_ind])
        all_ones = torch.ones(batch_size * batch_replication, device=self._device)
        
        if self._device.type == 'cuda':
            mask = torch.cuda.sparse.FloatTensor(ind, all_ones, 
                torch.Size([batch_size * batch_replication, batch_size]), device=self._device)
        else:
            mask = torch.sparse.FloatTensor(ind, all_ones, 
                torch.Size([batch_size * batch_replication, batch_size]), device=self._device)
        
        mask_transpose = mask.transpose(0, 1)
        return (mask, mask_transpose)

    def _compute_variable_function_map(self, graph_map, batch_variable_map, batch_function_map, edge_feature):
        edge_num = graph_map.size()[1]
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        all_ones = torch.ones(edge_num, device=self._device)

        if self._device.type == 'cuda':
            mask = torch.cuda.sparse.FloatTensor(graph_map.long(), all_ones, 
                torch.Size([variable_num, function_num]), device=self._device)
            signed_mask = torch.cuda.sparse.FloatTensor(graph_map.long(), edge_feature.squeeze(1), 
                torch.Size([variable_num, function_num]), device=self._device)
        else:
            mask = torch.sparse.FloatTensor(graph_map.long(), all_ones, 
                torch.Size([variable_num, function_num]), device=self._device)
            signed_mask = torch.sparse.FloatTensor(graph_map.long(), edge_feature.squeeze(1), 
                torch.Size([variable_num, function_num]), device=self._device)

        mask_transpose = mask.transpose(0, 1)
        signed_mask_transpose = signed_mask.transpose(0, 1)

        return (mask, mask_transpose, signed_mask, signed_mask_transpose)

    def _compute_batch_map(self, batch_variable_map, batch_function_map):
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        variable_all_ones = torch.ones(variable_num, device=self._device)
        function_all_ones = torch.ones(function_num, device=self._device)
        variable_range = torch.arange(variable_num, dtype=torch.int64, device=self._device)
        function_range = torch.arange(function_num, dtype=torch.int64, device=self._device)
        batch_size = (batch_variable_map.max() + 1).long().item()

        variable_sparse_ind = torch.stack([variable_range, batch_variable_map.long()])
        function_sparse_ind = torch.stack([function_range, batch_function_map.long()])

        if self._device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, variable_all_ones, 
                torch.Size([variable_num, batch_size]), device=self._device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, function_all_ones, 
                torch.Size([function_num, batch_size]), device=self._device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, variable_all_ones, 
                torch.Size([variable_num, batch_size]), device=self._device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, function_all_ones, 
                torch.Size([function_num, batch_size]), device=self._device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return (variable_mask, variable_mask_transpose, function_mask, function_mask_transpose)

    def _compute_graph_mask(self, graph_map, batch_variable_map, batch_function_map, edge_values=None):
        edge_num = graph_map.size()[1]
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]

        if edge_values is None:
            edge_values = torch.ones(edge_num, device=self._device)

        edge_num_range = torch.arange(edge_num, dtype=torch.int64, device=self._device)

        variable_sparse_ind = torch.stack([graph_map[0, :].long(), edge_num_range])
        function_sparse_ind = torch.stack([graph_map[1, :].long(), edge_num_range])

        if self._device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, edge_values, 
                torch.Size([variable_num, edge_num]), device=self._device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, edge_values, 
                torch.Size([function_num, edge_num]), device=self._device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_values, 
                torch.Size([variable_num, edge_num]), device=self._device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, edge_values, 
                torch.Size([function_num, edge_num]), device=self._device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return (variable_mask, variable_mask_transpose, function_mask, function_mask_transpose)

    def _peel(self):
        "Implements the peeling algorithm."
        
        vf_map, vf_map_transpose, signed_vf_map, _ = self._vf_mask_tuple

        variable_degree = torch.mm(vf_map, self._active_functions)
        signed_variable_degree = torch.mm(signed_vf_map, self._active_functions)

        while True:
            single_variables = (variable_degree == signed_variable_degree.abs()).float() * self._active_variables

            if torch.sum(single_variables) <= 0:
                break

            single_functions = (torch.mm(vf_map_transpose, single_variables) > 0).float() * self._active_functions
            degree_delta = torch.mm(vf_map, single_functions) * self._active_variables
            signed_degree_delta = torch.mm(signed_vf_map, single_functions) * self._active_variables
            self._solution[single_variables[:, 0] == 1] = (signed_variable_degree[single_variables[:, 0] == 1, 0].sign() + 1) / 2.0

            variable_degree -= degree_delta
            signed_variable_degree -= signed_degree_delta

            self._active_variables[single_variables[:, 0] == 1, 0] = 0
            self._active_functions[single_functions[:, 0] == 1, 0] = 0

    def _set_variable_core(self, assignment):
        "Fixes variables to certain binary values."

        _, vf_map_transpose, _, signed_vf_map_transpose = self._vf_mask_tuple

        assignment *= self._active_variables

        # Compute the number of inputs for each function node
        input_num = torch.mm(vf_map_transpose, assignment.abs())

        # Compute the signed evaluation for each function node
        function_eval = torch.mm(signed_vf_map_transpose, assignment)

        # Compute the de-activated functions
        deactivated_functions = (function_eval > -input_num).float() * self._active_functions

        # De-activate functions and variables
        self._active_variables[assignment[:, 0].abs() == 1, 0] = 0
        self._active_functions[deactivated_functions[:, 0] == 1, 0] = 0

        # Update the solution
        self._solution[assignment[:, 0].abs() == 1] = (assignment[assignment[:, 0].abs() == 1, 0] + 1) / 2.0

    def _propagate_single_clauses(self):
        "Implements unit clause propagation algorithm."

        vf_map, vf_map_transpose, signed_vf_map, _ = self._vf_mask_tuple
        b_variable_mask, b_variable_mask_transpose, b_function_mask, _ = self._batch_mask_tuple

        while True:
            function_degree = torch.mm(vf_map_transpose, self._active_variables)
            single_functions = (function_degree == 1).float() * self._active_functions

            if torch.sum(single_functions) <= 0:
                break

            # Compute the number of inputs for each variable node
            input_num = torch.mm(vf_map, single_functions)

            # Compute the signed evaluation for each variable node
            variable_eval = torch.mm(signed_vf_map, single_functions)

            # Detect and de-activate the UNSAT examples
            conflict_variables = (variable_eval.abs() != input_num).float() * self._active_variables
            if torch.sum(conflict_variables) > 0:

                # Detect the UNSAT examples
                unsat_examples = torch.mm(b_variable_mask_transpose, conflict_variables)
                self._is_sat[unsat_examples[:, 0] >= 1] = 0

                # De-activate the function nodes related to unsat examples
                unsat_functions = torch.mm(b_function_mask, unsat_examples) * self._active_functions
                self._active_functions[unsat_functions[:, 0] == 1, 0] = 0

                # De-activate the variable nodes related to unsat examples
                unsat_variables = torch.mm(b_variable_mask, unsat_examples) * self._active_variables
                self._active_variables[unsat_variables[:, 0] == 1, 0] = 0

            # Compute the assigned variables
            assigned_variables = (variable_eval.abs() == input_num).float() * self._active_variables

            # Compute the variable assignment
            assignment = torch.sign(variable_eval) * assigned_variables

            # De-activate single functions
            self._active_functions[single_functions[:, 0] == 1, 0] = 0

            # Set the corresponding variables
            self._set_variable_core(assignment)

    def set_variables(self, assignment):
        "Fixes variables to certain binary values and simplifies the CNF accordingly."

        self._set_variable_core(assignment)
        self.simplify()

    def simplify(self):
        "Simplifies the CNF."

        self._propagate_single_clauses()
        self._peel()


###############################################################
### The Solver Classes
###############################################################


class PropagatorDecimatorSolverBase(nn.Module):
    "The base class for all PDP SAT solvers."

    def __init__(self, device, name, propagator, decimator, predictor, local_search_iterations=0, epsilon=0.05):

        super(PropagatorDecimatorSolverBase, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()

        self._propagator = propagator
        self._decimator = decimator
        self._predictor = predictor

        self._module_list.append(self._propagator)
        self._module_list.append(self._decimator)
        self._module_list.append(self._predictor)

        self._global_step = nn.Parameter(torch.tensor([0], dtype=torch.float, device=self._device), requires_grad=False)
        self._name = name
        self._local_search_iterations = local_search_iterations
        self._epsilon = epsilon

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, export_path_base):
        torch.save(self.state_dict(), os.path.join(export_path_base, self._name))

    def load(self, import_path_base):
        self.load_state_dict(torch.load(os.path.join(import_path_base, self._name)))

    def forward(self, init_state,
        graph_map, batch_variable_map, batch_function_map, edge_feature, 
        meta_data, is_training=True, iteration_num=1, check_termination=None, simplify=True, batch_replication=1):
        
        init_propagator_state, init_decimator_state = init_state
        batch_replication = 1 if is_training else batch_replication
        sat_problem = SATProblem((graph_map, batch_variable_map, batch_function_map, edge_feature, meta_data, None), self._device, batch_replication)

        if simplify and not is_training:
            sat_problem.simplify()

        if self._propagator is not None and self._decimator is not None:
            propagator_state, decimator_state = self._forward_core(init_propagator_state, init_decimator_state, 
                sat_problem, iteration_num, is_training, check_termination)
        else:
            decimator_state = None
            propagator_state = None

        prediction = self._predictor(decimator_state, sat_problem, True)

        # Post-processing local search
        if not is_training:
            prediction = self._local_search(prediction, sat_problem, batch_replication)

        prediction = self._update_solution(prediction, sat_problem)
        
        if batch_replication > 1:
            prediction, propagator_state, decimator_state = self._deduplicate(prediction, propagator_state, decimator_state, sat_problem)

        return (prediction, (propagator_state, decimator_state))

    def _forward_core(self, init_propagator_state, init_decimator_state, sat_problem, iteration_num, is_training, check_termination):

        propagator_state = init_propagator_state
        decimator_state = init_decimator_state

        if check_termination is None:
            active_mask = None
        else:
            active_mask = torch.ones(sat_problem._batch_size, 1, dtype=torch.uint8, device=self._device)

        for _ in torch.arange(iteration_num, dtype=torch.int32, device=self._device):

            propagator_state = self._propagator(propagator_state, decimator_state, sat_problem, is_training, active_mask)
            decimator_state = self._decimator(decimator_state, propagator_state, sat_problem, is_training, active_mask)

            sat_problem._edge_mask = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables) * \
                torch.mm(sat_problem._graph_mask_tuple[3], sat_problem._active_functions)

            if sat_problem._edge_mask.sum() < sat_problem._edge_num:
                decimator_state += (sat_problem._edge_mask,)

            if check_termination is not None:
                prediction = self._predictor(decimator_state, sat_problem)
                prediction = self._update_solution(prediction, sat_problem)

                check_termination(active_mask, prediction, sat_problem)
                num_active = active_mask.sum()

                if num_active <= 0:
                    break

        return propagator_state, decimator_state

    def _update_solution(self, prediction, sat_problem):
        "Updates the the SAT problem object's solution according to the cuerrent prediction."

        if prediction[0] is not None:
            variable_solution = sat_problem._active_variables * prediction[0] + \
                (1.0 - sat_problem._active_variables) * sat_problem._solution.unsqueeze(1)
            sat_problem._solution[sat_problem._active_variables[:, 0] == 1] = \
                variable_solution[sat_problem._active_variables[:, 0] == 1, 0]
        else:
            variable_solution = None

        return variable_solution, prediction[1]

    def _deduplicate(self, prediction, propagator_state, decimator_state, sat_problem):
        "De-duplicates the current batch (to neutralize the batch replication) by finding the replica with minimum energy for each problem instance. "

        if sat_problem._batch_replication <= 1 or sat_problem._replication_mask_tuple is None:
            return None, None, None

        assignment = 2 * prediction[0] - 1.0
        energy, _ = self._compute_energy(assignment, sat_problem)
        max_ind = util.sparse_argmax(-energy.squeeze(1), sat_problem._replication_mask_tuple[0], device=self._device)

        batch_flag = torch.zeros(sat_problem._batch_size, 1, device=self._device)
        batch_flag[max_ind, 0] = 1

        flag = torch.mm(sat_problem._batch_mask_tuple[0], batch_flag)
        variable_prediction = (flag * prediction[0]).view(sat_problem._batch_replication, -1).sum(dim=0).unsqueeze(1)

        flag = torch.mm(sat_problem._graph_mask_tuple[1], flag)
        new_propagator_state = ()
        for x in propagator_state:
            new_propagator_state += ((flag * x).view(sat_problem._batch_replication, sat_problem._edge_num / sat_problem._batch_replication, -1).sum(dim=0),)

        new_decimator_state = ()
        for x in decimator_state:
            new_decimator_state += ((flag * x).view(sat_problem._batch_replication, sat_problem._edge_num / sat_problem._batch_replication, -1).sum(dim=0),)

        function_prediction = None
        if prediction[1] is not None:
            flag = torch.mm(sat_problem._batch_mask_tuple[2], batch_flag)
            function_prediction = (flag * prediction[1]).view(sat_problem._batch_replication, -1).sum(dim=0).unsqueeze(1)

        return (variable_prediction, function_prediction), new_propagator_state, new_decimator_state

    def _local_search(self, prediction, sat_problem, batch_replication):
        "Implements the Walk-SAT algorithm for post-processing."

        assignment = (prediction[0] > 0.5).float()
        assignment = sat_problem._active_variables * (2*assignment - 1.0)

        sat_problem._edge_mask = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables) * \
            torch.mm(sat_problem._graph_mask_tuple[3], sat_problem._active_functions)

        for _ in range(self._local_search_iterations):
            unsat_examples, unsat_functions = self._compute_energy(assignment, sat_problem)
            unsat_examples = (unsat_examples > 0).float()

            if batch_replication > 1:
                compact_unsat_examples = 1 - (torch.mm(sat_problem._replication_mask_tuple[1], 1 - unsat_examples) > 0).float()
                if compact_unsat_examples.sum() == 0:
                    break
            elif unsat_examples.sum() == 0:
                break

            delta_energy = self._compute_energy_diff(assignment, sat_problem)
            max_delta_ind = util.sparse_argmax(-delta_energy.squeeze(1), sat_problem._batch_mask_tuple[0], device=self._device)

            unsat_variables = torch.mm(sat_problem._vf_mask_tuple[0], unsat_functions) * sat_problem._active_variables
            unsat_variables = (unsat_variables > 0).float() * torch.rand([sat_problem._variable_num, 1], device=self._device)
            random_ind = util.sparse_argmax(unsat_variables.squeeze(1), sat_problem._batch_mask_tuple[0], device=self._device)

            coin = (torch.rand(sat_problem._batch_size, device=self._device) > self._epsilon).long()
            max_ind = coin * max_delta_ind + (1 - coin) * random_ind
            max_ind = max_ind[unsat_examples[:, 0] > 0]

            # Flipping the selected variables
            assignment[max_ind, 0] = -assignment[max_ind, 0]

        return (assignment + 1) / 2.0, prediction[1]

    def _compute_energy_diff(self, assignment, sat_problem):
        "Computes the delta energy if each variable to be flipped during the local search."

        distributed_assignment = torch.mm(sat_problem._signed_mask_tuple[1], assignment * sat_problem._active_variables)
        aggregated_assignment = torch.mm(sat_problem._graph_mask_tuple[2], distributed_assignment)
        aggregated_assignment = torch.mm(sat_problem._graph_mask_tuple[3], aggregated_assignment)
        aggregated_assignment = aggregated_assignment - distributed_assignment

        function_degree = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables)
        function_degree = torch.mm(sat_problem._graph_mask_tuple[2], function_degree)
        function_degree = torch.mm(sat_problem._graph_mask_tuple[3], function_degree)

        critical_edges = (aggregated_assignment == (1 - function_degree)).float() * sat_problem._edge_mask
        delta = torch.mm(sat_problem._graph_mask_tuple[0], critical_edges * distributed_assignment)

        return delta

    def _compute_energy(self, assignment, sat_problem):
        "Computes the energy of each CNF instance present in the batch."

        aggregated_assignment = torch.mm(sat_problem._signed_mask_tuple[1], assignment * sat_problem._active_variables)
        aggregated_assignment = torch.mm(sat_problem._graph_mask_tuple[2], aggregated_assignment)

        function_degree = torch.mm(sat_problem._graph_mask_tuple[1], sat_problem._active_variables)
        function_degree = torch.mm(sat_problem._graph_mask_tuple[2], function_degree)

        unsat_functions = (aggregated_assignment == -function_degree).float() * sat_problem._active_functions
        return torch.mm(sat_problem._batch_mask_tuple[3], unsat_functions), unsat_functions

    def get_init_state(self, graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication=1):
        "Initializes the propgator and the decimator messages in each direction."

        if self._propagator is None:
            init_propagator_state = None
        else:
            init_propagator_state = self._propagator.get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication)
        
        if self._decimator is None:
            init_decimator_state = None
        else:
            init_decimator_state = self._decimator.get_init_state(graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat, randomized, batch_replication)

        return init_propagator_state, init_decimator_state


###############################################################


class NeuralPropagatorDecimatorSolver(PropagatorDecimatorSolverBase):
    "Implements a fully neural PDP SAT solver with both the propagator and the decimator being neural."

    def __init__(self, device, name, edge_dimension, meta_data_dimension, 
                propagator_dimension, decimator_dimension, 
                mem_hidden_dimension, agg_hidden_dimension, mem_agg_hidden_dimension, prediction_dimension,
                variable_classifier=None, function_classifier=None, dropout=0, 
                local_search_iterations=0, epsilon=0.05):

        super(NeuralPropagatorDecimatorSolver, self).__init__(
            device=device, name=name, 
            propagator=pdp_propagate.NeuralMessagePasser(device, edge_dimension, decimator_dimension, 
                meta_data_dimension, propagator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, dropout), 
            decimator=pdp_decimate.NeuralDecimator(device, propagator_dimension, meta_data_dimension, 
                decimator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, edge_dimension, dropout),
            predictor=pdp_predict.NeuralPredictor(device, decimator_dimension, prediction_dimension, 
                edge_dimension, meta_data_dimension, mem_hidden_dimension, agg_hidden_dimension, 
                mem_agg_hidden_dimension, variable_classifier, function_classifier),
            local_search_iterations=local_search_iterations, epsilon=epsilon)


###############################################################


class NeuralSurveyPropagatorSolver(PropagatorDecimatorSolverBase):
    "Implements a PDP solver with the SP propgator and a neural decimator."

    def __init__(self, device, name, edge_dimension, meta_data_dimension, 
            decimator_dimension, 
            mem_hidden_dimension, agg_hidden_dimension, mem_agg_hidden_dimension, prediction_dimension,
            variable_classifier=None, function_classifier=None, dropout=0,
            local_search_iterations=0, epsilon=0.05):

        super(NeuralSurveyPropagatorSolver, self).__init__(
            device=device, name=name, 
            propagator=pdp_propagate.SurveyPropagator(device, decimator_dimension, include_adaptors=True), 
            decimator=pdp_decimate.NeuralDecimator(device, (3, 1), meta_data_dimension, 
                decimator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, edge_dimension, dropout),
            predictor=pdp_predict.NeuralPredictor(device, decimator_dimension, prediction_dimension, 
                edge_dimension, meta_data_dimension, mem_hidden_dimension, agg_hidden_dimension, 
                mem_agg_hidden_dimension, variable_classifier, function_classifier),
            local_search_iterations=local_search_iterations, epsilon=epsilon)


###############################################################


class SurveyPropagatorSolver(PropagatorDecimatorSolverBase):
    "Implements the classical SP-guided decimation solver via the PDP framework."

    def __init__(self, device, name, tolerance, t_max, local_search_iterations=0, epsilon=0.05):

        super(SurveyPropagatorSolver, self).__init__(
            device=device, name=name, 
            propagator=pdp_propagate.SurveyPropagator(device, decimator_dimension=1, include_adaptors=False), 
            decimator=pdp_decimate.SequentialDecimator(device, message_dimension=(3, 1), 
                scorer=pdp_predict.SurveyScorer(device, message_dimension=1, include_adaptors=False), tolerance=tolerance, t_max=t_max),
            predictor=pdp_predict.IdentityPredictor(device=device, random_fill=True),
            local_search_iterations=local_search_iterations, epsilon=epsilon)


###############################################################


class WalkSATSolver(PropagatorDecimatorSolverBase):
    "Implements the classical Walk-SAT solver via the PDP framework."

    def __init__(self, device, name, iteration_num, epsilon=0.05):

        super(WalkSATSolver, self).__init__(
            device=device, name=name, propagator=None, decimator=None,
            predictor=pdp_predict.IdentityPredictor(device=device, random_fill=True),
            local_search_iterations=iteration_num, epsilon=epsilon)


###############################################################


class ReinforceSurveyPropagatorSolver(PropagatorDecimatorSolverBase):
    "Implements the classical Reinforce solver via the PDP framework."

    def __init__(self, device, name, pi=0.1, decimation_probability=0.5, local_search_iterations=0, epsilon=0.05):

        super(ReinforceSurveyPropagatorSolver, self).__init__(
            device=device, name=name, 
            propagator=pdp_propagate.SurveyPropagator(device, decimator_dimension=1, include_adaptors=False, pi=pi), 
            decimator=pdp_decimate.ReinforceDecimator(device, 
                scorer=pdp_predict.SurveyScorer(device, message_dimension=1, include_adaptors=False, pi=pi), 
                decimation_probability=decimation_probability),
            predictor=pdp_predict.ReinforcePredictor(device=device),
            local_search_iterations=local_search_iterations, epsilon=epsilon)


###############################################################


class NeuralSequentialDecimatorSolver(PropagatorDecimatorSolverBase):
    "Implements a PDP solver with a neural propgator and the sequential decimator."

    def __init__(self, device, name, edge_dimension, meta_data_dimension, 
                propagator_dimension, decimator_dimension, 
                mem_hidden_dimension, agg_hidden_dimension, mem_agg_hidden_dimension, 
                classifier_dimension, dropout, tolerance, t_max, 
                local_search_iterations=0, epsilon=0.05):

        super(NeuralSequentialDecimatorSolver, self).__init__(
            device=device, name=name, 
            propagator=pdp_propagate.NeuralMessagePasser(device, edge_dimension, decimator_dimension, 
                meta_data_dimension, propagator_dimension, mem_hidden_dimension,
                mem_agg_hidden_dimension, agg_hidden_dimension, dropout), 
            decimator=pdp_decimate.SequentialDecimator(device, message_dimension=(3, 1), 
                scorer=pdp_predict.NeuralPredictor(device, decimator_dimension, 1, 
                    edge_dimension, meta_data_dimension, mem_hidden_dimension, agg_hidden_dimension, 
                    mem_agg_hidden_dimension, variable_classifier=util.PerceptronTanh(decimator_dimension, 
                    classifier_dimension, 1), function_classifier=None), 
                tolerance=tolerance, t_max=t_max),
            predictor=pdp_predict.IdentityPredictor(device=device, random_fill=True),
            local_search_iterations=local_search_iterations, epsilon=epsilon)
