# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# PDP_solver_trainer.py : Implements a factor graph trainer for various types of PDP SAT solvers.

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

from model_pytorch import factor_graph_trainer
from model_pytorch.PDP import solver, util


##########################################################################################################################


class Perceptron(nn.Module):
    "Implements a 1-layer perceptron."

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False)

    def forward(self, inp):
        return F.sigmoid(self._layer2(F.relu(self._layer1(inp))))


##########################################################################################################################

class SatFactorGraphTrainer(factor_graph_trainer.FactorGraphTrainerBase):
    "Implements a factor graph trainer for various types of PDP SAT solvers."

    def __init__(self, config, use_cuda, logger):
        super(SatFactorGraphTrainer, self).__init__(config=config, 
            has_meta_data=False, error_dim=config['error_dim'], loss=None, 
            evaluator=nn.L1Loss(), use_cuda=use_cuda, logger=logger)

        self._eps = 1e-8 * torch.ones(1, device=self._device)
        self._loss_evaluator = util.SatLossEvaluator(alpha = self._config['exploration'], device = self._device)
        self._cnf_evaluator = util.SatCNFEvaluator(device = self._device)
        self._counter = 0
        self._max_coeff = 10.0

    def _build_graph(self, config):
        model_list = []

        if config['model_type'] == 'np-nd-np':
            model_list += [solver.NeuralPropagatorDecimatorSolver(device=self._device, name=config['model_name'], 
                    edge_dimension=config['edge_feature_dim'], meta_data_dimension=config['meta_feature_dim'], 
                    propagator_dimension=config['hidden_dim'], decimator_dimension=config['hidden_dim'],
                    mem_hidden_dimension=config['mem_hidden_dim'], 
                    agg_hidden_dimension=config['agg_hidden_dim'], mem_agg_hidden_dimension=config['mem_agg_hidden_dim'], 
                    prediction_dimension=config['prediction_dim'], 
                    variable_classifier=Perceptron(config['hidden_dim'], config['classifier_dim'], config['prediction_dim']),
                    function_classifier=None, dropout=config['dropout'],
                    local_search_iterations=config['local_search_iteration'], epsilon=config['epsilon'])]

        elif config['model_type'] == 'p-nd-np':
            model_list += [solver.NeuralSurveyPropagatorSolver(device=self._device, name=config['model_name'], 
                    edge_dimension=config['edge_feature_dim'], meta_data_dimension=config['meta_feature_dim'], 
                    decimator_dimension=config['hidden_dim'],
                    mem_hidden_dimension=config['mem_hidden_dim'], 
                    agg_hidden_dimension=config['agg_hidden_dim'], mem_agg_hidden_dimension=config['mem_agg_hidden_dim'], 
                    prediction_dimension=config['prediction_dim'], 
                    variable_classifier=Perceptron(config['hidden_dim'], config['classifier_dim'], config['prediction_dim']),
                    function_classifier=None, dropout=config['dropout'],
                    local_search_iterations=config['local_search_iteration'], epsilon=config['epsilon'])]

        elif config['model_type'] == 'np-d-np':
            model_list += [solver.NeuralSequentialDecimatorSolver(device=self._device, name=config['model_name'], 
                    edge_dimension=config['edge_feature_dim'], meta_data_dimension=config['meta_feature_dim'], 
                    propagator_dimension=config['hidden_dim'], decimator_dimension=config['hidden_dim'],
                    mem_hidden_dimension=config['mem_hidden_dim'], 
                    agg_hidden_dimension=config['agg_hidden_dim'], mem_agg_hidden_dimension=config['mem_agg_hidden_dim'], 
                    classifier_dimension=config['classifier_dim'], 
                    dropout=config['dropout'], tolerance=config['tolerance'], t_max=config['t_max'],
                    local_search_iterations=config['local_search_iteration'], epsilon=config['epsilon'])]

        elif config['model_type'] == 'p-d-p':
            model_list += [solver.SurveyPropagatorSolver(device=self._device, name=config['model_name'], 
                    tolerance=config['tolerance'], t_max=config['t_max'],
                    local_search_iterations=config['local_search_iteration'], epsilon=config['epsilon'])]

        elif config['model_type'] == 'walk-sat':
            model_list += [solver.WalkSATSolver(device=self._device, name=config['model_name'],
                    iteration_num=config['local_search_iteration'], epsilon=config['epsilon'])]

        elif config['model_type'] == 'reinforce':
            model_list += [solver.ReinforceSurveyPropagatorSolver(device=self._device, name=config['model_name'],
                    pi=config['pi'], decimation_probability=config['decimation_probability'],
                    local_search_iterations=config['local_search_iteration'], epsilon=config['epsilon'])]

        if config['verbose']:
            self._logger.info("The model parameter count is %d." % model_list[0].parameter_count())
        return model_list

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map, 
        batch_function_map, edge_feature, meta_data):

        return self._loss_evaluator(variable_prediction=prediction[0], label=label, graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=meta_data, global_step=model._global_step, 
            eps=self._eps, max_coeff=self._max_coeff, loss_sharpness=self._config['loss_sharpness'])

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, graph_map, 
        batch_variable_map, batch_function_map, edge_feature, meta_data):

        output, _ = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=meta_data)

        recall = torch.sum(label * ((output > 0.5).float() - label).abs()) / torch.max(torch.sum(label), self._eps)
        accuracy = evaluator((output > 0.5).float(), label).unsqueeze(0)
        loss_value = self._loss_evaluator(variable_prediction=prediction[0], label=label, graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=meta_data, global_step=model._global_step, 
            eps=self._eps, max_coeff=self._max_coeff, loss_sharpness=self._config['loss_sharpness']).unsqueeze(0)

        return torch.cat([accuracy, recall, loss_value], 0)

    def _post_process_predictions(self, model, prediction, graph_map, 
        batch_variable_map, batch_function_map, edge_feature, graph_feat, label, misc_data):
        "Formats the prediction and the output solution into JSON format."

        message = ""
        labs = label.detach().cpu().numpy()

        res = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=graph_feat)
        output, unsat_clause_num = [a.detach().cpu().numpy() for a in res]

        for i in range(output.shape[0]):
            instance = {
                'ID': misc_data[i][0] if len(misc_data[i]) > 0 else "",
                'label': int(labs[i, 0]),
                'solved': int(output[i].flatten()[0] == 1),
                'unsat_clauses': int(unsat_clause_num[i].flatten()[0]),
                'solution': (prediction[0][batch_variable_map == i, 0].detach().cpu().numpy().flatten() > 0.5).astype(int).tolist()
            }
            message += (str(instance).replace("'", '"') + "\n")
            self._counter += 1

        return message

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        "De-actives the CNF examples which the model has already found a SAT solution for."

        output, _ = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=sat_problem._graph_map, 
            batch_variable_map=sat_problem._batch_variable_map, batch_function_map=sat_problem._batch_function_map, 
            edge_feature=sat_problem._edge_feature, meta_data=sat_problem._meta_data)#.detach().cpu().numpy()

        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (output > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
        else:
            active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5)
