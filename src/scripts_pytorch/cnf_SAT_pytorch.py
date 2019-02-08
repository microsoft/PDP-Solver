import numpy as np
import matplotlib.pyplot as pp
import os, yaml, csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model_pytorch import factor_graph_trainer
from model_pytorch.PDP import solver, util
from model_pytorch import generators

##########################################################################################################################


class Perceptron(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False)

        # self._bn1 = nn.BatchNorm1d(hidden_dimension)
        # self._bn2 = nn.BatchNorm1d(output_dimension)

    def forward(self, inp):
        # return F.sigmoid(self._bn2(self._layer2(F.relu(self._bn1(self._layer1(inp))))))
        return F.sigmoid(self._layer2(F.relu(self._layer1(inp))))


##########################################################################################################################

class SatFactorGraphTrainer(factor_graph_trainer.FactorGraphTrainerBase):
    def __init__(self, config, use_cuda):
        super(SatFactorGraphTrainer, self).__init__(config=config, 
            has_meta_data=False, error_dim=config['error_dim'], loss=None, evaluator=nn.L1Loss(), use_cuda=use_cuda)

        self._eps = 1e-8 * torch.ones(1, device=self._device)
        self._loss_evaluator = util.SatLossEvaluator(alpha = self._config['exploration'], device = self._device)
        self._cnf_evaluator = util.SatCNFEvaluator(device = self._device)
        self._counter = 0
        self._max_coeff = 10.0

    def _build_graph(self, config):
        model_list = []

        # model_list += [factor_graph_nbp.BeliefPropagator(device=self._device, name=config['model_name'], 
        #         input_dimension=config['edge_feature_dim'], meta_data_dimension=config['meta_feature_dim'], 
        #         hidden_dimension=config['hidden_dim'], mem_hidden_dimension=config['mem_hidden_dim'], 
        #         agg_hidden_dimension=config['agg_hidden_dim'], mem_agg_hidden_dimension=config['mem_agg_hidden_dim'], 
        #         prediction_dim=config['prediction_dim'], 
        #         variable_classifier=Perceptron(config['hidden_dim'] + config['meta_feature_dim'], 
        #             config['classifier_dim'], config['prediction_dim']),
        #         function_classifier=None, dropout=config['dropout'])]

        # model_list += [factor_graph_mp.NeuralPropagatorDecimatorSolver(device=self._device, name=config['model_name'], 
        #         edge_dimension=config['edge_feature_dim'], meta_data_dimension=config['meta_feature_dim'], 
        #         propagator_dimension=config['hidden_dim'], decimator_dimension=config['hidden_dim'],
        #         mem_hidden_dimension=config['mem_hidden_dim'], 
        #         agg_hidden_dimension=config['agg_hidden_dim'], mem_agg_hidden_dimension=config['mem_agg_hidden_dim'], 
        #         prediction_dim=config['prediction_dim'], 
        #         variable_classifier=Perceptron(config['hidden_dim'], config['classifier_dim'], config['prediction_dim']),
        #         function_classifier=None, dropout=config['dropout'])]

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

        print("The model parameter count is %d.\n" % model_list[0].parameter_count())
        return model_list

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map, 
        batch_function_map, edge_feature, meta_data):

        return self._loss_evaluator(variable_prediction=prediction[0], label=label, graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=meta_data, global_step=model._global_step, 
            eps=self._eps, max_coeff=self._max_coeff, loss_sharpness=self._config['loss_sharpness'])

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, graph_map, 
        batch_variable_map, batch_function_map, edge_feature, meta_data):

        output = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=graph_map, 
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
        batch_variable_map, batch_function_map, edge_feature, graph_feat, label):

        message = ""
        labs = label.detach().cpu().numpy()

        output = self._cnf_evaluator(variable_prediction=prediction[0], label=label, graph_map=graph_map, 
            batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
            edge_feature=edge_feature, meta_data=meta_data).detach().cpu().numpy()

        for i in range(output.shape[0]):
            values = output[i].flatten()
            solution = prediction[0].detach().cpu().numpy().flatten()
            message += "\nExample {:s}, Label: {:s}, Prediction: {:s}, Output: {:s}, Solution: \n{:s}\n".format(
                str(self._counter), str(labs[i, :]), "SAT" if values > 0.5 else "UNSAT", str(values), np.array_str(solution))
            self._counter += 1

        return message

    def _check_recurrence_termination(self, active, prediction, sat_problem):

        output = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=sat_problem._graph_map, 
            batch_variable_map=sat_problem._batch_variable_map, batch_function_map=sat_problem._batch_function_map, 
            edge_feature=sat_problem._edge_feature, meta_data=sat_problem._meta_data)#.detach().cpu().numpy()

        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (output > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
        else:
            active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5)

    # def _check_recurrence_termination(self, active, prediction, 
    #     graph_map, batch_variable_map, batch_function_map, edge_feature, graph_feat):

    #     output = self._cnf_evaluator(variable_prediction=prediction[0], graph_map=graph_map, 
    #         batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
    #         edge_feature=edge_feature, meta_data=graph_feat)#.detach().cpu().numpy()
        
    #     active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5)

##########################################################################################################################

def write_to_csv(result_list, file_path):
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in result_list:
            writer.writerow([row[0], row[1][1, 0]])

def write_to_csv_time(result_list, file_path):
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in result_list:
            writer.writerow([row[0], row[2]])

def run(random_seed, config_file, is_training, load_model, cpu, reset_step, use_generator, batch_replication):
    
    if not use_generator:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Set the configurations (from either JSON or YAML file)
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    # Check if the input path is a list or on
    if not isinstance(config['train_path'], list):
        config['train_path'] = [os.path.join(config['train_path'], f) \
            for f in os.listdir(config['train_path']) if os.path.isfile(os.path.join(config['train_path'], f)) and f.endswith('.json')]

    if not isinstance(config['validation_path'], list):
        config['validation_path'] = [os.path.join(config['validation_path'], f) \
            for f in os.listdir(config['validation_path']) if os.path.isfile(os.path.join(config['validation_path'], f)) and f.endswith('.json')]

    print("Training file(s):")
    print(config['train_path'])

    print("Validation file(s):")
    print(config['validation_path'])

    best_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                        config['model_name'], config['version'], "best")

    last_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                        config['model_name'], config['version'], "last")

    if not os.path.exists(best_model_path_base):
        os.makedirs(best_model_path_base)

    if not os.path.exists(last_model_path_base):
        os.makedirs(last_model_path_base)

    trainer = SatFactorGraphTrainer(config=config, use_cuda=not cpu)

    # Training
    if is_training:
        if config['verbose']:
            print("Starting the training phase...")

        generator = None

        if use_generator:
            if config['generator'] == 'modular':
                generator = generators.ModularCNFGenerator(config['min_k'], config['min_n'], config['max_n'], config['min_q'],
                    config['max_q'], config['min_c'], config['max_c'], config['min_alpha'], config['max_alpha'])
            elif config['generator'] == 'v-modular':
                generator = generators.VariableModularCNFGenerator(config['min_k'], config['max_k'], config['min_n'], config['max_n'], config['min_q'],
                    config['max_q'], config['min_c'], config['max_c'], config['min_alpha'], config['max_alpha'])
            else:
                generator = generators.UniformCNFGenerator(config['min_n'], config['max_n'], config['min_k'], config['max_k'], config['min_alpha'], config['max_alpha'])

        model_list, errors, losses = trainer.train(
        	train_list=config['train_path'], validation_list=config['validation_path'], 
            optimizer=optim.Adam(trainer.get_parameter_list(), lr=config['learning_rate'],
            weight_decay=config['weight_decay']), last_export_path_base=last_model_path_base,
            best_export_path_base=best_model_path_base, metric_index=config['metric_index'],
            load_model=load_model, reset_step=reset_step, generator=generator, 
            train_epoch_size=config['train_epoch_size'])

        '''colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        line_styles = ['solid', 'dashed', 'dotted']
        names = ['Accuracy', 'Recall']
        mean_err = np.mean(errors, axis=3)
        mean_loss = np.mean(losses, axis=2)

        pp.subplot(2, 1, 1)
        for i in range(len(model_list)):
            for j in range(config['error_dim']):
                pp.plot(mean_err[j, i, :], c=colors[i], ls=line_styles[j], label=model_list[i]._name + '-' + names[j])
        pp.yscale('log')
        pp.legend(loc='best', shadow=True)

        pp.subplot(2, 1, 2)
        for i in range(len(model_list)):
            pp.plot(mean_loss[i, :], c=colors[i], label=model_list[i]._name)
        pp.yscale('log')
        pp.legend(loc='best', shadow=True)

        pp.show()'''

    if config['verbose']:
        print("\nStarting the test/prediction phase...")

    for test_files in config['test_path']:
        if config['verbose']:
            print("\nTesting " + test_files)

        if load_model == "last":
            import_path_base = last_model_path_base
        elif load_model == "best":
            import_path_base = best_model_path_base
        else:
            import_path_base = None

        result = trainer.test(test_list=test_files, import_path_base=import_path_base, batch_replication=batch_replication)

        if config['verbose']:
            for row in result:
                filename, errors, _ = row
                print('Dataset: ' + filename)
                print("Accuracy: \t%s" % (1 - errors[0]))
                print("Recall: \t%s" % (1 - errors[1]))

        if os.path.isdir(test_files):
            write_to_csv(result, os.path.join(test_files, config['model_type'] + '_' + config['model_name'] + '_' + config['version'] + '-results.csv'))
            write_to_csv_time(result, os.path.join(test_files, config['model_type'] + '_' + config['model_name'] + '_' + config['version'] + '-results-time.csv'))

        # if config['verbose']:
        #     print("\nGenerating prediction file for " + test_files[0])

        # trainer._counter = 0
        # trainer.predict(test_list=test_files, import_path_base=last_model_path_base if load_model == "last" else best_model_path_base, 
        #     post_processor=trainer._post_process_predictions, batch_replication=batch_replication)
