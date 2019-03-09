# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# factor_graph_trainer.py : Defines the trainer base class for the PDP framework.

import os
import time
import math
import multiprocessing

import numpy as np

import torch
import torch.nn as nn

from pdp.factorgraph.dataset import FactorGraphDataset


def _module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# pylint: disable=protected-access
class FactorGraphTrainerBase:
    "Base class of the Factor Graph trainer pipeline (abstract)."

    # pylint: disable=unused-argument
    def __init__(self, config, has_meta_data, error_dim, loss, evaluator, use_cuda, logger):

        self._config = config
        self._logger = logger
        self._use_cuda = use_cuda and torch.cuda.is_available()
        
        if config['verbose']:
            if self._use_cuda:
                self._logger.info('Using GPU...')
            else:
                self._logger.info('Using CPU...')
        
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

        self._error_dim = error_dim
        self._num_cores = multiprocessing.cpu_count()
        self._loss = loss
        self._evaluator = evaluator

        if config['verbose']:
            self._logger.info("The number of CPU cores is %s." % self._num_cores)

        torch.set_num_threads(self._num_cores)

        # Build the network
        self._model_list = [self._set_device(model) for model in self._build_graph(self._config)]

    def _build_graph(self, config):
        "Builds the forward computational graph."
        
        raise NotImplementedError("Subclass must implement abstract method")

    # pylint: disable=unused-argument
    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map, 
        batch_function_map, edge_feature, meta_data):
        "Computes the loss function."
        
        return loss(prediction, label)

    # pylint: disable=unused-argument
    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, graph_map, 
        batch_variable_map, batch_function_map, edge_feature, meta_data):
        "Computes the evaluation function."
        
        return evaluator(prediction, label)

    def _load(self, import_path_base):
        "Loads the model(s) from file."
        
        for model in self._model_list:
            _module(model).load(import_path_base)

    def _save(self, export_path_base):
        "Saves the model(s) to file."

        for model in self._model_list:
            _module(model).save(export_path_base)

    def _reset_global_step(self):
        "Resets the global step counter."
        
        for model in self._model_list:
            _module(model)._global_step.data = torch.tensor(
                [0], dtype=torch.float, device=self._device)

    def _set_device(self, model):
        "Sets the CPU/GPU device."

        if self._use_cuda:
            return nn.DataParallel(model).cuda(self._device)
        return model.cpu()

    def _to_cuda(self, data):
        if isinstance(data, list):
            return data

        if data is not None and self._use_cuda:
            return data.cuda(self._device, non_blocking=True)
        return data

    def get_parameter_list(self):
        "Returns list of dictionaries with models' parameters."
        return [{'params': filter(lambda p: p.requires_grad, model.parameters())}
                for model in self._model_list]

    def _train_epoch(self, train_loader, optimizer):

        train_batch_num = math.ceil(len(train_loader.dataset) / self._config['batch_size'])

        total_loss = np.zeros(len(self._model_list), dtype=np.float32)
        total_example_num = 0

        for (j, data) in enumerate(train_loader, 1):
            segment_num = len(data[0])

            for i in range(segment_num):

                (graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label, _) = [self._to_cuda(d[i]) for d in data]
                total_example_num += (batch_variable_map.max() + 1)

                self._train_batch(total_loss, optimizer, graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label)

                if self._config['verbose']:
                    print("Training epoch with batch of size {:4d} ({:4d}/{:4d}): {:3d}% complete...".format(
                        batch_variable_map.max().item(), total_example_num % self._config['batch_size'], self._config['batch_size'],
                        int(j * 100.0 / train_batch_num)), end='\r')

                del graph_map
                del batch_variable_map
                del batch_function_map
                del edge_feature
                del graph_feat
                del label

            for model in self._model_list:
                _module(model)._global_step += 1

        return total_loss / total_example_num  # max(1, len(train_loader))

    def _train_batch(self, total_loss, optimizer, graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label):

        optimizer.zero_grad()
        lambda_value = torch.tensor([self._config['lambda']], dtype=torch.float32, device=self._device)

        for (i, model) in enumerate(self._model_list):

            state = _module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, 
                edge_feature, graph_feat, self._config['randomized'])

            loss = torch.zeros(1, device=self._device)

            for t in torch.arange(self._config['train_outer_recurrence_num'], dtype=torch.int32, device=self._device):

                prediction, state = model(
                    init_state=state, graph_map=graph_map, batch_variable_map=batch_variable_map, 
                    batch_function_map=batch_function_map, edge_feature=edge_feature, 
                    meta_data=graph_feat, is_training=True, iteration_num=self._config['train_inner_recurrence_num'])

                loss += self._compute_loss(
                            model=_module(model), loss=self._loss, prediction=prediction,
                            label=label, graph_map=graph_map, batch_variable_map=batch_variable_map, 
                            batch_function_map=batch_function_map, edge_feature=edge_feature, meta_data=graph_feat) * \
                    lambda_value.pow((self._config['train_outer_recurrence_num'] - t - 1).float())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self._config['clip_norm'])
            total_loss[i] += loss.detach().cpu().numpy()

            for s in state:
                del s

        optimizer.step()

    def _test_epoch(self, validation_loader, batch_replication):

        test_batch_num = math.ceil(len(validation_loader.dataset) / self._config['batch_size'])

        with torch.no_grad():

            error = np.zeros(
                (self._error_dim, len(self._model_list)), dtype=np.float32)
            total_example_num = 0

            for (j, data) in enumerate(validation_loader, 1):
                segment_num = len(data[0])

                for i in range(segment_num):

                    (graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label, _) = [self._to_cuda(d[i]) for d in data]
                    total_example_num += (batch_variable_map.max() + 1).detach().cpu().numpy()

                    self._test_batch(error, graph_map, batch_variable_map, batch_function_map, 
                        edge_feature, graph_feat, label, batch_replication)

                    if self._config['verbose']:
                        print("Testing epoch with batch of size {:4d} ({:4d}/{:4d}): {:3d}% complete...".format(
                            batch_variable_map.max().item(), total_example_num % self._config['batch_size'], self._config['batch_size'],
                            int(j * 100.0 / test_batch_num)), end='\r')

                del graph_map
                del batch_variable_map
                del batch_function_map
                del edge_feature
                del graph_feat
                del label

                # if self._use_cuda:
                #     torch.cuda.empty_cache()

        return error / total_example_num

    def _test_batch(self, error, graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label, batch_replication):

        this_batch_size = batch_variable_map.max() + 1 
        edge_num = graph_map.size(1)

        for (i, model) in enumerate(self._model_list):

            state = _module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, 
                edge_feature, graph_feat, randomized=True, batch_replication=batch_replication)

            prediction, _ = model(
                init_state=state, graph_map=graph_map, batch_variable_map=batch_variable_map, 
                batch_function_map=batch_function_map, edge_feature=edge_feature, 
                meta_data=graph_feat, is_training=False, iteration_num=self._config['test_recurrence_num'],
                check_termination=self._check_recurrence_termination, batch_replication=batch_replication)

            error[:, i] += (this_batch_size.float() * self._compute_evaluation_metrics(
                model=_module(model), evaluator=self._evaluator,
                prediction=prediction, label=label, graph_map=graph_map, 
                batch_variable_map=batch_variable_map, batch_function_map=batch_function_map, 
                edge_feature=edge_feature, meta_data=graph_feat)).detach().cpu().numpy()

            for p in prediction:
                del p

            for s in state:
                del s

    def _predict_epoch(self, validation_loader, post_processor, batch_replication, file):

        test_batch_num = math.ceil(len(validation_loader.dataset) / self._config['batch_size'])

        with torch.no_grad():

            for (j, data) in enumerate(validation_loader, 1):
                segment_num = len(data[0])

                for i in range(segment_num):

                    (graph_map, batch_variable_map, batch_function_map, 
                    edge_feature, graph_feat, label, misc_data) = [self._to_cuda(d[i]) for d in data]

                    self._predict_batch(graph_map, batch_variable_map, batch_function_map, 
                        edge_feature, graph_feat, label, misc_data, post_processor, batch_replication, file)

                    del graph_map
                    del batch_variable_map
                    del batch_function_map
                    del edge_feature
                    del graph_feat
                    del label

                # if self._config['verbose']:
                #     print("Predicting epoch: %3d%% complete..."
                #           % (j * 100.0 / test_batch_num), end='\r')

    def _predict_batch(self, graph_map, batch_variable_map, batch_function_map, 
        edge_feature, graph_feat, label, misc_data, post_processor, batch_replication, file):

        edge_num = graph_map.size(1)

        for (i, model) in enumerate(self._model_list):

            state = _module(model).get_init_state(graph_map, batch_variable_map, batch_function_map, 
                edge_feature, graph_feat, randomized=False, batch_replication=batch_replication)

            prediction, _ = model(
                init_state=state, graph_map=graph_map, batch_variable_map=batch_variable_map, 
                batch_function_map=batch_function_map, edge_feature=edge_feature, 
                meta_data=graph_feat, is_training=False, iteration_num=self._config['test_recurrence_num'],
                check_termination=self._check_recurrence_termination, batch_replication=batch_replication)

            if post_processor is not None and callable(post_processor):
                message = post_processor(_module(model), prediction, graph_map,
                    batch_variable_map, batch_function_map, edge_feature, graph_feat, label, misc_data)
                print(message, file=file)

            for p in prediction:
                del p

            for s in state:
                del s

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        "De-actives the CNF examples which the model has already found a SAT solution for."
        pass

    def train(self, train_list, validation_list, optimizer, last_export_path_base=None,
              best_export_path_base=None, metric_index=0, load_model=None, reset_step=False,
              generator=None, train_epoch_size=0):
        "Trains the PDP model."

        # Build the input pipeline
        train_loader = FactorGraphDataset.get_loader(
            input_file=train_list[0], limit=self._config['train_batch_limit'],
            hidden_dim=self._config['hidden_dim'], batch_size=self._config['batch_size'], shuffle=True,
            num_workers=self._num_cores, max_cache_size=self._config['max_cache_size'], generator=generator, 
            epoch_size=train_epoch_size)

        validation_loader = FactorGraphDataset.get_loader(
            input_file=validation_list[0], limit=self._config['test_batch_limit'],
            hidden_dim=self._config['hidden_dim'], batch_size=self._config['batch_size'], shuffle=False,
            num_workers=self._num_cores, max_cache_size=self._config['max_cache_size'])

        model_num = len(self._model_list)

        errors = np.zeros(
            (self._error_dim, model_num, self._config['epoch_num'],
             self._config['repetition_num']), dtype=np.float32)

        losses = np.zeros(
            (model_num, self._config['epoch_num'], self._config['repetition_num']),
            dtype=np.float32)

        best_errors = np.repeat(np.inf, model_num)

        if self._use_cuda:
            torch.backends.cudnn.benchmark = True

        for rep in range(self._config['repetition_num']):

            if load_model == "best" and best_export_path_base is not None:
                self._load(best_export_path_base)
            elif load_model == "last" and last_export_path_base is not None:
                self._load(last_export_path_base)

            if reset_step:
                self._reset_global_step()

            for epoch in range(self._config['epoch_num']):

                # Training
                start_time = time.time()
                losses[:, epoch, rep] = self._train_epoch(train_loader, optimizer)

                if self._use_cuda:
                    torch.cuda.empty_cache()

                # Validation
                errors[:, :, epoch, rep] = self._test_epoch(validation_loader, 1)
                duration = time.time() - start_time

                # Checkpoint the best models so far
                if last_export_path_base is not None:
                    for (i, model) in enumerate(self._model_list):
                        _module(model).save(last_export_path_base)

                if best_export_path_base is not None:
                    for (i, model) in enumerate(self._model_list):
                        if errors[metric_index, i, epoch, rep] < best_errors[i]:
                            best_errors[i] = errors[metric_index, i, epoch, rep]
                            _module(model).save(best_export_path_base)

                if self._use_cuda:
                    torch.cuda.empty_cache()

                if self._config['verbose']:
                    message = ''
                    for (i, model) in enumerate(self._model_list):
                        name = _module(model)._name
                        message += 'Step {:d}: {:s} error={:s}, {:s} loss={:5.5f} |'.format(
                            _module(model)._global_step.int()[0], name,
                            np.array_str(errors[:, i, epoch, rep].flatten()),
                            name, losses[i, epoch, rep])

                    self._logger.info('Rep {:2d}, Epoch {:2d}: {:s}'.format(rep + 1, epoch + 1, message))
                    self._logger.info('Time spent: %s seconds' % duration)

        if self._use_cuda:
            torch.backends.cudnn.benchmark = False

        if best_export_path_base is not None:
            # Save losses and errors
            base = os.path.relpath(best_export_path_base)
            np.save(os.path.join(base, "losses"), losses, allow_pickle=False)
            np.save(os.path.join(base, "errors"), errors, allow_pickle=False)

            # Save the model
            self._save(best_export_path_base)

        return self._model_list, errors, losses

    def test(self, test_list, import_path_base=None, batch_replication=1):
        "Tests the PDP model and generates test stats."

        if isinstance(test_list, list):
            test_files = test_list
        elif os.path.isdir(test_list):
            test_files = [os.path.join(test_list, f) for f in os.listdir(test_list) \
                if os.path.isfile(os.path.join(test_list, f)) and f[-5:].lower() == '.json' ]
        elif isinstance(test_list, str):
            test_files = [test_list]
        else:
            return None

        result = []

        for file in test_files:
            # Build the input pipeline
            test_loader = FactorGraphDataset.get_loader(
                input_file=file, limit=self._config['test_batch_limit'],
                hidden_dim=self._config['hidden_dim'], batch_size=self._config['batch_size'], shuffle=False,
                num_workers=self._num_cores, max_cache_size=self._config['max_cache_size'], batch_replication=batch_replication)

            if import_path_base is not None:
                self._load(import_path_base)

            start_time = time.time()
            error = self._test_epoch(test_loader, batch_replication)
            duration = time.time() - start_time

            if self._use_cuda:
                torch.cuda.empty_cache()

            if self._config['verbose']:
                message = ''
                for (i, model) in enumerate(self._model_list):
                    message += '{:s}, dataset:{:s} error={:s}|'.format(
                        _module(model)._name, file, np.array_str(error[:, i].flatten()))

                self._logger.info(message)
                self._logger.info('Time spent: %s seconds' % duration)

            result += [[file, error, duration]]

        return result

    def predict(self, test_list, out_file, import_path_base=None, post_processor=None, batch_replication=1):
        "Produces predictions for the trained PDP model."

        # Build the input pipeline
        test_loader = FactorGraphDataset.get_loader(
            input_file=test_list, limit=self._config['test_batch_limit'],
            hidden_dim=self._config['hidden_dim'], batch_size=self._config['batch_size'], shuffle=False,
            num_workers=self._num_cores, max_cache_size=self._config['max_cache_size'], batch_replication=batch_replication)

        if import_path_base is not None:
            self._load(import_path_base)

        start_time = time.time()
        self._predict_epoch(test_loader, post_processor, batch_replication, out_file)

        duration = time.time() - start_time

        if self._use_cuda:
            torch.cuda.empty_cache()

        if self._config['verbose']:
            self._logger.info('Time spent: %s seconds' % duration)
