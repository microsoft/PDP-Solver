#!/usr/bin/env python3
"""The main entry point to the PDP trainer/tester/predictor."""

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import numpy as np
import torch
import torch.optim as optim
import logging
import argparse, os, yaml, csv

from pdp.generator import *
from pdp.trainer import SatFactorGraphTrainer


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
    "Runs the train/test/predict procedures."
    
    if not use_generator:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Set the configurations (from either JSON or YAML file)
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    # Set the logger
    format = '[%(levelname)s] %(asctime)s - %(name)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format)
    logger = logging.getLogger(config['model_name'] + ' (' + config['version'] + ')')

    # Check if the input path is a list or on
    if not isinstance(config['train_path'], list):
        config['train_path'] = [os.path.join(config['train_path'], f) \
            for f in os.listdir(config['train_path']) if os.path.isfile(os.path.join(config['train_path'], f)) and f.endswith('.json')]

    if not isinstance(config['validation_path'], list):
        config['validation_path'] = [os.path.join(config['validation_path'], f) \
            for f in os.listdir(config['validation_path']) if os.path.isfile(os.path.join(config['validation_path'], f)) and f.endswith('.json')]

    if config['verbose']:
        if use_generator:
            logger.info("Generating training examples via %s generator." % config['generator'])
        else:
            logger.info("Training file(s): %s" % config['train_path'])
        logger.info("Validation file(s): %s" % config['validation_path'])

    best_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                        config['model_name'], config['version'], "best")

    last_model_path_base = os.path.join(os.path.relpath(config['model_path']),
                                        config['model_name'], config['version'], "last")

    if not os.path.exists(best_model_path_base):
        os.makedirs(best_model_path_base)

    if not os.path.exists(last_model_path_base):
        os.makedirs(last_model_path_base)

    trainer = SatFactorGraphTrainer(config=config, use_cuda=not cpu, logger=logger)

    # Training
    if is_training:
        if config['verbose']:
            logger.info("Starting the training phase...")

        generator = None

        if use_generator:
            if config['generator'] == 'modular':
                generator = ModularCNFGenerator(config['min_k'], config['min_n'], config['max_n'], config['min_q'],
                    config['max_q'], config['min_c'], config['max_c'], config['min_alpha'], config['max_alpha'])
            elif config['generator'] == 'v-modular':
                generator = VariableModularCNFGenerator(config['min_k'], config['max_k'], config['min_n'], config['max_n'], config['min_q'],
                    config['max_q'], config['min_c'], config['max_c'], config['min_alpha'], config['max_alpha'])
            else:
                generator = UniformCNFGenerator(config['min_n'], config['max_n'], config['min_k'], config['max_k'], config['min_alpha'], config['max_alpha'])

        model_list, errors, losses = trainer.train(
        	train_list=config['train_path'], validation_list=config['validation_path'], 
            optimizer=optim.Adam(trainer.get_parameter_list(), lr=config['learning_rate'],
            weight_decay=config['weight_decay']), last_export_path_base=last_model_path_base,
            best_export_path_base=best_model_path_base, metric_index=config['metric_index'],
            load_model=load_model, reset_step=reset_step, generator=generator, 
            train_epoch_size=config['train_epoch_size'])

    if config['verbose']:
        logger.info("Starting the test phase...")

    for test_files in config['test_path']:
        if config['verbose']:
            logger.info("Testing " + test_files)

        if load_model == "last":
            import_path_base = last_model_path_base
        elif load_model == "best":
            import_path_base = best_model_path_base
        else:
            import_path_base = None

        result = trainer.test(test_list=test_files, import_path_base=import_path_base, 
                        batch_replication=batch_replication)

        if config['verbose']:
            for row in result:
                filename, errors, _ = row
                print('Dataset: ' + filename)
                print("Accuracy: \t%s" % (1 - errors[0]))
                print("Recall: \t%s" % (1 - errors[1]))

        if os.path.isdir(test_files):
            write_to_csv(result, os.path.join(test_files, config['model_type'] + '_' + config['model_name'] + '_' + config['version'] + '-results.csv'))
            write_to_csv_time(result, os.path.join(test_files, config['model_type'] + '_' + config['model_name'] + '_' + config['version'] + '-results-time.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The configuration JSON file')
    parser.add_argument('-t', '--test', help='The test mode', action='store_true')
    parser.add_argument('-l', '--load_model', help='Load the previous model')
    parser.add_argument('-c', '--cpu_mode', help='Run on CPU', action='store_true')
    parser.add_argument('-r', '--reset', help='Reset the global step', action='store_true')
    parser.add_argument('-g', '--use_generator', help='Reset the global step', action='store_true')
    parser.add_argument('-b', '--batch_replication', help='Batch replication factor', type=int, default=1)

    args = parser.parse_args()
    run(0, args.config, not args.test, args.load_model, 
            args.cpu_mode, args.reset, args.use_generator, args.batch_replication)
