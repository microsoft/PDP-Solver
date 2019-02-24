# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# satyr.py : The main script to run a trained PDP solver against a test dataset.

import argparse
import yaml, os, logging
import numpy as np
import torch
from datetime import datetime
from scripts_pytorch import PDP_solver_trainer
import dimacs2json


def run(config, logger):
    "Runs the prediction engine."
    
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    if config['verbose']:
        logger.info("Building the computational graph...")

    predicter = PDP_solver_trainer.SatFactorGraphTrainer(config=config, use_cuda=not config['cpu_mode'], logger=logger)

    if config['verbose']:
        logger.info("Starting the prediction phase...")

    predicter._counter = 0
    predicter.predict(test_list=config['test_path'], import_path_base=config['model_path'], 
        post_processor=predicter._post_process_predictions, batch_replication=config['batch_replication'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', help='The model configuration yaml file')
    parser.add_argument('test_path', help='The input test path')
    parser.add_argument('test_recurrence_num', help='The number of iterations for the PDP', type=int)
    parser.add_argument('-b', '--batch_replication', help='Batch replication factor', type=int, default=1)
    parser.add_argument('-z', '--batch_size', help='Batch size', type=int, default=5000)
    parser.add_argument('-m', '--max_cache_size', help='Maximum cache size', type=int, default=100000)
    parser.add_argument('-l', '--test_batch_limit', help='Memory limit for mini-batches', type=int, default=40000000)
    parser.add_argument('-w', '--local_search_iteration', help='Number of iterations for post-processing local search', type=int, default=100)
    parser.add_argument('-e', '--epsilon', help='Epsilon probablity for post-processing local search', type=float, default=0.5)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    parser.add_argument('-c', '--cpu_mode', help='Run on CPU', action='store_true')
    parser.add_argument('-d', '--dimacs', help='The input folder contains DIMACS files', action='store_true')
    parser.add_argument('-s', '--random_seed', help='Random seed', type=int, default=int(datetime.now().microsecond))

    args = vars(parser.parse_args())

    # Load the model config
    with open(args['model_config'], 'r') as f:
        model_config = yaml.load(f)

    # Set the logger
    format = '[%(levelname)s] %(asctime)s - %(name)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format)
    logger = logging.getLogger(model_config['model_name'])

    # Convert DIMACS input files into JSON
    if args['dimacs']:
        if args['verbose']:
            logger.info("Converting DIMACS files into JSON...")
        temp_file_name = 'temp_problem_file.json'
        
        if os.path.isfile(args['test_path']):
            head, _ = os.path.split(args['test_path'])
            temp_file_name = os.path.join(head, temp_file_name)
            dimacs2json.convert_file(args['test_path'], temp_file_name, False)
        else:
            temp_file_name = os.path.join(args['test_path'], temp_file_name)
            dimacs2json.convert_directory(args['test_path'], temp_file_name, False)

        args['test_path'] = temp_file_name

    # Merge model config and other arguments into one config dict
    config = {**model_config, **args}

    if config['model_type'] == 'p-d-p' or config['model_type'] == 'walk-sat' or config['model_type'] == 'reinforce':
        config['model_path'] = None
        config['hidden_dim'] = 3

    if config['model_type'] == 'walk-sat':
        config['local_search_iteration'] = config['test_recurrence_num']

    config['dropout'] = 0
    config['error_dim'] = 1
    config['exploration'] = 0

    # Run the prediction engine
    run(config, logger)

    if args['dimacs']:
        os.remove(temp_file_name)
