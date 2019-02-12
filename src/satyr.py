# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# satyr.py : The main script to run a trained PDP solver against a test dataset.

import argparse
import time, yaml, sys
import numpy as np
import torch
from scripts_pytorch import PDP_solver_trainer


def run(config):
    "Runs the prediction engine."
    
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    if config['verbose']:
        print("\nBuilding the computational graph...", file=sys.stderr)

    predicter = PDP_solver_trainer.SatFactorGraphTrainer(config=config, use_cuda=not config['cpu_mode'])

    if config['verbose']:
        print("\nStarting the prediction phase...", file=sys.stderr)

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
    parser.add_argument('-l', '--local_search_iteration', help='Number of iterations for post-processing local search', type=int, default=100)
    parser.add_argument('-e', '--epsilon', help='Epsilon probablity for post-processing local search', type=float, default=0.5)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    parser.add_argument('-c', '--cpu_mode', help='Run on CPU', action='store_true')
    parser.add_argument('-s', '--random_seed', help='Random seed', type=int, default=int(round(time.time() * 1000)))

    args = parser.parse_args()

    # Load the model config
    with open(args['model_config'], 'r') as f:
        model_config = yaml.load(f)

    # Merge model config and other arguments into one config dict
    config = {**model_config, **args}

    config['drop_out'] = 0

    # Run the prediction engine
    run(config)
