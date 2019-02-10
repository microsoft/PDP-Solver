import argparse
from scripts_pytorch import PDP_solver_trainer

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
    PDP_solver_trainer.run(0, args.config, not args.test, args.load_model, 
            args.cpu_mode, args.reset, args.use_generator, args.batch_replication)
