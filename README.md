# PDP Framework for Neural Constraint Satisfaction Solving

The PDP framework is a generic framework based on the idea of Propagation, Decimation and Prediction (PDP) for learning and implementing message passing-based solvers for constraint satisfaction problems (CSP). In particular, it provides an elegant unsupervised framework for training neural solvers based on the idea of energy minimization. Our SAT solver adaptation of the PDP framework, referred as **SATYR**, supports a wide spectrum of solvers from fully neural architectures to classical inference-based techniques (such as Survey Propagation) with hybrid methods in between. For further theoretical details of the framework, please refer to our paper:

**Saeed Amizadeh, Sergiy Matusevych and Markus Weimer, [PDP: A General Neural Framework for Learning Constraint Satisfaction Solvers](https://arxiv.org/abs/1903.01969), arXiv preprint arXiv:1903.01969, 2019.**

```
@article{amizadeh2019pdp,
  title={PDP: A General Neural Framework for Learning Constraint Satisfaction Solvers},
  author={Amizadeh, Saeed and Matusevych, Sergiy and Weimer, Markus},
  journal={arXiv preprint arXiv:1903.01969},
  year={2019}
}
```

We also note the present work is still far away from competing with modern industrial solvers; nevertheless, we believe it is a significant step in the right direction for machine learning-based methods. Hence, we are glad to open source our code to the researchers in related fields including the neuro-symbolic community as well as the classical SAT community.

# SATYR

SATYR is the adaptation of the PDP framework for training and deploying neural Boolean Satisfiability solvers. In particular, SATYR implements:

1. Fully or partially neural SAT solvers that can be trained toward solving SAT for a specific distribution of problem instances. The training is based on unsupervised energy minimization and can be performed on an infinite stream of unlabeled, random instances sampled from the target distribution.

2. Non-learnable classical solvers based on message passing in graphical models (e.g. Survey Propagation). Even though, these solvers are non-learnable, they still benefit from the embarrassingly parallel implementation via the PDP framework on GPUs.

It should be noted that all the SATYR solvers try to find a satisfying assignment for input SAT formulas. However, if the SATYR solvers cannot find a satisfying solution for a given problem within their iteration number budget, it does NOT necessarily mean that the input problem in UNSAT. In other words, none of the SATYR solvers provide the proof of unsatisfiability.

# Setup

## Prerequisites

* Python 3.5 or higher.
* PyTorch 0.4.0 or higher.

Run:

```
> python setup.py
```

# Usage

The SATYR solvers can be used in two main modes: (1) apply an already-trained model or non-ML algorithm to test data, and (2) train/test new models.

## Running a (Trained) SATYR Solver

The usage for running a SATYR solver againts a set of SAT problems (represented as Conjunctive Normal Form (CNF)) is:

```
> python satyr.py [-h] [-b BATCH_REPLICATION] [-z BATCH_SIZE]
                [-m MAX_CACHE_SIZE] [-l TEST_BATCH_LIMIT]
                [-w LOCAL_SEARCH_ITERATION] [-e EPSILON] [-v] [-c] [-d]
                [-s RANDOM_SEED] [-o OUTPUT]
                model_config test_path test_recurrence_num
```

The commandline arguments are:

+ **-h** or **--help**: Shows the commandline options.

+ **-b BATCH_REPLICATION** or **--batch_replication BATCH_REPLICATION**: BATCH_REPLICATION is the replication factor for input problems to further benefit from parallelization (default: 1).

+ **-z BATCH_SIZE** or **--batch_size BATCH_SIZE**: BATCH_SIZE is the batch size (default: 5000).

+ **-m MAX_CACHE_SIZE** or **--max_cache_size MAX_CACHE_SIZE**: MAX_CACHE_SIZE is the maximum size of the cache containing the parsed CNFs loaded from disk (mostly useful for iterative training) (default: 100000).

+ **-l TEST_BATCH_LIMIT** or **--test_batch_limit TEST_BATCH_LIMIT**: TEST_BATCH_LIMIT is the memory limit used for dynamic batching. It must be empirically tuned by the user depending on the available GPU memory (default: 40000000).

+ **-w LOCAL_SEARCH_ITERATION** or **--local_search_iteration LOCAL_SEARCH_ITERATION**: LOCAL_SEARCH_ITERATION is the maximum number of local search (i.e. Walk-SAT) iterations that can be optionally applied as a post-processing step after the main solver terminates (default: 100).

+ **-e EPSILON** or **--epsilon EPSILON**: EPSILON is the probability with which the post-processing local search picks a random variable instead of the best option for flipping (default: 0.5).

+ **-v** or **--verbose**: Prints the log messages to STDOUT.

+ **-c** or **--cpu_mode**: Forces the solver to run on CPU.

+ **-d** or **--dimacs**: Notifies the solver that the input path is a directory of DIMACS files.

+ **-s RANDOM_SEED** or **--random_seed RANDOM_SEED**: RANDOM_SEED is the random seed directly affecting the randomized initial values of the messages in the PDP solvers.

+ **-o OUTPUT** or **--output OUTPUT**: OUTPUT is the path to the output JSON file that would contain the solutions for the input CNFs. If not specified, the output is directed to STDOUT.

+ **model_config**: The path to YAML config file specifying the model used for SAT solving. A few example config files are provided [here](https://github.com/Microsoft/PDP-Solver/tree/master/config/Predict). A model config file specifies the following properties:

    * **model_type**: The type of solver. So far, we have implemented six different types of solvers in SATYR:

        * *'np-nd-np'*: A fully neural PDP solver.

        * *'p-d-p'*: A PDP solver that implements the classical Survey Propagation with greedy sequential decimation.[[1]](#reference-1)

        * *'p-nd-np'*: A PDP solver that implements the classical Survey Propagation except with neural decimation.

        * *'np-d-np'*: A PDP solver that implements neural propagation with a greedy sequential decimation.

        * *'reinforce'*: A PDP solver that implements the classical Survey Propagation with concurrent distributed decimation (The REINFORCE algorithm).[[2]](#reference-2)

        * *'walk-sat'*: A PDP solver that implements the classical local search Walk-SAT algorithm.[[3]](#reference-3)

    * **has_meta_data**: whether the input problem instances contain meta features other than the CNF itself (Note: loading of such features is not supported by the current input pipeline).

    * **model_name**: The name picked for the model by the user.

    * **model_path**: The path to the saved weights for the trained model.

    * **label_dim**: Always set to 1 for SATYR.

    * **edge_feature_dim**: Always set to 1 for SATYR.

    * **meta_feature_dim**: The dimensionality of the meta features (0 for now).

    * **prediction_dim**: Always set to 1 for SATYR.

    If **model_type** is *'np-nd-np'*, *'p-nd-np'* or *'np-d-np'*:

    * **hidden_dim**: The dimensionality of messages between the propagator and the decimator.

    If **model_type** is *'np-nd-np'* or *'np-d-np'*:

    * **mem_hidden_dim**: The dimensionality of the hidden layer for the perceptron that is applied to messages *before* the aggregation step in the propagator.

    * **agg_hidden_dim**: The dimensionality of the hidden layer for the perceptron that is applied to messages *after* the aggregation step in the propagator.

    * **mem_agg_hidden_dim**: The output dimensionality of the perceptron that is applied to messages *before* the aggregation step in the propagator.

    If **model_type** is *'np-nd-np'*, *'p-nd-np'* or *'np-d-np'*:

    * **classifier_dim**: The dimensionality of the hidden layer for the perceptron that is used as the final predictor.

    If **model_type** is *'p-d-p'* or *'np-d-np'*:

    * **tolerance**: The convergence tolerance for the propagator before sequential decimator is invoked.

    * **t_max**: The maximum iteration number for the propagator before sequential decimator is invoked.

    If **model_type** is *'reinforce'*:

    * **pi**: The external force magnitude parameter for the REINFORCE algorithm.

    * **decimation_probability**: The probability with which the distributed decimation is invoked in the REINFORCE algorithm.

+ **test_path**: The path to the input JSON file containing the test CNFs (in the case of '-d' option, the path to the directory containing the test DIMACS files.)

+ **test_recurrence_num**: The maximum number of iterations the solver is allowed to run before termination.


## Training/Testing a SATYR Solver

The usage for training/testing new SATYR models is:

```
> python satyr-train-test.py [-h] [-t] [-l LOAD_MODEL] [-c] [-r] [-g]
                           [-b BATCH_REPLICATION]
                           config
```

The commandline arguments are:

+ **-h** or **--help**: Shows the commandline options.

+ **-t** or **--test**: Skips the training stage directly to the testing stage.

+ **-l LOAD_MODEL** or **-load LOAD_MODEL**: LOAD_MODEL is:

    * *best*: The model is initialized by the best model (according to the validation metric) saved from the previous run.

    * *last*: The model is initialized by the last model saved from the previous run.

    * Otherwise, the model is initialized by random weights.

+ **-c** or **--cpu_mode**: Forces the training/testing to run on CPU.

+ **-r** or **--reset**: Resets the global time parameter to 0 (used for annealing the temperature).

+ **-g** or **--use_generator**: Makes the training process use one of the provided CNF generators to generate unlabeled training CNF instances on the fly.

+ **-b BATCH_REPLICATION** or **--batch_replication BATCH_REPLICATION**: BATCH_REPLICATION is the replication factor for input problems to further benefit from parallelization (default: 1).

+ **config**: The path to YAML config file specifying the model as well as the training parameters. A few example training config files are provided [here](https://github.com/Microsoft/PDP-Solver/tree/master/config/Train). A training config file specifies the following properties:

    * **model_name**: The name picked for the model by the user.

    * **model_type**: The model type explained above.

    * **version**: The model version.

    * **has_meta_data**: whether the input problem instances contain meta features other than the CNF itself (Note: loading of such features is not supported by the current input pipeline).

    * **train_path**: A one-element list containing the path to the training JSON file. Will be ignored in the case of using option -g.

    * **validation_path**: A one-element list containing the path to the validation JSON file. Validation set is used for picking the best model during each training run.

    * **test_path**: A list containing the path(s) to test JSON files.

    * **model_path**: The parent directory of the location where the best and the last models are saved.

    * **repetition_num**: Number of repetitions for the training process (for regular scenarios: 1).

    * **train_epoch_size**: The size of one epoch in the case of using CNF generators via option -g.

    * **epoch_num**: The number of epochs for training.

    * **label_dim**: Always set to 1 for SATYR.

    * **edge_feature_dim**: Always set to 1 for SATYR.

    * **meta_feature_dim**: The dimensionality of the meta features (0 for now).

    * **error_dim**: The number error metrics the model reports on the validation/test sets (3 for now: accuracy, recall and test loss).

    * **metric_index**: The 0-based index of the error metric used to pick the best model.

    * **prediction_dim**: Always set to 1 for SATYR.

    * **batch_size**: The batch size used for training/testing.

    * **learning_rate**: The learning rate for ADAM optimization algorithm used for training.

    * **exploration**: The exploration factor used for annealing the temperature.

    * **verbose**: If TRUE prints log messages to STDOUT.

    * **randomized**: If TRUE initializes the propagator and the decimator messages with random values; otherwise with zeros.

    * **train_inner_recurrence_num**: The number of inner loop iterations before the loss function is computed (typically is set to 1).

    * **train_outer_recurrence_num**: The number of outer loop iterations (T in the paper) used during training.

    * **test_recurrence_num**: The number of outer loop iterations (T in the paper) used during testing.

    * **max_cache_size**: The maximum size of the cache containing the parsed CNFs loaded from disk during training.

    * **dropout**: The dropout factor during training.

    * **clip_norm**: The clip norm ratio used for gradient clipping during training.

    * **weight_decay**: The weight decay coefficient for ADAM optimizer used for training.

    * **loss_sharpness**: The sharpness of the step function used for calculating loss (the kappa parameter in the paper).

    + **train_batch_limit**: The memory limit used for dynamic batching during training. It must be empirically tuned by the user depending on the available GPU memory.

    + **test_batch_limit**: The memory limit used for dynamic batching during testing. It must be empirically tuned by the user depending on the available GPU memory.

    * **generator**: The type of CNF generator incorporated in the case -g option is deployed:

        * *'uniform'*: The uniform random k-SAT generator.

        * *'modular'*: The modular random k-SAT generator with fixed k (specified by **min_k**) according to the Community Attachment model[[4]](#reference-4).

        * *'v-modular'*: The modular random k-SAT generator with variable size k according to the Community Attachment model.

    * **min_n**: The minimum number of variables for a random training CNF instance in the case -g option is deployed.

    * **max_n**: The maximum number of variables for a random training CNF instance in the case -g option is deployed.

    * **min_alpha**: The minimum clause/variable ratio for a random training CNF instance in the case -g option is deployed.

    * **max_alpha**: The maximum clause/variable ratio for a random training CNF instance in the case -g option is deployed.

    * **min_k**: The minimum clause size for a random training CNF instance in the case -g option is deployed.

    * **max_k**: The maximum clause size for a random training CNF instance in the case -g option is deployed (not supported for *'v-modular'* generator).

    * **min_q**: The minimum modularity value for a random training CNF instance generated according to the Community Attachment model in the case -g option is deployed (not supported for *'uniform'* generator).

    * **max_q**: The maximum modularity value for a random training CNF instance generated according to the Community Attachment model in the case -g option is deployed (not supported for *'uniform'* generator).

    * **min_c**: The minimum number of communities for a random training CNF instance generated according to the Community Attachment model in the case -g option is deployed (not supported for *'uniform'* generator).

    * **max_c**: The maximum number of communities for a random training CNF instance generated according to the Community Attachment model in the case -g option is deployed (not supported for *'uniform'* generator).

    * **local_search_iteration**: The maximum number of local search (i.e. Walk-SAT) iterations that can be optionally applied as a post-processing step after the main solver terminates during testing.

    * **epsilon**: The probability with which the optional post-processing local search picks a random variable instead of the best option for flipping.

    * **lambda**: The discounting factor in (0, 1] used for loss calculation (the lambda parameter in the paper).

    If **model_type** is *'np-nd-np'*, *'p-nd-np'* or *'np-d-np'*:

    * **hidden_dim**: The dimensionality of messages between the propagator and the decimator.

    If **model_type** is *'np-nd-np'* or *'np-d-np'*:

    * **mem_hidden_dim**: The dimensionality of the hidden layer for the perceptron that is applied to messages *before* the aggregation step in the propagator.

    * **agg_hidden_dim**: The dimensionality of the hidden layer for the perceptron that is applied to messages *after* the aggregation step in the propagator.

    * **mem_agg_hidden_dim**: The output dimensionality of the perceptron that is applied to messages *before* the aggregation step in the propagator.

    If **model_type** is *'np-nd-np'*, *'p-nd-np'* or *'np-d-np'*:

    * **classifier_dim**: The dimensionality of the hidden layer for the perceptron that is used as the final predictor.

    If **model_type** is *'p-d-p'* or *'np-d-np'*:

    * **tolerance**: The convergence tolerance for the propagator before sequential decimator is invoked.

    * **t_max**: The maximum iteration number for the propagator before sequential decimator is invoked.

    If **model_type** is *'reinforce'*:

    * **pi**: The external force magnitude parameter for the REINFORCE algorithm.

    * **decimation_probability**: The probability with which the distributed decimation is invoked in the REINFORCE algorithm.

# Input/Output Formats

## Input

SATYR effectively works with the standard DIMACS format for representing CNF formulas. However, in order to increase the ingressing efficiency, the actual solvers work directly with an intermediate JSON format instead of the DIMACS representation for consuming input CNF data. A key feature of the intermediate JSON format is that an entire set of DIMACS files can be represented by a single JSON file where each row in the JSON file associates with one DIMACS file.

The train/test script assumes the train/validation/test sets are already in the JSON format. In order to convert a set of DIMACS files into a single JSON file, we have provided the following script:

```
> python dimacs2json.py [-h] [-s] [-p] in_dir out_file
```

where the commandline arguments are:

+ **-h** or **--help**: Shows the commandline options.

+ **-s** or **--simplify**: Performs elementary clause propagation simplification on the CNF formulas before converting them to JSON format. This option is not recommended for large formulas as it takes quadratic memory and time in terms of the number of clauses.

+ **-p** or **--positives**: Writes only the satisfiable examples in the output JSON file. This option is specially useful for creating all SAT validation/test sets. Note that this option does NOT invoke any external solver to find out whether an example is SAT or not. Instead, it only works if the SAT/UNSAT labels are already provided in the names of the DIMACS files. In particular, if the name of an input DIMACS file ends in '1', it will be regarded as a SAT (positive) example.

+ **in_dir**: The path to the parent directory of the input DIMACS files.

+ **out_file**: The path of the output JSON file.

The solver script, however, does not require the input problems to be in the JSON format; they can be in the DIMACS format as long as the -d option is deployed. Nevertheless, for repetitive applications of the solver script on the same input set, we would recommend externally converting the input DIMACS files into the JSON format once and only consume the JSON file afterwards.

## Output

The output of the solver script is a JSON file where each line corresponds to one input CNF instance and is a dictionary with the following key:value pairs:

+ **"ID"**: The DIMACS file name associated with a CNF example.

+ **"label"**: The binary SAT/UNSAT (0/1) label associated with a CNF example (only if it is already provided in the DIMACS filename).

+ **"solved"**: The binary flag showing whether the provided solution satisfies the CNF.

+ **"unsat_clauses"**: The number of clauses in the CNF that are not satisfied by the provided solution (0 if the CNF is satisfied by the solution).

+ **"solution"**: The provided solution by the solver. The variable assignments in the list are ordered based on the increasing variable indices in the original DIMACS file.

# Main Contributors

+ [Saeed Amizadeh](mailto:saeed.amizadeh@gmail.com), Microsoft Inc.
+ [Sergiy Matusevych](mailto:sergiy.matusevych@gmail.com), Microsoft Inc.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Extending The PDP Framework

The PDP framework supports a wide range of solvers from fully neural solvers to hybrid, neuro-symbolic models all the way to classical, non-learnable algorithms. So far, we have only implemented six different types, but there is definitely room for more. Therefore, we highly encourage contributions in the form of other types of PDP-based solvers. Furthermore, we welcome contributions with PDP-based adaptations for other types of constraint satisfaction problems beyond SAT.

# References

1. <a id="reference-1"></a>Mezard, M. and Montanari, A. Information, physics, and computation. Oxford University Press, 2009.
2. <a id="reference-2"></a>Chavas, J., Furtlehner, C., Mezard, M., and Zecchina, R. Survey-propagation decimation through distributed local computations. Journal of Statistical Mechanics: Theory and Experiment, 2005(11):P11016, 2005.
3. <a id="reference-3"></a>Hoos, Holger H. On the Run-time Behaviour of Stochastic Local Search Algorithms for SAT. In AAAI/IAAI, pp. 661-666. 1999.
4. <a id="reference-4"></a>Giraldez-Cru, J. and Levy, J. Generating sat instances with community structure. Artificial Intelligence, 238:119–134, 2016.
