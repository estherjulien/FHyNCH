# Code for the paper:
> ### Inferring Phylogenetic Networks from Multifurcating Trees via Cherry-Picking and Machine Learning
> *Giulia Bernardini, Leo van Iersel, Esther Julien and Leen Stougie*

## Description

This directory contains the code for the cherry-picking heuristic (CPH), with (trivial) random and ML-informed cherry selection. Cherry picking is a solution approach for the Hybridization problem in phylogenetics, with CPH a heuristic solver.

This directory contains the folders `Heuristic`, `LearningCherries`, `NetworkGen` and `RealData`:
- `heuristic`: includes the code for the cherry-picking heuristic (CPH):
  - `heuristic/CPH.py`: code base for the cherry-picking heuristic, (trivial) random and ML-based.
  - `heuristic/features.py`: code for the features used in ML-informed CPH, including updating.
- `learning`: includes the code for making the cherry prediction and leaf prediction ML models:
  - `learning/dm.py`: data manager which includes the pipeline for training data for the ML models.
  - `learning/train_data.py`: code for generating training data.
- `network_gen`: includes the code for making a phylogenetic network:
  - `network_gen/LGT_network.py`: code for making a lateral gene transfer (LGT) network.
  - `network_gen/network_to_tree.py`: code for extracting trees from a network to generate a tree set.
  - `network_gen/tree_to_newick.py`: code for writing a networkx trees to newick format.

It also contains the following execution files:
- `run_cph.py`: 
- `test_data_gen.py`:
- `train_data_gen.py`:
- `train_rf.py`:

## Dependencies
The following Python (3.10.10) packages are required to run this code: 
- `networkx 2.8.4`
- `numpy 1.23.5`
- `pandas 1.5.3`
- `scikit-learn 1.2.2`

## Run experiments
This section includes commands to run the code for a benchmark.

**_Generate Test Instances_**

For synthetic LGT-based instances:
```commandline
python test_data_gen.py --n_inst <> --n_jobs <> 
                         --mis_l_list <> --con_e_list <> --leaf_list <> --tree_list <> --ret_list <> 
```
- `n_inst` is the number of instances per parameter combination.
- `n_jobs` is the number of processes used for generating test data.
- `mis_l_list` is a list containing missing leaf percentages. (default = [0, 10, 20])
- `con_e_list` is a list containing contracted edges percentages. Non-binary parameter for trees. (default = [0, 10, 20])
- `leaf_list` is a list containing number of leaves per tree in the tree set. (default = [20, 50, 100])
- `tree_list` is a list containing tree set sizes. (default = [20, 50, 100])
- `ret_list` is a list containing the number of reticulations of the LGT network. (default = [10, 20, 30])

Example: 
```commandline
python test_data_gen.py --n_inst 10 --n_jobs 4 --mis_l_list 0 --con_e_list 0 --leaf_list 20 --tree_list 20 50 --ret_list 10 30
```

**_Generate Training Data_**
```commandline
python train_data_gen.py --job_id <> --n_inst <> --n_jobs <> --mis_l <>
```
- `job_id` is the job id, used for parallelization over computing nodes.
- `n_inst` is the number of training instances per node.
- `n_jobs` is the number of processes used for generating training data per node.
- `mis_l` missing leaf percentage (in integer).

The ML models in the paper are based on 2000 train instances, which are generated over multiple nodes.

Example:
```commandline
python train_data_gen.py --job_id 1 --n_inst 20 --n_jobs 4 --mis_l 0
```

**_Train Random Forest Models (cherry and leaf)_**
```commandline
python train_rf.py
```

**_Execute Cherry-Picking Heuristic_**
```commandline
python run_cph.py --test_case <> --inst <> --n_trees <> --n_leaves <> --n_rets <>
                  --n_tr_nets <> --mis_l <> --con_e <> --chosen_leaves <>
```
- `test_case` is the test case. Choose between `test/LGT`, `Beiko/small`, and `Beiko/large`.
- `inst` is the instance id.
- `n_trees` the number of trees in the tree set.
- `n_leaves` is the number of leaves per tree.
- `n_rets` is the number of reticulations in the reference network. For `test/LGT` instances, this is the number of reticulations of the original LGT network.
- `mis_l` is the missing leaves percentage of the tree set.
- `con_e` is the contracted edges percentage of the tree set. Serves as parameter for non-binary trees.
- `chosen_leaves` is the number of leaves selected per iteration of CPH.

Example: 
```commandline
python run_cph.py --test_case test/LGT --inst 1 --n_trees 20 --n_leaves 20 --n_rets 10 --n_tr_nets 20 --mis_l 0 --con_e 0 --chosen_leaves 1
```
