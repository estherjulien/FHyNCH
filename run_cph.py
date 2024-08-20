from test_data_gen import *
from utils.tree_child_prep import read_tree_set
from network_gen.network_to_tree import *
import heuristic.CPH as CPH


from datetime import datetime
import pandas as pd
import numpy as np
import copy
import argparse
import csv
import warnings
import os


warnings.simplefilter(action='ignore', category=FutureWarning)


def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
                  progress=False, pick_triv=False, pick_ml=False, pick_ml_triv=False,
                  pick_random=False, cherry_model_name=None, leaf_model_name=None, relabel=False,
                  problem_type="", num_chosen_leaves=1, job_id=None,
                  debug_mode=False, input_trees=False):
    # READ TREE SET
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Start at {now}")

    if tree_set is None and input_trees and tree_set_newick is not None:
        trees, leaves = read_tree_set(newick_file=tree_set_newick)
        for t, tree in trees.items():
            tree = nx.relabel_nodes(tree, {1: len(tree) + 1, 2: len(tree) + 2})
            tree = nx.relabel_nodes(tree, {0: 2})
            tree.add_weighted_edges_from([(1, 2, 0)], weight='length')
            trees[t] = tree
        tree_set = CPH.Input_Set(tree_set=trees, leaves=leaves)

    if tree_set is None and tree_set_newick is not None:
        # Empty set of inputs
        inputs = []

        # Read each line of the input file with name set by "option_file_argument"
        f = open(tree_set_newick, "rt")
        reader = csv.reader(f, delimiter='~', quotechar='|')
        for row in reader:
            new_line = str(row[0])
            if new_line[-1] == ";":
                inputs.append(new_line[:-1])
            else:
                inputs.append(new_line)
        f.close()

        # Make the set of inputs usable for all algorithms: use the CPH class
        tree_set = CPH.Input_Set(newick_strings=inputs, instance=inst_num, job_id=job_id, debug_mode=debug_mode)

    # RUN HEURISTIC CHERRY PICKING SEQUENCE
    # Run the heuristic to find a cherry-picking sequence `seq' for the set of input trees.
    # Arguments are set as given by the terminal arguments
    seq_dist, seq, leaf_probs, cherry_probs = tree_set.CPSBound(repeats=repeats,
                                                                progress=progress,
                                                                time_limit=time_limit,
                                                                pick_triv=pick_triv,
                                                                pick_ml=pick_ml,
                                                                relabel=relabel,
                                                                pick_ml_triv=pick_ml_triv,
                                                                pick_random=pick_random,
                                                                cherry_model_name=cherry_model_name,
                                                                leaf_model_name=leaf_model_name,
                                                                problem_type=problem_type,
                                                                num_chosen_leaves=num_chosen_leaves)

    # Output the computation time for the heuristic
    now = datetime.now().time()
    if progress:
        print(f"Instance {inst_num} {problem_type}: Finish at {now}")
        print(f"Instance {inst_num} {problem_type}: Computation time heuristic: {tree_set.CPS_Compute_Time}")
        print(f"Instance {inst_num} {problem_type}: Reticulation number = {min(tree_set.RetPerTrial.values())}")
    return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq, leaf_probs, cherry_probs


def run_CPH(args, retic=None, repeats=1000, time_limit=None,
            progress=False, file_name="", tree_set_newick="", input_trees=None, job_id=""):
    # ML MODEL
    cherry_model_name = f"data/rf_models/cherry/rf_cherries_{args.n_tr_nets}n.joblib"
    leaf_model_name = f"data/rf_models/leaf/rf_leaves_{args.n_tr_nets}n.joblib"

    # save results
    if args.test_case == "test/LGT":
        columns = ["TrivialML", "TrivialRand", "UB"]
    else:
        columns = ["TrivialML", "TrivialRand"]
    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[args.inst], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)

    # ML Trivial HEURISTIC
    reps = 1
    ret_score, time_score, seq_ml_triv, leaf_probs, cherry_probs = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=args.inst,
        repeats=reps,
        time_limit=time_limit,
        pick_ml_triv=True,
        relabel=True,
        cherry_model_name=cherry_model_name,
        leaf_model_name=leaf_model_name,
        problem_type="TrivialML",
        num_chosen_leaves=args.chosen_leaves,
        progress=progress,
        job_id=job_id,
        input_trees=input_trees)
    for r, ret in ret_score.items():
        score.loc[args.inst, "RetNum", r]["TrivialML"] = copy.copy(ret)
        score.loc[args.inst, "Time", r]["TrivialML"] = copy.copy(time_score[r])
    ml_triv_time = score.loc[args.inst, "Time", :]["TrivialML"].sum()
    ml_triv_ret = int(min(score.loc[args.inst, "RetNum"]["TrivialML"]))

    # TRIVIAL RANDOM
    ret_score, time_score, seq_tr, _, _ = run_heuristic(
        tree_set_newick=tree_set_newick,
        inst_num=args.inst,
        repeats=repeats,
        time_limit=ml_triv_time,
        pick_triv=True,
        relabel=True,
        problem_type="TrivialRand",
        progress=progress,
        job_id=job_id,
        input_trees=input_trees)
    for r, ret in ret_score.items():
        score.loc[args.inst, "RetNum", r]["TrivialRand"] = copy.copy(ret)
        score.loc[args.inst, "Time", r]["TrivialRand"] = copy.copy(time_score[r])
    tr_ret = int(min(score.loc[args.inst, "RetNum"]["TrivialRand"]))
    tr_time = score.loc[args.inst, "Time"]["TrivialRand"].sum()

    # upper bound of ret
    idx = pd.IndexSlice
    if args.test_case == "test/LGT":
        score.loc[idx[args.inst, "RetNum", :], "UB"] = retic
        # print results
        print(args.inst, ml_triv_ret, tr_ret, retic, tr_time)
        os.makedirs("data/test/LGT/results", exist_ok=True)
    else:
        print(args.inst, ml_triv_ret, tr_ret, tr_time)

    # SAVE DATAFRAMES
    # scores
    score.dropna(axis=0, how="all").to_pickle(file_name)


def main(args):
    job_id = f"JOB {args.inst} ("
    if args.test_case == "test/LGT":
        file_info = f"TEST[LGT_L{args.n_leaves}_R{args.n_rets}_T{args.n_trees}_MisL{args.mis_l}_ConE{args.con_e}" \
                    f"_ML[DUO_N{args.n_tr_nets}_CL{args.chosen_leaves}_RF]"
        job_id += f"L{args.n_leaves}, R{args.n_rets}, T{args.n_trees}, MisL{args.mis_l}, ConE{args.con_e}, RF)"
        input_trees = False
        newick = f"data/{args.test_case}/newick/tree_set_newick_L{args.n_leaves}_R{args.n_rets}_T{args.n_trees}_MisL{args.mis_l}_ConE{args.con_e}_LGT_{args.inst}.txt"
        file_name = f"data/{args.test_case}/results/heuristic_scores_{file_info}_{args.inst}"
        file_name += ".pkl"
    else:
        input_trees = True
        job_id += f"{args.test_case}, L{args.n_leaves}, T{args.n_trees})"
        newick = f"data/{args.test_case}/newick/test_T{args.n_trees}_L{args.n_leaves}_MisL{args.mis_l}_{args.inst}.txt"
        file_name = f"data/{args.test_case}/results/heuristic_scores_T{args.n_trees}_L{args.n_leaves}_MisL{args.mis_l}_{args.inst}"
        file_name += ".pkl"

    os.makedirs("data/" + args.test_case + "/results", exist_ok=True)
    # RUN CPH
    run_CPH(args, progress=False, file_name=file_name, tree_set_newick=newick, input_trees=input_trees,
            job_id=job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_case', type=str, default="test/LGT", choices=["test/LGT",
                                                                              "Beiko/small",
                                                                              "Beiko/large"])
    parser.add_argument('--inst', type=int)
    parser.add_argument('--n_trees', type=int)
    parser.add_argument('--n_leaves', type=int)
    parser.add_argument('--n_rets', type=int, default=-1, help="number of reticulations in the reference network")
    parser.add_argument('--n_tr_nets', type=int, default=2000)
    parser.add_argument('--mis_l', type=int, default=0, help="missing leaves percentage")
    parser.add_argument('--con_e', type=int, default=0, help="contracted edges percentage")
    parser.add_argument('--chosen_leaves', type=int, default=1, help="number of chosen leaves per MultiML iteration")

    args = parser.parse_args()
    main(args)
