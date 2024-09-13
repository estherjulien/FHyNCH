from test_data_gen import *
from utils.tree_child_prep import read_tree_set
from network_gen.network_to_tree import *
import heuristic.CPH as CPH
from phylox import DiNetwork

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
                  debug_mode=False, input_trees=False, return_network=False):
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
        tree_set = CPH.Input_Set(newick_strings=inputs, instance=inst_num, job_id=job_id)

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

    net_newick_str = None
    if return_network:
        net_newick_str = DiNetwork.from_cherry_picking_sequence(seq).newick(simple=True)

    return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq, leaf_probs, cherry_probs, net_newick_str


def run_CPH(args, retic=None, repeats=1000, time_limit=None,
            progress=False, results_path="", tree_set_newick="", input_trees=None, job_id="",
            return_network=False):
    # save results
    columns = []
    if args.run_multi_ml:
        columns.append("MultiML")
    if args.run_trivial_rand:
        columns.append("TrivialRand")
    if args.data_from_paper and args.test_path == "test/LGT":
        columns.append("UB")
    score = pd.DataFrame(
        index=pd.MultiIndex.from_product([[args.inst], ["RetNum", "Time"], np.arange(repeats)]),
        columns=columns, dtype=float)

    # MultiML HEURISTIC
    if args.run_multi_ml:
        # ML MODEL
        cherry_model_name = f"data/rf_cherries_{args.n_tr_nets}n.joblib"
        leaf_model_name = f"data/rf_leaves_{args.n_tr_nets}n.joblib"
        reps = 1
        ret_score, time_score, seq_ml_triv, leaf_probs, cherry_probs, multi_ml_net = run_heuristic(
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
            input_trees=input_trees,
            return_network=return_network)
        for r, ret in ret_score.items():
            score.loc[args.inst, "RetNum", r]["MultiML"] = copy.copy(ret)
            score.loc[args.inst, "Time", r]["MultiML"] = copy.copy(time_score[r])
        ml_triv_time = score.loc[args.inst, "Time", :]["MultiML"].sum()
        ml_triv_ret = int(min(score.loc[args.inst, "RetNum"]["MultiML"]))
    else:
        ml_triv_time, ml_triv_ret, multi_ml_net = None, None, None

    # TRIVIAL RANDOM
    if args.run_trivial_rand:
        if args.run_multi_ml and args.multiml_time_limit:
            tr_time_limit = ml_triv_time
        else:
            tr_time_limit = time_limit

        ret_score, time_score, seq_tr, _, _, tr_net= run_heuristic(
            tree_set_newick=tree_set_newick,
            inst_num=args.inst,
            repeats=repeats,
            time_limit=tr_time_limit,
            pick_triv=True,
            relabel=True,
            problem_type="TrivialRand",
            progress=progress,
            job_id=job_id,
            input_trees=input_trees,
            return_network=return_network)
        for r, ret in ret_score.items():
            score.loc[args.inst, "RetNum", r]["TrivialRand"] = copy.copy(ret)
            score.loc[args.inst, "Time", r]["TrivialRand"] = copy.copy(time_score[r])
        tr_time = score.loc[args.inst, "Time"]["TrivialRand"].sum()
        tr_ret = int(min(score.loc[args.inst, "RetNum"]["TrivialRand"]))
    else:
        tr_time, tr_ret, tr_net = None, None, None
    # upper bound of ret
    if args.data_from_paper and args.test_path == "LGT":
        idx = pd.IndexSlice
        score.loc[idx[args.inst, "RetNum", :], "UB"] = retic
        # print results
        print(args.inst, ml_triv_ret, tr_ret, retic, tr_time)
        os.makedirs(f"data/{args.test_path}/results", exist_ok=True)
    else:
        print(args.inst, ml_triv_ret, tr_ret, tr_time)

    # SAVE OUTPUT
    # scores
    score_path = results_path + "_retics_time" + ".csv"
    score.dropna(axis=0, how="all").to_csv(score_path)

    # networks
    if args.return_network:
        net_path = results_path + "_networks" + ".txt"
        f = open(net_path, "w")
        if args.run_multi_ml:
            f.write(f"MultiML seq: " + ', '.join([f"({item[0]}, {item[1]})" for item in seq_ml_triv]) + "\n")
            f.write(f"MultiML newick: " + multi_ml_net + "\n")
        if args.run_trivial_rand:
            f.write(f"TrivialRand seq: " + ', '.join([f"({item[0]}, {item[1]})" for item in seq_tr]) + "\n")
            f.write(f"TrivialRand newick: " + tr_net + "\n")
        f.close()

def main(args):
    job_id = f"JOB {args.inst} ("

    if not args.data_from_paper:
        newick = args.input_path
        results_path = args.results_path
        input_trees = True

    elif args.data_from_paper and args.test_path == "LGT":
        file_info = f"TEST[LGT_L{args.n_leaves}_R{args.n_rets}_T{args.n_trees}_MisL{args.mis_l}_ConE{args.con_e}" \
                    f"_ML[DUO_N{args.n_tr_nets}_CL{args.chosen_leaves}_RF]"
        job_id += f"L{args.n_leaves}, R{args.n_rets}, T{args.n_trees}, MisL{args.mis_l}, ConE{args.con_e}, RF)"
        input_trees = False
        newick = f"data/{args.test_path}/newick/tree_set_newick_L{args.n_leaves}_R{args.n_rets}_T{args.n_trees}_MisL{args.mis_l}_ConE{args.con_e}_LGT_{args.inst}.txt"
        results_path = f"data/{args.test_path}/results/heuristic_scores_{file_info}_{args.inst}"
        os.makedirs("data/" + args.test_path + "/results", exist_ok=True)
    elif args.data_from_paper:
        input_trees = True
        job_id += f"{args.test_path}, L{args.n_leaves}, T{args.n_trees})"
        newick = f"data/{args.test_path}/newick/test_T{args.n_trees}_L{args.n_leaves}_MisL{args.mis_l}_{args.inst}.txt"
        results_path = f"data/{args.test_path}/results/heuristic_scores_T{args.n_trees}_L{args.n_leaves}_MisL{args.mis_l}_{args.inst}"
        os.makedirs("data/" + args.test_path + "/results", exist_ok=True)

    if args.time_limit == -1:
        time_limit = None
    else:
        time_limit = 60 * args.time_limit
    # RUN CPH
    run_CPH(args, progress=args.verbose, results_path=results_path, time_limit=time_limit, repeats=args.n_triv_rand_its,
            tree_set_newick=newick, input_trees=input_trees, job_id=job_id, return_network=args.return_network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default="", help="The name of the input file that contains the set of"
                                                                   " input trees in Newick format, one tree per line.")
    parser.add_argument('--results_path', type=str, default="")
    parser.add_argument('--run_multi_ml', type=int, default=1, help="Boolean for running MultiML")
    parser.add_argument('--run_trivial_rand', type=int, default=1, help="Boolean for running TrivialRand")
    parser.add_argument('--return_network', type=int, default=1, help="Boolean for return network(s) in newick format")
    parser.add_argument('--time_limit', type=int, default=-1, help="In minutes, -1 is no time limit")
    parser.add_argument('--verbose', type=int, default=0)

    # arguments for MultiML
    parser.add_argument('--n_tr_nets', type=int, default=2000)
    parser.add_argument('--chosen_leaves', type=int, default=1, help="number of chosen leaves per MultiML iteration")

    # arguments for TrivialRand
    parser.add_argument('--n_triv_rand_its', type=int, default=1000, help="Max number of TrivialRand iterations")
    parser.add_argument('--multiml_time_limit', type=int, default=1, help="Time limit of TrivialRand equal to duration of MultiML")

    # arguments for experiments paper
    parser.add_argument('--data_from_paper', type=int, default=0)
    parser.add_argument('--test_path', type=str, default="LGT", choices=["LGT",
                                                                              "Beiko/small",
                                                                              "Beiko/large"])
    parser.add_argument('--inst', type=int, default=1)
    parser.add_argument('--n_trees', type=int, default=20)
    parser.add_argument('--n_leaves', type=int, default=20)
    parser.add_argument('--n_rets', type=int, default=-1, help="number of reticulations in the reference network")
    parser.add_argument('--mis_l', type=int, default=0, help="missing leaves percentage")
    parser.add_argument('--con_e', type=int, default=0, help="contracted edges percentage")

    args = parser.parse_args()
    main(args)
