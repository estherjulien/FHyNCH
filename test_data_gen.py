from network_gen.network_to_tree import *
from network_gen.LGT_network import simulation as lgt_simulation
from network_gen.tree_to_newick import *

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import argparse
import pickle
import time
import os


def make_test_lgt(net_num, l, ret, num_trees, missing_leaves=0, missing_edges=0, print_failed=False):
    network_gen = "LGT"
    tree_info = f"_L{l}_R{ret}_T{num_trees}_MisL{missing_leaves}_ConE{missing_edges}_LGT"

    # MAKE NETWORK
    st = time.time()
    beta = 1
    distances = True
    n = l - 2 + ret

    while True:
        alpha = np.random.uniform(0.1, 0.5)

        net, ret_num = lgt_simulation(n, alpha, 1, beta)
        num_leaves = len(leaves(net))
        if num_leaves == l and ret_num == ret:
            break
        elif print_failed:
            print(f"{network_gen} NETWORK GEN FAILED. r = {ret_num}, l = {num_leaves} ")

    # EXTRACT TREES
    net_nodes = int(len(net.nodes))
    trial = 1
    while True:
        rng = np.random.RandomState(10000 + net_num + trial)
        tree_set, tree_lvs, num_union_leaves = net_to_tree_non_binary(net, rng, num_trees, distances=distances,
                                                                       net_lvs=num_leaves, mis_l=missing_leaves,
                                                                       mis_edge=missing_edges)
        if num_union_leaves == num_leaves:
            break
        trial += 1

    tree_to_newick_fun(tree_set, net_num, tree_info=tree_info, network_gen=network_gen)

    # SAVE INSTANCE
    metadata_index = ["network_type", "rets", "nodes", "net_leaves", "chers", "ret_chers", "trees", "n", "alpha",
                      "beta", "missing_lvs_perc", "min_lvs", "mean_lvs", "max_lvs", "runtime"]

    net_cher, net_ret_cher = network_cherries(net)
    min_lvs = min(tree_lvs)
    mean_lvs = np.mean(tree_lvs)
    max_lvs = max(tree_lvs)
    metadata = pd.Series([0, ret_num, net_nodes, num_leaves, len(net_cher)/2, len(net_ret_cher),
                          num_trees, n, alpha, beta,
                          missing_leaves/100, min_lvs, mean_lvs, max_lvs,
                          time.time() - st],
                         index=metadata_index,
                         dtype=float)
    output = {"net": net, "forest": tree_set, "metadata": metadata}
    save_map = "LGT"
    os.makedirs(f"data/test/{save_map}/instances", exist_ok=True)

    with open(
            f"data/test/{save_map}/instances/tree_data{tree_info}_{net_num}.pkl", "wb") as handle:
        pickle.dump(output, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_inst', type=int, default=1000, help='Number of instances.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of processes')

    parser.add_argument('--mis_l_list', nargs='+', type=int, default=[0, 10, 20], help='List containing missing leaf percentages ')
    parser.add_argument('--con_e_list', nargs='+', type=int, default=[0, 10, 20], help='List containing contracted edges percentages ')
    parser.add_argument('--leaf_list', nargs='+', type=int, default=[20, 50, 100], help='List containing number of leaves ')
    parser.add_argument('--tree_list', nargs='+', type=int, default=[20, 50, 100], help='List containing tree set sizes ')
    parser.add_argument('--ret_list', nargs='+', type=int, default=[10, 20, 30], help='List containing the number of reticulations of the LGT network ')

    args = parser.parse_args()

    Parallel(n_jobs=args.n_jobs)(delayed(make_test_lgt)(i, l, ret, num_trees, mis_l, con_e)
                                 for i in range(args.n_inst)
                                 for mis_l in args.mis_l_list
                                 for con_e in args.con_e_list
                                 for l in args.leaf_list
                                 for ret in args.ret_list
                                 for num_trees in args.tree_list)
