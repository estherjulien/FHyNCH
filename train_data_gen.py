from learning.train_data import *
from network_gen.LGT_network import simulation as simulation

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import time
import os


def make_train_data(seed, num_red=100, num_chosen_leaves=10, missing_leaves=False):
    rng = np.random.RandomState(seed=seed)

    distances = True
    st = time.time()

    # name of data set
    network_path = f"data/train_network/n{seed}.txt"
    # random L and T
    L, T = rng.randint(20, 100, 2)
    if missing_leaves:
        mis_l = rng.choice([0, 10, 20, 30])
    else:
        mis_l = 0
    mis_edge = rng.choice([0, 10, 20])

    # MAKE/GET NETWORK
    if os.path.exists(network_path):
        net, num_ret, num_leaves, net_nodes = read_network(network_path)
        dur_net = -1
    else:
        start_net = time.time()
        net, num_ret, num_leaves, net_nodes = make_network(seed, any_L=1, L=L, T=T, file=network_path)
        dur_net = time.time() - start_net

    # MAKE DATA
    if T == -1:
        num_trees = None
    else:
        num_trees = T

    print(f"JOB {seed} (L = {num_leaves}, T = {T}, conE = {mis_edge}) ({datetime.now().time()}): "
          f"Start creating DATA SET (R = {num_ret})")

    st_data = time.time()
    num_leaf_data, num_cherry_data, num_cher, num_ret_cher, tree_set_num, cherry_feat_time, leaf_feat_time, relabel_time, real_red = \
        net_to_reduced_trees_duo_model(rng=rng, seed=seed, net=net, mis_l=mis_l, mis_edge=mis_edge,
                                       num_red=num_red, num_rets=num_ret,
                                       num_trees=num_trees, distances=distances,
                                       net_lvs=num_leaves, num_chosen_leaves=num_chosen_leaves)

    print(f"JOB {seed} (L = {num_leaves}, T = {T}, misE = {mis_edge}) ({datetime.now().time()}): "
          f"DATA GENERATION NETWORK FINISHED in {np.round(time.time() - st_data, 3)}s "
          f"(R = {num_ret}, X_cherry = {num_cherry_data}, X_leaf = {num_leaf_data})")

    # SAVE METADATA
    metadata_index = ["num_leaf_data", "num_cherry_data", "rets", "reductions", "nodes", "net_leaves", "chers", "ret_chers", "trees",
                      "runtime", "network_gen_runtime", "cherry_feat_time", "leaf_feat_time", "relabel_time"]
    metadata = pd.Series([num_leaf_data, num_cherry_data, num_ret, real_red, net_nodes, num_leaves, np.mean(num_cher),
                          np.mean(num_ret_cher), num_trees, time.time() - st, dur_net, cherry_feat_time,
                          leaf_feat_time, relabel_time],
                         index=metadata_index,
                         dtype=float)
    # SAVE SOLUTIONS
    save_dir = "data/train/metadata"
    os.makedirs(save_dir, exist_ok=True)
    metadata.to_csv(f"{save_dir}/train_metadata_{seed}.txt")


def make_network(seed, any_L=1, L=100, T=-1, file=""):
    # params of LGT generator
    beta = 1

    now = datetime.now().time()

    # choose n
    if T != -1:
        min_ret = np.ceil(np.log(T) / np.log(2))
        max_ret = 10
    else:
        min_ret = 0
        max_ret = 9

    min_n = L - 2 + min_ret
    max_n = L - 2 + max_ret

    # make network
    network_gen_st = time.time()
    print(f"JOB {seed} ({now}): Start creating NETWORK (L = {L})")
    while True:
        n = np.random.randint(min_n, max_n)
        alpha = np.random.uniform(0.1, 0.5)

        net, num_ret = simulation(n, alpha, 1, beta)
        num_leaves = len(leaves(net))
        if not any_L and min_ret <= num_ret <= max_ret and num_leaves == L:
            break
        if any_L and min_ret <= num_ret <= max_ret and 20 <= num_leaves <= 100:
            break
    print(f"JOB {seed}: finished creating training network in {np.round(time.time() - network_gen_st, 2)}s "
          f"(L = {num_leaves}, R = {num_ret})")

    # SAVE NETWORK
    os.makedirs("data/train_network", exist_ok=True)
    f = open(file, "w")
    for x, y in net.edges:
        length = net.edges[(x, y)]["length"]
        f.write(f"{x} {y} {length}\n")
    f.close()

    net_nodes = int(len(net.nodes))

    return net, num_ret, num_leaves, net_nodes


def read_network(file):
    # open edges file
    with open(file, "r") as handle:
        edges_file = handle.readlines()
    matrix = np.loadtxt(edges_file, dtype=str)
    edges = matrix[:, :2].astype(int)
    weights = matrix[:, -1].astype(float)

    # add edges
    net = nx.DiGraph()
    for i in range(len(weights)):
        net.add_edge(*edges[i], length=weights[i])

    num_ret = len(reticulations(net))
    num_leaves = len(leaves(net))
    net_nodes = int(len(net.nodes))

    return net, num_ret, num_leaves, net_nodes


def main(seed, num_red=100, num_chosen_leaves=10, missing_leaves=False):
    if os.path.exists(f"data/train/metadata/train_metadata_{seed}.txt"):
        print(f"JOB {seed} ALREADY COMPLETED")
        return None
    # try:
    make_train_data(seed, num_red, num_chosen_leaves, missing_leaves)
    # except Exception as e:
    #     print(f"JOB {seed}: FAILED - {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0, help='JOB ID.')
    parser.add_argument('--n_inst', type=int, default=100, help='Number of instances per job.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of cores.')
    parser.add_argument('--mis_l', type=int, default=0)
    args = parser.parse_args()

    first_inst = 1 + args.n_inst*(args.job_id - 1)
    last_inst = args.n_inst*args.job_id
    mis_l = bool(args.mis_l)
    Parallel(n_jobs=args.n_jobs)(delayed(main)(s, missing_leaves=mis_l) for s in range(first_inst, last_inst+1))
