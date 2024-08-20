import argparse
import os
import glob
from Bio import Phylo
from CPH import *
import threading
import pickle as pkl


def get_tree_from_newick(newick):
    file = open(f"tmp_{threading.get_native_id()}.tre", "w")
    file.write(newick)
    file.close()
    tree_bio = Phylo.read(f"tmp_{threading.get_native_id()}.tre", "newick", rooted=True)
    os.remove(f"tmp_{threading.get_native_id()}.tre")
    G = Phylo.to_networkx(tree_bio)
    Gi = nx.convert_node_labels_to_integers(G, label_attribute="Clade")
    assert check_root(Gi)
    final_tree = nx.relabel_nodes(Gi, {v: Gi.nodes[v]['Clade'].name for v in Gi.nodes if
                                       Gi.nodes[v]["Clade"].name is not None})
    change_weight_to_length(final_tree)
    add_node_attributes(final_tree, distances=True, root=0)
    return final_tree


def check_root(tree):
    root_is_zero = True
    for n in tree.nodes:
        if n == 0 and tree.in_degree(n) != 0:
            root_is_zero = False
            break
        if tree.in_degree(n) == 0 and n != 0:
            root_is_zero = False
            break
    return root_is_zero


def change_weight_to_length(tree):
    nx.set_edge_attributes(tree, {e: tree.edges[e]["weight"] for e in tree.edges}, "length")


def add_init_missing_leaves(trees, leaves):
    forest_env = Input_Set(tree_set=trees, leaves=leaves)
    forest_env.find_all_pairs()
    forest_env.init_missing_leaves()
    # first add leaves to same cherries
    forest_env.add_leaves()

    return forest_env


def add_leftover_leaves(forest_env, num_baseline=1):
    rng = np.random.RandomState(seed=num_baseline)
    forest_env.add_leaf_random(rng)
    for t, tree in forest_env.trees.items():
        if forest_env.leaves.difference(tree.leaves) or tree.leaves.difference(forest_env.leaves):
            print("not the same")
            break
    return forest_env


def read_tree_set(args=None, test_case_path=None, newick_file=None):
    if newick_file is None and args.test_case == "Beiko":
        file_name = test_case_path + f"sampled/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}.txt"
    elif newick_file is None and (args.test_case == "covid" or args.test_case == "languages"):
        file_name = test_case_path + f"newick/test_T0_L0_MisL0_{args.inst}.txt"
    else:
        file_name = newick_file
    file = open(file_name, "r")
    newick_strings = file.readlines()
    file.close()

    # read the input trees in 'newick_strings'
    trees = dict()
    leaves = set()
    for n in newick_strings:
        input_tree = get_tree_from_newick(n)
        tree = PhT(tree=input_tree)
        trees[len(trees)] = input_tree
        leaves = leaves.union(tree.leaves)

    return trees, leaves


def check_nonbinary_misl(trees, leaves):
    print(f"Tree set {args.inst}")
    for t, input_tree in trees.items():
        tree = PhT(tree=input_tree)
        misl = len(leaves.difference(tree.leaves))
        non_binary_num = sum([tree.nw.out_degree(n) - 2 for n in tree.nw.nodes if tree.nw.out_degree(n) > 2])
        non_binary_nodes = [n for n in tree.nw.nodes if tree.nw.out_degree(n) > 2]
        num_non_binary_nodes = len(non_binary_nodes)
        print(f"Tree {t}: MISL% = {misl/len(leaves)}, Non-binary-num = {non_binary_num}, # Non-binary-nodes = {num_non_binary_nodes}, Non-binary_nodes = {non_binary_nodes}")


def write_tree_to_newick(g, root=None):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(write_tree_to_newick(g, root=child))
        else:
            subgs.append(child)
    return "(" + ','.join(subgs) + ")"


def write_tree_to_newick_dist(g, root=None):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    dists = []
    for child in g[root]:
        dists.append(g.edges[root, child]['length'])
        if len(g[child]) > 0:
            subgs.append(write_tree_to_newick_dist(g, root=child))
        else:
            subgs.append(child)
    return "(" + ','.join([f"{c}:{d}" for c, d in zip(subgs, dists)]) + ")"
    # return "(" + ','.join(subgs) + ")",


def trees_to_newick(args, trees, test_case_path, save_map="nonbinary", num_baseline=None):
    newick_str = []
    for t, tree in trees.items():
        if args.test_size == "small":
            newick = write_tree_to_newick(tree.nw, root=None)
        else:
            newick = write_tree_to_newick_dist(tree, root=None)
        newick_str.append(newick + ";\n")
    os.makedirs(test_case_path + save_map, exist_ok=True)
    if num_baseline is None:
        file = open(test_case_path + save_map + f"/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}.txt", "w")
    else:
        file = open(test_case_path + save_map + f"/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}_BL{num_baseline}.txt", "w")

    file.writelines(newick_str)
    file.close()


def main(args):
    test_case_path = f"./data/{args.test_case}/{args.test_size}/"
    # add leaves
    trees, leaves = read_tree_set(args, test_case_path)
    forest_env = add_init_missing_leaves(trees, leaves)
    for i in range(1, args.num_baselines + 1):
        # then add randomly with different seed
        forest_env_new = add_leftover_leaves(deepcopy(forest_env), i)
        trees_to_newick(args, forest_env_new.trees, test_case_path, num_baseline=i)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_instances', type=int, default=10)
    parser.add_argument('--test_case', type=str, default="Beiko", choices=["Beiko", "covid", "languages"])
    parser.add_argument('--test_size', type=str, default="small", choices=["small", "large", "all"])
    parser.add_argument('--inst', type=int, default=1)
    parser.add_argument('--num_trees', type=int, default=2)    # depends on test case
    parser.add_argument('--num_leaves', type=int, default=10)   # depends on test case
    parser.add_argument('--mis_l_perc', type=int, default=30)
    parser.add_argument('--num_baselines', type=int, default=50)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    main(args)
