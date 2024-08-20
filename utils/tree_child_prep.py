import argparse
import os
import glob
from Bio import Phylo
from heuristic.CPH import *
import threading
import pickle as pkl


def get_Beiko_info():
    file_all = open("data/Beiko/Beiko_trees/AllTrees_names.tre", "rb")
    newick_str = file_all.readlines()
    file_all.close()

    leaf_sets = []
    leaf_set_size = []
    for t in newick_str:
        n = t.decode('utf-8')
        final_tree = get_tree_from_newick(n)
        tree = PhT(tree=final_tree)
        # leaf_sets.append(tree.leaves)
        leaf_set_size.append(len(tree.leaves))
    return leaf_sets, leaf_set_size


def make_Beiko_instances(args, test_case_path, time_limit=600):
    rng = np.random.RandomState(args.seed)
    # read Beiko file
    file_all = open("data/Beiko/Beiko_trees/AllTrees_names.tre", "rb")
    newick_str = file_all.readlines()
    file_all.close()

    inst_trees = []
    num_beiko_trees = len(newick_str)
    # make instances
    os.makedirs(test_case_path + "sampled", exist_ok=True)
    for inst_num in range(1, args.num_instances + 1):
        st = time.time()
        while time.time() - st < time_limit:
            tree_ids = []
            curr_leaves = {}
            tree_leaf_set = []
            it = 0
            while len(tree_ids) < args.num_trees and it < 1000*args.num_trees:
                it += 1
                t = rng.choice(num_beiko_trees)
                if t in tree_ids:
                    continue
                # get tree
                n = newick_str[t].decode('utf-8')
                final_tree = get_tree_from_newick(n)
                tree = PhT(tree=final_tree)
                # ADDING TREE TO SET CHECKS
                if not curr_leaves and not args.num_leaves * (1-args.mis_l_perc/100) <= len(tree.leaves) <= args.num_leaves:
                    continue
                elif not args.num_leaves * (1-args.mis_l_perc/100) <= len(tree.leaves) <= args.num_leaves:
                    continue

                if not curr_leaves:
                    curr_leaves = tree.leaves
                    tree_leaf_set.append(tree.leaves)
                    tree_ids.append(t)
                    continue
                curr_leaves_new = curr_leaves.union(tree.leaves)
                if len(curr_leaves_new) > args.num_leaves:
                    continue
                if len(curr_leaves_new.difference(curr_leaves)) > args.num_leaves * (args.mis_l_perc/100):
                    continue
                if len(curr_leaves.difference(curr_leaves_new)) > args.num_leaves * (args.mis_l_perc/100):
                    continue

                fine = True
                for _leaves in tree_leaf_set:
                    if len(curr_leaves_new.difference(_leaves)) > args.num_leaves * (args.mis_l_perc/100):
                        fine = False
                    elif len(_leaves.difference(curr_leaves_new)) > args.num_leaves * (args.mis_l_perc/100):
                        fine = False
                    if not fine:
                        break
                if not fine:
                    continue

                curr_leaves = curr_leaves_new
                tree_leaf_set.append(tree.leaves)
                tree_ids.append(t)

            # ADD INSTANCE
            if len(tree_ids) == args.num_trees:
                print(f"S = {args.test_size} - L = {args.num_leaves} - T = {args.num_trees} - MisL = {args.mis_l_perc}: "
                      f"Instance {inst_num} generated")
                # save instances
                file = open(test_case_path + f"sampled/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{inst_num}.txt", "w")
                for t in tree_ids:
                    n = newick_str[t].decode('utf-8')
                    file.write(n)
                file.close()
                tree_ids_str = ""
                for t in tree_ids:
                    tree_ids_str += str(t) + " "
                tree_ids_str += "\n"

                inst_trees.append(tree_ids_str)
                break
    # save chosen trees
    file = open(test_case_path + f"sampled/test_trees_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_I{args.num_instances}.txt", "w")
    file.writelines(inst_trees)
    file.close()


def make_covid_instances(args, test_case_path):
    if args.test_size == "small":
        tree_dir = "binary-treechild-semitemp"
    else:
        tree_dir = "nonbinary-original"

    os.makedirs(test_case_path + "/newick", exist_ok=True)
    # make instances
    inst_to_test = dict()
    gene_test = glob.glob(f"data/covid/trees/genes/{tree_dir}/*")
    inst = 1
    for test in gene_test:
        inst_to_test[inst] = test.split("/genes_", 1)[1][:-5]
        inst_dir = test_case_path + f"/newick/test_T0_L0_MisL0_{inst}.txt"
        os.system(f'cp {test} {inst_dir}')
        inst += 1
    block_test = glob.glob(f"data/covid/trees/blocks/{tree_dir}/*")
    for test in block_test:
        inst_to_test[inst] = test.split("/blocks_", 1)[1][:-5]
        inst_dir = test_case_path + f"/newick/test_T0_L0_MisL0_{inst}.txt"
        os.system(f'cp {test} {inst_dir}')
        inst += 1
    with open(test_case_path + "/newick/inst_to_test.pkl", "wb") as handle:
        pkl.dump(inst_to_test, handle)


def make_languages_instances(args, test_case_path):
    os.makedirs(test_case_path + "/newick", exist_ok=True)

    # inst 1 = Semitic
    trees = []
    for i in range(1, 11):
        file = open(f"data/languages/languages_input/Semitic/out_{i}.pl.txt.newick", "rb")
        trees += file.readlines()
        file.close()
    file_inst = open(test_case_path + "/newick/test_T0_L0_MisL0_1.txt", "wb")
    file_inst.writelines(trees)
    file_inst.close()

    # inst 2 = Constructed (artificial)
    trees = []
    for i in range(1, 11):
        file = open(f"data/languages/languages_input/Constructed/out_{i}.pl.txt.newick", "rb")
        trees += file.readlines()
        file.close()
    file_inst = open(test_case_path + "/newick/test_T0_L0_MisL0_2.txt", "wb")
    file_inst.writelines(trees)
    file_inst.close()
    pass


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


def add_missing_leaves(trees, leaves):
    forest_env = Input_Set(tree_set=trees, leaves=leaves)
    forest_env.find_all_pairs()
    forest_env.init_missing_leaves()
    # CHECK IF ALL TREES NOW HAVE ALL LEAVES
    all_equal = False
    while not all_equal:
        all_equal = True
        forest_env.add_leaves()
        for t, tree in forest_env.trees.items():
            if not leaves.difference(tree.leaves):
                continue
            all_equal = False
            break
    return forest_env


def read_tree_set(args=None, test_case_path=None, newick_file=None, save_map="samples"):
    if newick_file is None and args.test_case == "Beiko":
        file_name = test_case_path + f"{save_map}/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}.txt"
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


def trees_to_newick(args, trees, test_case_path, save_map="nonbinary"):
    newick_str = []
    for t, tree in trees.items():
        newick = write_tree_to_newick_dist(tree, root=None)
        newick_str.append(newick + ";\n")
    os.makedirs(test_case_path + save_map, exist_ok=True)
    file = open(test_case_path + save_map + f"/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}.txt", "w")
    file.writelines(newick_str)
    file.close()


def main(args):
    test_case_path = f"./data/{args.test_case}/{args.test_size}/"
    # if args.test_case == "Beiko":
    #     make_Beiko_instances(args, test_case_path)
    # elif args.test_case == "covid":
    #     make_covid_instances(args, test_case_path)
    # if args.test_case == "languages":
    #     make_languages_instances(args, test_case_path)
    # if args.test_case == "covid" and args.test_size == "large" or args.test_case == "languages":
    #     trees, leaves = read_tree_set(args, test_case_path)
    #     check_nonbinary_misl(trees, leaves)
    #     return None
    if args.test_case == "Beiko":
        trees, leaves = read_tree_set(args, test_case_path)
        trees_to_newick(args, trees, test_case_path, save_map="newick")
        return None
    #
    # # add leaves
    # trees, leaves = read_tree_set(args, test_case_path)
    # forest_env = add_missing_leaves(trees, leaves)
    # trees_to_newick(args, forest_env.trees, test_case_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_instances', type=int, default=10)
    parser.add_argument('--test_case', type=str, default="covid", choices=["Beiko", "covid", "languages"])
    parser.add_argument('--test_size', type=str, default="small", choices=["small", "large", "all"])
    parser.add_argument('--inst', type=int, default=1)
    parser.add_argument('--num_trees', type=int, default=2)    # depends on test case
    parser.add_argument('--num_leaves', type=int, default=10)   # depends on test case
    parser.add_argument('--mis_l_perc', type=int, default=20)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    main(args)

    # for making instances!
    # if os.path.exists(f"data/Beiko/{args.test_size}/sampled/test_trees_T{args.num_trees}_"
    #                   f"L{args.num_leaves}_MisL{args.mis_l_perc}_I{args.num_instances}.txt"):
    #     print(f"S = {args.test_size} - L = {args.num_leaves} - T = {args.num_trees} - MisL = {args.mis_l_perc}: DONE")
    # elif args.test_size == "large" and args.num_trees > 10 \
    #         and os.path.getsize(f"data/Beiko/{args.test_size}/sampled/test_trees_T{args.num_trees-10}_"
    #                   f"L{args.num_leaves}_MisL{args.mis_l_perc}_I{args.num_instances}.txt") == 0:
    #     print(f"S = {args.test_size} - L = {args.num_leaves} - T = {args.num_trees} - MisL = {args.mis_l_perc}: PREVIOUS TREE SET SIZE EMPTY")
    #     file = open(f"data/Beiko/{args.test_size}/sampled/test_trees_T{args.num_trees}_"
    #                   f"L{args.num_leaves}_MisL{args.mis_l_perc}_I{args.num_instances}.txt", "w")
    #     file.close()
    # elif args.mis_l_perc < 30 and os.path.getsize(f"data/Beiko/{args.test_size}/sampled/test_trees_T{args.num_trees}_"
    #                   f"L{args.num_leaves}_MisL{args.mis_l_perc+10}_I{args.num_instances}.txt") == 0:
    #     print(f"S = {args.test_size} - L = {args.num_leaves} - T = {args.num_trees} - MisL = {args.mis_l_perc}: PREVIOUS MISSING LEAVES PERC EMPTY")
    #     file = open(f"data/Beiko/{args.test_size}/sampled/test_trees_T{args.num_trees}_"
    #                   f"L{args.num_leaves}_MisL{args.mis_l_perc}_I{args.num_instances}.txt", "w")
    #     file.close()
    # else:
    #     print(f"S = {args.test_size} - L = {args.num_leaves} - T = {args.num_trees} - MisL = {args.mis_l_perc}: STARTING")
    #     main(args)
