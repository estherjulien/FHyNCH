import argparse
import os
from tree_child_prep import read_tree_set
from tree_child_leaves import add_init_missing_leaves


def main(args):
    test_case_path = f"./data/{args.test_case}/{args.test_size}/"
    # add leaves
    trees, leaves = read_tree_set(args, test_case_path, save_map="newick")
    forest_env = add_init_missing_leaves(trees, leaves)
    num_leaves = len(forest_env.leaves)
    os.makedirs(test_case_path + "tc_input", exist_ok=True)
    for i in range(1, args.num_baselines + 1):
        input_file = test_case_path + f"nonbinary/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}_BL{i}.txt"
        output_file = test_case_path + f"tc_input/test_T{args.num_trees}_L{args.num_leaves}_MisL{args.mis_l_perc}_{args.inst}_BL{i}.txt"
        if os.path.exists(output_file):
            continue
        os.system(f"MakeTestData-exe {num_leaves} {input_file} {output_file}")


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
