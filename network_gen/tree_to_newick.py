import numpy as np
import os


def sub_tree_to_newick(G, root=None):        # only for binary?
    subgs = []
    for child in G[root]:
        try:
            length = np.round(G.edges[(root, child)]["length"], 3)
        except KeyError:
            length = np.round(G.edges[(root, child)]["lenght"], 3)
        if len(G[child]) > 0:
            subgs.append(sub_tree_to_newick(G, root=child) + f":{length}")
        else:
            subgs.append(str(child) + f":{length}")
    return "(" + ','.join(subgs) + ")"


def tree_to_newick_fun(tree_set, net_num=None, network_gen="LGT", tree_info="", file_name=None):
    if file_name is None:
        os.makedirs(f"data/test/{network_gen}/newick", exist_ok=True)
        file_name = f"data/test/{network_gen}/newick/tree_set_newick{tree_info}_{net_num}.txt"

    file = open(file_name, "w+")
    for tree in tree_set.values():
        tree_line = sub_tree_to_newick(tree, 0)
        file.write(tree_line)
        file.write("\n")
    file.close()
