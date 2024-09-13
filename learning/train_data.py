import os
import pickle
from datetime import datetime

from network_gen.network_to_tree import *


def net_to_reduced_trees_duo_model(rng, seed, net, mis_l=0, mis_edge=0, num_red=0, num_rets=0, num_trees=None,
                                   distances=True, net_lvs=None, num_chosen_leaves=5):
    # extract trees from network
    if num_rets == 0:
        tree = deepcopy(net)
        add_node_attributes(tree, distances=distances, root=0)
        tree_set = {0: tree}
    else:
        tree_set, _, _ = net_to_tree_non_binary(net, rng, num_trees=num_trees, distances=distances, net_lvs=net_lvs,
                                                mis_l=mis_l, mis_edge=mis_edge)

    # make network and forest environments
    net_env = deepcopy(PhN(net))
    init_forest_env = Input_Set(tree_set=tree_set, leaves=net_env.leaves)
    forest_env = deepcopy(init_forest_env)
    # get cherries from network and forest
    net_cher, net_ret_cher = network_cherries(net_env.nw)
    forest_env.find_all_pairs()

    # output information
    num_cher = [len(net_cher)]
    num_ret_cher = [len(net_ret_cher)]
    tree_set_num = [len(forest_env.trees)]

    # Leaf features
    st_leaf = time.time()
    leaf_features = LeafFeatures(forest_env)
    leaf_feat_time = time.time() - st_leaf
    x_leaf = deepcopy(leaf_features.data)
    x_leaf.index = [f"{l}_s{seed}_r0" for l in list(leaf_features.data.index)]
    y_leaf = leaf_labels(net_cher, net_ret_cher, list(leaf_features.data.index), seed=seed, r=0)

    # Cherry features
    st_cherry = time.time()
    cherry_features = CherryFeatures(forest_env.trees, train_phase=True)
    cherry_feat_time = time.time() - st_cherry

    x_cherry = cherry_features.data
    y_cherry = pd.DataFrame(columns=np.arange(4), dtype=np.int8)

    # Relabel time
    relabel_time = 0
    # INITIALIZE CHERRY PICKING SEQUENCE
    CPS = []
    num_leaf_data = 0
    num_cherry_data = 0
    # now, reduce tree_set and net at the same time to get more labelled data!
    for r in np.arange(num_red+1):
        # ################################# CHERRY/LEAF SELECTION PHASE ##################################
        # PICK TRIVIAL CHERRY
        chosen_cherry, _ = forest_env.pick_trivial(seed)

        if chosen_cherry is None:
            triv_picked = False
        else:
            triv_picked = True

        if not triv_picked:
            # GET PICKABLE CHERRIES
            pickable_chers = (net_cher.union(net_ret_cher)).intersection(set(forest_env.reducible_pairs))
            # PICK RANDOM CHERRY
            if pickable_chers:  # reticulate cherries
                chosen_cherry = list(pickable_chers)[rng.choice(len(pickable_chers))]
            else:
                print(f"JOB {seed} (L = {net_lvs}, T = {num_trees}, conE = {mis_edge}) ({datetime.now().time()}): "
                      f"STOPPED. NO PICKABLE CHERRIES. NUM REDUCTIONS = {r}")
                break

            # GET RANDOM CHOSEN LEAVES
            chosen_leaves = set(rng.choice(list(chosen_cherry), min(2, num_chosen_leaves)))
            if num_chosen_leaves - 2 > 0:
                left_leaves = set(leaf_features.data.index).difference(set(chosen_leaves))
                chosen_leaves.update(set(rng.choice(list(left_leaves), min(len(left_leaves), num_chosen_leaves-2))))

        CPS.append(chosen_cherry)
        # ################################# RELABELLING PHASE ##################################
        # check whether we need to relabel
        x, y = chosen_cherry
        first_check = [(x in tree.leaves and y not in tree.leaves)
                       for t, tree in forest_env.trees.items()
                       if t not in forest_env.reducible_pairs[chosen_cherry]]
        extra_check = [(x in tree.leaves and y not in tree.leaves) or
                       (not {x, y}.intersection(tree.leaves)) or
                       (y in tree.leaves and x not in tree.leaves)
                       for t, tree in forest_env.trees.items()
                       if t not in forest_env.reducible_pairs[chosen_cherry]]
        if extra_check:
            relabel_needed = any(first_check) and all(extra_check)
        else:
            relabel_needed = False
        if relabel_needed:
            st_relabel = time.time()
            merged_cherries, relabel_in_tree = forest_env.relabel_trivial(*chosen_cherry)
            st_leaf = time.time()
            leaf_features.relabel_trivial_cherries(*chosen_cherry, len(forest_env.trees), relabel_in_tree)
            leaf_feat_time += time.time() - st_leaf
            relabel_time += time.time() - st_relabel
        # ################################# MAKE DATA PHASE ##################################
        if not triv_picked:
            # MAKE CHERRY DATA
            st_cherry = time.time()
            cherries = cherry_features.make_data(forest_env.reducible_pairs, chosen_leaves, forest_env.trees)
            cherry_feat_time += time.time() - st_cherry

            # save cherry features
            x_cherry = deepcopy(cherry_features.data)
            # change index of X
            x_cherry.index = [f"{c}_s{seed}_r{r}" for c in x_cherry.index]
            y_cherry = cherry_labels(net_cher, net_ret_cher, list(cherries), x_cherry.index, seed, r)
        # ################################# SAVING ML DATA PHASE ##################################
        if not triv_picked:
            output = {"x_cherry": x_cherry, "y_cherry": y_cherry, "x_leaf": x_leaf, "y_leaf": y_leaf}
            os.makedirs("data/train/inst_results", exist_ok=True)
            with open(f"data/train/inst_results/ML_data_{seed}_r{r}.pkl", "wb") as handle:
                pickle.dump(output, handle)
            num_leaf_data += len(y_leaf)
            num_cherry_data += len(y_cherry)
        if num_red == 0:
            break

        # UPDATE LEAF DATA BEFORE PICKING
        st_leaf = time.time()
        deleted_leaves = leaf_features.update_leaf_features_before(chosen_cherry, forest_env.reducible_pairs, forest_env.trees)
        leaf_feat_time += time.time() - st_leaf

        # UPDATE TREE HEIGHT FEATURE
        st_cherry = time.time()
        cherry_features.tree_height.update_height(chosen_cherry, forest_env.reducible_pairs[chosen_cherry],
                                                  forest_env.trees)
        cherry_feat_time += time.time() - st_cherry
        # ################################# CHERRY PICKING PHASE ##################################
        # PICK CHERRY
        forest_env.update_node_comb_length(*chosen_cherry)
        new_reduced, deleted_trees = forest_env.reduce_pair_in_all(chosen_cherry)
        # CHECK
        if any([any([trees.nw.in_degree(n) == 2 for n in trees.nw.nodes]) for t, trees in
                forest_env.trees.items()]):
            print(f"JOB {seed} (L = {net_lvs}, T = {num_trees}, misL = {mis_l}) ({datetime.now().time()}): "
                  f"STOPPED. RET HAPPENED. NUM REDUCTIONS = {r}")
            break
        # TERMINATE IF NO MORE TREES
        forest_env.update_reducible_pairs(new_reduced)
        if len(forest_env.trees) == 0:
            break

        # REDUCE NETWORK
        net_env.reduce_pair(*chosen_cherry)
        net_cher, net_ret_cher = network_cherries(net_env.nw)

        # ################################# FINALIZING + STORING DATA PHASE ##################################
        # UPDATE LEAF FEATURES AFTER
        st_leaf = time.time()
        leaf_features.update_leaf_features_after(forest_env.trees, deleted_leaves, deleted_trees)
        leaf_feat_time += time.time() - st_leaf

        # output information
        num_cher += [len(net_cher)/2]
        num_ret_cher += [len(net_ret_cher)]
        tree_set_num += [len(forest_env.trees)]

        # SAVE DATA
        # leaf features
        x_leaf = deepcopy(leaf_features.data)
        # change index of X
        x_leaf.index = [f"{l}_s{seed}_r{r+1}" for l in list(leaf_features.data.index)]
        y_leaf = leaf_labels(net_cher, net_ret_cher, list(leaf_features.data.index), seed=seed, r=r+1)

    return num_leaf_data, num_cherry_data, num_cher, num_ret_cher, tree_set_num, cherry_feat_time, leaf_feat_time, \
        relabel_time, r


# CHERRY LABELS
def cherry_labels(net_cher, net_ret_cher, tree_cher, index, num_net=0, r=0):
    # LABELS
    df_labels = pd.DataFrame(0, index=index, columns=np.arange(4), dtype=np.int8)
    for c in tree_cher:
        # cherry in network
        if c in net_cher:
            df_labels.loc[f"{c}_s{num_net}_r{r}", 1] = 1
        elif c in net_ret_cher:
            df_labels.loc[f"{c}_s{num_net}_r{r}", 2] = 1
        elif c[::-1] in net_ret_cher:
            df_labels.loc[f"{c}_s{num_net}_r{r}", 3] = 1
        else:
            df_labels.loc[f"{c}_s{num_net}_r{r}", 0] = 1
    return df_labels


# LEAF LABELS
def leaf_labels(net_cher, net_ret_cher, leaves, seed=0, r=0):
    # LABELS
    index = [f"{l}_s{seed}_r{r}" for l in leaves]
    df_labels = pd.Series(0, index=index, dtype=np.int8)
    leaf_in_net_cherry = set()
    # cherry in network
    done_cherry = set()
    for cherry_type in [net_cher, net_ret_cher]:
        for x, y in cherry_type:
            if (x, y) in done_cherry:
                continue
            done_cherry.add((y, x))
            if x in leaves:
                df_labels[f"{x}_s{seed}_r{r}"] = 1
                leaf_in_net_cherry.add(x)
            if y in leaves:
                df_labels[f"{y}_s{seed}_r{r}"] = 1
                leaf_in_net_cherry.add(y)
    return df_labels
