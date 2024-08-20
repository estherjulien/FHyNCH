import networkx as nx
import pandas as pd
import numpy as np
import copy

'''
CODE FOR FEATURES:
Composes of two main classes: CherryFeatures and LeafFeatures
'''


#####################################################################################
#######################             CHERRY FEATURES             #####################
#####################################################################################
class CherryFeatures:
    def __init__(self, tree_set, train_phase):
        self.feature_names = []
        self.num_trees = len(tree_set)

        self.train_phase = train_phase
        if self.train_phase:
            self.root = 0
        else:
            self.root = 2
        # FEATURE CLASSES
        # independent of height
        self.data = pd.DataFrame(dtype=float)

        self.trivial = Trivial()

        self.leaf_pair = LeafPair(self.num_trees)

        self.red_after_pick = RedAfterPick(self.root)

        # tree height
        self.tree_height = TreeHeight(self.root)

        # dependent of height
        self.cherry_height = CherryHeight()

        self.leaf_dist = LeafDist()

        self.leaf_height = LeafHeight()

        # RUN INIT FUN
        self.init_fun(tree_set)

    def init_fun(self, tree_set):
        # all things that have to be done once
        self.data = pd.DataFrame(columns=self.feature_names, dtype=float)
        self.tree_height.get_tree_height_metric(tree_set)

    def make_data(self, reducible_pairs, chosen_leaves, tree_set):
        cherries = {c: trees for c, trees in reducible_pairs.items() if chosen_leaves.intersection(set(c))}
        assert cherries
        self.data = pd.DataFrame(index=list(cherries), columns=self.feature_names, dtype=float)
        unique_cherries = set([tuple(sorted(c)) for c in cherries])

        # LEAF PAIRS
        new_data_column = self.leaf_pair.make_data(cherries, tree_set)
        self.data[self.leaf_pair.feature_names] = new_data_column

        # TRIVIAL
        new_data_column = self.trivial.make_data(cherries, tree_set, unique_cherries, self.num_trees,
                                                 self.leaf_pair.n)
        self.data[self.trivial.feature_names] = new_data_column

        # TREE HEIGHT
        new_data_column = self.tree_height.make_data(cherries, unique_cherries, cherry_in_forest=self.trivial.n)
        self.data[self.tree_height.feature_names] = new_data_column
        # REDUCTION AFTER PICK
        new_data_column = self.red_after_pick.make_data(cherries, tree_set, unique_cherries, self.trivial.n)
        self.data[self.red_after_pick.feature_names] = new_data_column

        # HEIGHT DEPENDENT FEATURES
        # cherry height
        new_data_column = self.cherry_height.make_data(cherries, tree_set, unique_cherries,
                                                       self.tree_height.dist, self.tree_height.comb,
                                                       self.trivial.n)
        self.data[self.cherry_height.feature_names] = new_data_column

        # leaf distance
        new_data_column = self.leaf_dist.make_data(cherries, tree_set, unique_cherries,
                                                   self.tree_height.dist, self.tree_height.comb,
                                                   self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)
        self.data[self.leaf_dist.feature_names] = new_data_column

        # leaf height
        new_data_column = self.leaf_height.make_data(cherries, tree_set, unique_cherries,
                                                     self.tree_height.dist, self.tree_height.comb,
                                                     self.leaf_pair.leaf_pair_cont_tree, self.leaf_pair.n)
        self.data[self.leaf_height.feature_names] = new_data_column

        self.data.index = pd.MultiIndex.from_tuples(self.data.index)
        return cherries


# TREE HEIGHT
class TreeHeight:
    def __init__(self, root=2):
        # PARAMETERS
        self.name = "TreeHeight"
        self.feature_names = ["Tree depth (t)", "Tree depth (d)"]
        self.data_column = pd.DataFrame()
        self.root = root
        # distances
        self.tree_level_width_dist = {}
        self.tree_level_dist_id = {}
        self.height_level_dist = {}
        self.max_id_dist = {}
        self.dist = pd.Series(dtype=float)
        self.prev_dist = pd.Series(dtype=float)
        self.dist_n = pd.Series(dtype=float)
        self.max_dist = 0

        # combinatorial
        self.tree_level_width_comb = {}
        self.tree_level_comb_id = {}

        self.comb = pd.Series(dtype=float)
        self.prev_comb = pd.Series(dtype=float)
        self.comb_n = pd.Series(dtype=float)
        self.max_comb = 0

        self.d = pd.Series(dtype=float)

    def get_tree_height_metric(self, tree_set):
        self.dist = pd.Series(index=tree_set, dtype=float)
        self.comb = pd.Series(index=tree_set, dtype=float)

        for t, tree in tree_set.items():
            self.metric_per_tree(t, tree)
        self.max_dist = self.dist.max()
        self.max_comb = self.comb.max()

    def metric_per_tree(self, t, tree):
        try:
            tree_input = tree.nw
            tree_leaves = tree.leaves
        except AttributeError:
            tree_input = tree
            tree_leaves = [u for u in tree.nodes() if tree.out_degree(u) == 0]

        self.tree_level_width_comb[t] = []
        self.tree_level_width_dist[t] = []
        self.height_level_dist[t] = []
        tmp_tree_level_comb = {}
        tmp_tree_level_dist = {}
        for n in tree_input.nodes:
            # skip leaves!
            if n in tree_leaves or (self.root == 2 and n == 1):
                continue
            # COMBINATORIAL
            comb_height = tree_input.nodes[n]["node_comb"]
            if comb_height in tmp_tree_level_comb:
                tmp_tree_level_comb[comb_height] += 1
            else:
                tmp_tree_level_comb[comb_height] = 1
            # DISTANCES
            height = np.round(tree_input.nodes[n]["node_length"], 10)
            if height in tmp_tree_level_dist:
                tmp_tree_level_dist[height].add(n)
            else:
                tmp_tree_level_dist[height] = {n}
        # find max
        # COMBINATORIAL
        sorted_tmp_comb = dict(sorted(tmp_tree_level_comb.items()))
        for comb_height, num_nodes in sorted_tmp_comb.items():
            self.tree_level_width_comb[t].append(num_nodes)
        self.comb[t] = len(self.tree_level_width_comb[t]) - 1
        # DISTANCES
        sorted_tmp_dist = dict(sorted(tmp_tree_level_dist.items()))
        self.tree_level_dist_id[t] = {}
        for i, (dist_height, nodes) in enumerate(sorted_tmp_dist.items()):
            self.height_level_dist[t].append(dist_height)
            self.tree_level_width_dist[t].append(len(nodes))
            for _n in nodes:
                self.tree_level_dist_id[t][_n] = i
        self.max_id_dist[t] = len(self.height_level_dist[t]) - 1
        self.dist[t] = self.height_level_dist[t][self.max_id_dist[t]]

    def make_data(self, reducible_pairs, unique_cherries, cherry_in_forest):
        self.data_column = pd.DataFrame(index=reducible_pairs, columns=self.feature_names, dtype=float)
        self.dist_n = pd.Series(index=reducible_pairs, dtype=float)
        self.comb_n = pd.Series(index=reducible_pairs, dtype=float)
        self.d = cherry_in_forest

        for c in unique_cherries:
            trees = list(reducible_pairs[c])

            self.dist_n[c] = self.dist[trees].sum()
            self.dist_n[c[::-1]] = self.dist_n[c]

            if self.max_dist:
                tree_dist_val = self.dist_n[c] / self.max_dist / self.d[c]
            else:
                tree_dist_val = 0
            self.data_column.loc[c, "Tree depth (d)"] = tree_dist_val
            self.data_column.loc[c[::-1], "Tree depth (d)"] = tree_dist_val

            self.comb_n[c] = self.comb[trees].sum()
            self.comb_n[c[::-1]] = self.comb_n[c]

            if self.max_comb:
                tree_comb_val = self.comb_n[c] / self.max_comb / self.d[c]
            else:
                tree_comb_val = 0
            self.data_column.loc[c, "Tree depth (t)"] = tree_comb_val
            self.data_column.loc[c[::-1], "Tree depth (t)"] = tree_comb_val

        return self.data_column

    def update_height(self, chosen_cherry, new_reduced, tree_set):
        change_height_dist_bool = False
        change_height_comb_bool = False
        for t in new_reduced:
            if t not in tree_set:
                continue
            # check if there are more siblings in current level, if so nothing changes
            for p in tree_set[t].nw.predecessors(chosen_cherry[0]):
                p_c = p
            # SKIP FOR SOME STUFF!
            if tree_set[t].nw.out_degree(p_c) > 2:
                continue

            # COMBINATORIAL
            height = tree_set[t].nw.nodes[p_c]["node_comb"]
            self.tree_level_width_comb[t][height] -= 1
            if self.tree_level_width_comb[t][height] == 0:
                if abs(self.max_comb - self.comb[t]) < 1e-3:
                    change_height_comb_bool = True
                self.comb[t] = height - 1

            # DISTANCES
            level_reduced = self.tree_level_dist_id[t][p_c]
            self.tree_level_width_dist[t][level_reduced] -= 1
            if self.tree_level_width_dist[t][level_reduced] == 0 and self.max_id_dist[t] == level_reduced:
                # update max height tree
                if abs(self.max_dist - self.dist[t]) < 1e-3:
                    change_height_dist_bool = True
                i = 1
                while level_reduced - i >= 0:
                    if self.tree_level_width_dist[t][level_reduced - i] > 0:
                        self.max_id_dist[t] = level_reduced - i
                        break
                    i += 1
                self.dist[t] = self.height_level_dist[t][self.max_id_dist[t]]

        # update max height level of all trees
        if change_height_dist_bool:
            new_max_dist = self.dist.max()
            if abs(new_max_dist - self.max_dist) > 10e-3:
                self.max_dist = new_max_dist
        if change_height_comb_bool:
            new_max_comb = self.comb.max()
            if abs(new_max_comb - self.max_comb) > 10e-3:
                self.max_comb = new_max_comb

    def update_metric_deleted_leaves(self, tree_set, deleted_tree_leaves):
        for t in deleted_tree_leaves:
            if t not in tree_set:
                continue
            tree = tree_set[t]
            self.metric_per_tree(t, tree)

        self.max_comb = self.comb.max()
        self.max_dist = self.dist.max()


# LEAF PAIR
class LeafPair:
    def __init__(self, num_trees):
        self.name = "LeafPair"
        self.feature_names = ["Leaves in tree"]
        self.data_column = pd.DataFrame()

        self.n = pd.Series(dtype=float)
        self.d = num_trees
        self.leaf_pair_cont_tree = pd.DataFrame(dtype=float)

    def make_data(self, reducible_pairs, tree_set):
        self.data_column = pd.DataFrame(columns=self.feature_names)
        self.leaf_pair_cont_tree = pd.DataFrame(True, index=reducible_pairs, columns=tree_set)

        self.n = pd.Series(0, index=reducible_pairs, dtype=float)
        for t, tree in tree_set.items():
            for c in reducible_pairs:
                if set(c).issubset(tree.leaves):
                    self.n[c] += 1
                    self.leaf_pair_cont_tree.loc[c, t] = True
                else:
                    self.leaf_pair_cont_tree.loc[c, t] = False
        self.data_column["Leaves in tree"] = self.n / self.d
        return self.data_column


# TRIVIAL
class Trivial:
    def __init__(self):
        # PARAMETERS
        self.name = "Trivial"
        self.feature_names = ["Trivial", "Cherry in tree"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        self.trivial = pd.DataFrame()
        self.n = pd.Series(dtype=float)

        self.x_in_cherry = pd.DataFrame()
        self.x_in_cherry_in_tree = pd.DataFrame()
        self.x_n = pd.Series(dtype=float)

        self.y_in_cherry = pd.DataFrame()
        self.y_in_cherry_in_tree = pd.DataFrame()
        self.y_n = pd.Series(dtype=float)

        self.triv_d = pd.Series(dtype=float)
        self.cher_d = 1
        self.xy_cher_d = 1

    def make_data(self, reducible_pairs, tree_set, unique_cherries, num_trees, leaf_pair_forest):
        self.data_column = pd.DataFrame(columns=self.feature_names)
        self.trivial = pd.DataFrame(False, index=reducible_pairs, columns=tree_set, dtype=bool)

        for c in unique_cherries:
            trees = list(reducible_pairs[c])
            self.trivial.loc[c, trees] = True
            self.trivial.loc[c[::-1], trees] = True

        # trivial data
        self.n = self.trivial.sum(axis=1)
        self.triv_d = leaf_pair_forest

        # cherry in tree data
        self.cher_d = num_trees
        # trivial data
        self.data_column["Trivial"] = self.n / self.triv_d
        # cherry in tree data
        self.data_column["Cherry in tree"] = self.n / self.cher_d
        return self.data_column


# CHERRY HEIGHT
class CherryHeight:
    def __init__(self):
        # PARAMETERS
        self.name = "CherryHeight"
        self.feature_names = ["Cherry depth (t)", "Cherry depth (d)"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        # distances
        self.dist_n = pd.Series(dtype=float)
        self.dist = pd.DataFrame(dtype=float)

        # combinatorial
        self.comb_n = pd.Series(dtype=float)
        self.comb = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # tree height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

    def make_data(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                  cherry_in_forest):
        self.data_column = pd.DataFrame(columns=self.feature_names)
        self.dist = pd.DataFrame(index=reducible_pairs, columns=tree_set)
        self.comb = pd.DataFrame(index=reducible_pairs, columns=tree_set)
        self.d = cherry_in_forest

        self.dist_n = pd.Series(0, index=reducible_pairs)
        self.comb_n = pd.Series(0, index=reducible_pairs)

        for c in unique_cherries:
            for t in reducible_pairs[c]:
                tree = tree_set[t]
                dist, comb = self.get_cherry_height(tree, *c)
                self.dist.loc[c, t] = dist
                self.dist.loc[c[::-1], t] = dist
                self.comb.loc[c, t] = comb
                self.comb.loc[c[::-1], t] = comb

        # DATA
        for c in unique_cherries:
            self.dist_n[c] = 0
            self.dist_n[c[::-1]] = 0
            self.comb_n[c] = 0
            self.comb_n[c[::-1]] = 0
            for t in reducible_pairs[c]:
                if not tree_height_dist[t]:
                    dist_n_val = 0
                else:
                    dist_n_val = self.dist.loc[c, t] / tree_height_dist[t]
                self.dist_n[c] += dist_n_val
                self.dist_n[c[::-1]] += dist_n_val

                if not tree_height_comb[t]:
                    comb_n_val = 0
                else:
                    comb_n_val = self.comb.loc[c, t] / tree_height_comb[t]
                self.comb_n[c] += comb_n_val
                self.comb_n[c[::-1]] += comb_n_val

        self.data_column["Cherry depth (d)"] = self.dist_n / self.d
        self.data_column["Cherry depth (t)"] = self.comb_n / self.d
        return self.data_column

    @staticmethod
    def get_cherry_height(tree, x, y):
        for p in tree.nw.predecessors(x):
            p_cherry = p
        height = tree.nw.nodes[p_cherry]["node_length"]
        height_comb = tree.nw.nodes[p_cherry]["node_comb"]

        return height, height_comb


# REDUCTION AFTER PICKING A CHERRY
class RedAfterPick:
    def __init__(self, root=2):
        # PARAMETERS
        self.name = "RedAfterPick"
        self.feature_names = ["New cherries", "Before/after"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        self.root = root

        self.red_n = pd.Series(dtype=float)
        self.red_after_pick = pd.DataFrame()

        self.cher_red_n = pd.Series(dtype=float)
        self.cher_red_d = 1
        self.prev_cher_red_d = 1
        self.cherry_red_set = dict()

    def make_data(self, reducible_pairs, tree_set, unique_cherries, cherry_in_forest):
        self.data_column = pd.DataFrame(columns=self.feature_names)
        self.red_after_pick = pd.DataFrame(index=reducible_pairs, columns=tree_set)
        # new cherries/old cherries, so doesn't have to be scaled anymore
        # network cherries works with just one tree, tree cherries with a set
        num_unique_cherries = len(unique_cherries)
        self.cher_red_n = pd.Series(num_unique_cherries - 1, index=reducible_pairs)
        self.cher_red_d = num_unique_cherries

        for c in unique_cherries:
            self.cherry_red_set[c] = set()
            self.cherry_red_set[c[::-1]] = set()
            for t in reducible_pairs[c]:
                num_after_pick, new_cat_cherry_a, new_cat_cherry_b = self.get_num_red_after_pick(tree_set[t], *c)
                self.red_after_pick.loc[c, t] = num_after_pick
                self.red_after_pick.loc[c[::-1], t] = num_after_pick
                if new_cat_cherry_a:
                    self.cherry_red_set[c].update(new_cat_cherry_a)
                    self.cherry_red_set[c[::-1]].update(new_cat_cherry_b)

            for red_cher in self.cherry_red_set[c]:
                if red_cher not in unique_cherries:
                    self.cher_red_n[c] += 1
            for red_cher in self.cherry_red_set[c[::-1]]:
                if red_cher not in unique_cherries:
                    self.cher_red_n[c[::-1]] += 1

        # DATA
        self.data_column["Before/after"] = self.cher_red_n / self.cher_red_d
        self.red_n = self.red_after_pick.sum(axis=1)
        self.data_column["New cherries"] = self.red_n / cherry_in_forest
        return self.data_column

    def get_num_red_after_pick(self, tree, x, y):
        # new cherries/old cherries, so doesn't have to be scaled anymore
        # network cherries works with just one tree, tree cherries with a set
        for p in tree.nw.predecessors(x):
            p_cherry = p
        if p_cherry == self.root:
            return 0, set(), set()
        if tree.nw.out_degree(p_cherry) > 2:
            return 0, set(), set()
            # check if there are any other siblings of x, y
            # get all new ones
            # for ch in tree.nw.successors(p_cherry):
            #     if ch in [x, y]:
            #         continue
            #     new_cherries_x.add(tuple(sorted([ch, y])))
            #     new_cherries_y.add(tuple(sorted([ch, x])))
            # return len(new_cherries_x), new_cherries_x, new_cherries_y

        new_cherries_x = set()
        new_cherries_y = set()
        for p in tree.nw.predecessors(p_cherry):
            gp_cherry = p
        if tree.nw.out_degree(gp_cherry) == 1 and self.root == 2 and gp_cherry in [1, 2]:
            return 0, new_cherries_x, new_cherries_y
        for ch in tree.nw.successors(gp_cherry):
            if ch == p_cherry:
                continue
            if tree.nw.out_degree(ch) == 0:
                new_cherries_x.add(tuple(sorted([ch, y])))
                new_cherries_y.add(tuple(sorted([ch, x])))
        return len(new_cherries_x), new_cherries_x, new_cherries_y


# LEAF DISTANCE
class LeafDist:
    def __init__(self):
        # PARAMETERS
        self.name = "LeafDist"
        self.feature_names = ["Leaf distance (t)", "Leaf distance (d)", "LCA depth (t)", "LCA depth (d)",
                              "LCA distance (t)", "LCA distance (d)"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        # distances
        self.dist_n = pd.Series(dtype=float)
        self.dist = pd.DataFrame(dtype=float)

        self.dist_frac_n = pd.Series(dtype=float)
        self.dist_frac = pd.DataFrame(dtype=float)

        self.dist_lca_n = pd.Series(dtype=float)
        self.dist_lca = pd.DataFrame(dtype=float)

        # combinatorial
        self.comb_n = pd.Series(dtype=float)
        self.comb = pd.DataFrame(dtype=float)

        self.comb_frac_n = pd.Series(dtype=float)
        self.comb_frac = pd.DataFrame(dtype=float)

        self.comb_lca_n = pd.Series(dtype=float)
        self.comb_lca = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # tree height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

        self.changed_before = pd.Series(dtype=int)

    def make_data(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                  leaf_pair_in_forest, leaf_pair_in):
        self.data_column = pd.DataFrame(columns=self.feature_names)

        self.tree_dist_prev = copy.deepcopy(tree_height_dist)
        self.tree_comb_prev = copy.deepcopy(tree_height_comb)

        self.dist = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.dist_frac = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.dist_lca = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.comb_frac = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.comb_lca = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)

        self.dist_n = pd.Series(0, index=reducible_pairs, dtype=float)
        self.dist_frac_n = pd.Series(0, index=reducible_pairs, dtype=float)
        self.dist_lca_n = pd.Series(0, index=reducible_pairs, dtype=float)
        self.comb_n = pd.Series(0, index=reducible_pairs, dtype=float)
        self.comb_frac_n = pd.Series(0, index=reducible_pairs, dtype=float)
        self.comb_lca_n = pd.Series(0, index=reducible_pairs, dtype=float)

        for c in unique_cherries:
            for t, tree in tree_set.items():
                if not leaf_pair_in_forest.loc[c, t]:
                    continue
                leaf_dist_comb, x_comb, comb_lca, leaf_dist, x_dist, dist_lca = self.get_leaf_dist(tree, *c)
                # data stuff
                self.comb.loc[c, t] = leaf_dist_comb
                self.comb.loc[c[::-1], t] = leaf_dist_comb
                self.comb_frac.loc[c, t] = x_comb
                self.comb_frac.loc[c[::-1], t] = 1 - x_comb
                self.comb_lca.loc[c, t] = comb_lca
                self.comb_lca.loc[c[::-1], t] = comb_lca

                self.dist.loc[c, t] = leaf_dist
                self.dist.loc[c[::-1], t] = leaf_dist
                self.dist_frac.loc[c, t] = x_dist
                self.dist_frac.loc[c[::-1], t] = 1 - x_dist
                self.dist_lca.loc[c, t] = dist_lca
                self.dist_lca.loc[c[::-1], t] = dist_lca

        # DATA
        for c in unique_cherries:
            self.dist_n[c] = 0
            self.dist_n[c[::-1]] = 0
            self.comb_n[c] = 0
            self.comb_n[c[::-1]] = 0

            self.dist_lca_n[c] = 0
            self.dist_lca_n[c[::-1]] = 0
            self.comb_lca_n[c] = 0
            self.comb_lca_n[c[::-1]] = 0

            for t in tree_set:
                if np.isnan(self.dist.loc[c, t]):
                    continue
                # leaf distance
                if not tree_height_dist[t]:
                    dist_n_val = 0
                    dist_lca_val = 0
                else:
                    dist_n_val = self.dist.loc[c, t] / tree_height_dist[t]
                    dist_lca_val = self.dist_lca.loc[c, t] / tree_height_dist[t]
                self.dist_n[c] += dist_n_val
                self.dist_n[c[::-1]] += dist_n_val

                self.dist_lca_n[c] += dist_lca_val
                self.dist_lca_n[c[::-1]] += dist_lca_val

                if not tree_height_comb[t]:
                    comb_n_val = 0
                    comb_lca_val = 0
                else:
                    comb_n_val = self.comb.loc[c, t] / tree_height_comb[t]
                    comb_lca_val = self.comb_lca.loc[c, t] / tree_height_comb[t]
                self.comb_n[c] += comb_n_val
                self.comb_n[c[::-1]] += comb_n_val

                self.comb_lca_n[c] += comb_lca_val
                self.comb_lca_n[c[::-1]] += comb_lca_val

        # leaf distance frac
        self.dist_frac_n = self.dist_frac.sum(axis=1)
        self.comb_frac_n = self.comb_frac.sum(axis=1)

        self.d = leaf_pair_in

        # leaf distance
        self.data_column["Leaf distance (t)"] = self.comb_n / self.d
        self.data_column["Leaf distance (d)"] = self.dist_n / self.d

        # LCA depth
        self.data_column["LCA depth (t)"] = self.comb_lca_n / self.d
        self.data_column["LCA depth (d)"] = self.dist_lca_n / self.d

        # leaf distance frac
        self.data_column["LCA distance (t)"] = self.comb_frac_n / self.d
        self.data_column["LCA distance (d)"] = self.dist_frac_n / self.d
        return self.data_column

    @staticmethod
    def get_leaf_dist(tree, x, y):
        # so-called up down distance. Find first common ancestor,
        # and then compute distance from this node to both leaves
        lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(tree.nw, x, y)

        # LENGTH TO X
        for p in tree.nw.predecessors(x):
            p_x = p
        # LENGTH TO Y
        for p in tree.nw.predecessors(y):
            p_y = p

        # ignore length of branch to leaf itself
        dist_length = tree.nw.nodes[p_x]["node_length"] + tree.nw.nodes[p_y]["node_length"] - 2 * tree.nw.nodes[lca][
            "node_length"]
        x_dist = (tree.nw.nodes[x]["node_length"] - tree.nw.nodes[lca]["node_length"]) / max(
            [(tree.nw.nodes[x]["node_length"]
              + tree.nw.nodes[y]["node_length"]
              - 2 * tree.nw.nodes[lca][
                  "node_length"]), 0.001])
        dist_lca = tree.nw.nodes[lca]["node_length"]

        comb_length = tree.nw.nodes[p_x]["node_comb"] + tree.nw.nodes[p_y]["node_comb"] - 2 * tree.nw.nodes[lca][
            "node_comb"]
        x_comb = (tree.nw.nodes[x]["node_comb"] - tree.nw.nodes[lca]["node_comb"]) / (tree.nw.nodes[x]["node_comb"] +
                                                                                      tree.nw.nodes[y]["node_comb"] - 2
                                                                                      * tree.nw.nodes[lca]["node_comb"])
        comb_lca = tree.nw.nodes[lca]["node_comb"]

        return comb_length, x_comb, comb_lca, dist_length, x_dist, dist_lca


# LEAF HEIGHT
class LeafHeight:
    def __init__(self):
        # PARAMETERS
        self.name = "LeafHeight"
        self.feature_names = ["Leaf depth x (t)", "Leaf depth x (d)", "Leaf depth y (t)", "Leaf depth y (d)",
                              "Depth x/y (t)", "Depth x/y (d)"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        # distances
        self.x_dist_n = pd.Series(dtype=float)
        self.x_dist = pd.DataFrame(dtype=float)

        self.y_dist_n = pd.Series(dtype=float)
        self.y_dist = pd.DataFrame(dtype=float)

        self.xy_dist_n = pd.Series(dtype=float)
        self.x_vs_y = pd.DataFrame(dtype=float)

        # combinatorial
        self.x_comb_n = pd.Series(dtype=float)
        self.x_comb = pd.DataFrame(dtype=float)

        self.y_comb_n = pd.Series(dtype=float)
        self.y_comb = pd.DataFrame(dtype=float)

        self.xy_comb_n = pd.Series(dtype=float)
        self.x_vs_y_comb = pd.DataFrame(dtype=float)

        self.d = pd.Series(dtype=float)
        # height parameters
        self.tree_dist_prev = pd.Series(dtype=float)
        self.tree_comb_prev = pd.Series(dtype=float)

        self.changed_before = pd.Series(dtype=float)

    def make_data(self, reducible_pairs, tree_set, unique_cherries, tree_height_dist, tree_height_comb,
                  leaf_pair_in_forest, leaf_pair_in):   # almost same as leaf features
        self.data_column = pd.DataFrame(columns=self.feature_names, dtype=float)

        self.tree_dist_prev = copy.deepcopy(tree_height_dist)
        self.tree_comb_prev = copy.deepcopy(tree_height_comb)

        self.x_dist = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.y_dist = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.x_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.y_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.x_vs_y = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)
        self.x_vs_y_comb = pd.DataFrame(index=reducible_pairs, columns=tree_set, dtype=float)

        self.d = leaf_pair_in

        for c in unique_cherries:
            for t, tree in tree_set.items():
                if not leaf_pair_in_forest.loc[c, t]:
                    continue
                x_height, x_height_comb, y_height, y_height_comb, x_vs_y, x_vs_y_comb = self.get_leaf_height(tree, *c)
                # data stuff
                # distances
                self.x_dist.loc[c, t] = x_height
                self.x_dist.loc[c[::-1], t] = y_height

                self.y_dist.loc[c, t] = y_height
                self.y_dist.loc[c[::-1], t] = x_height

                self.x_vs_y.loc[c, t] = x_vs_y
                self.x_vs_y.loc[c[::-1], t] = 1 / x_vs_y
                # combinatorial
                self.x_comb.loc[c, t] = x_height_comb
                self.x_comb.loc[c[::-1], t] = y_height_comb

                self.y_comb.loc[c, t] = y_height_comb
                self.y_comb.loc[c[::-1], t] = x_height_comb

                self.x_vs_y_comb.loc[c, t] = x_vs_y_comb
                self.x_vs_y_comb.loc[c[::-1], t] = 1 / x_vs_y_comb

        # DATA
        # leaf heights
        self.x_dist_n = (self.x_dist / self.tree_dist_prev).sum(axis=1)
        self.y_dist_n = (self.y_dist / self.tree_dist_prev).sum(axis=1)

        self.x_comb_n = (self.x_comb / self.tree_comb_prev).sum(axis=1)
        self.y_comb_n = (self.y_comb / self.tree_comb_prev).sum(axis=1)

        self.data_column["Leaf depth x (d)"] = self.x_dist_n / self.d
        self.data_column["Leaf depth y (d)"] = self.y_dist_n / self.d

        self.data_column["Leaf depth x (t)"] = self.x_comb_n / self.d
        self.data_column["Leaf depth y (t)"] = self.y_comb_n / self.d

        # leaf distance frac
        self.xy_dist_n = self.x_vs_y.sum(axis=1)
        self.xy_comb_n = self.x_vs_y_comb.sum(axis=1)

        # normalize per c!
        for c in unique_cherries:
            dist_both = np.array([self.xy_dist_n[c] / self.d[c], self.xy_dist_n[c[::-1]] / self.d[c[::-1]]])
            dist_scaled = dist_both / sum(dist_both)
            self.data_column.loc[c, "Depth x/y (d)"] = dist_scaled[0]
            self.data_column.loc[c[::-1], "Depth x/y (d)"] = dist_scaled[1]

            comb_both = np.array([self.xy_comb_n[c] / self.d[c], self.xy_comb_n[c[::-1]] / self.d[c[::-1]]])
            comb_scaled = comb_both / sum(comb_both)
            self.data_column.loc[c, "Depth x/y (t)"] = comb_scaled[0]
            self.data_column.loc[c[::-1], "Depth x/y (t)"] = comb_scaled[1]

        return self.data_column

    @staticmethod
    def get_leaf_height(tree, x, y):
        for p in tree.nw.predecessors(x):
            p_x = p
        for p in tree.nw.predecessors(y):
            p_y = p

        x_height = tree.nw.nodes[p_x]["node_length"]
        y_height = tree.nw.nodes[p_y]["node_length"]

        x_vs_y = max([tree.nw.nodes[x]["node_length"], 0.001]) / max([tree.nw.nodes[y]["node_length"], 0.001])

        x_height_comb = tree.nw.nodes[p_x]["node_comb"]
        y_height_comb = tree.nw.nodes[p_y]["node_comb"]

        x_vs_y_comb = max([tree.nw.nodes[x]["node_comb"], 0.001]) / max([tree.nw.nodes[y]["node_comb"], 0.001])

        return x_height, x_height_comb, y_height, y_height_comb, x_vs_y, x_vs_y_comb


#####################################################################################
#######################             LEAF FEATURES             #######################
#####################################################################################
class LeafFeatures:
    def __init__(self, tree_env, train_phase=True):
        self.train_phase = train_phase

        self.name = []
        self.num_trees = len(tree_env.trees)
        self.data = pd.DataFrame(dtype=float)
        if self.train_phase:
            self.all_leaves = tree_env.leaves
            self.root = 0
        else:
            self.all_leaves = set(tree_env.labels.values())
            self.root = 2

        # LEAF IN TREES
        self.leaf_in_trees = LeafInTrees(self.all_leaves)

        # SIBLINGS
        self.siblings = Siblings()

        # DEPTH
        self.depth = Depth(self.root)

        self.init_fun(tree_env.reducible_pairs, tree_env.trees)

    def init_fun(self, reducible_pairs, tree_set):
        # LEAF IN TREES
        new_data_column = self.leaf_in_trees.init_fun(reducible_pairs, tree_set)
        # initialize data set based on current leaves
        self.data = pd.DataFrame(index=list(self.leaf_in_trees.pick_leaves), columns=self.name, dtype=float)

        self.data[self.leaf_in_trees.feature_names] = new_data_column

        # SIBLINGS
        new_data_column = self.siblings.init_fun(reducible_pairs, self.num_trees, self.leaf_in_trees.pick_leaves,
                                                 self.leaf_in_trees.all_leaves,
                                                 self.leaf_in_trees.leaf_in_tree_curr,
                                                 self.leaf_in_trees.num_leaf_in_trees)
        self.data[self.siblings.feature_names] = new_data_column

        # DEPTH
        new_data_column = self.depth.init_fun(tree_set, self.leaf_in_trees.pick_leaves, self.leaf_in_trees.all_leaves,
                                              self.leaf_in_trees.num_leaf_in_trees)
        self.data[self.depth.feature_names] = new_data_column

    # UPDATE FUNCTIONS
    def update_leaf_features_before(self, chosen_cherry, reducible_pairs, tree_set):
        # update LEAF IN TREES
        new_leaves, deleted_leaves, new_leaf_per_tree = self.leaf_in_trees.update(chosen_cherry, reducible_pairs,
                                                                                  tree_set)
        # update SIBLINGS
        self.siblings.update(reducible_pairs, new_leaves, chosen_cherry, self.leaf_in_trees.leaf_in_tree_curr, tree_set)
        # update DEPTH
        self.depth.update(reducible_pairs, new_leaves, new_leaf_per_tree, chosen_cherry, tree_set)

        return deleted_leaves

    def update_leaf_features_after(self, tree_set, deleted_leaves, deleted_trees):
        self.num_trees = len(tree_set)
        # get data LEAF IN TREES
        self.leaf_in_trees.update_after(deleted_trees)
        # get data SIBLINGS
        self.siblings.update_after(self.leaf_in_trees.pick_leaves, self.leaf_in_trees.all_leaves,
                                   deleted_trees, self.leaf_in_trees.num_leaf_in_trees)
        # get data DEPTH
        self.depth.update_after(self.leaf_in_trees.pick_leaves, deleted_trees, self.leaf_in_trees.num_leaf_in_trees)

        self.chosen_leaf_cleaning(deleted_leaves)
        self.update_data()
        pass

    def chosen_leaf_cleaning(self, deleted_leaves):
        if len(deleted_leaves):
            self.data.drop(list(deleted_leaves), inplace=True)
            # LEAF IN TREES
            self.leaf_in_trees.chosen_leaf_cleaning(deleted_leaves)
            # SIBLINGS
            self.siblings.chosen_leaf_cleaning(deleted_leaves)
            # DEPTH
            self.depth.chosen_leaf_cleaning(deleted_leaves)

    def update_data(self):
        self.data = pd.DataFrame(index=list(self.leaf_in_trees.pick_leaves), columns=self.name, dtype=float)
        # LEAF IN TREES
        self.data[self.leaf_in_trees.feature_names] = self.leaf_in_trees.data_column
        # SIBLINGS
        self.data[self.siblings.feature_names] = self.siblings.data_column
        # DEPTH
        self.data[self.depth.feature_names] = self.depth.data_column

    # RELABEL FUNCTION
    def relabel_trivial_cherries(self, x, y, num_trees, relabel_in_tree):
        # LEAF IN TREES
        self.leaf_in_trees.relabel(x, y, num_trees)
        # initialize data
        self.data = pd.DataFrame(columns=self.name)

        self.data[self.leaf_in_trees.feature_names] = self.leaf_in_trees.data_column

        # SIBLINGS
        self.siblings.relabel(x, y)
        self.data[self.siblings.feature_names] = self.siblings.data_column

        # SIBLINGS
        self.depth.relabel(x, y, relabel_in_tree)
        self.data[self.depth.feature_names] = self.depth.data_column


class LeafInTrees:
    def __init__(self, all_leaves):
        self.name = "LeafInTrees"
        # how many times does this leaf appear in the tree set compared to the leaf that appears max/avg
        self.feature_names = ["Leaf pickable", "Leaf in tree"]  # latter stays constant
        self.data_column = pd.DataFrame(columns=self.feature_names, dtype=float)
        self.all_leaves = all_leaves

        self.pick_leaves = set()
        self.leaf_in_tree_curr = pd.DataFrame(dtype=int)
        self.leaf_in_tree_cherry = pd.DataFrame(dtype=int)

        # d_pickable and n_tree are the same, namely, the number of trees the leaf is in: self.num_leaf_in_trees
        # d_cherry and d_missing are also the same: self.num_trees
        self.num_leaf_in_trees = pd.Series(dtype=float)
        self.n_pickable = pd.Series(dtype=float)
        self.num_trees = -1

    def init_fun(self, reducible_pairs, tree_set):
        self.num_trees = len(tree_set)
        # get current cherries
        for (x, y) in reducible_pairs:
            self.pick_leaves.update({x, y})
        self.leaf_in_tree_curr = pd.DataFrame(0, index=list(self.all_leaves), columns=list(tree_set))

        self.n_pickable = pd.Series(0, index=list(self.pick_leaves))
        self.num_leaf_in_trees = pd.Series(0, index=list(self.pick_leaves))

        # get initial missing leaves
        for t, tree in tree_set.items():
            self.leaf_in_tree_curr[t].loc[list(tree.leaves)] = 1

        # leaf in tree cherry
        self.leaf_in_tree_cherry = pd.DataFrame(0, index=list(self.pick_leaves), columns=list(tree_set))

        done_cherries = set()
        for (x, y), trees in reducible_pairs.items():
            if (x, y) in done_cherries:
                continue
            done_cherries.add((y, x))
            self.leaf_in_tree_cherry.loc[[x, y], list(trees)] = 1

        # get leaves in tree (not necessarily pickable)
        for l in self.pick_leaves:
            self.num_leaf_in_trees[l] = self.leaf_in_tree_curr.loc[l].sum()
            self.n_pickable[l] = self.leaf_in_tree_cherry.loc[l].sum()

        self.data_column["Leaf pickable"] = self.n_pickable / self.num_leaf_in_trees
        self.data_column["Leaf in tree"] = self.num_leaf_in_trees / self.num_trees

        return self.data_column

    def update(self, chosen_cherry, reducible_pairs, tree_set):
        # find new leaves
        x_chosen, y_chosen = chosen_cherry
        new_leaf_per_tree = dict()
        new_leaves = set()
        new_unique_cherries = set()

        # and then for deleted ones
        self.leaf_in_tree_curr.loc[x_chosen, list(reducible_pairs[chosen_cherry])] = 0
        self.leaf_in_tree_cherry.loc[x_chosen, list(reducible_pairs[chosen_cherry])] = 0
        x_is_leaf, y_is_leaf = False, False

        # get new_leaves
        for t in reducible_pairs[chosen_cherry]:
            self.n_pickable[x_chosen] -= 1
            self.num_leaf_in_trees[x_chosen] -= 1
            if len(tree_set[t].leaves) == 2:
                self.n_pickable[y_chosen] -= 1
                self.leaf_in_tree_cherry.loc[y_chosen, t] = 0
                continue
            for p in tree_set[t].nw.predecessors(x_chosen):
                p_c = p
            if tree_set[t].nw.out_degree(p_c) > 2:
                for c in tree_set[t].nw.successors(p_c):
                    any_leaf = False
                    if c in chosen_cherry:
                        continue
                    if c in tree_set[t].leaves:
                        any_leaf = True
                        y_is_leaf = True
                        break
                if not any_leaf:
                    self.n_pickable[y_chosen] -= 1
                    self.leaf_in_tree_cherry.loc[y_chosen, t] = 0
                continue

            # CHECK FOR ALL NEW LEAVES
            for p in tree_set[t].nw.predecessors(p_c):
                gp_c = p
            any_leaf = False
            for l in tree_set[t].nw.successors(gp_c):
                if l == p_c:
                    continue
                if l in tree_set[t].leaves:
                    y_is_leaf = True
                    any_leaf = True
                    if l not in self.pick_leaves:
                        new_leaves.add(l)
                    elif not self.leaf_in_tree_cherry.loc[l, t]:
                        self.leaf_in_tree_cherry.loc[l, t] = 1
                        self.n_pickable[l] += 1
                    if (l, y_chosen) not in reducible_pairs:
                        new_unique_cherries.add(tuple(sorted((l, y_chosen))))
                    if t in new_leaf_per_tree:
                        new_leaf_per_tree[t].add(l)
                    else:
                        new_leaf_per_tree[t] = {l}
            if not any_leaf:
                self.n_pickable[y_chosen] -= 1
                self.leaf_in_tree_cherry.loc[y_chosen, t] = 0

        self.pick_leaves.update(new_leaves)

        # delete leaves (only x or y)
        del_leaves = set()
        # check for current cherries
        not_picked_trees = [t for t in tree_set if t not in reducible_pairs[chosen_cherry]]
        if not_picked_trees:
            x_is_leaf = self.leaf_in_tree_cherry.loc[x_chosen][not_picked_trees].any() # only possible for x

        if not y_is_leaf:
            for c in reducible_pairs:
                if c == chosen_cherry or c == chosen_cherry[::-1]:
                    continue
                if y_chosen in c:
                    y_is_leaf = True
                    break

        if not x_is_leaf:
            self.pick_leaves.remove(x_chosen)
            del_leaves.add(x_chosen)
        if not y_is_leaf:
            self.pick_leaves.remove(y_chosen)
            del_leaves.add(y_chosen)

        # first new ones
        for l in new_leaves:
            if l not in self.leaf_in_tree_cherry.index:
                new_row = 0
                self.leaf_in_tree_cherry.loc[l] = new_row
            if l not in self.n_pickable:
                self.n_pickable[l] = 0
        for t, leaves in new_leaf_per_tree.items():
            for l in leaves:
                if l not in new_leaves:
                    continue
                self.leaf_in_tree_cherry.loc[l, t] = 1
                self.n_pickable[l] += 1
        for l in new_leaves:
            self.num_leaf_in_trees[l] = self.leaf_in_tree_curr.loc[l].sum()

        # check if x chosen is fully deleted
        if not x_is_leaf and self.leaf_in_tree_curr.loc[x_chosen].sum() == 0:
            self.all_leaves.remove(x_chosen)
            self.leaf_in_tree_curr.drop(x_chosen, axis=0, inplace=True)
            self.leaf_in_tree_cherry.drop(x_chosen, axis=0, inplace=True)

        return new_leaves, del_leaves, new_leaf_per_tree

    def chosen_leaf_cleaning(self, deleted_leaves):
        for l in deleted_leaves:
            self.n_pickable.drop(l, axis=0, inplace=True)
            self.num_leaf_in_trees.drop(l, axis=0, inplace=True)

    def update_after(self, deleted_trees):
        self.num_trees -= len(deleted_trees)
        self.data_column = pd.DataFrame(index=list(self.pick_leaves), columns=self.feature_names, dtype=float)

        # delete columns from some information sets
        if len(deleted_trees):
            self.leaf_in_tree_curr.drop(deleted_trees, axis=1, inplace=True)
            self.leaf_in_tree_cherry.drop(deleted_trees, axis=1, inplace=True)

        # update data set
        for l in self.pick_leaves:
            self.data_column.loc[l, "Leaf pickable"] = self.n_pickable[l] / self.num_leaf_in_trees[l]
            self.data_column.loc[l, "Leaf in tree"] = self.num_leaf_in_trees[l] / self.num_trees

    def relabel(self, x, y, num_trees):
        self.leaf_in_tree_curr.loc[y] = self.leaf_in_tree_curr.loc[[x, y]].max()
        self.leaf_in_tree_curr.loc[x] = 0
        self.leaf_in_tree_cherry.loc[y] = self.leaf_in_tree_cherry.loc[[x, y]].max()
        self.leaf_in_tree_cherry.loc[x] = 0

        self.n_pickable[y] = self.leaf_in_tree_cherry.loc[y].sum()
        self.num_leaf_in_trees[y] = self.leaf_in_tree_curr.loc[y].sum()

        self.data_column.loc[y, "Leaf pickable"] = self.n_pickable[y] / self.num_leaf_in_trees[y]
        self.data_column.loc[y, "Leaf in tree"] = self.num_leaf_in_trees[y] / self.num_trees

class Siblings:
    def __init__(self):
        # Mean number of siblings per tree: about the number, not about which siblings
        self.name = "Siblings"
        self.feature_names = ["Siblings mean", "Siblings std"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        self.siblings_per_tree = pd.DataFrame(dtype=int)
        self.n_mean = pd.Series(dtype=float)
        self.n_std = pd.Series(dtype=float)
        self.d_mean = pd.Series(dtype=float)
        self.d_std = -1

    def init_fun(self, reducible_pairs, tree_num, pick_leaves, all_leaves, leaf_in_tree_curr, num_leaf_in_trees):
        self.d_mean = num_leaf_in_trees * len(all_leaves)
        self.d_std = len(all_leaves)

        self.siblings_per_tree = pd.DataFrame(np.nan, index=list(pick_leaves), columns=range(tree_num))
        self.n_mean = pd.Series(0, index=list(pick_leaves))
        self.n_std = pd.Series(0, index=list(pick_leaves))

        for l in pick_leaves:
            self.siblings_per_tree.loc[l] = [np.nan if not l_in_t else 0 for l_in_t in leaf_in_tree_curr.loc[l]]
        done = set()
        for (x, y), trees in reducible_pairs.items():
            if (x, y) in done:
                continue
            done.add((y, x))
            self.siblings_per_tree.loc[x][list(trees)] += 1
            self.siblings_per_tree.loc[y][list(trees)] += 1

        for l in pick_leaves:
            self.n_mean[l] = self.siblings_per_tree.loc[l].sum()
            self.n_std[l] = self.siblings_per_tree.loc[l].std()
            if np.isnan(self.n_std[l]) and (1 - self.siblings_per_tree.loc[l].isna()).sum() == 1:
                self.n_std[l] = 0

        self.data_column["Siblings mean"] = self.n_mean / self.d_mean
        self.data_column["Siblings std"] = self.n_std / self.d_std

        return self.data_column

    def update(self, reducible_pairs, new_leaves, chosen_cherry, leaf_in_tree_curr, tree_set):
        for l in new_leaves:
            if l not in self.n_mean:
                self.n_mean[l] = 0
            if l not in self.siblings_per_tree.index:
                self.siblings_per_tree.loc[l] = [np.nan if not l_in_t else 0 for l_in_t in leaf_in_tree_curr.loc[l]]

        x_chosen, y_chosen = chosen_cherry
        all_leaves_changed = set()
        for t in reducible_pairs[chosen_cherry]:
            all_leaves_changed = self.update_per_leaf_tree(tree_set[t], x_chosen, t, all_leaves_changed,
                                                           y_chosen=y_chosen)
        for l in all_leaves_changed:
            self.n_std[l] = self.siblings_per_tree.loc[l].std()
            if np.isnan(self.n_std[l]) and (1 - self.siblings_per_tree.loc[l].isna()).sum() == 1:
                self.n_std[l] = 0

    def update_per_leaf_tree(self, tree, l, t, all_leaves_changed, leaves_del_in_tree=None, y_chosen=None):
        if leaves_del_in_tree is None:
            leaves_del_in_tree = {l}

        self.siblings_per_tree.loc[l, t] = np.nan

        for p in tree.nw.predecessors(l):
            p_l = p

        siblings = set()
        for c in tree.nw.successors(p_l):
            if c in leaves_del_in_tree:
                continue
            if c not in tree.leaves:
                continue
            siblings.add(c)

        for c in siblings:
            self.n_mean[c] -= 1
            self.siblings_per_tree.loc[c, t] -= 1
            all_leaves_changed.add(c)

        del_under_parent = 1
        if len(leaves_del_in_tree) > 1:
            del_under_parent = 0
            for c in tree.nw.successors(p_l):
                if c not in leaves_del_in_tree:
                    continue
                del_under_parent += 1

        if tree.nw.out_degree(p_l) - del_under_parent > 1 or len(tree.leaves) == 2:
            return all_leaves_changed

        # if subtraction node happens
        if y_chosen is not None:
            y = y_chosen
        else:
            y = list(siblings)[0]

        # get grandparent of l
        for p in tree.nw.predecessors(p_l):
            gp_l = p
        num_siblings = 0
        for c in tree.nw.successors(gp_l):
            if c == p_l:
                continue
            if c not in tree.leaves:
                continue
            num_siblings += 1
            self.n_mean[c] += 1
            self.siblings_per_tree.loc[c, t] += 1
            all_leaves_changed.add(c)

        self.n_mean[y] += num_siblings
        self.siblings_per_tree.loc[y, t] = num_siblings
        return all_leaves_changed

    def chosen_leaf_cleaning(self, deleted_leaves):
        for l in deleted_leaves:
            self.n_mean.drop(l, axis=0, inplace=True)
            self.n_std.drop(l, axis=0, inplace=True)

    def update_after(self, pick_leaves, all_leaves, deleted_trees, num_leaf_in_trees):
        self.d_mean = num_leaf_in_trees * len(all_leaves)
        self.d_std = len(all_leaves)
        self.data_column = pd.DataFrame(index=list(pick_leaves), columns=self.feature_names, dtype=float)

        if len(deleted_trees):
            self.siblings_per_tree.drop(deleted_trees, axis=1, inplace=True)
        for l in pick_leaves:
            self.data_column.loc[l, "Siblings mean"] = self.n_mean[l] / self.d_mean[l]
            self.data_column.loc[l, "Siblings std"] = self.n_std[l] / self.d_std

    def relabel(self, x, y):
        self.siblings_per_tree.loc[y] = self.siblings_per_tree.loc[[x, y]].max()

        self.n_mean[y] = self.siblings_per_tree.loc[y].sum()
        self.n_std[y] = self.siblings_per_tree.loc[y].std()
        if np.isnan(self.n_std[y]) and (1 - self.siblings_per_tree.loc[y].isna()).sum() == 1:
            self.n_std[y] = 0

        self.data_column.loc[y, "Siblings mean"] = self.n_mean[y] / self.d_mean[y]
        self.data_column.loc[y, "Siblings std"] = self.n_std[y] / self.d_std


class Depth:
    def __init__(self, root):
        # Mean number of siblings per tree: about the number, not about which siblings
        self.root = root
        self.name = "Depth"
        self.feature_names = ["Topo. depth mean", "Topo. depth std", "Dist. depth mean", "Dist. depth std"]
        self.data_column = pd.DataFrame(columns=self.feature_names)

        self.depth_per_tree_comb = pd.DataFrame(dtype=int)
        self.depth_per_tree_dist = pd.DataFrame(dtype=int)

        self.n_comb_mean = pd.Series(dtype=float)
        self.n_comb_std = pd.Series(dtype=float)
        self.n_dist_mean = pd.Series(dtype=float)
        self.n_dist_std = pd.Series(dtype=float)
        self.d_comb = pd.Series(dtype=float)
        self.d_dist = pd.Series(dtype=float)

        self.tree_comb = pd.Series()
        self.tree_dist = pd.Series()
        # distances
        self.tree_level_width_dist = {}
        self.tree_level_dist_id = {}
        self.height_level_dist = {}
        self.max_id_dist = {}
        self.max_dist = -1

        # combinatorial
        self.tree_level_width_comb = {}
        self.max_comb = -1

    def init_fun(self, tree_set, pick_leaves, all_leaves, num_leaf_in_trees):
        self.depth_per_tree_comb = pd.DataFrame(np.nan, index=list(all_leaves), columns=tree_set.keys())
        self.depth_per_tree_dist = pd.DataFrame(np.nan, index=list(all_leaves), columns=tree_set.keys())

        self.n_comb_mean = pd.Series(0, index=list(pick_leaves))
        self.n_comb_std = pd.Series(0, index=list(pick_leaves))
        self.n_dist_mean = pd.Series(0, index=list(pick_leaves))
        self.n_dist_std = pd.Series(0, index=list(pick_leaves))

        for t, tree in tree_set.items():
            for l in tree.leaves:
                self.depth_per_tree_dist.loc[l, t] = tree.nw.nodes[l]["node_length"]
                self.depth_per_tree_comb.loc[l, t] = tree.nw.nodes[l]["node_comb"]

        for l in pick_leaves:
            self.n_comb_mean[l] = self.depth_per_tree_comb.loc[l].sum()
            self.n_comb_std[l] = self.depth_per_tree_comb.loc[l].std()
            if np.isnan(self.n_comb_std[l]) and (1 - self.depth_per_tree_comb.loc[l].isna()).sum() == 1:
                self.n_comb_std[l] = 0
            self.n_dist_mean[l] = self.depth_per_tree_dist.loc[l].sum()
            self.n_dist_std[l] = self.depth_per_tree_dist.loc[l].std()
            if np.isnan(self.n_dist_std[l]) and (1 - self.depth_per_tree_dist.loc[l].isna()).sum() == 1:
                self.n_dist_std[l] = 0

        self.get_tree_leaf_depth_metric(tree_set)

        # INSTEAD OF self.depth_per_tree_comb/self.tree_comb, we take the mean of self.tree_comb (alternatively the max)
        self.d_comb = num_leaf_in_trees * self.max_comb
        self.d_dist = num_leaf_in_trees * self.max_dist

        self.data_column["Topo. depth mean"] = self.n_comb_mean / self.d_comb
        self.data_column["Topo. depth std"] = self.n_comb_std / self.max_comb
        self.data_column["Dist. depth mean"] = self.n_dist_mean / self.d_dist
        self.data_column["Dist. depth std"] = self.n_dist_std / self.max_dist

        return self.data_column

    def update(self, reducible_pairs, new_leaves, new_leaf_per_tree, chosen_cherry, tree_set):
        self.update_metric(reducible_pairs[chosen_cherry], chosen_cherry[0], tree_set)
        x_chosen, y_chosen = chosen_cherry
        for t in reducible_pairs[chosen_cherry]:
            self.n_comb_mean[x_chosen] -= self.depth_per_tree_comb.loc[x_chosen, t]
            self.depth_per_tree_comb.loc[x_chosen, t] = np.nan
            self.n_dist_mean[x_chosen] -= self.depth_per_tree_dist.loc[x_chosen, t]
            self.depth_per_tree_dist.loc[x_chosen, t] = np.nan
            # topo depth of y is changed
            for p in tree_set[t].nw.predecessors(x_chosen):
                p_x = p
            if tree_set[t].nw.out_degree(p_x) == 2:
                self.depth_per_tree_comb.loc[y_chosen, t] -= 1
                self.n_comb_mean[y_chosen] -= 1

        # NEW LEAVES
        for l in new_leaves:
            # comb
            if l not in self.n_comb_mean:
                self.n_comb_mean[l] = self.depth_per_tree_comb.loc[l].sum()
            # dist
            if l not in self.n_dist_mean:
                self.n_dist_mean.loc[l] = self.depth_per_tree_dist.loc[l].sum()

        changed_leaves = {y_chosen}
        for t, leaves in new_leaf_per_tree.items():
            changed_leaves.update(leaves)
            for l in leaves:
                if l in new_leaves:
                    continue
                # BINARY: should become a set of leaves
                self.n_comb_mean[l] += tree_set[t].nw.nodes[l]["node_comb"]
                self.n_dist_mean[l] += tree_set[t].nw.nodes[l]["node_length"]

        # std needs to be done again of y_chosen
        for l in changed_leaves:
            self.n_comb_std[l] = self.depth_per_tree_comb.loc[l].std()
            if np.isnan(self.n_comb_std[l]) and (1 - self.depth_per_tree_comb.loc[l].isna()).sum() == 1:
                self.n_comb_std[l] = 0
            self.n_dist_std[l] = self.depth_per_tree_dist.loc[l].std()
            if np.isnan(self.n_dist_std[l]) and (1 - self.depth_per_tree_dist.loc[l].isna()).sum() == 1:
                self.n_dist_std[l] = 0

    def chosen_leaf_cleaning(self, deleted_leaves):
        for l in deleted_leaves:
            self.n_comb_mean.drop(l, axis=0, inplace=True)
            self.n_comb_std.drop(l, axis=0, inplace=True)
            self.n_dist_mean.drop(l, axis=0, inplace=True)
            self.n_dist_std.drop(l, axis=0, inplace=True)

    def update_after(self, pick_leaves, deleted_trees, num_leaf_in_trees):
        self.d_comb = num_leaf_in_trees * self.max_comb
        self.d_dist = num_leaf_in_trees * self.max_dist
        self.data_column = pd.DataFrame(index=list(pick_leaves), columns=self.feature_names, dtype=float)
        if len(deleted_trees):
            self.depth_per_tree_comb.drop(deleted_trees, axis=1, inplace=True)
            self.depth_per_tree_dist.drop(deleted_trees, axis=1, inplace=True)

        for l in pick_leaves:
            self.data_column.loc[l, "Topo. depth mean"] = self.n_comb_mean[l] / self.d_comb[l]
            self.data_column.loc[l, "Topo. depth std"] = self.n_comb_std[l] / self.max_comb
            self.data_column.loc[l, "Dist. depth mean"] = self.n_dist_mean[l] / self.d_dist[l]
            self.data_column.loc[l, "Dist. depth std"] = self.n_dist_std[l] / self.max_dist

    def relabel(self, x, y, relabel_in_tree):
        self.depth_per_tree_comb.loc[y] = self.depth_per_tree_comb.loc[[x, y]].max()
        self.depth_per_tree_dist.loc[y] = self.depth_per_tree_dist.loc[[x, y]].max()

        self.n_comb_mean[y] = self.depth_per_tree_comb.loc[y].sum()
        self.n_comb_std[y] = self.depth_per_tree_comb.loc[y].std()
        if np.isnan(self.n_comb_std[y]) and (1 - self.depth_per_tree_comb.loc[y].isna()).sum() == 1:
            self.n_comb_std[y] = 0
        self.n_dist_mean[y] = self.depth_per_tree_dist.loc[y].sum()
        self.n_dist_std[y] = self.depth_per_tree_dist.loc[y].std()
        if np.isnan(self.n_dist_std[y]) and (1 - self.depth_per_tree_dist.loc[y].isna()).sum() == 1:
            self.n_dist_std[y] = 0

        self.data_column.loc[y, "Topo. depth mean"] = self.n_comb_mean[y] / self.d_comb[y]
        self.data_column.loc[y, "Topo. depth std"] = self.n_comb_std[y] / self.max_comb
        self.data_column.loc[y, "Dist. depth mean"] = self.n_dist_mean[y] / self.d_dist[y]
        self.data_column.loc[y, "Dist. depth std"] = self.n_dist_std[y] / self.max_dist

        # relabel id of tree depth
        for t in relabel_in_tree:
            self.tree_level_dist_id[t][y] = self.tree_level_dist_id[t][x]
            del self.tree_level_dist_id[t][x]

    def get_tree_leaf_depth_metric(self, tree_set):
        self.tree_comb = pd.Series(index=tree_set)
        self.tree_dist = pd.Series(index=tree_set)
        for t, tree in tree_set.items():
            self.metric_per_tree(t, tree)

        self.max_comb = self.tree_comb.max()
        self.max_dist = self.tree_dist.max()

    def metric_per_tree(self, t, tree):
        self.tree_level_width_comb[t] = []
        self.tree_level_width_dist[t] = []
        self.height_level_dist[t] = []
        tmp_tree_level_comb = {}
        tmp_tree_level_dist = {}
        for n in tree.nw.nodes:
            if self.root == 2 and n == 1:
                continue
            # COMBINATORIAL
            comb_height = tree.nw.nodes[n]["node_comb"]
            if comb_height in tmp_tree_level_comb:
                tmp_tree_level_comb[comb_height] += 1
            else:
                tmp_tree_level_comb[comb_height] = 1
            # DISTANCES
            if n not in tree.leaves:
                continue
            height = np.round(tree.nw.nodes[n]["node_length"], 10)
            if height in tmp_tree_level_dist:
                tmp_tree_level_dist[height].add(n)
            else:
                tmp_tree_level_dist[height] = {n}
        # find max
        # COMBINATORIAL
        sorted_tmp_comb = dict(sorted(tmp_tree_level_comb.items()))
        for comb_height, num_nodes in sorted_tmp_comb.items():
            self.tree_level_width_comb[t].append(num_nodes)
        self.tree_comb[t] = len(self.tree_level_width_comb[t]) - 1
        # DISTANCES
        sorted_tmp_dist = dict(sorted(tmp_tree_level_dist.items()))
        self.tree_level_dist_id[t] = {}
        for i, (dist_height, nodes) in enumerate(sorted_tmp_dist.items()):
            self.height_level_dist[t].append(dist_height)
            self.tree_level_width_dist[t].append(len(nodes))
            for _n in nodes:
                self.tree_level_dist_id[t][_n] = i
        self.max_id_dist[t] = len(self.height_level_dist[t]) - 1
        self.tree_dist[t] = self.height_level_dist[t][self.max_id_dist[t]]

    def update_metric(self, trees, l, tree_set):  # change metric
        change_height_dist_bool = False
        change_height_comb_bool = False
        for t in trees:
            more_siblings_curr_level = False
            # check if there are more siblings in current level
            for p in tree_set[t].nw.predecessors(l):
                p_y = p
            if tree_set[t].nw.out_degree(p_y) > 2:
                more_siblings_curr_level = True
            # COMBINATORIAL
            height = int(self.depth_per_tree_comb.loc[l, t])
            if more_siblings_curr_level:
                self.tree_level_width_comb[t][height] -= 1  # only x is gone from this level
            else:
                self.tree_level_width_comb[t][height] -= 2  # both x and y are gone from this level
                if self.tree_level_width_comb[t][height] == 0:
                    if abs(self.max_comb - self.tree_comb[t]) < 1e-3:
                        change_height_comb_bool = True
                    self.tree_comb[t] = height - 1

            # DISTANCES
            level_reduced = self.tree_level_dist_id[t][l]
            self.tree_level_width_dist[t][level_reduced] -= 1

            if self.tree_level_width_dist[t][level_reduced] == 0 and self.max_id_dist[t] == level_reduced:
                # update max height tree
                if abs(self.max_dist - self.tree_dist[t]) < 1e-3:
                    change_height_dist_bool = True
                i = 1
                while True:
                    if self.tree_level_width_dist[t][level_reduced - i] > 0:
                        self.max_id_dist[t] = level_reduced - i
                        break
                    i += 1
                self.tree_dist[t] = self.height_level_dist[t][self.max_id_dist[t]]

        # update max height level of all trees
        if change_height_comb_bool:
            new_max_comb = self.tree_comb.max()
            if abs(new_max_comb - self.max_comb) > 10e-3:
                self.max_comb = new_max_comb
        if change_height_dist_bool:
            new_max_dist = self.tree_dist.max()
            if abs(new_max_dist - self.max_dist) > 10e-3:
                self.max_dist = new_max_dist

