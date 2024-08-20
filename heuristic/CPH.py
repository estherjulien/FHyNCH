import networkx as nx
import ast
import re
import time
import joblib
import pandas as pd

import numpy as np
from copy import deepcopy
from copy import copy
from heuristic.features import CherryFeatures, LeafFeatures

import warnings
warnings.filterwarnings("ignore")
'''
This script consists of 4 parts: 
- Help functions (mainly for handling the input)
- Input_Set CLASS: code for opening the input and running the Cherry Picking Heuristic (CPH)
- PhT CLASS: environment for a phylogenetic tree
- PhN CLASS: environment for a phylogenetic network
'''

#######################################################################
########                   HELP FUNCTIONS                      ########
#######################################################################


# Write length newick: convert ":" to "," and then evaluate as list of lists using ast.literal_eval
# Then, in each list, the node is followed by the length of the incoming arc.
# This only works as long as each branch has length and all internal nodes are labeled.
def newick_to_tree(newick, current_labels=dict()):
    # newick = newick[:-1]
    distances = False
    # presence of : indicates the use of lengths in the trees
    if ":" in newick:
        distances = True
        # taxon names may already be enclosed by " or ', otherwise, we add these now
        if "'" not in newick and '"' not in newick:
            newick = re.sub(r"([,\(])([a-zA-Z\d]+)", r"\1'\2", newick)
            newick = re.sub(r"([a-zA-Z\d]):", r"\1':", newick)
        newick = newick.replace(":", ",")
    else:
        # taxon names may already be enclosed by " or ', otherwise, we add these now
        if not "'" in newick and not '"' in newick:
            newick = re.sub(r"([,\(])([a-zA-Z\d]+)", r"\1'\2", newick)
            newick = re.sub(r"([a-zA-Z\d])([,\(\)])", r"\1'\2", newick)
    # turn the string into a pyhton nested list using [ instead of (
    newick = newick.replace("(", "[")
    newick = newick.replace(")", "]")
    nestedtree = ast.literal_eval(newick)

    # parse the nested list into a list of edges with some additional information about the leaves
    # we start with the root 2, so that we can append a root edge (1,2)
    edges, leaves, current_labels, current_node = nested_list_to_tree(nestedtree, 2, current_labels, distances=distances)
    # put all this information into a networkx DiGraph with or without distances/lengths
    tree = nx.DiGraph()
    if distances:
        edges.append((1, 2, 0))
        tree.add_weighted_edges_from(edges, weight='length')

    else:
        edges.append((1, 2, 0))
        # tree.add_edges_from(edges)
        tree.add_weighted_edges_from(edges, weight='length')
    add_node_attributes(tree, distances=distances, root=2)
    return tree, leaves, deepcopy(leaves), current_labels, distances


# Auxiliary function to convert list of lists to tree (graph)
# Works recursively, where we keep track of the nodes we have already used
# Leaves are nodes with negative integer as ID, and already existing taxa are coupled to node IDs by current_labels.
def nested_list_to_tree(nestedList, next_node, current_labels, distances=False):
    edges = []
    leaves = set()
    top_node = next_node
    current_node = next_node + 1
    if distances:
        # each element in the sublist has 2 properties, the subtree, and the length, which are adjacent in nestedList
        for i in range(0, len(nestedList), 2):
            t = nestedList[i]
            length = nestedList[i + 1]
            if type(t) == list:  # Not a leaf
                edges.append((top_node, current_node, length))
                extra_edges, extra_leaves, current_labels, current_node = nested_list_to_tree(t, current_node,
                                                                                              current_labels,
                                                                                              distances=distances)
            else:  # A leaf
                if str(t) not in current_labels:
                    current_labels[str(t)] = -len(current_labels)
                edges.append((top_node, current_labels[str(t)], length))
                extra_edges = []
                extra_leaves = {current_labels[str(t)]}
            edges = edges + extra_edges
            leaves = leaves.union(extra_leaves)
    else:
        # no lengths/distances, so each subtree is simply an element of nestedList
        for t in nestedList:
            length = 1
            if type(t) == list:
                edges.append((top_node, current_node, length))
                extra_edges, extra_leaves, current_labels, current_node = nested_list_to_tree(t, current_node,
                                                                                              current_labels)
            else:
                if str(t) not in current_labels:
                    current_labels[str(t)] = -len(current_labels)
                edges.append((top_node, current_labels[str(t)], length))
                extra_edges = []
                extra_leaves = {current_labels[str(t)]}
            edges = edges + extra_edges
            leaves = leaves.union(extra_leaves)
    return edges, leaves, current_labels, current_node


def nested_list_to_tree_network(nestedList, next_node):
    edges = []
    leaves = set()
    labels = []
    top_node = next_node
    current_node = next_node + 1
    for t in nestedList:
        edges.append((top_node, current_node))
        if type(t) == list:
            extra_edges, extra_leaves, extra_labels, current_node = nested_list_to_tree_network(t, current_node)
        else:
            extra_edges = []
            extra_leaves = set([str(t)])
            extra_labels = [[str(t), current_node]]
            current_node += 1
        edges = edges + extra_edges
        leaves = leaves.union(extra_leaves)
        labels = labels + extra_labels
    return edges, leaves, labels, current_node


# per node, add the edge based and comb height of the node as an attribute.
def add_node_attributes(tree, distances=True, root=0):
    attrs = dict()
    for x in tree.nodes:
        if root == 2 and x == 1:
            attrs[x] = {"node_length": None,
                        "node_comb": None}
        elif x == root:
            attrs[x] = {"node_length": 0,
                        "node_comb": 0}
        else:
            node_length, node_comb = find_leaf_depth(tree, x, root, distances)
            attrs[x] = {"node_length": node_length,
                        "node_comb": node_comb}

    nx.set_node_attributes(tree, attrs)


def find_leaf_depth(tree, x, root, distances=True):
    # get path from root to x
    px = x
    path = [x]
    while px != root:
        for p in tree.predecessors(px):
            px = p
        path.append(px)

    node_comb = len(path) - 1
    if distances:
        node_dist = 0
        for i, p in enumerate(path[:-1]):
            node_dist += tree.edges[path[i+1], p]["length"]
    else:
        node_dist = node_comb

    return node_dist, node_comb


# Modifies a cherry-picking sequence so that it represents a network with exactly one root.
# A sequence may be such that reconstructing a network from the sequence results in multiple roots
# This function adds some pairs to the sequence so that the network has a single root.
# returns the new sequence, and also modifies the sets of trees reduced by each pair in the sequence,
# so that the new pairs are also represented (they reduce no trees)
def sequence_add_roots(seq, red_trees):
    leaves_encountered = set()
    roots = set()
    # The roots can be found by going back through the sequence and finding pairs where the second element has not been
    # encountered in the sequence yet
    for pair in reversed(seq):
        if pair[1] not in leaves_encountered:
            roots.add(pair[1])
        leaves_encountered.add(pair[0])
        leaves_encountered.add(pair[1])
    roots = list(roots)
    # Now add some pairs to make sure each second element is already part of some pair in the sequence read backwards,
    # except for the last pair in the sequence
    for i in range(len(roots) - 1):
        seq.append((roots[i], roots[i + 1]))
        # none of the trees are reduced by the new pairs.
        red_trees.append(set())
    return seq, red_trees


# return leaves from network
def get_leaves(net):
    return {u for u in net.nodes() if net.out_degree(u) == 0}

#######################################################################
########            INPUT SET CLASS with CPH method            ########
#######################################################################


# Methods for sets of phylogenetic trees
class Input_Set:
    def __init__(self, newick_strings=[], tree_set=None, instance=0, leaves=None, job_id=None, debug_mode=False):
        # The dictionary of trees
        self.trees = dict()
        # the set of leaf labels of the trees
        self.labels = dict()
        self.labels_reversed = dict()
        self.leaves = set()
        self.reducible_pairs = dict()
        self.root = -1
        self.debug_mode = debug_mode
        self.missing_leaves = dict()
        self.added_to_leaf = dict()
        self.leaf_to_tree = dict()

        self.instance = instance
        self.job_id = job_id if job_id is not None else instance
        # the current best sequence we have found for this set of trees
        self.best_seq = None
        # the list of reduced trees for each of the pairs in the best sequence
        self.best_red_trees = None

        # the best sequence for the algorithm using lengths as input as well
        self.best_seq_with_lengths = None
        # the sets of reduced trees for each pair in this sequence
        self.best_seq_with_lengths_red_trees = None
        # the height of each pair in this sequence
        self.best_seq_with_lengths_heights = None

        # true if distances are used
        self.distances = True
        # computation times
        self.CPS_Compute_Time = 0
        self.CPS_Compute_Reps = 0
        self.DurationPerTrial = []
        self.RetPerTrial = []
        self.SeqPerTrial = []
        self.PartSeqPerTrial = []

        if tree_set is None:
            # read the input trees in 'newick_strings'
            tree_num = 0
            for n in newick_strings:
                tree = PhT()
                self.trees[len(self.trees)] = tree
                self.labels, distances_in_tree = tree.tree_from_newick(newick=n, current_labels=self.labels)
                self.distances = self.distances and distances_in_tree
                tree_num += 1
            # ONLY UNION
            self.leaves = set(self.labels.keys())
        else:
            self.trees = {t: PhT(tree) for t, tree in tree_set.items()}
            self.leaves = leaves
            self.labels = {l: l for l in self.leaves}

        self.num_leaves = len(self.leaves)

        # make a reverse dictionary for the leaf labels, to look up the label of a given node
        for l, i in self.labels.items():
            self.labels_reversed[i] = l

    # Make a deepcopy of an instance
    def __deepcopy__(self, memodict={}):
        copy_inputs = Input_Set()
        copy_inputs.trees = deepcopy(self.trees, memodict)
        copy_inputs.labels = deepcopy(self.labels, memodict)
        copy_inputs.labels_reversed = deepcopy(self.labels_reversed, memodict)
        copy_inputs.leaves = deepcopy(self.leaves, memodict)
        copy_inputs.missing_leaves = deepcopy(self.missing_leaves)
        copy_inputs.debug_mode = deepcopy(self.debug_mode)
        return copy_inputs

    # Find new cherry-picking sequences for the trees and update the best found
    def CPSBound(self, repeats=1,
                 progress=False,
                 time_limit=None,
                 pick_triv=False,
                 pick_ml=False,
                 pick_ml_triv=False,
                 pick_random=False,
                 cherry_model_name=None,
                 leaf_model_name=None,
                 relabel=False,
                 ml_draw=False,
                 problem_type=None,
                 num_chosen_leaves=1,
                 return_seqs=False,
                 train_phase=False):
        # Set the specific heuristic that we use, based on the user input and whether the trees have lengths
        # self.full_leaf_set = full_leaf_set
        # Initialize the recorded best sequences and corresponding data
        best = None
        red_trees_best = []
        starting_time = time.time()
        self.DurationPerTrial = dict()
        self.RetPerTrial = dict()

        if return_seqs:
            self.SeqPerTrial = dict()
            self.PartSeqPerTrial = dict()
        # Try as many times as required by the integer 'repeats'
        all_leaf_probs = []
        all_cherry_probs = []
        for i in np.arange(repeats):
            start_trial = time.time()
            if self.debug_mode and (not pick_ml_triv or ml_draw) :
                print(f"Trial {i}")

            rng = np.random.RandomState(i)
            # RUN HEURISTIC
            new, reduced_trees, leaf_probs, cherry_probs = self.CPHeuristicDuo(progress=progress,
                                                                               pick_triv=pick_triv,
                                                                               pick_ml=pick_ml,
                                                                               pick_ml_triv=pick_ml_triv,
                                                                               pick_random=pick_random,
                                                                               cherry_model_name=cherry_model_name,
                                                                               leaf_model_name=leaf_model_name,
                                                                               relabel=relabel,
                                                                               ml_draw=ml_draw,
                                                                               problem_type=problem_type,
                                                                               num_chosen_leaves=num_chosen_leaves,
                                                                               rng=rng,
                                                                               train_phase=train_phase)
            all_leaf_probs.append(leaf_probs)
            all_cherry_probs.append(cherry_probs)
            if progress:
                print(f"Instance {self.instance} {problem_type}: found sequence of length: {len(new)}")

            if return_seqs:
                self.PartSeqPerTrial[i] = deepcopy(new)
            # COMPLETE PARTIAL SEQUENCE
            new, reduced_trees = sequence_add_roots(new, reduced_trees)
            if return_seqs:
                self.SeqPerTrial[i] = deepcopy(new)
            if progress:
                print(f"Instance {self.instance} {problem_type}: length after completing sequence: {len(new)}")

            self.CPS_Compute_Reps += 1
            self.DurationPerTrial[i] = time.time() - start_trial

            # FIND RETICULATION NUMBER
            seq_length = len(new)
            self.RetPerTrial[i] = seq_length - self.num_leaves + 1
            # store best sequence
            if best is None or seq_length < best_length:
                best = new
                best_length = copy(seq_length)
                red_trees_best = reduced_trees
            if progress:
                print(f"Instance {self.instance} {problem_type}: best sequence has length: {best_length}")
            if time_limit and time.time() - starting_time > time_limit:
                break

        # storing stuff of heuristic
        self.CPS_Compute_Time += time.time() - starting_time
        # storing best network
        new_seq = best
        if not self.best_seq_with_lengths or len(new_seq) < len(self.best_seq_with_lengths):
            converted_new_seq = []
            for pair in new_seq:
                converted_new_seq += [(self.labels_reversed[pair[0]], self.labels_reversed[pair[1]])]
            self.best_seq_with_lengths = converted_new_seq
            self.best_seq_with_lengths_red_trees = red_trees_best
        seq_return = [(self.labels_reversed[x], self.labels_reversed[y]) for x, y in new_seq]
        return self.best_seq_with_lengths, seq_return, all_leaf_probs, all_cherry_probs

    def CPHeuristicDuo(self, progress=False, pick_triv=True, pick_ml=False, pick_ml_triv=False,
                       cherry_model_name=None, leaf_model_name=None, pick_random=False, relabel=False,
                       ml_draw=False, problem_type=None, num_chosen_leaves=1, rng=None,
                       train_phase=False):
        # Works in a copy of the input trees, copy_of_inputs, because trees have to be reduced somewhere.
        copy_of_inputs = deepcopy(self)
        CPS = []
        reduced_trees = []
        if self.debug_mode:
            print(f"\nstart instance {self.instance}")
        if rng is None:
            rng = np.random
        # Make dict of reducible pairs
        leaf_probs = []
        cherry_probs = []
        copy_of_inputs.find_all_pairs()
        if pick_ml or pick_ml_triv:
            # create initial features
            start_time_init = time.time()
            leaf_features = LeafFeatures(copy_of_inputs, train_phase=train_phase)
            cherry_features = CherryFeatures(copy_of_inputs.trees, train_phase=train_phase)
            if progress:
                print(f"Instance {self.instance} {problem_type}:"
                      f" Initial features found in {np.round(time.time() - start_time_init, 3)}s")
            # open prediction model
            if leaf_model_name is None:
                leaf_model_name = "data/rf_models/leaf/rf_leaves_2000n.joblib"
            model_leaf = joblib.load(leaf_model_name)
            if cherry_model_name is None:
                cherry_model_name = "data/rf_models/cherry/rf_cherries_2000n.joblib"
            model_cherry = joblib.load(cherry_model_name)
        else:
            leaf_features = None
            cherry_features = None
            model_leaf = None
            model_cherry = None
        # START ALGORITHM
        while copy_of_inputs.trees:
            triv_picked = False
            chosen_cherry = None
            if progress and (pick_ml or pick_ml_triv):
                print(f"Instance {self.instance} {problem_type}: Sequence has length {len(CPS)}")
                print(f"Instance {self.instance} {problem_type}: {len(copy_of_inputs.trees)} trees left.\n")

            if pick_triv or pick_ml_triv:
                chosen_cherry, triv_picked = copy_of_inputs.pick_trivial(rng)
                if chosen_cherry is None:
                    pick_random = True
                    if pick_ml_triv:
                        pick_random = False
                        pick_ml = True
                else:
                    pick_random = False
                    if pick_ml_triv:
                        pick_ml = False
            if pick_ml:
                # predict leaves
                leaf_prediction = pd.DataFrame(model_leaf.predict_proba(leaf_features.data),
                                                   index=leaf_features.data.index)[1]
                if num_chosen_leaves == 1:
                    chosen_leaves = {leaf_prediction.index[leaf_prediction.argmax()]}
                else:
                    best_leaves = leaf_prediction.nlargest(num_chosen_leaves)
                    chosen_leaves = set(best_leaves.index[[0, 1]])
                    for l in best_leaves.index[2:]:
                        if best_leaves.loc[l] >= 0.5:
                            chosen_leaves.add(l)
                leaf_probs += leaf_prediction[list(chosen_leaves)].tolist()

                # make cherry data
                cherry_features.make_data(copy_of_inputs.reducible_pairs, chosen_leaves, copy_of_inputs.trees)

                # predict if cherry
                cherry_prediction = pd.DataFrame(model_cherry.predict_proba(cherry_features.data),
                                                     index=cherry_features.data.index)[1]
                if ml_draw:
                    # make probability density from cherry predictions
                    prob = cherry_prediction / cherry_prediction.sum()
                    cherry_num = rng.choice(len(cherry_prediction.index), p=prob)
                    chosen_cherry = list(cherry_prediction.index)[cherry_num]
                else:
                    max_cherry = cherry_prediction.argmax()
                    chosen_cherry = cherry_prediction.index[max_cherry]
                chosen_cherry_prob = cherry_prediction.loc[chosen_cherry]
                cherry_probs.append(chosen_cherry_prob)

                if not pick_ml_triv and cherry_features.data.loc[chosen_cherry]["Trivial"] == 1:
                    triv_picked = True

            if pick_random:
                random_cherry_num = rng.choice(len(copy_of_inputs.reducible_pairs))
                chosen_cherry = list(copy_of_inputs.reducible_pairs)[random_cherry_num]

            # RELABEL
            if triv_picked:
                relabel_needed, chosen_cherry = copy_of_inputs.pick_order(*chosen_cherry,
                                                                          return_relabel_needed=True)
                if relabel and relabel_needed:
                    if progress:
                        print(f"Instance {self.instance} {problem_type}: RELABEL chosen cherry = {chosen_cherry}")
                    merged_cherries, relabel_in_tree = copy_of_inputs.relabel_trivial(*chosen_cherry)
                    if self.debug_mode and relabel_in_tree:
                        print(f"relabel {chosen_cherry}")
                    if (pick_ml or pick_ml_triv) and relabel_in_tree:
                        leaf_features.relabel_trivial_cherries(*chosen_cherry, len(copy_of_inputs.trees),
                                                               relabel_in_tree)
            CPS += [chosen_cherry]

            # UPDATE SOME FEATURES BEFORE
            deleted_leaves = set()
            if pick_ml or pick_ml_triv:
                deleted_leaves = leaf_features.update_leaf_features_before(chosen_cherry,
                                                                           copy_of_inputs.reducible_pairs,
                                                                           copy_of_inputs.trees)
                cherry_features.tree_height.update_height(chosen_cherry, copy_of_inputs.reducible_pairs[chosen_cherry],
                                                          copy_of_inputs.trees)
                copy_of_inputs.update_node_comb_length(*chosen_cherry)

            # REDUCE CHOSEN CHERRY FROM FOREST
            new_reduced, deleted_trees = copy_of_inputs.reduce_pair_in_all(chosen_cherry)
            copy_of_inputs.update_reducible_pairs(new_reduced)
            reduced_trees += [new_reduced]
            if self.debug_mode:
                print(f"chosen cherry = {chosen_cherry} in {new_reduced}")

            # UPDATE FEATURES AFTER REDUCTION
            if pick_ml or pick_ml_triv:
                leaf_features.update_leaf_features_after(copy_of_inputs.trees, deleted_leaves,
                                                         deleted_trees)

        return CPS, reduced_trees, leaf_probs, cherry_probs

    @staticmethod
    def update_del_leaves(dict1, dict2):
        del_leaves = deepcopy(dict1)
        if dict1 and dict2:
            for t, leaves in dict2.items():
                if t in del_leaves:
                    del_leaves[t].update(leaves)
                else:
                    del_leaves[t] = leaves
        elif dict2:
            del_leaves = dict2
        return del_leaves

    # select order of chosen cherry
    def pick_order(self, x, y, return_cherry=False, return_relabel_needed=False):
        first_check = [(x in tree.leaves and y not in tree.leaves)
                       for t, tree in self.trees.items()
                       if t not in self.reducible_pairs[x, y]]
        extra_check = [(x in tree.leaves and y not in tree.leaves) or
                       (not {x, y}.intersection(tree.leaves)) or
                       (y in tree.leaves and x not in tree.leaves)
                       for t, tree in self.trees.items()
                       if t not in self.reducible_pairs[x, y]]

        if extra_check:
            relabel_needed = any(first_check) and all(extra_check)
        else:
            relabel_needed = False

        leaf_x_left = 0
        leaf_y_left = 0
        for t, tree in self.trees.items():
            if t in self.reducible_pairs[x, y]:
                continue
            if x in tree.leaves:
                leaf_x_left += 1
            if y in tree.leaves:
                leaf_y_left += 1
        if return_cherry:
            # FAVOR X, Y OVER Y, X
            if leaf_x_left <= leaf_y_left:
                return x, y
            else:
                return y, x
        elif return_relabel_needed:
            if leaf_x_left <= leaf_y_left:
                return relabel_needed, (x, y)
            else:
                return relabel_needed, (y, x)
        else:
            return leaf_x_left, leaf_y_left

    # when using machine learning, update the topological/combinatorial length of nodes
    def update_node_comb_length(self, x, y):
        for t in self.reducible_pairs[x, y]:
            for p in self.trees[t].nw.predecessors(y):
                p_y = p
            if self.trees[t].nw.out_degree(p_y) > 2:
                continue
            self.trees[t].nw.nodes[y]["node_comb"] -= 1

    # Finds the set of reducible pairs in all trees
    # Returns a dictionary with reducible pairs as keys, and the trees they reduce as values.
    def find_all_pairs(self):
        self.reducible_pairs = dict()
        for i, t in self.trees.items():
            red_pairs_t = t.find_all_reducible_pairs()
            for pair in red_pairs_t:
                if pair in self.reducible_pairs:
                    self.reducible_pairs[pair].add(i)
                else:
                    self.reducible_pairs[pair] = {i}

    # Returns the updated dictionary of reducible pairs in all trees after a reduction
    # (with the trees they reduce as values)
    # we only need to update for the trees that got reduced: 'new_red_treed'
    def update_reducible_pairs(self, new_red_trees):
        # Remove trees to update from all pairs
        pair_del = []
        for pair, trees in self.reducible_pairs.items():
            trees.difference_update(new_red_trees)
            if len(trees) == 0:
                pair_del.append(pair)
        for pair in pair_del:
            del self.reducible_pairs[pair]
        # Add the trees to the right pairs again
        for t in new_red_trees:
            if t in self.trees:
                tree = self.trees[t]
                red_pairs_t = tree.find_all_reducible_pairs()
                for pair in red_pairs_t:
                    if pair in self.reducible_pairs:
                        self.reducible_pairs[pair].add(t)
                    else:
                        self.reducible_pairs[pair] = {t}

    # reduces the given pair in all trees
    # Returns the set of trees that were reduced
    # CHANGES THE SET OF TREES, ONLY PERFORM IN A COPY OF THE CLASS INSTANCE
    def reduce_pair_in_all(self, pair):
        deleted_trees = []
        reduced_trees_for_pair = []
        if pair in self.reducible_pairs:
            trees_to_reduce = self.reducible_pairs[pair]
        else:
            trees_to_reduce = deepcopy(self.trees)
        for t in trees_to_reduce:
            if t in self.trees:
                tree = self.trees[t]
                if tree.reduce_pair(*pair):
                    reduced_trees_for_pair += [t]
                    if (self.trees[t].root == 0 and len(tree.nw.edges()) <= 1) or \
                            (self.trees[t].root == 2 and len(tree.nw.edges()) <= 2):
                        del self.trees[t]
                        if t in self.missing_leaves:
                            del self.missing_leaves[t]
                        deleted_trees.append(t)
        return set(reduced_trees_for_pair), deleted_trees

    # TRIVIAL CHERRY
    def pick_trivial(self, rng=None):
        if rng is None:
            rng = np.random
        trivial_cherries = []
        trivial_in_all_cherries = []
        for c, trees in self.reducible_pairs.items():
            if len(trees) == len(self.trees):
                trivial_in_all_cherries.append(c)
                continue
            trivial_check = self.trivial_check(c, trees)
            if trivial_check:
                trivial_cherries.append(c)

        if trivial_in_all_cherries:
            chosen_cherry = trivial_in_all_cherries[rng.choice(len(trivial_in_all_cherries))]
            triv_picked = False
        elif trivial_cherries:
            chosen_cherry = trivial_cherries[rng.choice(len(trivial_cherries))]
            triv_picked = True
        else:
            chosen_cherry = None
            triv_picked = False

        return chosen_cherry, triv_picked

    def trivial_check(self, c, trees):
        if len(trees) == len(self.trees):
            return False
        return len([t for t, tree in self.trees.items() if (set(c).issubset(tree.leaves) and t not in trees)]) == 0

    # RELABEL
    def relabel_trivial(self, x, y):
        relabel_in_tree = set()
        merged_cherries = set()
        new_cherries = set()
        for t, tree in self.trees.items():
            if t in self.reducible_pairs[(x, y)]:
                continue
            if x in tree.leaves:
                relabel_in_tree.add(t)
                # change leaf set
                tree.leaves.remove(x)
                tree.leaves.add(y)

                # relabel x to y
                tree.nw = nx.relabel_nodes(tree.nw, {x: y})

                # check if we have a new cherry now
                for p in tree.nw.predecessors(y):
                    for c in tree.nw.successors(p):
                        if c == y:
                            continue
                        if c not in tree.leaves:
                            continue
                        if (c, y) in self.reducible_pairs:
                            self.reducible_pairs[(c, y)].add(t)
                            self.reducible_pairs[(y, c)].add(t)
                            try:
                                del self.reducible_pairs[(c, x)], self.reducible_pairs[(x, c)]
                                merged_cherries.add((x, c))
                            except KeyError:
                                pass
                        else:
                            # add to reducible_pairs?
                            self.reducible_pairs[(c, y)] = {t}
                            self.reducible_pairs[(y, c)] = {t}
                            new_cherries.add((c, y))
                            try:
                                del self.reducible_pairs[(c, x)], self.reducible_pairs[(x, c)]
                            except KeyError:
                                pass

        return merged_cherries, relabel_in_tree

#######################################################################
########              PHYLOGENETIC TREE CLASS                  ########
#######################################################################


# A class representing a phylogenetic tree
# Contains methods to reduce trees
class PhT:
    def __init__(self, tree=None):
        if tree is None:
            # the actual graph
            self.nw = nx.DiGraph()
            # the set of leaf labels of the network
            self.leaves = set()
            self.root = 2
        else:
            self.nw = tree
            self.root = 0
            self.leaves = get_leaves(self.nw)
            self.init_leaves = deepcopy(self.leaves)

    # Builds a tree from a newick string
    def tree_from_newick(self, newick=None, current_labels=dict()):
        self.nw, self.leaves, self.init_leaves, current_labels, distances = newick_to_tree(newick, current_labels)
        return current_labels, distances

    # Checks whether the pair (x,y) forms a cherry in the tree
    def is_cherry(self, x, y):
        if (x not in self.leaves) or (y not in self.leaves):
            return False
        px = -1
        py = -1
        for p in self.nw.predecessors(x):
            px = p
        for p in self.nw.predecessors(y):
            py = p
        return px == py

    # Returns the height of (x,y) if it is a cherry:
    #     i.e.: length(p,x)+length(p,y)/2
    # Returns false otherwise
    def height_of_cherry(self, x, y):
        if (x not in self.leaves) or (y not in self.leaves):
            return False
        px = -1
        py = -1
        for p in self.nw.predecessors(x):
            px = p
        for p in self.nw.predecessors(y):
            py = p
        if px == py:
            height = [float(self.nw[px][x]['length']), float(self.nw[py][y]['length'])]
            return height
        return False

        # suppresses a degree-2 node v and returns true if successful

    # the new arc has length length(p,v)+length(v,c)
    # returns false if v is not a degree-2 node
    def clean_node(self, v):
        if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
            pv = -1
            for p in self.nw.predecessors(v):
                pv = p
            cv = -1
            for c in self.nw.successors(v):
                cv = c
            self.nw.add_edges_from([(pv, cv, self.nw[pv][v])])
            if 'length' in self.nw[pv][v] and 'length' in self.nw[v][cv]:
                self.nw[pv][cv]['length'] = self.nw[pv][v]['length'] + self.nw[v][cv]['length']
            self.nw.remove_node(v)
            return True
        return False

    # reduces the pair (x,y) in the tree if it is present as cherry
    # i.e., removes the leaf x and its incoming arc, and then cleans up its parent node.
    # note that if px, and py have different lengths, the length of px is lost in the new network.
    # returns true if successful and false otherwise
    def reduce_pair(self, x, y):
        if x not in self.leaves or y not in self.leaves:
            return False
        py = - 1
        for p in self.nw.predecessors(y):
            py = p
        if self.is_cherry(x, y):
            self.nw.remove_node(x)
            self.leaves.remove(x)

            self.clean_node(py)
            return True
        return False

    # Returns all reducible pairs in the tree involving x, where x is the first element
    def find_pairs_with_first(self, x):
        pairs = set()
        px = -1
        for p in self.nw.predecessors(x):
            px = p
        if self.nw.out_degree(px) > 1:
            for cpx in self.nw.successors(px):
                if cpx in self.leaves:
                    if cpx == x:
                        continue
                    pairs.add((x, cpx))
        return pairs - {x, x}

    # Returns all reducible pairs in the tree
    def find_all_reducible_pairs(self):
        red_pairs = set()
        for l in self.leaves:
            red_pairs = red_pairs.union(self.find_pairs_with_first(l))
        return red_pairs


#######################################################################
########              PHYLOGENETIC NETWORK CLASS               ########
#######################################################################


# A class for phylogenetic networks
class PhN:
    def __init__(self, net=None, seq=set(), newick=None, best_tree_from_network=None, reduced_trees=None, heights=None):
        # the actual graph
        self.nw = nx.DiGraph()
        # the set of leaf labels of the network
        self.leaves = set()
        # a dictionary giving the node for a given leaf label
        self.labels = dict()
        # the number of nodes in the graph
        self.no_nodes = 0
        self.leaf_nodes = dict()
        self.TCS = seq
        self.CPS = seq
        self.newick = newick
        self.reducible_pairs = set()
        self.reticulated_cherries = set()
        self.cherries = set()
        self.level = None
        self.no_embedded_trees = 0
        # if a cherry-picking sequence is given, build the network from this sequence
        if seq:
            total_len = len(seq)
            current_trees_embedded = set()
            # Creates a phylogenetic network from a cherry picking sequence:
            if reduced_trees:
                for i, pair in enumerate(reversed(seq)):
                    if heights:
                        self.add_pair(*pair, red_trees=reduced_trees[total_len - 1 - i],
                                      current_trees=current_trees_embedded, height=heights[total_len - 1 - i])
                        current_trees_embedded = current_trees_embedded | reduced_trees[total_len - 1 - i]
                    else:
                        self.add_pair(*pair, red_trees=reduced_trees[total_len - 1 - i],
                                      current_trees=current_trees_embedded)
                self.no_embedded_trees = len(current_trees_embedded)
            else:
                for pair in reversed(seq):
                    self.add_pair(*pair)
        # if a newick string is given, build the network from the newick string
        elif newick:
            self.newick = newick
            network = self.newick_to_network(newick)
            self.leaves = get_leaves(network)
            self.labels = {l: l for l in self.leaves}
            self.nw = network
            self.no_nodes = len(list(self.nw))
            self.compute_leaf_nodes()
        # if a network 'best_tree_from_network' is given, extract the best tree from this network and use this tree
        # as the network
        elif best_tree_from_network:
            self.nw.add_edges_from(best_tree_from_network.Best_Tree())
            self.labels = best_tree_from_network.labels
            self.leaf_nodes = best_tree_from_network.leaf_nodes
            self.leaves = best_tree_from_network.leaves
            self.no_nodes = best_tree_from_network.no_nodes
            # self.Clean_Up()

        elif net:
            self.nw = net
            self.leaves = get_leaves(self.nw)
            self.labels = {l: l for l in self.leaves}

    # Takes an extended newick string and returns a network
    def newick_to_network(self, newick):
        # Processing the newick string so it represents a tree, where the tips are the leaves and the reticulation nodes of the network
        newick = newick[:-1]
        newick = newick.replace("(", "[")
        newick = newick.replace(")", "]")
        newick = re.sub(r"\]\#H([\d]+)", r",#R\1]", newick)
        newick = re.sub(r"#([RH])([\d]+)", r"'#\1\2'", newick)
        # Parsing the proccessed string as a list of lists
        nestedtree = ast.literal_eval(newick)
        # Converting the list of lists to a set of edges with root node 1
        edges, leaves, label_set, current_node = nested_list_to_tree_network(nestedtree, 1)
        # Add a root edge (0,1)
        edges.append([0, 1])
        ret_labels = dict()
        leaf_labels = dict()
        for l in leaves:
            # leaves are strings, check if they are reticulation nodes
            if len(l) > 2 and (l[:2] == "#H" or l[:2] == "#R"):
                ret_labels[l[2:]] = []
            else:
                leaf_labels[l] = []
        for l in label_set:
            if len(l[0]) > 2 and (l[0][:2] == "#H" or l[0][:2] == "#R"):
                if l[0][1] == 'H':
                    ret_labels[l[0][2:]] += [l[1]]
                else:
                    ret_labels[l[0][2:]] = [l[1]] + ret_labels[l[0][2:]]
            else:
                leaf_labels[l[0]] += [l[1]]
        network = nx.DiGraph()
        network.add_edges_from(edges)
        # Merge corresponding reticulation nodes
        for retic in ret_labels:
            r = ret_labels[retic]
            receiving = r[0]
            parent_receiving = 0
            for p in network.predecessors(receiving):
                parent_receiving = p
            network.remove_node(receiving)
            for v in r[1:]:
                network.add_edge(v, parent_receiving)
                network = nx.contracted_edge(network, (v, parent_receiving))
                network.remove_edge(v, v)
                parent_receiving = v
        # Compute the leaves and their labels
        leaves = set()
        leaf_nodes = dict()
        for l in leaf_labels:
            leaf_labels[l] = leaf_labels[l][0]
            leaf_nodes[leaf_labels[l]] = l
            leaves.add(l)
        # Relabel the nodes
        for node in leaf_nodes:
            leaf_nodes[node] = "L_" + str(leaf_nodes[node])
        network = nx.relabel_nodes(network, leaf_nodes)
        # Return the network
        return network

    def is_cherry(self, x, y):
        if (x not in self.leaves) or (y not in self.leaves):
            return False
        px = -1
        py = -1
        for p in self.nw.predecessors(x):
            px = p
        for p in self.nw.predecessors(y):
            py = p
        return px == py

    # Returns true if (x_label,y_label) forms a reticulate cherry in the network, false otherwise
    def is_ret_cherry(self, x_label, y_label):
        if not x_label in self.leaves or not x_label in self.leaves:
            return False
        x = self.labels[x_label]
        y = self.labels[y_label]
        px = -1
        py = -1
        for p in self.nw.predecessors(x):
            px = p
        for p in self.nw.predecessors(y):
            py = p
        return (self.nw.in_degree(px) > 1) and self.nw.out_degree(px) == 1 and (py in self.nw.predecessors(px))

    # Returns the leaf nodes of the network
    def compute_leaf_nodes(self):
        self.leaf_nodes = dict()
        for v in self.labels:
            self.leaf_nodes[self.labels[v]] = v

    def reticulations_non_binary(self):
        return [self.nw.in_degree(v)-1 for v in self.nw.nodes() if self.nw.in_degree(v) >= 2]

    # Adds a pair to the network, using the construction from a cherry-picking sequence
    # returns false if y is not yet in the network and the network is not empty
    def add_pair(self, x, y, red_trees=set(), current_trees=set(), height=[1, 1]):
        # if the network is empty, create a cherry (x,y)
        if len(self.leaves) == 0:
            self.nw.add_edge(0, 1, no_of_trees=len(red_trees), length=0)
            self.nw.add_edge(1, 2, no_of_trees=len(red_trees), length=height[0])
            self.nw.add_edge(1, 3, no_of_trees=len(red_trees), length=height[1])
            self.leaves = {x, y}
            self.labels[x] = 2
            self.labels[y] = 3
            self.leaf_nodes[2] = x
            self.leaf_nodes[3] = y
            self.no_nodes = 4
            return True
        # if y is not in the network return false, as there is no way to add the pair and get a phylogenetic network
        if y not in self.leaves:
            return False
        # add the pair to the existing network
        node_y = self.labels[y]
        parent_node_y = -1
        for p in self.nw.predecessors(node_y):
            parent_node_y = p

        # first add all edges around y
        length_incoming_y = self.nw[parent_node_y][node_y]['length']
        no_of_trees_incoming_y = self.nw[parent_node_y][node_y]['no_of_trees']
        height_goal_x = height[0]
        if height[1] < length_incoming_y:
            height_pair_y_real = height[1]
        else:
            height_pair_y_real = length_incoming_y
            height_goal_x += height[1] - height_pair_y_real

        self.nw.add_edge(node_y, self.no_nodes, no_of_trees=no_of_trees_incoming_y + len(red_trees - current_trees),
                         length=height_pair_y_real)
        self.nw[parent_node_y][node_y]['length'] = length_incoming_y - height_pair_y_real
        self.leaf_nodes.pop(self.labels[y], False)
        self.labels[y] = self.no_nodes
        self.leaf_nodes[self.no_nodes] = y

        # Now also add edges around x
        # x is not yet in the network, so make a cherry (x,y)
        if x not in self.leaves:
            self.nw.add_edge(node_y, self.no_nodes + 1, no_of_trees=len(red_trees), length=height_goal_x)
            self.leaves.add(x)
            self.labels[x] = self.no_nodes + 1
            self.leaf_nodes[self.no_nodes + 1] = x
            self.no_nodes += 2
        # x is already in the network, so create a reticulate cherry (x,y)
        else:
            node_x = self.labels[x]
            for parent in self.nw.predecessors(node_x):
                px = parent
            length_incoming_x = self.nw[px][node_x]['length']
            no_of_trees_incoming_x = self.nw[px][node_x]['no_of_trees']
            # if x is below a reticulation, and the height of the new pair is above the height of this reticulation,
            # add the new hybrid arc to the existing reticulation
            if self.nw.in_degree(px) > 1 and length_incoming_x <= height_goal_x:
                self.nw.add_edge(node_y, px, no_of_trees=len(red_trees), length=height_goal_x - length_incoming_x)
                self.nw[px][node_x]['no_of_trees'] += len(red_trees)
                self.no_nodes += 1
            # create a new reticulation vertex above x to attach the hybrid arc to
            else:
                height_pair_x = min(height_goal_x, length_incoming_x)
                self.nw.add_edge(node_y, node_x, no_of_trees=len(red_trees), length=height_goal_x - height_pair_x)
                self.nw.add_edge(node_x, self.no_nodes + 1, no_of_trees=no_of_trees_incoming_x + len(red_trees),
                                 length=height_pair_x)
                self.nw[px][node_x]['length'] = length_incoming_x - height_pair_x
                self.leaf_nodes.pop(self.labels[x], False)
                self.labels[x] = self.no_nodes + 1
                self.leaf_nodes[self.no_nodes + 1] = x
                self.no_nodes += 2
        return True

    # suppresses v if it is a degree-2 node
    def Clean_Node(self, v):
        if self.nw.out_degree(v) == 1 and self.nw.in_degree(v) == 1:
            pv = -1
            for p in self.nw.predecessors(v):
                pv = p
            cv = -1
            for c in self.nw.successors(v):
                cv = c
            self.nw.add_edges_from([(pv, cv, self.nw[v][cv])])
            if self.nw[pv][v]['length'] and self.nw[v][cv]['length']:
                self.nw[pv][cv]['length'] = self.nw[pv][v]['length'] + self.nw[v][cv]['length']
            self.nw.remove_node(v)
            return True
        return False

    # reduces the pair (x_label,y_label) if it is reducible in the network
    # returns a new set reducible pairs that involve the leaves x_label and y_label
    def reduce_pair(self, x_label, y_label):
        if x_label not in self.leaves or not y_label in self.leaves:
            return set()
        x = self.labels[x_label]
        y = self.labels[y_label]
        px = -1
        py = -1
        for p in self.nw.predecessors(x):
            px = p
        for p in self.nw.predecessors(y):
            py = p
        if self.is_cherry(x_label, y_label):
            self.reducible_pairs.difference_update({(x_label, y_label), (y_label, x_label)})
            self.nw.remove_node(x)
            self.leaves.remove(x_label)
            self.labels.pop(x_label, False)
            self.Clean_Node(py)
            # AddCherriesInvolving y
            new_pairs = {("no_leaf", "no_leaf")} | self.Find_Pairs_With_First(y_label) | self.Find_Pairs_With_Second(
                y_label)
            self.reducible_pairs = self.reducible_pairs.union(new_pairs - {("no_leaf", "no_leaf")})
            return new_pairs
        if self.is_ret_cherry(x_label, y_label):
            self.reducible_pairs.difference_update({(x_label, y_label), (y_label, x_label)})
            self.nw.remove_edge(py, px)
            self.Clean_Node(px)
            self.Clean_Node(py)
            # AddCherriesInvolving x and y
            new_pairs = {("no_leaf", "no_leaf")} | self.Find_Pairs_With_First(x_label) | self.Find_Pairs_With_Second(
                x_label) | self.Find_Pairs_With_First(y_label) | self.Find_Pairs_With_Second(y_label)
            self.reducible_pairs = self.reducible_pairs.union(new_pairs - {("no_leaf", "no_leaf")})
            return new_pairs
        return set()

    # Returns all reducible pairs in the network where x_label is the first element of the pair
    def Find_Pairs_With_First(self, x_label):
        pairs = set()
        x = self.labels[x_label]
        px = -1
        for p in self.nw.predecessors(x):
            px = p
        if self.nw.in_degree(px) > 1:
            for ppx in self.nw.predecessors(px):
                for cppx in self.nw.successors(ppx):
                    if cppx in self.leaf_nodes:
                        pairs.add((x_label, self.leaf_nodes[cppx]))
        if self.nw.out_degree(px) > 1:
            for cpx in self.nw.successors(px):
                if cpx in self.leaf_nodes:
                    pairs.add((x_label, self.leaf_nodes[cpx]))
        return pairs - {(x_label, x_label)}

    # Returns all reducible pairs in the network where x_label is the second element of the pair
    def Find_Pairs_With_Second(self, x_label):
        pairs = set()
        x = self.labels[x_label]
        px = -1
        for p in self.nw.predecessors(x):
            px = p
        if self.nw.out_degree(px) > 1:
            for cpx in self.nw.successors(px):
                if cpx in self.leaf_nodes:
                    pairs.add((self.leaf_nodes[cpx], x_label))
                if self.nw.in_degree(cpx) > 1:
                    for ccpx in self.nw.successors(cpx):
                        if ccpx in self.leaf_nodes:
                            pairs.add((self.leaf_nodes[ccpx], x_label))
        return pairs - {(x_label, x_label)}
