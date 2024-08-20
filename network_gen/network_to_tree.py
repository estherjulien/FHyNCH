from heuristic.CPH import *

import networkx as nx
import numpy as np
import itertools
import copy


# return reticulation nodes
def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


# for non-binary give ret number per reticulation node
def reticulations_non_binary(G):
    return [G.in_degree(i)-1 for i in G.nodes if G.in_degree(i) >= 2]


# return leaves from network
def leaves(net):
    return {u for u in net.nodes() if net.out_degree(u) == 0}


# MAKE TREES FROM NETWORK
def net_to_tree(net, rng=None, num_trees=None, distances=True, net_lvs=0, mis_l=0, add_atr_now=True):
    # we only consider binary networks here
    tree_set = dict()
    rets = reticulations(net)
    ret_num = len(rets)
    tree_lvs = []
    if ret_num == 0:
        return False, None, None

    if num_trees is None:
        ret_dels_tmp = itertools.product(*[np.arange(2)] * ret_num)
        ret_dels = None
        for opt in ret_dels_tmp:
            opt = np.array(opt).reshape([1, -1])
            try:
                ret_dels = np.vstack([ret_dels, opt])
            except:
                ret_dels = opt
    else:
        ret_dels_set = set()
        # probability per reticulation
        p_ret = rng.uniform(0.1, 0.9, ret_num)
        while len(ret_dels_set) < num_trees:
            set_tree_ret = []
            for r in range(ret_num):
                set_tree_ret.append(rng.choice([0, 1], p=[p_ret[r], 1 - p_ret[r]]))
            ret_dels_set.add(tuple(set_tree_ret))
        ret_dels = np.array([list(opt) for opt in ret_dels_set])

    t = 0
    for opt in ret_dels:
        if opt[0] is None:
            continue
        tree = copy.deepcopy(net)
        tree_lvs.append(net_lvs)
        for i in np.arange(ret_num):
            # STEP 1: DELETE ONE OF THE RETICULATION EDGES
            if opt[i] is None:
                continue
            ret = rets[i]
            # check if reticulation still has indegree 2!
            if ret not in tree.nodes:
                continue
            if tree.in_degree(ret) < 2:
                continue
            ret_pre_both = list(tree.pred[ret]._atlas.keys())
            ret_pre_del = ret_pre_both[opt[i]]
            # delete reticulation edge
            tree.remove_edge(ret_pre_del, ret)

            # STEP 2: SUBTRACTING
            in_out_one_nodes = [n for n in tree.nodes if (tree.in_degree(n) == 1 and tree.out_degree(n) == 1)]
            for n in in_out_one_nodes:
                tree = subtract_node(tree, n)

            # STEP 3: DELETE "NEW LEAVES"
            while True:
                wrong_new_leaves = leaves(tree).difference(leaves(net))
                if not wrong_new_leaves:
                    break
                for n in wrong_new_leaves:
                    tree.remove_node(n)

            # STEP 4: REROOTING
            if tree.out_degree(0) == 1:
                for c in tree.successors(0):
                    child = c
                tree.remove_node(0)
                tree = nx.relabel_nodes(tree, {child: 0})

        # STEP 5: CHECK IF THERE ARE ANY IN-DEGREE 1 OUT-DEGREE 1 NODES
        tree_nodes = copy.deepcopy(tree.nodes)
        for n in tree_nodes:
            if n not in tree.nodes:
                continue
            if tree.in_degree(n) == 1 and tree.out_degree(n) == 1:
                # subtract
                tree = subtract_node(tree, n)

        # STEP 6: MISSING LEAVES
        if mis_l > 0:
            # have missing leaves percentage per tree
            max_ms_perc = float(mis_l)/100
            ms_perc = rng.uniform(0, max_ms_perc)
            lvs = leaves(tree)
            # delete nodes with outdegree zero and not part of leave
            k = 0
            for l in lvs:
                if tree_lvs[-1] <= 2 or rng.rand() > ms_perc:
                    continue
                k += 1
                # regraft parent
                tree_lvs[-1] -= 1
                for p in tree.predecessors(l):
                    pl = p
                if pl == 0:
                    # PARENT IS ROOT, REROOTING
                    tree.remove_node(l)
                    for c in tree.successors(0):
                        child = c
                    tree.remove_node(0)
                    tree = nx.relabel_nodes(tree, {child: 0})
                    continue

                for g in tree.predecessors(pl):
                    gl = g
                for s in tree.successors(pl):
                    if s == l:
                        continue
                    sl = s

                # add new edge
                try:
                    len_1 = tree.edges[(gl, pl)]["length"]
                except KeyError:
                    len_1 = tree.edges[(gl, pl)]["lenght"]

                try:
                    len_2 = tree.edges[(pl, sl)]["length"]
                except KeyError:
                    len_2 = tree.edges[(pl, sl)]["lenght"]

                tree.add_edge(gl, sl, length=len_1 + len_2)
                # remove parent and leaf node
                tree.remove_node(l)
                tree.remove_node(pl)

        if add_atr_now:
            add_node_attributes(tree, distances=distances, root=0)
        if max([tree.in_degree(l) for l in tree.nodes]) == 1:
            tree_set[t] = tree
            t += 1
        else:
            print("Bad tree after missing leaves")

    if mis_l > 0:
        unique_leaves = leaves(tree_set[0])
        for t, tree in tree_set.items():
            if t == 0:
                continue
            unique_leaves = leaves(tree).union(unique_leaves)
        num_union_leaves = len(unique_leaves)
    else:
        num_union_leaves = len(leaves(tree_set[0]))

    return tree_set, tree_lvs, num_union_leaves


def net_to_tree_non_binary(net, rng=None, num_trees=None, distances=True, net_lvs=0, mis_l=0, mis_edge=0):
    if mis_edge:
        add_atr_now = False
    else:
        add_atr_now = True
    tree_set, tree_lvs, num_unique_leaves = net_to_tree(net, rng, num_trees, distances, net_lvs, mis_l,
                                                        add_atr_now=add_atr_now)

    if mis_edge > 0:
        # delete edges
        for t, tree in tree_set.items():
            max_ms_perc = float(mis_edge)/100
            ms_perc = rng.uniform(0, max_ms_perc)
            del_edges = set()
            # choose edges to delete
            for x, y in tree.edges:
                if tree.out_degree(y) == 0:
                    continue
                if rng.rand() > ms_perc:
                    continue
                # check all successors of
                del_edges.add((x, y))

            # delete edges
            changed_node = dict()
            for x_pre, y in del_edges:
                while x_pre in changed_node:
                    x_pre = changed_node[x_pre]
                x = x_pre
                # add edges
                len_1 = tree.edges[(x, y)]["length"]
                for c in tree.successors(y):
                    # add new edge
                    len_2 = tree.edges[(y, c)]["length"]
                    tree.add_edge(x, c, length=len_1 + len_2)
                    # check for changed leaves
                    changed_node[y] = x
                # delete node
                tree.remove_node(y)

            # add tree node attributes
            add_node_attributes(tree, distances=distances, root=0)

    return tree_set, tree_lvs, num_unique_leaves


def subtract_node(tree, n):
    for p in tree.predecessors(n):
        pn = p
    for c in tree.successors(n):
        cn = c
    # predecessor length
    try:
        pre_len = tree.edges[(pn, n)]["length"]
    except KeyError:
        pre_len = tree.edges[(pn, n)]["lenght"]
    # successor length
    try:
        succ_len = tree.edges[(n, cn)]["length"]
    except KeyError:
        succ_len = tree.edges[(n, cn)]["lenght"]
    # remove node
    tree.remove_node(n)
    # add edge
    tree.add_edge(pn, cn, length=pre_len + succ_len)
    return tree


def network_cherries(net):
    cherries = set()
    retic_cherries = set()
    lvs = leaves(net)

    for l in lvs:
        for p in net.pred[l]:
            if net.out_degree(p) > 1:
                for cp in net.succ[p]:
                    if cp == l:
                        continue
                    if cp in lvs:
                        cherries.add((l, cp))
                        cherries.add((cp, l))
                    elif net.in_degree(cp) > 1:
                        for ccp in net.succ[cp]:
                            if ccp in lvs:
                                retic_cherries.add((ccp, l))

    return cherries, retic_cherries


def tree_cherries(tree_set):
    cherries = set()
    reducible_pairs = dict()
    t = 0
    for tree in tree_set.values():
        lvs = leaves(tree)

        for l in lvs:
            for p in tree.pred[l]:
                if tree.out_degree(p) > 1:
                    for cp in tree.succ[p]:
                        if cp == l:
                            continue
                        if cp in lvs:
                            cherry = (l, cp)
                            cherries.add(cherry)
                            cherries.add(cherry[::-1])

                            # add tree to cherry
                            if cherry not in reducible_pairs:
                                reducible_pairs[cherry] = {t}
                                reducible_pairs[cherry[::-1]] = {t}
                            else:
                                reducible_pairs[cherry].add(t)
                                reducible_pairs[cherry[::-1]].add(t)
        t += 1
    return cherries, reducible_pairs


# check if cherry is reducible
def is_cherry(tree, x, y):
    lvs = leaves(tree)
    if (x not in lvs) or (y not in lvs):
        return False
    # tree, so no reticulations
    px = tree.pred[x]._atlas.keys()
    py = tree.pred[y]._atlas.keys()
    return px == py


def is_ret_cherry(net, x, y):
    for p in net.pred[y]:
        if net.out_degree(p) > 1:
            for cp in net.succ[p]:
                if cp == y:
                    continue
                if net.in_degree(cp) > 1:
                    for ccp in net.succ[cp]:
                        if ccp == x:
                            return True
    return False


