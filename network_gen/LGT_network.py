import networkx as nx
import numpy as np


'''
Code for generating LGT network
- Apdated from Remie Janssen's code for the paper:
Janssen, R., & Liu, P. (2021). Comparing the topology of phylogenetic network generators. 
Journal of Bioinformatics and Computational Biology, 19(06), 2140012.

- Originally, the method is from the paper:
Joan Carles Pons, Celine Scornavacca, and Gabriel Cardona. Generation of level-k LGT
networks. IEEE/ACM transactions on computational biology and bioinformatics, 17(1):158–164,
2019.
'''


# get last node from network
def last_node(net):
    return max(net.nodes())


# speciation event
def speciate(net, leaf):
    l = last_node(net)
    net.add_edge(leaf, l+1, length=np.random.random())
    net.add_edge(leaf, l+2, length=np.random.random())


# lateral gene transfer event
def lgt(net, leaf1, leaf2):
    # change length of either first or second leaf, depending on which one is higher.
    leaf1_path_tmp = nx.edge_dfs(net, leaf1, orientation='reverse')
    leaf1_path = {(p1, p2) for p1, p2, _ in leaf1_path_tmp}

    len_leaf1 = 0
    for edge in leaf1_path:
        try:
            len_leaf1 += net.edges[edge]["length"]
        except KeyError:
            len_leaf1 += net.edges[edge]["lenght"]

    leaf2_path_tmp = nx.edge_dfs(net, leaf2, orientation='reverse')
    leaf2_path = {(p1, p2) for p1, p2, _ in leaf2_path_tmp}

    len_leaf2 = 0
    for edge in leaf2_path:
        try:
            len_leaf2 += net.edges[edge]["length"]
        except KeyError:
            len_leaf2 += net.edges[edge]["lenght"]

    change_leaf_ind = np.argmin([len_leaf1, len_leaf2])
    change_leaf = [leaf1, leaf2][change_leaf_ind]
    change_leaf_length = [len_leaf1, len_leaf2][change_leaf_ind]
    other_leaf_length = [len_leaf1, len_leaf2][1 - change_leaf_ind]

    for p in net.predecessors(change_leaf):
        leaf_parent = p

    try:
        old_path_length = net.edges[(leaf_parent, change_leaf)]["length"]
    except KeyError:
        old_path_length = net.edges[(leaf_parent, change_leaf)]["lenght"]

    new_path_length = other_leaf_length - (change_leaf_length - old_path_length)
    net.remove_edge(leaf_parent, change_leaf)
    net.add_edge(leaf_parent, change_leaf, length=new_path_length)

    # add new edge
    net.add_edge(leaf1, leaf2, secondary=True, length=0)
    l = last_node(net)
    net.add_edge(leaf1, l+1, length=np.random.random())
    net.add_edge(leaf2, l+2, length=np.random.random())


# return leaves from network
def leaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]


# return the non trivial blobs of the network
def non_trivial_blobs(net):
    blobs = list(nx.biconnected_components(nx.Graph(net)))
    return [bl for bl in blobs if len(bl) > 2]


def internal_blobs(net):
    internal_nodes = set([u for u in net.nodes() if net.out_degree(u)>0])
    blobs = list(nx.biconnected_components(nx.Graph(net)))
    blobs = [bl for bl in blobs if len(bl) > 2]
    nodes_in_blobs = set().union(*blobs)
    nodes_not_in_blobs = internal_nodes - nodes_in_blobs
    blobs.extend([{u} for u in nodes_not_in_blobs])
    return blobs


def compute_hash(net):
    mapping_blobs = {}
    blobs = internal_blobs(net)
    for blob in blobs:
        for node in blob:
            mapping_blobs[node] = blob

    mapping = {}
    for l in leaves(net):
        # parent = net.predecessors(l)
        parent = list(net.pred[l])[0]
        mapping[l] = mapping_blobs[parent]
    return mapping


# return internal and external pair of leaves
def internal_and_external_pairs_bugged(net):
    lvs = leaves(net)
    pairs = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    mapping = compute_hash(net)
    internal_pairs = []
    external_pairs = []
    for pair in pairs:
        # check if pair has same parent node
        if net.in_degree(pair[0]) == 1 and net.in_degree(pair[1]) == 1:
            # same parent not allowed
            p_both = []
            for l in pair:
                for p in net.predecessors(l):
                    p_both.append(p)
            if p_both[0] == p_both[1]:
                continue
            # if grandparent of x is parent of y            ## TODO: MAYBE NOT??
            try:
                for gp in net.predecessors(p_both[0]):
                    gp_x = gp
                if gp_x == p_both[1]:
                    continue
            except NameError:
                pass
            if is_ret_cherry(net, *pair[::-1]) or is_ret_cherry(net, *pair):
                continue
        else:
            pass
        if mapping[pair[0]] == mapping[pair[1]]:
            internal_pairs.append(pair)
        else:
            external_pairs.append(pair)
    return internal_pairs, external_pairs


def internal_and_external_pairs(net):
    lvs = leaves(net)
    pairs = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    mapping = compute_hash(net)
    internal_pairs = []
    external_pairs = []
    for pair in pairs:
        # check if pair has same parent node
        if net.in_degree(pair[0]) == 1 and net.in_degree(pair[1]) == 1:
            # same parent not allowed
            p_both = []
            for l in pair:
                for p in net.predecessors(l):
                    p_both.append(p)
            if p_both[0] == p_both[1]:
                continue
            if is_ret_cherry(net, *pair):
                continue
        else:
            pass
        if mapping[pair[0]] == mapping[pair[1]]:
            internal_pairs.append(pair)
        else:
            external_pairs.append(pair)
    return internal_pairs, external_pairs


# return a random pair of leaves from the network
def random_pair(net, wint, wext):
    int_pairs, ext_pairs = internal_and_external_pairs(net)
    prob = np.array([wint]*len(int_pairs)+[wext]*len(ext_pairs)) / (len(int_pairs) + len(ext_pairs))
    id = np.random.choice(np.arange(len(int_pairs) + len(ext_pairs)), p=prob)
    return (int_pairs+ext_pairs)[id]


# SIMULATION
def simulation(num_steps, prob_lgt, wint, wext):
    # probl_lgt: alpha
    # wint: weight internal, i.e. 1
    # wext: weight external, i.e. beta

    # initialize network
    net = nx.DiGraph()
    net.add_edge(0, 1, length=np.random.random())
    net.add_edge(0, 2, length=np.random.random())
    num_ret = 0
    for i in range(num_steps):
        if len(net.nodes) == 3:
            event = "spec"
        else:
            event = np.random.choice(['spec', 'lgt'], p=[1-prob_lgt, prob_lgt])
        if event == 'spec':
            l = np.random.choice(leaves(net))
            speciate(net, l)
        else:
            num_ret += 1
            pair = random_pair(net, wint, wext)
            # change length of highest hanging leaf to lowest hanging leaf. Only then we can have an LGT event.
            lgt(net, pair[0], pair[1])
    return net, num_ret


# return reticulation nodes
def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) > 1]


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


# find level per blob
def local_level(G, bicc):
    rets = list(set(reticulations(G)).intersection(bicc))
    if len(rets) == 0:
        return 0
    else:
        bicc_edges = [e for e in G.edges() if ((e[0] in bicc) & (e[1] in bicc))]
        end_nodes = [e[1] for e in bicc_edges]
        return len([ret for ret in rets if ret in end_nodes])




