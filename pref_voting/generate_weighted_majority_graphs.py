'''
    File: generate_margin_graphs.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: July 14, 2022
    
    Functions to generate a margin graph
    
'''


import networkx as nx
from itertools import combinations
from pref_voting.weighted_majority_graphs import MarginGraph
import random

def generate_edge_ordered_tournament(num_cands): 
    """Generate a random uniquely weighted MarginGraph for ``num_cands`` candidates.  

    :param num_cands: the number of candidates
    :type num_cands: int
    :returns: a uniquely weighted margin graph
    :rtype: MarginGraph

    .. note:: This function randomly generates a tournament with a linear order over the edges.  A **tournament** is an asymmetric directed graph with an edge between every two nodes.  The linear order of the edges is represented by assigning to each edge a number  :math:`2, \ldots, 2*n`, where :math:`n` is the number of the edges. 
    """
    mg = nx.DiGraph()
    mg.add_nodes_from(list(range(num_cands)))
    candidates = list(range(num_cands))
    _edges = list()
    for c1 in candidates: 
        for c2 in candidates: 
            if c1 != c2: 
                if (c1, c2) not in _edges and (c2, c1) not in _edges:
                    if random.choice([True, False]): 
                        _edges.append((c1, c2))
                    else: 
                        _edges.append((c2, c1))
                   
    edges = list()
    edge_indices = list(range(len(_edges)))
    random.shuffle(edge_indices)
    
    for i, e_idx in enumerate(edge_indices):
        edges.append((_edges[e_idx][0], _edges[e_idx][1], 2 * (i+1))) 
    
    return MarginGraph(candidates, edges)

def generate_edge_ordered_weighted_majority_graph(num_cands): 
    """Generate a random  weighted MarginGraph (allowing for ties in the margins) for ``num_cands`` candidates.  

    :param num_cands: the number of candidates
    :type num_cands: int
    :returns: a uniquely weighted margin graph
    :rtype: MarginGraph

    """

    candidates = list(range(num_cands))
    edges = list()
    pairs_of_cands = list(combinations(candidates, 2))

    for c1, c2 in pairs_of_cands:

        margin = random.choice([2 * pidx for pidx in range(len(pairs_of_cands) + 1)])
        
        if margin != 0: 
            if random.choice([True, False]): 
                edges.append((c1, c2, margin))
            else:
                edges.append((c2, c1, margin))

    return MarginGraph(candidates, edges)


 