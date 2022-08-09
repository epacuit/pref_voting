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
import numpy as np
from scipy.stats import multivariate_normal

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

    Args:
        num_cands (int): the number of candidates
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns:
        A uniquely weighted margin graph (MarginGraph)

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


### 

# Turn a code into a pair
def depair(pair_vector, k):
    return pair_vector[k]


# This function defines the i,jth entry of the covariance matrix
def entries(pair_vector, i,j):
    x = depair(pair_vector, i)
    y = depair(pair_vector, j)
    if x[0] == y[0] and x[1] == y[1]:
        return 1
    if x[1] == y[0]:
        return -1/3
    if x[1] == y[1]:
        return 1/3
    if x[0] == y[0]:
        return 1/3
    if x[0] == y[1]:
        return -1/3
    return 0

def generate_covariance_matrix(num_candidates):
    
    num_pairs = num_candidates *(num_candidates -1)//2

    # Store the vector mapping codes to pairs
    pair_vector = [0]*num_pairs

    # Populate the vector of pairs
    k=0
    for i in range(num_candidates):
        for j in range(i+1,num_candidates):
            pair_vector[k] = [i,j]
            k = k+1

    # Populate the covariance matrix
    cov = np.empty((num_pairs,num_pairs))
    for i in range(num_pairs):
        for j in range(num_pairs):
            cov[i,j] = entries(pair_vector, i,j)
            
    return cov


def generate_edge_ordered_tournament_infinite_limit(num_candidates): 
    """
    Using the ideas from Section 9 of the paper 
    *An Analysis of Random Elections with Large Numbers of Voters∗ by Matthew Harrison-Trainor 
    (https://arxiv.org/abs/2009.02979) and the code provided at  
    https://github.com/MatthewHT/RandomMarginGraphs/, generate a qualitative margin graph for 
    ``num_candidates`` candidates.
    
    .. important:: 
        
        The weights of the generated margin graphs are real numbers, representing a linear ordering of the edges. 
        Only qualitative margin graph invariant voting methods, such as Split Cycle, Beat Path, Minimax, 
        Ranked Pairs, etc., should be used on the generated graphs. 
        
    Args:
        
        num_candidates (int): the number of candidates
        
    Returns: 
    
        MarginGraph
    
    """
    
    candidates = range(num_candidates)
    cov_matrix = generate_covariance_matrix(num_candidates)
    # random_var is a random variable with the multivariate normal distribution of margin graphs
    random_var = multivariate_normal(None, cov_matrix)
    rv = random_var.rvs()
    
    def pair(p):
        return p[1]-2*p[0]-1 + (num_candidates)*(num_candidates+1)//2 - (num_candidates-p[0])*(num_candidates-p[0]+1)//2

    mg = [[-np.inf for _ in candidates] for _ in candidates]
    
    for c1 in candidates:
        for c2 in candidates:
            if (c1 < c2 and rv[pair([c1,c2])] > 0):
                mg[c1][c2] = rv[pair([c1,c2])]
            if (c1 > c2 and rv[pair([c2,c1])] < 0):
                mg[c1][c2] = -rv[pair([c2,c1])]
            if (c1 == c2):
                mg[c1][c2] = 0

    w_edges = [(c1, c2, mg[c1][c2]) 
               for c1 in candidates 
               for c2 in candidates if c1 != c2 if mg[c1][c2] > 0]
    
    return MarginGraph(candidates, w_edges)



 