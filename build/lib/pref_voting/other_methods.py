'''
    File: mg_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 12, 2022
    
    Implementations of 
'''

from pref_voting.voting_method import  *
from pref_voting.profiles import _find_updated_profile
from pref_voting.helper import get_mg
from itertools import combinations, permutations, chain
import networkx as nx

## Banks
#

def seqs(iterable):
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(len(s)+1))

def is_transitive(G, p):
    for c1_idx, c1 in enumerate(p[:-1]):
        for c2 in p[c1_idx+1::]:            
            if not G.has_edge(c1,c2):
                return False
    return True

def is_subsequence(x, y):
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)

@vm(
    name = "Banks",
    description = """The Banks set is the maximla elements of all maximal transitive subrelations of the Majority Graph.
    """)
def banks(edata, curr_cands = None): 
    
    mg = get_mg(edata, curr_cands = curr_cands)
    trans_paths = list()
    for s in seqs(mg.nodes):
        if nx.algorithms.simple_paths.is_simple_path(mg, s):
            if is_transitive(mg, s): 
                trans_paths.append(s)

    maximal_paths = list()
    #print("max paths")
    for s in trans_paths:
        is_max = True
        for other_s in trans_paths: 
            if s != other_s:
                if is_subsequence(s, other_s): 
                    is_max = False
                    break
        if is_max:
            maximal_paths.append(s)
    
    return sorted(list(set([p[0] for p in maximal_paths])))


## Slater
#

def distance_to_margin_graph(edata, rel, exp = 1, curr_cands = None): 
    
    candidates = edata.candidates if curr_cands is None else curr_cands
    
    penalty = 0
    for a,b in combinations(candidates, 2): 
        if edata.majority_prefers(a, b) and (b,a) in rel: 
            penalty += (edata.margin(a, b) ** exp)
        elif edata.majority_prefers(b, a) and (a,b) in rel: 
            penalty += (edata.margin(b, a) ** exp)
        elif edata.majority_prefers(a, b) and (a,b) not in rel and (b,a) not in rel: 
            penalty += (edata.margin(a, b) ** exp) / 2 
        elif edata.majority_prefers(b, a) and (a,b) not in rel and (b,a) not in rel: 
            penalty += (edata.margin(b, a) ** exp)  / 2
    return penalty


def lin_order_to_rel(lin_order): 
    """convert a linear order (a list of items) into a set of ordered pairs"""
    els = sorted(lin_order)
    rel = []
    for a,b in combinations(els, 2):
        if lin_order.index(a) < lin_order.index(b): 
            rel.append((a,b))
        elif lin_order.index(b) < lin_order.index(a): 
            rel.append((b,a))     
    return rel


def slater_rankings(edata, curr_cands = None): 
    
    candidates = edata.candidates if curr_cands is None else curr_cands
    min_dist = np.inf
    
    rankings = list()
    for lin_order in permutations(candidates): 
        #print(lin_order)
        lo_rel = lin_order_to_rel(lin_order)
        
        dist = distance_to_margin_graph(edata, lo_rel, exp = 0, curr_cands = curr_cands)
        if dist < min_dist: 
            min_dist = dist
            rankings = [lin_order]
        elif dist == min_dist: 
            rankings.append(lin_order)
    return rankings, min_dist

        
@vm(
    name = "Slater",
    description = """A Slater ranking is a linear order $R$ of the candidates thatminimises the number of edges in the majority graph we have to turn
around before we obtain $R$.   A candidate is a winner if the candidate is the top element of some Slater ranking. 
""")
def slater(edata, curr_cands = None): 
    
    rankings, dist = slater_rankings(edata, curr_cands = curr_cands)
    
    return sorted(list(set([r[0] for r in rankings])))


## Kemmeny-Young Method 
#
def kendalltau_dist(rank_a, rank_b):
    rank_a = tuple(rank_a)
    rank_b = tuple(rank_b)
    tau = 0
    candidates = sorted(rank_a)
    for i, j in combinations(candidates, 2):
        tau += (np.sign(rank_a.index(i) - rank_a.index(j)) ==
                -np.sign(rank_b.index(i) - rank_b.index(j)))
    return tau


def kemmeny_young_rankings(rankings, rcounts, candidates): 
    rankings_dist = dict()
    for ranking in permutations(candidates): 
        rankings_dist[tuple(ranking)] = sum(c * kendalltau_dist(tuple(r), ranking) 
                                            for r,c in zip(rankings, rcounts))
    min_dist = min(rankings_dist.values())

    lin_orders = [r for r in rankings_dist.keys() if rankings_dist[r] == min_dist]
    
    return lin_orders, min_dist

@vm(
    name = "Kemmeny-Young",
    description = """A Kemmeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters' rankings.  The Kemmeny-Young winners are the candidates that are ranked first by some Kemmeny-Young ranking.
    """)
def kemmeny_young(profile, curr_cands = None): 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]),  profile.num_cands)
    ky_rankings, min_dist = kemmeny_young_rankings(list(rankings), list(profile._rcounts), candidates)
    
    return sorted(list(set([r[0] for r in ky_rankings])))

other_vms = [
    banks,
    slater,
    kemmeny_young
]