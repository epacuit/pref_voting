'''
    File: mg_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 10, 2022
    
    Implementations of voting methods that work on both profiles and margin graphs
'''

from pref_voting.voting_method import  *
from pref_voting.helper import get_mg
import copy
import math
from itertools import product, permutations
import networkx as nx
import matplotlib.pyplot as plt

@vm(
    name = "Condorcet",
    description ="""Return the Condorcet winner if one exists, otherwise return all the candidates.  
    """)
def condorcet(edata, curr_cands = None):
    
    candidates = edata.candidates if curr_cands is None else curr_cands
    cond_winner = edata.condorcet_winner(curr_cands = curr_cands)
    
    return [cond_winner] if cond_winner is not None else sorted(candidates)


@vm(
    name = "Copeland",
    description ="""The Copeland score for c is the number of candidates that c is majority preferred to minus the number of candidates majority preferred to c.   The Copeland winners are the candidates with the maximum Copeland score.  
    """)
def copeland(edata, curr_cands = None):
    
    c_scores = edata.copeland_scores(curr_cands = curr_cands)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])


@vm(
    name = "Llull",
    description ="""The Llull score for a candidate c is the number of candidates that c is weakly majority preferred to.  The Llull winners are the candidates with the greatest Llull score.
    """)
def llull(edata, curr_cands = None):
    
    l_scores = edata.copeland_scores(curr_cands = curr_cands, scores = (1,1,0))
    max_score = max(l_scores.values())
    return sorted([c for c in l_scores.keys() if l_scores[c] == max_score])

def left_covers(dom, c1, c2):
    # left covers: c1 left covers c2 when all the candidates that are majority preferred to c1
    # are also majority preferred to c2.
    
    # weakly left covers: c1 weakly left covers c2 when all the candidates that are majority preferred to or tied with c1
    # are also majority preferred to or tied with c2.
    
    return dom[c1].issubset(dom[c2])

def right_covers(dom, c1, c2):
    # right covers: c1 right covers c2 when all the candidates that c2  majority preferrs are majority
    # preferred by c1
      
    return dom[c2].issubset(dom[c1])


@vm(
    name = "Uncovered Set",
    description = """(Gillies version) Given candidates a and b, say that a defeats b in the profile P, a defeats b 
    if a is majority preferred to b and a left covers b: i.e., for all c, if c is majority preferred to a, 
    then c majority preferred to b. Then the winners are the set of  candidates who are undefeated in P. """)
def uc_gill(edata, curr_cands = None): 

    candidates = edata.candidates if curr_cands is None else curr_cands
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))


@vm(
    name = "Uncovered Set - Fishburn",
    description = """(Fishburn version)  Given candidates a and b, say that a defeats b if a left covers b: i.e., for all c, if c is majority preferred to a, then c majority preferred to b, and
    b does not left cover a. Then the winners are the set of candidates who are undefeated.""")
def uc_fish(edata, curr_cands = None): 

    candidates = edata.candidates if curr_cands is None else curr_cands
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in candidates:
            if c1 != c2:
                # check if c2 left covers  c1 but c1 does not left cover c2
                if left_covers(dom, c2, c1)  and not left_covers(dom, c1, c2):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))


@vm(
    name = "Uncovered Set - Bordes",
    description = """(Bordes version)  Given candidates a and b, say that a Bordes covers b if a is majority preferred to b and for all c, if c is majority preferred or tied with a, then c is majority preferred to tied with b. Returns the candidates that are not Bordes covered. """)
def uc_bordes(edata, curr_cands = None): 

    candidates = edata.candidates if curr_cands is None else curr_cands

    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)).union([_c for _c in candidates if (not edata.majority_prefers(c, _c) and not edata.majority_prefers(_c, c))]) for c in candidates}
    
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))  

@vm(
    name = "Uncovered Set - McKelvey",
    description = """(McKelvey version)  Given candidates a and b, say that a McKelvey covers b if a Gillies covers b and a Bordes covers b. Returns the candidates that are not McKelvey covered.
    """)
def uc_mckelvey(edata, curr_cands = None): 

    candidates = edata.candidates if curr_cands is None else curr_cands

    strict_dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}    
    weak_dom = {c: strict_dom[c].union([_c for _c in candidates if (not edata.majority_prefers(c, _c) and not edata.majority_prefers(_c, c))]) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(strict_dom, c2, c1) and left_covers(weak_dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))      

@vm(
    name = "GETCHA",
    description = """The smallest set of candidates such that every candidate inside the set 
    is majority preferred to every candidate outside the set.  Also known as GETCHA or the Smith set.""")
def getcha(edata, curr_cands = None):
    
    mg = get_mg(edata, curr_cands = curr_cands)
    scc = list(nx.strongly_connected_components(mg))
    min_indegree = min([max([mg.in_degree(n) for n in comp]) for comp in scc])
    smith = [comp for comp in scc if max([mg.in_degree(n) for n in comp]) == min_indegree][0]
    return sorted(list(smith))

# Create some aliases for GETCHA
getcha.set_name("Top Cycle")
top_cycle = copy.deepcopy(getcha)
getcha.set_name("Smith Set")
smith_set = copy.deepcopy(getcha)

# reset the name GETCHA
getcha.set_name("GETCHA")


@vm(
    name = "GOCHA",
    description = """The GOCHA set (also known as the Schwartz set) is the smallest set of candidates with the property
    that every candidate inside the set is not majority preferred by every candidate outside the set.""")
def gocha(edata, curr_cands = None):
    mg = get_mg(edata, curr_cands = curr_cands)
    transitive_closure =  nx.algorithms.dag.transitive_closure(mg)
    schwartz = set()
    for ssc in nx.strongly_connected_components(transitive_closure):
        if not any([transitive_closure.has_edge(c2,c1) 
                    for c1 in ssc for c2 in transitive_closure.nodes if c2 not in ssc]):
            schwartz =  schwartz.union(ssc)
    return sorted(list(schwartz))

# Create some aliases for GOCHA
gocha.set_name("Schwartz Set")
schwartz_set = copy.deepcopy(gocha)

# reset the name GETCHA
gocha.set_name("GOCHA")


@vm(
    name = "Minimax",
    description = """Return the candidates with the smallest maximum pairwise defeat.  That is, for each candidate c determine the biggest margin of a candidate c1 over c, then elect the candidates with the smallest such loss. Alson known as the Simpson-Kramer Rule.
    """)
def minimax(edata, curr_cands = None, strength_function = None):   

    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function

    scores = {c: max([strength_function(_c, c) for _c in edata.dominators(c)]) if len(edata.dominators(c)) > 0 else 0 
              for c in candidates}
    min_score = min(scores.values())
    return sorted([c for c in candidates if scores[c] == min_score])

def minimax_scores(edata, curr_cands = None, score_method="margins"):
    """Return the minimax scores for each candidate, where the minimax score for c in 
    the smallest maximum pairwise defeat. 
    """
    
    candidates = edata.candidates if curr_cands is None else curr_cands

    if len(candidates) == 1:
        return {c: 0 for c in candidates}
    
    # there are different scoring functions that can be used to measure the worse loss for each 
    # candidate. These all produce the same set of winners when voters submit linear orders. 
    score_functions = {
        "winning": lambda cs, c: max([edata.support(_c,c) for _c in cs]) if len(cs) > 0 else 0,
        "margins": lambda cs, c: max([edata.margin(_c,c) for _c in cs]) if len(cs) > 0 else 0,
        "pairwise_opposition": lambda cs, c: max([edata.support(_c,c) for _c in cs])
    } 
    
    cands = {
        "winning": lambda c: edata.dominators(c, curr_cands = curr_cands),
        "margins": lambda c: edata.dominators(c, curr_cands = curr_cands),
        "pairwise_opposition": lambda c: [_c for _c in candidates if _c != c]
    } 

    return {c: -1 * score_functions[score_method](cands[score_method](c), c) for c in candidates}


def maximal_elements(g): 
    """return the nodes in g with no incoming arrows."""
    return [n for n in g.nodes if g.in_degree(n) == 0]


@vm(
    name="Beat Path",
    description="""For candidates a and b, a *path from a to b in P* is a sequence 
    x_1,...,x_n of distinct candidates  with  x_1=a and x_n=b such that 
    for 1 <= k <= n-1$, x_k is majority preferred to x_{k+1}.  The *strength of a path* 
    is the minimal margin along that path.  Say that a defeats b in P if 
    the strength of the strongest path from a to b is greater than the strength of 
    the strongest path from b to a. Then Beat Path winners are the undefeated candidates. 
    Also known as the Schulze Rule.
    
    This is an implementation of Beat Path using a variation of the Floyd Warshall-Algorithm
    See https://en.wikipedia.org/wiki/Schulze_method#Implementation
    """)
def beat_path(edata, curr_cands = None, strength_function = None):   

    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function
    
    mg = get_mg(edata, curr_cands = curr_cands)
    
    beat_paths_weights = {c: {c2:0 for c2 in candidates if c2 != c} for c in candidates}
    for c in candidates: 
        for other_c in beat_paths_weights[c].keys():
            all_paths =  list(nx.all_simple_paths(mg, c, other_c))
            if len(all_paths) > 0:
                beat_paths_weights[c][other_c] = max([min([strength_function(p[i], p[i+1]) 
                                                           for i in range(0,len(p)-1)]) 
                                                      for p in all_paths])
    
    winners = list()
    for c in candidates: 
        if all([beat_paths_weights[c][c2] >= beat_paths_weights[c2][c] 
                for c2 in candidates  if c2 != c]):
            winners.append(c)

    return sorted(list(winners))


@vm(
    name="Beat Path",
    description="""For candidates a and b, a *path from a to b in P* is a sequence 
    x_1,...,x_n of distinct candidates with  x_1=a and x_n=b such that 
    for 1 <= k <= n-1$, x_k is majority preferred to x_{k+1}.  The *strength of a path* 
    is the minimal margin along that path.  Say that a defeats b in P if 
    the strength of the strongest path from a to b is greater than the strength of 
    the strongest path from b to a. Then Beat Path winners are the undefeated candidates. 
    Also known as the Schulze Rule.
    
    This is an implementation of Beat Path using a variation of the Floyd Warshall-Algorithm
    See https://en.wikipedia.org/wiki/Schulze_method#Implementation""")
def beat_path_faster(edata, curr_cands = None, strength_function = None):   

    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function
        
    s_matrix = [[-np.inf for _ in candidates] for _ in candidates]
    for c1_idx, c1 in enumerate(candidates):
        for c2_idx, c2 in enumerate(candidates):
            if (edata.majority_prefers(c1, c2) or c1 == c2):
                s_matrix[c1_idx][c2_idx] = strength_function(c1, c2) 
    strength = list(map(lambda i : list(map(lambda j : j , i)) , s_matrix))
    for i_idx, i in enumerate(candidates):         
        for j_idx, j in enumerate(candidates): 
            if i!= j:
                for k_idx, k in enumerate(candidates): 
                    if i!= k and j != k:
                        strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))
    winners = {i:True for i in candidates}
    for i_idx, i in enumerate(candidates): 
        for j_idx, j in enumerate(candidates):
            if i!=j:
                if strength[j_idx][i_idx] > strength[i_idx][j_idx]:
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])


@vm(
    name="Split Cycle",
    description="""A *majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.  The *strength of* a majority is the minimal margin in the cycle.  Say that a defeats b in P if the margin of a over b is positive and greater than the strength of the strongest majority cycle containing a and b. The Split Cycle winners are the undefeated candidates.
    """)
def split_cycle(edata, curr_cands = None, strength_function = None):
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function 
    
    # create the majority graph
    mg = get_mg(edata, curr_cands = None) 
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph
        
        # get all the margins (i.e., the weights) of the edges in the cycle
        strengths = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            strengths.append(strength_function(c1, c2))
            
        split_number = min(strengths) # the split number of the cycle is the minimal margin
        
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        
    
    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if edata.majority_prefers(c1, c2) and strength_function(c1,c2) > cycle_number[(c1,c2)]])
   
    # the winners are candidates not defeated by any other candidate
    winners = maximal_elements(defeat)
    
    return sorted(list(set(winners)))


@vm(
    name="Split Cycle",
    description="""A *majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates  with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.  The *strength of* a majority is the minimal margin in the cycle.  Say that a defeats b in P if the margin of a over b is positive and greater than the strength of the strongest majority cycle containing a and b. The Split Cycle winners are the undefeated candidates.
    
    Implementation of Split Cycle using a variation of the Floyd Warshall-Algorithm""")
def split_cycle_faster(edata, curr_cands = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function 
 
    weak_condorcet_winners = {c:True for c in candidates}
    s_matrix = [[-np.inf for _ in candidates] for _ in candidates]
    
    # Weak Condorcet winners are Split Cycle winners
    for c1_idx, c1 in enumerate(candidates):
        for c2_idx, c2 in enumerate(candidates):
            if (edata.majority_prefers(c1, c2) or c1 == c2):
                s_matrix[c1_idx][c2_idx] = strength_function(c1, c2) 
                weak_condorcet_winners[c2] = weak_condorcet_winners[c2] and (c1 == c2)
    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , s_matrix))
    for i_idx, i in enumerate(candidates): 
        for j_idx, j in enumerate(candidates):
            if i!= j:
                if not weak_condorcet_winners[j]: # weak Condorcet winners are Split Cycle winners
                    for k_idx, k in enumerate(candidates): 
                        if i != k and j != k:
                            strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))
    winners = {i:True for i in candidates}
    for i_idx, i in enumerate(candidates):
        for j_idx, j in enumerate(candidates):
            if i != j:
                if s_matrix[j_idx][i_idx] > strength[i_idx][j_idx]: # the main difference with Beat Path
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])


@vm(
    name="Iterated Split Cycle",
    description="""Iteratively calculate the split cycle winners until there is a
    unique winner or all remaining candidates are split cycle winners.""")
def iterated_split_cycle(edata, curr_cands = None, strength_function = None):
    
    prev_sc_winners = edata.candidates if curr_cands is None else curr_cands
    sc_winners = split_cycle_faster(edata, curr_cands = curr_cands, strength_function = strength_function)
    
    while len(sc_winners) != 1 and sc_winners != prev_sc_winners: 
        prev_sc_winners = sc_winners
        sc_winners = split_cycle_faster(edata, curr_cands = sc_winners, strength_function = strength_function)
        
    return sorted(sc_winners)


# flatten a 2d list - turn a 2d list into a single list of items
flatten = lambda l: [item for sublist in l for item in sublist]

def does_create_cycle(g, edge):
    '''return True if adding the edge to g create a cycle.
    it is assumed that edge is already in g'''
    source = edge[0]
    target = edge[1]
    for n in g.predecessors(source):
        if nx.has_path(g, target, n): 
            return True
    return False


@vm(
    name="Ranked Pairs",
    description="""Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking inear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according to some tie-breaking rule. Also known as Tideman's Rule.""")
def ranked_pairs(edata, curr_cands = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function    

    cw = edata.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2)]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        sorted_edges = [[e for e in w_edges if e[2] == s] for s in strengths]
        tbs = product(*[permutations(edges) for edges in sorted_edges])
        for tb in tbs:
            edges = flatten(tb)
            rp_defeat = nx.DiGraph() 
            for e in edges: 
                rp_defeat.add_edge(e[0], e[1], weight=e[2])
                if does_create_cycle(rp_defeat, e):
                    rp_defeat.remove_edge(e[0], e[1])
            winners.append(maximal_elements(rp_defeat)[0])
    return sorted(list(set(winners)))



@vm(
    name="Ranked Pairs",
    description="""Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according to some tie-breaking rule. Also known as Tideman's Rule.
    
    Includes a test to determined if it will take too long to compute the Ranked Pairs winners. If the calculation of the winners will take too long, return None
    """)
def ranked_pairs_with_test(edata, curr_cands = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function   

    cw = edata.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2)]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        sorted_edges = [[e for e in w_edges if e[2] == s] for s in strengths]
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 2000: 
            return None
        else: 
            tbs = product(*[permutations(edges) for edges in sorted_edges])
            for tb in tbs:
                edges = flatten(tb)
                rp_defeat = nx.DiGraph() 
                for e in edges: 
                    rp_defeat.add_edge(e[0], e[1], weight=e[2])
                    if does_create_cycle(rp_defeat, e):
                        rp_defeat.remove_edge(e[0], e[1])
                winners.append(maximal_elements(rp_defeat)[0])
    return sorted(list(set(winners)))



@vm(
    name="Ranked Pairs T",
    description="""Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according to some tie-breaking rule. Also known as Tideman's Rule.
    
    Use a fixed linear order on the candidates to break any ties in the margins.   
    Since the tie_breaker is a linear order, this method is resolute.""")
def ranked_pairs_tb(edata, curr_cands = None, tie_breaker = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function
    
    tb_ranking = tie_breaker if tie_breaker is not None else sorted(list(candidates))

    cw = edata.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2)]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        
        rp_defeat = nx.DiGraph() 
        for s in strengths: 
            edges = [e for e in w_edges if e[2] == s]
            
            # break ties using the lexicographic ordering on tuples given tb_ranking
            sorted_edges = sorted(edges, key = lambda e: (tb_ranking.index(e[0]), tb_ranking.index(e[1])), reverse=False)
            for e in sorted_edges: 
                rp_defeat.add_edge(e[0], e[1], weight=e[2])
                if does_create_cycle(rp_defeat, e):
                    rp_defeat.remove_edge(e[0], e[1])
        winners.append(maximal_elements(rp_defeat)[0])

    return sorted(list(set(winners)))


@vm(
    name="Ranked Pairs ZT",
    description="""Ranked pairs (see the Ranked Pairs description for the explanation) where a fixed voter breaks any ties in the margins.  It is always the voter in position 0 that breaks the ties.  Since voters have strict preferences, this method is resolute.  This is known as Ranked Pairs ZT, for Zavist Tideman. """)
def ranked_pairs_zt(profile, curr_cands = None, strength_function = None):   
    
    candidates = profile.candidates if curr_cands is None else curr_cands    
    
    # the tie-breaker is always the first voter. 
    tb_ranking = tuple([c for c in list(profile._rankings[0]) if c in candidates])
    
    return ranked_pairs_tb(profile, curr_cands = curr_cands, tie_breaker = tb_ranking, strength_function = strength_function)



@vm(
    name="River",
    description="""Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle and edges in which there is already an edge pointing to the target.  Break ties using a tie-breaking  linear ordering over the edges.  A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. See https://electowiki.org/wiki/River.""")
def river(edata, curr_cands = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function    

    cw = edata.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2)]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        sorted_edges = [[e for e in w_edges if e[2] == s] for s in strengths]
        tbs = product(*[permutations(edges) for edges in sorted_edges])
        for tb in tbs:
            edges = flatten(tb)
            river_defeat = nx.DiGraph() 
            for e in edges: 
                if e[1] not in river_defeat.nodes or len(list(river_defeat.in_edges(e[1]))) == 0:
                    river_defeat.add_edge(e[0], e[1], weight=e[2])
                    if does_create_cycle(river_defeat, e):
                        river_defeat.remove_edge(e[0], e[1])
            winners.append(maximal_elements(river_defeat)[0])
    return sorted(list(set(winners)))


@vm(
    name="River",
    description="""Order the edges in the weak margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle and edges in which there is already an edge pointing to the target.  Break ties using a tie-breaking  linear ordering over the edges.  A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. See https://electowiki.org/wiki/River.

    Includes a test to determined if it will take too long to compute the Ranked Pairs winners. If the calculation of the winners will take too long, return None. 
    """)
def river_with_test(edata, curr_cands = None, strength_function = None):   
    
    candidates = edata.candidates if curr_cands is None else curr_cands    
    strength_function = edata.margin if strength_function is None else strength_function    

    cw = edata.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        w_edges = [(c1, c2, strength_function(c1, c2)) for c1 in candidates for c2 in candidates 
                   if edata.majority_prefers(c1, c2) or edata.is_tied(c1, c2)]
        winners = list()            
        strengths = sorted(list(set([e[2] for e in w_edges])), reverse=True)
        sorted_edges = [[e for e in w_edges if e[2] == s] for s in strengths]
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 2000: 
            return None
        else: 
            tbs = product(*[permutations(edges) for edges in sorted_edges])
            for tb in tbs:
                edges = flatten(tb)
                river_defeat = nx.DiGraph() 
                for e in edges: 
                    if e[1] not in river_defeat.nodes or len(list(river_defeat.in_edges(e[1]))) == 0:
                        river_defeat.add_edge(e[0], e[1], weight=e[2])
                        if does_create_cycle(river_defeat, e):
                            river_defeat.remove_edge(e[0], e[1])
                winners.append(maximal_elements(river_defeat)[0])
    return sorted(list(set(winners)))

# Simple Stable Voting 
def simple_stable_voting_(edata, curr_cands = None, mem_sv_winners = {}, strength_function = None): 
    '''
    Determine the Simple Stable Voting winners while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else edata.candidates 
    strength_function = edata.margin if strength_function is None else strength_function  
    
    sv_winners = list()

    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b]
    strengths = list(set([strength_function(a, b) for a,b in matches]))
    
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for s in sorted(strengths, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if strength_function(ab_match[0], ab_match[1]) == s]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = simple_stable_voting_(edata, 
                                                               curr_cands = [c for c in curr_cands if c != b],
                                                               mem_sv_winners = mem_sv_winners,
                                                               strength_function = strength_function)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
                
@vm(
    name = "Simple Stable Voting",
    description = """Implementation of  Simple Stable Voting from https://arxiv.org/abs/2108.00542
    """)
def simple_stable_voting(edata, curr_cands = None, strength_function = None): 
    
    return simple_stable_voting_(edata, curr_cands = curr_cands, mem_sv_winners = {}, strength_function = strength_function)[0]


@vm(
    name = "Simple Stable Voting",
    description = """Implementation of  Simple Stable Voting from https://arxiv.org/abs/2108.00542. 

    Simple Stable Voting is Condorcet consistent.   It is faster to skip executing the recursive algorithm when there is a Condorcet winner
    """)
def simple_stable_voting_faster(edata, curr_cands = None, strength_function = None): 
    '''First check if there is a Condorcet winner.  If so, return the Condorcet winner, otherwise find the stable voting winnner using stable_voting_'''
    
    cw = edata.condorcet_winner(curr_cands = None)
    if cw is not None: 
        return [cw]
    else: 
        return simple_stable_voting_(edata, curr_cands = curr_cands, mem_sv_winners = {}, strength_function = strength_function)[0]

    
def stable_voting_(edata, curr_cands = None, mem_sv_winners = {}, strength_function = None): 
    '''
    Determine the Stable Voting winners for the profile while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else edata.candidates
    strength_function = edata.margin if strength_function is None else strength_function  

    sv_winners = list()
    
    undefeated_candidates = split_cycle_faster(edata, curr_cands = curr_cands)
    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b if a in undefeated_candidates]
    strengths = list(set([strength_function(a, b) for a,b in matches]))
        
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for s in sorted(strengths, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if strength_function(ab_match[0], ab_match[1])  == s]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_(edata, 
                                                        curr_cands = [c for c in curr_cands if c != b],
                                                        mem_sv_winners = mem_sv_winners,
                                                        strength_function = strength_function)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
        
@vm(
    name = "Stable Voting",
    description = """Implementation of Stable Voting from https://arxiv.org/abs/2108.00542. 
    """)
def stable_voting(edata, curr_cands = None, strength_function = None): 
    return stable_voting_(edata, curr_cands = curr_cands, mem_sv_winners = {}, strength_function = strength_function)[0]


@vm(
    name = "Stable Voting",
    description = """Implementation of Stable Voting from https://arxiv.org/abs/2108.00542. 
    
    Stable Voting is Condorcet consistent.   It is faster to skip executing the recursive algorithm when there is a Condorcet winner
    """)
def stable_voting_faster(edata, curr_cands = None, strength_function = None): 
    
    cw = edata.condorcet_winner(curr_cands = curr_cands)
    if cw is not None: 
        return [cw]
    else: 
        return stable_voting_(edata, curr_cands = curr_cands, mem_sv_winners = {}, strength_function = strength_function)[0]


### TODO: Move to MarginGraph?
#

def cycle_number(edata, curr_cands = None):
    
    candidates = edata.candidates if curr_cands is None else curr_cands
    
    # get the margin graph
    mg = get_mg(edata, curr_cands = curr_cands)
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(edata.margin(c1, c2))
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        
    return cycle_number

def split_cycle_defeat(edata, curr_cands = None):
    """A majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = edata.candidates if curr_cands is None else curr_cands
    
    # create the margin graph
    mg = get_mg(edata, curr_cands = curr_cands)
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(edata.margin(c1, c2))
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_weighted_edges_from([(c1,c2, edata.margin(c1, c2))  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if edata.margin(c1,c2) > cycle_number[(c1,c2)]])

    return defeat


def display_mg_with_sc(edata, curr_cands = None, cmap = None):
    
    cmap = cmap if cmap is not None else edata.cmap
    
    cmap_to_cand = {cm: c for c, cm in cmap.items()}
    candidates = edata.candidates if curr_cands is None else curr_cands

    mg = nx.DiGraph()
    mg.add_nodes_from([cmap[c] for c in candidates])
    mg.add_weighted_edges_from([(cmap[c1], cmap[c2], edata.margin(c1,c2))
                                for c1 in candidates for c2 in candidates if c1 != c2 and edata.margin(c1, c2) > 0])
    sc_defeat = split_cycle_defeat(edata, curr_cands = curr_cands)
    sc_winners =  maximal_elements(sc_defeat)
    sc_edges = sc_defeat.edges()
    
    edges = mg.edges()
    colors = ['blue' if (cmap_to_cand[e[0]], cmap_to_cand[e[1]]) in sc_edges else 'black' for e in edges]
    widths = [3 if (cmap_to_cand[e[0]], cmap_to_cand[e[1]]) in sc_edges else 1.5 for e in edges]
    
    pos = nx.circular_layout(mg)
    nx.draw(mg, pos, edges=edges, edge_color=colors, width=widths,
            font_size=20, node_color=['blue' if cmap_to_cand[n] in sc_winners else 'red' for n in mg.nodes], font_color='white', node_size=700, 
            with_labels=True)
    labels = nx.get_edge_attributes(mg,'weight')
    nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)
    plt.show()




mg_vms = [
    condorcet,
    copeland,
    llull,
    uc_gill,
    uc_fish,
    uc_bordes,
    uc_mckelvey,
    getcha,
    gocha,
    minimax, 
    split_cycle,
    split_cycle_faster,
    beat_path,
    beat_path_faster,
    #ranked_pairs,
    #ranked_pairs_with_test,
    ranked_pairs_zt,
    ranked_pairs_tb,
    #river,
    #river_with_test, 
    iterated_split_cycle,
    simple_stable_voting,
    simple_stable_voting_faster,
    stable_voting,
    stable_voting_faster,
]


mg_vms_all = [
    condorcet,
    copeland,
    llull,
    uc_gill,
    uc_fish,
    uc_bordes,
    uc_mckelvey,
    getcha,
    gocha,
    minimax, 
    split_cycle,
    split_cycle_faster,
    beat_path,
    beat_path_faster,
    ranked_pairs,
    ranked_pairs_with_test,
    ranked_pairs_zt,
    ranked_pairs_tb,
    river,
    river_with_test, 
    iterated_split_cycle,
    simple_stable_voting,
    simple_stable_voting_faster,
    stable_voting,
    stable_voting_faster,
]
