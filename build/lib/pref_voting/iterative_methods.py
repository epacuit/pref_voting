'''
    File: iterative_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    
    Implementations of iterative voting methods
'''

from pref_voting.voting_method import  *
from pref_voting.voting_method import _num_rank_last, _num_rank_first
from pref_voting.profiles import  _borda_score, _find_updated_profile, _num_rank

import copy
from itertools import permutations, product
import numpy as np

@vm(
    name = "Instant Runoff",
    description ="""If there is a majority winner then that candidate is the  winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest number of voters.  Continue removing candidates with the fewest number first-place votes until there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    
    Note: This voting method is known as "Ranked Choice", "Hare", "Alternative Voter", and "STV"
    """)
def instant_runoff(profile, curr_cands = None):
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: # removed all of the candidates 
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)

# Create some aliases for instant runoff
instant_runoff.set_name("Hare")
hare = copy.deepcopy(instant_runoff)
instant_runoff.set_name("Ranked Choice")
ranked_choice = copy.deepcopy(instant_runoff)

# reset the name Instant Runoff
instant_runoff.set_name("Instant Runoff")


@vm(
    name = "Instant Runoff TB",
    description ="""If there is a majority winner then that candidate is the  winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest number of voters.  Continue removing candidates with the fewest number first-place votes until there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then remove the candidate with lowest in the tie_breaker ranking from the profile. 
    """)
def instant_runoff_tb(profile, curr_cands = None, tie_breaker = None):

    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates if not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if plurality_scores[c] == min_plurality_score])
        
        cand_to_remove = lowest_first_place_votes[0]
        for c in lowest_first_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, cand_to_remove), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)


@vm(
    name = "Instant Runoff PUT",
    description ="""Instant Runoff with parallel universe tie-breaking (PUT).  Apply the Instant Runoff method with a tie-breaker for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates. 
    """)
def instant_runoff_put(profile, curr_cands = None):
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
        
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    if len(winners) == 0:
        # run Instant Runoff with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += instant_runoff_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 
    return sorted(list(set(winners)))


# Create some aliases for instant runoff
instant_runoff_put.set_name("Hare PUT")
hare_put = copy.deepcopy(instant_runoff_put)
instant_runoff.set_name("Ranked Choice PUT")
ranked_choice_put = copy.deepcopy(instant_runoff_put)

# reset the name Instant Runoff
instant_runoff_put.set_name("Instant Runoff PUT")


@vm(
    name = "Instant Runoff (with data)",
    description ="""If there is a majority winner then that candidate is the  winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest number of voters.  Continue removing candidates with the fewest number first-place votes until there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then *all* such candidates are removed from the profile. 
    
    Note: This voting method is known as "Ranked Choice", "Hare", "Alternative Voter", and "STV".
    
    Also returns the order of elimination.
    """)
def instant_runoff_with_data(profile, curr_cands = None):
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
    elims_list = list()

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        elims_list.append(list(lowest_first_place_votes))

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: # removed all of the candidates 
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners), elims_list


@vm(
    name = "PluralityWRunoff",
    description ="""If there is a majority winner then that candidate is the plurality with runoff winner. If there is no majority winner, then hold a runoff with  the top two candidates: either two (or more candidates)  with the most first place votes or the candidate with the most first place votes and the candidate with the 2nd highest first place votes are ranked first by the fewest number of voters.  
    
    A candidate is a Plurality with Runoff winner if it is a winner in a runoff between two pairs of first- or second- ranked candidates. 
    
    Note: If the candidates are all tied for the most first place votes, then all candidates are winners. 
    """)
def plurality_with_runoff(profile, curr_cands = None):
    
    curr_cands = profile.candidates if curr_cands is None else curr_cands
    
    if len(curr_cands) == 1: 
        return list(curr_cands)
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)  

    max_plurality_score = max(plurality_scores.values())
    
    first = [c for c in curr_cands if plurality_scores[c] == max_plurality_score]
    second = list()
    if len(first) == 1:
        second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
        second = [c for c in curr_cands if plurality_scores[c] == second_plurality_score]

    if len(second) > 0:
        all_runoff_pairs = product(first, second)
    else: 
        all_runoff_pairs = [(c1,c2) for c1,c2 in product(first, first) if c1 != c2]

    winners = list()
    for c1, c2 in all_runoff_pairs: 
        
        if profile.margin(c1,c2) > 0:
            winners.append(c1)
        elif profile.margin(c1,c2) < 0:
            winners.append(c2)
        elif profile.margin(c1,c2) == 0:
            winners.append(c1)
            winners.append(c2)
    
    return sorted(list(set(winners)))


@vm(
    name = "Coombs",
    description ="""If there is a majority winner then that candidate is the Coombs winner
    If there is no majority winner, then remove all candidates that are ranked last by the greatest number of voters.  Continue removing candidates with the most last-place votes until there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the most number of last-place votes, then *all* such candidates are removed from the profile. 
    """)
def coombs(profile, curr_cands = None):
    
    # need the total number of all candidates in a profile to check when all candidates have been removed   
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands: # removed all candidates 
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)


@vm(
    name = "Coombs TB",
    description ="""Coombs with a fixed tie-breaking rule:  If there is a majority winner then that candidate is the Coombs winner.  If there is no majority winner, then remove all candidates that are ranked last by the greatest  number of voters.  If there are ties, then choose the candidate according to a fixed tie-breaking rule (given below). Continue removing candidates with the most last-place votes until there is a candidate with a majority of first place votes.  
    
    The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule is to order the candidates as follows: 0,....,num_cands-1.
    """)
def coombs_tb(profile, curr_cands = None, tie_breaker=None):
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = [c for c in last_place_scores.keys() 
                                     if  last_place_scores[c] == max_last_place_score]

        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = greatest_last_place_votes[0]
        for c in greatest_last_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

@vm(
    name = "Coombs PUT",
    description ="""Coombs with parallel universe tie-breaking (PUT).  Apply the Coombs method with a tie-breaker for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates. 
    """)
def coombs_put(profile, curr_cands = None):

    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, np.empty(0), c) >= strict_maj_size]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += coombs_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 

    return sorted(list(set(winners)))


@vm(
    name = "Coombs (with data)",
    description ="""If there is a majority winner then that candidate is the Coombs winner
    If there is no majority winner, then remove all candidates that are ranked last by the greatest number of voters.  Continue removing candidates with the most last-place votes until there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the most number of last-place votes, then *all* such candidates are removed from the profile.
    
    Also returns the order of elimination.
    """)
def coombs_with_data(profile, curr_cands = None):
        
    num_cands = profile.num_cands 
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    elims_list = list()
    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        elims_list.append(list(greatest_last_place_votes))
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners), elims_list

@vm(
    name = "Baldwin",
    description ="""Iteratively remove all candidates with the lowest Borda score until a single candidate remains.  If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """)
def baldwin(profile, curr_cands = None):

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(
    name = "Baldwin TB",
    description ="""Iteratively remove all candidates with the lowest Borda score until a single  candidate remains.  If, at any stage, all candidates have the same Borda score, 
    then all (remaining) candidates are winners.

    The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule is to order the candidates as follows: 0,....,num_cands-1.
    """)
def baldwin_tb(profile, curr_cands = None, tie_breaker=None):
 
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cand_to_remove = last_place_borda_scores[0]
    for c in last_place_borda_scores[1:]: 
        if tb.index(c) < tb.index(cand_to_remove):
            cand_to_remove = c
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = last_place_borda_scores[0]
        for c in last_place_borda_scores[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners)

@vm(
    name = "Baldwin PUT",
    description ="""Baldwin with parallel universe tie-breaking (PUT).  Apply the baldwin method with a tie-breaker for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates.
    """)
def baldwin_put(profile, curr_cands=None):

    candidates = profile.candidates if curr_cands is None else curr_cands    
    cw = profile.condorcet_winner(curr_cands=curr_cands)
    
    winners = list() if cw is None else [cw]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += baldwin_tb(profile, curr_cands = curr_cands, tie_breaker = tb) 

    return sorted(list(set(winners)))


@vm(
    name = "Baldwin",
    description ="""Iteratively remove all candidates with the lowest Borda score until a single candidate remains.  If, at any stage, all candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    
    Return the list of candidates that are eliminated.
    """)
def baldwin_with_data(profile, curr_cands = None):

    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    elims_list = list()
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
    elims_list.append({c: min_borda_score for c in last_place_borda_scores})
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(last_place_borda_scores)), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  all_num_cands: # all candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        elims_list.append({c: min_borda_score for c in last_place_borda_scores})
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(last_place_borda_scores)), axis=None)
                
        if cands_to_ignore.shape[0] == all_num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    return sorted(winners), elims_list


@vm(
    name = "Strict Nanson",
    description ="""Iteratively remove all candidates with the  Borda score strictly below the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.
    """)
def strict_nanson(profile, curr_cands = None):
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] < avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (below_borda_avg_candidates.shape[0] == 0) or ((all_num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners


@vm(
    name = "Strict Nanson",
    description ="""Iteratively remove all candidates with the  Borda score strictly below the average Borda score until one candidate remains.   If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.
    
    Return the elimination list (including the average Borda score and Borda scores for each removed candidate). 
    """)
def strict_nanson_with_data(profile, curr_cands = None):
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0)
    elim_list = list()
    
    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = [c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score]
    
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in candidates}})
        winners = sorted(candidates)
    else: 
        num_cands = len(candidates)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in below_borda_avg_candidates}})
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_score = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                      if borda_scores[c] < avg_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in below_borda_avg_candidates}})
                
        if (len(below_borda_avg_candidates) == 0) or ((all_num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners, elim_list


@vm(
    name = "Weak Nanson",
    description ="""Iteratively remove all candidates with the  Borda score less than or equal to the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.
    """)
def weak_nanson(profile, curr_cands = None):
    
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])

    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}

    avg_borda_score = np.mean(list(borda_scores.values()))

    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] <= avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
        winners = [c for c in candidates if not isin(cands_to_ignore, c)]
    else: 
        num_cands = len(candidates)
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        

        avg_borda_score = np.mean(list(borda_scores.values()))

        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] <= avg_borda_score])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
        
        if cands_to_ignore.shape[0] == all_num_cands:  # all remaining candidates have been removed
            winners = sorted(below_borda_avg_candidates)
        elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
            winners = [c for c in candidates if not isin(cands_to_ignore, c)]
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners


@vm(
    name = "Weak Nanson",
    description ="""Iteratively remove all candidates with the  Borda score less than or equal to the average Borda score until one candidate remains.  If, at any stage, all  candidates have the same Borda score, then all (remaining) candidates are winners.
    
    Return the elimination list (including the average Borda score and Borda scores for each removed candidate). 
    """)
def weak_nanson_with_data(profile, curr_cands = None):
    all_num_cands = profile.num_cands   
    candidates = profile.candidates if curr_cands is None else curr_cands
        
    rcounts = profile._rcounts # get all the ranking data
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), all_num_cands)
    cands_to_ignore = np.empty(0) if curr_cands is None else np.array([c for c in profile.candidates if c not in curr_cands])
    elim_list = list()
    
    borda_scores = {c: _borda_score(rankings, rcounts, len(candidates), c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                  if borda_scores[c] <= avg_borda_score]
    
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
    winners = list()
    if cands_to_ignore.shape[0] == all_num_cands:  # all candidates have same Borda score
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in candidates}})
        winners = sorted(candidates)
    elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in below_borda_avg_candidates}})
        winners = [c for c in candidates if not isin(cands_to_ignore, c)]
    else: 
        num_cands = len(candidates)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in below_borda_avg_candidates}})
        updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
        

    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_rankings, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore, c)}
        
        avg_borda_score = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = [c for c in borda_scores.keys() 
                                      if borda_scores[c] <= avg_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array(below_borda_avg_candidates)), axis=None)
        elim_list.append({"avg_borda_score": avg_borda_score, 
                          "elim_cands": {c: borda_scores[c] for c in below_borda_avg_candidates}})
                
        if cands_to_ignore.shape[0] == all_num_cands:  # all remaining candidates have been removed
            winners = sorted(below_borda_avg_candidates)
        elif all_num_cands - cands_to_ignore.shape[0]  == 1: # one candidate remains
            winners = [c for c in candidates if not isin(cands_to_ignore, c)]
        else:
            updated_rankings = _find_updated_profile(rankings, cands_to_ignore, num_cands)
            
    return winners, elim_list


@vm(
    name = "Simplified Bucklin",
    description ="""If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes. 
    """)
def simplified_bucklin(profile, curr_cands = None): 

    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
            
    return sorted([c for c in candidates if cand_scores[c] >= strict_maj_size])


@vm(
    name = "Simplified Bucklin",
    description ="""If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes. 
    """)
def simplified_bucklin_with_data(profile, curr_cands = None): 

    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
            
    return sorted([c for c in candidates if cand_scores[c] >= strict_maj_size]), cand_scores


@vm(
    name = "Bucklin",
    description ="""If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes.  Return the candidates with the greatest overall score.  
    """)
def bucklin(profile, curr_cands = None): 

    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    max_score = max(cand_scores.values())
    return sorted([c for c in candidates if cand_scores[c] >= max_score])


@vm(
    name = "Bucklin",
    description ="""If a candidate has a strict majority of first-place votes, then that candidate is the winner. If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate has a strict majority of first- or second-place voters, then that candidate is the winner. If no such winner is found move on to the 3rd, 4th, etc. place votes.  Return the candidates with the greatest overall score.  

    Return the candidate scores. 
    """)
def bucklin_with_data(profile, curr_cands = None): 

    strict_maj_size = profile.strict_maj_size()
    
    candidates = profile.candidates if curr_cands is None else curr_cands
    
    num_cands = candidates
    
    rankings = profile._rankings if curr_cands is None else _find_updated_profile(profile._rankings, np.array([c for c in profile.candidates if c not in curr_cands]), profile.num_cands)

    rcounts = profile._rcounts
    
    num_cands = len(candidates)
    ranks = range(1, num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: _num_rank(rankings, rcounts, c, r)
                                      for c in candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    max_score = max(cand_scores.values())
    return sorted([c for c in candidates if cand_scores[c] >= max_score]), cand_scores


@vm(
    name = "Iterated Removal Condorcet Loser",
    description ="""The winners are the candidates that survive iterated removal of Condorcet losers. 
    """)
def iterated_removal_cl(profile, curr_cands = None):

    condorcet_loser = profile.condorcet_loser(curr_cands = curr_cands)  
    
    remaining_cands = profile.candidates if curr_cands is None else curr_cands
    
    while len(remaining_cands) > 1 and  condorcet_loser is not None:    
        remaining_cands = [c for c in remaining_cands if c not in [condorcet_loser]]
        condorcet_loser = profile.condorcet_loser(curr_cands = remaining_cands)
            
    return sorted(remaining_cands)


@vm(
    name = "Iterated Removal Condorcet Loser",
    description ="""The winners are the candidates that survive iterated removal of Condorcet losers. 

    Also return the list of candidates that are eliminated.
    """)
def iterated_removal_cl_with_data(profile, curr_cands = None):

    elim_list = list()
    condorcet_loser = profile.condorcet_loser(curr_cands = curr_cands)  
    
    remaining_cands = profile.candidates if curr_cands is None else curr_cands
    
    while len(remaining_cands) > 1 and  condorcet_loser is not None: 
        elim_list.append(condorcet_loser)   
        remaining_cands = [c for c in remaining_cands if c not in [condorcet_loser]]
        condorcet_loser = profile.condorcet_loser(curr_cands = remaining_cands)
            
    return sorted(remaining_cands), elim_list

iterated_vms = [
    instant_runoff,
    instant_runoff_tb,
    instant_runoff_put,
    hare,
    ranked_choice,
    plurality_with_runoff,
    coombs,
    coombs_tb,
    coombs_put,
    strict_nanson,
    weak_nanson,
    baldwin,
    baldwin_tb,
    baldwin_put,
    bucklin,
    simplified_bucklin,
    iterated_removal_cl,
]

iterated_vms_with_data = [
    instant_runoff_with_data,
    coombs_with_data,
    baldwin_with_data,
    strict_nanson_with_data,
    weak_nanson_with_data,
    simplified_bucklin_with_data,
    bucklin_with_data,
    iterated_removal_cl_with_data,
]

