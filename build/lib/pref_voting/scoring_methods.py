'''
    File: scoring_rules.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    
    Implemenation of scoring rules. 
'''
from pref_voting.voting_method import  *
from pref_voting.voting_method import _num_rank_last

@vm(
    name = "Majority",
    description ="""The majority winner is the candidate with a strict majority  of first place votes.  Returns an empty list if there is no candidate with a strict majority of first place votes.
    """)
def majority(profile, curr_cands = None):

    maj_size = profile.strict_maj_size()
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    maj_winner = [c for c in curr_cands if plurality_scores[c] >= maj_size]
    return sorted(maj_winner)


@vm(
    name = "Plurality",
    description ="""The Plurality score of a candidate $c$ is the number of voters that rank $c$ in first place. The Plurality winners are the candidates with the largenst Plurality score. 
    """)
def plurality(profile, curr_cands = None):

    curr_cands = profile.candidates if curr_cands is None else curr_cands

    # get the Plurality scores for all the candidates in curr_cands
    plurality_scores = profile.plurality_scores(curr_cands = curr_cands)
    
    max_plurality_score = max(plurality_scores.values())

    return sorted([c for c in curr_cands if plurality_scores[c] == max_plurality_score])


@vm(
    name = "Borda",
    description ="""The Borda score of a candidate is calculated as follows: If there are $m$ candidates, then the Borda score of candidate $c$ is $\sum_{r=1}^{m} (m - r) * Rank(c,r)$ where $Rank(c,r)$ is the number of voters that rank candidate $c$ in position $r$. The Borda winners are the candidates with the largest Borda score. 
    """)
def borda(profile, curr_cands = None):
 
    curr_cands = profile.candidates if curr_cands is None else curr_cands

    # get the Borda scores for all the candidates in curr_cands
    borda_scores = profile.borda_scores(curr_cands = curr_cands)
    
    max_borda_score = max(borda_scores.values())
    
    return sorted([c for c in curr_cands if borda_scores[c] == max_borda_score])


@vm(
    name = "Anti-Plurality",
    description = """The Anti-Plurlity score of a candidate $c$ is the number of voters that rank $c$ in last place.  The Anti-Plurality winnners are the candidates with the smallest Anit-Plurality score. 
    """)
def anti_plurality(profile, curr_cands = None):
    
    # get ranking data
    rankings, rcounts = profile.rankings_counts

    curr_cands = profile.candidates if curr_cands is None else curr_cands
    cands_to_ignore = np.array([c for c in profile.candidates if c not in curr_cands])
    
    last_place_scores = {c: _num_rank_last(rankings, rcounts, cands_to_ignore, c) for c in curr_cands}
    min_last_place_score = min(list(last_place_scores.values()))
    
    return sorted([c for c in curr_cands if last_place_scores[c] == min_last_place_score])


scoring_vms = [
    majority,
    plurality, 
    borda, 
    anti_plurality
]