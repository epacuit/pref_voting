'''
    File: iterative_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 6, 2022
    
    Implementations of voting methods that combine multiple methods
'''

from pref_voting.voting_method import  *
from pref_voting.scoring_methods import plurality, borda
from pref_voting.iterative_methods import iterated_removal_cl, instant_runoff
from pref_voting.margin_based_methods import smith_set, copeland


@vm(
    name = "Daunou",
    description = """Implementation of Daunou's voting method as described in the paper: 
    https://link.springer.com/article/10.1007/s00355-020-01276-w
    
    If there is a Condorcet winner, then that candidate is the winner.  Otherwise, 
    iteratively remove all Condorcet losers then select the plurality winner from among 
    the remaining conadidates."""
)
def daunou(profile, curr_cands = None):

    candidates = profile.candidates if curr_cands is None else curr_cands
    cw = profile.condorcet_winner(curr_cands = curr_cands)
    if cw is not None: 
        winners = [cw]
    else: 
        cands_survive_it_rem_cl = iterated_removal_cl(profile, curr_cands = curr_cands)
        winners = plurality(profile, curr_cands = cands_survive_it_rem_cl)
        
    return sorted(winners)


@vm(
    name = "Blacks",
    description = """If a Condorcet winner exists return that winner. Otherwise, return the Borda winning set.
    """)
def blacks(profile, curr_cands = None):
    
    cw = profile.condorcet_winner(curr_cands = curr_cands)
    
    if cw is not None:
        winners = [cw]
    else:
        winners = borda(profile, curr_cands = curr_cands)
        
    return winners


@vm(
    name = "Smith IRV",
    description = """Find the Smith Set.  Then return the Instant Runoff winners after restricting to the Smith Set.
    """)
def smith_irv(profile, curr_cands = None): 
        
    smith = smith_set(profile, curr_cands = curr_cands)
    
    return instant_runoff(profile, curr_cands = smith)

#compose two voting methods.
def compose(vm1, vm2): 

    def vm(edata, curr_cands = None): 
        
        vm1_ws = vm1(edata, curr_cands = curr_cands)

        return vm2(edata, curr_cands = vm1_ws)
            
    return VotingMethod(vm, name = f"{vm1.name}-{vm2.name}", description = f"Find the {vm1.name} winners.  After restricting to this set of winners, find the {vm2.name} winners.")

copeland_borda = compose(copeland, borda)


combined_vms = [
    daunou,
    blacks,
    smith_irv,
    copeland_borda
]