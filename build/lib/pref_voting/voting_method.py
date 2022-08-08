'''
    File: voting_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: November 6, 2021
    
    The VotingMethod class and helper functions for voting methdods
'''

import numpy as np
from numba import jit
import random

class VotingMethod(object): 
    
    def __init__(self, vm, name = None, description = None): 
        
        self.vm = vm
        self.name = name
        self.description = description
        
    def __call__(self, edata, curr_cands = None, **kwargs):
        
        #if curr_cands is not None and len(curr_cands) == 1: 
        #    return curr_cands

        #if "tie_breaker" in kwargs.keys(): 
        ##    tie_breaker = kwargs["tie_breaker"]
        #    return self.vm(edata, curr_cands = curr_cands, tie_breaker = tie_breaker)
        
        #else: 
        return self.vm(edata, curr_cands = curr_cands, **kwargs)
    
    def choose(self, edata, curr_cands = None): 
        # randomly choose a winner from the winning set
        ws = self.__call__(edata, curr_cands = curr_cands)
        return random.choice(ws)
    
    def display(self, edata, curr_cands = None, cmap = None, **kwargs): 
        # display the winning set 
        cmap = cmap if cmap is not None else edata.cmap

        ws = self.__call__(edata, curr_cands = curr_cands, **kwargs)

        if ws is None:  # some voting methods, such as Ranked Pairs, may return None if it is too long to computer the winner.
            print(f"{self.name} winning set is not available")
        else: 
            w_str = f"{self.name} winner is " if len(ws) == 1 else f"{self.name} winners are "
            print(w_str + "{" + ", ".join([str(cmap[c]) for c in ws]) + "}")
        
    def set_name(self, new_name):
        # set the name of the voting method
        self.name = new_name

    def __str__(self): 
        return f"{self.name}\n{self.description}"

def vm(name = None, description = None):
    def wrapper(f):
        return VotingMethod(f, name=name, description=description)
    return wrapper

@jit(fastmath=True)
def isin(arr, val):
    """compiled function testing if the value val is in the array arr
    """
    
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False

@jit(nopython=True)
def _num_rank_first(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand first after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    top_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(0, len(rankings[vidx])):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                top_cands_indices[vidx] = level
                break                
    top_cands = np.array([rankings[vidx][top_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = top_cands == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 


@jit(nopython=True)
def _num_rank_last(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand last after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    last_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(len(rankings[vidx]) - 1,-1,-1):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                last_cands_indices[vidx] = level
                break                
    bottom_cands = np.array([rankings[vidx][last_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = bottom_cands  == cand
    return np.sum(is_cand * rcounts) 

def vm_name(vm_name):
    def wrapper(f):
        f.name = vm_name
        return f
    return wrapper
