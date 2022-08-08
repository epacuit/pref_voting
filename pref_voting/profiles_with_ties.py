'''
    File: profiles_with_ties.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022
    Updated: July 13, 2022
    
    Functions to reason about profiles of (truncated) strict weak orders.
'''

from math import ceil
import copy
import numpy as np
from  tabulate import tabulate
from pref_voting.weighted_majority_graphs import MajorityGraph, MarginGraph, SupportGraph

class Ranking(object):
    """A ranking of a set of candidates.

    A ranking is a map from candidates to ranks (integers).  There is no assumption that all candidates in an election are ranked.   The ranks can be any numbers () or that the    

    :param rmap: Dictionary in which the keys are the candidates and the values are the ranks.
    :type rmap: dict[int or str: int]
    :param cmap: Dictionary mapping candidates (keys of the ``rmap``) to candidate names (strings).  If not provied, each candidate  is mapped to itself. 
    :type cmap: dict[int: str], optional

    :Example:
    
    The following code creates three rankings: 

    1. ``rank1`` is the ranking where 0 is ranked first, 2 is ranked in second-place, and 1 is ranked last.  
    2. ``rank2`` is the ranking where 0 and 1 are tied for first place, and 2 is ranked last.  
    3. ``rank3`` is the ranking where 0 is ranked first, and 2 is ranked in last place.    

    .. code-block:: python

            rank1 = Ranking({0:1, 1:3, 2:2})
            rank2 = Ranking({0:1, 1:1, 2:2})
            rank3 = Ranking({0:1, 2:3})

    .. important::
        The numerical value of the ranks do not mean anything.  They are only used to make ordinal comparisons.  For instance, each of the following represents the same ranking: 
        0 is ranked  first, 2 is ranked second, and 1 is ranked in last place.

        .. code-block:: python

            rank1 = Ranking({0:1, 1:3, 2:2})
            rank2 = Ranking({0:1, 1:10, 2:3})
            rank3 = Ranking({0:10, 1:100, 2:30})

    """

    def __init__(self, rmap, cmap = None):
        """constructer method"""

        self.rmap = rmap
        self.cmap = cmap if cmap is not None else {c:str(c) for c in rmap.keys()}
        
    @property
    def ranks(self): 
        """Returns a sorted list of the ranks."""
        return sorted(set(self.rmap.values()))
    
    @property
    def cands(self):
        """Returns a sorted list of the candidates that are ranked."""
        return sorted(list(self.rmap.keys()))
    
    def cands_at_rank(self, r):
        """Returns a list of the candidates that are assigned the rank ``r``."""
        return [c for c in self.rmap.keys() if self.rmap[c] == r]
    
    def is_ranked(self, c): 
        """Returns True if the candidate ``c`` is ranked."""

        return c in self.rmap.keys()

    def strict_pref(self, c1, c2):
        """Returns True if ``c1`` is strictly preferred to ``c2``. 

        The return value is True when both ``c1`` and ``c2`` are ranked and the rank of ``c1`` is strictly smaller than the rank of ``c2``.  
        """

        return (self.is_ranked(c1) and self.is_ranked(c2)) and self.rmap[c1] < self.rmap[c2]

    def extended_strict_pref(self, c1, c2):
        """Returns True when either ``c1`` is ranked and ``c2`` is not ranked or the rank of ``c1`` is strictly smaller than the rank of ``c2``.  
        """

        return (self.is_ranked(c1) and not self.is_ranked(c2)) or\
    ((self.is_ranked(c1) and self.is_ranked(c2)) and self.rmap[c1] < self.rmap[c2])

    def indiff(self, c1, c2):
        """Returns True if ``c1`` is indifferent with ``c2``. 

        The return value is True when  both ``c1`` and  ``c2`` are  ranked and the rank of ``c1`` is equals the rank of ``c2``.  
        
        """

        return self.is_ranked(c1) and self.is_ranked(c2) and self.rmap[c1] == self.rmap[c2]

    def extended_indiff(self, c1, c2):
        """Returns True  when either both ``c1`` and  ``c2`` are not ranked or the rank of ``c1`` is equals the rank of ``c2``.  
        
        """

        return (not self.is_ranked(c1) and not self.is_ranked(c2)) or (self.is_ranked(c1) and self.is_ranked(c2) and self.rmap[c1] == self.rmap[c2])

    def weak_pref(self, c1, c2):
        """Returns True if ``c1`` is weakly preferred to ``c2``. 

        The return value is True either ``c1`` is indifferent with ``c2`` or ``c1`` is strictly preferred to ``c2``.  
        """

        return self.strict_pref(c1,c2) or self.indiff(c1,c2)
    
    def extended_weak_pref(self, c1, c2):
        """Returns True when either ``c1`` and ``c2`` are extended indifferent or ``c1`` is extended strictly preferred to ``c2``.  
        """

        return self.extended_strict_pref(c1,c2) or self.extended_indiff(c1,c2)

    def is_linear(self):
        """Returns True if there are no ties."""

        return len(list(self.rmap.keys())) == len(list(self.rmap.values()))
        
    def remove_cand(self, a): 
        """Returns a Ranking with the candidate ``a`` removed. 
        """

        new_rmap = {c: self.rmap[c] for c in self.rmap.keys() if c != a}
        new_cmap = {c: self.cmap[c] for c in self.cmap.keys() if c != a}
        return Ranking(new_rmap, cmap = new_cmap)
    
    def first(self, cs = None):
        """Returns the list of candidates from ``cs`` that have the best ranking.   If ``cs`` is None, then use all the ranked candidates. 
        """

        _ranks = list(self.rmap.value()) if cs is None else [self.rmap[c] for c in cs]
        _cands = list(self.rmap.keys())  if cs is None else cs
        min_rank = min(_ranks)
        return sorted([c for c in _cands if self.rmap[c] == min_rank])

    def last(self, cs = None):
        """Returns the list of candidates from ``cs`` that have the worst ranking.   If ``cs`` is None, then use all the ranked candidates. 
        """

        _ranks = list(self.rmap.value()) if cs is None else [self.rmap[c] for c in cs]
        _cands = list(self.rmap.keys())  if cs is None else cs
        max_rank = max(_ranks)
        return sorted([c for c in _cands if self.rmap[c] == max_rank])

    def normalize_ranks(self):
        """Change the ranks so that they start with 1, and the next rank is the next integer after the previous rank. 
        
        :Example:

        .. exec_code:: python

            from pref_voting.profiles_with_ties import Ranking
            r = Ranking({0:1, 1:3, 2:2})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)
            
            r = Ranking({0:1, 1:10, 2:3})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

            r = Ranking({0:-100, 1:123, 2:0})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

            r = Ranking({0:10, 1:10, 2:100})
            print(r.rmap)
            r.normalize_ranks()
            print("After normalizing: ", r.rmap)

        """
        self.rmap = {c: self.ranks.index(r) + 1 for c,r in self.rmap.items()} 

    ## set preferences
    def AAdom(self, c1s, c2s, use_extended_preferences = False):  
        """
        Returns True if every candidate in ``c1s`` is weakly preferred to every candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended weak preference.
        """       
        
        weak_pref = self.extended_weak_pref if  use_extended_preferences else self.weak_pref

        return all([all([weak_pref(c1, c2) for c2 in c2s]) for c1 in c1s])
    
    def strong_dom(self, c1s, c2s, use_extended_preferences = False):   
        """
        Returns True if ``AAdom(c1s, c2s)`` and there is some candidate in ``c1s`` that is strictly preferred to every candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended  preferences.
        """       

        strict_pref = self.extended_strict_pref if  use_extended_preferences else self.strict_pref

        return self.AAdom(c1s, c2s,use_extended_preferences=use_extended_preferences ) and any([all([strict_pref(c1, c2) for c2 in c2s]) for c1 in c1s])

    def weak_dom(self, c1s, c2s, use_extended_preferences=False):         
        """
        Returns True if ``AAdom(c1s, c2s)`` and there is some candidate in ``c1s`` that is strictly preferred to some candidate in ``c2s``. If ``use_extended_preferences`` is True, then use the extended  preferences.
        """ 
             
        strict_pref = self.extended_strict_pref if  use_extended_preferences else self.strict_pref
                
        return self.AAdom(c1s, c2s, use_extended_preferences=use_extended_preferences) and any([any([strict_pref(c1, c2) for c2 in c2s]) for c1 in c1s])

    def __str__(self):
        """
        Display the ranking as a string. 
        """
        r_str = ''
        
        for r in self.ranks:
            cands_at_rank = self.cands_at_rank(r)
            if len(cands_at_rank) == 1:
                r_str += str(self.cmap[cands_at_rank[0]])
            else: 
                r_str += '(' + ''.join(map(lambda c: self.cmap[c], cands_at_rank)) + ')'
        return r_str


class ProfileWithTies(object):
    """An anonymous profile of (truncated) strict weak orders of :math:`n` candidates.  It is assumed that the candidates are named :math:`0, 1, \ldots, n-1` and a ranking of the candidates is a list of candidate names.  For instance, the list ``[0, 2, 1]`` represents the ranking in which :math:`0` is ranked above :math:`2`, :math:`2` is ranked above :math:`1`, and :math:`0` is ranked above :math:`1`.   

    :param rankings: List of rankings in the profile, where a ranking is either a :class:`Ranking` object or a dictionary.
    :type rankings: list[dict[int or str: int]] or list[Ranking]
    :param rcounts: List of the number of voters associated with each ranking.  Should be the same length as rankings.   If not provided, it is assumed that 1 voters submitted each element of ``rankings``.   
    :type rcounts: list[int], optional
    :param candidates: List of candidates in the profile.  If not provied, this is the list that is ranked by at least on voter. 
    :type candidates: list[int] or list[str], optional
    :param cmap: Dictionary mapping candidates (integers) to candidate names (strings).  If not provied, each candidate name is mapped to itself. 
    :type cmap: dict[int: str], optional

    :Example:
    
    The following code creates a profile in which 
    2 voters submitted the ranking 0 ranked first, 1 ranked second, and 2 ranked third; 3 voters submitted the ranking 1 and 2 are tied for first place and 0 is ranked second; and 1 voter submitted the ranking in which 2 is ranked first and 0 is ranked second: 

    .. code-block:: python

            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
    """

    def __init__(self, rankings, rcounts=None, candidates=None, cmap=None):
        """constructor method"""
        
        assert rcounts is None or len(rankings) == len(rcounts), "The number of rankings much be the same as the number of rcounts"
        
        self.candidates = sorted(candidates) if candidates is not None else sorted(list(set([c for r in rankings for c in r.keys()])))
        """The candidates in the profile. """

        self.num_cands = len(self.candidates)
        """The number of candidates in the profile."""
        
        self.ranks = list(range(1, self.num_cands + 1))

        # mapping candidates to candidate names
        self.cmap = cmap if cmap is not None else {c:c for c in self.candidates}
        
        self.rankings = [Ranking(r, cmap=self.cmap) if type(r) == dict else Ranking(r.rmap, cmap=self.cmap) for r in rankings]
        """The list of rankings in the Profile (each ranking is a :class:`Ranking` object). 
        """
        
        self.rcounts = [1]*len(rankings) if rcounts is None else list(rcounts) 
          
        self.num_voters = np.sum(self.rcounts)
        """The number of voters in the profile. """

        # memoize the supports
        self._supports = {c1: {c2: sum(n for r, n in zip(self.rankings, self.rcounts) if r.strict_pref(c1, c2)) 
                              for c2 in self.candidates}
                          for c1 in self.candidates}
    
    def use_extended_strict_preference(self): 
        """Redefine the supports so that extended strict preferences are used. Using extended strict preference may change the margins between candidates.
        """

        self._supports = {c1: {c2: sum(n for r, n in zip(self.rankings, self.rcounts) if r.extended_strict_pref(c1, c2)) 
                              for c2 in self.candidates}
                          for c1 in self.candidates}

    def use_strict_preference(self): 
        """Redefine the supports so that strict preferences are used. Using extended strict preference may change the margins between candidates.
        """
        
        self._supports = {c1: {c2: sum(n for r, n in zip(self.rankings, self.rcounts) if r.strict_pref(c1, c2)) 
                              for c2 in self.candidates}
                          for c1 in self.candidates}

    @property
    def rankings_counts(self):
        """Returns the rankings and the counts of each ranking."""

        return self.rankings, self.rcounts
    
    def support(self, c1, c2, use_extended_preferences = False):
        """Returns the support of candidate ``c1`` over candidate ``c2``, where the support is the number of voters that rank ``c1`` strictly above ``c2``."""
            
        return self._supports[c1][c2]
    
    def margin(self, c1, c2):
        """Returns the margin of candidate ``c1`` over candidate ``c2``, where the maring is the number of voters that rank ``c1`` strictly above ``c2`` minus the number of voters that rank ``c2`` strictly above ``c1``."""

        return self._supports[c1][c2] - self._supports[c2][c1]

    def ratio(self, c1, c2):
        """Returns the ratio of the support of ``c1`` over ``c2`` to the support ``c2`` over ``c1``. 
        """
        
        if self.support(c1, c2) > 0 and self.support(c2, c1) > 0: 
            return self.support(c1, c2) / self.support(c2, c1)
        elif self.support(c1, c2) > 0 and self.support(c2, c1) == 0:
            return float(self.num_voters + self.support(c1, c2))
        elif self.support(c1, c2) == 0 and self.support(c2, c1) > 0:
            return 1 / (self.num_voters + self.support(c2, c1))
        elif self.support(c1, c2) == 0 and self.support(c2, c1) == 0: 
            return 1

    def majority_prefers(self, c1, c2):
        """Returns True if ``c1`` is majority preferred to ``c2``.
        """

        return self.margin(c1, c2) > 0

    def condorcet_winner(self, curr_cands = None): 
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate. 
        """
        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cw = None
        for c in curr_cands: 

            if all([self.majority_prefers(c, c1) for c1 in curr_cands if c1 != c]): 
                cw = c
                break
        return cw

    def condorcet_loser(self, curr_cands = None):
        """Returns the Condorcet loser in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        A candidate :math:`c` is a  **Condorcet loser** if every other candidate  is majority preferred to :math:`c`.
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        cl = None
        for c1 in curr_cands: 
            if all([self.majority_prefers(c2, c1) for c2 in curr_cands if c1 != c2]): 
                cl = c1
                break # if a Condorcet loser exists, then it is unique
        return cl
    
    def weak_condorcet_winner(self, curr_cands = None):
        """Returns a list of the weak Condorcet winners in the profile restricted to ``curr_cands`` (which may be empty).

        A candidate :math:`c` is a  **weak Condorcet winner** if there is no other candidate that is majority preferred to :math:`c`.

        .. note:: While the Condorcet winner is unique if it exists, there may be multiple weak Condorcet winners.    
        """

        curr_cands = curr_cands if curr_cands is not None else self.candidates

        weak_cw = list()
        for c1 in curr_cands: 
            if not any([self.majority_prefers(c2,c1) for c2 in curr_cands if c1 != c2]): 
                weak_cw.append(c1)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def strict_maj_size(self):
        """Returns the strict majority of the number of voters.  
        """

        return int(self.num_voters/2 + 1 if self.num_voters % 2 == 0 else int(ceil(float(self.num_voters)/2)))

    def margin_graph(self): 
        """Returns the margin graph of the profile.  See :class:`.MarginGraph`.  

        :Example: 

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                mg = prof.margin_graph()
                print(mg.edges)
                print(mg.m_matrix)
        """
    
        return MarginGraph.from_profile(self)

    def support_graph(self): 
        """Returns the support graph of the profile.  See :class:`.SupportGraph`.  
        
        :Example: 

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                sg = prof.support_graph()
                print(sg.edges)
                print(sg.s_matrix)

        """
    
        return SupportGraph.from_profile(self)

    def majority_graph(self): 
        """Returns the majority graph of the profile.  See :class:`.MarginGraph`.

        :Example: 

        .. exec_code:: python

                from pref_voting.profiles_with_ties import ProfileWithTies
                prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])

                mg = prof.majority_graph()
                print(mg.edges)
  
        """
    
        return MajorityGraph.from_profile(self)

    def cycles(self): 
        """Return a list of the cycles in the profile.
        """

        return self.margin_graph().cycles()
        
    def remove_candidates(self, cands_to_ignore):
        """Remove all candidates from ``cands_to_ignore`` from the profile. 

        :param cands_to_ignore: list of candidates to remove from the profile
        :type cands_to_ignore: list[int]
        :returns: a profile with candidates from ``cands_to_ignore`` removed. 

        :Example: 

        .. exec_code::

            from pref_voting.profiles_with_ties import ProfileWithTies
            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
            prof.display()
            new_prof = prof.remove_candidates([1])
            new_prof.display()
            print(new_prof.ranks)
        """        
        
        updated_rankings = [{c:r for c,r in rank.rmap.items() if c not in cands_to_ignore} for rank in self.rankings]
        new_candidates = [c for c in self.candidates if c not in cands_to_ignore]
        return ProfileWithTies(updated_rankings, rcounts=self.rcounts, candidates=new_candidates, cmap=self.cmap)

    def display(self, cmap=None, style="pretty", curr_cands=None):
        """Display a profile (restricted to ``curr_cands``) as an ascii table (using tabulate).

        :param cmap: the candidate map to use (overrides the cmap associated with this profile)
        :type cmap: dict[int,str], optional
        :param style: the candidate map to use (overrides the cmap associated with this profile)
        :type style: str ---  "pretty" or "fancy_grid" (or any other style option for tabulate)
        :param curr_cands: list of candidates
        :type curr_cands: list[int], optional
        :rtype: None

        :Example: 

        .. exec_code::

            from pref_voting.profiles_with_ties import ProfileWithTies
            prof = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
            prof.display()
            prof.display(cmap={0:"a", 1:"b", 2:"c"})

        """

        _rankings = copy.deepcopy(self.rankings)
        _rankings = [r.normalize_ranks() or r for r in _rankings]
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        cmap = cmap if cmap is not None else self.cmap
        
        print(tabulate([[' '.join([str(cmap[c]) for c in r.cands_at_rank(rank) if c in curr_cands]) for r in _rankings] for rank in self.ranks], self.rcounts, tablefmt=style))    
        

    def display_margin_graph(self, cmap=None, curr_cands = None):
        """ 
        Display the margin graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.MarginGraph`. 
        """

        cmap = cmap if cmap is not None else self.cmap
        MarginGraph.from_profile(self, cmap=cmap).display(curr_cands = curr_cands)

    def display_support_graph(self, cmap=None, curr_cands = None):
        """ 
        Display the support graph of the profile (restricted to ``curr_cands``) using the ``cmap``.  See :class:`.SupportGraph`. 
        """

        cmap = cmap if cmap is not None else self.cmap
        SupportGraph.from_profile(self, cmap=cmap).display(curr_cands = curr_cands)


