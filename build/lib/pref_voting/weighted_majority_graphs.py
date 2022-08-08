'''
    File: weighted_majority_graphs.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 5, 2022
    
    Majority Graphs, Margin Graphs and Support Graphs
'''

import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
import string

class MajorityGraph(object):
    """An majority graph is an asymmetric directed graph.  The nodes are the candidates and an edge from candidate :math:`c` to :math:`d` means that :math:`c` is majority preferred to :math:`d`. 

    :param candidates: List of the candidates.  To be used as nodes in the majority graph.
    :type candidates: list[int] or  list[str]
    :param edges: List of the pairs of candidates describing the edges in the majority graph.   If :math:`(c,d)` is in the list of edges, then there is an edge from :math:`c` to :math:`d`. 
    :type edges: list
    :param cmap: Dictionary mapping candidates to candidate names (strings).  If not provied, each candidate name is mapped to itself. 
    :type cmap: dict[int: str], optional

    :Example:
    
    The following code creates a majority graph in which 0 is majority preferred to 1, 1 is majority preferred to 2, and 2 is majority preferred to 0: 

    .. code-block:: python

            mg = MajorityGraph([0, 1, 2], [(0,1), (1,2), (2,0)])
    
    .. warning:: Currently, there is no check to that the edges are asymteric.  It is assumed that the user provides an approriate set of edges.  
    """

    def __init__(self, candidates, edges, cmap = None): 
        """constructer method"""

        mg = nx.DiGraph()
        mg.add_nodes_from(candidates)
        mg.add_edges_from(edges)
        self.mg = mg
        """A networkx DiGraph object representing the majority graph."""

        self.cmap = cmap if cmap is not None else {c: str(c) for c in candidates}
        self.cindx = {c:cidx for cidx, c in enumerate(candidates)}
        self.candidates = candidates
        self.cindices = range(len(candidates))
        
        self.maj_matrix = [[False for c2 in self.cindices] for c1 in self.cindices]
        """A networkx DiGraph object representing the majority graph."""
        
        for c1 in self.cindices: 
            for c2 in self.cindices: 
                if mg.has_edge(c1, c2): 
                    self.maj_matrix[c1][c2] = True 
                    self.maj_matrix[c2][c1] = False
                elif mg.has_edge(c2, c1): 
                    self.maj_matrix[c2][c1] = True
                    self.maj_matrix[c1][c2] = False 
        
    def margin(self, c1, c2): 
        raise NotImplemented
        
    def support(self, c1, c2): 
        raise NotImplemented
        
    def ratio(self, c1, c2): 
        raise NotImplemented
    
    def majority_prefers(self, c1, c2): 
        """Returns true if there is an edge from `c1` to `c2`."""
        return self.mg.has_edge(c1, c2) 

    def is_tied(self, c1, c2): 
        """Returns true if there is no edge from `c1` to `c2` or from `c2` to `c1`."""
        return not self.mg.has_edge(c1, c2) and not self.mg.has_edge(c2, c1)
    
    def copeland_scores(self, curr_cands = None, scores = (1,0,-1)):
        """The Copeland scores in the profile restricted to the candidates in ``curr_cands``. 

        The **Copeland score** for candidate :math:`c` is calculated as follows:  :math:`c` receives ``scores[0]`` points for every candidate that  :math:`c` is majority preferred to, ``scores[1]`` points for every candidate that is tied with :math:`c`, and ``scores[2]`` points for every candidate that is majority preferred to :math:`c`. The default ``scores`` is ``(1, 0, -1)``. 
        

        :param curr_cands: restrict attention to candidates in this list. Defaults to all candidates in the profile if not provided. 
        :type curr_cands: list[int], optional
        :param scores: the scores used to calculate the Copeland score of a candidate :math:`c`: ``scores[0]`` is for the candidates that :math:`c` is majority preferred to; ``scores[1]`` is the number of candidates tied with :math:`c`; and ``scores[2]`` is the number of candidate majority preferred to :math:`c`.  The default value is ``scores = (1, 0, -1)`` 
        :type scores: tuple[int], optional
        :returns: a dictionary associating each candidate in ``curr_cands`` with its Copeland score. 

        """        
    
        wscore, tscore, lscore = scores
        candidates = self.candidates if curr_cands is None else curr_cands
        c_scores = {c: 0.0 for c in candidates}
        for c1 in candidates:
            for c2 in candidates:
                if self.majority_prefers(c1, c2):
                    c_scores[c1] += wscore
                elif self.majority_prefers(c2, c1):
                    c_scores[c1] += lscore
                elif c1 != c2: 
                    c_scores[c1] += tscore
        return c_scores

    def dominators(self, cand, curr_cands = None): 
        """Returns the list of candidates that are majority preferred to ``cand`` in the majority graph restricted to ``curr_cands``.
        """
        candidates = self.candidates if curr_cands is None else curr_cands
        
        return [c for c in candidates if self.majority_prefers(c, cand)]
    
    def condorcet_winner(self, curr_cands = None):
        """Returns the Condorcet winner in the profile restricted to ``curr_cands`` if one exists, otherwise return None.

        The **Condorcet winner** is the candidate that is majority preferred to every other candidate. 
        """
        
        curr_cands = curr_cands if curr_cands is not None else self.candidates
        
        cw = None
        for c1 in curr_cands: 
            if all([self.majority_prefers(c1,c2) for c2 in curr_cands if c1 != c2]): 
                cw = c1
                break # if a Condorcet winner exists, then it is unique
        return cw

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
    
    def has_cycle(self):
        """Returns True if the margin graph has a cycle.
        
        This uses the networkx method ``networkx.find_cycle`` to find the cycles in ``self.mg``.

        :Example: 

        .. exec_code::

            from pref_voting.weighted_majority_graphs import MajorityGraph 
            mg = MajorityGraph([0,1,2], [[(0,1), (1,2), (0,2)]])
            print(f"The majority graph has a cycle: {mg.has_cycle()}")
            mg = MajorityGraph([0,1,2], [[(0,1), (1,2), (2,0)]])
            print(f"The majority graph has a cycle: {mg.has_cycle()}")
            mg = MajorityGraph([0,1,2,3], [[(0,1), (3,0), (1,2), (3,1), (2,0), (3,2)]])
            print(f"The majority graph has a cycle: {mg.has_cycle()}")

        """                

        try:
            cycles =  nx.find_cycle(self.mg)
        except:
            cycles = list()
        return len(cycles) != 0

    def display(self, cmap=None, curr_cands = None):
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

            from pref_voting.profiles import Profile 
            prof = Profile([[0,1,2], [1,2,0], [2,0,1]], [2, 3, 1])
            prof.display()
            prof.display(cmap={0:"a", 1:"b", 2:"c"})

        """                
        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_edges_from([(cmap[c1], cmap[c2]) for c1, c2 in self.mg.edges if c1 in curr_cands and c2 in curr_cands])

        pos = nx.circular_layout(mg)

        nx.draw(mg, pos, 
                font_size=20, font_color='white', node_size=700, 
                width=1.5, with_labels=True)
        plt.show()

    def to_latex(self, cmap=None):

        if len(self.candidates) == 3: 
            return three_cand_tikz_str(self, cmap = cmap)
        elif len(self.candidates) == 4: 
            return four_cand_tikz_str(self, cmap = cmap)
        elif len(self.candidates) == 5: 
            return five_cand_tikz_str(self, cmap = cmap)
        else: 
            pos = nx.circular_layout(self.mg)
            return to_tikz_str(self, pos, cmap = cmap)
        
    @classmethod
    def from_profile(cls, profile, cmap=None):
        # generate a MajorityGraph from a Profile
        cmap = profile.cmap if cmap is None else cmap
        return cls(profile.candidates, 
                   [(c1, c2) for c1 in profile.candidates 
                    for c2 in profile.candidates if profile.majority_prefers(c1,c2)],
                  cmap=cmap)


class MarginGraph(MajorityGraph): 
    
    def __init__(self, candidates, w_edges, cmap = None):
        
        super().__init__(candidates, [(e[0], e[1]) for e in w_edges], cmap=cmap)
        
        self.m_matrix = [[0 for c2 in self.cindices] for c1 in self.cindices]
        for c1, c2, margin in w_edges: 
            self.m_matrix[self.cindx[c1]][self.cindx[c2]] = margin
            self.m_matrix[self.cindx[c2]][self.cindx[c1]] = -1 * margin
    
    def margin(self, c1, c2): 
        return self.m_matrix[self.cindx[c1]][self.cindx[c2]]
    
    def majority_prefers(self, c1, c2):
        return self.m_matrix[self.cindx[c1]][self.cindx[c2]] > 0

    def is_tied(self, c1, c2):
        return self.m_matrix[self.cindx[c1]][self.cindx[c2]] == 0

    def is_uniquely_weighted(self):
        # return true if all the margins are unique
        has_zero_margins = any([self.margin(c1, c2) == 0 for c1 in self.candidates for c2 in self.candidates if c1 != c2])
        return not has_zero_margins and len(list(set([self.margin(e[0], e[1]) for e in self.mg.edges ]))) == len(self.mg.edges)

    def display(self, curr_cands = None, cmap=None):
        # display the margin graph
        
        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands

        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_weighted_edges_from([(cmap[c1], cmap[c2], self.margin(c1,c2))
                                    for c1, c2 in self.mg.edges if c1 in curr_cands and c2 in curr_cands])

        pos = nx.circular_layout(mg)

        nx.draw(mg, pos, 
                font_size=20, font_color='white', node_size=700, 
                width=1.5, with_labels=True)
        labels = nx.get_edge_attributes(mg,'weight')
        nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)

        plt.show()
        
    @classmethod
    def from_profile(cls, profile, cmap=None):
        # generate a MarginGraph from a Profile
        cmap = profile.cmap if cmap is None else cmap
        return cls(profile.candidates, 
                   [(c1, c2, profile.margin(c1,c2)) for c1 in profile.candidates 
                    for c2 in profile.candidates if profile.majority_prefers(c1,c2)],
                  cmap=cmap)
        
class SupportGraph(MajorityGraph): 
    
    def __init__(self, candidates, w_edges, cmap = None):
        
        super().__init__(candidates, [(e[0], e[1]) if e[2][0] > e[2][1] else (e[1], e[0]) 
                                      for e in w_edges if e[2][0] != e[2][1]], cmap=cmap)
        
        self.s_matrix = [[0 for c2 in self.cindices] for c1 in self.cindices]
        
        for c1, c2, support in w_edges: 
            self.s_matrix[self.cindx[c1]][self.cindx[c2]] = support[0]
            self.s_matrix[self.cindx[c2]][self.cindx[c1]] = support[1]
    
    def margin(self, c1, c2): 
        return self.s_matrix[self.cindx[c1]][self.cindx[c2]] - self.s_matrix[self.cindx[c2]][self.cindx[c1]]

    def support(self, c1, c2): 
        return self.s_matrix[self.cindx[c1]][self.cindx[c2]]
    
    def majority_prefers(self, c1, c2):
        return self.s_matrix[self.cindx[c1]][self.cindx[c2]] > self.s_matrix[self.cindx[c2]][self.cindx[c1]]
    
    def is_tied(self, c1, c2):
        return self.s_matrix[self.cindx[c1]][self.cindx[c2]] == self.s_matrix[self.cindx[c2]][self.cindx[c1]]    
    
    def display(self, curr_cands = None, cmap=None):
        # display the margin graph
        
        cmap = cmap if cmap is not None else self.cmap
        curr_cands = self.candidates if curr_cands is None else curr_cands
        
        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in curr_cands])
        mg.add_weighted_edges_from([(cmap[c1], cmap[c2], self.support(c1,c2))
                                    for c1, c2 in self.mg.edges if c1 in curr_cands and c2 in curr_cands])

        pos = nx.circular_layout(mg)

        nx.draw(mg, pos, 
                font_size=20, font_color='white', node_size=700, 
                width=1.5, with_labels=True)
        labels = nx.get_edge_attributes(mg,'weight')
        nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)
   
        plt.show()
        
    @classmethod
    def from_profile(cls, profile, cmap=None):
        # generate a SupportGraph from a Profile
        
        cmap = profile.cmap if cmap is None else cmap
        return cls(profile.candidates, 
                   [(c1, c2, (profile.support(c1,c2), profile.support(c2,c1))) for c1 in profile.candidates 
                    for c2 in profile.candidates],
                  cmap=cmap)
        

###
# funcitons to display graphs in tikz 
##
def three_cand_tikz_str(g, cmap = None): 
    a = g.candidates[0]
    b = g.candidates[1]
    c = g.candidates[2]

    if type(g) == MarginGraph:
        w = lambda c,d : f'node[fill=white] {{${g.margin(c,d)}$}}'
    elif type(g) == SupportGraph: 
        w = lambda c,d : f'node[fill=white] {{${g.support(c,d)}$}}'
    else: 
        w = lambda c,d : ''
        
    cmap = g.cmap if cmap is None else cmap

    nodes = f'''
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (0,0) (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3,0) (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,1.5) (b) {{${cmap[b]}$}};\n'''
    
    if g.majority_prefers(a,b): 
        ab_edge = f'\\path[->,draw,thick] (a) to {w(a,b)} (b);\n'
    elif g.majority_prefers(b,a):
        ab_edge = f'\\path[->,draw,thick] (b) to {w(b,a)} (a);\n'
    else: 
        ab_edge = ''

    if g.majority_prefers(b,c): 
        bc_edge = f'\\path[->,draw,thick] (b) to {w(b,c)} (c);\n'
    elif g.majority_prefers(c,b):
        bc_edge = f'\\path[->,draw,thick] (c) to {w(c,b)} (b);\n'
    else: 
        bc_edge = ''

    if g.majority_prefers(a,c): 
        ac_edge = f'\\path[->,draw,thick] (a) to {w(a,c)} (c);\n'
    elif g.majority_prefers(c,a):
        ac_edge = f'\\path[->,draw,thick] (c) to {w(c,a)} (a);\n'
    else: 
        ac_edge = ''

    return nodes + ab_edge + bc_edge + ac_edge + '\\end{tikzpicture}'

def four_cand_tikz_str(g, cmap = None): 
    a = g.candidates[0]
    b = g.candidates[1]
    c = g.candidates[2]
    d = g.candidates[3]

    if type(g) == MarginGraph:
        w = lambda c,d : f'node[fill=white] {{${g.margin(c,d)}$}}'
    elif type(g) == SupportGraph: 
        w = lambda c,d : f'node[fill=white] {{${g.support(c,d)}$}}'
    else: 
        w = lambda c,d : ''
        
    cmap = g.cmap if cmap is None else cmap

    nodes = f'''
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (0,0)      (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3,0)      (b) {{${cmap[b]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,1.5)  (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (1.5,-1.5) (d) {{${cmap[d]}$}};\n'''   
 
    if g.majority_prefers(a,b): 
        ab_edge = f'\\path[->,draw,thick] (a) to[pos=.7] {w(a,b)} (b);\n'
    elif g.majority_prefers(b,a):
        ab_edge = f'\\path[->,draw,thick] (b) to[pos=.7] {w(b,a)} (a);\n'
    else: 
        ab_edge = ''

    if g.majority_prefers(a,c): 
        ac_edge = f'\\path[->,draw,thick] (a) to {w(a,c)} (c);\n'
    elif g.majority_prefers(c,a):
        ac_edge = f'\\path[->,draw,thick] (c) to {w(c,a)} (a);\n'
    else: 
        ac_edge = ''

    if g.majority_prefers(a,d): 
        ad_edge = f'\\path[->,draw,thick] (a) to {w(a,d)} (d);\n'
    elif g.majority_prefers(d,a):
        ad_edge = f'\\path[->,draw,thick] (d) to {w(d,a)} (a);\n'
    else: 
        ad_edge = ''

    if g.majority_prefers(b,c): 
        bc_edge = f'\\path[->,draw,thick] (b) to {w(b,c)} (c);\n'
    elif g.majority_prefers(c,b):
        bc_edge = f'\\path[->,draw,thick] (c) to {w(c,b)} (b);\n'
    else: 
        bc_edge = ''

    if g.majority_prefers(b,d): 
        bd_edge = f'\\path[->,draw,thick] (b) to {w(b,d)} (d);\n'
    elif g.majority_prefers(d,b):
        bd_edge = f'\\path[->,draw,thick] (d) to {w(d,b)} (b);\n'
    else: 
        bd_edge = ''

    if g.majority_prefers(c,d): 
        cd_edge = f'\\path[->,draw,thick] (c) to[pos=.7]  {w(c,d)} (d);\n'
    elif g.majority_prefers(d,c):
        cd_edge = f'\\path[->,draw,thick] (d) to[pos=.7]  {w(d,c)} (c);\n'
    else: 
        cd_edge = ''

    return nodes + ab_edge + ac_edge + ad_edge + bc_edge + bd_edge + cd_edge + '\\end{tikzpicture}'

def five_cand_tikz_str(g, cmap = None): 
    a = g.candidates[0]
    b = g.candidates[1]
    c = g.candidates[2]
    d = g.candidates[3]
    e = g.candidates[4]

    node_id = {
        a: 'a', 
        b: 'b',
        c: 'c', 
        d: 'd', 
        e: 'e'
    }
    if type(g) == MarginGraph:
        w = lambda c,d : f'node[fill=white] {{${g.margin(c,d)}$}}'
    elif type(g) == SupportGraph: 
        w = lambda c,d : f'node[fill=white] {{${g.support(c,d)}$}}'
    else: 
        w = lambda c,d : ''
        
    cmap = g.cmap if cmap is None else cmap

    nodes = f'''
\\begin{{tikzpicture}}
\\node[circle,draw,minimum width=0.25in] at (2,1.5)  (a) {{${cmap[a]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (0,1.5)  (b) {{${cmap[b]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (0,-1.5) (c) {{${cmap[c]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (2,-1.5) (d) {{${cmap[d]}$}}; 
\\node[circle,draw,minimum width=0.25in] at (3.5,0)  (e) {{${cmap[e]}$}};\n'''
    edges = [(a,b), (a,d), (a,e), (b,c), (c,d), (d,e)]
    edges_with_pos = [(a,c), (b, d), (b, e), (c, e)]
    
    edge_tikz_str = list()
    for c1, c2 in edges: 
        if g.majority_prefers(c1, c2): 
            edge_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c1]}) to {w(c1,c2)} ({node_id[c2]});\n')
        elif g.majority_prefers(c2, c1): 
            edge_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c2]}) to {w(c2,c1)} ({node_id[c1]});\n')
        else: 
            edge_tikz_str.append('')
    for c1, c2 in edges_with_pos: 
        if g.majority_prefers(c1, c2): 
            edge_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c1]}) to[pos=.7] {w(c1,c2)} ({node_id[c2]});\n')
        elif g.majority_prefers(c2, c1): 
            edge_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c2]}) to[pos=.7] {w(c2,c1)} ({node_id[c1]});\n')
        else: 
            edge_tikz_str.append('')

    return nodes + ''.join(edge_tikz_str)  + '\\end{tikzpicture}'

def to_tikz_str(g, pos, cmap = None): 
    
    node_id = {c:string.ascii_lowercase[cidx] for cidx,c in enumerate(g.candidates)}

    if type(g) == MarginGraph:
        w = lambda c,d : f'node[fill=white] {{${g.margin(c,d)}$}}'
    elif type(g) == SupportGraph: 
        w = lambda c,d : f'node[fill=white] {{${g.support(c,d)}$}}'
    else: 
        w = lambda c,d : ''
        
    cmap = g.cmap if cmap is None else cmap
    
    node_tikz_str = list()
    
    for c in g.candidates: 
        node_tikz_str.append(f'\\node[circle,draw,minimum width=0.25in] at ({float(2.4*list(pos[c])[0])},{float(2.6*list(pos[c])[1])})  ({node_id[c]}) {{${cmap[c]}$}};\n')

    edges_tikz_str = list()
    for c1 in g.candidates: 
        for c2 in g.candidates: 
            if g.majority_prefers(c1, c2): 
                edges_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c1]}) to[pos=.7] {w(c1,c2)} ({node_id[c2]});\n')
            elif g.majority_prefers(c2, c1): 
                edges_tikz_str.append(f'\\path[->,draw,thick] ({node_id[c2]}) to[pos=.7] {w(c2,c1)} ({node_id[c1]});\n')
            else: 
                edges_tikz_str.append('')

    return '\\begin{tikzpicture}\n' + ''.join(node_tikz_str) + ''.join(edges_tikz_str)  + '\\end{tikzpicture}'
