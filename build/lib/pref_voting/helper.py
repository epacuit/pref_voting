
from pref_voting.profiles import Profile
from pref_voting.weighted_majority_graphs import MajorityGraph
import networkx as nx

def get_mg(edata, curr_cands = None): 
    
    if curr_cands == None: 
        if type(edata) == Profile: 
            mg = MajorityGraph.from_profile(edata).mg
        else:
            mg = edata.mg
    else: 
        if type(edata) == Profile: 
            mg = nx.DiGraph()
            mg.add_nodes_from(curr_cands)
            mg.add_edges_from([(c1,c2) for c1 in curr_cands for c2 in curr_cands if edata.majority_prefers(c1, c2)])
        else:
            mg = edata.mg.copy()
            mg.remove_nodes_from([c for c in edata.candidates if c not in curr_cands])
    return mg
