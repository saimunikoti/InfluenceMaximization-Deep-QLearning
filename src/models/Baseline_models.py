import numpy as np
import time
import pickle
from src.data import config as cn
from src.data import utils as ut
from src.visualization import visualize as vs
import networkx as nx

## original GHC algorithm.
"""
The results of spread coincides with influence capacity metric
"""

def IC(g, S, p=0.5, mc=200):

    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    # Loop over the Monte-Carlo Simulations
    spread = []

    for i in range(mc):
        # print(i)
        # Simulate propagation process
        new_active, A = S[:], S[:]

        while new_active :

            # For each newly active node, find its neighbors that become activated
            new_ones = []

            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                outn = [n for n in g.neighbors(node)]
                success = np.random.uniform(0, 1, len(outn)) < p
                new_ones += list(np.extract(success, outn))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        spread.append(len(A))

    return np.mean(spread)

def greedy(g, k, candidatenodelist, p=0.5, mc=200):

    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    # S, spread, timelapse = [], [], []

    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0

        # candidatenodelist = range(len(g.nodes))

        for j in set(candidatenodelist) - set(S):

            # Get the expected spread
            s = IC(g, S + [j], p, mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
        print("k", _)

    return S, spread, timelapse

candidatenodes = np.arange(len(g.nodes))

st_time = time.time()
s, spread, timelist = greedy(g, 5, candidatenodelist = candidatenodes)
end_time = time.time()-st_time

## define modified independent cascade model and hill climbing greedy algorithm

g = nx.read_gpickle(cn.datapath + "\\ca-CSphd\\g200plcval.gpickle")

def mIC(g, S, p=0.5,mc=500):

    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    # Loop over the Monte-Carlo Simulations
    spread = []

    new_active, A = S[:], S[:]

    while new_active:

        # For each newly active node, find its neighbors that become activated
        new_ones = []

        for node in new_active:
            # Determine neighbors that become infected
            # np.random.seed(1)
            outn = [n for n in g.neighbors(node)]
            success = np.random.uniform(0, 1, len(outn)) < p
            new_ones += list(np.extract(success, outn))

        new_active = list(set(new_ones) - set(A))

        # Add newly activated nodes to the set of activated nodes
        A += new_active

    spread.append(len(A))

    return np.mean(spread)

def aim_greedy(g, k, candidatenodelist, p=0.5, mc=200):

    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    # S, spread, timelapse = [], [], []

    # Find k nodes with largest marginal gain
    for countb in range(k):
        # print("node", countb)
        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0

        for j in set(candidatenodelist) - set(S):
            s = []
            for count in range(mc):

                np.random.seed(count)

                if np.random.uniform(0, 1) < g.nodes[j]['alpha']:

                    # Get the spread
                    s.append( mIC(g, S + [j], p=0.5, mc=100))

            # Update the winning node and spread so far
            if np.sum(s) >= best_spread:
                best_spread = np.sum(s)
                best_node = j

        # Add the selected node to the seed set
        S.append(best_node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
        print("k", countb)

    return S, spread, timelapse

candidatenodes = np.arange(len(g.nodes))

#==== WEIGHT NEW NETWORK ===

# for u,d in g.nodes(data=True):
#     d['alpha'] = np.random.uniform(0, 1)
st_time = time.time()
s, spread, timelist_all = aim_greedy(g, 10, candidatenodelist = candidatenodes)
end_time = time.time()-st_time

st_time = time.time()
spread = IC(g, s, p = 0.5, mc=200)
end_time = time.time()-st_time

## save files

snew, spreadnew, timelistnew = greedy(g, 10, candidatenodelist = candidatenodes)

with open(cn.datapath + "\\ca-CSphd\\S2k_10nodes.pickle", 'wb') as b:
    pickle.dump(s, b)
with open(cn.datapath + "\\ca-CSphd\\Spread2k_10nodes.pickle", 'wb') as b:
    pickle.dump(spread, b)
with open(cn.datapath + "\\ca-CSphd\\time2k_10nodes.pickle", 'wb') as b:
    pickle.dump(timelist, b)

## infleunce capapcity centrality

for u,v,d in g.edges(data=True):
    d['weight'] = 0.5

nx.write_weighted_edgelist(g, filepath)

def get_inflcapapcity( g, uniinfweight):
    nodelist = list(g.nodes)
    il = np.zeros((len(nodelist), 1))
    ig = np.zeros((len(nodelist), 1))

    degn = max([nx.degree(g, ind) for ind in g.nodes])

    for countnode in range(len(nodelist)):
        tempw = 0
        for neighbnode in g.neighbors(nodelist[countnode]):
            tempw = tempw + uniinfweight * uniinfweight * nx.degree(g, neighbnode)

        # local score
        il[nodelist[countnode]] = 1 + list(g.degree([nodelist[countnode]], weight='weight'))[0][1] + tempw

        # global score
        ig[nodelist[countnode]] = nx.core_number(g)[nodelist[countnode]] * (
                1 + (nx.degree(g, nodelist[countnode])) / (degn))

    # overall score
    ic = np.array([(il[nodelist[countnode]] / np.max(il)) * (ig[nodelist[countnode]] / np.max(ig)) for countnode in
                   range(len(nodelist))])

    return ic

icscore = get_inflcapapcity(g, 0.5)

ind = np.argsort(icscore, axis=0)

# method of scores
class1 = np.where(icscore<=0.05*1.0)[0]
class2 = np.where( (icscore>0.05*1.0) & (icscore<0.1))[0]
class3 = np.where(icscore>=0.10*1.0)[0]

# Method of top-k %
class1 = ind[0:int(0.25*len(ind)), 0]
class2 = ind[int(0.25*len(ind)): int(0.65*len(ind)), 0]
class3 = ind[int(0.65*len(ind)): int(1.0*len(ind)), 0]
class0 = np.append(class1, class3)

## top-k% based on common neighbors

ranked = np.argsort(icscore, axis=0)
n = len(icscore)
ind = ranked[::-1][:n]
ind = ind[:,0]

# def getclass1candnodes(g, ind):
#
# indcopy = list(ind)
# class1 = []

# while len(class1)<= int(0.3*n) :

for _ in range(50) :
    u = indcopy[0]
    for neighnode in nx.neighbors(g,u):
        if len(list(nx.common_neighbors(g, u, neighnode)))> 1:
            try:
                indcopy.remove(neighnode)
                print("n r")
            except:
                pass

    class1.append(u)
    indcopy.remove(u)

## load seed nodes from ripples

filepath = r"C:\Users\saimunikoti\Manifestation\InfluenceMaximization_DRL\ripples-master\outputPLC20k.json"

with open(filepath) as f:
  data = json.load(f)

data = data[0]
IMseednodes= data['Seeds']

## plot comparison of computation times

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,sharey=False,figsize=(8,6))
seed = np.arange(1,11)
ax1.plot(seed, timelist_org, '-o', color="coral",linewidth=4, markersize=9)
ax1.plot(seed, timelist_new, '-o', color="cornflowerblue", linewidth=4,  markersize=9)
ax1.set_xticks(np.arange(1, 11, 1))
vs.plot_base(ax1, "seed nodes","Execution time (s)", " ")
ax1.legend(['Conventional GHC', 'Proposed GHC'], fontsize=26)
plt.tight_layout()

def IC(g, S, p=0.5, mc=500):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S[:], S[:]

        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                outn = [n for n in g.neighbors(node)]
                success = np.random.uniform(0, 1, len(outn)) < p
                new_ones += list(np.extract(success, outn))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        spread.append(len(A))

    return np.mean(spread)

def get_random_fromdist(nodelist, p):
    # choose node woth probability
    random_variable = rv_discrete(values=(nodelist, p))
    temp = random_variable.rvs(size=10)[1]
    return temp

def get_newprob(g, S, nodeavailable):

    nodegain = []
    # compute prob for each node
    if len(S)!=0:
        for node in nodeavailable:
            Stemp = S.copy()
            Stemp.append(node)
            temp = IC(g, Stemp) - IC(g, S)
            nodegain.append(temp)
    else:
        for node in nodeavailable:
            Stemp = S.copy()
            Stemp.append(node)
            temp = IC(g, Stemp)
            nodegain.append(temp)

    den = sum(nodegain)
    p = np.array([number / den for number in nodegain])

    return p

def get_probgreedy(g, tolerance=0.5):
    nodelist = list(g.nodes)
    gain = 10
    S = []
    gainlist = []
    while gain > tolerance:
        nodeavailable = [item for item in nodelist if item not in S]

        p = get_newprob(g, S, nodeavailable)

        # nodeselected = get_random_fromdist(nodeavailable, p)
        temp = random.choices(nodeavailable, p, k=100)
        u, count = np.unique(temp, return_counts=True)
        count_sort_ind = np.argsort(-count)
        nodeselected = u[count_sort_ind][0]

        Snew = S.copy()
        Snew.append(nodeselected)

        # gain = p[np.where(np.array(nodelist) == nodeselected)[0][0]]
        gain = IC(g,Snew) - IC(g,S)
        gainlist.append(gain)
        S = Snew.copy()
        print("iter gain: ", gain)
        # gain2 = gain2-0.001
    return S

def get_probcandidatenodes(Listgraph):

    Stensor = []

    for countg in range(len(Listgraph)):
        Slist=[]
        for count in range(10):
            temp = get_probgreedy(Listgraph[countg])
            Slist.append(temp)
            print("=== count== ", countg, count)
        Stensor.append(Slist)

    return Stensor

## activtion informed IM with greedy HC

def ainf_greedy(g, k, candidatenodelist, Listgraph, p=0.5, mc=500):

    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    # S, spread, timelapse = [], [], []

    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0

        for j in set(candidatenodelist) - set(S):
            s = 0
            for countg in Listgraph:

                if np.random.uniform(0, 1) > countg.nodes[j]['alpha']:

                    # Get the spread
                    s = s + IC(g, S + [j], p, mc=100)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread = s
                best_node = j

        # Add the selected node to the seed set
        S.append(best_node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
        print("k", _)

    return S, spread, timelapse


