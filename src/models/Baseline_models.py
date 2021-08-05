import numpy as np
import time
import pickle
from src.data import config as cn
import networkx as nx
## define independent cascade model and hill climbing greedy algorithm

g = nx.read_gpickle(cn.datapath + "\\ca-CSphd\\g200.gpickle")

def IC(g, S, p=0.5, mc=500):

    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    # Loop over the Monte-Carlo Simulations
    spread = []

    for i in range(mc):
        print(i)
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

def greedy(g, k, candidatenodelist, p=0.5, mc=500):
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

            # Get the spread
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

s, spread, timelist = greedy(g, 10, candidatenodelist = candidatenodes)

snew, spreadnew, timelistnew = greedy(g, 5, candidatenodelist = candidatenodes)

with open(cn.datapath + "\\ca-CSphd\\S400_10nodes.pickle", 'wb') as b:
    pickle.dump(s, b)
with open(cn.datapath + "\\ca-CSphd\\Spread400_10nodes.pickle", 'wb') as b:
    pickle.dump(spread, b)
with open(cn.datapath + "\\ca-CSphd\\time400_10nodes.pickle", 'wb') as b:
        pickle.dump(timelist, b)

## centrality

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

def getclass1candnodes(g, ind):

indcopy = list(ind)
class1 = []

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

## plot times of computation
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,sharey=False,figsize=(8,6))
seed = np.arange(1,11)
ax1.plot(seed, timelistorg, '-o',color="coral",linewidth=3, markersize=8)
ax1.plot(seed, timelist, '-o',color="cornflowerblue", linewidth=3,  markersize=8)
ax1.set_xticks(np.arange(1,11,1))
plot_base(ax1, "seed","Execution time (s)", "Comparison of trivial Hill climbing greedy with proposed approach")
plt.tight_layout()

