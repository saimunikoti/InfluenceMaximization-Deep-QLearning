
import networkx as nx
from src.data import utils as ut
from src.data import config as cnf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

def generate_RandomPLC(nodesize, noofgraphs):
    Listgraph=[]

    for countg in range(noofgraphs):
        Listgraph.append(nx.generators.random_graphs.powerlaw_cluster_graph(nodesize, 3, 0.05))
    return Listgraph

Listgraph = generate_RandomPLC(1000,1)

def generate_RandomBA(nodesize, noofgraphs):

    Listgraph = []

    for countg in range(noofgraphs):
        Listgraph.append(nx.generators.random_graphs.barabasi_albert_graph(n=nodesize, m=2))

    return Listgraph

Listgraph = generate_RandomBA(1000,1)

# def generate_RandomER(nodesize, noofgraphs):
#
#     Listgraph = []
#
#     for countg in range(noofgraphs):
#         Listgraph.append(nx.generators.random_graphs.erdos_renyi_graph(n=nodesize, p=0.25))
#
#     return Listgraph

# Listgraph = generate_RandomER(200,1)

# def generate_RandomLFR(nodesize, noofgraphs):
#
#     Listgraph = []
#
#     for countg in range(noofgraphs):
#         Listgraph.append(nx.generators.community.LFR_benchmark_graph(
#         n=nodesize, tau1=3, tau2=1.5, mu=0.5, average_degree=15, min_community=30, seed=10
#         ))
#
#     return Listgraph

# Listgraph = generate_RandomLFR(60,1)

def generate_SBM(sizes, noofgraphs):

    Listgraph = []
    dim = len(sizes)
    # sizes = [55, 50, 50, 20, 25]
    # probs = [[0.20, 0.03, 0.02, 0.02, 0.01], [0.03, 0.22, 0.05, 0.02, 0.01], [0.02, 0.05, 0.20, 0.01, 0.02],
    #          [0.02, 0.02, 0.01, 0.21, 0.02], [0.01, 0.01, 0.02, 0.02, 0.23]]
    probs = np.zeros((dim,dim))

    for ci in range(0,dim):
        for cj in range(ci, dim):
            if ci == cj:
                probs[ci,cj] = np.random.choice([0.20, 0.22, 0.23, 0.22, 0.2, 0.24, 0.22, 0.23, 0.21, 0.2])
            else:
                probs[ci,cj] = np.random.choice([0.01,0.02,0.03,0.04,0.05])
                probs[cj,ci] = probs[ci,cj]

    for countg in range(noofgraphs):
        Listgraph.append(nx.stochastic_block_model(sizes, probs))

    return Listgraph

sizelist = [75, 70, 70, 40, 45, 75, 70, 70, 40, 45 ]
sizelist = [55, 55, 55, 20, 25, 55, 55, 55, 20, 25]
sizelist = [110, 130, 120, 120, 120]

Listgraph = generate_SBM(sizes = sizelist, noofgraphs= 1)

g = Listgraph[0]

# weighted graph for influence prob
for u,v,d in g.edges(data=True):
    d['weight'] = 0.5

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    # node_data["alpha"] = np.random.uniform(0, 1)

# get random subgraphs
Listrandomgraph = ut.get_randomsubgraph(g,100)

## make data based on IFC centrality scores

Imutils = ut.IMutil()

def getgraphtargetdf(Listgraph):
    Listlabel = []
    finaldf = pd.DataFrame(columns=['nodename', 'label'])

    for countg in range(len(Listgraph)):

        icscore = Imutils.get_inflcapapcity(Listgraph[countg], 0.5)
        ind = np.argsort(icscore, axis=0)

        # Method of top-k %
        class1 = ind[0:int(0.25 * len(ind))]
        class2 = ind[int(0.25 * len(ind)): int(0.65 * len(ind))]
        class3 = ind[int(0.65 * len(ind)): int(1.0 * len(ind))]

        tempdict = {}
        tempdict['class1'] = class1
        tempdict['class2'] = class2
        tempdict['class3'] = class3

        Listlabel.append(tempdict)

        # targetdf
        targetdf = pd.DataFrame()
        nodelist = list(Listgraph[countg].nodes)
        targetdf['nodename'] = nodelist
        targetdf['label'] = np.zeros((len(nodelist)), dtype=int)

        for ind in class2[:, 0]:
            targetdf.loc[targetdf['nodename'] == ind, 'label'] = 1

        # for ind in class3[:, 0]:
        #     targetdf.loc[targetdf['nodename'] == ind, 'label'] = 2

        finaldf = pd.concat([finaldf, targetdf])
        print("count", countg )

    finaldf = finaldf.reset_index(drop=True)

    return finaldf, Listlabel

Listgraph = [g]

targetdf, Listlabel = getgraphtargetdf(Listgraph)

targetdf.drop(columns=['nodename'], inplace=True)

## assign node label and intrinsic probabilityto gobs

for node_id, node_data in g.nodes(data=True):
    node_data["label"] = list(targetdf.loc[node_id])
    node_data["alpha"] = np.random.uniform(0,1)

##
filepath = cnf.datapath + "\\ca-CSphd\\g600sbmval2.gpickle"

nx.write_gpickle(g, filepath, protocol=4)

filepath = cnf.datapath + "\\ca-CSphd\\g600sbmval.txt"

nx.write_weighted_edgelist(g, filepath)

