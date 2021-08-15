import networkx as nx
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import random
from scipy.stats import rv_discrete
from src.data import config as cn
from src.data import utils as ut
import pandas as pd
import pickle
import time

## load directed collaboration network

# filepath = cn.datapath + "\\ca-CSphd\\ca-CSphd.txt"
# g = ut.get_graphtxt(filepath)
#
# g1 = nx.ego_graph(g, 500, undirected=True, radius=8)
# g2 = ut.get_node_renumbering(g1)
#
# nx.write_gpickle(g2, cn.datapath +"\\ca-CSphd\\ca-CSphd.gpickle")

## create random bipartite  BP graph

# def generate_RandomBP(nodesize, noofgraphs, prob=0.1):
#     Listgraph=[]
#     top = int(0.2*nodesize)
#     bottom = int(0.8*nodesize)
#
#     for countg in range(noofgraphs):
#
#         g = nx.algorithms.bipartite.random_graph(top, bottom, prob)
#         components = sorted(nx.connected_components(g), key=len, reverse=True)
#         largest_component = components[0]
#         C = g.subgraph(largest_component)
#         Listgraph.append(C)
#
#     return Listgraph
#
# Listgraph = generate_RandomBP(150,5,0.1)

## power law cluster graph

def generate_RandomPLC(nodesize, noofgraphs):
    Listgraph=[]

    for countg in range(noofgraphs):
        Listgraph.append(nx.generators.random_graphs.powerlaw_cluster_graph(nodesize, 2, 0.05))
    return Listgraph

Listgraph = generate_RandomPLC(4000,1)

def generate_RandomBA(nodesize, noofgraphs):

    Listgraph = []

    for countg in range(noofgraphs):
        Listgraph.append(nx.generators.random_graphs.barabasi_albert_graph(n=nodesize, m=2))

    return Listgraph

Listgraph = generate_RandomBA(1000,1)

def combine_graphs(graphlist):
    U = nx.disjoint_union_all(graphlist)
    return U

# combine graphs one disjoint union graph
g = combine_graphs(Listgraph)

for u,v,d in g.edges(data=True):
    d['weight'] = 0.5

# save edgelist
filepath = cn.datapath + "\\ca-CSphd\\g400test.txt"
nx.write_weighted_edgelist(g, filepath)

filepath = cn.datapath + "\\ca-CSphd\\g400BA.gpickle"
nx.write_gpickle(g, filepath)

# top = nx.bipartite.sets(C)[0]
# pos = nx.bipartite_layout(C, top)

## ==== generate node class targets l for listgraphs ====

## gen target labels for nodes

Stensor = get_probcandidatenodes(Listgraph)

with open(cn.datapath +"\\ca-CSphd" + "\\Listgraph_PLC_6000.pickle", 'wb') as b:
    pickle.dump(Listgraph, b)

with open(cn.datapath +"\\ca-CSphd" + "\\Combinedgraph_PLC_4000.pickle", 'wb') as b:
    pickle.dump(g, b)

with open(cn.datapath + "\\ca-CSphd" + "\\Stensor3_PLC_400.pickle", 'wb') as b:
    pickle.dump(Stensor, b)

# union of S
Ssetlist=[]

for countg in range(len(Listgraph)):
    Sset = [ set(ind) for ind in Stensor[countg]] # selets top 5 nodes from each sol set
    Ssetlist.append(list(set.union(*Sset)))

print("average length of Sset", len(Ssetlist[4]))

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

targetdf, Listlabel = getgraphtargetdf(Listgraph)

targetdf.drop(columns=['nodename'], inplace=True)

targetdftest, Listlabeltest = getgraphtargetdf(Listgraphtest)
targetdftest.drop(columns=['nodename'], inplace=True)

targetdf = pd.get_dummies(targetdf.label)
targetdftest = pd.get_dummies(targetdftest.label)

with open(cn.datapath + "\\ca-CSphd" + "\\Listlabel_PLC_6000.pickle", 'wb') as b:
    pickle.dump(Listlabel, b)

with open(cn.datapath + "\\ca-CSphd" + "\\targetdf_PLC_400test.pickle", 'wb') as b:
    pickle.dump(targetdf, b)

## ======= target dataframe =======

# def getgraphtargetdf(Listgraph, Listlabel):
#
#     finaldf = pd.DataFrame(columns=['nodename', 'label'])
#
#     for countsset in range(len(Listlabel)):
#         targetdf = pd.DataFrame()
#         nodelist = list(Listgraph[countsset].nodes)
#         targetdf['nodename'] = nodelist
#         targetdf['label'] = np.zeros((len(nodelist)), dtype=int)
#
#         for ind in Listlabel[countsset]:
#             targetdf.loc[targetdf['nodename'] == ind, 'label'] = 1
#
#         finaldf = pd.concat([finaldf, targetdf])
#
#     finaldf = finaldf.reset_index(drop=True)
#
#     return finaldf
#
# targetdf = getgraphtargetdf(Listgraph, Ssetlist)
#
# targetdf = pd.get_dummies(targetdf.label)
#
# with open(cn.datapath + "\\ca-CSphd" + "\\targetdf_PLC_6000.pickle", 'wb') as b:
#     pickle.dump(targetdf, b)

## feature vector

for node_id, node_data in g.nodes(data=True):
    node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]

## load already saved graph and labe;s

with open(cn.datapath +"\\ca-CSphd" + "\\Listgraph_PLC_6000.pickle", 'rb') as b:
    Listgraph = pickle.load(b)

with open(cn.datapath + "\\ca-CSphd" + "\\Listlabel_PLC_6000.pickle", 'rb') as b:
    Listlabel = pickle.load(b)

with open(cn.datapath + "\\ca-CSphd" + "\\targetdf_PLC_6000.pickle", 'rb') as b:
    targetdf1 = pickle.load(b)

with open(cn.datapath + "\\ca-CSphd" + "\\Stensor2_PLC_400.pickle", 'rb') as b:
    Stensor3 = pickle.load(b)

with open(cn.datapath +"\\ca-CSphd" + "\\Combinedgraph_PLC_4000.pickle", 'wb') as b:
    pickle.dump(g, b)