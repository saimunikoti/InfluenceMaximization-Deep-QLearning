from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.manifold import TSNE

import numpy as np

from src.data import config as cnf
import networkx as nx
import matplotlib.pyplot as plt
from src.data import utils as ut
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

##### get performance metrics
def getacuracy(y_true, y_pred):
    cm = np.array([confusion_matrix(y_true[ind,:], y_pred[ind,:]) for ind in range(y_true.shape[0])])
    ac = np.array([accuracy_score(y_true[ind,:], y_pred[ind,:]) for ind in range(y_true.shape[0])])
    pr = np.array([precision_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    rc = np.array([recall_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    f1 = np.array([f1_score(y_true[ind,:], y_pred[ind,:],average='weighted') for ind in range(y_true.shape[0])])
    # pr = np.mean(pr)
    # rc = np.mean(pr)
    # f1 = np.mean(pr)
    # ac = np.mean(ac)
    
    return ac, pr,rc,f1

def plot_multiplegraphs(listgraph):
    graphscount = len(listgraph)
    fig1, ax = plt.subplots(nrows=2,ncols = 2,sharex=False, sharey=False, figsize=(8, 8))

    for countplot in range(graphscount):
        ix = np.unravel_index(countplot, ax.shape)
        plt.sca(ax[ix])
        nx.draw_networkx(listgraph[countplot], with_labels=True, node_color='coral')
        ax[ix].set_title(str(countplot), fontsize=10)

    plt.show()

### check the variation of graph in training and test data

def checkgraphvariation(xtrain, xtest):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()

    for i in range(4):
        train = np.reshape(xtrain[i+12],(xtest.shape[1],xtest.shape[1]))
        test = np.reshape(xtest[i+40],(xtest.shape[1],xtest.shape[1]))
        if i<2:
            Gtrain = nx.from_numpy_matrix(train)
            pos= nx.circular_layout(Gtrain)       
            nx.draw_networkx(Gtrain,pos, with_labels=True, node_color='lightgreen', ax=ax[i])
        else:
            Gtest = nx.from_numpy_matrix(test)
            pos = nx.circular_layout(Gtest)
            nx.draw_networkx(Gtest,pos, with_labels=True, node_color='peachpuff',ax=ax[i])
        ax[i].set_axis_off()

    plt.show()

#### plot the cicular layout grapoh
def plot_graph():
    g = nx.random_geometric_graph(22, 0.2)
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos, with_labels=True)

## visualize the comparison plots of degre  betweenness , egr
def visualize_corregr(g, egr):
    btw = []
    degree = []
    nodes = []

    for node, value in nx.betweenness_centrality(g).items():
        print(node, value)
        btw.append(value)
        degree.append(g.degree(node))
        nodes.append(node)
    plt.style.use('dark_background')
    plt.plot(nodes, degree, color="dodgerblue")
    plt.plot(nodes, btw, color="green")
    plt.plot(nodes, egr, color="coral")
    print(np.corrcoef(degree, egr))

def get_topnaccuracy(y_testdf, y_pred, margin):
    topindnodes_true = y_testdf[y_testdf['btw'] >= margin].index.values
    topindnodes_pr =  y_testdf[y_testdf['btw'] >= margin].index.values

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

def plot_base(tempax, xlabel, ylabel, figtitle):
    tempax.xaxis.set_tick_params(labelsize=22)
    tempax.yaxis.set_tick_params(labelsize=22)
    tempax.set_ylabel(ylabel, fontsize=28)
    tempax.set_xlabel(xlabel, fontsize=28)
    tempax.set_title(figtitle, fontsize=24)
    plt.grid(True)

def gen_rankresults(margin, graphsizelist, y_test, y_pred):

    result = np.zeros( ((len(graphsizelist)-1), 8))
    for countgraph in range(len(graphsizelist)-1):

       temp_ytest = y_test[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]
       temp_ypred = y_pred[sum(graphsizelist[0:countgraph+1]) : sum(graphsizelist[0:countgraph+2])]

       rank_test = np.array([1 if ind >= (1-margin)*np.max(temp_ytest) else 0 for ind in temp_ytest])
       rank_pred = np.array([1 if ind >= (1-margin)*np.max(temp_ypred) else 0 for ind in temp_ypred])
       # overall accuracy
       result[countgraph, 0] = accuracy_score(rank_test, rank_pred)

       try:
            result[countgraph, 1] = precision_score(rank_test, rank_pred)
            result[countgraph, 2] = recall_score(rank_test, rank_pred)
       except:
            print("precision not defined")

       ind = np.where(rank_test == 1)[0]
       # Top N accuracy
       result[countgraph, 3] = sum(rank_pred[ind]) / len(ind)

       rank_test = np.array([1 if ind <= margin*np.max(temp_ytest) else 0 for ind in temp_ytest])
       rank_pred = np.array([1 if ind <= margin*np.max(temp_ypred) else 0 for ind in temp_ypred])

       result[countgraph, 4] = accuracy_score(rank_test, rank_pred)
       try:
            result[countgraph, 5] = precision_score(rank_test, rank_pred)
            result[countgraph, 6] = recall_score(rank_test, rank_pred)
       except:
           print("precision not work")
       ind = np.where(rank_test == 1)[0]
       result[countgraph, 7] = sum(rank_pred[ind]) / len(ind)

    return result

def get_tsnevisualization(vector, ndim):
    X_embedded = TSNE(n_components=ndim).fit_transform(vector)
    return X_embedded

def plot_tsne_classwise(xembd, y_target):
    y_target = np.array(y_target)
    ytarget = np.array([np.where(y_target[ind]==1)[0] for ind in range(y_target.shape[0])])
    class0_indices = np.where(ytarget==0)[0]
    class1_indices = np.where(ytarget==1)[0]
    class2_indices = np.where(ytarget==2)[0]

    try:
        ax = plt.axes(projection='3d')
        ax.scatter3D(xembd[class0_indices,0], xembd[class0_indices,1], xembd[class0_indices,2], color="dodgerblue")
        ax.scatter3D(xembd[class1_indices,0], xembd[class1_indices,1], xembd[class1_indices,2], color="violet")
        ax.scatter3D(xembd[class2_indices,0], xembd[class2_indices,1], xembd[class2_indices,2], color="coral")
    except:
        plt.scatter(xembd[class0_indices,0], xembd[class0_indices,1], color="dodgerblue")
        plt.scatter(xembd[class1_indices,0], xembd[class1_indices,1], color="purple")
        plt.scatter(xembd[class2_indices,0], xembd[class2_indices,1], color="coral")

## Plot robustness metric
class new_resmetric():

    def __init__(self):
        print("metric class is invoked")

    #    def get_servicefactor(self, G):
    #        sumq = 0
    #
    #        for (node, val) in G.degree(weight='weight'):
    #            tempval = val/(G.nodes[node]['qo'])
    #            G.nodes[node]['qcurrent'] = tempval
    #
    #        for countn in G.nodes():
    #            sumq = sumq + (G.nodes[countn]['qcurrent'])/(G.nodes[countn]['crf'])
    #
    #        servicefac = sumq/(len(G.nodes))
    #
    #        return servicefac

    def get_egr_resistancedist(self, G):
        N = len(G.nodes)
        Rg = 0
        Gund = G.to_undirected()
        nodelist = list(Gund.nodes)
        for i in range(N):
            for j in range(i + 1, N):
                rab = nx.resistance_distance(Gund, nodelist[i], nodelist[j], weight='weight', invert_weight=True)
                Rg = Rg + rab

        return (N - 1) / Rg

    def get_weff(self, G):
        N = len(G)
        for u, v in G.edges:
            G[u][v]['newweight'] = 1 / (G[u][v]['weight'])
        sparray = np.zeros((N, N))
        nodelist = list(G.nodes)
        for i in range(N):
            for j in range(N):
                if i != j:
                    try:
                        sparray[i][j] = nx.shortest_path_length(G, nodelist[i], nodelist[j], weight='newweight')
                    except:
                        continue
        return np.mean(sparray)

    def get_weightedeff(self, G):
        tempsum = []
        tempsize = []
        splength = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G, weight='weight'))
        for key in splength.keys():
            tempsum.append(sum(splength[key].values()))
            tempsize.append(len(splength[key].values()) - 1)
        try:
            weff = sum(tempsum) / sum(tempsize)
        except:
            weff = 0
        return weff

    def network_criticality(self, G):
        n = len(G.nodes)
        Gcopy = G.copy()
        Gcopy = Gcopy.to_undirected()
        # eig = np.linalg.pinv(nx.directed_laplacian_matrix(G, weight='weight'))

        eig = np.linalg.pinv(nx.laplacian_matrix(Gcopy, weight='weight'))

        ncr = np.trace(eig)
        return (2 / (n - 1)) * ncr

    ###  normalized egr
    def get_egr(self, G):
        Gcopy = G.copy()
        Gcopy = Gcopy.to_undirected()

        eig = nx.linalg.spectrum.laplacian_spectrum(Gcopy, weight='weight')
        n = len(G.nodes)

        # Laplacian = nx.directed_laplacian_matrix(G, weight='weight') # diercted weighted laplacian
        # eig, v = la.eig(np.squeeze(np.asarray(Laplacian)))

        try:
            eig = [(1 / num) for num in eig[1:] if num != 0]
            egr = np.round(sum(np.abs(eig)), 3)
        except:
            print("zero encountered in Laplacian eigen values")

        Rg = (n - 1) / (n * egr)

        return np.round(Rg, 3)

        #### indegree robustness of tjhe graph - new designed metric

    def indegree_robust(self, G):

        for (node, val) in G.in_degree(weight='weight'):
            G.nodes[node]['indegcurnt'] = val

        sumindegree = 0
        for countn in G.nodes():
            try:
                sumindegree = sumindegree + (G.nodes[countn]['indegcurnt']) / (G.nodes[countn]['indegwt'])
            except:
                continue

        for node in G.nodes:
            orgindegreewtcount = G.nodes[node]['orgindgwtcount']
            break

        return sumindegree / orgindegreewtcount

    def component_robust(self, G, weightflag=True):
        for u in G.nodes():
            orgedgeweights = G.nodes[u]['orgedgewtcount']
            break
        try:
            tempcomp = [xind for xind in nx.weakly_connected_components(G)]
        except:
            tempcomp = [xind for xind in nx.connected_components(G)]
        sumedgecount = 0
        for countcomp in range(len(tempcomp)):
            H = G.subgraph(tempcomp[countcomp])
            if weightflag == True:
                edgecount = H.size(weight='weight')
            else:
                edgecount = H.size(weight=None)

            sumedgecount = sumedgecount + edgecount

        comprobustness = sumedgecount / orgedgeweights

        return comprobustness

    def get_servicefactor(self, G, weightflag="weight"):
        for (node, val) in G.degree(weight=weightflag):
            G.nodes[node]['qcurrent'] = val
        sumqnew = 0
        sumqorg = 0
        if weightflag == "weight":

            for countn in G.nodes():
                sumqnew = sumqnew + (G.nodes[countn]['qcurrent']) * (G.nodes[countn]['crf'])
                sumqorg = sumqorg + (G.nodes[countn]['qo']) * (G.nodes[countn]['crf'])
        else:
            for countn in G.nodes():
                sumqnew = sumqnew + (G.nodes[countn]['qcurrent']) * (G.nodes[countn]['crf'])
                sumqorg = sumqorg + (G.nodes[countn]['orgunwtdegree']) * (G.nodes[countn]['crf'])

        return (sumqnew / sumqorg)

    def get_edgerobustness(self, G, weightflag=True):

        for u in G.nodes():
            n = G.nodes[u]['orgnodecount']
            break
        # n= len(G.nodes)
        try:
            tempcomp = [xind for xind in nx.weakly_connected_components(G)]
        except:
            tempcomp = [xind for xind in nx.connected_components(G)]

        sumedgecount = 0
        for countcomp in range(len(tempcomp)):

            H = G.subgraph(tempcomp[countcomp])
            if weightflag == True:
                # edgecount = H.size(weight='weight')
                edgecount = len(H.nodes)
            else:
                # edgecount = H.size(weight=None)
                edgecount = len(H.nodes)
            sumedgecount = sumedgecount + edgecount * (edgecount - 1)

        edgerobustness = sumedgecount / (n * (n - 1))

        return edgerobustness

    def get_countstages(self, lcc, fr, idr):
        output = np.zeros((3, 3))
        for countmetric, metric in enumerate([lcc, fr, idr]):
            for countcol, residualratio in enumerate([0.8, 0.5, 0.2]):
                output[countmetric, countcol] = np.where(np.array(metric) < residualratio)[0][0]

        return output

    def get_percentfall_resmetric(self, lcc, efr, sf):

        restable = np.zeros((3, 3))
        pb = [int(0.2 * len(lcc)), int(0.6 * len(lcc)), int(1 * len(lcc))]
        startindex = [0, pb[0], pb[1]]
        endindex = [pb[0] - 1, pb[1] - 1, pb[2] - 1]

        for countmetric, metric in enumerate([lcc, efr, sf]):

            for countcol, (startind, endind) in enumerate(zip(startindex, endindex)):
                try:
                    restable[countmetric, countcol] = metric[startind] - metric[endind]  #
                    # restable[countmetric, countcol] = ((metric[startind]- metric[endind])/metric[startind])*100 # difference in percentage
                except:
                    restable[countmetric, countcol] = 0.00001
        return restable

    def plot_resmetric(self, lcc, ncc, sf, efr):

        xind = np.arange(0, len(lcc))

        def plot_base(tempax, y, ylabel, Colour):
            tempax.plot(xind, y, marker='o', color=Colour)
            tempax.xaxis.set_tick_params(labelsize=24)
            tempax.yaxis.set_tick_params(labelsize=24)

            tempax.set_ylabel(ylabel, fontsize=24)
            tempax.grid(True)

        fig1, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(8, 6))
        plot_base(ax[0], lcc, "LCC", "limegreen")
        plot_base(ax[1], ncc, "NCC", "slateblue")
        ax[1].set_xlabel("Percolation stage", fontsize=24)

        fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(8, 6))
        plot_base(ax[0], sf, "Service factor", "coral")
        plot_base(ax[1], efr, "Edge flow robustness", "dodgerblue")
        ax[1].set_xlabel("Percolation stage", fontsize=24)

    def plot2d_resmetric(self, lcc, ncc, sf, efr, titlename):

        xind = np.arange(0, len(lcc))
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0].plot(xind, lcc, marker='o', markersize=9, color="limegreen", label="LCC", linewidth=3)

        ax02 = ax[0].twinx()

        lns2 = ax02.plot(xind, ncc, marker='^', markersize=10, color="slateblue", label="NCC", linewidth=3)

        ax[0].set_ylabel("LCC", fontsize=26, color="limegreen", fontweight="bold")
        ax[0].tick_params(labelsize=22)

        ax02.set_ylabel("NCC", fontsize=26, color="slateblue", fontweight="bold")
        ax02.tick_params(labelsize=22)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc=0, fontsize=20)
        ax[0].grid(True)

        ################ subplo
        lns1 = ax[1].plot(xind, sf, marker='o', markersize=9, color="coral", label="SF", linewidth=3)

        ax12 = ax[1].twinx()

        lns2 = ax12.plot(xind, efr, marker='^', markersize=10, color="dodgerblue", label="EFR", linewidth=3)

        ax[1].set_ylabel("SF", fontsize=26, color="coral", fontweight="bold")
        ax[1].tick_params(labelsize=22)
        ax12.set_ylabel("EFR", fontsize=26, color="dodgerblue", fontweight="bold")
        ax12.tick_params(labelsize=22)
        ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1].set_xlabel('Percolation stage', fontsize=24)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc=0, fontsize=20)
        ax[1].grid(True)

        fig.tight_layout()
        plt.show()

    def plot4d_2compresmetric(self, lcccomp, lccpartial, ncccomp, nccpartial, efrcomp, efrpartial, idrcomp, idrpartial):
        xind = np.arange(0, len(lcccomp))

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0, 0].plot(lcccomp, marker='o', markersize=8, color="coral", label="LCC", linewidth=3)
        lns1 = ax[0, 0].plot(lccpartial, marker='^', markersize=8, color="dodgerblue", label="LCC", linewidth=3)
        ax[0, 0].set_ylabel("LCC", fontsize=26)
        ax[0, 0].tick_params(labelsize=22)
        ax[0, 0].set_xlabel('Percolation stage', fontsize=24)
        ax[0, 0].legend(['random', 'target'], fontsize=20)

        lns2 = ax[0, 1].plot(ncccomp, marker='o', markersize=8, color="coral", label="NCC", linewidth=3)
        lns2 = ax[0, 1].plot(nccpartial, marker='^', markersize=8, color="dodgerblue", label="NCC", linewidth=3)
        ax[0, 1].set_ylabel("NCC", fontsize=26)
        ax[0, 1].tick_params(labelsize=22)
        ax[0, 1].set_xlabel('Percolation stage', fontsize=24)
        ax[0, 1].legend(['random', 'target'], fontsize=20)

        lns3 = ax[1, 0].plot(efrcomp, marker='o', markersize=8, color="coral", label="egr", linewidth=3)
        lns3 = ax[1, 0].plot(efrpartial, marker='^', markersize=8, color="dodgerblue", label="egr", linewidth=3)
        ax[1, 0].set_ylabel("FR", fontsize=26)
        ax[1, 0].tick_params(labelsize=22)
        ax[1, 0].set_xlabel('Percolation stage', fontsize=24)
        ax[1, 0].legend(['random', 'target'], fontsize=20)

        lns4 = ax[1, 1].plot(idrcomp, marker='o', markersize=8, color="coral", label="idr", linewidth=3)
        lns4 = ax[1, 1].plot(idrpartial, marker='^', markersize=8, color="dodgerblue", label="idr", linewidth=3)
        ax[1, 1].set_ylabel("SR", fontsize=26)
        ax[1, 1].tick_params(labelsize=22)
        ax[1, 1].set_xlabel('Percolation stage', fontsize=24)
        ax[1, 1].legend(['random', 'target'], fontsize=20)

        ax[0, 0].grid(True)
        ax[0, 1].grid(True)
        ax[1, 0].grid(True)
        ax[1, 1].grid(True)

    def plot4d_resmetric(self, lcccomp, lccpartial, lccoutd, ncccomp, nccpartial, nccoutd, efrcomp, efrpartial, efroutd,
                         idrcomp, idrpartial, idroutd):
        xind = np.arange(0, len(lcccomp))

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0, 0].plot(lcccomp, marker='o', markersize=8, color="coral", label="LCC", linewidth=3)
        lns1 = ax[0, 0].plot(lccpartial, marker='^', markersize=8, color="dodgerblue", label="LCC", linewidth=3)
        lns1 = ax[0, 0].plot(lccoutd, marker='P', markersize=8, color="limegreen", label="LCC", linewidth=3)
        ax[0, 0].set_ylabel("LCC", fontsize=26)
        ax[0, 0].tick_params(labelsize=22)
        ax[0, 0].set_xlabel('Percolation stage', fontsize=24)
        ax[0, 0].legend(['random', 'betweenness', 'outdegree'], fontsize=20)

        lns2 = ax[0, 1].plot(ncccomp, marker='o', markersize=8, color="coral", label="NCC", linewidth=3)
        lns2 = ax[0, 1].plot(nccpartial, marker='^', markersize=8, color="dodgerblue", label="NCC", linewidth=3)
        lns2 = ax[0, 1].plot(nccoutd, marker='P', markersize=8, color="limegreen", label="NCC", linewidth=3)
        ax[0, 1].set_ylabel("NCC", fontsize=26)
        ax[0, 1].tick_params(labelsize=22)
        ax[0, 1].set_xlabel('Percolation stage', fontsize=24)
        ax[0, 1].legend(['random', 'betweenness', 'outdegree'], fontsize=20)

        lns3 = ax[1, 0].plot(efrcomp, marker='o', markersize=8, color="coral", label="egr", linewidth=3)
        lns3 = ax[1, 0].plot(efrpartial, marker='^', markersize=8, color="dodgerblue", label="egr", linewidth=3)
        lns3 = ax[1, 0].plot(efroutd, marker='P', markersize=8, color="limegreen", label="egr", linewidth=3)
        ax[1, 0].set_ylabel("FR", fontsize=26)
        ax[1, 0].tick_params(labelsize=22)
        ax[1, 0].set_xlabel('Percolation stage', fontsize=24)
        ax[1, 0].legend(['random', 'betweenness', 'outdegree'], fontsize=20)

        lns4 = ax[1, 1].plot(idrcomp, marker='o', markersize=8, color="coral", label="idr", linewidth=3)
        lns4 = ax[1, 1].plot(idrpartial, marker='^', markersize=8, color="dodgerblue", label="idr", linewidth=3)
        lns4 = ax[1, 1].plot(idroutd, marker='P', markersize=8, color="limegreen", label="idr", linewidth=3)
        ax[1, 1].set_ylabel("SR", fontsize=26)
        ax[1, 1].tick_params(labelsize=22)
        ax[1, 1].set_xlabel('Percolation stage', fontsize=24)
        ax[1, 1].legend(['random', 'betweenness', 'outdegree'], fontsize=20)

        ax[0, 0].grid(True)
        ax[0, 1].grid(True)
        ax[1, 0].grid(True)
        ax[1, 1].grid(True)

    def plot2d_resmetric_3plots(self, lcc, ncc, sf, sfwt, efr, efrwt, titlename, nplots=3):

        xind = np.arange(0, len(lcc))
        fig, ax = plt.subplots(nrows=nplots, ncols=1, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0].plot(xind, lcc, marker='o', markersize=9, color="limegreen", label="LCC")

        ax02 = ax[0].twinx()

        lns2 = ax02.plot(xind, ncc, marker='^', markersize=9, color="slateblue", label="NCC")

        ax[0].set_ylabel("LCC", fontsize=26, color="limegreen", fontweight='bold')
        ax[0].tick_params(labelsize=22)

        ax02.set_ylabel("NCC", fontsize=24, color="slateblue", fontweight='bold')
        ax02.tick_params(labelsize=22)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc=0, fontsize=20)
        ax[0].grid(True)

        #        ax[0].set_title(titlename, fontsize=20)

        lns1 = ax[1].plot(xind, sf, marker='o', markersize=9, color="coral", label="SF")

        ax12 = ax[1].twinx()

        lns2 = ax12.plot(xind, efr, marker='^', markersize=9, color="dodgerblue", label="EFR")

        ax[1].set_ylabel("SF", fontsize=24, color="coral", fontweight='bold')
        ax[1].tick_params(labelsize=22)
        ax12.set_ylabel("EFR", fontsize=24, color="dodgerblue", fontweight='bold')
        ax12.tick_params(labelsize=22)
        ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #        ax[1].set_xlabel('Percolation stage', fontsize=24)

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc=0, fontsize=20)
        ax[1].grid(True)

        ##################
        if nplots == 3:
            lns1 = ax[2].plot(xind, sfwt, marker='o', markersize=9, color="coral", label="SF")

            ax12 = ax[2].twinx()

            lns2 = ax12.plot(xind, efrwt, marker='^', markersize=9, color="dodgerblue", label="EFR")

            ax[2].set_ylabel("SFWt", fontsize=24, color="coral", fontweight='bold')
            ax[2].tick_params(labelsize=22)
            ax12.set_ylabel("EFRWt", fontsize=24, color="dodgerblue", fontweight='bold')
            ax12.tick_params(labelsize=22)
            ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax[2].legend(lns, labs, loc=0, fontsize=20)
            ax[2].grid(True)

        ax[-1].set_xlabel('Percolation stage', fontsize=24)
        fig.tight_layout()
        plt.show()

## plots for GraMeR

class plots_MGRL():

    def __init__(self):
        print("plot class is invoked")
        self.plcdf = pd.read_excel(cnf.modelpath + "\Resultsdf_plc.xlsx", engine='openpyxl')
        self.badf = pd.read_excel(cnf.modelpath + "\Resultsdf_ba.xlsx", engine='openpyxl')
        self.sbmdf = pd.read_excel(cnf.modelpath + "\Resultsdf_sbm.xlsx", engine='openpyxl', sheet_name="600_v2")
        self.budgetlist = [5,10,15,20]

    def plot_base(self, tempax, xind, y, xlabel, ylabel, Colour, markerlist, precision, yllimit, yulimit, marksize=15):

        tempax.plot(xind, y, color=Colour, marker= markerlist, markersize=marksize, linewidth=4)
        # tempax.set_ylim([0.5, 1.5])
        # tempax.set_ylim([-10, 400])
        tempax.set_ylim([yllimit, yulimit])
        # tempax.set_xticks([200, 600,1000,2000])
        tempax.xaxis.set_tick_params(labelsize=18)
        tempax.yaxis.set_tick_params(labelsize=18)
        tempax.yaxis.set_major_formatter(FormatStrFormatter(precision))

        tempax.set_xlabel(xlabel, fontsize=20)
        tempax.set_ylabel(ylabel, fontsize=20)

        tempax.grid(True)

    # prformance vs budget
    def plot_accuracy(self):

        budgetlist = self.budgetlist
        fig1, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(8, 6))

        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f',0.5,1.5, )
        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o",'%.1f',0.5,1.5)
        ax[0, 0].set_title('PLC', fontsize=20)

        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o",'%.1f', 0.5,1.5)

        ax[0, 1].set_title('BA', fontsize=20)

        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f', 0.5,1.5)
        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o", '%.1f', 0.5,1.5)

        ax[0, 2].set_title('SBM', fontsize=20)

        # ax[0,1].set_title("Spread vs Budget", fontsize=22)
        ax[0,2].legend(["MGHC","MS2V-DQN","GraMeR"], fontsize=15)

        # probability plot in second row
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o", '%.1f', 0.5,1.5)
        ax[1, 0].set_title('PLC', fontsize=20)

        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o",'%.1f',0.5,1.5)

        ax[1, 1].set_title('BA', fontsize=20)

        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o",'%.1f',0.5,1.5)

        ax[1, 2].set_title('SBM', fontsize=20)

        # ax[1,1].set_title("Mean intrinsic probability vs Budget", fontsize=21)
        ax[1,2].legend(["MGHC", "MS2V-DQN", "GraMeR"], fontsize=15)

        # fig1.suptitle("Spread/Intrinsic probability vs budget", fontsize=22)

    # running time vs budget
    def plot_time(self):

        budgetlist = self.budgetlist

        fig2, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(8, 5))

        self.plot_base(ax[0], budgetlist, self.plcdf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f', 0, 400)
        self.plot_base(ax[0], budgetlist, self.plcdf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', 0, 400, marksize=17)
        self.plot_base(ax[0], budgetlist, self.plcdf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',0, 400,)
        ax[0].set_title('PLC', fontsize=20)
        ax[0].set_yscale('log')

        self.plot_base(ax[1], budgetlist, self.badf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f',-5,6)
        self.plot_base(ax[1], budgetlist, self.badf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', -5, 6, marksize=17)
        self.plot_base(ax[1], budgetlist, self.badf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',-5,6)
        ax[1].set_title('BA', fontsize=20)
        ax[1].set_yscale('log')

        self.plot_base(ax[2], budgetlist, self.sbmdf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f',-5,6)
        self.plot_base(ax[2], budgetlist, self.sbmdf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', -5, 6, marksize=17)
        self.plot_base(ax[2], budgetlist, self.sbmdf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',-5,6)
        ax[2].set_title('SBM', fontsize=20)
        ax[2].set_yscale('log')

        # ax[1].set_title("Algorithms running time vs budget", fontsize=22)
        #
        ax[2].legend(["MGHC", "MS2V-DQN", "GraMeR"], fontsize=16)

    # running time & performance vs node size

    def plot_scalability(self):

        self.scalable_plcdf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="PLC")
        self.scalable_badf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="BA")
        self.scalable_sbmdf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="SBM")

        glist = [200,600,1000,2000]

        fig2, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(8, 5))

        self.plot_base(ax[0,0], glist, self.plcdf['norm_spread_mghc'], "Graph size (|V|)", "Influence spread", "dodgerblue", "s", '%.1f', 0.5,1.5)
        self.plot_base(ax[0,0], glist, self.plcdf['norm_spread_mgrl'], "Graph size (|V|)", "Influence spread", "coral", "o",'%.1f', 0.5,1.5)
        ax[0, 0].set_ylim([0.5, 1.5])
        ax[0, 0].set_title('PLC', fontsize=20)

        self.plot_base(ax[0,1], glist, self.badf['norm_spread_mghc'], "Graph size |V|", "Influence spread", "dodgerblue", "s",'%.1f',0.5,1.5)
        self.plot_base(ax[0,1], glist, self.badf['norm_spread_mgrl'], "Graph size |V|", "Influence spread", "coral", "o",'%.1f',0.5,1.5)
        ax[0, 1].set_ylim([0.5, 1.5])
        ax[0, 1].set_title('BA', fontsize=20)

        self.plot_base(ax[0,2], glist, self.sbmdf['norm_spread_mghc'], "Graph size |V|", "Influence spread", "dodgerblue", "s",'%.1f',0.5,1.5 )
        self.plot_base(ax[0,2], glist, self.sbmdf['norm_spread_mgrl'], "Graph size |V|", "Influence spread", "coral", "o", '%.1f', 0.5,1.5)
        ax[0, 2].set_ylim([0.5, 1.5])
        ax[0, 2].set_title('SBM', fontsize=20)

        # time vs graph size

        ax[0,2].legend(["MGHC", "GraMeR"], fontsize=16)

        self.plot_base(ax[1,0], glist, self.scalable_plcdf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,0], glist, self.scalable_plcdf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,0].set_title('PLC', fontsize=20)
        ax[1,0].set_yscale('log')

        self.plot_base(ax[1,1], glist, self.scalable_badf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,1], glist, self.scalable_badf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,1].set_title('BA', fontsize=20)
        ax[1, 1].set_yscale('log')

        self.plot_base(ax[1,2], glist, self.scalable_sbmdf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,2], glist, self.scalable_sbmdf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,2].set_title('SBM', fontsize=20)
        ax[1, 2].set_yscale('log')

        ax[1, 2].legend(["MGHC", "GraMeR"], fontsize=16)

    def plot_time_candwocand(self):

        self.candwocand_plcmgrldf = pd.read_excel(cnf.modelpath + "\Resultsdfablation_plc.xlsx", engine='openpyxl',
                                            sheet_name="comparison_mgrl")
        self.candwocand_plcmghcdf = pd.read_excel(cnf.modelpath + "\Resultsdfablation_plc.xlsx", engine='openpyxl',
                                            sheet_name="comparison_mghc")

        glist = [200, 600, 1000, 2000]

        fig2, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 5))

        # ax2 = ax.twinx()

        self.plot_base(ax[0], glist, self.candwocand_plcmghcdf['wocand']/60, "Graph size |V|", "Time (min)", "limegreen", "s", '%.1f',0,900)
        self.plot_base(ax[0], glist, self.candwocand_plcmghcdf['wcand']/60, "Graph size |V|", "Time (min)", "slateblue", "o", '%.1f',0,900)
        ax[0].set_title('MGHC', fontsize=21)

        self.plot_base(ax[1], glist, self.candwocand_plcmgrldf['wocand']/60, "Graph size |V|", "Time (min)", "coral", "s", '%.1f',0,7)
        self.plot_base(ax[1], glist, self.candwocand_plcmgrldf['wcand']/60, "Graph size |V|", "Time (min)", "dodgerblue", "o", '%.1f',0,7)
        ax[1].set_title('GraMeR', fontsize=21)

        ax[0].legend(["without_cand", "with_cand"], fontsize=16)
        ax[1].legend(["without_cand", "with_cand"], fontsize=16)


