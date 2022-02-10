
from src.models.dqnAgent import Agent
import numpy as np
import torch
from tqdm import tqdm
from src.models.models import genv
from src.data import config as cnf
import networkx as nx
import matplotlib.pyplot as plt
import time
from src.data import utils as ut
import pandas as pd

##

g = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g600plcval.gpickle")
Listgraph = [g]

candnodelist = []
for countg in Listgraph:
    templist = [ nodeind for nodeind in countg.nodes if countg.nodes[nodeind]['alpha'] > 0.1]
    candnodelist.append(templist)

## ##============ spread with mgrl ========

in_feats = 5
hid_feats = 64
hid_mlp = 16
budget = 20

agent = Agent(Listgraph, in_feats, hid_feats, hid_mlp, candnodelist, seed=2, tuningweight=0.1)
checkpointpath1 = cnf.modelpath + "\checkpoint_AIM1_0.05r0.85_b20.pth"
checkpointpath2 = cnf.modelpath + "\checkpoint_AIM2_0.05r0.85_b20.pth"

agent.qnetwork_local1.load_state_dict(torch.load(checkpointpath1))
agent.qnetwork_local2.load_state_dict(torch.load(checkpointpath2))

# genv
genv = genv(Listgraph, candnodelist, budget, weighingfactor=0.1, intr_threshold=0.8)

#
action_arr = []
spread_arr = []

st_time = time.time()

for i in tqdm(range(1)):

    # state, candnodelist, gindex = genv.knownreset(start_node=908)
    state, candnodelist, gindex = genv.reset()
    done = False
    # rew = 0
    actionlist = []

    while not done:
        # A = agent.get_action(obs, env.action_space.n, epsilon
        action = agent.act_morl(state, candnodelist, gindex, eps=0)

        next_state, reward, done, reward1, reward2 = genv.step(action, gindex)

        # rew += reward
        actionlist.append(action)

    # reward_arr.append(rew)

    action_arr.append(next_state)

    spread_arr.append(genv.spreadlist)

    end_time_mgrl = time.time() - st_time

    # print("average reward per episode :", sum(reward_arr) / len(reward_arr))
    s_mgrl = action_arr[0]

    prob_mgrl = np.round(np.array([g.nodes[ind]['alpha'] for ind in s_mgrl]), 3)
    # print(prob)
    # print(np.mean(prob))

## === spread with mghc ====

candidatenodes = np.arange(len(g.nodes))
st_time = time.time()
s_mghc, spread, timelist = ut.aim_greedy(g, 20, candidatenodelist=candidatenodes)
end_time_mghc = time.time() - st_time
prob_mghc = np.round(np.array([g.nodes[ind]['alpha'] for ind in s_mghc ]), 3)

## combingin results in dataframe

Resultsdf = pd.DataFrame(columns=['budget', 'slist_mghc','spread_mghc', 'iprob_mghc', 'meaniprob_mghc', 'time_mghc', 'slist_mgrl', 'spread_mgrl', 'iprob_mgrl',
                 'meaniprob_mgrl', 'time_mgrl'])

for cind in range(4):

    lastind = 5*(cind+1)
    temps_mghc = s_mghc[0:lastind]
    temps_mgrl = s_mgrl[0:lastind]

    spread_mghc = ut.IC(g, temps_mghc, p=0.5, mc=1000)
    spread_mgrl = ut.IC(g, temps_mgrl, p=0.5, mc=1000)

    Resultsdf.at[cind,'budget'] = lastind
    Resultsdf.at[cind,'slist_mgrl'] = (temps_mgrl)
    Resultsdf.at[cind,'spread_mgrl'] = spread_mgrl
    Resultsdf.at[cind, 'iprob_mgrl'] = prob_mgrl[0:lastind]
    Resultsdf.at[cind,'meaniprob_mgrl'] = np.mean(prob_mgrl[0:lastind])
    Resultsdf.at[cind,'time_mgrl'] = (end_time_mgrl/20)*lastind + 0.05*lastind

    #========== get mghc baseline values ======

    Resultsdf.at[cind,'slist_mghc'] = temps_mghc
    Resultsdf.at[cind,'spread_mghc'] = spread_mghc
    Resultsdf.at[cind,'iprob_mghc'] = prob_mghc[0:lastind]
    Resultsdf.at[cind, 'meaniprob_mghc'] = np.mean(prob_mghc[0:lastind])
    Resultsdf.at[cind,'time_mghc'] = (end_time_mghc/20)*lastind + 0.05*lastind

filepath = cnf.modelpath + "\Resultsdf_plc1k.xlsx"
Resultsdf.to_excel(filepath, sheet_name="1k")

## plots definition

from matplotlib.ticker import FormatStrFormatter

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
        # tempax.set_ylim([yllimit, yulimit])
        # tempax.set_xticks([200, 600,1000,2000])
        tempax.xaxis.set_tick_params(labelsize=22)
        tempax.yaxis.set_tick_params(labelsize=22)
        tempax.yaxis.set_major_formatter(FormatStrFormatter(precision))

        tempax.set_xlabel(xlabel, fontsize=25)
        tempax.set_ylabel(ylabel, fontsize=25)

        tempax.grid(True)

    # prformance vs budget
    def plot_accuracy(self):

        budgetlist = self.budgetlist
        fig1, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(8, 6))

        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f',0.5,1.5, )
        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,0], budgetlist, self.plcdf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o",'%.1f',0.5,1.5)
        ax[0, 0].set_title('PLC', fontsize=24)

        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,1], budgetlist, self.badf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o",'%.1f', 0.5,1.5)

        ax[0, 1].set_title('BA', fontsize=24)

        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_mghc'], "Budget", "Influence spread", "dodgerblue", "s", '%.1f', 0.5,1.5)
        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_s2vdqn'], "Budget", "Influence spread", "sienna", "*",'%.1f',0.5,1.5)
        self.plot_base(ax[0,2], budgetlist, self.sbmdf['norm_spread_mgrl'], "Budget", "Influence spread", "coral", "o", '%.1f', 0.5,1.5)

        ax[0, 2].set_title('SBM', fontsize=24)

        # ax[0,1].set_title("Spread vs Budget", fontsize=22)
        ax[0,2].legend(["MGHC","MS2V-DQN","GraMeR"], fontsize=18)

        # probability plot in second row
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,0], budgetlist, self.plcdf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o", '%.1f', 0.5,1.5)
        ax[1, 0].set_title('PLC', fontsize=24)

        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,1], budgetlist, self.badf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o",'%.1f',0.5,1.5)

        ax[1, 1].set_title('BA', fontsize=24)

        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_mghc'], "Budget", "Mean intrinsic probability", "limegreen", "s", '%.1f',0.5,1.5)
        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_s2vdqn'], "Budget", "Mean intrinsic probability", "chocolate", "*", '%.1f', 0.5,1.5)
        self.plot_base(ax[1,2], budgetlist, self.sbmdf['meaniprob_mgrl'], "Budget", "Mean intrinsic probability", "slateblue", "o",'%.1f',0.5,1.5)

        ax[1, 2].set_title('SBM', fontsize=24)

        # ax[1,1].set_title("Mean intrinsic probability vs Budget", fontsize=21)
        ax[1,2].legend(["MGHC", "MS2V-DQN", "GraMeR"], fontsize=18)

        # fig1.suptitle("Spread/Intrinsic probability vs budget", fontsize=22)

    # running time vs budget
    def plot_time(self):

        budgetlist = self.budgetlist

        fig2, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(8, 5))

        self.plot_base(ax[0], budgetlist, self.plcdf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f', 0, 400)
        self.plot_base(ax[0], budgetlist, self.plcdf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', 0, 400, marksize=17)
        self.plot_base(ax[0], budgetlist, self.plcdf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',0, 400,)
        ax[0].set_title('PLC', fontsize=24)
        ax[0].set_yscale('log')

        self.plot_base(ax[1], budgetlist, self.badf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f',-5,6)
        self.plot_base(ax[1], budgetlist, self.badf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', -5, 6, marksize=17)
        self.plot_base(ax[1], budgetlist, self.badf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',-5,6)
        ax[1].set_title('BA', fontsize=24)
        ax[1].set_yscale('log')

        self.plot_base(ax[2], budgetlist, self.sbmdf['time_mghc']/60, "Budget", "Time (min), log scale", "coral", "s", '%.1f',-5,6)
        self.plot_base(ax[2], budgetlist, self.sbmdf['time_s2vdqn']/60, "Budget", "Time (min), log scale", "sienna", "*", '%.1f', -5, 6, marksize=17)
        self.plot_base(ax[2], budgetlist, self.sbmdf['time_mgrl']/60, "Budget", "Time (min), log scale", "slateblue", "o", '%.1f',-5,6)
        ax[2].set_title('SBM', fontsize=24)
        ax[2].set_yscale('log')

        # ax[1].set_title("Algorithms running time vs budget", fontsize=22)
        #
        ax[2].legend(["MGHC", "MS2V-DQN", "GraMeR"], fontsize=18)

    # running time & performance vs node size

    def plot_scalability(self):

        self.scalable_plcdf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="PLC")
        self.scalable_badf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="BA")
        self.scalable_sbmdf = pd.read_excel(cnf.modelpath + "\Resultsdf_scalability.xlsx", engine='openpyxl', sheet_name="SBM")

        glist = [200,600,1000,2000]

        fig2, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(8, 5))

        self.plot_base(ax[0,0], glist, self.plcdf['norm_spread_mghc'], "Graph size |V|", "Influence spread", "dodgerblue", "s", '%.1f', 0.5,1.5)
        self.plot_base(ax[0,0], glist, self.plcdf['norm_spread_mgrl'], "Graph size |V|", "Influence spread", "coral", "o",'%.1f', 0.5,1.5)
        ax[0, 0].set_ylim([0.5, 1.5])
        ax[0, 0].set_title('PLC', fontsize=24)

        self.plot_base(ax[0,1], glist, self.badf['norm_spread_mghc'], "Graph size |V|", "Influence spread", "dodgerblue", "s",'%.1f',0.5,1.5)
        self.plot_base(ax[0,1], glist, self.badf['norm_spread_mgrl'], "Graph size |V|", "Influence spread", "coral", "o",'%.1f',0.5,1.5)
        ax[0, 1].set_ylim([0.5, 1.5])
        ax[0, 1].set_title('BA', fontsize=24)

        self.plot_base(ax[0,2], glist, self.sbmdf['norm_spread_mghc'], "Graph size |V|", "Influence spread", "dodgerblue", "s",'%.1f',0.5,1.5 )
        self.plot_base(ax[0,2], glist, self.sbmdf['norm_spread_mgrl'], "Graph size |V|", "Influence spread", "coral", "o", '%.1f', 0.5,1.5)
        ax[0, 2].set_ylim([0.5, 1.5])
        ax[0, 2].set_title('SBM', fontsize=24)

        # time vs graph size

        ax[0,2].legend(["MGHC", "GraMeR"], fontsize=18)

        self.plot_base(ax[1,0], glist, self.scalable_plcdf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,0], glist, self.scalable_plcdf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,0].set_title('PLC', fontsize=24)
        ax[1,0].set_yscale('log')

        self.plot_base(ax[1,1], glist, self.scalable_badf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,1], glist, self.scalable_badf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,1].set_title('BA', fontsize=24)
        ax[1, 1].set_yscale('log')

        self.plot_base(ax[1,2], glist, self.scalable_sbmdf['time_mghc']/60, "Graph size |V|", "Time (min), log scale", "limegreen", "s", '%.1f',-100,1800)
        self.plot_base(ax[1,2], glist, self.scalable_sbmdf['time_mgrl']/60, "Graph size |V|", "Time (min), log scale", "slateblue", "o", '%.1f',-100,1800)
        ax[1,2].set_title('SBM', fontsize=24)
        ax[1, 2].set_yscale('log')

        ax[1, 2].legend(["MGHC", "GraMeR"], fontsize=18)

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
        ax[0].set_title('MGHC', fontsize=24)

        self.plot_base(ax[1], glist, self.candwocand_plcmgrldf['wocand']/60, "Graph size |V|", "Time (min)", "coral", "s", '%.1f',0,7)
        self.plot_base(ax[1], glist, self.candwocand_plcmgrldf['wcand']/60, "Graph size |V|", "Time (min)", "dodgerblue", "o", '%.1f',0,7)
        ax[1].set_title('GraMeR', fontsize=24)

        ax[0].legend(["without_cand", "with_cand"], fontsize=18)
        ax[1].legend(["without_cand", "with_cand"], fontsize=18)

vs = plots_MGRL()

vs.plot_accuracy()
plt.tight_layout(pad=2.5)

vs.plot_time()
plt.tight_layout(pad=2.5)

vs.plot_scalability()
plt.tight_layout(pad=2.5)

vs.plot_time_candwocand()
plt.tight_layout(pad=2.5)