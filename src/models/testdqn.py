
from src.models.dqnAgent import Agent
import numpy as np
import torch
from tqdm import tqdm
from src.models.models import genv
from src.data import config as cnf
import networkx as nx

##
in_feats = 5
hid_feats = 64
budget = 5
g = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g1ktest.gpickle")
Listgraph = [g]

candnodelist = [ list(gind.nodes) for gind in Listgraph ]

agent = Agent( Listgraph, in_feats, hid_feats, candnodelist, seed=0)
checkpointpath = cnf.modelpath + "\checkpoint_AIM_wtpdreward.pth"
agent.qnetwork_local.load_state_dict(torch.load(checkpointpath))

# genv
genv = genv( Listgraph, candnodelist, budget, weighingfactor=0.5)

##
# reward_arr = []

action_arr = []
spread_arr = []

for i in tqdm(range(2)):

    # state, candnodelist, gindex = genv.knownreset(start_node=49)
    state, candnodelist , gindex = genv.reset()
    done = False
    # rew = 0
    actionlist = []

    while not done:

        # A = agent.get_action(obs, env.action_space.n, epsilo
        action = agent.act(state, candnodelist,  gindex, eps=0)

        next_state, reward, done, reward1, reward2 = genv.step(action, gindex)

        # rew += reward
        actionlist.append(action)

    # reward_arr.append(rew)

    action_arr.append(next_state)

    spread_arr.append(genv.spreadlist)

# print("average reward per episode :", sum(reward_arr) / len(reward_arr))
s = action_arr[0]
prob = np.round(np.array([g.nodes[ind]['alpha'] for ind in s ]), 3)
print(prob)
print(np.mean(prob))

##

