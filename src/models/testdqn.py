
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

candnodelist = list(g.nodes)

agent = Agent( Listgraph, in_feats, hid_feats, candnodelist, seed=0)
checkpointpath = cnf.modelpath + "\checkpoint_infmaxv2.pth"
agent.qnetwork_local.load_state_dict(torch.load(checkpointpath))

# genv
genv = genv( Listgraph, candnodelist, budget)

##
# reward_arr = []
action_arr = []
spread_arr = []

for i in tqdm(range(5)):

    state, candnodelist = genv.knownreset(start_node=2)
    # state, candnodelist = genv.reset()
    done = False
    # rew = 0
    actionlist = []

    while not done:
        # A = agent.get_action(obs, env.action_space.n, epsilon=0)
        action = agent.act(state, candnodelist, eps=0)

        next_state, reward, done = genv.step(action)

        # rew += reward
        actionlist.append(action)

    # reward_arr.append(rew)
    action_arr.append(next_state)
    spread_arr.append(genv.spreadlist)

# print("average reward per episode :", sum(reward_arr) / len(reward_arr))

##

