import gym

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
g = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400test.gpickle")
Listgraph = [g]

candnodelist = list(g.nodes)

agent = Agent( Listgraph, in_feats, hid_feats, candnodelist, seed=0)
checkpointpath = r"C:\Users\saimunikoti\Manifestation\InfluenceMaximization_DRL\models\checkpoint_infmax.pth"
agent.qnetwork_local.load_state_dict(torch.load(checkpointpath))

# genv
genv = genv( Listgraph, candnodelist, budget)

##
reward_arr = []
for i in tqdm(range(100)):
    state, candnodelist = genv.reset()
    done = False
    rew = 0

    while not done:
        # A = agent.get_action(obs, env.action_space.n, epsilon=0)
        action = agent.act(state, candnodelist, eps=0)

        next_state, reward, done = genv.step(action)

        rew += reward

    reward_arr.append(rew)

print("average reward per episode :", sum(reward_arr) / len(reward_arr))

##

