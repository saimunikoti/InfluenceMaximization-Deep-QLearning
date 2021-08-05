from collections import deque
from src.models.dqnAgent import Agent
import numpy as np
import torch
from tqdm import tqdm
from src.models.models import genv
from src.data import config as cnf
import networkx as nx

## define environment

in_feats = 5
hid_feats = 64
budget = 5
g = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400test.gpickle")
Listgraph = [g]

candnodelist = list(g.nodes)

agent = Agent( Listgraph, in_feats, hid_feats, candnodelist, seed=0)
model = agent.qnetwork_local

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# define environment
genv = genv( Listgraph, candnodelist, budget)

## define main function for running algorithm

def dqn(checkpointpath, n_episodes= 100, eps_start=1.0, eps_end = 0.05, decay_step=0.0002, exp_replay_size=256):

    """Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        exp_replay_size: buffer size
    """

    # ====== initialoze experience replay buffer ======

    index = 0

    for i in range(exp_replay_size):

        state, candnodelist = genv.reset()
        done = False
        forbreak=0
        while not done:
            action = agent.act(state, candnodelist, eps=1)
            next_step, reward, done = genv.step(action)
            agent.save_buffer(state, action, reward, next_step, done)
            # agent.memory.memory
            state = next_step.copy()
            print(index)
            index += 1
            if index > exp_replay_size:
                forbreak=1
                break
        if forbreak==1:
                break

    #===== main loop for training ====

    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    scores_window_prev = deque(maxlen=100) # last 100 scores
    scores_window_prev.append(-9999) # large negative number for initializing previous start window

    eps = eps_start

    for i_episode in tqdm(range(1, n_episodes+1)):
        print("episode: ", i_episode)
        state, candnodelist = genv.reset()
        score = 0
        done = 0

        while not done:

            action = agent.act(state, candnodelist, eps)

            next_state, reward, done = genv.step(action)

            # print("state, action, reward, next_state", state, action, reward, next_state)

            agent.train(state, action, reward, next_state, done)

            state = next_state
            score += reward

            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score

            eps = max(eps - decay_step, eps_end)

            if np.mean(scores_window) > np.mean(scores_window_prev):
                torch.save(agent.qnetwork_local.state_dict(), checkpointpath)
                scores_window_prev = scores_window.copy()
                # print("model saved")

            if np.mean(scores_window)>=180.0:
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                           np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), checkpointpath)
                break

    return scores

checkpointpth = cnf.modelpath + "\\checkpoint_infmaxv2.pth"

scores = dqn(checkpointpath=checkpointpth)

