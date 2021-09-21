from collections import deque
from src.models.dqnAgent import Agent
import numpy as np
import torch
from tqdm import tqdm
from src.models.models import genv
from src.data import config as cnf
import networkx as nx
import time as time
import pickle

## define environment

in_feats = 5
hid_feats = 64
budget = 5
g1 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g200test.gpickle")
g2 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400test.gpickle")
g3 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g1ktest.gpickle")
g4 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g200BAtest.gpickle")
g5 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400BAtest.gpickle")
g6 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g1kBAtest.gpickle")

Listgraph = [g1,g2,g3, g4,g5,g6]

candnodelist = [ list(gind.nodes) for gind in Listgraph ]

# define agent
agent = Agent( Listgraph, in_feats, hid_feats, candnodelist, seed=0, trainmodel_flag=0)
model = agent.qnetwork_local
print("=== model # params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# define environment
genv = genv( Listgraph, candnodelist, budget, weighingfactor=0.4)

## define main function for running algorithm

def dqn(checkpointpath, n_episodes = 1000, eps_start=1.0, eps_end = 0.05, decay_step=0.0002, exp_replay_size=1000, buffer_flag=0):

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

    # ====== initialize experience replay buffer ======

    index = 0
    st = time.time()

    if buffer_flag == 0:

        # new_maxreward = np.ones((6, 4))

        for i in range(exp_replay_size):

            state, candnodelist, gindex = genv.reset()
            done = False
            forbreak = 0

            while not done:

                action = agent.act(state, candnodelist, gindex, eps=1)
                next_step, reward, done, reward1, reward2 = genv.step(action, gindex)
                agent.save_buffer(state, action, reward, next_step, done, gindex, reward1, reward2)

                # agent.memory.memory
                state = next_step.copy()

                # if index % 10 == 0:
                #     new_maxreward = agent.get_newmaxreward(new_maxreward)

                print(index)
                index += 1
                if index > exp_replay_size:
                    forbreak = 1
                    break

            if forbreak == 1:
                break

        # get average reqard for each state
        # new_maxreward = agent.get_newmaxreward(new_maxreward)

        # with open(cnf.datapath + "\\ca-CSphd\\max_reward2.pickle", 'wb') as f:
        #     pickle.dump(new_maxreward, f)

        # save filled buffer as pcile file

        filledbuffer_wopadding = agent.get_filledbuffer_wopadding()

        filledbuffer = agent.get_filledbuffer()

        with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_PLC-BA_AIM3.pickle", 'wb') as f:
            pickle.dump(filledbuffer, f)

        with open(cnf.datapath + "\\ca-CSphd\\filledbufferwopad_nonrandom_PLC-BA_AIM3.pickle", 'wb') as f:
            pickle.dump(filledbuffer_wopadding, f)

    else:
        # load saved data into replaybuffer

        with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_PLC-BA_AIM3.pickle", 'rb') as f:
            filledbuffer = pickle.load(f)

        agent.load_filledbuffer(filledbuffer)

        with open(cnf.datapath + "\\ca-CSphd\\filledbufferwopad_random_PLC-BA_AIM3.pickle", 'rb') as f:
            filledbuffer_wopadding = pickle.load(f)

    print("buffer loading time: ", time.time()-st)

    #===== main loop for training ====

    # scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    scores_window_prev = deque(maxlen=100) # last 100 scores
    scores_window_prev.append(-9999) # large negative number for initializing previous start window

    eps = eps_start
    st = time.time()
    meanreward = []

    for i_episode in tqdm(range(1, n_episodes+1)):
        print("eps: ", i_episode)
        meanreward.append(np.mean(scores_window))
        print("mean s", np.mean(scores_window))
        print("mean l", np.mean(agent.trainloss[-5:]))

        state, candnodelist, gindex = genv.reset()
        score = 0
        done = 0

        while not done:

            action = agent.act(state, candnodelist, gindex, eps)

            next_state, reward, done, reward1, reward2 = genv.step(action, gindex)

            # print("state, action, reward, next_state", state, action, reward, next_state)

            agent.train(state, action, reward, next_state, done, gindex,  reward1, reward2)

            state = next_state

            score += reward

            scores_window.append(score) ## save the most recent score

            # scores.append(score)  ## save the most recent score

            eps = max(eps - decay_step, eps_end)

            if np.mean(scores_window) > np.mean(scores_window_prev):
                torch.save(agent.qnetwork_local.state_dict(), checkpointpath)
                scores_window_prev = scores_window.copy()
                # print("model saved")
                # save update filled buffer as pickle file

                updatedbuffer = agent.get_filledbuffer()
                filledbuffer_wopadding = agent.get_filledbuffer_wopadding()

                with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_PLC-BA_AIM3.pickle", 'wb') as f:
                    pickle.dump(updatedbuffer, f)

                with open(cnf.datapath + "\\ca-CSphd\\filledbufferwopad_random_PLC-BA_AIM3.pickle", 'wb') as f:
                    pickle.dump(filledbuffer_wopadding, f)

                # with open(cnf.datapath + "\\ca-CSphd\\max_reward.pickle", 'wb') as f:
                #     pickle.dump(new_maxreward, f)

            # if np.mean(scores_window)>=180.0:
            #     print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
            #                                                                                np.mean(scores_window)))
            #     torch.save(agent.qnetwork_local.state_dict(), checkpointpath)
            #     break

        # if i_episode % 10 == 0:
        #     new_maxreward = agent.get_newmaxreward(new_maxreward)

    print("training time: ", time.time() - st)
    return scores_window

checkpointpth = cnf.modelpath + "\\checkpoint_AIM_wtpdreward.pth"

scores_window = dqn(checkpointpath=checkpointpth)

##

