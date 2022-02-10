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
hid_mlp = 16
TARGET_UPDATE = 8
budget = 20

g1 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g200test.gpickle")
g2 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400test.gpickle")
g3 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g1ktest.gpickle")
g4 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g200BAtest.gpickle")
g5 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g400BAtest.gpickle")
g6 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g1kBAtest.gpickle")
g7 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g200sbmtest.gpickle")
g8 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g300sbmtest.gpickle")
g9 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g600plctest.gpickle")
g10 = nx.read_gpickle(cnf.datapath + "\\ca-CSphd\\g800plctest.gpickle")

Listgraph = [g1,g2,g3, g4,g5,g6,g7,g8,g9,g10]

# candnodelist = [ list(gind.nodes) for gind in Listgraph ]

candnodelist = []
for countg in Listgraph:
    templist = [ nodeind for nodeind in countg.nodes if countg.nodes[nodeind]['alpha'] > 0.5]
    candnodelist.append(templist)
    # candnodelist = [ list(gind.nodes) for nodes in gind.nodes ]

# define agent
agent = Agent( Listgraph, in_feats, hid_feats, hid_mlp, candnodelist, seed=0, tuningweight=0.05, trainmodel_flag=0)
model = agent.qnetwork_local1
print("=== model # params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
model = agent.qnetwork_local2
print("=== model # params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# define environment
genv = genv( Listgraph, candnodelist, budget, weighingfactor=0.05, intr_threshold=0.85)

## define main function for running algorithm

def dqn(checkpointpath1, checkpointpath2, n_episodes = 2000, eps_start=1, eps_end = 0.01, decay_rate=0.996, exp_replay_size=5000, buffer_flag=0):

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

                action = agent.act_morl(state, candnodelist, gindex, eps=1)
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

        with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_AIM_0.05_r0.85_b20.pickle", 'wb') as f:
            pickle.dump(filledbuffer, f)

        with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_nonrandom_AIM_0.05_r0.85_b20.pickle", 'wb') as f:
            pickle.dump(filledbuffer_wopadding, f)

    else:
        # load saved data into replaybuffer

        with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_AIM_0.05_r0.85_b20.pickle", 'rb') as f:
            filledbuffer = pickle.load(f)

        agent.load_filledbuffer(filledbuffer)

        # with open(cnf.datapath + "\\ca-CSphd\\filledbufferwopad_nonrandom_PLC-BA_AIM3.pickle", 'rb') as f:
        #     filledbuffer_wopadding = pickle.load(f)

    print("buffer loading time: ", time.time()-st)

    #===== main loop for training ====

    # scores = [] # list containing score from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    best_score = -np.inf
    # scores_window_prev = deque(maxlen=100) # last 100 scores
    # scores_window_prev.append(-9999) # large negative number for initializing previous start window

    eps = eps_start
    st = time.time()

    for i_episode in tqdm(range(1, n_episodes+1)):
        print("eps: ", i_episode)
        print("mean s", np.mean(scores_window))
        print("mean l1", np.mean(agent.trainloss1[-5:]))
        print("mean l2", np.mean(agent.trainloss2[-5:]))

        state, candnodelist, gindex = genv.reset()
        score = 0
        done = 0

        while not done:

            action = agent.act_morl(state, candnodelist, gindex, eps)

            next_state, reward, done, reward1, reward2 = genv.step(action, gindex)

            # print("state, action, reward, next_state", state, action, reward, next_state)

            agent.train(state, action, reward, next_state, done, gindex,  reward1, reward2)

            state = next_state

            meanreward = np.mean([reward1, reward2])

            score += meanreward

            # scores_window.append(score) ## save the most recent score

            # scores.append(score)  ## save the most recent score

            eps = max(eps*decay_rate, eps_end)

            # evaluate/monitor performance after each train step

            # if np.mean(scores_window) > np.mean(scores_window_prev):
            #
            #     torch.save(agent.qnetwork_local1.state_dict(), checkpointpath1)
            #     torch.save(agent.qnetwork_local2.state_dict(), checkpointpath2)
            #
            #     scores_window_prev = scores_window.copy()
            #     # print("model saved")
            #     # save update filled buffer as pickle file
            #
            #     updatedbuffer = agent.get_filledbuffer()
            #     filledbuffer_wopadding = agent.get_filledbuffer_wopadding()
            #
            #     with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_PLC-BA_AIM3.pickle", 'wb') as f:
            #         pickle.dump(updatedbuffer, f)
            #
            #     with open(cnf.datapath + "\\ca-CSphd\\filledbufferwopad_random_PLC-BA_AIM3.pickle", 'wb') as f:
            #         pickle.dump(filledbuffer_wopadding, f)

                # with open(cnf.datapath + "\\ca-CSphd\\max_reward.pickle", 'wb') as f:
                #     pickle.dump(new_maxreward, f)

            # if np.mean(scores_window)>=180.0:
            #     print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
            #                                                                                np.mean(scores_window)))
            #     torch.save(agent.qnetwork_local.state_dict(), checkpointpath)
            #     break

        scores_window.append(score)
        avg_scores = np.mean(scores_window)

        if avg_scores > best_score:
            torch.save(agent.qnetwork_local1.state_dict(), checkpointpath1)
            torch.save(agent.qnetwork_local2.state_dict(), checkpointpath2)

            updatedbuffer = agent.get_filledbuffer()
            filledbuffer_wopadding = agent.get_filledbuffer_wopadding()

            with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_AIM_0.05_r0.85_b20.pickle", 'wb') as f:
                pickle.dump(updatedbuffer, f)

            with open(cnf.datapath + "\\ca-CSphd\\filledbuffer_wopadding_AIM_0.05_r0.85_b20.pickle", 'wb') as f:
                pickle.dump(filledbuffer_wopadding, f)

            best_score = avg_scores

        if i_episode % TARGET_UPDATE == 0:
            agent.qnetwork_target1.load_state_dict(agent.qnetwork_local1.state_dict())
            agent.qnetwork_target2.load_state_dict(agent.qnetwork_local2.state_dict())

    print("training time: ", time.time() - st)
    return scores_window

checkpointpth1 = cnf.modelpath + "\\checkpoint_AIM1_0.05r0.85_b20.pth"
checkpointpth2 = cnf.modelpath + "\\checkpoint_AIM2_0.05r0.85_b20.pth"

scores_window = dqn(checkpointpath1=checkpointpth1, checkpointpath2= checkpointpth2)

##

