#load the weights from file
from collections import deque
from src.models.dqnAgent import Agent
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
from IPython.display import display

## define environment
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = Agent(state_size=4,action_size=2,seed=0)
agent.qnetwork_local.load_state_dict(torch.load(checkpointpth))

for i in range(3):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(100):
        action = agent.act(state, eps=0)
        img.set_data(env.render(mode='rbg_array'))
        plt.axix('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state,reward,done,_ = env.step(action)
        if done:
            break

env.close()