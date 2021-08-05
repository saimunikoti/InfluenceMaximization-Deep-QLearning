import gym
from tqdm import tqdm
from time import sleep
from src.models.dqnAgent import Agent
import torch

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "record_dir", force='True')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = Agent(state_size=input_dim,action_size=output_dim,seed=0)
checkpointpath = r"C:\Users\saimunikoti\Manifestation\InfluenceMaximization_DRL\src\models\test_new\checkpoint.pth"

agent.qnetwork_local.load_state_dict(torch.load(checkpointpath))

reward_arr = []
for i in tqdm(range(100)):
    state, done, rew = env.reset(), False, 0
    while not done:
        # A = agent.get_action(obs, env.action_space.n, epsilon=0)
        action = agent.act(state)
        state,reward,done,_ = env.step(action)
        rew += reward
        # sleep(0.01)
        # env.render()

    reward_arr.append(rew)
print("average reward per episode :", sum(reward_arr) / len(reward_arr))
