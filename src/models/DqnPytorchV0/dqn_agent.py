
import copy
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_unit=64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class DQN_Agent():

    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        seed = torch.manual_seed(seed)
        self.q_net = QNetwork(4, 2, seed)
        # self.q_net = self.build_nn(layer_sizes)
        self.target_net = QNetwork(4, 2, seed)
        # self.target_net = copy.deepcopy(self.q_net)
        self.q_net.cuda()
        self.target_net.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().cuda()
        self.experience_replay = deque(maxlen=exp_replay_size)

    # def build_nn(self, layer_sizes):
    #     assert len(layer_sizes) > 1
    #     layers = []
    #     for index in range(len(layer_sizes) - 1):
    #         linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
    #         act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
    #         layers += (linear, act)
    #     return nn.Sequential(*layers)

    def load_pretrained_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_net.state_dict(), model_path)

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference

        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().cuda())

        Q, A = torch.max(Qp, axis=0)

        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))

        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)

        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.cuda())
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.cuda())
        target_return = rn.cuda() + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()


