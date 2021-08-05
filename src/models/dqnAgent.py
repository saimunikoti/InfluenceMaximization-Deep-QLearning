import numpy as np
import random 
from collections import namedtuple, deque 
import dgl
##Importing the model (function approximator for Q-table)
from src.models.models import GraphQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 256     # replay buffer size
BATCH_SIZE = 16        # minibatch size
GAMMA = 0.95           # discount factor
TAU = 1e-3             # for soft update of target parameters
LR = 1e-3              # learning rate
UPDATE_EVERY = 1       # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, glist, in_feats, hid_feats, candnodelist, seed):

        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.graphlist = [dgl.from_networkx(glist[ind], node_attrs=['feature', 'label']).to(device) for ind in range(len(glist))]
        self.seed = random.seed(seed)
        self.candidatenodelist = candnodelist

        #Q- Network
        self.qnetwork_local = GraphQNetwork(in_feats= in_feats, hid_feats=hid_feats).to(device)
        self.qnetwork_target = GraphQNetwork(in_feats= in_feats, hid_feats=hid_feats).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory) > (BATCH_SIZE*5):
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def train(self, state, action, reward, next_step, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)
        # print("state, action, reward, next_state", state, action, reward, next_step)
        # print("state, action, reward, next_state", state, action, reward, next_step)

        # train every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1) % UPDATE_EVERY

        if self.t_step == 0:
            experience = self.memory.sample()
            self.learn(experience, GAMMA)

    def save_buffer(self, state, action, reward, next_step, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

    def act(self, state, candnodelist, eps = 0):

        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        """

        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # state = torch.tensor(state).to(device)
        state = torch.tensor(state).to(device)
        candnodelist = torch.tensor(candnodelist).to(device)

        action_values = []

        self.qnetwork_local.eval()
        train_nfeat = self.graphlist[0].ndata['feature']
        train_nfeat = torch.tensor(train_nfeat).to(device)

        with torch.no_grad():
            for action in candnodelist:
                action_values.append(self.qnetwork_local(self.graphlist[0], train_nfeat, state, action).cpu().data.numpy())

        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            action_index = np.argmax(action_values)
        else:
            action_index = random.choice(np.arange(len(candnodelist)))

        return candnodelist[action_index]

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        =======

            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples

            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print("states, actions, rewards, next_states", states.shape, actions.shape, rewards.shape, next_states.shape)

        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()

        # predicted_targets = []
        # #shape of output from the model (batch_size,action_dim) = (64,4)
        train_nfeat = self.graphlist[0].ndata['feature']

        for count, (count_state, count_action) in enumerate(zip(states, actions)):

            ## predicted_targets.append(self.qnetwork_local(self.graphlist[0], train_nfeat, count_state, count_action).item())
            temp_pred = self.qnetwork_local(self.graphlist[0], train_nfeat, count_state, count_action)

            if count == 0:
                predicted_targets = temp_pred.clone()
            else:
                predicted_targets = torch.cat([predicted_targets, temp_pred], dim=0)

        # predicted_targets = self.qnetwork_local(self.graphlist[0], train_nfeat, states[0], actions[0])

        # print("predicted_targets raw", self.qnetwork_local(states).shape)
        # print("predicted_targets processed", predicted_targets.shape)

        with torch.no_grad():
            labels_next = []
            for counts, count_state in enumerate(next_states):
                candnodes = self.candidatenodelist.copy()
                candnodes = [ele for ele in candnodes if ele not in count_state]
                candnodes = torch.tensor(candnodes).to(device)

                # temp_label = []
                for counta, count_action in enumerate(candnodes):
                    # temp_label.append(self.qnetwork_target(self.graphlist[0], train_nfeat, count_state, count_action).item())
                    if counta == 0:
                        temp_label = self.qnetwork_target(self.graphlist[0], train_nfeat, count_state, count_action)
                    else:
                        temp_label = torch.cat([temp_label, self.qnetwork_target(self.graphlist[0], train_nfeat, count_state, count_action)], dim=0)

                # labels_next.append(max(temp_label))
                temp_label = torch.reshape(torch.max(temp_label), (1,1))
                if counts == 0:
                    labels_next = temp_label
                else:
                    labels_next = torch.cat([labels_next, temp_label], dim=0)
                try:
                    del temp_label
                except:
                    pass

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        # labels = torch.tensor([rewards[ind] + (gamma*labels_next[ind]*(1-dones[ind])) for ind in range(int(states.shape[0])) ])
        # labels = torch.tensor([rewards[ind] + (gamma*labels_next[ind]*(1-dones[ind])) for ind in range(int(states.shape[0])) ])
        labels = rewards + torch.mul( (torch.mul(labels_next, gamma)),(1-dones))
        # predicted_targets = torch.tensor(predicted_targets)

        loss = criterion(predicted_targets, labels).to(device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)
            
    def soft_update(self, local_model, target_model, tau):

        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        # self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):

        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # states = torch.tensor(torch.tensor([e.state for e in experiences if e is not None])).float().to(device)
        # states = torch.from_numpy(np.vstack([e.state.item() for e in experiences if e is not None])).float().to(device)
        # states = np.array([torch.tensor(e.state).to(device) for e in experiences if e is not None])
        temp = [(e.state) for e in experiences if e is not None]
        max_cols = max([len(batch) for batch in temp ])
        # NEEDS CHANGES IF AGGREGATION OF STATE VECTOR FUNCTION CHNAGES FROM MAXIMUM FUNCTION
        padded = [batch + [batch[0]]*(max_cols - len(batch)) for batch in temp]
        states = torch.tensor(padded).to(device)

        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        actions = torch.from_numpy(np.vstack([e.action.item() for e in experiences if e is not None])).long().to(device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # next_states = np.array([torch.tensor(e.next_state).to(device) for e in experiences if e is not None])

        tempns = [(e.next_state) for e in experiences if e is not None]

        max_cols = max([len(batch) for batch in tempns])
        # NEEDS CHANGES IF AGGREGATION OF STATE VECTOR FUNCTION CHNAGES FROM MAXIMUM FUNCTION
        paddedns = [batch + [batch[0]]*(max_cols - len(batch)) for batch in tempns]
        next_states = torch.tensor(paddedns).to(device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)