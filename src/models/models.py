import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from tqdm import tqdm
import random
import numpy as np

## Qnetwork pyth model

class GraphQNetwork(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats=1):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')

        self.fc1 = nn.Linear(2*hid_feats, out_feats)
        self.dropout = nn.Dropout(0.1)

    def forward(self, graph, inputs, states, actions):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(graph, h)
        h = F.relu(h)

        # state_indices = states[0]
        # action_index = torch.tensor(actions)

        states_vector = torch.index_select(h, 0, states)
        actions_vector = torch.index_select(h, 0, actions)

        # max aggregation
        states_aggvector = torch.amax(states_vector, dim=(0), keepdim=True)

        # mean aggregation
        # states_aggvector = torch.mean(states_vector, dim=0)

        h = torch.cat((states_aggvector, actions_vector), 1)

        out = self.fc1(h)

        return out

    # def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
    #     self.n_layers = n_layers
    #     self.n_hidden = n_hidden
    #     self.n_classes = n_classes
    #     self.layers = nn.ModuleList()
    #     if n_layers > 1:
    #         self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'pool'))
    #         for i in range(1, n_layers - 1):
    #             self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'pool'))
    #         self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'pool'))
    #     else:
    #         self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'pool'))
    #
    #     self.dropout = nn.Dropout(dropout)
    #     self.activation = activation

    # def forward(self, blocks, x):
    #     h = x
    #     for l, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h = layer(block, h)
    #         if l != len(self.layers) - 1:
    #             h = self.activation(h)
    #             h = self.dropout(h)
    #     return h

    # def inference(self, g, x, device, batch_size, num_workers):
    #
    #     """
    #     Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
    #     g : the entire graph.
    #     x : the input of entire node set.
    #
    #     The inference code is written in a fashion that it could handle any number of nodes and
    #     layers.
    #     """
    #     # During inference with sampling, multi-layer blocks are very inefficient because
    #     # lots of computations in the first few layers are repeated.
    #     # Therefore, we compute the representation of all nodes layer by layer.  The nodes
    #     # on each layer are of course splitted in batches.
    #     # TODO: can we standardize this?
    #
    #     for l, layer in enumerate(self.layers):
    #         y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
    #
    #         sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    #         dataloader = dgl.dataloading.NodeDataLoader(
    #             g,
    #             torch.arange(g.num_nodes()).to(g.device),
    #             sampler,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             drop_last=False,
    #             num_workers=num_workers)
    #
    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             block = blocks[0]
    #
    #             block = block.int().to(device)
    #             h = x[input_nodes].to(device)
    #             h = layer(block, h)
    #             if l != len(self.layers) - 1:
    #                 h = self.activation(h)
    #                 h = self.dropout(h)
    #
    #             y[output_nodes] = h.cpu()
    #
    #         x = y
    #     return y

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):
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
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

## environment class

class genv():

    def __init__(self, glist, candnodelist, b):
        self.graphlist = glist  # take graphs
        self.candidatenodelist = candnodelist
        self.budget = b
        # print("environemnt is invoked")

    def IC(self, g, S, p=0.5, mc=500):
        """
        Input:  graph object, set of seed nodes, propagation probability
                and the number of Monte-Carlo simulations
        Output: average number of nodes influenced by the seed nodes
        """
        # Loop over the Monte-Carlo Simulations
        spread = []

        for i in range(mc):

            # Simulate propagation process
            new_active, A = S[:], S[:]

            while new_active:

                # For each newly active node, find its neighbors that become activated
                new_ones = []
                for node in new_active:
                    # Determine neighbors that become infected
                    np.random.seed(i)
                    outn = [n for n in g.neighbors(node)]
                    success = np.random.uniform(0, 1, len(outn)) < p
                    new_ones += list(np.extract(success, outn))

                new_active = list(set(new_ones) - set(A))

                # Add newly activated nodes to the set of activated nodes
                A += new_active

            spread.append(len(A))

        return np.mean(spread)

    def reset(self):

        self.state = []
        self.next_state=[]
        self.spreadlist =[]

        self.candnodelist = self.candidatenodelist.copy()

        start_node = random.choice(self.candnodelist)
        self.candnodelist.remove(start_node)

        self.state.append(start_node)
        self.next_state.append(start_node)
        state = self.state.copy()

        self.spreadlist.append(self.IC(g=self.graphlist[0], S=state))

        return state, self.candnodelist

    def knownreset(self, start_node):

        self.state = []
        self.next_state = []
        self.spreadlist =[]

        self.candnodelist = self.candidatenodelist.copy()

        self.candnodelist.remove(start_node)

        self.state.append(start_node)
        self.next_state.append(start_node)
        state = self.state.copy()

        self.spreadlist.append(self.IC(g=self.graphlist[0], S=state))

        return state, self.candnodelist

    def step(self, action):

        # update candidate node list and next_state
        self.candnodelist.remove(action)
        # print(self.next_state)
        self.next_state.append(action.item())
        next_state = self.next_state.copy()

        # print(next_state)
        # reward
        self.spreadlist.append(self.IC(g= self.graphlist[0], S=next_state))
        reward = self.spreadlist[-1]-self.spreadlist[-2]

        # episode termination criterion
        if len(next_state) >= self.budget:
            done = True
        else:
            done = False

        return next_state, reward, done

















