import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import dgl.nn as dglnn
from tqdm import tqdm

## Qnetwork pyth model

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
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):

        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?

        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

## environment class

class S2V_QN_1(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T, args=None):
        super(S2V_QN_1, self).__init__()
        self.args = args
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        # self.mu_1 = torch.nn.Linear(1, embed_dim)
        # torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)
        self.mu_1 = torch.nn.Parameter(
            torch.Tensor(1 if self.args.use_state_abs else 3, embed_dim))  # [#batch, #nodes, 3]
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)
        self.list_pre_pooling = []

        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []

        for i in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)

        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear((2 if self.args.use_state_abs else 4) * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear((2 if self.args.use_state_abs else 4) * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj, mask=None):
        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]

        def _mask_out(mu, mask, minibatch_size):
            # batch and type1 mode
            if minibatch_size > 1 and self.args.model_scheme == 'type1':
                mask = mask.view(minibatch_size, -1, 1)
                mask = mask.repeat(1, 1, mu.shape[-1])
                mu = mu * mask
            return mu

        for t in range(self.T):
            if t == 0:
                # mu = self.mu_1(xv).clamp(0)
                mu = torch.matmul(xv, self.mu_1).clamp(0)
                mu = _mask_out(mu, mask, minibatch_size)

                # mu.transpose_(1,2)
                # mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                # mu = torch.add(mu_1, mu_2).clamp(0)
            else:
                # mu_1 = self.mu_1(xv).clamp(0)
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
                mu_1 = _mask_out(mu_1, mask, minibatch_size)

                # mu_1.transpose_(1,2)
                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj.float(), mu)
                mu_pool = _mask_out(mu_pool, mask, minibatch_size)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu_2 = _mask_out(mu_2, mask, minibatch_size)

                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(xv.transpose(1, 2), mu)).view(minibatch_size, 1, -1)
        q_1 = q_1.expand(minibatch_size, nbr_node, self.embed_dim * xv.shape[-1])
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)  # 64+64/ 64,3*64

        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_ = q_.clamp(0)
            q = self.q(q_)

        # mask out
        if self.args.model_scheme == 'type1' and minibatch_size > 1:
            q[mask.view(minibatch_size, -1, 1) == 0] = -99999
        return q
