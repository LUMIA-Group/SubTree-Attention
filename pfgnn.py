import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
import numpy as np

from pfprop import MessageProp, KeyProp


class PFGT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout, K, alpha):
        super(PFGT, self).__init__()
        self.input_trans = Linear(num_features, hidden_channels)
        self.linQ = Linear(hidden_channels, hidden_channels)
        self.linK = Linear(hidden_channels, hidden_channels)
        self.linV = Linear(hidden_channels, num_classes)

        self.propM = MessageProp()
        self.propK = KeyProp()

        self.c = hidden_channels
        self.dropout = dropout
        self.K = K
        self.alpha = alpha

        self.cst = 10e-6

        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_trans(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        Q = self.linQ(x)
        K = self.linK(x)
        V = self.linV(x)

        Q = 1 + F.elu(Q)
        K = 1 + F.elu(K)

        # M = K.repeat(1, V.size(1)).view(-1, V.size(1), K.size(1)).transpose(-1, -2) * V.repeat(1, K.size(1)).view(-1, K.size(1), V.size(1))
        M = torch.einsum('ni,nj->nij',[K,V])

        hidden = V*(self.temp[0])
        for hop in range(self.K):
            M = self.propM(M, edge_index)
            K = self.propK(K, edge_index)         
            # H = (Q.repeat(1, M.size(-1)).view(-1, M.size(-1),
                #  Q.size(-1)).transpose(-1, -2) * M).sum(dim=-2)
            H = torch.einsum('ni,nij->nj',[Q,M])
            # C = (Q * K).sum(dim=-1, keepdim=True) + self.cst
            C = torch.einsum('ni,ni->n',[Q,K]).unsqueeze(-1) + self.cst
            H = H / C
            gamma = self.temp[hop+1]
            hidden = hidden + gamma*H

        return hidden
