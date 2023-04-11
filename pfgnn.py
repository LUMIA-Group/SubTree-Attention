import torch
from torch.nn import Parameter, Linear, init
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


class MHPFGT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout, K, alpha, num_heads, ind_gamma, gamma_softmax, multi_concat):
        super(MHPFGT, self).__init__()
        self.headc = headc = hidden_channels // num_heads
        self.input_trans = Linear(num_features, hidden_channels)
        self.linQ = Linear(hidden_channels, headc * num_heads)
        self.linK = Linear(hidden_channels, headc * num_heads)
        self.linV = Linear(hidden_channels, num_classes * num_heads)
        self.output = Linear(num_classes * num_heads, num_classes)

        self.propM = MessageProp(node_dim=-4)
        self.propK = KeyProp(node_dim=-3)

        self.dropout = dropout
        self.K = K
        self.alpha = alpha
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.multi_concat = multi_concat
        self.ind_gamma = ind_gamma
        self.gamma_softmax = gamma_softmax

        self.cst = 10e-6

        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K

        if (ind_gamma):
            if (gamma_softmax):
                self.hopwise = Parameter(torch.tensor(TEMP))
                self.temp = Parameter(torch.empty(self.num_heads, self.K + 1, requires_grad=True, dtype=torch.float, device='cuda'))
                init.uniform_(self.temp)
            else:
                self.temp = Parameter(torch.tensor(TEMP).unsqueeze(
                    0).repeat(self.num_heads, 1).float())
        else:
            self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if (self.ind_gamma):
            if (self.gamma_softmax):
                for k in range(self.K+1):
                    self.hopwise.data[k] = self.alpha*(1-self.alpha)**k
                self.hopwise.data[-1] = (1-self.alpha)**self.K
                init.uniform_(self.temp)
            else:
                for h in range(self.num_heads):
                    for k in range(self.K+1):
                        self.temp.data[h, k] = self.alpha*(1-self.alpha)**k
                    self.temp.data[h, -1] = (1-self.alpha)**self.K
        else:
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

        Q = Q.view(-1, self.num_heads, self.headc)
        K = K.view(-1, self.num_heads, self.headc)
        V = V.view(-1, self.num_heads, self.num_classes)

        M = torch.einsum('nhi,nhj->nhij', [K, V])

        if (self.ind_gamma):
            hidden = V * (self.temp[:, 0].unsqueeze(-1))
        else:
            hidden = V * (self.temp[0])

        if ((self.ind_gamma) and (self.gamma_softmax)):
            layerwise = F.softmax(self.temp, dim=-2)

        for hop in range(self.K):
            M = self.propM(M, edge_index)
            K = self.propK(K, edge_index)
            H = torch.einsum('nhi,nhij->nhj', [Q, M])
            C = torch.einsum('nhi,nhi->nh', [Q, K]).unsqueeze(-1) + self.cst
            H = H / C
            if (self.ind_gamma):
                if (self.gamma_softmax):
                    gamma = self.hopwise[hop+1] * layerwise[:, hop+1].unsqueeze(-1)
                else:
                    gamma = self.temp[:, hop+1].unsqueeze(-1)
            else:
                gamma = self.temp[hop+1]
            hidden = hidden + gamma * H

        if (self.multi_concat):
            hidden = hidden.view(-1, self.num_classes * self.num_heads)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.output(hidden)
        else:
            hidden = hidden.mean(dim=-2)

        return hidden
