import math
import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree

from staprop import MessageProp_random_walk, KeyProp_random_walk


class STAGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout, K, global_attn):
        super(STAGNN, self).__init__()
        self.input_trans = Linear(num_features, hidden_channels)
        self.linQ = Linear(hidden_channels, hidden_channels)
        self.linK = Linear(hidden_channels, hidden_channels)
        self.linV = Linear(hidden_channels, num_classes)


        self.propM = MessageProp_random_walk(node_dim=-3)
        self.propK = KeyProp_random_walk(node_dim=-2)        

        self.c = hidden_channels
        self.dropout = dropout
        self.K = K

        self.cst = 10e-6

        self.hopwise = Parameter(torch.ones(K+1, dtype=torch.float))
        self.teleport = Parameter(torch.ones(1, dtype=torch.float))
        self.global_attn = global_attn

    def reset_parameters(self):
        self.input_trans.reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        torch.nn.init.ones_(self.hopwise)
        torch.nn.init.ones_(self.teleport)


    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']


        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row]

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
        
        if (self.global_attn):
            num_nodes = x.size(0)
            teleportM = torch.sum(M, dim=0, keepdim=True) / num_nodes
            teleportK = torch.sum(K, dim=0, keepdim=True) / num_nodes
            teleportH = torch.einsum('ni,nij->nj',[Q,teleportM])
            teleportC = torch.einsum('ni,ni->n',[Q,teleportK]).unsqueeze(-1) + self.cst
            teleportH = teleportH / teleportC


        hidden = V*(self.hopwise[0])
        for hop in range(self.K):

            M = self.propM(M, edge_index, norm.view(-1,1,1))
            K = self.propK(K, edge_index, norm.view(-1,1))
       
            # H = (Q.repeat(1, M.size(-1)).view(-1, M.size(-1),
                #  Q.size(-1)).transpose(-1, -2) * M).sum(dim=-2)
            H = torch.einsum('ni,nij->nj',[Q,M])
            # C = (Q * K).sum(dim=-1, keepdim=True) + self.cst
            C = torch.einsum('ni,ni->n',[Q,K]).unsqueeze(-1) + self.cst
            H = H / C
            gamma = self.hopwise[hop+1]
            hidden = hidden + gamma*H

        if (self.global_attn):
            hidden = hidden + self.teleport*teleportH

        return hidden


class MSTAGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout, K, num_heads, ind_gamma, gamma_softmax, multi_concat, global_attn):
        super(MSTAGNN, self).__init__()
        self.headc = headc = hidden_channels // num_heads
        self.input_trans = Linear(num_features, hidden_channels)
        self.linQ = Linear(hidden_channels, headc * num_heads)
        self.linK = Linear(hidden_channels, headc * num_heads)
        self.linV = Linear(hidden_channels, num_classes * num_heads)
        if (multi_concat):
            self.output = Linear(num_classes * num_heads, num_classes)


        self.propM = MessageProp_random_walk(node_dim=-4)
        self.propK = KeyProp_random_walk(node_dim=-3)        


        self.dropout = dropout
        self.K = K
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.multi_concat = multi_concat
        self.ind_gamma = ind_gamma
        self.gamma_softmax = gamma_softmax
        self.global_attn = global_attn

        self.cst = 10e-6

        if (ind_gamma):
            if (gamma_softmax):
                self.hopwise = Parameter(torch.ones(K+1))
                self.headwise = Parameter(torch.zeros(size=(self.num_heads,K)))
            else:
                self.hopwise = Parameter(torch.ones(size=(self.num_heads,K+1)))
        else:
            self.hopwise = Parameter(torch.ones(K+1))
        
        self.teleport = Parameter(torch.ones(1))

    def reset_parameters(self):
        if (self.ind_gamma and self.gamma_softmax):
            torch.nn.init.ones_(self.hopwise)
            torch.nn.init.zeros_(self.headwise)
        else:
            torch.nn.init.ones_(self.hopwise)
        self.input_trans.reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        if (self.multi_concat):
            self.output.reset_parameters()
        torch.nn.init.ones_(self.teleport)

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']



        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row]

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
            if (self.gamma_softmax):
                hidden = V * (self.hopwise[0])
            else:
                hidden = V * (self.hopwise[:, 0].unsqueeze(-1))
        else:
            hidden = V * (self.hopwise[0])

        if ((self.ind_gamma) and (self.gamma_softmax)):
            layerwise = F.softmax(self.headwise, dim=-2)

        if (self.global_attn):
            num_nodes = x.size(0)
            teleportM = torch.sum(M, dim=0, keepdim=True) / num_nodes
            teleportK = torch.sum(K, dim=0, keepdim=True) / num_nodes
            teleportH = torch.einsum('nhi,nhij->nhj',[Q,teleportM])
            teleportC = torch.einsum('nhi,nhi->nh',[Q,teleportK]).unsqueeze(-1) + self.cst
            teleportH = teleportH / teleportC
            teleportH = teleportH.sum(dim=-2)

        for hop in range(self.K):

            M = self.propM(M, edge_index, norm.view(-1,1,1,1))
            K = self.propK(K, edge_index, norm.view(-1,1,1))

            H = torch.einsum('nhi,nhij->nhj', [Q, M])
            C = torch.einsum('nhi,nhi->nh', [Q, K]).unsqueeze(-1) + self.cst
            H = H / C
            if (self.ind_gamma):
                if (self.gamma_softmax):
                    gamma = self.hopwise[hop+1] * layerwise[:, hop].unsqueeze(-1)
                else:
                    gamma = self.hopwise[:, hop+1].unsqueeze(-1)
            else:
                gamma = self.hopwise[hop+1]
            hidden = hidden + gamma * H

        if (self.multi_concat):
            hidden = hidden.view(-1, self.num_classes * self.num_heads)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.output(hidden)
        else:
            hidden = hidden.sum(dim=-2)
    
        if (self.global_attn):
            hidden = hidden + self.teleport*teleportH

        return hidden