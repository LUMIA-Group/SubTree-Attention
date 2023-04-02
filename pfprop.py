from torch_geometric.nn import MessagePassing

class MessageProp(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=-3)  

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j




class KeyProp(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  


    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j
