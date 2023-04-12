from torch_geometric.nn import MessagePassing

class MessageProp(MessagePassing):
    def __init__(self, aggr, node_dim=-3):
        super().__init__(aggr=aggr, node_dim=node_dim)  

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j




class KeyProp(MessagePassing):
    def __init__(self, aggr, node_dim=-2):
        super().__init__(aggr=aggr, node_dim=node_dim)  


    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j






class MessageProp_normalized_laplacian(MessagePassing):
    def __init__(self, node_dim=-3):
        super().__init__(aggr='add', node_dim=node_dim)  

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j




class KeyProp_normalized_laplacian(MessagePassing):
    def __init__(self, node_dim=-2):
        super().__init__(aggr='add', node_dim=node_dim)  

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j