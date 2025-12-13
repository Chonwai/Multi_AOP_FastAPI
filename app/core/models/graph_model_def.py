"""
Graph model definition using MPNN (Message Passing Neural Network)
"""

import torch
import torch.nn as nn
import torch_geometric.nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class MPNNConv(MessagePassing):
    """
    Message Passing Neural Network Convolution Layer
    
    Args:
        in_channels: Number of input node features
        out_channels: Number of output node features
        edge_features: Number of edge features
    """
    def __init__(self, in_channels, out_channels, edge_features):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_features, out_channels)
        self.lin_message = nn.Linear(2 * out_channels + out_channels, out_channels)
        self.lin_update = nn.Linear(2 * out_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        x = self.lin_node(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr is not None and edge_index.size(1) > edge_attr.size(0):
            # Add self-loop edge features
            self_loop_attr = torch.zeros(
                (x.size(0), edge_attr.size(1)),
                device=edge_attr.device,
                dtype=edge_attr.dtype
            )
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Compute messages between nodes"""
        if edge_attr is not None:
            edge_embedding = self.lin_edge(edge_attr)
        else:
            edge_embedding = torch.zeros(
                (x_i.size(0), self.out_channels),
                device=x_i.device
            )
        return self.lin_message(torch.cat([x_i, edge_embedding, x_j], dim=1))

    def update(self, aggr_out, x):
        """Update node features"""
        return self.lin_update(torch.cat([x, aggr_out], dim=1))


class MPNN(nn.Module):
    """
    Message Passing Neural Network for molecular graph processing
    
    Args:
        node_features: Number of node features (default: 12)
        edge_features: Number of edge features (default: 3)
        hidden_size: Hidden layer size (default: 128)
        dropout_rate: Dropout rate (default: 0.5)
    """
    def __init__(self, node_features=12, edge_features=3, hidden_size=128, dropout_rate=0.5):
        super(MPNN, self).__init__()

        self.conv1 = MPNNConv(node_features, hidden_size, edge_features)
        self.conv2 = MPNNConv(hidden_size, hidden_size, edge_features)
        self.conv3 = MPNNConv(hidden_size, hidden_size, edge_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Graph-level features [batch_size, hidden_size]
        """
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index, edge_attr))
        # Get graph-level representation
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x


