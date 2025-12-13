"""
Combined model definition for AOP prediction
Combines sequence model (xLSTM) and graph model (MPNN)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from app.core.models.seq_model_def import SequenceModel
from app.core.models.graph_model_def import MPNN


# Sequence Feature Pooling Options
class SequencePooling(nn.Module):
    """
    Sequence feature pooling module
    
    Supports multiple pooling strategies:
    - attention: Self-attention pooling
    - max: Max pooling
    - mean: Mean pooling
    """
    def __init__(self, embedding_dim, pooling_type='attention'):
        super(SequencePooling, self).__init__()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(embedding_dim // 2, 1)
            )

    def forward(self, x):
        """
        Pool sequence features
        
        Args:
            x: Sequence features [batch_size, seq_len, embedding_dim]
        
        Returns:
            Pooled features [batch_size, embedding_dim]
        """
        if self.pooling_type == 'max':
            return torch.max(x, dim=1)[0]
        elif self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pooling_type == 'attention':
            attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
            return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class HierarchicalFusion(nn.Module):
    """
    Hierarchical feature fusion module
    Combines sequence and graph features
    """
    def __init__(self, seq_dim=128, graph_dim=128, hidden_dim=128, dropout_rate=0.5):
        super(HierarchicalFusion, self).__init__()

        self.seq_pooling = SequencePooling(seq_dim, pooling_type='attention')

        # Projection layers for sequence and graph features
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, seq_features, graph_features):
        """
        Fuse sequence and graph features
        
        Args:
            seq_features: Sequence features [batch_size, seq_len, seq_dim]
            graph_features: Graph features [batch_size, graph_dim]
        
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        # Pool sequence features
        pooled_seq = self.seq_pooling(seq_features)
        # Project features to common space
        seq_proj = self.seq_proj(pooled_seq)
        graph_proj = self.graph_proj(graph_features)
        # Concatenate and fuse
        combined = torch.cat([seq_proj, graph_proj], dim=1)
        fused = self.fusion(combined)
        return fused


class CombinedModel(nn.Module):
    """
    Combined model for AOP prediction
    Integrates sequence model (xLSTM) and graph model (MPNN)
    """
    def __init__(self, fusion_method='hierarchical',
                 fusion_hidden_dim=128, dropout_rate=0.5, device=None):
        super(CombinedModel, self).__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sequence model (xLSTM)
        self.sequence_model = SequenceModel(device=device)
        # Graph model (MPNN)
        self.graph_model = MPNN()

        # Feature fusion
        self.fusion_module = HierarchicalFusion()

        # Classifier layers
        self.fc1 = nn.Linear(fusion_hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Store fusion method for reference
        self.fusion_method = fusion_method

    def forward(self, sequences, x, edge_index, edge_attr, batch):
        """
        Forward pass
        
        Args:
            sequences: Sequence tensors [batch_size, seq_length]
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Tuple of (seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs)
        """
        # Process sequence data
        seq_features = self.sequence_model(sequences)
        # Process graph data
        graph_features = self.graph_model(x, edge_index, edge_attr, batch)

        # Pool sequence features
        pooled_seq = self.fusion_module.seq_pooling(seq_features)

        # Fuse features
        fused_features = self.fusion_module(seq_features, graph_features)

        # Classification
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.fc2(x)
        last_hidden = self.relu(x)
        x = self.dropout(last_hidden)
        x = self.fc3(x)
        outputs = self.sigmoid(x)

        return seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs


