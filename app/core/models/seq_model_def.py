"""
Sequence model definition using xLSTM
"""

import torch
import torch.nn as nn

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig
from xlstm import mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig


def create_xlstm_config(device: torch.device):
    """Create xLSTM configuration based on device"""
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=8,
                num_heads=4
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda" if device.type == "cuda" else "vanilla",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=2.0,
                act_fn="gelu",
                dropout=0.5
            ),
        ),
        context_length=256,
        num_blocks=3,
        embedding_dim=128,
        slstm_at=[1, 2]
    )


class SequenceModel(nn.Module):
    """
    xLSTM-based sequence model for amino acid sequences
    
    Args:
        vocab_size: Vocabulary size (21 for amino acids + padding)
        seq_length: Maximum sequence length
        device: Device to run the model on
    """
    def __init__(self, vocab_size=21, seq_length=50, device=None):
        super(SequenceModel, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        cfg = create_xlstm_config(device)
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.seq_length = seq_length
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_length]
        
        Returns:
            Sequence features of shape [batch_size, seq_length, embedding_dim]
        """
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        return x

