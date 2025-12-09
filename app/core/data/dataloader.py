"""
Data loader for in-memory sequence processing
Supports both single sequences and batches
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from app.core.data.processors import process_sequence


class InMemorySequenceDataset(Dataset):
    """
    In-memory dataset for amino acid sequences
    Does not require CSV files, works directly with sequence lists
    """
    def __init__(self, sequences: List[str], labels: Optional[List[float]] = None, seq_length: int = 50):
        """
        Initialize dataset
        
        Args:
            sequences: List of amino acid sequence strings
            labels: Optional list of labels (for compatibility, not used in prediction)
            seq_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels if labels is not None else [0.0] * len(sequences)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Process sequence
        sequence_tensor, graph_data = process_sequence(sequence, self.seq_length)

        # Convert label
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return {
            'sequence': sequence_tensor,
            'graph': graph_data,
            'label': label_tensor
        }


def create_in_memory_loader(
    sequences: List[str],
    batch_size: int,
    seq_length: int = 50,
    shuffle: bool = False,
    labels: Optional[List[float]] = None
) -> DataLoader:
    """
    Create a DataLoader from a list of sequences (in-memory)
    
    Args:
        sequences: List of amino acid sequence strings
        batch_size: Batch size
        seq_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        labels: Optional list of labels
    
    Returns:
        DataLoader instance
    """
    dataset = InMemorySequenceDataset(sequences, labels, seq_length)
    
    def collate_fn(batch):
        """Collate function to handle both graph and sequence data"""
        sequences = torch.stack([item['sequence'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        graphs = [item['graph'] for item in batch]

        # Create batched graph data
        batch_idx = []
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for i, graph in enumerate(graphs):
            num_nodes = graph.x.size(0)
            batch_idx.extend([i] * num_nodes)
            x_list.append(graph.x)

            if graph.edge_index.size(1) > 0:
                # Adjust edge indices for batching
                edge_index = graph.edge_index.clone()
                edge_index[0] += sum(g.x.size(0) for g in graphs[:i])
                edge_index[1] += sum(g.x.size(0) for g in graphs[:i])
                edge_index_list.append(edge_index)
                edge_attr_list.append(graph.edge_attr)
        
        # Combine graph data
        if len(edge_index_list) > 0:
            batched_x = torch.cat(x_list, dim=0)
            batched_edge_index = torch.cat(edge_index_list, dim=1)
            batched_edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            batched_x = torch.cat(x_list, dim=0)
            batched_edge_index = torch.zeros((2, 0), dtype=torch.long, device=sequences.device)
            batched_edge_attr = torch.zeros((0, 3), dtype=torch.float, device=sequences.device)

        batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long)

        return {
            'sequences': sequences,
            'x': batched_x,
            'edge_index': batched_edge_index,
            'edge_attr': batched_edge_attr,
            'batch': batch_idx_tensor,
            'labels': labels
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

