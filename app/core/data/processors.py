"""
Data processing utilities for amino acid sequences

DEPRECATED: This module is being phased out.
These functions now delegate to CoreAdapter which wraps the upstream core algorithm.
For new code, import from app.adapters.core_adapter directly.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Import CoreAdapter to delegate to upstream core algorithm
from app.adapters.core_adapter import get_core_adapter

# Get adapter instance
_adapter = get_core_adapter()


def aa_to_int(sequence: str) -> list[int]:
    """
    Convert amino acid sequence to integer encoding
    
    DEPRECATED: Use CoreAdapter.aa_to_int() instead.
    This function delegates to the upstream core algorithm via CoreAdapter.
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        List of integers representing amino acids
    """
    return _adapter.aa_to_int(sequence)


def aa_to_smiles(sequence: str) -> str:
    """
    Convert amino acid sequence to SMILES representation
    
    DEPRECATED: Use CoreAdapter.aa_to_smiles() instead.
    This function delegates to the upstream core algorithm via CoreAdapter.
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        SMILES string
    """
    return _adapter.aa_to_smiles(sequence)


def mol_to_graph(mol) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object
    
    DEPRECATED: Use CoreAdapter.mol_to_graph() instead.
    This function delegates to the upstream core algorithm via CoreAdapter.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        PyTorch Geometric Data object
    """
    return _adapter.mol_to_graph(mol)


def process_sequence(sequence: str, seq_length: int = 50) -> tuple[torch.Tensor, Data]:
    """
    Process a single sequence into model inputs
    
    Args:
        sequence: Amino acid sequence string
        seq_length: Maximum sequence length
    
    Returns:
        Tuple of (sequence_tensor, graph_data)
    """
    # Process sequence data
    sequence_int = aa_to_int(sequence)
    if len(sequence_int) > seq_length:
        sequence_int = sequence_int[:seq_length]
    else:
        sequence_int = sequence_int + [0] * (seq_length - len(sequence_int))
    sequence_tensor = torch.tensor(sequence_int, dtype=torch.long)

    # Process graph data
    smiles = aa_to_smiles(sequence)
    mol = Chem.MolFromSmiles(smiles)
    graph_data = mol_to_graph(mol)

    return sequence_tensor, graph_data

