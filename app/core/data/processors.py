"""
Data processing utilities for amino acid sequences
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data


def aa_to_int(sequence: str) -> list[int]:
    """
    Convert amino acid sequence to integer encoding
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        List of integers representing amino acids
    """
    aa_to_int_dict = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    return [aa_to_int_dict.get(aa.upper(), -1) for aa in sequence]


def aa_to_smiles(sequence: str) -> str:
    """
    Convert amino acid sequence to SMILES representation
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        SMILES string
    """
    aa_to_smiles_dict = {
        'A': 'CC(N)C(=O)O', 'R': 'NC(=N)NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
        'D': 'OC(=O)CC(N)C(=O)O', 'C': 'SC(C(N)C(=O)O)', 'E': 'OC(=O)CCC(N)C(=O)O',
        'Q': 'NC(=O)CCC(N)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1c[nH]cn1)C(=O)O',
        'I': 'CC(C)CC(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
        'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
        'S': 'OCC(N)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
        'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O'
    }
    smiles_list = [aa_to_smiles_dict.get(aa.upper(), '') for aa in sequence]
    return '.'.join([s for s in smiles_list if s])


def get_atom_features(atom) -> list:
    """
    Extract atom features for graph representation
    
    Args:
        atom: RDKit atom object
    
    Returns:
        List of atom features
    """
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        int(atom.GetHybridization()),
        atom.GetNumImplicitHs(),
        int(atom.GetChiralTag()),
        len(atom.GetNeighbors()),
        atom.IsInRing(),
        atom.GetMass(),
        atom.GetTotalValence()
    ]
    return features


def mol_to_graph(mol) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        PyTorch Geometric Data object
    """
    if mol is None:
        # Return empty graph if molecule is None
        return Data(
            x=torch.zeros((1, 12), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 3), dtype=torch.float)
        )
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices and features
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

        feature = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.GetIsAromatic()
        ]
        edge_features.extend([feature, feature])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


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

