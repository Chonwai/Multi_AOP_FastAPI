"""
Input validation utilities for amino acid sequences and batch operations

This module provides validation functions for:
- Single amino acid sequences
- Batch of sequences
- Batch sizes
"""

from typing import List, Tuple

# Standard 20 amino acids
STANDARD_AMINO_ACIDS = set(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 50


def validate_sequence(
    sequence: str, 
    min_length: int = MIN_SEQUENCE_LENGTH, 
    max_length: int = MAX_SEQUENCE_LENGTH,
    normalize: bool = True
) -> Tuple[bool, str, str]:
    """
    Validate amino acid sequence
    
    Args:
        sequence: Amino acid sequence string
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        normalize: Whether to normalize the sequence (strip and uppercase)
    
    Returns:
        Tuple of (is_valid, error_message, normalized_sequence)
        If invalid, normalized_sequence will be empty string
    """
    if not isinstance(sequence, str):
        return False, "Sequence must be a string", ""
    
    # Normalize sequence
    if normalize:
        sequence = sequence.strip().upper()
    else:
        sequence = sequence.strip()
    
    # Check if empty after stripping
    if not sequence:
        return False, "Sequence cannot be empty", ""
    
    # Check length
    if len(sequence) < min_length:
        return False, (
            f"Sequence length must be at least {min_length} amino acids. "
            f"Got {len(sequence)}."
        ), ""
    
    if len(sequence) > max_length:
        return False, (
            f"Sequence length must be at most {max_length} amino acids. "
            f"Got {len(sequence)}."
        ), ""
    
    # Check for invalid amino acid characters
    invalid_chars = set(sequence) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        invalid_list = ', '.join(sorted(invalid_chars))
        return False, (
            f"Sequence contains invalid amino acid characters: {invalid_list}. "
            f"Only standard 20 amino acids are allowed: {', '.join(sorted(STANDARD_AMINO_ACIDS))}"
        ), ""
    
    return True, "", sequence


def validate_sequences(
    sequences: List[str],
    min_length: int = MIN_SEQUENCE_LENGTH,
    max_length: int = MAX_SEQUENCE_LENGTH,
    max_batch_size: int = 100
) -> Tuple[bool, str, List[str]]:
    """
    Validate a batch of amino acid sequences
    
    Args:
        sequences: List of amino acid sequence strings
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        max_batch_size: Maximum number of sequences allowed
    
    Returns:
        Tuple of (is_valid, error_message, normalized_sequences)
        If invalid, normalized_sequences will be empty list
    """
    if not isinstance(sequences, list):
        return False, "Sequences must be a list", []
    
    if len(sequences) == 0:
        return False, "Batch cannot be empty", []
    
    if len(sequences) > max_batch_size:
        return False, (
            f"Batch size cannot exceed {max_batch_size}. "
            f"Got {len(sequences)} sequences."
        ), []
    
    # Validate each sequence
    normalized_sequences = []
    for idx, sequence in enumerate(sequences):
        is_valid, error_msg, normalized_seq = validate_sequence(
            sequence, min_length, max_length
        )
        if not is_valid:
            return False, f"Sequence at index {idx}: {error_msg}", []
        normalized_sequences.append(normalized_seq)
    
    return True, "", normalized_sequences


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> Tuple[bool, str]:
    """
    Validate batch size
    
    Args:
        batch_size: Number of sequences in batch
        max_batch_size: Maximum allowed batch size
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(batch_size, int):
        return False, "Batch size must be an integer"
    
    if batch_size < 1:
        return False, f"Batch size must be at least 1. Got {batch_size}."
    
    if batch_size > max_batch_size:
        return False, (
            f"Batch size cannot exceed {max_batch_size}. "
            f"Got {batch_size}."
        )
    
    return True, ""


def normalize_sequence(sequence: str) -> str:
    """
    Normalize amino acid sequence (strip whitespace and convert to uppercase)
    
    Args:
        sequence: Amino acid sequence string
    
    Returns:
        Normalized sequence string
    """
    return sequence.strip().upper()

