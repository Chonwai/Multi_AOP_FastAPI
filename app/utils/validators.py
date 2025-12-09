"""
Input validation utilities
"""

# Standard 20 amino acids
STANDARD_AMINO_ACIDS = set(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 50


def validate_sequence(sequence: str, min_length: int = MIN_SEQUENCE_LENGTH, 
                     max_length: int = MAX_SEQUENCE_LENGTH) -> tuple[bool, str]:
    """
    Validate amino acid sequence
    
    Args:
        sequence: Amino acid sequence string
        min_length: Minimum sequence length
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(sequence, str):
        return False, "Sequence must be a string"
    
    sequence = sequence.strip().upper()
    
    if len(sequence) < min_length:
        return False, f"Sequence length must be at least {min_length} amino acids"
    
    if len(sequence) > max_length:
        return False, f"Sequence length must be at most {max_length} amino acids"
    
    # Check for invalid amino acid characters
    invalid_chars = set(sequence) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        return False, f"Sequence contains invalid amino acid characters: {', '.join(sorted(invalid_chars))}"
    
    return True, ""


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> tuple[bool, str]:
    """
    Validate batch size
    
    Args:
        batch_size: Number of sequences in batch
        max_batch_size: Maximum allowed batch size
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if batch_size < 1:
        return False, "Batch size must be at least 1"
    
    if batch_size > max_batch_size:
        return False, f"Batch size cannot exceed {max_batch_size}"
    
    return True, ""

