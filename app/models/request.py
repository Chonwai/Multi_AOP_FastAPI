"""
Request models for API endpoints
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class SinglePredictionRequest(BaseModel):
    """
    Request model for single sequence prediction
    
    Attributes:
        sequence: Amino acid sequence string (2-50 characters)
    """
    sequence: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Amino acid sequence to predict (2-50 amino acids)"
    )
    
    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate and normalize sequence"""
        v = v.strip().upper()
        if not v:
            raise ValueError("Sequence cannot be empty")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "sequence": "MKLLVVVFCLVLAAP"
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch sequence prediction
    
    Attributes:
        sequences: List of amino acid sequence strings (1-100 sequences)
    """
    sequences: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of amino acid sequences to predict (1-100 sequences)"
    )
    
    @field_validator('sequences')
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        """Validate and normalize sequences"""
        if not v:
            raise ValueError("Sequences list cannot be empty")
        
        normalized = []
        for i, seq in enumerate(v):
            if not isinstance(seq, str):
                raise ValueError(f"Sequence at index {i} must be a string")
            seq = seq.strip().upper()
            if not seq:
                raise ValueError(f"Sequence at index {i} cannot be empty")
            normalized.append(seq)
        
        return normalized
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "sequences": [
                    "MKLLVVVFCLVLAAP",
                    "ACDEFGHIKLMNPQRSTVWY"
                ]
            }
        }
    }


