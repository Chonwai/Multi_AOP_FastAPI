"""
Response models for API endpoints
"""

from typing import List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class PredictionResult(BaseModel):
    """
    Single prediction result
    
    Attributes:
        sequence: Input sequence
        prediction: Binary prediction (0 or 1)
        probability: Prediction probability (0-1)
        confidence: Confidence level (low/medium/high)
        is_aop: Boolean indicating if predicted as AOP
    """
    sequence: str = Field(..., description="Input amino acid sequence")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=non-AOP, 1=AOP)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Confidence level")
    is_aop: bool = Field(..., description="Whether the sequence is predicted as AOP")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "sequence": "MKLLVVVFCLVLAAP",
                "prediction": 1,
                "probability": 0.85,
                "confidence": "high",
                "is_aop": True
            }
        }
    }


class SinglePredictionResponse(BaseModel):
    """
    Response model for single prediction endpoint
    """
    sequence: str = Field(..., description="Input amino acid sequence")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Confidence level")
    is_aop: bool = Field(..., description="Whether predicted as AOP")
    message: str = Field(default="Prediction completed successfully", description="Status message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "sequence": "MKLLVVVFCLVLAAP",
                "prediction": 1,
                "probability": 0.85,
                "confidence": "high",
                "is_aop": True,
                "message": "Prediction completed successfully"
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction endpoint
    """
    total: int = Field(..., ge=0, description="Total number of sequences processed")
    results: List[PredictionResult] = Field(..., description="List of prediction results")
    processing_time_seconds: float = Field(..., ge=0.0, description="Time taken for prediction in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 2,
                "results": [
                    {
                        "sequence": "MKLLVVVFCLVLAAP",
                        "prediction": 1,
                        "probability": 0.85,
                        "confidence": "high",
                        "is_aop": True
                    },
                    {
                        "sequence": "ACDEFGHIKLMNPQRSTVWY",
                        "prediction": 0,
                        "probability": 0.23,
                        "confidence": "low",
                        "is_aop": False
                    }
                ],
                "processing_time_seconds": 2.5
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """
    Response model for model information endpoint
    """
    model_version: str = Field(..., description="Model version")
    model_path: str = Field(..., description="Path to the model file")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    seq_length: int = Field(..., description="Maximum sequence length")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")
    is_loaded: bool = Field(..., description="Whether model is currently loaded")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_version": "1.0.0",
                "model_path": "predict/model/best_model_Oct13.pth",
                "device": "cpu",
                "seq_length": 50,
                "loaded_at": "2024-12-19T10:00:00Z",
                "is_loaded": True
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint
    """
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")
    environment: str = Field(..., description="Environment (development/production)")
    message: str = Field(default="", description="Additional status message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-12-19T10:00:00Z",
                "environment": "development",
                "message": ""
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Unified error response model
    """
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    status_code: int = Field(..., description="HTTP status code")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Sequence length must be between 2 and 50 amino acids",
                "error_type": "ValidationError",
                "status_code": 422
            }
        }
    }


