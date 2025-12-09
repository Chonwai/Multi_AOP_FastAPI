"""
Prediction Service
Handles prediction logic for single and batch sequences
"""

from typing import List, Dict, Any
import torch
import time

from app.config import settings
from app.core.data.dataloader import create_in_memory_loader
from app.services.model_manager import ModelManager
from app.utils.exceptions import PredictionError, ValidationError
from app.utils.validators import validate_sequences, normalize_sequence
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    Prediction service for AOP prediction
    
    Uses ModelManager to get the model and processes sequences
    through the data pipeline for prediction.
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Initialize prediction service
        
        Args:
            model_manager: ModelManager instance (creates new if None)
        """
        self.model_manager = model_manager or ModelManager()
        self.seq_length = settings.SEQ_LENGTH
        self.batch_size = settings.BATCH_SIZE
    
    def predict_single(self, sequence: str) -> Dict[str, Any]:
        """
        Predict AOP for a single sequence
        
        Args:
            sequence: Amino acid sequence string
        
        Returns:
            Dictionary containing prediction results:
            - sequence: Input sequence
            - prediction: Binary prediction (0 or 1)
            - probability: Prediction probability (0-1)
            - confidence: Confidence level (low/medium/high)
            - is_aop: Boolean indicating if predicted as AOP
        
        Raises:
            ValidationError: If sequence validation fails
            PredictionError: If prediction fails
        """
        try:
            # Validate and normalize sequence
            from app.utils.validators import validate_sequence
            
            is_valid, error_msg, normalized_seq = validate_sequence(
                sequence,
                min_length=2,
                max_length=self.seq_length
            )
            
            if not is_valid:
                raise ValidationError(error_msg, "sequence")
            
            # Get model
            model = self.model_manager.get_model()
            device = self.model_manager.get_device()
            
            # Create data loader
            data_loader = create_in_memory_loader(
                sequences=[normalized_seq],
                batch_size=1,
                seq_length=self.seq_length,
                shuffle=False
            )
            
            # Perform prediction
            with torch.no_grad():
                for batch in data_loader:
                    # Move data to device
                    sequences = batch['sequences'].to(device)
                    x = batch['x'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_attr = batch['edge_attr'].to(device)
                    batch_idx = batch['batch'].to(device)
                    
                    # Forward pass
                    _, _, _, _, _, outputs = model(
                        sequences, x, edge_index, edge_attr, batch_idx
                    )
                    
                    # Extract probability
                    probability = outputs.squeeze().cpu().item()
                    prediction = 1 if probability > 0.5 else 0
                    
                    # Determine confidence
                    confidence = self._get_confidence(probability)
                    
                    return {
                        "sequence": normalized_seq,
                        "prediction": int(prediction),
                        "probability": float(probability),
                        "confidence": confidence,
                        "is_aop": bool(prediction == 1)
                    }
            
            raise PredictionError("No prediction result returned", sequence)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Prediction error for sequence '{sequence[:20]}...': {str(e)}")
            raise PredictionError(str(e), sequence) from e
    
    def predict_batch(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Predict AOP for a batch of sequences
        
        Args:
            sequences: List of amino acid sequence strings
        
        Returns:
            Dictionary containing:
            - total: Total number of sequences
            - results: List of prediction results (same format as predict_single)
            - processing_time_seconds: Time taken for prediction
        
        Raises:
            ValidationError: If batch validation fails
            PredictionError: If prediction fails
        """
        start_time = time.time()
        
        try:
            # Validate sequences
            is_valid, error_msg, normalized_sequences = validate_sequences(
                sequences,
                min_length=2,
                max_length=self.seq_length,
                max_batch_size=settings.MAX_BATCH_SIZE
            )
            
            if not is_valid:
                raise ValidationError(error_msg, "sequences")
            
            # Get model
            model = self.model_manager.get_model()
            device = self.model_manager.get_device()
            
            # Create data loader
            data_loader = create_in_memory_loader(
                sequences=normalized_sequences,
                batch_size=self.batch_size,
                seq_length=self.seq_length,
                shuffle=False
            )
            
            # Perform batch prediction
            all_results = []
            with torch.no_grad():
                for batch in data_loader:
                    # Move data to device
                    sequences_tensor = batch['sequences'].to(device)
                    x = batch['x'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_attr = batch['edge_attr'].to(device)
                    batch_idx = batch['batch'].to(device)
                    
                    # Forward pass
                    _, _, _, _, _, outputs = model(
                        sequences_tensor, x, edge_index, edge_attr, batch_idx
                    )
                    
                    # Process outputs
                    probabilities = outputs.squeeze().cpu().numpy()
                    if probabilities.ndim == 0:
                        probabilities = [probabilities.item()]
                    else:
                        probabilities = probabilities.tolist()
                    
                    # Create results for this batch
                    batch_start_idx = len(all_results)
                    for i, prob in enumerate(probabilities):
                        seq_idx = batch_start_idx + i
                        if seq_idx < len(normalized_sequences):
                            probability = float(prob)
                            prediction = 1 if probability > 0.5 else 0
                            confidence = self._get_confidence(probability)
                            
                            all_results.append({
                                "sequence": normalized_sequences[seq_idx],
                                "prediction": int(prediction),
                                "probability": probability,
                                "confidence": confidence,
                                "is_aop": bool(prediction == 1)
                            })
            
            processing_time = time.time() - start_time
            
            return {
                "total": len(all_results),
                "results": all_results,
                "processing_time_seconds": round(processing_time, 3)
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise PredictionError(f"Batch prediction failed: {str(e)}") from e
    
    def _get_confidence(self, probability: float) -> str:
        """
        Determine confidence level based on probability
        
        Args:
            probability: Prediction probability (0-1)
        
        Returns:
            Confidence level string (low/medium/high)
        """
        distance_from_threshold = abs(probability - 0.5)
        
        if distance_from_threshold >= 0.3:
            return "high"
        elif distance_from_threshold >= 0.15:
            return "medium"
        else:
            return "low"

