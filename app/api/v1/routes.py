"""
API v1 routes
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, status

from app.models.request import SinglePredictionRequest, BatchPredictionRequest
from app.models.response import (
    SinglePredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    PredictionResult
)
from app.api.dependencies import PredictionServiceDep, ModelManagerDep
from app.services.model_manager import ModelManager
from app.services.predictor import PredictionService
from app.utils.exceptions import ValidationError, ModelLoadError, PredictionError
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post(
    "/predict/single",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict single sequence",
    description="Predict whether a single amino acid sequence is an AOP (Antioxidant Peptide)"
)
async def predict_single(
    request: SinglePredictionRequest,
    prediction_service: PredictionServiceDep
) -> SinglePredictionResponse:
    """
    Predict AOP for a single sequence
    
    Args:
        request: Single prediction request containing sequence
        prediction_service: PredictionService instance (injected)
    
    Returns:
        SinglePredictionResponse with prediction results
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Single prediction request for sequence: {request.sequence[:20]}...")
        result = prediction_service.predict_single(request.sequence)
        
        return SinglePredictionResponse(
            sequence=result["sequence"],
            prediction=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            is_aop=result["is_aop"],
            message="Prediction completed successfully"
        )
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e.message}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in single prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict batch of sequences",
    description="Predict whether multiple amino acid sequences are AOPs (Antioxidant Peptides)"
)
async def predict_batch(
    request: BatchPredictionRequest,
    prediction_service: PredictionServiceDep
) -> BatchPredictionResponse:
    """
    Predict AOP for a batch of sequences
    
    Args:
        request: Batch prediction request containing list of sequences
        prediction_service: PredictionService instance (injected)
    
    Returns:
        BatchPredictionResponse with prediction results for all sequences
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Batch prediction request for {len(request.sequences)} sequences")
        result = prediction_service.predict_batch(request.sequences)
        
        # Convert results to PredictionResult objects
        prediction_results = [
            PredictionResult(
                sequence=r["sequence"],
                prediction=r["prediction"],
                probability=r["probability"],
                confidence=r["confidence"],
                is_aop=r["is_aop"]
            )
            for r in result["results"]
        ]
        
        return BatchPredictionResponse(
            total=result["total"],
            results=prediction_results,
            processing_time_seconds=result["processing_time_seconds"]
        )
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {e.message}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Get information about the loaded model"
)
async def get_model_info(
    model_manager: ModelManagerDep
) -> ModelInfoResponse:
    """
    Get model information
    
    Args:
        model_manager: ModelManager instance (injected)
    
    Returns:
        ModelInfoResponse with model information
    
    Raises:
        HTTPException: If model is not loaded
    """
    try:
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Please wait for model initialization."
            )
        
        model_path = model_manager.get_model_path()
        device = model_manager.get_device()
        
        return ModelInfoResponse(
            model_version="1.0.0",
            model_path=str(model_path) if model_path else settings.MODEL_PATH,
            device=str(device) if device else settings.DEVICE,
            seq_length=settings.SEQ_LENGTH,
            loaded_at=datetime.utcnow().isoformat() + "Z",  # TODO: Store actual load time
            is_loaded=True
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model information: {str(e)}"
        )

