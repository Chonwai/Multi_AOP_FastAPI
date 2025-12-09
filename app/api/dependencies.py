"""
Dependency injection for FastAPI
"""

from typing import Annotated
from fastapi import Depends

from app.services.model_manager import ModelManager
from app.services.predictor import PredictionService


def get_model_manager() -> ModelManager:
    """
    Dependency function to get ModelManager instance (Singleton)
    
    Returns:
        ModelManager instance
    """
    return ModelManager()


def get_prediction_service(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)]
) -> PredictionService:
    """
    Dependency function to get PredictionService instance
    
    Args:
        model_manager: ModelManager instance (injected)
    
    Returns:
        PredictionService instance
    """
    return PredictionService(model_manager=model_manager)


# Type aliases for cleaner dependency injection
ModelManagerDep = Annotated[ModelManager, Depends(get_model_manager)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]

