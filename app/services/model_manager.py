"""
Model Manager (Singleton Pattern)
Manages the lifecycle of the PyTorch model

Modified to use CoreAdapter for loading upstream core algorithm models.
"""

import threading
from pathlib import Path
from typing import Optional
import torch

from app.config import settings
from app.adapters.core_adapter import get_core_adapter
from app.utils.exceptions import ModelLoadError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Model Manager using Singleton Pattern
    
    Ensures the model is loaded only once and provides thread-safe access.
    The model is loaded lazily on first access.
    """
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation with thread safety"""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager (only called once due to singleton)"""
        if self._initialized:
            return
        
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._model_path: Optional[Path] = None
        self._load_lock = threading.Lock()
        self._initialized = True
    
    def load_model(self, model_path: Optional[str] = None, device: Optional[str] = None) -> torch.nn.Module:
        """
        Load the model (thread-safe)
        
        Args:
            model_path: Path to model file (uses config if None)
            device: Device to load model on (uses config if None)
        
        Returns:
            Loaded model instance
        
        Raises:
            ModelLoadError: If model loading fails
        """
        with self._load_lock:
            if self._model is not None:
                logger.info("Model already loaded, returning existing instance")
                return self._model
            
            try:
                # Determine model path
                if model_path is None:
                    model_path = settings.get_model_path()
                else:
                    model_path = Path(model_path)
                    if not model_path.is_absolute():
                        project_root = Path(__file__).parent.parent.parent
                        model_path = project_root / model_path
                    model_path = model_path.resolve()
                
                self._model_path = model_path
                
                # Check if model file exists
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Determine device
                if device is None:
                    device_str = settings.DEVICE
                else:
                    device_str = device
                
                if device_str == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device_str = "cpu"
                
                self._device = torch.device(device_str)
                
                logger.info(f"Loading model from {model_path} on device {self._device}")
                
                # Use CoreAdapter to load model from upstream core algorithm
                # This delegates to predict/aop_def.py CombinedModel
                adapter = get_core_adapter()
                model = adapter.load_model(str(model_path), self._device)
                
                # Model is already loaded and configured by adapter
                # No need for manual checkpoint loading or state_dict filtering
                
                self._model = model
                
                logger.info(f"Model loaded successfully via CoreAdapter on {self._device}")
                return self._model
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to load model: {error_msg}")
                raise ModelLoadError(error_msg, str(model_path)) from e
    
    def get_model(self) -> torch.nn.Module:
        """
        Get the loaded model (loads if not already loaded)
        
        Returns:
            Model instance
        
        Raises:
            ModelLoadError: If model is not loaded and loading fails
        """
        if self._model is None:
            return self.load_model()
        return self._model
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
    
    def get_device(self) -> Optional[torch.device]:
        """Get the device the model is loaded on"""
        return self._device
    
    def get_model_path(self) -> Optional[Path]:
        """Get the path to the loaded model"""
        return self._model_path
    
    def reload_model(self, model_path: Optional[str] = None, device: Optional[str] = None) -> torch.nn.Module:
        """
        Reload the model (useful for testing or model updates)
        
        Args:
            model_path: Path to model file
            device: Device to load model on
        
        Returns:
            Reloaded model instance
        """
        with self._load_lock:
            self._model = None
            return self.load_model(model_path, device)

