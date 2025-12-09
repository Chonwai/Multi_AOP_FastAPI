"""
Configuration management using Pydantic Settings (Singleton Pattern)

This module provides a singleton configuration manager that loads settings
from environment variables and .env files. It uses Pydantic Settings for
type-safe configuration management with validation.
"""

import os
import threading
from pathlib import Path
from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """
    Application settings (Singleton pattern via module-level instance)
    
    Settings are loaded from:
    1. Environment variables
    2. .env file (if present)
    3. Default values
    
    All settings can be overridden via environment variables.
    """
    
    # API Configuration
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API port"
    )
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins (comma-separated list or JSON array)"
    )
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="predict/model/best_model_Oct13.pth",
        description="Path to the trained model file (relative to project root or absolute)"
    )
    DEVICE: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device to use for inference (cpu/cuda)"
    )
    
    # Sequence Processing Configuration
    SEQ_LENGTH: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum sequence length"
    )
    BATCH_SIZE: int = Field(
        default=16,
        ge=1,
        le=500,
        description="Batch size for prediction"
    )
    MAX_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum batch size for batch prediction endpoint"
    )
    
    # Logging Configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Environment
    ENVIRONMENT: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Environment"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            # Handle comma-separated string
            if v.startswith("[") and v.endswith("]"):
                # JSON array string
                import json
                return json.loads(v)
            else:
                # Comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @field_validator("MODEL_PATH")
    @classmethod
    def validate_model_path(cls, v):
        """Validate model path exists (warning only, not strict for MVP)"""
        # For MVP, we'll just check if it's a valid path format
        # Actual file existence will be checked when loading the model
        if not v:
            raise ValueError("MODEL_PATH cannot be empty")
        return v
    
    @model_validator(mode="after")
    def validate_batch_sizes(self):
        """Ensure MAX_BATCH_SIZE >= BATCH_SIZE"""
        if self.MAX_BATCH_SIZE < self.BATCH_SIZE:
            raise ValueError(
                f"MAX_BATCH_SIZE ({self.MAX_BATCH_SIZE}) must be >= BATCH_SIZE ({self.BATCH_SIZE})"
            )
        return self
    
    def get_model_path(self) -> Path:
        """Get model path as Path object, resolving relative to project root"""
        model_path = Path(self.MODEL_PATH)
        if not model_path.is_absolute():
            # Resolve relative to project root (where .env or main.py is)
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_path
        return model_path.resolve()
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"


# Singleton instance with thread-safe initialization
_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """
    Get settings instance (Thread-safe Singleton pattern)
    
    Returns:
        Settings: The singleton settings instance
        
    Raises:
        ValueError: If settings validation fails
    """
    global _settings
    if _settings is None:
        with _settings_lock:
            # Double-check locking pattern
            if _settings is None:
                try:
                    _settings = Settings()
                except Exception as e:
                    raise ValueError(
                        f"Failed to load settings: {e}. "
                        "Please check your .env file and environment variables."
                    ) from e
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing)
    
    Returns:
        Settings: New settings instance
    """
    global _settings
    with _settings_lock:
        _settings = None
    return get_settings()


# Module-level settings instance (lazy initialization)
settings = get_settings()

