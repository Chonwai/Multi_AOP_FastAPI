"""
Configuration management using Pydantic Settings (Singleton Pattern)
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings (Singleton pattern via module-level instance)"""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host address")
    API_PORT: int = Field(default=8000, description="API port")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="predict/model/best_model_Oct13.pth",
        description="Path to the trained model file"
    )
    DEVICE: str = Field(
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
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    
    # Environment
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment (development/production)"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings instance (Singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Module-level settings instance
settings = get_settings()

