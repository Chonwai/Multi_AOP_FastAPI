"""
Logging configuration for the application

For MVP version, we use Python's standard logging library.
For production, consider upgrading to loguru or structlog.
"""

import logging
import sys
from typing import Literal

from app.config import settings


def setup_logging(log_level: str | None = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
                  If None, uses LOG_LEVEL from settings
    """
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging format
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    
    # Basic configuration
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

