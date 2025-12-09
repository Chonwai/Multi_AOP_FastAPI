"""
FastAPI application entry point
"""

from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-AOP Prediction API",
    description="API for predicting antioxidant peptides using multi-view deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("Starting Multi-AOP FastAPI application")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API running on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Device: {settings.DEVICE}")
    # Model will be loaded here via dependency injection in future stages
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Multi-AOP FastAPI application")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-AOP Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status with model loading information
    """
    # TODO: Check model loading status when model manager is implemented
    return {
        "status": "healthy",
        "model_loaded": False,  # Will be updated when model manager is implemented
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": settings.ENVIRONMENT
    }

