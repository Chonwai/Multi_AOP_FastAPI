"""
FastAPI application entry point
"""

from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logging_config import setup_logging, get_logger
from app.services.model_manager import ModelManager
from app.api.v1.routes import router as v1_router
from app.api.middleware import register_exception_handlers
from app.models.response import HealthResponse

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

# Register exception handlers
register_exception_handlers(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(v1_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("Starting Multi-AOP FastAPI application")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API running on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Device: {settings.DEVICE}")
    
    # Pre-load model on startup (optional, model will be loaded lazily if not done here)
    try:
        model_manager = ModelManager()
        if not model_manager.is_loaded():
            logger.info("Pre-loading model on startup...")
            model_manager.load_model()
            logger.info("Model loaded successfully on startup")
        else:
            logger.info("Model already loaded")
    except Exception as e:
        logger.warning(f"Failed to pre-load model on startup: {str(e)}")
        logger.warning("Model will be loaded lazily on first prediction request")
    
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
        "docs": "/docs",
        "api_version": "v1"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status with model loading information
    """
    try:
        model_manager = ModelManager()
        model_loaded = model_manager.is_loaded()
        
        status_value = "healthy" if model_loaded else "unhealthy"
        message = "" if model_loaded else "Model not loaded. Please wait for initialization."
        
        return HealthResponse(
            status=status_value,
            model_loaded=model_loaded,
            timestamp=datetime.utcnow().isoformat() + "Z",
            environment=settings.ENVIRONMENT,
            message=message
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.utcnow().isoformat() + "Z",
            environment=settings.ENVIRONMENT,
            message=f"Health check failed: {str(e)}"
        )

