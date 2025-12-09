"""
FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

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
    # Model will be loaded here via dependency injection
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    pass


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
    """Health check endpoint"""
    # TODO: Check model loading status
    return {
        "status": "healthy",
        "model_loaded": False,  # Will be updated when model manager is implemented
        "timestamp": "2024-12-19T00:00:00"  # Will use actual timestamp
    }

