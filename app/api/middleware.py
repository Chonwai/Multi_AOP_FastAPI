"""
Exception handlers and middleware for FastAPI
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.response import ErrorResponse
from app.utils.exceptions import ValidationError, ModelLoadError, PredictionError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register custom exception handlers for the FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle ValidationError exceptions"""
        logger.warning(f"Validation error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=exc.message,
                error_type="ValidationError",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            ).model_dump()
        )
    
    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
        """Handle ModelLoadError exceptions"""
        logger.error(f"Model load error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=exc.message,
                error_type="ModelLoadError",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).model_dump()
        )
    
    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
        """Handle PredictionError exceptions"""
        logger.error(f"Prediction error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail=exc.message,
                error_type="PredictionError",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ).model_dump()
        )
    
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, 
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI RequestValidationError (Pydantic validation errors)"""
        errors = exc.errors()
        error_messages = []
        for error in errors:
            field = ".".join(str(loc) for loc in error["loc"])
            error_messages.append(f"{field}: {error['msg']}")
        
        detail = "; ".join(error_messages)
        logger.warning(f"Request validation error: {detail}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=detail,
                error_type="RequestValidationError",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            ).model_dump()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, 
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTPException"""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=str(exc.detail),
                error_type="HTTPException",
                status_code=exc.status_code
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An unexpected error occurred. Please contact support if the problem persists.",
                error_type="InternalServerError",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ).model_dump()
        )

