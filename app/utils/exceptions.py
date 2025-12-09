"""
Custom exception classes for the Multi-AOP API

These exceptions provide more context than generic exceptions
and can be caught by FastAPI error handlers for proper HTTP responses.
"""


class ValidationError(Exception):
    """
    Raised when input validation fails
    
    Attributes:
        message: Error message describing the validation failure
        field: Optional field name that failed validation
    """
    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.field:
            return f"Validation error for field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class ModelLoadError(Exception):
    """
    Raised when model loading fails
    
    Attributes:
        message: Error message describing the loading failure
        model_path: Path to the model file that failed to load
    """
    def __init__(self, message: str, model_path: str | None = None):
        self.message = message
        self.model_path = model_path
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.model_path:
            return f"Failed to load model from '{self.model_path}': {self.message}"
        return f"Model loading error: {self.message}"


class PredictionError(Exception):
    """
    Raised when prediction fails
    
    Attributes:
        message: Error message describing the prediction failure
        sequence: Optional sequence that caused the error
    """
    def __init__(self, message: str, sequence: str | None = None):
        self.message = message
        self.sequence = sequence
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.sequence:
            return f"Prediction error for sequence '{self.sequence[:20]}...': {self.message}"
        return f"Prediction error: {self.message}"

