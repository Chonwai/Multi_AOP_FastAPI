"""
Custom exception classes
"""


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass

