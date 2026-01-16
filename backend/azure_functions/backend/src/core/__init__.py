"""Core primitives shared across modules"""

from src.core.interfaces import (
    Chunk,
    Query,
    ModelAnswer,
    RetrievalResult,
    EvaluationScore,
)
from src.core.exceptions import (
    RAGEvalException,
    ConfigurationError,
    DatabaseError,
    AzureServiceError,
    ValidationError,
)
from src.core.config import Config
from src.core.logging import get_logger, setup_logging

__all__ = [
    # Interfaces
    "Chunk",
    "Query",
    "ModelAnswer",
    "RetrievalResult",
    "EvaluationScore",
    # Exceptions
    "RAGEvalException",
    "ConfigurationError",
    "DatabaseError",
    "AzureServiceError",
    "ValidationError",
    # Configuration
    "Config",
    # Logging
    "get_logger",
    "setup_logging",
]
