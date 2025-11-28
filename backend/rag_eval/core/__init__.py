"""Core primitives shared across modules"""

from rag_eval.core.interfaces import (
    Chunk,
    Query,
    ModelAnswer,
    RetrievalResult,
    EvaluationScore,
)
from rag_eval.core.exceptions import (
    RAGEvalException,
    ConfigurationError,
    DatabaseError,
    AzureServiceError,
    ValidationError,
)
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger, setup_logging

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
