"""RAG Evaluation Platform - Main Package

This package provides a complete RAG (Retrieval-Augmented Generation) evaluation
platform for testing and iterating on RAG system components.
"""

__version__ = "0.1.0"

# Core primitives and interfaces
from src.core import (
    Chunk,
    Query,
    ModelAnswer,
    RetrievalResult,
    EvaluationScore,
    Config,
    get_logger,
    setup_logging,
)

# Exceptions
from src.core import (
    RAGEvalException,
    ConfigurationError,
    DatabaseError,
    AzureServiceError,
    ValidationError,
)

__all__ = [
    # Version
    "__version__",
    # Core interfaces
    "Chunk",
    "Query",
    "ModelAnswer",
    "RetrievalResult",
    "EvaluationScore",
    # Configuration
    "Config",
    # Logging
    "get_logger",
    "setup_logging",
    # Exceptions
    "RAGEvalException",
    "ConfigurationError",
    "DatabaseError",
    "AzureServiceError",
    "ValidationError",
]
