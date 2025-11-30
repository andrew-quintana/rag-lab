"""Custom error types"""


class RAGEvalException(Exception):
    """Base exception for RAG Evaluation Platform"""
    pass


class ConfigurationError(RAGEvalException):
    """Raised when configuration is invalid or missing"""
    pass


class DatabaseError(RAGEvalException):
    """Raised when database operations fail"""
    pass


class AzureServiceError(RAGEvalException):
    """Raised when Azure service calls fail"""
    pass


class ValidationError(RAGEvalException):
    """Raised when input validation fails"""
    pass


class EvaluationError(RAGEvalException):
    """Raised when evaluation pipeline fails"""
    pass

