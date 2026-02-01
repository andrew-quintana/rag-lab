"""Core data structures, I/O utilities, and text processing"""

from .interfaces import (
    Query, Chunk, RetrievalResult, ModelAnswer, EvaluationScore,
    JudgeEvaluationResult, MetaEvaluationResult, JudgeMetricScores,
    JudgePerformanceMetrics, BEIRMetricsResult, EvaluationExample,
    BiasCorrectedAccuracy, EvaluationResult
)
from .io import DataLoader, RunManager, load_evaluation_dataset, save_evaluation_results, create_new_run
from .chunking import chunk_text_fixed_size, chunk_text

__all__ = [
    # Interfaces
    'Query', 'Chunk', 'RetrievalResult', 'ModelAnswer', 'EvaluationScore',
    'JudgeEvaluationResult', 'MetaEvaluationResult', 'JudgeMetricScores',
    'JudgePerformanceMetrics', 'BEIRMetricsResult', 'EvaluationExample',
    'BiasCorrectedAccuracy', 'EvaluationResult',
    # I/O
    'DataLoader', 'RunManager', 'load_evaluation_dataset', 'save_evaluation_results', 'create_new_run',
    # Chunking
    'chunk_text_fixed_size', 'chunk_text'
]