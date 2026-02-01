"""RAGLab - Lean RAG Evaluation Framework"""

# Core components
from .core import *
from .evaluation import *
from .indexing import *

# Main pipeline components
from .eval import (
    RAGEvaluationPipeline, run_evaluation, load_evaluation_results, print_evaluation_summary
)
from .agent import (
    SimpleRAGAgent, ConversationalRAGAgent, AgentEvaluator,
    create_simple_rag_agent, create_conversational_rag_agent, evaluate_agent_performance
)

__version__ = "0.1.0"

__all__ = [
    # Main pipeline
    'RAGEvaluationPipeline', 'run_evaluation', 'load_evaluation_results', 'print_evaluation_summary',
    # Agent components
    'SimpleRAGAgent', 'ConversationalRAGAgent', 'AgentEvaluator',
    'create_simple_rag_agent', 'create_conversational_rag_agent', 'evaluate_agent_performance',
    # All exports from core, evaluation, and indexing modules
]