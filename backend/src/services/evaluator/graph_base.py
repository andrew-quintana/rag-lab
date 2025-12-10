"""LangGraph infrastructure for evaluation workflows"""

from typing import List, Optional, TypedDict

from src.core.config import Config
from src.core.interfaces import RetrievalResult


class JudgeEvaluationState(TypedDict):
    """State schema for LLM-as-Judge evaluation workflow
    
    This state is passed between LangGraph nodes during the evaluation process.
    All fields are optional except those required for initial state.
    
    Attributes:
        query: Original user query (required for initial state)
        retrieved_context: List of retrieved chunks with similarity scores (required for initial state)
        model_answer: Generated answer from RAG system (required for initial state)
        reference_answer: Gold reference answer for comparison (required for initial state)
        correctness_binary: Correctness classification result (True/False)
        hallucination_binary: Hallucination classification result (True/False)
        risk_direction: System-level risk direction classification (-1 for care avoidance risk, +1 for unexpected cost risk)
        risk_impact: System-level impact magnitude calculation (0-3 scale)
        reasoning: List of reasoning strings from all invoked LLM nodes
        config: Application configuration (optional, defaults to Config.from_env())
    """
    query: str
    retrieved_context: List[RetrievalResult]
    model_answer: str
    reference_answer: str
    correctness_binary: Optional[bool]
    hallucination_binary: Optional[bool]
    risk_direction: Optional[int]
    risk_impact: Optional[float]
    reasoning: List[str]
    config: Optional[Config]


def validate_initial_state(state: JudgeEvaluationState) -> None:
    """Validate that initial state has all required fields.
    
    Args:
        state: State to validate
        
    Raises:
        ValueError: If required fields are missing or empty
    """
    # Check if required fields are present (not None)
    required_fields = ["query", "retrieved_context", "model_answer", "reference_answer"]
    missing_fields = [field for field in required_fields if field not in state or state.get(field) is None]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in initial state: {', '.join(missing_fields)}")
    
    # Validate non-empty strings
    if not state["query"] or not state["query"].strip():
        raise ValueError("Query cannot be empty")
    
    if not state["model_answer"] or not state["model_answer"].strip():
        raise ValueError("Model answer cannot be empty")
    
    if not state["reference_answer"] or not state["reference_answer"].strip():
        raise ValueError("Reference answer cannot be empty")
    
    # Validate retrieved_context is a list (can be empty)
    if not isinstance(state["retrieved_context"], list):
        raise ValueError("retrieved_context must be a list")


def get_config_from_state(state: JudgeEvaluationState) -> Config:
    """Get Config from state, defaulting to Config.from_env() if not present.
    
    Args:
        state: State dictionary
        
    Returns:
        Config instance
    """
    return state.get("config") or Config.from_env()

