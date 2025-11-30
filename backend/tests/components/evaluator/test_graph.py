"""Test graph implementation to validate LangGraph setup with existing evaluators"""

from typing import Literal

from langgraph.graph import StateGraph, END

from rag_eval.services.evaluator.graph_base import (
    JudgeEvaluationState,
    validate_initial_state,
    get_config_from_state,
)
from rag_eval.services.evaluator.correctness import CorrectnessEvaluator
from rag_eval.services.evaluator.hallucination import HallucinationEvaluator


def correctness_node(state: JudgeEvaluationState) -> dict:
    """LangGraph node that calls CorrectnessEvaluator.
    
    This is a pure function node following LangGraph best practices:
    - Takes state as input
    - Returns partial state update (immutable)
    - Does not mutate input state
    
    Args:
        state: Current workflow state
        
    Returns:
        Partial state update with correctness_binary and reasoning
    """
    config = get_config_from_state(state)
    evaluator = CorrectnessEvaluator(config=config)
    
    result = evaluator.classify_correctness(
        query=state["query"],
        model_answer=state["model_answer"],
        reference_answer=state["reference_answer"]
    )
    
    # Return partial state update
    reasoning_update = state.get("reasoning", [])
    reasoning_update.append(f"Correctness classification: {result}")
    
    return {
        "correctness_binary": result,
        "reasoning": reasoning_update
    }


def hallucination_node(state: JudgeEvaluationState) -> dict:
    """LangGraph node that calls HallucinationEvaluator.
    
    This is a pure function node following LangGraph best practices:
    - Takes state as input
    - Returns partial state update (immutable)
    - Does not mutate input state
    
    Args:
        state: Current workflow state
        
    Returns:
        Partial state update with hallucination_binary and reasoning
    """
    config = get_config_from_state(state)
    evaluator = HallucinationEvaluator(config=config)
    
    result = evaluator.classify_hallucination(
        retrieved_context=state["retrieved_context"],
        model_answer=state["model_answer"]
    )
    
    # Return partial state update
    reasoning_update = state.get("reasoning", [])
    reasoning_update.append(f"Hallucination classification: {result}")
    
    return {
        "hallucination_binary": result,
        "reasoning": reasoning_update
    }


def should_continue(state: JudgeEvaluationState) -> Literal["end", "continue"]:
    """Conditional edge function to determine if workflow should continue.
    
    For this test graph, we always end after hallucination check.
    In Phase 7, this will route to cost/impact nodes if hallucination detected.
    
    Args:
        state: Current workflow state
        
    Returns:
        "end" to end workflow, "continue" to proceed (not used in test graph)
    """
    return "end"


def create_test_graph() -> StateGraph:
    """Create a simple test graph with correctness and hallucination nodes.
    
    This graph validates that LangGraph works with existing evaluator infrastructure.
    The graph flow is:
    1. correctness_node (always executes)
    2. hallucination_node (always executes)
    3. END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(JudgeEvaluationState)
    
    # Add nodes
    workflow.add_node("correctness", correctness_node)
    workflow.add_node("hallucination", hallucination_node)
    
    # Set entry point
    workflow.set_entry_point("correctness")
    
    # Add edges
    workflow.add_edge("correctness", "hallucination")
    workflow.add_conditional_edges(
        "hallucination",
        should_continue,
        {
            "end": END,
        }
    )
    
    return workflow.compile()


def run_test_evaluation(
    query: str,
    retrieved_context: list,
    model_answer: str,
    reference_answer: str,
    config=None
) -> JudgeEvaluationState:
    """Run test evaluation using the test graph.
    
    This function validates that the LangGraph setup works correctly
    with existing evaluator classes.
    
    Args:
        query: Original user query
        retrieved_context: List of RetrievalResult objects
        model_answer: Generated answer from RAG system
        reference_answer: Gold reference answer
        config: Optional Config instance
        
    Returns:
        Final state after graph execution
    """
    # Create initial state
    initial_state: JudgeEvaluationState = {
        "query": query,
        "retrieved_context": retrieved_context,
        "model_answer": model_answer,
        "reference_answer": reference_answer,
        "correctness_binary": None,
        "hallucination_binary": None,
        "hallucination_cost": None,
        "hallucination_impact": None,
        "reasoning": [],
        "config": config,
    }
    
    # Validate initial state
    validate_initial_state(initial_state)
    
    # Create and run graph
    graph = create_test_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state

