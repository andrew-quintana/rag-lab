# Phase 3.5 Decisions — LangGraph Infrastructure Setup

## Context

This document captures implementation decisions made during Phase 3.5 that are not explicitly covered in PRD001.md or RFC001.md.

## Decision 1: LangGraph Version Selection

**Decision**: Use `langgraph>=0.2.0` and `langchain-core>=0.3.0` as dependencies.

**Rationale**:
- LangGraph 0.2.0+ provides stable StateGraph API and TypedDict support
- langchain-core 0.3.0+ is required dependency and provides core abstractions
- Minimum version pins ensure compatibility while allowing patch updates
- These versions are stable and well-documented

**Implementation**: Added to `backend/requirements.txt` with version pins.

## Decision 2: State Schema Design (TypedDict)

**Decision**: Use `TypedDict` for state schema definition instead of Pydantic models.

**Rationale**:
- TypedDict is the recommended pattern in LangGraph documentation
- Provides type safety without runtime validation overhead
- Simpler and more lightweight than Pydantic for state management
- Better integration with LangGraph's state update mechanism
- Allows partial state updates (immutability pattern)

**Implementation**: `JudgeEvaluationState` defined as TypedDict in `graph_base.py`.

## Decision 3: Pure Function Node Pattern

**Decision**: Implement LangGraph nodes as pure functions that wrap existing evaluator classes.

**Rationale**:
- Follows LangGraph best practices for node design
- Maintains immutability (nodes return partial state updates, don't mutate input)
- Keeps existing evaluator classes unchanged (no refactoring needed)
- Enhances testability (pure functions are easier to test)
- Clear separation of concerns (orchestration vs. evaluation logic)

**Implementation**: Node functions in `test_graph.py` extract from state, call evaluator methods, return partial updates.

## Decision 4: State Validation at Boundaries

**Decision**: Implement state validation function `validate_initial_state()` to check required fields.

**Rationale**:
- Catches errors early before graph execution
- Provides clear error messages for missing or invalid fields
- Follows LangGraph best practice of validation at node boundaries
- Prevents downstream errors from propagating through graph

**Implementation**: `validate_initial_state()` function in `graph_base.py` validates required fields and non-empty strings.

## Decision 5: Config Handling in State

**Decision**: Make `config` optional in state, defaulting to `Config.from_env()` when not present.

**Rationale**:
- Allows flexibility: can pass explicit config or use defaults
- Reduces boilerplate when config is not needed
- Consistent with existing evaluator patterns
- Helper function `get_config_from_state()` encapsulates logic

**Implementation**: `config` field is `Optional[Config]` in state, `get_config_from_state()` provides default.

## Decision 6: Test Graph Design

**Decision**: Create minimal test graph with correctness and hallucination nodes only (no conditional branching to cost/impact).

**Rationale**:
- Validates LangGraph setup works with existing evaluators
- Demonstrates key patterns needed for Phase 7 (state passing, node functions)
- Keeps test graph simple and focused on infrastructure validation
- Full orchestrator with conditional edges will be implemented in Phase 7

**Implementation**: `test_graph.py` contains simple 2-node graph: correctness → hallucination → END.

## Decision 7: Reasoning Collection Pattern

**Decision**: Store reasoning as a list of strings in state, with each node appending its reasoning.

**Rationale**:
- Allows accumulation of reasoning from multiple nodes
- Simple and flexible data structure
- Easy to combine into final reasoning trace in Phase 7
- Maintains immutability (nodes return new list, not mutate existing)

**Implementation**: `reasoning: List[str]` in state, nodes append to list in partial state update.

## Decision 8: File Organization

**Decision**: Separate graph infrastructure (`graph_base.py`) from test graph implementation (`test_graph.py`).

**Rationale**:
- Clear separation: infrastructure vs. example/test usage
- `graph_base.py` contains reusable components for Phase 7
- `test_graph.py` is for validation only (can be removed or kept as reference)
- Follows modular design principles

**Implementation**: Two separate files with distinct purposes.


