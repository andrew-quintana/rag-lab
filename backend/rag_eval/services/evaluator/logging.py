"""Logging and persistence for evaluation results

This module provides functions to log evaluation results to Supabase Postgres database.
All evaluation metrics are stored as JSONB for flexibility and extensibility.

The logging is optional - if query_executor is None, logging is skipped (local-only mode).
Logging failures are handled gracefully and do not fail the evaluation pipeline.
"""

import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import asdict, is_dataclass

from rag_eval.core.interfaces import (
    EvaluationResult,
    JudgeEvaluationResult,
    MetaEvaluationResult,
    BEIRMetricsResult,
    JudgePerformanceMetrics,
    JudgeMetricScores
)
from rag_eval.db.queries import QueryExecutor
from rag_eval.core.logging import get_logger

logger = get_logger("services.evaluator.logging")


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for dataclasses and datetime objects"""
    
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        return super().default(obj)


def _serialize_to_json(obj: Any) -> Dict[str, Any]:
    """
    Serialize a dataclass or object to a JSON-compatible dictionary.
    
    Args:
        obj: Object to serialize (dataclass, dict, or primitive)
        
    Returns:
        JSON-compatible dictionary
    """
    if obj is None:
        return None
    
    if is_dataclass(obj):
        return asdict(obj)
    
    if isinstance(obj, dict):
        return {k: _serialize_to_json(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [_serialize_to_json(item) for item in obj]
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # For primitives, return as-is
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # Fallback: try to convert to dict
    try:
        return asdict(obj)
    except (TypeError, ValueError):
        logger.warning(f"Could not serialize object of type {type(obj)}, using str representation")
        return str(obj)


def _serialize_judge_output(judge_output: JudgeEvaluationResult) -> Dict[str, Any]:
    """
    Serialize JudgeEvaluationResult to JSON-compatible dict.
    
    Args:
        judge_output: JudgeEvaluationResult to serialize
        
    Returns:
        JSON-compatible dictionary
    """
    return {
        "correctness_binary": judge_output.correctness_binary,
        "hallucination_binary": judge_output.hallucination_binary,
        "risk_direction": judge_output.risk_direction,
        "risk_impact": judge_output.risk_impact,
        "reasoning": judge_output.reasoning,
        "failure_mode": judge_output.failure_mode
    }


def _serialize_meta_eval_output(meta_eval_output: MetaEvaluationResult) -> Dict[str, Any]:
    """
    Serialize MetaEvaluationResult to JSON-compatible dict.
    
    Args:
        meta_eval_output: MetaEvaluationResult to serialize
        
    Returns:
        JSON-compatible dictionary including ground truth fields
    """
    return {
        "judge_correct": meta_eval_output.judge_correct,
        "explanation": meta_eval_output.explanation,
        "ground_truth_correctness": meta_eval_output.ground_truth_correctness,
        "ground_truth_hallucination": meta_eval_output.ground_truth_hallucination,
        "ground_truth_risk_direction": meta_eval_output.ground_truth_risk_direction,
        "ground_truth_risk_impact": meta_eval_output.ground_truth_risk_impact
    }


def _serialize_beir_metrics(beir_metrics: BEIRMetricsResult) -> Dict[str, Any]:
    """
    Serialize BEIRMetricsResult to JSON-compatible dict.
    
    Args:
        beir_metrics: BEIRMetricsResult to serialize
        
    Returns:
        JSON-compatible dictionary
    """
    return {
        "recall_at_k": beir_metrics.recall_at_k,
        "precision_at_k": beir_metrics.precision_at_k,
        "ndcg_at_k": beir_metrics.ndcg_at_k
    }


def _serialize_judge_performance_metrics(
    metrics: JudgePerformanceMetrics
) -> Dict[str, Any]:
    """
    Serialize JudgePerformanceMetrics to JSON-compatible dict.
    
    Args:
        metrics: JudgePerformanceMetrics to serialize
        
    Returns:
        JSON-compatible dictionary
    """
    result = {
        "correctness": _serialize_to_json(metrics.correctness),
        "hallucination": _serialize_to_json(metrics.hallucination)
    }
    
    if metrics.risk_direction is not None:
        result["risk_direction"] = _serialize_to_json(metrics.risk_direction)
    
    if metrics.risk_impact is not None:
        result["risk_impact"] = _serialize_to_json(metrics.risk_impact)
    
    return result


def log_evaluation_result(
    result: EvaluationResult,
    query_executor: Optional[QueryExecutor] = None
) -> Optional[str]:
    """
    Log a single evaluation result to Supabase Postgres database.
    
    This function stores evaluation results as JSONB in the evaluation_results table.
    If query_executor is None, logging is skipped (local-only mode).
    Logging failures are handled gracefully and do not fail the evaluation pipeline.
    
    Args:
        result: EvaluationResult to log
        query_executor: QueryExecutor instance for database operations (optional)
        
    Returns:
        result_id if logging successful, None otherwise (local-only mode or failure)
        
    Note:
        - Logging is optional - if query_executor is None, returns None immediately
        - Database failures are caught and logged, but do not raise exceptions
        - All evaluation metrics are stored as JSONB for flexibility
    """
    # Local-only mode: skip logging if query_executor is None
    if query_executor is None:
        logger.debug("query_executor is None, skipping database logging (local-only mode)")
        return None
    
    try:
        # Generate unique result_id
        result_id = str(uuid.uuid4())
        
        # Serialize all result components to JSON
        judge_output_json = json.dumps(_serialize_judge_output(result.judge_output))
        meta_eval_output_json = json.dumps(_serialize_meta_eval_output(result.meta_eval_output))
        beir_metrics_json = json.dumps(_serialize_beir_metrics(result.beir_metrics))
        
        # Insert into database
        insert_query = """
            INSERT INTO evaluation_results (
                id, example_id, timestamp,
                judge_output, meta_eval_output, beir_metrics
            )
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
        """
        
        params = (
            result_id,
            result.example_id,
            result.timestamp,
            judge_output_json,
            meta_eval_output_json,
            beir_metrics_json
        )
        
        inserted_id = query_executor.execute_insert(insert_query, params)
        
        if inserted_id:
            logger.info(f"Successfully logged evaluation result {result_id} for example {result.example_id}")
            return inserted_id
        else:
            logger.warning(f"Insert succeeded but no id returned for example {result.example_id}")
            return result_id  # Return generated ID even if RETURNING didn't work
            
    except Exception as e:
        # Graceful degradation: log error but don't fail evaluation pipeline
        logger.error(f"Failed to log evaluation result for example {result.example_id}: {e}", exc_info=True)
        return None


def log_evaluation_batch(
    results: List[EvaluationResult],
    query_executor: Optional[QueryExecutor] = None,
    judge_performance_metrics: Optional[JudgePerformanceMetrics] = None
) -> None:
    """
    Log a batch of evaluation results to Supabase Postgres database.
    
    This function performs batch insertion of evaluation results as JSONB.
    If query_executor is None, logging is skipped (local-only mode).
    Partial failures are handled gracefully - successful results are logged,
    failed results are logged as errors but don't fail the batch operation.
    
    Args:
        results: List of EvaluationResult objects to log
        query_executor: QueryExecutor instance for database operations (optional)
        judge_performance_metrics: Optional JudgePerformanceMetrics to log with batch
        
    Note:
        - Logging is optional - if query_executor is None, returns immediately
        - Batch failures are caught and logged, but do not raise exceptions
        - All evaluation metrics are stored as JSONB for flexibility
        - Ground truth fields from MetaEvaluationResult are included in logged data
    """
    # Local-only mode: skip logging if query_executor is None
    if query_executor is None:
        logger.debug(f"query_executor is None, skipping batch database logging (local-only mode) for {len(results)} results")
        return
    
    if not results:
        logger.warning("log_evaluation_batch called with empty results list")
        return
    
    successful_logs = 0
    failed_logs = 0
    
    try:
        # Prepare batch insert data
        batch_data = []
        for result in results:
            try:
                result_id = str(uuid.uuid4())
                
                # Serialize all result components to JSON
                judge_output_json = json.dumps(_serialize_judge_output(result.judge_output))
                meta_eval_output_json = json.dumps(_serialize_meta_eval_output(result.meta_eval_output))
                beir_metrics_json = json.dumps(_serialize_beir_metrics(result.beir_metrics))
                
                batch_data.append((
                    result_id,
                    result.example_id,
                    result.timestamp,
                    judge_output_json,
                    meta_eval_output_json,
                    beir_metrics_json
                ))
            except Exception as e:
                logger.error(f"Failed to serialize result for example {result.example_id}: {e}", exc_info=True)
                failed_logs += 1
        
        if not batch_data:
            logger.error("No valid results to log after serialization")
            return
        
        # Perform batch insert
        # Note: PostgreSQL doesn't support true batch inserts with psycopg2,
        # so we'll use execute_values or execute_batch for efficiency
        # For now, we'll use a simple loop with individual inserts for reliability
        # In production, consider using execute_values for better performance
        
        for data in batch_data:
            try:
                insert_query = """
                    INSERT INTO evaluation_results (
                        id, example_id, timestamp,
                        judge_output, meta_eval_output, beir_metrics
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                """
                
                query_executor.execute_insert(insert_query, data)
                successful_logs += 1
                
            except Exception as e:
                logger.error(f"Failed to insert result {data[0]} for example {data[1]}: {e}", exc_info=True)
                failed_logs += 1
        
        # Optionally log judge performance metrics if provided
        if judge_performance_metrics is not None:
            try:
                # Store judge performance metrics in metadata for the batch
                # Note: This is a simplified approach - in production, you might want
                # a separate table or different approach for storing batch-level metrics
                metrics_json = json.dumps(_serialize_judge_performance_metrics(judge_performance_metrics))
                
                # Update judge_performance_metrics column for all results in this batch
                # Use a tuple for the IN clause
                example_ids = tuple(result.example_id for result in results)
                if example_ids:
                    update_query = """
                        UPDATE evaluation_results
                        SET judge_performance_metrics = %s::jsonb
                        WHERE example_id = ANY(%s)
                    """
                    
                    query_executor.execute_query(
                        update_query,
                        (metrics_json, list(example_ids))
                    )
                    
                    logger.info(f"Updated {len(example_ids)} results with judge performance metrics")
                
            except Exception as e:
                logger.warning(f"Failed to log judge performance metrics: {e}", exc_info=True)
                # Don't fail batch operation if metrics logging fails
        
        logger.info(
            f"Batch logging completed: {successful_logs} successful, {failed_logs} failed "
            f"out of {len(results)} total results"
        )
        
    except Exception as e:
        # Graceful degradation: log error but don't fail evaluation pipeline
        logger.error(f"Failed to log evaluation batch: {e}", exc_info=True)

