"""Evaluation Pipeline Orchestrator

This module implements the complete evaluation pipeline that integrates all components
(retrieval, RAG generation, judge, meta-eval, BEIR metrics) to evaluate the RAG system.
"""

import logging
import time
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError, EvaluationError
from rag_eval.core.interfaces import (
    EvaluationExample,
    EvaluationResult,
    RetrievalResult,
    ModelAnswer,
    JudgeEvaluationResult,
    MetaEvaluationResult,
    BEIRMetricsResult
)
from rag_eval.services.evaluator.judge import evaluate_answer_with_judge
from rag_eval.services.evaluator.meta_eval import meta_evaluate_judge, calculate_judge_metrics
from rag_eval.services.evaluator.beir_metrics import compute_beir_metrics
from rag_eval.services.evaluator.cost_extraction import extract_costs

logger = logging.getLogger(__name__)


def evaluate_rag_system(
    evaluation_dataset: List[EvaluationExample],
    rag_retriever: Callable[[str, int], List[RetrievalResult]],
    rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer],
    config: Optional[Config] = None
) -> List[EvaluationResult]:
    """
    Run complete evaluation pipeline for a set of evaluation examples.
    
    This function orchestrates the complete evaluation pipeline:
    1. Retrieval: Retrieve chunks for each question
    2. RAG Generation: Generate answers using retrieved chunks
    3. Judge Evaluation: Evaluate answers with LLM-as-Judge
    4. Meta-Evaluation: Validate judge verdicts
    5. BEIR Metrics: Compute retrieval metrics
    
    Args:
        evaluation_dataset: List of evaluation examples with questions, reference answers, and ground-truth chunks
        rag_retriever: Function that retrieves chunks for a query (query: str, k: int) -> List[RetrievalResult]
        rag_generator: Function that generates answers (query: str, chunks: List[RetrievalResult]) -> ModelAnswer
        config: Application configuration (optional, defaults to Config.from_env())
        
    Returns:
        List of EvaluationResult objects containing:
        - example_id: Unique identifier for the evaluation example
        - judge_output: JudgeEvaluationResult with hallucination analysis
        - meta_eval_output: MetaEvaluationResult with judge validation
        - beir_metrics: BEIRMetricsResult with retrieval metrics
        - timestamp: Evaluation timestamp
        
    Raises:
        AzureServiceError: If Azure service calls fail
        ValueError: If evaluation dataset is invalid
        EvaluationError: If evaluation pipeline fails
    """
    # Validate evaluation dataset
    if not evaluation_dataset:
        raise ValueError("evaluation_dataset cannot be empty")
    
    # Use provided config or default
    if config is None:
        config = Config.from_env()
    
    logger.info(f"Starting evaluation pipeline for {len(evaluation_dataset)} examples")
    pipeline_start_time = time.time()
    
    results = []
    failed_examples = []
    
    # Process each example
    for i, example in enumerate(evaluation_dataset, start=1):
        logger.info(f"[{i}/{len(evaluation_dataset)}] Evaluating example: {example.example_id}")
        example_start_time = time.time()
        
        try:
            result = _evaluate_single_example(
                example=example,
                rag_retriever=rag_retriever,
                rag_generator=rag_generator,
                config=config
            )
            results.append(result)
            
            example_latency = time.time() - example_start_time
            logger.info(
                f"[{i}/{len(evaluation_dataset)}] Completed example {example.example_id} "
                f"in {example_latency:.2f}s"
            )
            
        except Exception as e:
            example_latency = time.time() - example_start_time
            logger.error(
                f"[{i}/{len(evaluation_dataset)}] Failed to evaluate example {example.example_id} "
                f"after {example_latency:.2f}s: {e}",
                exc_info=True
            )
            failed_examples.append((example.example_id, str(e)))
            # Continue with next example instead of failing entire pipeline
    
    pipeline_latency = time.time() - pipeline_start_time
    logger.info(
        f"Evaluation pipeline completed in {pipeline_latency:.2f}s. "
        f"Successfully evaluated {len(results)}/{len(evaluation_dataset)} examples"
    )
    
    if failed_examples:
        logger.warning(f"Failed examples: {failed_examples}")
    
    if not results:
        raise EvaluationError(
            f"All {len(evaluation_dataset)} examples failed evaluation. "
            f"Failed examples: {failed_examples}"
        )
    
    return results


def _evaluate_single_example(
    example: EvaluationExample,
    rag_retriever: Callable[[str, int], List[RetrievalResult]],
    rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer],
    config: Config
) -> EvaluationResult:
    """
    Evaluate a single evaluation example through the complete pipeline.
    
    This function handles the evaluation of a single example:
    1. Retrieve chunks using RAG retriever
    2. Generate answer using RAG generator
    3. Evaluate with LLM-as-Judge
    4. Meta-evaluate judge verdict
    5. Compute BEIR metrics
    6. Assemble EvaluationResult
    
    Args:
        example: Evaluation example to evaluate
        rag_retriever: Function that retrieves chunks for a query
        rag_generator: Function that generates answers
        config: Application configuration
        
    Returns:
        EvaluationResult object with all evaluation outputs
        
    Raises:
        AzureServiceError: If Azure service calls fail
        ValueError: If inputs are invalid
    """
    # Step 1: Retrieve chunks using RAG retriever
    logger.debug(f"Step 1: Retrieving chunks for example {example.example_id}")
    retrieval_start_time = time.time()
    try:
        retrieved_chunks = rag_retriever(example.question, k=5)
        retrieval_latency = time.time() - retrieval_start_time
        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_latency:.2f}s"
        )
    except Exception as e:
        logger.error(f"Retrieval failed for example {example.example_id}: {e}", exc_info=True)
        raise AzureServiceError(f"Retrieval failed: {str(e)}") from e
    
    # Step 2: Generate answer using RAG generator
    logger.debug(f"Step 2: Generating answer for example {example.example_id}")
    generation_start_time = time.time()
    try:
        model_answer = rag_generator(example.question, retrieved_chunks)
        generation_latency = time.time() - generation_start_time
        logger.debug(
            f"Generated answer ({len(model_answer.text)} characters) in {generation_latency:.2f}s"
        )
    except Exception as e:
        logger.error(f"Generation failed for example {example.example_id}: {e}", exc_info=True)
        raise AzureServiceError(f"Generation failed: {str(e)}") from e
    
    # Step 3: Evaluate with LLM-as-Judge
    logger.debug(f"Step 3: Evaluating with judge for example {example.example_id}")
    judge_start_time = time.time()
    try:
        judge_output = evaluate_answer_with_judge(
            query=example.question,
            retrieved_context=retrieved_chunks,
            model_answer=model_answer.text,
            reference_answer=example.reference_answer,
            config=config
        )
        judge_latency = time.time() - judge_start_time
        logger.debug(f"Judge evaluation completed in {judge_latency:.2f}s")
    except Exception as e:
        logger.error(f"Judge evaluation failed for example {example.example_id}: {e}", exc_info=True)
        raise AzureServiceError(f"Judge evaluation failed: {str(e)}") from e
    
    # Step 4: Meta-evaluate judge verdict
    logger.debug(f"Step 4: Meta-evaluating judge for example {example.example_id}")
    meta_eval_start_time = time.time()
    try:
        # Extract costs if needed (for meta-evaluation)
        extracted_costs = None
        actual_costs = None
        
        if judge_output.correctness_binary:
            # Extract costs from model answer
            try:
                extracted_costs = extract_costs(model_answer.text, config=config)
            except Exception as e:
                logger.warning(
                    f"Failed to extract costs from model answer for example {example.example_id}: {e}"
                )
            
            # Extract costs from retrieved chunks
            if retrieved_chunks:
                chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_chunks])
                try:
                    actual_costs = extract_costs(chunks_text, config=config)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract costs from retrieved chunks for example {example.example_id}: {e}"
                    )
        
        meta_eval_output = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=retrieved_chunks,
            model_answer=model_answer.text,
            reference_answer=example.reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        meta_eval_latency = time.time() - meta_eval_start_time
        logger.debug(f"Meta-evaluation completed in {meta_eval_latency:.2f}s")
    except Exception as e:
        logger.error(f"Meta-evaluation failed for example {example.example_id}: {e}", exc_info=True)
        raise EvaluationError(f"Meta-evaluation failed: {str(e)}") from e
    
    # Step 5: Compute BEIR metrics
    logger.debug(f"Step 5: Computing BEIR metrics for example {example.example_id}")
    beir_start_time = time.time()
    try:
        beir_metrics = compute_beir_metrics(
            retrieved_chunks=retrieved_chunks,
            ground_truth_chunk_ids=example.ground_truth_chunk_ids,
            k=5
        )
        beir_latency = time.time() - beir_start_time
        logger.debug(f"BEIR metrics computed in {beir_latency:.2f}s")
    except Exception as e:
        logger.error(f"BEIR metrics computation failed for example {example.example_id}: {e}", exc_info=True)
        raise EvaluationError(f"BEIR metrics computation failed: {str(e)}") from e
    
    # Step 6: Assemble EvaluationResult
    result = EvaluationResult(
        example_id=example.example_id,
        judge_output=judge_output,
        meta_eval_output=meta_eval_output,
        beir_metrics=beir_metrics,
        timestamp=datetime.now()
    )
    
    logger.debug(f"Completed evaluation for example {example.example_id}")
    return result

