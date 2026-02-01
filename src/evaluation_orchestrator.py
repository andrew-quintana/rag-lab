"""Evaluation Pipeline Orchestrator - Simplified for local use

This module implements the complete evaluation pipeline that integrates all components
(retrieval, RAG generation, judge, meta-eval, BEIR metrics) to evaluate the RAG system.
"""

import logging
import time
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime

from .core.interfaces import (
    EvaluationExample,
    EvaluationResult,
    RetrievalResult,
    ModelAnswer,
    JudgeEvaluationResult,
    MetaEvaluationResult,
    BEIRMetricsResult
)
from .evaluation.judge import evaluate_answer_with_judge
from .evaluation.meta_evaluator import (
    meta_evaluate_judge, 
    calculate_judge_metrics,
    compute_bias_corrected_accuracy_from_results
)
from .cost_extraction_evaluator import extract_costs

logger = logging.getLogger(__name__)


def evaluate_rag_system(
    evaluation_dataset: List[EvaluationExample],
    rag_retriever: Callable[[str, int], List[RetrievalResult]],
    rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer],
    llm_call_function: Callable[[str, float, int], str],
    compute_beir_metrics_func: Optional[Callable[[List[RetrievalResult], List[str], int], BEIRMetricsResult]] = None,
    prompt_templates: Optional[Dict[str, str]] = None
) -> List[EvaluationResult]:
    """
    Run complete evaluation pipeline for a set of evaluation examples.
    
    This function orchestrates the complete evaluation pipeline:
    1. Retrieval: Retrieve chunks for each question
    2. RAG Generation: Generate answers using retrieved chunks
    3. Judge Evaluation: Evaluate answers with LLM-as-Judge
    4. Meta-Evaluation: Validate judge verdicts and compute bias-corrected accuracy
    5. BEIR Metrics: Compute retrieval metrics (optional)
    
    Args:
        evaluation_dataset: List of evaluation examples with questions, reference answers, and ground-truth chunks
        rag_retriever: Function that retrieves chunks for a query (query: str, k: int) -> List[RetrievalResult]
        rag_generator: Function that generates answers (query: str, chunks: List[RetrievalResult]) -> ModelAnswer
        llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
        compute_beir_metrics_func: Optional function to compute BEIR metrics
        prompt_templates: Optional dict of custom prompt templates
        
    Returns:
        List of EvaluationResult objects containing:
        - example_id: Unique identifier for the evaluation example
        - judge_output: JudgeEvaluationResult with evaluation analysis
        - meta_eval_output: MetaEvaluationResult with judge validation
        - beir_metrics: BEIRMetricsResult with retrieval metrics (if compute_beir_metrics_func provided)
        - timestamp: Evaluation timestamp
        
    Raises:
        Exception: If LLM service calls fail
        ValueError: If evaluation dataset is invalid
    """
    # Validate evaluation dataset
    if not evaluation_dataset:
        raise ValueError("evaluation_dataset cannot be empty")
    
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
                llm_call_function=llm_call_function,
                compute_beir_metrics_func=compute_beir_metrics_func,
                prompt_templates=prompt_templates
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
        raise Exception(
            f"All {len(evaluation_dataset)} examples failed evaluation. "
            f"Failed examples: {failed_examples}"
        )
    
    # Compute bias-corrected accuracy as part of meta-evaluation
    try:
        evaluation_pairs = [
            (result.judge_output, result.meta_eval_output) 
            for result in results
        ]
        bias_corrected = compute_bias_corrected_accuracy_from_results(evaluation_pairs)
        logger.info(
            f"Bias-corrected accuracy: {bias_corrected['corrected_accuracy']:.3f} "
            f"(raw: {bias_corrected['raw_accuracy']:.3f}, "
            f"95% CI: [{bias_corrected['confidence_interval'][0]:.3f}, "
            f"{bias_corrected['confidence_interval'][1]:.3f}])"
        )
        logger.debug(
            f"Judge calibration: q0={bias_corrected['q0']:.3f}, "
            f"q1={bias_corrected['q1']:.3f}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to compute bias-corrected accuracy: {e}. "
            f"This is non-fatal and evaluation results are still valid.",
            exc_info=True
        )
    
    return results


def _evaluate_single_example(
    example: EvaluationExample,
    rag_retriever: Callable[[str, int], List[RetrievalResult]],
    rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer],
    llm_call_function: Callable[[str, float, int], str],
    compute_beir_metrics_func: Optional[Callable[[List[RetrievalResult], List[str], int], BEIRMetricsResult]] = None,
    prompt_templates: Optional[Dict[str, str]] = None
) -> EvaluationResult:
    """
    Evaluate a single evaluation example through the complete pipeline.
    
    This function handles the evaluation of a single example:
    1. Retrieve chunks using RAG retriever
    2. Generate answer using RAG generator
    3. Evaluate with LLM-as-Judge
    4. Meta-evaluate judge verdict
    5. Compute BEIR metrics (optional)
    6. Assemble EvaluationResult
    
    Args:
        example: Evaluation example to evaluate
        rag_retriever: Function that retrieves chunks for a query
        rag_generator: Function that generates answers
        llm_call_function: Function to call LLM
        compute_beir_metrics_func: Optional function to compute BEIR metrics
        prompt_templates: Optional dict of custom prompt templates
        
    Returns:
        EvaluationResult object with all evaluation outputs
        
    Raises:
        Exception: If service calls fail
        ValueError: If inputs are invalid
    """
    # Step 1: Retrieve chunks using RAG retriever
    logger.debug(f"Step 1: Retrieving chunks for example {example.example_id}")
    retrieval_start_time = time.time()
    try:
        retrieved_chunks = rag_retriever(example.question, 5)  # k=5 
        retrieval_latency = time.time() - retrieval_start_time
        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_latency:.2f}s"
        )
    except Exception as e:
        logger.error(f"Retrieval failed for example {example.example_id}: {e}", exc_info=True)
        raise Exception(f"Retrieval failed: {str(e)}") from e
    
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
        raise Exception(f"Generation failed: {str(e)}") from e
    
    # Step 3: Evaluate with LLM-as-Judge
    logger.debug(f"Step 3: Evaluating with judge for example {example.example_id}")
    judge_start_time = time.time()
    try:
        judge_output = evaluate_answer_with_judge(
            query=example.question,
            retrieved_context=retrieved_chunks,
            model_answer=model_answer.text,
            reference_answer=example.reference_answer,
            llm_call_function=llm_call_function,
            prompt_templates=prompt_templates
        )
        judge_latency = time.time() - judge_start_time
        logger.debug(f"Judge evaluation completed in {judge_latency:.2f}s")
    except Exception as e:
        logger.error(f"Judge evaluation failed for example {example.example_id}: {e}", exc_info=True)
        raise Exception(f"Judge evaluation failed: {str(e)}") from e
    
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
                extracted_costs = extract_costs(
                    model_answer.text, 
                    llm_call_function=llm_call_function
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract costs from model answer for example {example.example_id}: {e}"
                )
            
            # Extract costs from retrieved chunks
            if retrieved_chunks:
                chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_chunks])
                try:
                    actual_costs = extract_costs(
                        chunks_text,
                        llm_call_function=llm_call_function
                    )
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
        raise Exception(f"Meta-evaluation failed: {str(e)}") from e
    
    # Step 5: Compute BEIR metrics (optional)
    beir_metrics = None
    if compute_beir_metrics_func:
        logger.debug(f"Step 5: Computing BEIR metrics for example {example.example_id}")
        beir_start_time = time.time()
        try:
            beir_metrics = compute_beir_metrics_func(
                retrieved_chunks,
                example.ground_truth_chunk_ids,
                5  # k=5
            )
            beir_latency = time.time() - beir_start_time
            logger.debug(f"BEIR metrics computed in {beir_latency:.2f}s")
        except Exception as e:
            logger.error(f"BEIR metrics computation failed for example {example.example_id}: {e}", exc_info=True)
            raise Exception(f"BEIR metrics computation failed: {str(e)}") from e
    
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


def print_evaluation_summary(results: List[EvaluationResult]) -> None:
    """
    Print a summary of evaluation results.
    
    Args:
        results: List of evaluation results
    """
    if not results:
        print("No evaluation results to summarize.")
        return
    
    print(f"\n=== Evaluation Summary ({len(results)} examples) ===")
    
    # Calculate aggregate metrics
    correctness_count = sum(1 for r in results if r.judge_output.correctness_binary)
    hallucination_count = sum(1 for r in results if r.judge_output.hallucination_binary)
    
    # Risk direction distribution
    risk_directions = [r.judge_output.risk_direction for r in results if r.judge_output.risk_direction is not None]
    care_avoidance_count = sum(1 for rd in risk_directions if rd == -1)
    no_clear_direction_count = sum(1 for rd in risk_directions if rd == 0)
    unexpected_cost_count = sum(1 for rd in risk_directions if rd == 1)
    
    # Risk impact distribution
    risk_impacts = [r.judge_output.risk_impact for r in results if r.judge_output.risk_impact is not None]
    impact_counts = {i: sum(1 for ri in risk_impacts if ri == i) for i in range(4)}
    
    print(f"Correctness: {correctness_count}/{len(results)} ({correctness_count/len(results)*100:.1f}%)")
    print(f"Hallucination: {hallucination_count}/{len(results)} ({hallucination_count/len(results)*100:.1f}%)")
    
    if risk_directions:
        print(f"\nRisk Direction Distribution ({len(risk_directions)} examples):")
        print(f"  Care Avoidance (-1): {care_avoidance_count}")
        print(f"  No Clear Direction (0): {no_clear_direction_count}")
        print(f"  Unexpected Cost (+1): {unexpected_cost_count}")
    
    if risk_impacts:
        print(f"\nRisk Impact Distribution ({len(risk_impacts)} examples):")
        for impact_level, count in impact_counts.items():
            print(f"  Level {impact_level}: {count}")
    
    # Meta-evaluation validation
    judge_correct_count = sum(1 for r in results if r.meta_eval_output.judge_correct)
    print(f"\nJudge Validation: {judge_correct_count}/{len(results)} ({judge_correct_count/len(results)*100:.1f}%)")
    
    # BEIR metrics (if available)
    if results[0].beir_metrics:
        avg_recall = sum(r.beir_metrics.recall_at_k for r in results) / len(results)
        avg_precision = sum(r.beir_metrics.precision_at_k for r in results) / len(results)
        avg_ndcg = sum(r.beir_metrics.ndcg_at_k for r in results) / len(results)
        
        print(f"\nBEIR Metrics (average):")
        print(f"  Recall@5: {avg_recall:.3f}")
        print(f"  Precision@5: {avg_precision:.3f}")
        print(f"  NDCG@5: {avg_ndcg:.3f}")
    
    # Compute bias-corrected accuracy
    try:
        evaluation_pairs = [(r.judge_output, r.meta_eval_output) for r in results]
        bias_corrected = compute_bias_corrected_accuracy_from_results(evaluation_pairs)
        
        print(f"\nBias-Corrected Accuracy:")
        print(f"  Raw Accuracy: {bias_corrected['raw_accuracy']:.3f}")
        print(f"  Corrected Accuracy: {bias_corrected['corrected_accuracy']:.3f}")
        print(f"  95% CI: [{bias_corrected['confidence_interval'][0]:.3f}, {bias_corrected['confidence_interval'][1]:.3f}]")
        print(f"  Judge Specificity (q0): {bias_corrected['q0']:.3f}")
        print(f"  Judge Sensitivity (q1): {bias_corrected['q1']:.3f}")
    except Exception as e:
        print(f"\nCould not compute bias-corrected accuracy: {e}")
    
    print("=" * 50)