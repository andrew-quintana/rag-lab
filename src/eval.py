"""Main evaluation orchestration and pipeline"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
import traceback

from .core.interfaces import (
    EvaluationExample, EvaluationResult, JudgeEvaluationResult,
    MetaEvaluationResult, BEIRMetricsResult
)
from .evaluation.beir_metrics import compute_beir_metrics
from .core.io import RunManager, create_new_run
from .indexing.index import RAGRetriever
from .evaluation.judge import JudgeEvaluator, load_prompts_from_disk
from .core.registry import registry

logger = logging.getLogger(__name__)


class RAGEvaluationPipeline:
    """Complete RAG evaluation pipeline with BEIR + Judge + Meta-eval"""
    
    def __init__(
        self,
        retriever=None,
        generator_function: Optional[Callable[[str, List[str]], str]] = None,
        llm_function: Optional[Callable[[str, float, int], str]] = None,
        run_name: Optional[str] = None,
        prompts_dir: Optional[str] = None,
        # New component-based parameters
        judge_name: Optional[str] = None,
        chunker_name: Optional[str] = None,
        embedder_name: Optional[str] = None,
        retriever_name: Optional[str] = None,
        generator_name: Optional[str] = None,
        component_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize evaluation pipeline with component registry support

        Args:
            retriever: RAG retrieval system (legacy)
            generator_function: Function that takes (query, context_chunks) -> answer (legacy)
            llm_function: Function that takes (prompt, temperature, max_tokens) -> response (legacy)
            run_name: Optional name for evaluation run
            prompts_dir: Optional path to prompts directory (e.g. "prompts/evaluation") for judge templates
            judge_name: Name of judge implementation from registry
            chunker_name: Name of chunker implementation from registry
            embedder_name: Name of embedder implementation from registry  
            retriever_name: Name of retriever implementation from registry
            generator_name: Name of generator implementation from registry
            component_configs: Configuration dict for each component type
        """
        # Initialize component configurations
        self.component_configs = component_configs or {}
        
        # Initialize components from registry or use legacy parameters
        self.retriever = self._initialize_retriever(
            retriever, retriever_name, embedder_name
        )
        self.generator = self._initialize_generator(
            generator_function, generator_name, llm_function
        )
        self.judge = self._initialize_judge(
            llm_function, judge_name, prompts_dir
        )
        
        # Legacy support
        self.generator_function = generator_function
        self.llm_function = llm_function

        from .evaluation.meta_evaluator import MetaEvaluator
        self.meta_evaluator = MetaEvaluator()

        self.run_manager = RunManager()
        self.run_dir = create_new_run(run_name)

        logger.info(f"Initialized evaluation pipeline with run directory: {self.run_dir}")
    
    def _initialize_retriever(self, retriever, retriever_name, embedder_name):
        """Initialize retriever from registry or use provided instance"""
        if retriever is not None:
            return retriever
        
        if retriever_name:
            retriever_cls = registry.get('retrievers', retriever_name)
            config = self.component_configs.get('retriever', {})
            return retriever_cls(**config)
        
        return None
    
    def _initialize_generator(self, generator_function, generator_name, llm_function):
        """Initialize generator from registry or use provided function"""
        if generator_function is not None:
            return None  # Use legacy function
        
        if generator_name and llm_function:
            generator_cls = registry.get('generators', generator_name)
            config = self.component_configs.get('generator', {})
            return generator_cls(llm_function=llm_function, **config)
        
        return None
    
    def _initialize_judge(self, llm_function, judge_name, prompts_dir):
        """Initialize judge from registry or use legacy approach"""
        prompt_templates = {}
        if prompts_dir:
            prompt_templates = load_prompts_from_disk(prompts_dir)
            if prompt_templates:
                logger.info(f"Loaded {len(prompt_templates)} prompt(s) from {prompts_dir}")
        
        if judge_name and llm_function:
            judge_cls = registry.get('judges', judge_name)
            config = self.component_configs.get('judge', {})
            return judge_cls(llm_function=llm_function, prompts=prompt_templates, **config)
        
        # Legacy approach
        return JudgeEvaluator(llm_function, prompt_templates=prompt_templates)
    
    def _format_generation_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Default prompt formatting for generation"""
        if context_chunks:
            context = "\n\n".join(context_chunks)
            return f"""Context information:
{context}

Question: {question}

Answer the question based on the context provided above.

Answer:"""
        else:
            return f"Question: {question}\n\nAnswer:"
    
    def evaluate_single_example(
        self,
        example: EvaluationExample,
        retrieval_k: int = 5
    ) -> EvaluationResult:
        """
        Evaluate a single example through the complete pipeline
        
        Args:
            example: Evaluation example with question, reference answer, and ground truth
            retrieval_k: Number of chunks to retrieve for BEIR metrics
            
        Returns:
            Complete evaluation result
        """
        try:
            # 1. Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(example.question, k=retrieval_k)
            
            # 2. Generate answer
            context_texts = [chunk.chunk_text for chunk in retrieved_chunks]
            if self.generator is not None:
                # Use component-based generator
                from .core.interfaces import GenerationResult
                if hasattr(self.generator, 'generate_from_question'):
                    result = self.generator.generate_from_question(example.question, context_texts)
                else:
                    formatted_prompt = self._format_generation_prompt(example.question, context_texts)
                    result = self.generator.generate(formatted_prompt, context_chunks=context_texts)
                generated_answer = result.response
            else:
                # Use legacy generator function
                generated_answer = self.generator_function(example.question, context_texts)
            
            # 3. Compute BEIR metrics
            beir_metrics = compute_beir_metrics(
                retrieved_chunks, 
                example.ground_truth_chunk_ids, 
                k=retrieval_k
            )
            
            # 4. Run LLM-as-Judge evaluation
            judge_result = self.judge.evaluate(
                question=example.question,
                reference_answer=example.reference_answer,
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
            )
            
            # 5. Run meta-evaluation (pass retrieved_chunks for hallucination/risk validation)
            meta_result = self.meta_evaluator.validate_judge_output(
                judge_output=judge_result,
                question=example.question,
                reference_answer=example.reference_answer,
                generated_answer=generated_answer,
                retrieved_context=retrieved_chunks,
            )
            
            # 6. Create complete result
            result = EvaluationResult(
                example_id=example.example_id,
                judge_output=judge_result,
                meta_eval_output=meta_result,
                beir_metrics=beir_metrics,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Completed evaluation for example {example.example_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating example {example.example_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Return error result
            error_judge_result = JudgeEvaluationResult(
                correctness_binary=False,
                hallucination_binary=True,
                risk_direction=None,
                risk_impact=None,
                reasoning=f"Evaluation failed with error: {str(e)}",
                failure_mode="evaluation_error"
            )
            
            error_meta_result = MetaEvaluationResult(
                judge_correct=False,
                explanation="Meta-evaluation skipped due to evaluation error"
            )
            
            error_beir_result = BEIRMetricsResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                ndcg_at_k=0.0
            )
            
            return EvaluationResult(
                example_id=example.example_id,
                judge_output=error_judge_result,
                meta_eval_output=error_meta_result,
                beir_metrics=error_beir_result,
                timestamp=datetime.now()
            )
    
    def evaluate_dataset(
        self,
        examples: List[EvaluationExample],
        retrieval_k: int = 5,
        save_intermediate: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate complete dataset
        
        Args:
            examples: List of evaluation examples
            retrieval_k: Number of chunks to retrieve for BEIR metrics
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Starting evaluation of {len(examples)} examples")
        
        # Save configuration
        config = {
            "num_examples": len(examples),
            "retrieval_k": retrieval_k,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "complete_pipeline"
        }
        self.run_manager.save_config(config, self.run_dir)
        
        results = []
        traces = []
        
        for i, example in enumerate(examples):
            logger.info(f"Evaluating example {i+1}/{len(examples)}: {example.example_id}")
            
            # Record trace
            trace_start = datetime.now()
            
            # Evaluate example
            result = self.evaluate_single_example(example, retrieval_k)
            results.append(result)
            
            # Record trace
            trace_end = datetime.now()
            trace = {
                "example_id": example.example_id,
                "start_time": trace_start.isoformat(),
                "end_time": trace_end.isoformat(),
                "duration_seconds": (trace_end - trace_start).total_seconds(),
                "success": result.judge_output.failure_mode != "evaluation_error"
            }
            traces.append(trace)
            
            # Save intermediate results
            if save_intermediate and (i + 1) % 10 == 0:
                self._save_partial_results(results[:i+1], traces[:i+1], i+1)
        
        # Save final results
        self._save_final_results(results, traces)
        
        logger.info(f"Completed evaluation of {len(examples)} examples")
        return results
    
    def _save_partial_results(self, results: List[EvaluationResult], traces: List[Dict], count: int):
        """Save partial results during evaluation"""
        logger.info(f"Saving intermediate results ({count} examples)")
        
        # Convert results to dictionaries
        outputs = [self._result_to_dict(result) for result in results]
        
        self.run_manager.save_outputs(outputs, self.run_dir)
        self.run_manager.save_traces(traces, self.run_dir)
    
    def _save_final_results(self, results: List[EvaluationResult], traces: List[Dict]):
        """Save final results and compute summary metrics"""
        logger.info(f"Saving final results ({len(results)} examples)")
        
        # Convert results to dictionaries
        outputs = [self._result_to_dict(result) for result in results]
        
        # Save outputs and traces
        self.run_manager.save_outputs(outputs, self.run_dir)
        self.run_manager.save_traces(traces, self.run_dir)
        
        # Compute and save summary metrics
        metrics = self._compute_summary_metrics(results)
        self.run_manager.save_metrics(metrics, self.run_dir)
        
        logger.info(f"Evaluation complete. Results saved to {self.run_dir}")
    
    def _result_to_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary for serialization"""
        return {
            "example_id": result.example_id,
            "timestamp": result.timestamp.isoformat(),
            "judge_output": {
                "correctness_binary": result.judge_output.correctness_binary,
                "hallucination_binary": result.judge_output.hallucination_binary,
                "risk_direction": result.judge_output.risk_direction,
                "risk_impact": result.judge_output.risk_impact,
                "reasoning": result.judge_output.reasoning,
                "failure_mode": result.judge_output.failure_mode
            },
            "meta_eval_output": {
                "judge_correct": result.meta_eval_output.judge_correct,
                "explanation": result.meta_eval_output.explanation,
                "ground_truth_correctness": result.meta_eval_output.ground_truth_correctness,
                "ground_truth_hallucination": result.meta_eval_output.ground_truth_hallucination,
                "ground_truth_risk_direction": result.meta_eval_output.ground_truth_risk_direction,
                "ground_truth_risk_impact": result.meta_eval_output.ground_truth_risk_impact
            },
            "beir_metrics": {
                "recall_at_k": result.beir_metrics.recall_at_k,
                "precision_at_k": result.beir_metrics.precision_at_k,
                "ndcg_at_k": result.beir_metrics.ndcg_at_k
            }
        }
    
    def _compute_summary_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute summary metrics across all results"""
        if not results:
            return {"error": "No results to compute metrics"}
        
        # Filter out error results
        valid_results = [r for r in results if r.judge_output.failure_mode != "evaluation_error"]
        
        if not valid_results:
            return {"error": "No valid results to compute metrics"}
        
        # BEIR metrics
        beir_recall = [r.beir_metrics.recall_at_k for r in valid_results]
        beir_precision = [r.beir_metrics.precision_at_k for r in valid_results]
        beir_ndcg = [r.beir_metrics.ndcg_at_k for r in valid_results]
        
        # Judge metrics
        correctness_values = [r.judge_output.correctness_binary for r in valid_results]
        hallucination_values = [r.judge_output.hallucination_binary for r in valid_results]
        
        # Risk metrics (filter None values)
        risk_direction_values = [r.judge_output.risk_direction for r in valid_results 
                               if r.judge_output.risk_direction is not None]
        risk_impact_values = [r.judge_output.risk_impact for r in valid_results 
                            if r.judge_output.risk_impact is not None]
        
        # Meta-eval metrics
        judge_correct_values = [r.meta_eval_output.judge_correct for r in valid_results]
        
        metrics = {
            "total_examples": len(results),
            "valid_examples": len(valid_results),
            "error_rate": (len(results) - len(valid_results)) / len(results),
            
            # BEIR metrics
            "beir_metrics": {
                "recall_at_k": {
                    "mean": sum(beir_recall) / len(beir_recall),
                    "min": min(beir_recall),
                    "max": max(beir_recall)
                },
                "precision_at_k": {
                    "mean": sum(beir_precision) / len(beir_precision),
                    "min": min(beir_precision),
                    "max": max(beir_precision)
                },
                "ndcg_at_k": {
                    "mean": sum(beir_ndcg) / len(beir_ndcg),
                    "min": min(beir_ndcg),
                    "max": max(beir_ndcg)
                }
            },
            
            # Judge metrics
            "judge_metrics": {
                "correctness_rate": sum(correctness_values) / len(correctness_values),
                "hallucination_rate": sum(hallucination_values) / len(hallucination_values),
                "risk_direction_coverage": len(risk_direction_values) / len(valid_results),
                "risk_impact_coverage": len(risk_impact_values) / len(valid_results)
            },
            
            # Meta-evaluation metrics
            "meta_eval_metrics": {
                "judge_accuracy": sum(judge_correct_values) / len(judge_correct_values)
            },
            
            # Risk analysis (if available)
            "risk_analysis": {}
        }
        
        # Add risk analysis if we have risk data
        if risk_direction_values:
            metrics["risk_analysis"]["risk_direction"] = {
                "mean": sum(risk_direction_values) / len(risk_direction_values),
                "care_avoidance_rate": sum(1 for x in risk_direction_values if x == -1) / len(risk_direction_values),
                "unexpected_cost_rate": sum(1 for x in risk_direction_values if x == 1) / len(risk_direction_values),
                "neutral_rate": sum(1 for x in risk_direction_values if x == 0) / len(risk_direction_values)
            }
        
        if risk_impact_values:
            metrics["risk_analysis"]["risk_impact"] = {
                "mean": sum(risk_impact_values) / len(risk_impact_values),
                "distribution": {
                    "low_0": sum(1 for x in risk_impact_values if x == 0) / len(risk_impact_values),
                    "moderate_1": sum(1 for x in risk_impact_values if x == 1) / len(risk_impact_values),
                    "high_2": sum(1 for x in risk_impact_values if x == 2) / len(risk_impact_values),
                    "critical_3": sum(1 for x in risk_impact_values if x == 3) / len(risk_impact_values)
                }
            }
        
        return metrics


# Convenience functions
def run_evaluation(
    examples: List[EvaluationExample],
    retriever=None,
    generator_function: Optional[Callable[[str, List[str]], str]] = None,
    llm_function: Optional[Callable[[str, float, int], str]] = None,
    run_name: Optional[str] = None,
    retrieval_k: int = 5,
    prompts_dir: Optional[str] = None,
    # Component registry parameters
    judge_name: Optional[str] = None,
    chunker_name: Optional[str] = None,
    embedder_name: Optional[str] = None,
    retriever_name: Optional[str] = None,
    generator_name: Optional[str] = None,
    component_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[EvaluationResult], str]:
    """
    Run complete evaluation pipeline.

    Args:
        examples: List of evaluation examples
        retriever: RAG retrieval system
        generator_function: Function that takes (query, context_chunks) -> answer
        llm_function: Function that takes (prompt, temperature, max_tokens) -> response
        run_name: Optional name for evaluation run
        retrieval_k: Number of chunks to retrieve
        prompts_dir: Optional path to prompts directory (e.g. "prompts/evaluation")

    Returns:
        Tuple of (evaluation_results, run_directory_path)
    """
    pipeline = RAGEvaluationPipeline(
        retriever, generator_function, llm_function, run_name, prompts_dir=prompts_dir
    )
    results = pipeline.evaluate_dataset(examples, retrieval_k)
    return results, pipeline.run_dir


def load_evaluation_results(run_dir: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load evaluation results from run directory
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (outputs, metrics)
    """
    run_manager = RunManager()
    outputs = run_manager.load_outputs(run_dir)
    metrics = run_manager.load_metrics(run_dir)
    return outputs, metrics


def print_evaluation_summary(run_dir: str):
    """
    Print summary of evaluation results
    
    Args:
        run_dir: Path to run directory
    """
    try:
        _, metrics = load_evaluation_results(run_dir)
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Run Directory: {run_dir}")
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Valid Examples: {metrics['valid_examples']}")
        print(f"Error Rate: {metrics['error_rate']:.2%}")
        
        print(f"\n--- BEIR Metrics ---")
        beir = metrics['beir_metrics']
        print(f"Recall@K: {beir['recall_at_k']['mean']:.3f}")
        print(f"Precision@K: {beir['precision_at_k']['mean']:.3f}")
        print(f"nDCG@K: {beir['ndcg_at_k']['mean']:.3f}")
        
        print(f"\n--- Judge Metrics ---")
        judge = metrics['judge_metrics']
        print(f"Correctness Rate: {judge['correctness_rate']:.2%}")
        print(f"Hallucination Rate: {judge['hallucination_rate']:.2%}")
        
        print(f"\n--- Meta-Evaluation ---")
        meta = metrics['meta_eval_metrics']
        print(f"Judge Accuracy: {meta['judge_accuracy']:.2%}")
        
        if 'risk_analysis' in metrics and metrics['risk_analysis']:
            print(f"\n--- Risk Analysis ---")
            risk = metrics['risk_analysis']
            if 'risk_direction' in risk:
                rd = risk['risk_direction']
                print(f"Care Avoidance Rate: {rd['care_avoidance_rate']:.2%}")
                print(f"Unexpected Cost Rate: {rd['unexpected_cost_rate']:.2%}")
            if 'risk_impact' in risk:
                ri = risk['risk_impact']
                print(f"Mean Risk Impact: {ri['mean']:.2f}")
        
    except Exception as e:
        print(f"Error loading evaluation summary: {e}")