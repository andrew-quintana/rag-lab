"""System-level risk direction classification LLM node for evaluation"""

from typing import Optional, List
from pathlib import Path

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.interfaces import RetrievalResult
from rag_eval.services.evaluator.base_evaluator import BaseEvaluatorNode
from rag_eval.services.shared.llm_providers import LLMProvider
from rag_eval.db.queries import QueryExecutor

class RiskDirectionEvaluator(BaseEvaluatorNode):
    """System-level risk direction classification evaluator using LLM"""
    
    def __init__(
        self,
        prompt_version: Optional[str] = None,
        live: bool = True,
        query_executor: Optional[QueryExecutor] = None,
        config: Optional[Config] = None,
        llm_provider: Optional[LLMProvider] = None,
        prompt_path: Optional[Path] = None  # Backward compatibility for testing
    ):
        """
        Initialize risk direction evaluator.
        
        Args:
            prompt_version: Prompt version name (e.g., "v1"). Defaults to None (uses live version)
            query_executor: QueryExecutor instance for database operations (required for database prompt loading)
            config: Application configuration (optional, defaults to Config.from_env())
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
            prompt_path: Path to prompt template file (optional, for testing only).
                        Either query_executor or prompt_path must be provided.
        """
        super().__init__(
            prompt_version=prompt_version,
            prompt_type="evaluation",
            name="risk_direction_evaluator",
            live=live,
            query_executor=query_executor,
            config=config,
            llm_provider=llm_provider,
            prompt_path=prompt_path
        )
    
    def _format_retrieved_context(self, retrieved_context: List[RetrievalResult]) -> str:
        """
        Format retrieved context for prompt insertion.
        
        Concatenates chunk texts with chunk IDs for traceability.
        
        Args:
            retrieved_context: List of retrieval results
            
        Returns:
            Formatted context string
        """
        if not retrieved_context:
            return "[No retrieved context available]"
        
        formatted_chunks = []
        for result in retrieved_context:
            chunk_text = result.chunk_text.strip()
            chunk_id = result.chunk_id
            formatted_chunks.append(f"[Chunk ID: {chunk_id}] {chunk_text}")
        
        return "\n".join(formatted_chunks)
    
    def _construct_prompt(
        self,
        model_answer: str,
        retrieved_context: List[RetrievalResult]
    ) -> str:
        """
        Construct risk direction classification prompt by loading template and replacing placeholders.
        
        Args:
            model_answer: Generated answer from RAG system
            retrieved_context: List of retrieved chunks with similarity scores
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{model_answer}", "{retrieved_context}"]
        missing_placeholders = [
            placeholder for placeholder in required_placeholders
            if placeholder not in template
        ]
        
        if missing_placeholders:
            error_msg = (
                f"Prompt template is missing required placeholders: "
                f"{', '.join(missing_placeholders)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Format retrieved context
        formatted_context = self._format_retrieved_context(retrieved_context)
        
        # Replace placeholders
        try:
            prompt = template.replace("{retrieved_context}", formatted_context)
            prompt = prompt.replace("{model_answer}", model_answer)
        except Exception as e:
            error_msg = f"Failed to format prompt template: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        self.logger.debug(
            f"Constructed risk direction prompt for {len(retrieved_context)} chunks, "
            f"model answer length: {len(model_answer)}"
        )
        return prompt
    
    def classify_risk_direction(
        self,
        model_answer: str,
        retrieved_context: List[RetrievalResult]
    ) -> int:
        """
        Classify the risk direction of system-level deviations.
        
        Determines whether a deviation from the gold answer represents a care avoidance risk (-1)
        or unexpected cost risk (+1) by comparing the model answer's cost claims
        against the retrieved context. This evaluates the entire RAG pipeline as a black box,
        capturing deviations regardless of origin (retrieval, augmentation, context ordering,
        prompting, model reasoning, or hallucination).
        
        **Risk Classification**:
        - **-1 (Care Avoidance Risk)**: Model overestimated cost, dissuading user from seeking care
        - **+1 (Unexpected Cost Risk)**: Model underestimated cost, persuading user to pursue care
        
        **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
        in the evaluation system requirements. Note that LLM classification is inherently
        non-deterministic, so slight variations may occur across calls.
        
        **Critical Note**: This function does NOT use a reference answer. It only
        analyzes cost information in the model answer against cost information in
        the retrieved context. This is intentional - risk classification is about
        the direction of cost errors relative to ground truth evidence, not relative
        to a reference answer.
        
        Args:
            model_answer: Generated answer from RAG system
            retrieved_context: List of retrieved chunks with similarity scores
            
        Returns:
            int: -1 for care avoidance risk (overestimated), +1 for unexpected cost risk (underestimated)
            
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
            
        Example:
            >>> from rag_eval.core.config import Config
            >>> from rag_eval.core.interfaces import RetrievalResult
            >>> 
            >>> config = Config.from_env()
            >>> evaluator = RiskDirectionEvaluator(config)
            >>> 
            >>> retrieved_context = [
            ...     RetrievalResult(
            ...         chunk_id="chunk_001",
            ...         similarity_score=0.95,
            ...         chunk_text="The copay for a specialist visit is $50."
            ...     )
            ... ]
            >>> model_answer = "The copay for specialist visits is $75."
            >>> 
            >>> risk_classification = evaluator.classify_risk_direction(
            ...     model_answer, retrieved_context
            ... )
            >>> print(risk_classification)
            -1
        """
        # Validate inputs
        if not retrieved_context:
            raise ValueError("Retrieved context cannot be empty")
        
        if not model_answer or not model_answer.strip():
            raise ValueError("Model answer cannot be empty")
        
        # Get model name for logging
        model_name = "unknown"
        if hasattr(self.llm_provider, 'model'):
            model_name = self.llm_provider.model
        
        self.logger.info(
            f"Classifying risk direction for model answer (length: {len(model_answer)}) "
            f"with {len(retrieved_context)} retrieved chunks using model '{model_name}'"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(model_answer, retrieved_context)
            self.logger.debug(f"Constructed risk direction prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility (as specified in requirements)
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            classification = self._parse_json_response(response_text)
            
            # Step 4: Validate risk_direction field
            if "risk_direction" not in classification:
                raise ValueError(
                    f"Risk direction classification missing 'risk_direction' field. "
                    f"Response: {classification}"
                )
            
            # Validate risk_direction is -1 or +1
            risk_direction = classification["risk_direction"]
            if not isinstance(risk_direction, int):
                raise ValueError(
                    f"risk_direction must be an integer, got {type(risk_direction)}: {risk_direction}"
                )
            
            if risk_direction not in [-1, 1]:
                raise ValueError(
                    f"risk_direction must be -1 (care avoidance risk) or +1 (unexpected cost risk), "
                    f"got {risk_direction}"
                )
            
            reasoning = classification.get("reasoning", "No reasoning provided")
            
            risk_type = "care avoidance risk" if risk_direction == -1 else "unexpected cost risk"
            self.logger.info(
                f"Risk direction classification: {risk_direction} ({risk_type}) "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return risk_direction
            
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error classifying risk direction: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error classifying risk direction: {str(e)}"
            ) from e


# Backward compatibility: module-level function
def classify_risk_direction(
    model_answer: str,
    retrieved_context: List[RetrievalResult],
    config: Optional[Config] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_version: Optional[str] = None,
    live: bool = True
) -> int:
    """
    Classify the risk direction of system-level deviations.
    
    This is a convenience function that maintains backward compatibility.
    It creates a RiskDirectionEvaluator instance and calls classify_risk_direction().
    
    Determines whether a deviation from the gold answer represents a care avoidance risk (-1)
    or unexpected cost risk (+1) by comparing the model answer's cost claims
    against the retrieved context. This evaluates the entire RAG pipeline as a black box,
    capturing deviations regardless of origin (retrieval, augmentation, context ordering,
    prompting, model reasoning, or hallucination).
    
    **Risk Classification**:
    - **-1 (Care Avoidance Risk)**: Model overestimated cost, dissuading user from seeking care
    - **+1 (Unexpected Cost Risk)**: Model underestimated cost, persuading user to pursue care
    
    **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
    in the evaluation system requirements. Note that LLM classification is inherently
    non-deterministic, so slight variations may occur across calls.
    
    **Critical Note**: This function does NOT use a reference answer. It only
    analyzes cost information in the model answer against cost information in
    the retrieved context. This is intentional - risk classification is about
    the direction of cost errors relative to ground truth evidence, not relative
    to a reference answer.
    
    Args:
        model_answer: Generated answer from RAG system
        retrieved_context: List of retrieved chunks with similarity scores
        config: Application configuration (optional, defaults to Config.from_env())
        query_executor: QueryExecutor instance for database operations (optional, for database prompt loading)
        
    Returns:
        int: -1 for care avoidance risk (overestimated), +1 for unexpected cost risk (underestimated)
        
    Raises:
        AzureServiceError: If Azure Foundry LLM call fails
        ValueError: If inputs are invalid or empty, or if response cannot be parsed
        
    Example:
        >>> from rag_eval.core.config import Config
        >>> from rag_eval.core.interfaces import RetrievalResult
        >>> 
        >>> config = Config.from_env()
        >>> 
        >>> retrieved_context = [
        ...     RetrievalResult(
        ...         chunk_id="chunk_001",
        ...         similarity_score=0.95,
        ...         chunk_text="The copay for a specialist visit is $50."
        ...     )
        ... ]
        >>> model_answer = "The copay for specialist visits is $75."
        >>> 
        >>> risk_classification = classify_risk_direction(
        ...     model_answer, retrieved_context, config
        ... )
        >>> print(risk_classification)
        -1
    """
    evaluator = RiskDirectionEvaluator(
        config=config,
        query_executor=query_executor,
        prompt_version=prompt_version if prompt_version else None,
        live=live
    )
    return evaluator.classify_risk_direction(model_answer, retrieved_context)

