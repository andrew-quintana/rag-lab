"""Hallucination classification LLM node for evaluation"""

from typing import Optional, List
from pathlib import Path

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.interfaces import RetrievalResult
from rag_eval.services.evaluator.base_evaluator import BaseEvaluatorNode
from rag_eval.services.shared.llm_providers import LLMProvider

# Default prompt template path
_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "evaluation" / "hallucination_prompt.md"


class HallucinationEvaluator(BaseEvaluatorNode):
    """Hallucination classification evaluator using LLM"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        prompt_path: Optional[Path] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize hallucination evaluator.
        
        Args:
            config: Application configuration (optional, defaults to Config.from_env())
            prompt_path: Path to prompt template file (optional, uses default if None)
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
        """
        if prompt_path is None:
            prompt_path = _DEFAULT_PROMPT_PATH
        
        super().__init__(prompt_path=prompt_path, config=config, llm_provider=llm_provider)
    
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
        retrieved_context: List[RetrievalResult],
        model_answer: str
    ) -> str:
        """
        Construct hallucination classification prompt by loading template and replacing placeholders.
        
        Args:
            retrieved_context: List of retrieved chunks with similarity scores
            model_answer: Generated answer from RAG system
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{retrieved_context}", "{model_answer}"]
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
            f"Constructed hallucination prompt for {len(retrieved_context)} chunks, "
            f"model answer length: {len(model_answer)}"
        )
        return prompt
    
    def classify_hallucination(
        self,
        retrieved_context: List[RetrievalResult],
        model_answer: str
    ) -> bool:
        """
        Binary classification: detect if model answer contains hallucinations.
        
        Performs strict grounding analysis: checks if model answer contains
        information not supported by retrieved evidence. Reference answer is NOT used.
        
        **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
        in the evaluation system requirements. Note that LLM classification is inherently
        non-deterministic, so slight variations may occur across calls.
        
        **Critical Note**: This function does NOT use a reference answer. It only
        analyzes whether the model answer is grounded in the retrieved context.
        This is intentional - hallucination detection is about grounding, not correctness.
        
        Args:
            retrieved_context: List of retrieved chunks with similarity scores
            model_answer: Generated answer from RAG system
            
        Returns:
            bool: True if hallucination detected, False otherwise
            
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
            
        Example:
            >>> from rag_eval.core.config import Config
            >>> from rag_eval.core.interfaces import RetrievalResult
            >>> 
            >>> config = Config.from_env()
            >>> evaluator = HallucinationEvaluator(config)
            >>> 
            >>> retrieved_context = [
            ...     RetrievalResult(
            ...         chunk_id="chunk_001",
            ...         similarity_score=0.95,
            ...         chunk_text="The copay for a specialist visit is $50."
            ...     )
            ... ]
            >>> model_answer = "The copay for specialist visits is $50."
            >>> 
            >>> has_hallucination = evaluator.classify_hallucination(
            ...     retrieved_context, model_answer
            ... )
            >>> print(has_hallucination)
            False
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
            f"Classifying hallucination for model answer (length: {len(model_answer)}) "
            f"with {len(retrieved_context)} retrieved chunks using model '{model_name}'"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(retrieved_context, model_answer)
            self.logger.debug(f"Constructed hallucination prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility (as specified in requirements)
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            classification = self._parse_json_response(response_text)
            
            # Step 4: Validate hallucination_binary field
            if "hallucination_binary" not in classification:
                raise ValueError(
                    f"Hallucination classification missing 'hallucination_binary' field. "
                    f"Response: {classification}"
                )
            
            # Validate hallucination_binary is a boolean
            hallucination_binary = classification["hallucination_binary"]
            if not isinstance(hallucination_binary, bool):
                raise ValueError(
                    f"hallucination_binary must be a boolean, got {type(hallucination_binary)}: {hallucination_binary}"
                )
            
            reasoning = classification.get("reasoning", "No reasoning provided")
            
            self.logger.info(
                f"Hallucination classification: {hallucination_binary} "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return hallucination_binary
            
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error classifying hallucination: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error classifying hallucination: {str(e)}"
            ) from e


# Backward compatibility: module-level function
def classify_hallucination(
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    config: Optional[Config] = None
) -> bool:
    """
    Binary classification: detect if model answer contains hallucinations.
    
    This is a convenience function that maintains backward compatibility.
    It creates a HallucinationEvaluator instance and calls classify_hallucination().
    
    Performs strict grounding analysis: checks if model answer contains
    information not supported by retrieved evidence. Reference answer is NOT used.
    
    **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
    in the evaluation system requirements. Note that LLM classification is inherently
    non-deterministic, so slight variations may occur across calls.
    
    **Critical Note**: This function does NOT use a reference answer. It only
    analyzes whether the model answer is grounded in the retrieved context.
    This is intentional - hallucination detection is about grounding, not correctness.
    
    Args:
        retrieved_context: List of retrieved chunks with similarity scores
        model_answer: Generated answer from RAG system
        config: Application configuration (optional, defaults to Config.from_env())
        
    Returns:
        bool: True if hallucination detected, False otherwise
        
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
        >>> model_answer = "The copay for specialist visits is $50."
        >>> 
        >>> has_hallucination = classify_hallucination(
        ...     retrieved_context, model_answer, config
        ... )
        >>> print(has_hallucination)
        False
    """
    evaluator = HallucinationEvaluator(config=config)
    return evaluator.classify_hallucination(retrieved_context, model_answer)

