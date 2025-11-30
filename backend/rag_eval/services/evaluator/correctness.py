"""Correctness classification LLM node for evaluation"""

from typing import Optional
from pathlib import Path

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.services.evaluator.base_evaluator import BaseEvaluatorNode
from rag_eval.services.shared.llm_providers import LLMProvider

# Default prompt template path
_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "evaluation" / "correctness_prompt.md"


class CorrectnessEvaluator(BaseEvaluatorNode):
    """Correctness classification evaluator using LLM"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        prompt_path: Optional[Path] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize correctness evaluator.
        
        Args:
            config: Application configuration (optional, defaults to Config.from_env())
            prompt_path: Path to prompt template file (optional, uses default if None)
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
        """
        if prompt_path is None:
            prompt_path = _DEFAULT_PROMPT_PATH
        
        super().__init__(prompt_path=prompt_path, config=config, llm_provider=llm_provider)
    
    def _construct_prompt(
        self,
        query: str,
        model_answer: str,
        reference_answer: str
    ) -> str:
        """
        Construct correctness classification prompt by loading template and replacing placeholders.
        
        Args:
            query: Original user query
            model_answer: Generated answer from RAG system
            reference_answer: Gold reference answer for comparison
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{query}", "{model_answer}", "{reference_answer}"]
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
        
        # Replace placeholders
        try:
            prompt = template.replace("{query}", query)
            prompt = prompt.replace("{model_answer}", model_answer)
            prompt = prompt.replace("{reference_answer}", reference_answer)
        except Exception as e:
            error_msg = f"Failed to format prompt template: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        self.logger.debug(f"Constructed correctness prompt for query '{query[:50]}...'")
        return prompt
    
    def classify_correctness(
        self,
        query: str,
        model_answer: str,
        reference_answer: str
    ) -> bool:
        """
        Direct comparison: assess if model answer is correct compared to reference answer.
        
        Performs direct comparison between model answer and gold reference answer.
        Does not consider retrieved context - purely compares answer correctness.
        
        **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
        in the evaluation system requirements. Note that LLM classification is inherently
        non-deterministic, so slight variations may occur across calls.
        
        Args:
            query: Original user query
            model_answer: Generated answer from RAG system
            reference_answer: Gold reference answer for comparison
            
        Returns:
            bool: True if answer is correct, False otherwise
            
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
            
        Example:
            >>> from rag_eval.core.config import Config
            >>> 
            >>> config = Config.from_env()
            >>> evaluator = CorrectnessEvaluator(config)
            >>> query = "What is the copay for a specialist visit?"
            >>> model_answer = "The copay for a specialist visit is $50."
            >>> reference_answer = "Specialist visits have a $50 copay."
            >>> is_correct = evaluator.classify_correctness(query, model_answer, reference_answer)
            >>> print(is_correct)
            True
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")
        
        if not model_answer or not model_answer.strip():
            raise ValueError("Model answer cannot be empty")
        
        if not reference_answer or not reference_answer.strip():
            raise ValueError("Reference answer cannot be empty")
        
        # Get model name for logging
        model_name = "unknown"
        if hasattr(self.llm_provider, 'model'):
            model_name = self.llm_provider.model
        
        self.logger.info(
            f"Classifying correctness for query '{query[:50]}...' "
            f"using model '{model_name}'"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(query, model_answer, reference_answer)
            self.logger.debug(f"Constructed correctness prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility (as specified in requirements)
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            classification = self._parse_json_response(response_text)
            
            # Step 4: Validate correctness_binary field
            if "correctness_binary" not in classification:
                raise ValueError(
                    f"Correctness classification missing 'correctness_binary' field. "
                    f"Response: {classification}"
                )
            
            # Validate correctness_binary is a boolean
            correctness_binary = classification["correctness_binary"]
            if not isinstance(correctness_binary, bool):
                raise ValueError(
                    f"correctness_binary must be a boolean, got {type(correctness_binary)}: {correctness_binary}"
                )
            
            reasoning = classification.get("reasoning", "No reasoning provided")
            
            self.logger.info(
                f"Correctness classification: {correctness_binary} "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return correctness_binary
            
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error classifying correctness: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error classifying correctness: {str(e)}"
            ) from e


# Backward compatibility: module-level function
def classify_correctness(
    query: str,
    model_answer: str,
    reference_answer: str,
    config: Optional[Config] = None
) -> bool:
    """
    Direct comparison: assess if model answer is correct compared to reference answer.
    
    This is a convenience function that maintains backward compatibility.
    It creates a CorrectnessEvaluator instance and calls classify_correctness().
    
    Performs direct comparison between model answer and gold reference answer.
    Does not consider retrieved context - purely compares answer correctness.
    
    **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
    in the evaluation system requirements. Note that LLM classification is inherently
    non-deterministic, so slight variations may occur across calls.
    
    Args:
        query: Original user query
        model_answer: Generated answer from RAG system
        reference_answer: Gold reference answer for comparison
        config: Application configuration (optional, defaults to Config.from_env())
        
    Returns:
        bool: True if answer is correct, False otherwise
        
    Raises:
        AzureServiceError: If Azure Foundry LLM call fails
        ValueError: If inputs are invalid or empty, or if response cannot be parsed
        
    Example:
        >>> from rag_eval.core.config import Config
        >>> 
        >>> config = Config.from_env()
        >>> query = "What is the copay for a specialist visit?"
        >>> model_answer = "The copay for a specialist visit is $50."
        >>> reference_answer = "Specialist visits have a $50 copay."
        >>> is_correct = classify_correctness(query, model_answer, reference_answer, config)
        >>> print(is_correct)
        True
    """
    evaluator = CorrectnessEvaluator(config=config)
    return evaluator.classify_correctness(query, model_answer, reference_answer)
