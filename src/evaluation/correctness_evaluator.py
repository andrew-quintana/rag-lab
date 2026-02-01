"""Correctness classification LLM node for evaluation - Simplified for local use"""

from typing import Optional, Callable
from pathlib import Path

from .base_evaluator import BaseEvaluatorNode

# Default prompt template for correctness evaluation
DEFAULT_CORRECTNESS_PROMPT = """You are an expert evaluator tasked with determining whether a model-generated answer is correct compared to a reference answer.

Your task:
1. Compare the model answer to the reference answer
2. Determine if the model answer is factually correct and addresses the query appropriately
3. Focus on semantic equivalence, not exact wording

Query: {query}

Model Answer: {model_answer}

Reference Answer: {reference_answer}

Provide your evaluation as a JSON object with the following structure:
{{
    "correctness_binary": true/false,
    "reasoning": "Your detailed reasoning explaining why the answer is correct or incorrect"
}}

Important:
- Set correctness_binary to true if the model answer is factually correct and addresses the query appropriately
- Set correctness_binary to false if the model answer contains factual errors, is incomplete, or doesn't address the query
- Focus on semantic equivalence rather than exact wording
- Provide clear reasoning for your decision"""


class CorrectnessEvaluator(BaseEvaluatorNode):
    """Correctness classification evaluator using LLM"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None
    ):
        """
        Initialize correctness evaluator.
        
        Args:
            llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
            prompt_template: Optional custom prompt template
            prompt_path: Optional path to prompt template file
        """
        super().__init__(
            llm_call_function=llm_call_function,
            prompt_template=prompt_template or DEFAULT_CORRECTNESS_PROMPT,
            prompt_path=prompt_path,
            name="CorrectnessEvaluator"
        )
    
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
        
        Uses temperature=0.1 for reproducibility.
        
        Args:
            query: Original user query
            model_answer: Generated answer from RAG system
            reference_answer: Gold reference answer for comparison
            
        Returns:
            bool: True if answer is correct, False otherwise
            
        Raises:
            Exception: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")
        
        if not model_answer or not model_answer.strip():
            raise ValueError("Model answer cannot be empty")
        
        if not reference_answer or not reference_answer.strip():
            raise ValueError("Reference answer cannot be empty")
        
        self.logger.info(f"Classifying correctness for query '{query[:50]}...'")
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(query, model_answer, reference_answer)
            self.logger.debug(f"Constructed correctness prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility
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
            
        except Exception as e:
            if "LLM call failed" in str(e):
                raise
            self.logger.error(f"Unexpected error classifying correctness: {e}", exc_info=True)
            raise Exception(f"Unexpected error classifying correctness: {str(e)}") from e


# Convenience function
def classify_correctness(
    query: str,
    model_answer: str,
    reference_answer: str,
    llm_call_function: Callable[[str, float, int], str],
    prompt_template: Optional[str] = None
) -> bool:
    """
    Direct comparison: assess if model answer is correct compared to reference answer.
    
    This is a convenience function that creates a CorrectnessEvaluator instance 
    and calls classify_correctness().
    
    Args:
        query: Original user query
        model_answer: Generated answer from RAG system
        reference_answer: Gold reference answer for comparison
        llm_call_function: Function to call LLM
        prompt_template: Optional custom prompt template
        
    Returns:
        bool: True if answer is correct, False otherwise
    """
    evaluator = CorrectnessEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_template
    )
    return evaluator.classify_correctness(query, model_answer, reference_answer)