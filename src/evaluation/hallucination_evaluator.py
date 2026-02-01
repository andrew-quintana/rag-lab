"""Hallucination classification LLM node for evaluation - Simplified for local use"""

from typing import Optional, Callable, List
from pathlib import Path

from ..core.interfaces import RetrievalResult
from .base_evaluator import BaseEvaluatorNode

# Default prompt template for hallucination evaluation
DEFAULT_HALLUCINATION_PROMPT = """You are an expert evaluator tasked with determining whether a model-generated answer contains hallucinations.

Your task:
1. Analyze the model answer for factual claims
2. Check whether these claims are supported by the provided retrieved context
3. Identify any information in the model answer that is not grounded in the retrieved context

Retrieved Context:
{retrieved_context}

Model Answer:
{model_answer}

Provide your evaluation as a JSON object with the following structure:
{{
    "hallucination_binary": true/false,
    "reasoning": "Your detailed reasoning explaining whether hallucination was detected"
}}

Important:
- Set hallucination_binary to true if the model answer contains any claims not supported by the retrieved context
- Set hallucination_binary to false if all claims in the model answer are grounded in the retrieved context
- Focus on factual accuracy and grounding, not stylistic differences
- Consider that paraphrasing and reasonable inference from context is acceptable
- Provide clear reasoning for your decision"""


class HallucinationEvaluator(BaseEvaluatorNode):
    """Hallucination classification evaluator using LLM"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None
    ):
        """
        Initialize hallucination evaluator.
        
        Args:
            llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
            prompt_template: Optional custom prompt template
            prompt_path: Optional path to prompt template file
        """
        super().__init__(
            llm_call_function=llm_call_function,
            prompt_template=prompt_template or DEFAULT_HALLUCINATION_PROMPT,
            prompt_path=prompt_path,
            name="HallucinationEvaluator"
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
        
        Uses temperature=0.1 for reproducibility.
        
        Args:
            retrieved_context: List of retrieved chunks with similarity scores
            model_answer: Generated answer from RAG system
            
        Returns:
            bool: True if hallucination detected, False otherwise
            
        Raises:
            Exception: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
        """
        # Validate inputs
        if not retrieved_context:
            raise ValueError("Retrieved context cannot be empty")
        
        if not model_answer or not model_answer.strip():
            raise ValueError("Model answer cannot be empty")
        
        self.logger.info(
            f"Classifying hallucination for model answer (length: {len(model_answer)}) "
            f"with {len(retrieved_context)} retrieved chunks"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(retrieved_context, model_answer)
            self.logger.debug(f"Constructed hallucination prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility
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
            
        except Exception as e:
            if "LLM call failed" in str(e):
                raise
            self.logger.error(f"Unexpected error classifying hallucination: {e}", exc_info=True)
            raise Exception(f"Unexpected error classifying hallucination: {str(e)}") from e


# Convenience function
def classify_hallucination(
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    llm_call_function: Callable[[str, float, int], str],
    prompt_template: Optional[str] = None
) -> bool:
    """
    Binary classification: detect if model answer contains hallucinations.
    
    This is a convenience function that creates a HallucinationEvaluator instance 
    and calls classify_hallucination().
    
    Args:
        retrieved_context: List of retrieved chunks with similarity scores
        model_answer: Generated answer from RAG system
        llm_call_function: Function to call LLM
        prompt_template: Optional custom prompt template
        
    Returns:
        bool: True if hallucination detected, False otherwise
    """
    evaluator = HallucinationEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_template
    )
    return evaluator.classify_hallucination(retrieved_context, model_answer)