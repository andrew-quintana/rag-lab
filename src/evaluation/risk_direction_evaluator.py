"""System-level risk direction classification LLM node for evaluation - Simplified for local use"""

from typing import Optional, Callable, List
from pathlib import Path

from ..core.interfaces import RetrievalResult
from .base_evaluator import BaseEvaluatorNode

# Default prompt template for risk direction evaluation
DEFAULT_RISK_DIRECTION_PROMPT = """You are an expert evaluator tasked with determining the risk direction of insurance-related cost deviations.

Your task:
1. Analyze the model answer for cost-related claims (deductibles, copays, premiums, coverage limits, etc.)
2. Compare these claims against the cost information in the retrieved context
3. Determine the risk direction based on whether the model overestimated or underestimated costs

Model Answer:
{model_answer}

Retrieved Context:
{retrieved_context}

Risk Direction Categories:
- **-1 (Care Avoidance Risk)**: Model overestimated cost, potentially dissuading users from seeking care
- **0 (No Clear Risk Direction)**: Unable to determine clear risk direction or costs are similar
- **+1 (Unexpected Cost Risk)**: Model underestimated cost, potentially leading to unexpected financial burden

Provide your evaluation as a JSON object with the following structure:
{{
    "risk_direction": -1/0/1,
    "reasoning": "Your detailed reasoning explaining the risk direction classification"
}}

Important:
- Focus on insurance cost elements: deductibles, copays, coinsurance, premiums, out-of-pocket maximums
- Consider the direction of cost deviation and its potential impact on care-seeking behavior
- Set risk_direction to -1 if model overestimated costs (care avoidance risk)
- Set risk_direction to +1 if model underestimated costs (unexpected cost risk) 
- Set risk_direction to 0 if costs are similar or direction cannot be determined
- Provide clear reasoning for your classification"""


class RiskDirectionEvaluator(BaseEvaluatorNode):
    """System-level risk direction classification evaluator for insurance contexts"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None
    ):
        """
        Initialize risk direction evaluator.
        
        Args:
            llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
            prompt_template: Optional custom prompt template
            prompt_path: Optional path to prompt template file
        """
        super().__init__(
            llm_call_function=llm_call_function,
            prompt_template=prompt_template or DEFAULT_RISK_DIRECTION_PROMPT,
            prompt_path=prompt_path,
            name="RiskDirectionEvaluator"
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
        Classify the risk direction of insurance system-level deviations.
        
        Determines whether a deviation from the ground truth represents a care avoidance risk (-1)
        or unexpected cost risk (+1) by comparing the model answer's cost claims
        against the retrieved context.
        
        Risk Classification:
        - **-1 (Care Avoidance Risk)**: Model overestimated cost, dissuading user from seeking care
        - **0 (No Clear Risk Direction)**: Unable to determine clear risk direction
        - **+1 (Unexpected Cost Risk)**: Model underestimated cost, leading to unexpected costs
        
        Uses temperature=0.1 for reproducibility.
        
        Args:
            model_answer: Generated answer from RAG system
            retrieved_context: List of retrieved chunks with similarity scores
            
        Returns:
            int: -1 for care avoidance risk (overestimated), 0 for no clear direction, +1 for unexpected cost risk (underestimated)
            
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
            f"Classifying risk direction for model answer (length: {len(model_answer)}) "
            f"with {len(retrieved_context)} retrieved chunks"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(model_answer, retrieved_context)
            self.logger.debug(f"Constructed risk direction prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for classification
            # Temperature=0.1 for reproducibility
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
            
            # Validate risk_direction is -1, 0, or +1
            risk_direction = classification["risk_direction"]
            if not isinstance(risk_direction, int):
                raise ValueError(
                    f"risk_direction must be an integer, got {type(risk_direction)}: {risk_direction}"
                )
            
            if risk_direction not in [-1, 0, 1]:
                raise ValueError(
                    f"risk_direction must be -1 (care avoidance risk), 0 (no clear direction), or +1 (unexpected cost risk), "
                    f"got {risk_direction}"
                )
            
            reasoning = classification.get("reasoning", "No reasoning provided")
            
            if risk_direction == -1:
                risk_type = "care avoidance risk"
            elif risk_direction == 0:
                risk_type = "no clear direction"
            else:
                risk_type = "unexpected cost risk"
                
            self.logger.info(
                f"Risk direction classification: {risk_direction} ({risk_type}) "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return risk_direction
            
        except Exception as e:
            if "LLM call failed" in str(e):
                raise
            self.logger.error(f"Unexpected error classifying risk direction: {e}", exc_info=True)
            raise Exception(f"Unexpected error classifying risk direction: {str(e)}") from e


# Convenience function
def classify_risk_direction(
    model_answer: str,
    retrieved_context: List[RetrievalResult],
    llm_call_function: Callable[[str, float, int], str],
    prompt_template: Optional[str] = None
) -> int:
    """
    Classify the risk direction of insurance system-level deviations.
    
    This is a convenience function that creates a RiskDirectionEvaluator instance 
    and calls classify_risk_direction().
    
    Args:
        model_answer: Generated answer from RAG system
        retrieved_context: List of retrieved chunks with similarity scores
        llm_call_function: Function to call LLM
        prompt_template: Optional custom prompt template
        
    Returns:
        int: -1 for care avoidance risk, 0 for no clear direction, +1 for unexpected cost risk
    """
    evaluator = RiskDirectionEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_template
    )
    return evaluator.classify_risk_direction(model_answer, retrieved_context)