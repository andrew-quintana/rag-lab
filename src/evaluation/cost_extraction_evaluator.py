"""Cost extraction LLM node for evaluation - Simplified for local use"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .base_evaluator import BaseEvaluatorNode

# Default prompt template for cost extraction
DEFAULT_COST_EXTRACTION_PROMPT = """You are an expert evaluator tasked with extracting structured cost information from insurance-related text.

Your task:
1. Parse the text for cost-related information
2. Extract time-based costs, money-based costs, and step-based costs
3. Format the information in a structured way

Text to analyze:
{text}

Extract the following types of cost information if present:
- **Money**: Dollar amounts for deductibles, copays, coinsurance, premiums, out-of-pocket costs
- **Time**: Time periods for waiting periods, processing times, appointment durations
- **Steps**: Number of steps in processes, forms to fill out, approvals needed

Provide your extraction as a JSON object with the following structure:
{{
    "money": "extracted monetary cost or null if not present",
    "time": "extracted time cost or null if not present", 
    "steps": "extracted step count or null if not present",
    "reasoning": "Your detailed reasoning explaining what cost information was found"
}}

Important:
- Only include fields where cost information is clearly present in the text
- Preserve original format/units when possible (e.g., "$50", "2 weeks", "3 steps")
- Set fields to null if that type of cost information is not mentioned
- Provide clear reasoning for what you found or didn't find
- Focus on specific, quantifiable cost information"""


class CostExtractionEvaluator(BaseEvaluatorNode):
    """Cost extraction evaluator using LLM to parse structured cost information from text"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None
    ):
        """
        Initialize cost extraction evaluator.
        
        Args:
            llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
            prompt_template: Optional custom prompt template
            prompt_path: Optional path to prompt template file
        """
        super().__init__(
            llm_call_function=llm_call_function,
            prompt_template=prompt_template or DEFAULT_COST_EXTRACTION_PROMPT,
            prompt_path=prompt_path,
            name="CostExtractionEvaluator"
        )
    
    def _construct_prompt(self, text: str) -> str:
        """
        Construct cost extraction prompt by loading template and replacing placeholders.
        
        Args:
            text: Text containing cost information to extract
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{text}"]
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
            prompt = template.replace("{text}", text)
        except Exception as e:
            error_msg = f"Failed to format prompt template: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        self.logger.debug(f"Constructed cost extraction prompt for text length: {len(text)}")
        return prompt
    
    def extract_costs(self, text: str) -> Dict[str, Any]:
        """
        Extract structured cost information (time, money, steps) from unstructured text.
        
        Parses natural language text to identify time-based costs, money-based costs,
        and step-based costs. All cost fields are optional - if a cost type is not present
        in the text, that field will be omitted from the result.
        
        Uses temperature=0.1 for reproducibility.
        
        Args:
            text: Text containing cost information to extract
            
        Returns:
            Dictionary with optional fields:
            - `time`: Optional string or number representing time cost
            - `money`: Optional string or number representing monetary cost
            - `steps`: Optional integer or string representing step count
            - `reasoning`: Required string explaining what cost information was found
            
        Raises:
            Exception: If LLM call fails
            ValueError: If input is invalid or empty, or if response cannot be parsed
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self.logger.info(f"Extracting costs from text (length: {len(text)})")
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(text)
            self.logger.debug(f"Constructed cost extraction prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for extraction
            # Temperature=0.1 for reproducibility
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            # Step 4: Validate required reasoning field
            if "reasoning" not in parsed:
                raise ValueError(
                    f"Cost extraction response missing 'reasoning' field. "
                    f"Response: {parsed}"
                )
            
            # Step 5: Build result dictionary with only present fields
            result = {
                "reasoning": parsed["reasoning"]
            }
            
            # Add optional fields only if they exist in the parsed response and are not null
            if "time" in parsed and parsed["time"] is not None:
                result["time"] = parsed["time"]
            
            if "money" in parsed and parsed["money"] is not None:
                result["money"] = parsed["money"]
            
            if "steps" in parsed and parsed["steps"] is not None:
                result["steps"] = parsed["steps"]
            
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            self.logger.info(
                f"Cost extraction completed. Found fields: {list(result.keys())} "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return result
            
        except Exception as e:
            if "LLM call failed" in str(e):
                raise
            self.logger.error(f"Unexpected error extracting costs: {e}", exc_info=True)
            raise Exception(f"Unexpected error extracting costs: {str(e)}") from e


# Convenience function
def extract_costs(
    text: str,
    llm_call_function: Callable[[str, float, int], str],
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured cost information (time, money, steps) from unstructured text.
    
    This is a convenience function that creates a CostExtractionEvaluator instance 
    and calls extract_costs().
    
    Args:
        text: Text containing cost information to extract
        llm_call_function: Function to call LLM
        prompt_template: Optional custom prompt template
        
    Returns:
        Dictionary with optional cost fields and reasoning
    """
    evaluator = CostExtractionEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_template
    )
    return evaluator.extract_costs(text)