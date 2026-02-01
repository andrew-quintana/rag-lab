"""System-level risk impact magnitude calculation LLM node for evaluation - Simplified for local use"""

import json
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .base_evaluator import BaseEvaluatorNode

# Default prompt template for risk impact evaluation
DEFAULT_RISK_IMPACT_PROMPT = """You are an expert evaluator tasked with determining the impact magnitude of insurance cost deviations.

Your task:
1. Analyze the difference between model answer costs and actual costs from retrieved chunks
2. Consider mixed resource types (time, money, steps) and their relative importance
3. Assess the real-world impact magnitude on a discrete scale

Model Answer Cost:
{model_answer_cost}

Actual Cost (from retrieved chunks):
{actual_cost}

Impact Scale (discrete values):
- **0**: Minimal/no impact - very small differences that would not meaningfully affect decisions
- **1**: Low impact - small differences that might be noticed but unlikely to change behavior
- **2**: Moderate impact - significant differences that could influence healthcare decisions
- **3**: High/severe impact - major differences that would strongly influence care-seeking behavior

Provide your evaluation as a JSON object with the following structure:
{{
    "risk_impact": 0/1/2/3,
    "reasoning": "Your detailed reasoning explaining the impact magnitude assessment"
}}

Important:
- Consider the magnitude of cost differences across time, money, and steps
- Focus on real-world impact on healthcare decision-making
- Money differences generally have higher impact than time or steps
- Assess relative differences (percentages) rather than absolute amounts when possible
- Provide clear reasoning for your impact assessment"""


class RiskImpactEvaluator(BaseEvaluatorNode):
    """System-level risk impact magnitude calculation evaluator for insurance contexts"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None
    ):
        """
        Initialize risk impact evaluator.
        
        Args:
            llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
            prompt_template: Optional custom prompt template
            prompt_path: Optional path to prompt template file
        """
        super().__init__(
            llm_call_function=llm_call_function,
            prompt_template=prompt_template or DEFAULT_RISK_IMPACT_PROMPT,
            prompt_path=prompt_path,
            name="RiskImpactEvaluator"
        )
    
    def _format_cost_dict(self, cost_dict: Dict[str, Any]) -> str:
        """
        Format cost dictionary as JSON string for prompt insertion.
        
        Args:
            cost_dict: Cost dictionary with time, money, steps fields
            
        Returns:
            Formatted JSON string
        """
        try:
            return json.dumps(cost_dict, indent=2)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to format cost dict as JSON: {e}, using str()")
            return str(cost_dict)
    
    def _construct_prompt(
        self,
        model_answer_cost: Dict[str, Any],
        actual_cost: Dict[str, Any]
    ) -> str:
        """
        Construct risk impact calculation prompt by loading template and replacing placeholders.
        
        Args:
            model_answer_cost: Cost extracted from model answer (time/money/steps)
            actual_cost: Actual cost from retrieved chunks (ground truth)
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{model_answer_cost}", "{actual_cost}"]
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
        
        # Format cost dictionaries as JSON
        formatted_model_cost = self._format_cost_dict(model_answer_cost)
        formatted_actual_cost = self._format_cost_dict(actual_cost)
        
        # Replace placeholders
        try:
            prompt = template.replace("{model_answer_cost}", formatted_model_cost)
            prompt = prompt.replace("{actual_cost}", formatted_actual_cost)
        except Exception as e:
            error_msg = f"Failed to format prompt template: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        self.logger.debug(
            f"Constructed risk impact prompt with model cost: {formatted_model_cost[:100]}..., "
            f"actual cost: {formatted_actual_cost[:100]}..."
        )
        return prompt
    
    def calculate_risk_impact(
        self,
        model_answer_cost: Dict[str, Any],
        actual_cost: Dict[str, Any]
    ) -> int:
        """
        Calculate system-level risk impact magnitude (discrete scale: 0, 1, 2, or 3).
        
        Analyzes difference between model answer cost and actual cost from chunks,
        considering mixed resource types (time, money, steps) and their relative importance.
        Evaluates impact of deviations regardless of origin.
        
        Impact Scale (discrete values):
        - **0**: Minimal/no impact
        - **1**: Low impact  
        - **2**: Moderate impact
        - **3**: High/severe impact
        
        Uses temperature=0.1 for reproducibility.
        
        Args:
            model_answer_cost: Cost extracted from model answer (time/money/steps)
            actual_cost: Actual cost from retrieved chunks (ground truth)
            
        Returns:
            int: Impact magnitude as discrete value in {0, 1, 2, 3}
                
        Raises:
            Exception: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
        """
        # Validate inputs
        if not isinstance(model_answer_cost, dict) or not model_answer_cost:
            raise ValueError("model_answer_cost must be a non-empty dictionary")
        
        if not isinstance(actual_cost, dict) or not actual_cost:
            raise ValueError("actual_cost must be a non-empty dictionary")
        
        self.logger.info(
            f"Calculating risk impact for model cost: {model_answer_cost}, "
            f"actual cost: {actual_cost}"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(model_answer_cost, actual_cost)
            self.logger.debug(f"Constructed risk impact prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for impact calculation
            # Temperature=0.1 for reproducibility
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            classification = self._parse_json_response(response_text)
            
            # Step 4: Validate risk_impact field
            if "risk_impact" not in classification:
                raise ValueError(
                    f"Risk impact calculation missing 'risk_impact' field. "
                    f"Response: {classification}"
                )
            
            # Validate risk_impact is a number
            risk_impact = classification["risk_impact"]
            if not isinstance(risk_impact, (int, float)):
                raise ValueError(
                    f"risk_impact must be a number, got {type(risk_impact)}: {risk_impact}"
                )
            
            # Convert to int (discrete values only)
            risk_impact = int(risk_impact)
            
            # Validate risk_impact is a discrete value in {0, 1, 2, 3}
            if risk_impact not in [0, 1, 2, 3]:
                raise ValueError(
                    f"risk_impact must be a discrete value in {{0, 1, 2, 3}}, got {risk_impact}"
                )
            
            reasoning = classification.get("reasoning", "No reasoning provided")
            
            self.logger.info(
                f"Risk impact calculation: {risk_impact} "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return risk_impact
            
        except Exception as e:
            if "LLM call failed" in str(e):
                raise
            self.logger.error(f"Unexpected error calculating risk impact: {e}", exc_info=True)
            raise Exception(f"Unexpected error calculating risk impact: {str(e)}") from e


# Convenience function
def calculate_risk_impact(
    model_answer_cost: Dict[str, Any],
    actual_cost: Dict[str, Any],
    llm_call_function: Callable[[str, float, int], str],
    prompt_template: Optional[str] = None
) -> int:
    """
    Calculate system-level risk impact magnitude (discrete scale: 0, 1, 2, or 3).
    
    This is a convenience function that creates a RiskImpactEvaluator instance 
    and calls calculate_risk_impact().
    
    Args:
        model_answer_cost: Cost extracted from model answer (time/money/steps)
        actual_cost: Actual cost from retrieved chunks (ground truth)
        llm_call_function: Function to call LLM
        prompt_template: Optional custom prompt template
        
    Returns:
        int: Impact magnitude as discrete value in {0, 1, 2, 3}
    """
    evaluator = RiskImpactEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_template
    )
    return evaluator.calculate_risk_impact(model_answer_cost, actual_cost)