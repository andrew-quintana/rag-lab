"""System-level risk impact magnitude calculation LLM node for evaluation"""

import json
from typing import Optional, Dict, Any
from pathlib import Path

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.services.evaluator.base_evaluator import BaseEvaluatorNode
from rag_eval.services.shared.llm_providers import LLMProvider
from rag_eval.db.queries import QueryExecutor


class RiskImpactEvaluator(BaseEvaluatorNode):
    """System-level risk impact magnitude calculation evaluator using LLM"""
    
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
        Initialize risk impact evaluator.
        
        Args:
            prompt_version: Prompt version name (e.g., "0.1"). Defaults to None (uses live version)
            query_executor: QueryExecutor instance for database operations (required for database prompt loading)
            config: Application configuration (optional, defaults to Config.from_env())
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
            prompt_path: Path to prompt template file (optional, for testing only).
                        Either query_executor or prompt_path must be provided.
        """
        super().__init__(
            prompt_version=prompt_version,
            prompt_type="evaluation",
            name="risk_impact_evaluator",
            live=live,
            query_executor=query_executor,
            config=config,
            llm_provider=llm_provider,
            prompt_path=prompt_path
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
        Evaluates impact of deviations regardless of origin (retrieval, augmentation, context ordering,
        prompting, model reasoning, or hallucination).
        
        **Impact Scale** (discrete values):
        - **0**: Minimal/no impact
        - **1**: Low impact
        - **2**: Moderate impact
        - **3**: High/severe impact
        
        **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
        in the evaluation system requirements. Note that LLM classification is inherently
        non-deterministic, so slight variations may occur across calls.
        
        **Rationale for LLM Node**: Uses an LLM node (not a deterministic function) because
        it must handle mixed resource types (time, money, steps) and assess their relative
        importance, which requires nuanced reasoning about real-world consequences.
        
        Args:
            model_answer_cost: Cost extracted from model answer (time/money/steps)
            actual_cost: Actual cost from retrieved chunks (ground truth)
            
        Returns:
            int: Impact magnitude as discrete value in {0, 1, 2, 3}, where:
                - 0: Minimal/no impact
                - 1: Low impact
                - 2: Moderate impact
                - 3: High/severe impact
                
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If inputs are invalid or empty, or if response cannot be parsed
            
        Example:
            >>> from rag_eval.core.config import Config
            >>> 
            >>> config = Config.from_env()
            >>> evaluator = RiskImpactEvaluator(config)
            >>> 
            >>> model_answer_cost = {"money": 500.0, "time": "2 hours"}
            >>> actual_cost = {"money": 50.0, "time": "30 minutes"}
            >>> 
            >>> impact = evaluator.calculate_risk_impact(model_answer_cost, actual_cost)
            >>> print(impact)
            2
        """
        # Validate inputs
        if not isinstance(model_answer_cost, dict) or not model_answer_cost:
            raise ValueError("model_answer_cost must be a non-empty dictionary")
        
        if not isinstance(actual_cost, dict) or not actual_cost:
            raise ValueError("actual_cost must be a non-empty dictionary")
        
        # Get model name for logging
        model_name = "unknown"
        if hasattr(self.llm_provider, 'model'):
            model_name = self.llm_provider.model
        
        self.logger.info(
            f"Calculating risk impact for model cost: {model_answer_cost}, "
            f"actual cost: {actual_cost} using model '{model_name}'"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(model_answer_cost, actual_cost)
            self.logger.debug(f"Constructed risk impact prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for impact calculation
            # Temperature=0.1 for reproducibility (as specified in requirements)
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
            
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calculating risk impact: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error calculating risk impact: {str(e)}"
            ) from e


# Backward compatibility: module-level function
def calculate_risk_impact(
    model_answer_cost: Dict[str, Any],
    actual_cost: Dict[str, Any],
    config: Optional[Config] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_version: Optional[str] = None,
    live: bool = True
) -> int:
    """
    Calculate system-level risk impact magnitude (discrete scale: 0, 1, 2, or 3).
    
    This is a convenience function that maintains backward compatibility.
    It creates a RiskImpactEvaluator instance and calls calculate_risk_impact().
    
    Analyzes difference between model answer cost and actual cost from chunks,
    considering mixed resource types (time, money, steps) and their relative importance.
    Evaluates impact of deviations regardless of origin (retrieval, augmentation, context ordering,
    prompting, model reasoning, or hallucination).
    
    **Impact Scale** (discrete values):
    - **0**: Minimal/no impact
    - **1**: Low impact
    - **2**: Moderate impact
    - **3**: High/severe impact
    
    **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
    in the evaluation system requirements. Note that LLM classification is inherently
    non-deterministic, so slight variations may occur across calls.
    
    **Rationale for LLM Node**: Uses an LLM node (not a deterministic function) because
    it must handle mixed resource types (time, money, steps) and assess their relative
    importance, which requires nuanced reasoning about real-world consequences.
    
    Args:
        model_answer_cost: Cost extracted from model answer (time/money/steps)
        actual_cost: Actual cost from retrieved chunks (ground truth)
        config: Application configuration (optional, defaults to Config.from_env())
        query_executor: QueryExecutor instance for database operations (optional, for database prompt loading)
        prompt_version: Prompt version name (e.g., "0.1"). Defaults to None (uses live version)
        live: If True and prompt_version is None, loads the live version. Defaults to True
        
    Returns:
        int: Impact magnitude as discrete value in {0, 1, 2, 3}, where:
            - 0: Minimal/no impact
            - 1: Low impact
            - 2: Moderate impact
            - 3: High/severe impact
            
    Raises:
        AzureServiceError: If Azure Foundry LLM call fails
        ValueError: If inputs are invalid or empty, or if response cannot be parsed
        
    Example:
        >>> from rag_eval.core.config import Config
        >>> 
        >>> config = Config.from_env()
        >>> 
        >>> model_answer_cost = {"money": 500.0, "time": "2 hours"}
        >>> actual_cost = {"money": 50.0, "time": "30 minutes"}
        >>> 
            >>> impact = calculate_risk_impact(model_answer_cost, actual_cost, config)
            >>> print(impact)
            2
    """
    evaluator = RiskImpactEvaluator(
        config=config,
        query_executor=query_executor,
        prompt_version=prompt_version,
        live=live
    )
    return evaluator.calculate_risk_impact(model_answer_cost, actual_cost)

