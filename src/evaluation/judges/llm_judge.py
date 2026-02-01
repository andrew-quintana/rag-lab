"""LLM-based judge implementation using the original evaluation methodology"""

from typing import List, Dict, Any, Optional, Callable
from ..base.base_judge import MultiStageJudge
from ...core.interfaces import JudgeEvaluationResult
from ...core.registry import register_judge


@register_judge(
    name="llm_judge",
    description="LLM-based judge with multi-stage evaluation",
    methodology="Original RAG evaluator approach"
)
class LLMJudge(MultiStageJudge):
    """LLM-based judge implementation using multi-stage evaluation"""
    
    def __init__(self, llm_function: Callable[[str], str], prompts: Dict[str, str], **config):
        """
        Initialize LLM judge with language model function and prompts
        
        Args:
            llm_function: Function that takes prompt and returns LLM response
            prompts: Dictionary of evaluation prompts by stage
            **config: Additional configuration parameters
        """
        super().__init__(**config)
        self.llm_function = llm_function
        self.prompts = prompts
        
    def get_name(self) -> str:
        return "llm_judge"
        
    def get_description(self) -> str:
        return "LLM-based judge with multi-stage evaluation for correctness, hallucination, and risk assessment"
    
    def evaluate_correctness(self, question: str, reference_answer: str, generated_answer: str) -> bool:
        """Evaluate correctness using LLM"""
        prompt = self.prompts.get("correctness", "").format(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        
        response = self.llm_function(prompt)
        return self._parse_binary_response(response)
    
    def evaluate_hallucination(self, generated_answer: str, context_chunks: List[str]) -> bool:
        """Evaluate hallucination using LLM"""
        context = "\n\n".join(context_chunks)
        prompt = self.prompts.get("hallucination", "").format(
            generated_answer=generated_answer,
            context=context
        )
        
        response = self.llm_function(prompt)
        return self._parse_binary_response(response)
    
    def evaluate_risk_direction(self, question: str, generated_answer: str, context_chunks: List[str]) -> Optional[int]:
        """Evaluate risk direction using LLM"""
        context = "\n\n".join(context_chunks)
        prompt = self.prompts.get("risk_direction", "").format(
            question=question,
            generated_answer=generated_answer,
            context=context
        )
        
        response = self.llm_function(prompt)
        return self._parse_risk_direction_response(response)
    
    def evaluate_risk_impact(self, question: str, generated_answer: str, context_chunks: List[str]) -> Optional[int]:
        """Evaluate risk impact using LLM"""
        context = "\n\n".join(context_chunks)
        prompt = self.prompts.get("risk_impact", "").format(
            question=question,
            generated_answer=generated_answer,
            context=context
        )
        
        response = self.llm_function(prompt)
        return self._parse_risk_impact_response(response)
    
    def _parse_binary_response(self, response: str) -> bool:
        """Parse binary (yes/no, true/false) response from LLM"""
        response_lower = response.lower().strip()
        
        if any(word in response_lower for word in ["yes", "true", "correct", "accurate"]):
            return True
        elif any(word in response_lower for word in ["no", "false", "incorrect", "inaccurate"]):
            return False
        else:
            return False
    
    def _parse_risk_direction_response(self, response: str) -> Optional[int]:
        """Parse risk direction response (-1, 0, 1)"""
        response_lower = response.lower().strip()
        
        if any(word in response_lower for word in ["care avoidance", "overestimate", "-1"]):
            return -1
        elif any(word in response_lower for word in ["unexpected cost", "underestimate", "1"]):
            return 1
        elif any(word in response_lower for word in ["no clear", "neutral", "0"]):
            return 0
        else:
            return None
    
    def _parse_risk_impact_response(self, response: str) -> Optional[int]:
        """Parse risk impact response (0-3)"""
        response_lower = response.lower().strip()
        
        if "3" in response_lower or "severe" in response_lower or "high" in response_lower:
            return 3
        elif "2" in response_lower or "moderate" in response_lower or "medium" in response_lower:
            return 2
        elif "1" in response_lower or "mild" in response_lower or "low" in response_lower:
            return 1
        elif "0" in response_lower or "none" in response_lower or "minimal" in response_lower:
            return 0
        else:
            return None