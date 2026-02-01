"""Abstract base class for judge implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ...core.interfaces import JudgeEvaluationResult, RetrievalResult


class BaseJudge(ABC):
    """Abstract base class for judge implementations"""
    
    def __init__(self, **config):
        """
        Initialize judge with configuration
        
        Args:
            **config: Judge-specific configuration parameters
        """
        self.config = config
    
    @abstractmethod
    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        context_chunks: List[str],
        **kwargs
    ) -> JudgeEvaluationResult:
        """
        Evaluate a generated answer against reference
        
        Args:
            question: User question
            reference_answer: Ground truth answer
            generated_answer: Model-generated answer
            context_chunks: Retrieved context chunks
            **kwargs: Additional evaluation parameters
            
        Returns:
            JudgeEvaluationResult with evaluation scores
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this judge implementation"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of this judge"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used by this judge"""
        return self.config.copy()
    
    def validate_inputs(
        self, 
        question: str, 
        reference_answer: str, 
        generated_answer: str, 
        context_chunks: List[str]
    ) -> bool:
        """
        Validate inputs before evaluation
        
        Returns:
            True if inputs are valid, False otherwise
        """
        return (
            question and question.strip() and
            reference_answer and reference_answer.strip() and
            generated_answer and generated_answer.strip() and
            context_chunks and len(context_chunks) > 0
        )


class MultiStageJudge(BaseJudge):
    """Base class for judges that perform multi-stage evaluation"""
    
    @abstractmethod
    def evaluate_correctness(self, question: str, reference_answer: str, generated_answer: str) -> bool:
        """Evaluate correctness of the answer"""
        pass
    
    @abstractmethod
    def evaluate_hallucination(self, generated_answer: str, context_chunks: List[str]) -> bool:
        """Evaluate if answer contains hallucinations"""
        pass
    
    @abstractmethod
    def evaluate_risk_direction(self, question: str, generated_answer: str, context_chunks: List[str]) -> Optional[int]:
        """Evaluate risk direction (-1, 0, 1)"""
        pass
    
    @abstractmethod
    def evaluate_risk_impact(self, question: str, generated_answer: str, context_chunks: List[str]) -> Optional[int]:
        """Evaluate risk impact (0-3)"""
        pass
    
    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        context_chunks: List[str],
        **kwargs
    ) -> JudgeEvaluationResult:
        """
        Default implementation using multi-stage evaluation
        """
        if not self.validate_inputs(question, reference_answer, generated_answer, context_chunks):
            raise ValueError("Invalid inputs for evaluation")
        
        # Stage 1: Correctness
        correctness = self.evaluate_correctness(question, reference_answer, generated_answer)
        
        # Stage 2: Hallucination
        hallucination = self.evaluate_hallucination(generated_answer, context_chunks)
        
        # Stage 3 & 4: Risk evaluation (only if hallucination detected)
        risk_direction = None
        risk_impact = None
        if hallucination:
            risk_direction = self.evaluate_risk_direction(question, generated_answer, context_chunks)
            risk_impact = self.evaluate_risk_impact(question, generated_answer, context_chunks)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(correctness, hallucination, risk_direction, risk_impact)
        
        return JudgeEvaluationResult(
            correctness_binary=correctness,
            hallucination_binary=hallucination,
            risk_direction=risk_direction,
            risk_impact=risk_impact,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, correctness: bool, hallucination: bool, risk_direction: Optional[int], risk_impact: Optional[int]) -> str:
        """Generate reasoning text for the evaluation"""
        parts = []
        
        if correctness:
            parts.append("Answer is correct")
        else:
            parts.append("Answer is incorrect")
        
        if hallucination:
            parts.append("hallucination detected")
        else:
            parts.append("no hallucination detected")
        
        if risk_direction is not None:
            risk_desc = {-1: "care avoidance risk", 0: "no clear risk direction", 1: "unexpected cost risk"}
            parts.append(f"risk direction: {risk_desc.get(risk_direction, 'unknown')}")
        
        if risk_impact is not None:
            parts.append(f"risk impact: {risk_impact}")
        
        return "; ".join(parts) + "."