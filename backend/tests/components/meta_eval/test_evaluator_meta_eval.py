"""Unit tests for Meta-Evaluator (deterministic validation)"""

import pytest
from typing import List, Dict, Any, Optional

from rag_eval.core.interfaces import (
    JudgeEvaluationResult,
    MetaEvaluationResult,
    RetrievalResult,
    JudgePerformanceMetrics
)
from rag_eval.services.evaluator.meta_eval import (
    meta_evaluate_judge,
    calculate_judge_metrics,
    _validate_correctness,
    _validate_hallucination,
    _validate_cost_classification,
    _validate_impact_magnitude,
    _generate_explanation
)


@pytest.fixture
def sample_retrieved_context():
    """Create sample retrieved context"""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            similarity_score=0.95,
            chunk_text="The copay for a specialist visit is $50. The deductible is $1000 per year."
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            similarity_score=0.90,
            chunk_text="Coinsurance is 20% after deductible is met."
        )
    ]


@pytest.fixture
def sample_model_answer():
    """Sample model answer"""
    return "The copay for specialist visits is $50."


@pytest.fixture
def sample_reference_answer():
    """Sample reference answer"""
    return "Specialist visits have a $50 copay."


@pytest.fixture
def sample_judge_output_correct():
    """Sample judge output with correct verdicts"""
    return JudgeEvaluationResult(
        correctness_binary=True,
        hallucination_binary=False,
        risk_direction=None,
        risk_impact=None,
        reasoning="Model answer matches reference answer. No hallucinations detected.",
        failure_mode=None
    )


@pytest.fixture
def sample_judge_output_incorrect():
    """Sample judge output with incorrect verdicts"""
    return JudgeEvaluationResult(
        correctness_binary=False,
        hallucination_binary=True,
        risk_direction=None,
        risk_impact=None,
        reasoning="Model answer differs from reference. Hallucinations detected.",
        failure_mode="cost misstatement"
    )


class TestMetaEvaluateJudge:
    """Test the main meta_evaluate_judge function"""
    
    def test_input_validation_empty_model_answer(
        self, sample_judge_output_correct, sample_retrieved_context, sample_reference_answer
    ):
        """Test that empty model answer raises ValueError"""
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            meta_evaluate_judge(
                judge_output=sample_judge_output_correct,
                retrieved_context=sample_retrieved_context,
                model_answer="",
                reference_answer=sample_reference_answer
            )
    
    def test_input_validation_empty_reference_answer(
        self, sample_judge_output_correct, sample_retrieved_context, sample_model_answer
    ):
        """Test that empty reference answer raises ValueError"""
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            meta_evaluate_judge(
                judge_output=sample_judge_output_correct,
                retrieved_context=sample_retrieved_context,
                model_answer=sample_model_answer,
                reference_answer=""
            )
    
    def test_input_validation_empty_retrieved_context(
        self, sample_judge_output_correct, sample_model_answer, sample_reference_answer
    ):
        """Test that empty retrieved context raises ValueError"""
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            meta_evaluate_judge(
                judge_output=sample_judge_output_correct,
                retrieved_context=[],
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer
            )
    
    def test_judge_correct_all_validations_pass(
        self, sample_judge_output_correct, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test judge_correct classification when all verdicts are correct"""
        result = meta_evaluate_judge(
            judge_output=sample_judge_output_correct,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert isinstance(result, MetaEvaluationResult)
        assert result.judge_correct is True
        assert result.explanation is not None
        assert "All validations passed" in result.explanation
    
    def test_judge_incorrect_correctness_mismatch(
        self, sample_judge_output_incorrect, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test judge_incorrect classification when correctness verdict is wrong"""
        # Model answer matches reference, but judge says incorrect
        result = meta_evaluate_judge(
            judge_output=sample_judge_output_incorrect,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert isinstance(result, MetaEvaluationResult)
        assert result.judge_correct is False
        assert result.explanation is not None
        assert "Some validations failed" in result.explanation or "mismatch" in result.explanation
    
    def test_judge_correct_hallucination_verdict(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test judge_correct classification when hallucination verdict is correct"""
        # Model answer is grounded in context, judge says no hallucination
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="No hallucination detected.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert result.judge_correct is True
    
    def test_judge_incorrect_hallucination_verdict(
        self, sample_retrieved_context, sample_reference_answer
    ):
        """Test judge_incorrect classification when hallucination verdict is wrong"""
        # Model answer with ungrounded claim
        model_answer = "The copay is $200 and there is a $5000 annual fee."
        # Judge says no hallucination, but claim is not in context
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,  # Wrong - should be True
            risk_direction=None,
            risk_impact=None,
            reasoning="No hallucination detected.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert result.judge_correct is False
    
    def test_validation_risk_direction_with_costs(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test validation of risk_direction against ground truth costs"""
        # Model overestimates cost (extracted > actual) -> should be -1
        extracted_costs = {"money": "$100"}
        actual_costs = {"money": "$50"}
        
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=-1,  # Correct: overestimated
            risk_impact=2,
            reasoning="Cost overestimated.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        
        assert result.judge_correct is True
    
    def test_validation_risk_direction_incorrect(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test validation of risk_direction when judge verdict is incorrect"""
        # Model underestimates cost (extracted < actual) -> should be +1
        extracted_costs = {"money": "$30"}
        actual_costs = {"money": "$50"}
        
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=-1,  # Wrong: should be +1 (underestimated)
            risk_impact=1,
            reasoning="Cost direction analysis.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        
        assert result.judge_correct is False
    
    def test_validation_risk_impact_with_costs(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test validation of risk_impact against ground truth costs"""
        # Large cost difference (>50%) -> should be impact 3
        # extracted=$100, actual=$30: overestimated, so risk_direction should be -1
        extracted_costs = {"money": "$100"}
        actual_costs = {"money": "$30"}
        
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=-1,  # Correct: overestimated (extracted > actual)
            risk_impact=3,  # Correct: large difference (>50%)
            reasoning="Large cost impact.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        
        assert result.judge_correct is True
    
    def test_validation_risk_impact_within_tolerance(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test validation of risk_impact with tolerance (within 1 level)"""
        # Cost difference: (60-40)/40 = 50% -> expected impact 3 (>=50%)
        # But test with 20-50% range: extracted=$55, actual=$45 -> 22% difference -> expected impact 2
        extracted_costs = {"money": "$55"}
        actual_costs = {"money": "$45"}
        
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=-1,  # Correct: overestimated (extracted > actual)
            risk_impact=1,  # Within tolerance of expected 2 (2-1=1 <= 1)
            reasoning="Cost impact analysis.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        
        assert result.judge_correct is True
    
    def test_deterministic_explanation_generation(
        self, sample_judge_output_correct, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test deterministic explanation generation"""
        result = meta_evaluate_judge(
            judge_output=sample_judge_output_correct,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert result.explanation is not None
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
    
    def test_partial_judge_correctness(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test edge case: partial judge correctness (some verdicts correct, others incorrect)"""
        # Correctness correct, but hallucination wrong
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=True,  # Wrong: answer is grounded
            risk_direction=None,
            risk_impact=None,
            reasoning="Partial correctness.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer
        )
        
        assert result.judge_correct is False
    
    def test_missing_ground_truth_costs(
        self, sample_retrieved_context, sample_model_answer, sample_reference_answer
    ):
        """Test edge case: missing ground truth cost information"""
        # When costs are None, risk validations should pass (not applicable)
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="No cost information available.",
            failure_mode=None
        )
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            extracted_costs=None,
            actual_costs=None
        )
        
        assert result.judge_correct is True
    
    def test_zero_retrieved_chunks(
        self, sample_model_answer, sample_reference_answer
    ):
        """Test edge case: zero retrieved chunks"""
        # Empty context should raise ValueError
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=True,
            risk_direction=None,
            risk_impact=None,
            reasoning="No context available.",
            failure_mode=None
        )
        
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=[],
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer
            )
    
    def test_correctness_false_no_cost_validation(
        self, sample_retrieved_context, sample_reference_answer
    ):
        """Test that cost validations are skipped when correctness is False"""
        # Use a model answer that doesn't match reference, so judge saying False is correct
        model_answer = "The copay is $100."  # Different from reference
        reference = "Specialist visits have a $50 copay."
        
        # When correctness is False, risk validations should pass (not applicable)
        judge_output = JudgeEvaluationResult(
            correctness_binary=False,  # Correct: model doesn't match reference
            hallucination_binary=False,
            risk_direction=None,  # Not validated when correctness is False
            risk_impact=None,
            reasoning="Answer is incorrect.",
            failure_mode=None
        )
        
        extracted_costs = {"money": "$100"}
        actual_costs = {"money": "$50"}
        
        result = meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=sample_retrieved_context,
            model_answer=model_answer,
            reference_answer=reference,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs
        )
        
        # Risk validations should pass (not applicable when correctness is False)
        # Correctness verdict is correct (False matches ground truth), hallucination is correct
        assert result.judge_correct is True  # Only correctness/hallucination matter


class TestValidateCorrectness:
    """Test _validate_correctness helper function"""
    
    def test_correct_verdict_exact_match(self):
        """Test validation when judge correctly identifies exact match"""
        model = "The copay is $50."
        reference = "The copay is $50."
        
        assert _validate_correctness(True, model, reference) is True
    
    def test_incorrect_verdict_exact_match(self):
        """Test validation when judge incorrectly identifies exact match"""
        model = "The copay is $50."
        reference = "The copay is $50."
        
        assert _validate_correctness(False, model, reference) is False
    
    def test_correct_verdict_semantic_similar(self):
        """Test validation when judge correctly identifies semantic similarity"""
        model = "Specialist visits have a $50 copay."
        reference = "The copay for specialist visits is $50."
        
        assert _validate_correctness(True, model, reference) is True
    
    def test_incorrect_verdict_different_answers(self):
        """Test validation when judge correctly identifies different answers"""
        model = "The copay is $50."
        reference = "The copay is $100."
        
        assert _validate_correctness(False, model, reference) is True


class TestValidateHallucination:
    """Test _validate_hallucination helper function"""
    
    def test_correct_verdict_no_hallucination(self):
        """Test validation when judge correctly identifies no hallucination"""
        model = "The copay for specialist visits is $50."
        context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for a specialist visit is $50."
            )
        ]
        
        assert _validate_hallucination(False, model, context) is True
    
    def test_correct_verdict_hallucination_detected(self):
        """Test validation when judge correctly identifies hallucination"""
        model = "The copay is $200 and there is a $5000 annual fee."
        context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for a specialist visit is $50."
            )
        ]
        
        assert _validate_hallucination(True, model, context) is True
    
    def test_incorrect_verdict_missed_hallucination(self):
        """Test validation when judge misses hallucination"""
        model = "The copay is $200 and there is a $5000 annual fee."
        context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for a specialist visit is $50."
            )
        ]
        
        assert _validate_hallucination(False, model, context) is False
    
    def test_empty_context_always_hallucination(self):
        """Test that empty context always indicates hallucination"""
        model = "The copay is $50."
        context = []
        
        assert _validate_hallucination(True, model, context) is True
        assert _validate_hallucination(False, model, context) is False


class TestValidateCostClassification:
    """Test _validate_cost_classification helper function"""
    
    def test_correct_opportunity_cost(self):
        """Test validation when judge correctly identifies opportunity cost (-1)"""
        judge_cost = -1
        extracted = {"money": "$100"}
        actual = {"money": "$50"}
        
        assert _validate_cost_classification(judge_cost, extracted, actual) is True
    
    def test_correct_resource_cost(self):
        """Test validation when judge correctly identifies resource cost (+1)"""
        judge_cost = 1
        extracted = {"money": "$30"}
        actual = {"money": "$50"}
        
        assert _validate_cost_classification(judge_cost, extracted, actual) is True
    
    def test_correct_no_direction(self):
        """Test validation when judge correctly identifies no clear direction (0)"""
        judge_cost = 0
        extracted = {"money": "$50"}
        actual = {"money": "$52"}  # Within 10% threshold
        
        assert _validate_cost_classification(judge_cost, extracted, actual) is True
    
    def test_incorrect_direction(self):
        """Test validation when judge incorrectly identifies direction"""
        judge_cost = -1  # Says overestimated
        extracted = {"money": "$30"}
        actual = {"money": "$50"}  # Actually underestimated
        
        assert _validate_cost_classification(judge_cost, extracted, actual) is False
    
    def test_none_judge_cost(self):
        """Test validation when judge cost is None (not applicable)"""
        judge_cost = None
        extracted = {"money": "$50"}
        actual = {"money": "$50"}
        
        assert _validate_cost_classification(judge_cost, extracted, actual) is True


class TestValidateImpactMagnitude:
    """Test _validate_impact_magnitude helper function"""
    
    def test_correct_high_impact(self):
        """Test validation when judge correctly identifies high impact (3)"""
        judge_impact = 3
        extracted = {"money": "$100"}
        actual = {"money": "$30"}  # >50% difference
        
        assert _validate_impact_magnitude(judge_impact, extracted, actual) is True
    
    def test_correct_low_impact(self):
        """Test validation when judge correctly identifies low impact (1)"""
        judge_impact = 1
        extracted = {"money": "$55"}
        actual = {"money": "$50"}  # 5-20% difference
        
        assert _validate_impact_magnitude(judge_impact, extracted, actual) is True
    
    def test_impact_within_tolerance(self):
        """Test validation when judge impact is within tolerance"""
        # Cost difference: (55-45)/45 = 22% -> expected impact 2
        judge_impact = 1  # Expected is 2, but within 1 level tolerance (2-1=1 <= 1)
        extracted = {"money": "$55"}
        actual = {"money": "$45"}  # 20-50% difference (expected 2)
        
        assert _validate_impact_magnitude(judge_impact, extracted, actual) is True
    
    def test_impact_outside_tolerance(self):
        """Test validation when judge impact is outside tolerance"""
        judge_impact = 0  # Expected is 2, outside tolerance
        extracted = {"money": "$60"}
        actual = {"money": "$40"}  # 20-50% difference (expected 2)
        
        assert _validate_impact_magnitude(judge_impact, extracted, actual) is False
    
    def test_none_judge_impact(self):
        """Test validation when judge impact is None (not applicable)"""
        judge_impact = None
        extracted = {"money": "$50"}
        actual = {"money": "$50"}
        
        assert _validate_impact_magnitude(judge_impact, extracted, actual) is True


class TestGenerateExplanation:
    """Test _generate_explanation helper function"""
    
    def test_all_validations_pass(self):
        """Test explanation when all validations pass"""
        validation_results = {
            "correctness": True,
            "hallucination": True,
            "risk_direction": True,
            "risk_impact": True
        }
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="All good.",
            failure_mode=None
        )
        
        explanation = _generate_explanation(validation_results, judge_output)
        
        assert "All validations passed" in explanation
        assert "validated" in explanation
    
    def test_some_validations_fail(self):
        """Test explanation when some validations fail"""
        validation_results = {
            "correctness": True,
            "hallucination": False,
            "risk_direction": True,
            "risk_impact": True
        }
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="Some issues.",
            failure_mode=None
        )
        
        explanation = _generate_explanation(validation_results, judge_output)
        
        assert "Some validations failed" in explanation
        assert "mismatch" in explanation


class TestCalculateJudgeMetrics:
    """Test calculate_judge_metrics function for data science metrics"""
    
    def test_empty_results_raises_error(self):
        """Test that empty evaluation results raises ValueError"""
        with pytest.raises(ValueError, match="evaluation_results cannot be empty"):
            calculate_judge_metrics([])
    
    def test_correctness_metrics_perfect_score(self, sample_retrieved_context):
        """Test correctness metrics with perfect judge performance"""
        # Create 5 examples where judge is always correct
        # Use exact matches to ensure ground truth is True
        results = []
        for i in range(5):
            judge_output = JudgeEvaluationResult(
                correctness_binary=True,
                hallucination_binary=False,
                risk_direction=None,
                risk_impact=None,
                reasoning="Correct.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer="The copay is $50.",
                reference_answer="The copay is $50."  # Exact match
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        assert metrics.correctness.precision == 1.0
        assert metrics.correctness.recall == 1.0
        assert metrics.correctness.f1_score == 1.0
        assert metrics.correctness.true_positives == 5
        assert metrics.correctness.false_positives == 0
        assert metrics.correctness.false_negatives == 0
        assert metrics.correctness.total_samples == 5
    
    def test_correctness_metrics_with_errors(self, sample_retrieved_context):
        """Test correctness metrics with judge errors"""
        results = []
        
        # Test cases: (model_answer, reference_answer, judge_verdict, expected_ground_truth)
        # Ground truth: True if model matches reference, False otherwise
        test_cases = [
            ("The copay is $50.", "The copay is $50.", True, True),  # Judge correct: both True
            ("The copay is $50.", "The copay is $50.", True, True),  # Judge correct: both True
            ("The copay is $100.", "The copay is $50.", True, False),  # Judge wrong: says True, should be False
            ("The copay is $50.", "The copay is $50.", False, True),  # Judge wrong: says False, should be True
            ("The copay is $50.", "The copay is $100.", False, False),  # Judge correct: both False
        ]
        
        for model, reference, judge_verdict, expected_gt in test_cases:
            judge_output = JudgeEvaluationResult(
                correctness_binary=judge_verdict,
                hallucination_binary=False,
                risk_direction=None,
                risk_impact=None,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer=model,
                reference_answer=reference
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        # Should have some errors (not all perfect)
        # We have 2 errors out of 5: case 3 (FP) and case 4 (FN)
        assert metrics.correctness.total_samples == 5
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.67
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.67
        assert metrics.correctness.precision < 1.0
        assert metrics.correctness.recall < 1.0
        assert metrics.correctness.f1_score < 1.0
    
    def test_hallucination_metrics(self, sample_retrieved_context, sample_reference_answer):
        """Test hallucination metrics calculation"""
        results = []
        
        # Mix of hallucination and non-hallucination cases
        test_cases = [
            ("The copay is $50.", False),  # No hallucination (grounded)
            ("The copay is $200.", True),  # Hallucination (ungrounded claim)
            ("The copay is $50.", False),  # No hallucination
        ]
        
        for model, has_hallucination in test_cases:
            judge_output = JudgeEvaluationResult(
                correctness_binary=True,
                hallucination_binary=has_hallucination,
                risk_direction=None,
                risk_impact=None,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer=model,
                reference_answer=sample_reference_answer
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        assert metrics.hallucination.total_samples == 3
        assert metrics.hallucination.precision >= 0.0
        assert metrics.hallucination.recall >= 0.0
        assert metrics.hallucination.f1_score >= 0.0
    
    def test_risk_direction_metrics(self, sample_retrieved_context, sample_reference_answer):
        """Test risk direction metrics calculation"""
        results = []
        
        # Test cases with cost information
        test_cases = [
            ({"money": "$100"}, {"money": "$50"}, -1),  # Overestimated
            ({"money": "$30"}, {"money": "$50"}, 1),  # Underestimated
            ({"money": "$50"}, {"money": "$52"}, 0),  # No clear direction
        ]
        
        for extracted, actual, expected_direction in test_cases:
            judge_output = JudgeEvaluationResult(
                correctness_binary=True,
                hallucination_binary=False,
                risk_direction=expected_direction,
                risk_impact=1,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer="The copay is $50.",
                reference_answer=sample_reference_answer,
                extracted_costs=extracted,
                actual_costs=actual
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        assert metrics.risk_direction is not None
        assert metrics.risk_direction.total_samples == 3
        assert metrics.risk_direction.precision >= 0.0
        assert metrics.risk_direction.recall >= 0.0
        assert metrics.risk_direction.f1_score >= 0.0
    
    def test_risk_impact_metrics(self, sample_retrieved_context, sample_reference_answer):
        """Test risk impact metrics calculation"""
        results = []
        
        # Test cases with different impact levels
        test_cases = [
            ({"money": "$55"}, {"money": "$50"}, 1),  # Low impact (5-20%)
            ({"money": "$60"}, {"money": "$40"}, 2),  # Moderate impact (20-50%)
            ({"money": "$100"}, {"money": "$30"}, 3),  # High impact (>50%)
        ]
        
        for extracted, actual, expected_impact in test_cases:
            judge_output = JudgeEvaluationResult(
                correctness_binary=True,
                hallucination_binary=False,
                risk_direction=-1,
                risk_impact=expected_impact,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer="The copay is $50.",
                reference_answer=sample_reference_answer,
                extracted_costs=extracted,
                actual_costs=actual
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        assert metrics.risk_impact is not None
        assert metrics.risk_impact.total_samples == 3
        assert metrics.risk_impact.precision >= 0.0
        assert metrics.risk_impact.recall >= 0.0
        assert metrics.risk_impact.f1_score >= 0.0
    
    def test_missing_optional_metrics(self, sample_retrieved_context, sample_reference_answer):
        """Test that missing optional metrics (risk_direction, risk_impact) are handled"""
        results = []
        
        # Create results without cost information
        for i in range(3):
            judge_output = JudgeEvaluationResult(
                correctness_binary=True,
                hallucination_binary=False,
                risk_direction=None,
                risk_impact=None,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer="The copay is $50.",
                reference_answer=sample_reference_answer
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        # Correctness and hallucination should be present
        assert metrics.correctness.total_samples == 3
        assert metrics.hallucination.total_samples == 3
        
        # Risk metrics should be None (no cost data)
        assert metrics.risk_direction is None
        assert metrics.risk_impact is None
    
    def test_mixed_metrics_scenario(self, sample_retrieved_context, sample_reference_answer):
        """Test comprehensive metrics calculation with mixed scenarios"""
        results = []
        
        # Mix of scenarios
        scenarios = [
            # (model, reference, judge_correct, judge_hallucination, extracted_costs, actual_costs)
            ("The copay is $50.", "The copay is $50.", True, False, None, None),
            ("The copay is $100.", "The copay is $50.", False, True, None, None),
            ("The copay is $50.", "The copay is $50.", True, False, {"money": "$100"}, {"money": "$50"}),
        ]
        
        for model, reference, judge_correct, judge_hallucination, extracted, actual in scenarios:
            judge_output = JudgeEvaluationResult(
                correctness_binary=judge_correct,
                hallucination_binary=judge_hallucination,
                risk_direction=-1 if extracted else None,
                risk_impact=2 if extracted else None,
                reasoning="Test.",
                failure_mode=None
            )
            meta_eval = meta_evaluate_judge(
                judge_output=judge_output,
                retrieved_context=sample_retrieved_context,
                model_answer=model,
                reference_answer=reference,
                extracted_costs=extracted,
                actual_costs=actual
            )
            results.append((judge_output, meta_eval))
        
        metrics = calculate_judge_metrics(results)
        
        # All metrics should be calculated where data is available
        assert metrics.correctness.total_samples == 3
        assert metrics.hallucination.total_samples == 3
        # Only one result has cost data
        assert metrics.risk_direction is not None
        assert metrics.risk_direction.total_samples == 1
        assert metrics.risk_impact is not None
        assert metrics.risk_impact.total_samples == 1

