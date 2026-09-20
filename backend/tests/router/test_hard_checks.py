from app.router.hard_checks import HardCapabilityChecker
from app.router.prompt_analyzer import PromptAnalysis


def test_hard_check_context_overflow():
    checker = HardCapabilityChecker()
    analysis = PromptAnalysis(
        task_type="explanation",
        difficulty="hard",
        required_capabilities={"explanation": 0.9},
        complexity_score=0.4,
        reasoning_requirement=0.2,
        context_requirement=1.0,
        structured_output_required=False,
        prompt_length_chars=7500,
    )
    passed, reason = checker.check(analysis, total_chars=7500)
    assert passed is False
    assert "exceeds local model window limit" in reason


def test_hard_check_extreme_formal_reasoning():
    checker = HardCapabilityChecker()
    analysis = PromptAnalysis(
        task_type="reasoning",
        difficulty="hard",
        required_capabilities={"complex_reasoning": 0.95},
        complexity_score=0.85,
        reasoning_requirement=0.90,
        context_requirement=0.2,
        structured_output_required=False,
        prompt_length_chars=400,
    )
    passed, reason = checker.check(analysis, total_chars=400)
    assert passed is False
    assert "reasoning ceiling" in reason


def test_hard_check_normal_task_passes():
    checker = HardCapabilityChecker()
    analysis = PromptAnalysis(
        task_type="coding",
        difficulty="easy",
        required_capabilities={"coding": 0.85},
        complexity_score=0.2,
        reasoning_requirement=0.2,
        context_requirement=0.1,
        structured_output_required=False,
        prompt_length_chars=50,
    )
    passed, reason = checker.check(analysis, total_chars=50)
    assert passed is True
    assert "Passed" in reason
