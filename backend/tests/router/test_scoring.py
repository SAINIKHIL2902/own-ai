from app.router.prompt_analyzer import PromptAnalysis
from app.router.scoring import RoutingScorer


def test_scoring_easy_prompt_high_suitability():
    scorer = RoutingScorer()
    analysis = PromptAnalysis(
        task_type="explanation",
        difficulty="easy",
        required_capabilities={"explanation": 0.85, "general_knowledge": 0.80},
        complexity_score=0.15,
        reasoning_requirement=0.15,
        context_requirement=0.10,
        structured_output_required=False,
    )
    suitability, confidence = scorer.calculate_scores(analysis)
    assert suitability >= 0.75
    assert confidence >= 0.65


def test_scoring_hard_prompt_low_suitability():
    scorer = RoutingScorer()
    analysis = PromptAnalysis(
        task_type="reasoning",
        difficulty="hard",
        required_capabilities={"complex_reasoning": 0.95, "multi_step_reasoning": 0.90},
        complexity_score=0.80,
        reasoning_requirement=0.85,
        context_requirement=0.40,
        structured_output_required=False,
    )
    suitability, confidence = scorer.calculate_scores(analysis)
    assert suitability < 0.60


def test_scoring_uncertainty_state_low_confidence():
    scorer = RoutingScorer()
    # A borderline hard task near threshold
    analysis = PromptAnalysis(
        task_type="coding",
        difficulty="hard",
        required_capabilities={"coding": 0.75},
        complexity_score=0.55,
        reasoning_requirement=0.50,
        context_requirement=0.30,
        structured_output_required=False,
    )
    suitability, confidence = scorer.calculate_scores(analysis)
    # On hard borderline tasks, confidence is dialed down to prevent overconfidence
    assert confidence < 0.65 or suitability < 0.70
