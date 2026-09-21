from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from app.config.settings import settings
from app.router.hard_checks import HardCapabilityChecker, hard_capability_checker
from app.router.prompt_analyzer import PromptAnalysis, PromptAnalyzer, prompt_analyzer
from app.router.scoring import RoutingScorer, routing_scorer

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Explainable result of model routing."""
    selected_provider: str  # "local" or "gemini"
    selected_model: str
    local_suitability: float
    routing_confidence: float
    task_type: str
    difficulty: str
    required_capabilities: Dict[str, float]
    reason: str
    hard_check_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_provider": self.selected_provider,
            "selected_model": self.selected_model,
            "local_suitability": self.local_suitability,
            "routing_confidence": self.routing_confidence,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "required_capabilities": self.required_capabilities,
            "reason": self.reason,
            "hard_check_passed": self.hard_check_passed,
        }


class ModelRouter:
    """
    Intelligent Local vs. Gemini model router.
    Evaluates required capabilities, hard capability limits, local model capabilities,
    empirical evidence, and routing confidence to make evidence-based routing decisions.
    """

    def __init__(
        self,
        analyzer: Optional[PromptAnalyzer] = None,
        hard_checker: Optional[HardCapabilityChecker] = None,
        scorer: Optional[RoutingScorer] = None,
    ):
        self.analyzer = analyzer or prompt_analyzer
        self.hard_checker = hard_checker or hard_capability_checker
        self.scorer = scorer or routing_scorer

    def route(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> RoutingDecision:
        """
        Determines whether the request should be routed to Local Ollama or Google Gemini.
        """
        # 1. Analyze prompt requirements
        analysis = self.analyzer.analyze(prompt, conversation_history)

        # 2. Hard capability check (evaluates prompt demands against physical boundaries)
        hard_passed, hard_reason = self.hard_checker.check(analysis, analysis.prompt_length_chars)
        if not hard_passed:
            logger.info(f"ModelRouter: Hard capability limit exceeded -> GEMINI ({hard_reason})")
            return RoutingDecision(
                selected_provider="gemini",
                selected_model=settings.GEMINI_MODEL,
                local_suitability=0.20,
                routing_confidence=0.95,
                task_type=analysis.task_type,
                difficulty=analysis.difficulty,
                required_capabilities=analysis.required_capabilities,
                reason=f"Hard capability limit exceeded: {hard_reason}",
                hard_check_passed=False,
            )

        # 3. Calculate Local Suitability and Routing Confidence
        suitability, confidence = self.scorer.calculate_scores(analysis)

        # 4. Decision Logic with Explicit Uncertainty Handling
        min_suitability = settings.LOCAL_MIN_SUITABILITY
        min_confidence = settings.LOCAL_MIN_CONFIDENCE

        is_suitable = suitability >= min_suitability
        is_confident = confidence >= min_confidence

        if is_suitable and is_confident:
            selected_provider = "local"
            selected_model = settings.OLLAMA_MODEL
            reason = (
                f"Local model is sufficiently capable (suitability {suitability} >= {min_suitability}) "
                f"and routing confidence is high ({confidence} >= {min_confidence})."
            )
        elif not is_suitable:
            selected_provider = "gemini"
            selected_model = settings.GEMINI_MODEL
            reason = (
                f"Local capability is insufficient for {analysis.task_type} "
                f"(suitability {suitability} < {min_suitability})."
            )
        else:
            # Uncertain State (Medium/High suitability but Low confidence)
            selected_provider = "gemini"
            selected_model = settings.GEMINI_MODEL
            reason = (
                f"Routing confidence is low ({confidence} < {min_confidence}); "
                "defaulting safely to Gemini to guarantee quality."
            )

        logger.info(
            f"ModelRouter decision: provider={selected_provider}, model={selected_model}, "
            f"suitability={suitability}, confidence={confidence}, task={analysis.task_type}, diff={analysis.difficulty}"
        )

        return RoutingDecision(
            selected_provider=selected_provider,
            selected_model=selected_model,
            local_suitability=suitability,
            routing_confidence=confidence,
            task_type=analysis.task_type,
            difficulty=analysis.difficulty,
            required_capabilities=analysis.required_capabilities,
            reason=reason,
            hard_check_passed=True,
        )


model_router = ModelRouter()
