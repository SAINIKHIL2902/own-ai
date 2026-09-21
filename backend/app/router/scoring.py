from typing import Dict, Tuple

from app.router.capability import capability_manager
from app.router.prompt_analyzer import PromptAnalysis


class RoutingScorer:
    """
    Computes local suitability score and routing confidence from prompt analysis
    and local capability profiles.
    """

    CAPABILITY_WEIGHT = 0.45
    COMPLEXITY_WEIGHT = 0.25
    REASONING_WEIGHT = 0.20
    CONTEXT_WEIGHT = 0.10

    def calculate_scores(self, analysis: PromptAnalysis) -> Tuple[float, float]:
        """
        Returns (local_suitability, routing_confidence) normalized between 0.0 and 1.0.
        """
        # 1. Capability Match Score
        total_weight = 0.0
        weighted_match_sum = 0.0

        for cap, req_score in analysis.required_capabilities.items():
            local_cap = capability_manager.get_effective_capability(cap, analysis.difficulty)
            if local_cap >= req_score:
                match_val = 1.0
            else:
                deficit = req_score - local_cap
                match_val = max(0.0, 1.0 - deficit)

            weight = req_score
            weighted_match_sum += match_val * weight
            total_weight += weight

        cap_match = (weighted_match_sum / total_weight) if total_weight > 0 else 0.80

        # 2. Complexity Fit
        complexity_fit = max(0.0, 1.0 - analysis.complexity_score)

        # 3. Reasoning Fit
        local_reasoning_cap = capability_manager.get_effective_capability("complex_reasoning", analysis.difficulty)
        reasoning_deficit = max(0.0, analysis.reasoning_requirement - local_reasoning_cap)
        reasoning_fit = max(0.0, 1.0 - reasoning_deficit)

        # 4. Context Fit
        context_fit = max(0.0, 1.0 - analysis.context_requirement)

        # Composite Local Suitability Score
        raw_suitability = (
            cap_match * self.CAPABILITY_WEIGHT
            + complexity_fit * self.COMPLEXITY_WEIGHT
            + reasoning_fit * self.REASONING_WEIGHT
            + context_fit * self.CONTEXT_WEIGHT
        )
        local_suitability = min(max(round(raw_suitability, 3), 0.0), 1.0)

        # 5. Routing Confidence Calculation
        # Confidence is high when suitability is clearly distant from threshold (e.g. 0.70)
        # and prompt classification signals are unambiguous.
        margin = abs(local_suitability - 0.70)
        signal_clarity = 1.0 if analysis.difficulty in ["easy", "hard"] else 0.85

        if analysis.difficulty == "hard" and local_suitability > 0.60:
            # Uncertain borderline zone on hard tasks
            raw_confidence = 0.45 + (margin * 0.8)
        else:
            raw_confidence = 0.60 + (margin * 1.2) * signal_clarity

        routing_confidence = min(max(round(raw_confidence, 3), 0.1), 0.99)

        return local_suitability, routing_confidence


routing_scorer = RoutingScorer()
