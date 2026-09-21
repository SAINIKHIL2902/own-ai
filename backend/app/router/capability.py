from typing import Any, Dict, Optional

# Initial baseline capability profile for local Ollama model (e.g. llama3.2:1b).
# NOTE: These values represent configurable starting assumptions,
# which can be calibrated using empirical benchmark evaluation.
LOCAL_MODEL_CAPABILITIES: Dict[str, float] = {
    "simple_question_answering": 0.90,
    "general_knowledge": 0.85,
    "explanation": 0.85,
    "summarization": 0.85,
    "rewriting": 0.90,
    "translation": 0.80,
    "coding": 0.70,
    "debugging": 0.60,
    "mathematical_reasoning": 0.55,
    "multi_step_reasoning": 0.50,
    "complex_reasoning": 0.40,
    "long_context": 0.40,
    "structured_output": 0.75,
    "data_analysis": 0.50,
    "instruction_following": 0.75,
}

DIFFICULTY_MULTIPLIERS = {
    "easy": 1.15,
    "medium": 1.00,
    "hard": 0.70,
}


class CapabilityProfileManager:
    """
    Manages local model capabilities with support for difficulty adjustments
    and empirical benchmark result overlays.
    """

    def __init__(self, baseline: Optional[Dict[str, float]] = None):
        self._baseline = dict(baseline or LOCAL_MODEL_CAPABILITIES)
        self._empirical_overrides: Dict[str, float] = {}

    def get_effective_capability(self, capability: str, difficulty: str = "medium") -> float:
        """
        Calculate effective local capability score (0.0 - 1.0) taking into account
        empirical benchmark results and task difficulty tier.
        """
        base = self._empirical_overrides.get(capability, self._baseline.get(capability, 0.50))
        mult = DIFFICULTY_MULTIPLIERS.get(difficulty.lower(), 1.0)
        return min(max(round(base * mult, 3), 0.05), 1.0)

    def update_empirical_overrides(self, benchmark_scores: Dict[str, float]) -> None:
        """Update capability profile with measured empirical benchmark evidence."""
        self._empirical_overrides.update(benchmark_scores)

    def get_all_capabilities(self, difficulty: str = "medium") -> Dict[str, float]:
        return {
            cap: self.get_effective_capability(cap, difficulty)
            for cap in self._baseline.keys()
        }


capability_manager = CapabilityProfileManager()
