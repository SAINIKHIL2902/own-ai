from typing import Dict, List, Optional, Tuple

from app.config.settings import settings
from app.router.prompt_analyzer import PromptAnalysis


class HardCapabilityChecker:
    """
    Evaluates physical and structural constraints before numerical scoring.
    Determines if the local model is fundamentally incapable of processing the request.
    """

    def check(self, analysis: PromptAnalysis, total_chars: int) -> Tuple[bool, str]:
        """
        Returns (passes_hard_checks, reason_message).
        If passes_hard_checks is False, the request must immediately be routed to Gemini.
        """
        # 1. Hard Context Length Check
        if total_chars > settings.LOCAL_MAX_CONTEXT_CHARS:
            return (
                False,
                f"Context length ({total_chars} chars) exceeds local model window limit ({settings.LOCAL_MAX_CONTEXT_CHARS} chars).",
            )

        # 2. Hard Task-specific constraints
        if analysis.task_type == "long_context" and analysis.context_requirement >= 0.85:
            return (
                False,
                "Prompt requires extensive long-document context beyond local model memory capability.",
            )

        # 3. Hard Extreme Reasoning / Proofs
        if analysis.task_type == "reasoning" and analysis.difficulty == "hard" and analysis.reasoning_requirement >= 0.85:
            return (
                False,
                "Task requires multi-step formal deductive reasoning exceeding local 1B-3B model reasoning ceiling.",
            )

        return (True, "Passed hard capability checks.")


hard_capability_checker = HardCapabilityChecker()
