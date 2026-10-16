import re
from typing import Dict, List, Optional, Tuple

from app.memory.models import MemoryRecord, MemorySource


class ConfidenceManager:
    """
    Manages memory confidence calculations, evidence reinforcement,
    and contradiction resolution.
    """

    DEFAULT_CONFIDENCE: Dict[str, float] = {
        MemorySource.EXPLICIT_USER.value: 0.95,
        MemorySource.CONVERSATION.value: 0.70,
        MemorySource.BEHAVIOR.value: 0.60,
        MemorySource.FEEDBACK.value: 0.65,
        MemorySource.SYSTEM.value: 0.80,
    }

    # Opposing preference pairs (antonyms)
    OPPOSING_TRAITS: List[Tuple[re.Pattern, re.Pattern]] = [
        (
            re.compile(r"\b(concise|short|brief|bullet\s*points?|compact)\b", re.IGNORECASE),
            re.compile(r"\b(detailed|verbose|thorough|in-depth|extensive|long)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(step-by-step|step\s+by\s+step|walkthrough|gradual)\b", re.IGNORECASE),
            re.compile(r"\b(high-level|summary|bird's\s*eye|quick\s*overview)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(only\s+code|code\s+only|snippets?\s+only)\b", re.IGNORECASE),
            re.compile(r"\b(no\s+code|plain\s+english|conceptual\s+only)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(dark\s+mode|dark\s+theme)\b", re.IGNORECASE),
            re.compile(r"\b(light\s+mode|light\s+theme)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(fast|speed|quick)\b", re.IGNORECASE),
            re.compile(r"\b(thorough|deep|exhaustive)\b", re.IGNORECASE),
        ),
    ]

    def get_initial_confidence(self, source_type: str) -> float:
        """Returns baseline confidence for given source type."""
        return self.DEFAULT_CONFIDENCE.get(source_type, 0.50)


    def reinforce(self, current_confidence: float, evidence_count: int) -> Tuple[float, int]:
        """
        Asymptotically increase confidence when matching supporting evidence is observed.
        Formula: conf = conf + (1.0 - conf) * 0.15
        """
        new_conf = current_confidence + (1.0 - current_confidence) * 0.15
        new_evidence = evidence_count + 1
        return round(min(new_conf, 1.0), 3), new_evidence

    def penalize(self, current_confidence: float) -> float:
        """Reduce confidence upon receiving contradictory feedback."""
        new_conf = current_confidence * 0.80
        return round(max(new_conf, 0.10), 3)

    def is_contradiction(self, content_a: str, content_b: str) -> bool:
        """
        Determine if two preference statements are fundamentally contradictory.
        E.g. 'prefer concise answers' vs 'prefer detailed explanations'.
        """
        ca = content_a.lower()
        cb = content_b.lower()

        # Direct like vs dislike negation check
        if ("like" in ca and "dislike" in cb) or ("dislike" in ca and "like" in cb):
            return True
        if ("don't like" in ca and "prefer" in cb) or ("prefer" in ca and "don't like" in cb):
            return True

        # Opposing traits check
        for pat1, pat2 in self.OPPOSING_TRAITS:
            if (pat1.search(ca) and pat2.search(cb)) or (pat2.search(ca) and pat1.search(cb)):
                return True

        return False

    def reconcile_contradiction(
        self,
        existing: MemoryRecord,
        new_content: str,
        new_source: str,
    ) -> MemoryRecord:
        """
        Reconcile an existing memory when the user provides an updated, contradictory preference.
        The newer explicit instruction supersedes the previous state.
        """
        existing.content = new_content
        existing.source_type = new_source
        existing.confidence = self.get_initial_confidence(new_source)
        existing.evidence_count += 1
        existing.status = "active"
        return existing


confidence_manager = ConfidenceManager()
