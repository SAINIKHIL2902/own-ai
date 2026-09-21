from datetime import datetime, timezone
import math
import re
from typing import Set

from app.config.settings import settings
from app.memory.models import MemoryRecord, MemoryType


def _tokenize(text: str) -> Set[str]:
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "is", "are", "was", "were", "it", "that", "this", "user",
        "please", "can", "you", "me", "my", "i", "how", "what",
    }
    return {w for w in words if w not in stop_words}


class MemoryScorer:
    """
    Computes a normalized relevance score in [0.0, 1.0] between a user prompt
    and a stored memory record.
    """

    def score(self, prompt: str, memory: MemoryRecord) -> float:
        w_topic = settings.MEMORY_RELEVANCE_TOPIC_WEIGHT
        w_imp = settings.MEMORY_RELEVANCE_IMPORTANCE_WEIGHT
        w_conf = settings.MEMORY_RELEVANCE_CONFIDENCE_WEIGHT
        w_rec = settings.MEMORY_RELEVANCE_RECENCY_WEIGHT

        # 1. Topic / Keyword Match
        prompt_tokens = _tokenize(prompt)
        memory_tokens = _tokenize(memory.content)

        if not prompt_tokens or not memory_tokens:
            topic_match = 0.0
        else:
            intersection = len(prompt_tokens.intersection(memory_tokens))
            if intersection > 0:
                # Jaccard overlap relative to memory token size
                topic_match = min(1.0, intersection / float(len(memory_tokens)))
            else:
                # Only generic formatting/style preferences apply across all topics
                is_style_preference = (
                    memory.memory_type in [MemoryType.PREFERENCE.value, MemoryType.INSTRUCTION.value]
                    and bool(re.search(r"\b(concise|detailed|short|brief|bullet|verbose|code|snippet|format|markdown|summary)\b", memory.content, re.IGNORECASE))
                )
                if is_style_preference:
                    topic_match = 0.35
                else:
                    return 0.0  # Zero relevance if completely unrelated topic


        # 2. Importance
        importance_score = min(max(memory.importance, 0.0), 1.0)

        # 3. Confidence
        confidence_score = min(max(memory.confidence, 0.0), 1.0)

        # 4. Recency
        recency_score = self._compute_recency(memory.updated_at)

        # Composite Relevance Score
        raw = (
            w_topic * topic_match
            + w_imp * importance_score
            + w_conf * confidence_score
            + w_rec * recency_score
        )
        return min(max(round(raw, 3), 0.0), 1.0)

    @staticmethod
    def _compute_recency(updated_at_iso: str) -> float:
        if not updated_at_iso:
            return 0.80
        try:
            dt = datetime.fromisoformat(updated_at_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = max(0.0, (now - dt).total_seconds() / 86400.0)
            # Gentle exponential decay over 30 days
            return round(max(0.20, math.exp(-0.03 * days)), 3)
        except Exception:
            return 0.80


memory_scorer = MemoryScorer()
