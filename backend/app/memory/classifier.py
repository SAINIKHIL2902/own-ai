from datetime import datetime, timedelta, timezone
import re
from typing import Optional, Tuple

from app.memory.models import MemoryType


class MemoryClassifier:
    """
    Classifies memory candidate text into structured MemoryType and assigns
    baseline importance and potential expiration.
    """

    PREFERENCE_INDICATORS = re.compile(
        r"(?i)\b(prefer|like|dislike|concise|detailed|short|brief|verbose|step-by-step|examples|code|summary|plain text|markdown)\b"
    )
    GOAL_INDICATORS = re.compile(
        r"(?i)\b(goal|target|aim|want to learn|master|build|complete|achieve|studying)\b"
    )
    CONTEXT_INDICATORS = re.compile(
        r"(?i)\b(currently|working on|debugging|implementing|building|phase\s+\d+|temporary|sprint)\b"
    )
    INSTRUCTION_INDICATORS = re.compile(
        r"(?i)\b(always|never|must|format as|respond with|rule|follow)\b"
    )
    INTEREST_INDICATORS = re.compile(
        r"(?i)\b(kafka|python|machine learning|ai|fastapi|docker|kubernetes|sql|postgres|react|rust|golang|devops)\b"
    )

    def classify(self, content: str, pattern_category: str = "") -> Tuple[MemoryType, float, Optional[str]]:
        """
        Returns (memory_type, importance, expires_at_iso_or_none).
        """
        cleaned = content.strip()

        # Category hints from detector
        if pattern_category == "goal" or self.GOAL_INDICATORS.search(cleaned):
            return MemoryType.GOAL, 0.85, None

        if pattern_category == "context" or self.CONTEXT_INDICATORS.search(cleaned):
            # Temporary context expires in 7 days by default
            expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            return MemoryType.CONTEXT, 0.50, expires

        if pattern_category == "instruction_persistent" or self.INSTRUCTION_INDICATORS.search(cleaned):
            return MemoryType.INSTRUCTION, 0.90, None

        if pattern_category.startswith("preference") or self.PREFERENCE_INDICATORS.search(cleaned):
            return MemoryType.PREFERENCE, 0.85, None

        if self.INTEREST_INDICATORS.search(cleaned):
            return MemoryType.INTEREST, 0.75, None

        # Default classification
        return MemoryType.FACT, 0.80, None


memory_classifier = MemoryClassifier()
