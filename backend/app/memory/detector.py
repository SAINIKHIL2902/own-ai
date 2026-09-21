import re
from typing import Dict, List, Optional


class MemoryDetector:
    """
    Deterministic, high-precision pattern matcher that detects explicit user statements
    intended to be retained as memory.
    Avoids false positives on regular questions and conversational chat.
    """

    PATTERNS: List[Dict[str, Any]] = [
        {
            "category": "remember",
            "regex": re.compile(
                r"(?:please\s+)?\b(?:remember\s+that|remember|keep\s+in\s+mind\s+that)\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "preference",
            "regex": re.compile(
                r"\b(?:i\s+(?:usually\s+)?(?:prefer|love|enjoy))\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "preference",
            "regex": re.compile(
                r"\b(?:i\s+(?:don'?t\s+like|dislike|hate|avoid))\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "interest",
            "regex": re.compile(
                r"\b(?:i\s+am\s+(?:very\s+)?interested\s+in|i'?m\s+interested\s+in|i\s+like\s+(?:exploring|studying))\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "preference",
            "regex": re.compile(
                r"\b(?:i\s+like)\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "goal",
            "regex": re.compile(
                r"\b(?:my\s+goal\s+is\s+to|my\s+goal\s+is|i\s+want\s+to\s+learn|my\s+target\s+is\s+to|i\s+aim\s+to)\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "context",
            "regex": re.compile(
                r"\b(?:i'?m\s+(?:currently\s+)?working\s+on|i\s+am\s+(?:currently\s+)?working\s+on|currently\s+building)\s+(.+)",
                re.IGNORECASE,
            ),
        },
        {
            "category": "instruction",
            "regex": re.compile(
                r"\b(?:always\s+(?:respond\s+with|use|give|explain)|never\s+(?:use|give|include))\s+(.+)",
                re.IGNORECASE,
            ),
        },
    ]


    # Patterns that indicate a regular question or ephemeral command, NOT a persistent memory statement
    EXCLUSIONS = [
        re.compile(r"^(?:what|how|why|when|where|who|which|can\s+you|could\s+you|is\s+it|are\s+there)\b", re.IGNORECASE),
        re.compile(r"\?$", re.MULTILINE),
        re.compile(r"^(?:hello|hi|hey|thanks|thank\s+you|ok|okay|bye)\b", re.IGNORECASE),
    ]

    def detect(self, prompt: str) -> Optional[Dict[str, str]]:
        """
        Extract memory candidates from prompt if explicit pattern is matched with high confidence.
        Returns a dict with {"content": ..., "pattern_category": ...} or None.
        """
        cleaned = prompt.strip()
        if len(cleaned) < 5 or len(cleaned) > 500:
            return None

        # Check exclusion rules: standard questions and greetings must never create memory
        for excl in self.EXCLUSIONS:
            if excl.search(cleaned):
                # If question mark is present, only permit if explicitly preceded by "remember that"
                if not re.match(r"(?i)^(?:please\s+)?remember\b", cleaned):
                    return None

        for item in self.PATTERNS:
            match = item["regex"].search(cleaned)
            if match:
                extracted = match.group(1).strip()
                # Clean punctuation from end
                extracted = re.sub(r"[\.\,\;\!\?]+$", "", extracted).strip()
                if len(extracted) >= 3:
                    return {
                        "content": extracted,
                        "category": item["category"],
                    }

        return None


memory_detector = MemoryDetector()
