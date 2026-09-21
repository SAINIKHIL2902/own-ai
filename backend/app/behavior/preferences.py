import re
from typing import Dict, List, Optional

PREFERENCE_PATTERNS = {
    "prefers_code": [
        r"\b(?:code|script|snippet|implementation|function|program)\b",
        r"\b(?:write\s+a\s+python|give\s+me\s+the\s+code)\b",
    ],
    "prefers_step_by_step": [
        r"\b(?:step[\s\-]by[\s\-]step|steps|walkthrough|guide\s+me)\b",
    ],
    "prefers_concise_answers": [
        r"\b(?:concise|briefly|short|in\s+one\s+sentence|tldr|quick\s+summary)\b",
    ],
    "prefers_detailed_answers": [
        r"\b(?:detailed|in[\s\-]depth|deep\s+dive|elaborate|comprehensive|thoroughly)\b",
    ],
    "prefers_examples": [
        r"\b(?:example|examples|for\s+instance|sample)\b",
    ],
    "prefers_real_world_examples": [
        r"\b(?:real[\s\-]world|production|industry|case\s+study|practical\s+scenario)\b",
    ],
    "prefers_bullet_points": [
        r"\b(?:bullet[\s\-]points?|bulleted|bullets|list|in\s+points)\b",
    ],
    "prefers_simple_explanations": [
        r"\b(?:simple|explain\s+like\s+i'm\s+5|eli5|beginner|easy\s+to\s+understand|in\s+plain\s+english)\b",
    ],
    "prefers_troubleshooting": [
        r"\b(?:error|debug|fix|exception|traceback|issue|failing|broken)\b",
    ],
}


def extract_preferences(prompt: str) -> Dict[str, str]:
    """Inspects prompt for specific stylistic preference indicators."""
    detected = {}
    lower_prompt = prompt.lower()

    for pref_key, patterns in PREFERENCE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower_prompt):
                detected[pref_key] = "true"
                break

    # Check for user correction behavior (e.g. "No, explain using...")
    if re.search(r"^(?:no|not\s+like\s+that|instead|rather),?\s+", lower_prompt):
        if "example" in lower_prompt or "using" in lower_prompt:
            detected["prefers_real_world_examples"] = "true"

    return detected


def calculate_confidence(evidence_count: int, threshold: int = 5) -> float:
    """Confidence scales proportionally with evidence count up to 1.0 (requires 5+ observations)."""
    return min(1.0, round(evidence_count / float(threshold), 2))
