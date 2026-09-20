import re
from typing import Any, Dict, List, Set


# Common hedge words and vague phrasing that indicate ambiguity
AMBIGUITY_PATTERNS = [
    r"\b(?:maybe|perhaps|possibly|probably|might|could\s+be)\b",
    r"\b(?:sort\s+of|kind\s+of|somewhat|more\s+or\s+less)\b",
    r"\b(?:some\s+things|various\s+things|etc\.?|and\s+so\s+on)\b",
    r"\b(?:it\s+depends\s+widely|hard\s+to\s+say|not\s+really\s+clear)\b",
]

CODE_INTENT_PATTERNS = [
    r"\b(?:code|script|snippet|implementation|function|program|write|create|build)\b",
    r"\b(?:in\s+python|using\s+python|bash|sql|dockerfile)\b",
]


class ResponseStyleAnalyzer:
    """Analyzes assistant responses for structural styles, clarity, and ambiguity indicators."""

    @staticmethod
    def analyze(prompt: str, response: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        words = response.split()
        word_count = len(words)

        # 1. Structural features
        code_blocks = re.findall(r"```([a-zA-Z0-9_-]*)\n([\s\S]*?)```", response)
        has_code = len(code_blocks) > 0
        code_block_count = len(code_blocks)

        # Count lines of code vs total lines
        code_lines = sum(len(cb[1].strip().splitlines()) for cb in code_blocks)
        total_lines = len(response.splitlines()) or 1
        code_ratio = round(code_lines / float(total_lines), 2) if total_lines > 0 else 0.0

        # Numbered step lists (e.g. "1. ", "Step 1:", "### Step 1")
        step_matches = re.findall(r"(?:^|\n)(?:\d+\.|\bstep\s+\d+:?|###\s+step\s+\d+)", response_lower)
        is_step_by_step = len(step_matches) >= 2

        # Bullet lists (e.g. "- ", "* ")
        bullet_matches = re.findall(r"(?:^|\n)\s*[\-\*]\s+", response)
        has_bullets = len(bullet_matches) >= 2

        # Headings (e.g. "##", "###")
        has_headings = bool(re.search(r"(?:^|\n)#{1,4}\s+", response))

        # Brevity
        is_concise = word_count < 120
        is_dense = word_count > 350 and not has_bullets and not has_code

        # 2. Ambiguity & Confusion Features
        hedge_count = 0
        for pat in AMBIGUITY_PATTERNS:
            hedge_count += len(re.findall(pat, response_lower))

        hedge_density = round((hedge_count / max(1, word_count)) * 100.0, 2)
        is_ambiguous = hedge_density > 1.5 or (hedge_count >= 3 and word_count < 150)

        # Prompt wanted code, but response gave none
        wanted_code = any(re.search(p, prompt_lower) for p in CODE_INTENT_PATTERNS)
        missing_code = wanted_code and not has_code

        # Wall of text without breaks
        is_wall_of_text = word_count > 250 and not has_code and not has_bullets and not is_step_by_step and not has_headings

        # 3. Determine primary style key and display name
        style_tags: List[str] = []
        if has_code:
            style_tags.append("executable_code")
        if is_step_by_step:
            style_tags.append("step_by_step")
        if has_bullets:
            style_tags.append("bulleted_list")
        if is_concise:
            style_tags.append("concise_summary")
        if has_headings:
            style_tags.append("modular_headings")

        if is_step_by_step and has_code:
            primary_style = "step_by_step_code"
            style_name = "Step-by-Step Code Walkthrough"
            description = "Structured walkthrough with progressive steps and executable code"
        elif has_code and (code_ratio > 0.4 or word_count < 100):
            primary_style = "code_centric"
            style_name = "Direct Code Implementation"
            description = "Clean, actionable code snippets with minimal commentary"
        elif is_step_by_step:
            primary_style = "step_by_step_guide"
            style_name = "Step-by-Step Conceptual Guide"
            description = "Clear sequential breakdown of a process or concept"
        elif has_bullets and is_concise:
            primary_style = "concise_bullets"
            style_name = "Concise Bullet Points"
            description = "Quick-to-read, high-density bulleted summaries"
        elif has_headings or has_bullets:
            primary_style = "structured_overview"
            style_name = "Structured Overview"
            description = "Well-organized reference with headings and structured sections"
        elif is_concise:
            primary_style = "concise_answer"
            style_name = "Concise Direct Answer"
            description = "Brief, focused single-paragraph answer"
        else:
            primary_style = "deep_dive_conceptual"
            style_name = "Deep Dive Explanation"
            description = "Comprehensive explanatory prose and conceptual deep-dive"

        # 4. Ambiguity / Confusion tags
        confusion_triggers: List[str] = []
        if is_ambiguous:
            confusion_triggers.append("ambiguous_or_vague")
        if is_wall_of_text:
            confusion_triggers.append("dense_wall_of_text")
        if missing_code:
            confusion_triggers.append("missing_concrete_code")

        return {
            "primary_style": primary_style,
            "style_name": style_name,
            "description": description,
            "style_tags": style_tags,
            "has_code": has_code,
            "is_step_by_step": is_step_by_step,
            "is_concise": is_concise,
            "word_count": word_count,
            "hedge_density": hedge_density,
            "confusion_triggers": confusion_triggers,
        }


response_style_analyzer = ResponseStyleAnalyzer()
