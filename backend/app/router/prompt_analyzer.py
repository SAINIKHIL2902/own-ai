from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


@dataclass
class PromptAnalysis:
    """Structured result of prompt analysis."""
    task_type: str
    difficulty: str  # "easy", "medium", "hard"
    required_capabilities: Dict[str, float]
    complexity_score: float  # 0.0 - 1.0
    reasoning_requirement: float  # 0.0 - 1.0
    context_requirement: float  # 0.0 - 1.0
    structured_output_required: bool
    prompt_length_chars: int = 0


class PromptAnalyzer:
    """
    Deterministic, zero-latency analyzer that identifies required capabilities,
    task type, complexity, reasoning demands, and difficulty tier without recursive LLM calls.
    """

    TASK_PATTERNS = {
        "coding": re.compile(
            r"\b(write\s+(?:a\s+)?(?:\w+\s+)?(?:code|function|script|algorithm|program)|implement|def\s+\w+|class\s+\w+|algorithm\b|code\s+(?:in|for|to)|snippet|reverse a (?:string|list|array|linked list)|binary search)\b",
            re.IGNORECASE,
        ),
        "debugging": re.compile(
            r"\b(debug|fix|error|exception|traceback|bug|syntaxerror|issue|fails?|failing|resolve error)\b",
            re.IGNORECASE,
        ),
        "mathematics": re.compile(
            r"\b(math|calculate|integral|derivative|equation|algebra|probability|theorem|matrix|variance|statistics)\b|[\+\-\*\/\^\=]{2,}",
            re.IGNORECASE,
        ),
        "reasoning": re.compile(
            r"\b(why|prove|proofs?|deduce|logic|inference|evaluate tradeoffs|trade-?offs?|compare and contrast|decision matrix|causal|conclude|vs|versus|failure modes?)\b",
            re.IGNORECASE,
        ),
        "analysis": re.compile(
            r"\b(analyze|breakdown|evaluate|audit|investigate|benchmark|performance bottleneck|tradeoffs?)\b",
            re.IGNORECASE,
        ),
        "summarization": re.compile(
            r"\b(summarize|summary|tldr|brief overview|key takeaways|bullet points summary|synopsis)\b",
            re.IGNORECASE,
        ),
        "rewriting": re.compile(
            r"\b(rewrite|rephrase|paraphrase|proofread|edit|polish|reformat)\b",
            re.IGNORECASE,
        ),
        "translation": re.compile(
            r"\b(translate|translation|in spanish|in french|in german|in hindi|in japanese|in chinese)\b",
            re.IGNORECASE,
        ),
        "explanation": re.compile(
            r"\b(explain|how does|what is|what are|describe|walkthrough|teach me|introduction to|overview)\b",
            re.IGNORECASE,
        ),
        "structured_output": re.compile(
            r"\b(json|yaml|xml|csv|table|schema|markdown table|return as json|format as json)\b",
            re.IGNORECASE,
        ),
    }

    STRUCTURED_OUTPUT_INDICATORS = re.compile(
        r"(?i)\b(json format|valid json|strictly json|return (?:a |as )?json|table format|markdown table|csv format)\b"
    )

    COMPLEXITY_TRIGGERS = re.compile(
        r"(?i)\b(edge cases?|distributed system|distributed|concurrency|race condition|architecture design|high availability|fault tolerance|trade-?offs?|inconsistencies|deep dive|step-by-step mathematical proof|proofs?|byzantine|consensus|paxos|raft|replication|failure modes?|network partitions?)\b"
    )

    def analyze(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> PromptAnalysis:
        cleaned = prompt.strip()
        length = len(cleaned)

        # 1. Detect task type
        task_type = "question_answering"
        task_scores: Dict[str, int] = {}
        for ttype, pattern in self.TASK_PATTERNS.items():
            matches = len(pattern.findall(cleaned))
            if matches > 0:
                task_scores[ttype] = matches

        if task_scores:
            task_type = max(task_scores.items(), key=lambda x: x[1])[0]

        # Prioritize explicit explanation openings ("What is X?", "Explain Y")
        if re.match(r"(?i)^(what\s+is|what\s+are|how\s+does|explain\b|tell\s+me\s+about|describe\b)", cleaned):
            if not re.search(r"(?i)\b(write\s+(?:a\s+)?code|write\s+(?:a\s+)?function|implement)\b", cleaned):
                task_type = "explanation"

        # 2. Check structured output requirement
        structured_output_required = bool(self.STRUCTURED_OUTPUT_INDICATORS.search(cleaned))
        if structured_output_required and task_type in ["question_answering", "explanation"]:
            task_type = "structured_output"

        # 3. Context requirement
        # Evaluate current prompt length and recent conversation turns (up to 4 turns)
        total_chars = length
        if conversation_history:
            recent_history = conversation_history[-4:]
            total_chars += sum(len(m.get("content", "")) for m in recent_history)
        context_requirement = min(round(total_chars / 6000.0, 3), 1.0)

        # Only classify as long_context if the prompt itself is a long text (>3500 chars)
        if length > 3500 and task_type in ["question_answering", "explanation"]:
            task_type = "long_context"

        # 4. Reasoning requirement
        reasoning_hits = len(self.TASK_PATTERNS["reasoning"].findall(cleaned))
        analysis_hits = len(self.TASK_PATTERNS["analysis"].findall(cleaned))
        complexity_hits = len(self.COMPLEXITY_TRIGGERS.findall(cleaned))
        raw_reasoning = (reasoning_hits * 0.25) + (analysis_hits * 0.2) + (complexity_hits * 0.35)
        reasoning_requirement = min(round(max(raw_reasoning, 0.1), 2), 1.0)

        # 5. Complexity score
        raw_complexity = (
            (length / 2000.0) * 0.25
            + (complexity_hits * 0.25)
            + (reasoning_requirement * 0.3)
            + (0.2 if structured_output_required else 0.0)
        )
        complexity_score = min(round(raw_complexity, 2), 1.0)

        # 6. Difficulty tier: easy, medium, hard
        # Inherent difficulty of the prompt based on complexity and reasoning demands
        if complexity_score >= 0.65 or reasoning_requirement >= 0.70 or length > 4000:
            difficulty = "hard"
        elif complexity_score >= 0.35 or reasoning_requirement >= 0.40:
            difficulty = "medium"
        else:
            difficulty = "easy"

        # 7. Required capabilities mapping (normalized 0.0 - 1.0)
        req_caps: Dict[str, float] = {
            "instruction_following": 0.70,
        }

        if task_type == "coding":
            req_caps["coding"] = 0.90 if difficulty == "easy" else (0.95 if difficulty == "medium" else 1.0)
            req_caps["instruction_following"] = 0.80
            if "debug" in cleaned.lower():
                req_caps["debugging"] = 0.85
        elif task_type == "debugging":
            req_caps["debugging"] = 0.95
            req_caps["coding"] = 0.85
            req_caps["multi_step_reasoning"] = 0.65
        elif task_type == "mathematics":
            req_caps["mathematical_reasoning"] = 0.90
            req_caps["multi_step_reasoning"] = 0.75
        elif task_type == "reasoning" or task_type == "analysis":
            req_caps["complex_reasoning"] = 0.90 if difficulty == "hard" else 0.75
            req_caps["multi_step_reasoning"] = 0.80
            req_caps["general_knowledge"] = 0.70
        elif task_type == "explanation":
            req_caps["explanation"] = 0.90
            req_caps["general_knowledge"] = 0.80
            req_caps["instruction_following"] = 0.70
        elif task_type == "summarization":
            req_caps["summarization"] = 0.90
            req_caps["instruction_following"] = 0.80
        elif task_type == "rewriting":
            req_caps["rewriting"] = 0.90
            req_caps["instruction_following"] = 0.80
        elif task_type == "translation":
            req_caps["translation"] = 0.95
        elif task_type == "long_context":
            req_caps["long_context"] = 1.0
            req_caps["complex_reasoning"] = 0.70
        elif task_type == "structured_output":
            req_caps["structured_output"] = 0.95
            req_caps["instruction_following"] = 0.90
        else:
            req_caps["simple_question_answering"] = 0.85
            req_caps["general_knowledge"] = 0.80

        if structured_output_required:
            req_caps["structured_output"] = 0.90
        if context_requirement >= 0.6:
            req_caps["long_context"] = context_requirement

        return PromptAnalysis(
            task_type=task_type,
            difficulty=difficulty,
            required_capabilities=req_caps,
            complexity_score=complexity_score,
            reasoning_requirement=reasoning_requirement,
            context_requirement=context_requirement,
            structured_output_required=structured_output_required,
            prompt_length_chars=length,
        )


prompt_analyzer = PromptAnalyzer()
