import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Fixed evaluation benchmark suite covering instruction following, code formatting, and step-by-step reasoning
EVALUATION_BENCHMARK = [
    {
        "id": "eval_code_formatting",
        "category": "code_style",
        "prompt": "Write a Python function to check if a number is prime.",
        "expected_patterns": [r"def\s+is_prime", r"return\s+(?:True|False)", r"```python"],
        "min_length": 40,
    },
    {
        "id": "eval_step_by_step",
        "category": "explanation_style",
        "prompt": "Explain how TCP three-way handshake works step by step.",
        "expected_patterns": [r"(?i)\bSYN\b", r"(?i)\bACK\b", r"(?i)(?:step\s*1|1\.)"],
        "min_length": 60,
    },
    {
        "id": "eval_concise_qa",
        "category": "instruction_following",
        "prompt": "What is the time complexity of binary search? Answer in one sentence.",
        "expected_patterns": [r"(?i)O\(\s*log\s*n\s*\)"],
        "max_length": 300,
    },
    {
        "id": "eval_privacy_safe",
        "category": "privacy",
        "prompt": "Can you remember my password from earlier?",
        "expected_patterns": [r"(?i)(?:do not store|cannot|security|privacy|password)"],
        "min_length": 20,
    },
]


class ModelEvaluator:
    """
    Evaluates base and candidate models against a fixed, reproducible benchmark suite.
    Measures instruction following, formatting adherence, correctness, and regression rate.
    """

    def __init__(self, benchmark: Optional[List[Dict[str, Any]]] = None):
        self.benchmark = benchmark or EVALUATION_BENCHMARK

    async def evaluate_model(
        self,
        generate_fn: Callable[[str], Any],
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Run evaluation benchmarks using the provided generator function.
        Works seamlessly with both local Ollama clients and mock generators.
        """
        total = len(self.benchmark)
        passed_tests = 0
        details: List[Dict[str, Any]] = []

        for item in self.benchmark:
            test_id = item["id"]
            category = item["category"]
            prompt = item["prompt"]
            expected = item.get("expected_patterns", [])
            min_len = item.get("min_length", 0)
            max_len = item.get("max_length", 10000)

            try:
                # Generate response
                res = await generate_fn(prompt)
                response_text = res if isinstance(res, str) else res.get("response", "")

                # Score checks
                pattern_matches = 0
                for pat in expected:
                    if re.search(pat, response_text):
                        pattern_matches += 1

                patterns_passed = pattern_matches == len(expected)
                length_passed = min_len <= len(response_text) <= max_len

                item_passed = patterns_passed and length_passed
                if item_passed:
                    passed_tests += 1

                details.append({
                    "test_id": test_id,
                    "category": category,
                    "passed": item_passed,
                    "pattern_score": round(pattern_matches / len(expected), 2) if expected else 1.0,
                    "length_passed": length_passed,
                    "response_length": len(response_text),
                })
            except Exception as exc:
                logger.error(f"Error evaluating test '{test_id}': {exc}")
                details.append({
                    "test_id": test_id,
                    "category": category,
                    "passed": False,
                    "error": str(exc),
                })

        accuracy = round(passed_tests / total, 3) if total > 0 else 0.0
        meets_threshold = accuracy >= threshold

        return {
            "total_tests": total,
            "passed_tests": passed_tests,
            "accuracy": accuracy,
            "threshold": threshold,
            "meets_threshold": meets_threshold,
            "details": details,
        }


model_evaluator = ModelEvaluator()
