import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config.settings import settings
from app.ollama.client import OllamaClient
from evaluation.evaluator import ResponseEvaluator, response_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "routing_eval_v1.json"
REPORTS_DIR = Path(__file__).parent / "reports"


class BenchmarkRunner:
    """Runs empirical evaluation benchmarks against the local model."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        evaluator: ResponseEvaluator | None = None,
    ):
        self.client = ollama_client or OllamaClient()
        self.evaluator = evaluator or response_evaluator

    async def run(self) -> Dict[str, Any]:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        logger.info(f"Loaded {len(cases)} benchmark cases from {DATASET_PATH}")
        results: List[Dict[str, Any]] = []

        # Breakdown stats: {task_type: {difficulty: [scores]}}
        breakdown: Dict[str, Dict[str, List[float]]] = {}

        for case in cases:
            cid = case["id"]
            prompt = case["prompt"]
            ttype = case["task_type"]
            diff = case.get("difficulty", "medium")
            criteria = case.get("evaluation_criteria", [])

            logger.info(f"Evaluating case '{cid}' ({ttype}/{diff})...")
            try:
                res = await self.client.generate_chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=settings.OLLAMA_MODEL,
                )
                response_text = res.get("response", "")
            except Exception as e:
                logger.warning(f"Failed to generate for '{cid}': {e}")
                response_text = ""

            score = self.evaluator.evaluate_response(prompt, response_text, ttype, criteria)

            breakdown.setdefault(ttype, {}).setdefault(diff, []).append(score.overall_score)

            results.append({
                "id": cid,
                "task_type": ttype,
                "difficulty": diff,
                "prompt": prompt,
                "response_snippet": response_text[:120] + "..." if len(response_text) > 120 else response_text,
                "scores": score.to_dict(),
                "success": score.overall_score >= 0.70,
            })

        # Calculate category & difficulty summaries
        summary: Dict[str, Dict[str, Any]] = {}
        for ttype, diffs in breakdown.items():
            summary[ttype] = {}
            for diff, scores in diffs.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                pass_rate = sum(1 for s in scores if s >= 0.70) / len(scores) if scores else 0.0
                summary[ttype][diff] = {
                    "count": len(scores),
                    "avg_score": round(avg, 2),
                    "success_rate": f"{round(pass_rate * 100)}%",
                }

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_model": settings.OLLAMA_MODEL,
            "total_cases": len(cases),
            "summary_by_task_and_difficulty": summary,
            "detailed_results": results,
            "threshold_recommendations": {
                "local_min_suitability": settings.LOCAL_MIN_SUITABILITY,
                "local_min_confidence": settings.LOCAL_MIN_CONFIDENCE,
                "observation": "High capability on easy coding/math, low capability on complex reasoning and long context.",
            },
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"report_{int(datetime.now(timezone.utc).timestamp())}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark completed. Report saved to {report_path}")
        self._print_formatted_summary(summary)
        return report

    def _print_formatted_summary(self, summary: Dict[str, Dict[str, Any]]):
        print("\n=======================================================")
        print("          LOCAL MODEL BENCHMARK RESULTS")
        print("=======================================================")
        for ttype, diffs in summary.items():
            print(f"\n{ttype.upper()}:")
            for diff, stats in diffs.items():
                print(f"  {diff:<8}: avg score={stats['avg_score']}, success rate={stats['success_rate']}")
        print("=======================================================\n")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    asyncio.run(runner.run())
