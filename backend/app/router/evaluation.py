import asyncio
import sys
from pathlib import Path

# Add backend and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from evaluation.runner import BenchmarkRunner


def main():
    runner = BenchmarkRunner()
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
