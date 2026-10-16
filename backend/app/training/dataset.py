from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import PROJECT_ROOT
from app.training.extractor import ConsentAwareExtractor, consent_aware_extractor
from app.training.models import TrainingConfig, TrainingExample

logger = logging.getLogger(__name__)

TRAINING_DIR = PROJECT_ROOT / "data" / "training"


# High-quality behavioral style demonstrations aligned with Phase 2 preferences
CURATED_BEHAVIORAL_EXAMPLES = [
    {
        "messages": [
            {"role": "user", "content": "How do I implement exponential backoff in Python?"},
            {
                "role": "assistant",
                "content": "```python\nimport time\nimport random\n\ndef retry_with_backoff(func, max_retries=5, base_delay=1.0):\n    for attempt in range(max_retries):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_retries - 1:\n                raise\n            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)\n            time.sleep(delay)\n```\nKey steps:\n1. Calculates exponential multiplier `2 ** attempt`.\n2. Adds random jitter to prevent thundering herd problems.",
            },
        ],
        "metadata": {"style": "code_centric", "category": "utility"},
    },
    {
        "messages": [
            {"role": "user", "content": "Explain database indexing step by step."},
            {
                "role": "assistant",
                "content": "### Step-by-Step Overview of Database Indexing\n\n1. **Without an Index (Full Table Scan)**: The database engine inspects every row sequentially (O(N) time complexity).\n2. **B-Tree Structure**: The index creates a balanced multi-way tree sorted by the indexed column.\n3. **Search Traversal**: Lookups traverse tree nodes in O(log N) time to locate the row pointer.\n4. **Trade-offs**: Read queries speed up significantly, but write operations (INSERT/UPDATE) have slight overhead to maintain the tree.",
            },
        ],
        "metadata": {"style": "step_by_step_guide", "category": "conceptual"},
    },
]


class DatasetBuilder:
    """
    Builds, deduplicates, validates, and partitions JSONL datasets for local/cloud fine-tuning.
    """

    def __init__(
        self,
        extractor: Optional[ConsentAwareExtractor] = None,
        output_base_dir: Optional[Path] = None,
    ):
        self.extractor = extractor or consent_aware_extractor
        self.output_base_dir = output_base_dir or TRAINING_DIR

    def build_dataset(
        self,
        config: TrainingConfig,
        user_id: str = "local_user",
        include_curated_style_demos: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds a versioned dataset from eligible interactions and approved style demonstrations.
        Returns a dictionary containing dataset paths, statistics, and provenance.
        """
        # 1. Extract consent-verified user interactions
        report = self.extractor.extract_user_examples(user_id=user_id)
        if not report.consent_opted_in:
            raise PermissionError(f"User '{user_id}' has not granted consent for model training.")
        all_examples: List[Dict[str, Any]] = [ex.to_dict() for ex in report.examples]

        # 2. Optionally append curated behavioral demonstrations
        if include_curated_style_demos:
            all_examples.extend(CURATED_BEHAVIORAL_EXAMPLES)

        # 3. Deduplicate by content hash
        seen_hashes = set()
        unique_examples: List[Dict[str, Any]] = []

        for ex in all_examples:
            msgs = ex.get("messages", [])
            content_key = ":::".join(m.get("content", "").strip() for m in msgs)
            h = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_examples.append(ex)

        if len(unique_examples) < 2:
            return {
                "success": False,
                "reason": "INSUFFICIENT_EXAMPLES",
                "message": f"Only {len(unique_examples)} eligible examples found. Minimum required is 2.",
                "user_accepted": report.accepted_count,
                "rejected_counts": report.rejected_counts,
            }

        # 4. Deterministic train/validation split
        rng = random.Random(config.random_seed)
        shuffled = list(unique_examples)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * config.train_val_split)
        split_idx = max(1, min(split_idx, len(shuffled) - 1))  # Ensure at least 1 in val

        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]

        # 5. Version and save to disk
        version_id = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        target_dir = self.output_base_dir / version_id
        target_dir.mkdir(parents=True, exist_ok=True)

        train_path = target_dir / "train.jsonl"
        val_path = target_dir / "val.jsonl"
        manifest_path = target_dir / "manifest.json"

        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps({"messages": item["messages"]}) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps({"messages": item["messages"]}) + "\n")

        manifest = {
            "version": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "total_examples": len(unique_examples),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "user_extracted_count": report.accepted_count,
            "curated_demos_included": include_curated_style_demos,
            "config": config.model_dump(),
            "paths": {
                "train": str(train_path),
                "val": str(val_path),
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Successfully built dataset '{version_id}' at {target_dir}: "
            f"{len(train_data)} train, {len(val_data)} val."
        )

        return {
            "success": True,
            "version": version_id,
            "dataset_dir": str(target_dir),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "manifest_path": str(manifest_path),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "total_count": len(unique_examples),
            "total_examples": len(unique_examples),
            "user_accepted": report.accepted_count,
            "rejected_counts": report.rejected_counts,
        }

    def validate_dataset_files(self, dataset_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """Validates that train.jsonl and val.jsonl exist and conform to schema."""
        train_p = dataset_dir / "train.jsonl"
        val_p = dataset_dir / "val.jsonl"

        if not train_p.exists() or not val_p.exists():
            return False, "Missing train.jsonl or val.jsonl file", {}

        stats = {"train_rows": 0, "val_rows": 0}
        for path, key in [(train_p, "train_rows"), (val_p, "val_rows")]:
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        return False, f"Invalid JSON on line {idx+1} in {path.name}", {}

                    if "messages" not in record or not isinstance(record["messages"], list):
                        return False, f"Missing 'messages' list on line {idx+1} in {path.name}", {}

                    for m in record["messages"]:
                        if "role" not in m or "content" not in m:
                            return False, f"Invalid message structure on line {idx+1} in {path.name}", {}
                    stats[key] += 1

        return True, "Dataset is valid and ready for training.", stats


dataset_builder = DatasetBuilder()
