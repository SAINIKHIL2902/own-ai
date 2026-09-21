import hashlib
import re
from typing import List, Optional
from app.storage.models import ContextRecord, utc_now

CONTEXT_PATTERNS = [
    # "I'm currently learning Kafka for my MLOps project"
    re.compile(r"(?:i am|i'm)?\s*(?:currently\s+)?learning\s+([a-zA-Z0-9_\-\.\+]+)\s+for\s+(?:my\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\.|$|,)", re.IGNORECASE),
    # "working on an ecommerce backend" / "building a chatbot"
    re.compile(r"(?:working on|building|developing|creating)\s+(?:a|an|my)?\s+([a-zA-Z0-9_\-\.\s]+?)(?:\s+using|\s+with|\.|$|,)", re.IGNORECASE),
]


def extract_contexts(prompt: str) -> List[ContextRecord]:
    """Extracts active project and learning contexts with timestamps."""
    records: List[ContextRecord] = []
    now = utc_now()

    # Pattern 1: Learning X for Y
    m1 = CONTEXT_PATTERNS[0].search(prompt)
    if m1:
        topic = m1.group(1).strip().capitalize()
        project = m1.group(2).strip()
        ctx_id = f"ctx_{hashlib.md5(f'{topic}_{project}'.encode()).hexdigest()[:8]}"
        records.append(
            ContextRecord(
                id=ctx_id,
                topic=topic,
                activity="learning",
                context=project,
                status="current",
                created_at=now,
                updated_at=now,
            )
        )

    # Pattern 2: Building X
    m2 = CONTEXT_PATTERNS[1].search(prompt)
    if m2:
        project = m2.group(1).strip()
        if len(project) > 3 and not any(r.context == project for r in records):
            ctx_id = f"ctx_{hashlib.md5(project.encode()).hexdigest()[:8]}"
            records.append(
                ContextRecord(
                    id=ctx_id,
                    topic="Engineering",
                    activity="building",
                    context=project,
                    status="current",
                    created_at=now,
                    updated_at=now,
                )
            )

    return records
