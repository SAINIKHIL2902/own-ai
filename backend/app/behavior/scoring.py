from datetime import datetime, timezone
import math
from typing import Optional

from app.config.settings import settings


def calculate_recency_decay(
    last_seen_iso: str,
    decay_rate: Optional[float] = None,
    now: Optional[datetime] = None,
) -> float:
    """Calculate exponential time-decay factor based on days elapsed since last seen."""
    rate = decay_rate if decay_rate is not None else settings.INTEREST_DECAY_RATE
    current_time = now or datetime.now(timezone.utc)

    try:
        last_seen_dt = datetime.fromisoformat(last_seen_iso)
        if last_seen_dt.tzinfo is None:
            last_seen_dt = last_seen_dt.replace(tzinfo=timezone.utc)
        elapsed_days = (current_time - last_seen_dt).total_seconds() / 86400.0
        # Decay factor: (rate ^ elapsed_days) clamped between 0.1 and 1.0
        return max(0.1, min(1.0, math.pow(rate, max(0.0, elapsed_days))))
    except Exception:
        return 1.0


def calculate_interaction_weight(
    prompt_len: int,
    feedback_type: Optional[str] = None,
) -> float:
    """Calculate weighted contribution for a single interaction."""
    # Frequency component (base increment)
    freq_contrib = settings.INTEREST_FREQUENCY_WEIGHT * 0.2

    # Recency component (immediate interaction gives max recency boost)
    recency_contrib = settings.INTEREST_RECENCY_WEIGHT * 0.3

    # Engagement component (longer prompts indicate higher user investment)
    depth_factor = min(1.0, prompt_len / 100.0)
    engagement_contrib = settings.INTEREST_ENGAGEMENT_WEIGHT * depth_factor

    # Feedback component
    feedback_factor = 0.0
    if feedback_type == "positive":
        feedback_factor = 1.0
    elif feedback_type == "negative":
        feedback_factor = -0.5
    feedback_contrib = settings.INTEREST_FEEDBACK_WEIGHT * feedback_factor

    total_weight = freq_contrib + recency_contrib + engagement_contrib + feedback_contrib
    return max(0.05, total_weight)


def update_interest_score(
    current_score: float,
    last_seen_iso: str,
    prompt_len: int = 50,
    feedback_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> float:
    """Apply decay to current score and add the new interaction weight, normalized between 0.0 and 1.0."""
    decay = calculate_recency_decay(last_seen_iso, now=now)
    interaction_weight = calculate_interaction_weight(prompt_len, feedback_type)

    decayed_score = current_score * decay
    # Diminishing returns scaling: asymptotically approaches 1.0
    new_score = decayed_score + (1.0 - decayed_score) * (interaction_weight * 0.5)

    return round(max(0.01, min(1.0, new_score)), 4)
