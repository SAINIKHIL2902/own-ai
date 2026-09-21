from datetime import datetime, timedelta, timezone
from app.behavior.scoring import (
    calculate_interaction_weight,
    calculate_recency_decay,
    update_interest_score,
)


def test_recency_decay_immediate():
    now = datetime.now(timezone.utc)
    decay = calculate_recency_decay(now.isoformat(), decay_rate=0.95, now=now)
    assert decay == 1.0


def test_recency_decay_after_days():
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=10)
    decay = calculate_recency_decay(past.isoformat(), decay_rate=0.95, now=now)
    # (0.95)^10 ~= 0.5987
    assert 0.55 < decay < 0.65


def test_interaction_weight_scaling():
    short_w = calculate_interaction_weight(prompt_len=10)
    long_w = calculate_interaction_weight(prompt_len=200)
    assert long_w > short_w

    pos_w = calculate_interaction_weight(prompt_len=50, feedback_type="positive")
    neg_w = calculate_interaction_weight(prompt_len=50, feedback_type="negative")
    assert pos_w > neg_w


def test_update_interest_score_growth():
    now = datetime.now(timezone.utc)
    score1 = update_interest_score(current_score=0.2, last_seen_iso=now.isoformat(), prompt_len=50, now=now)
    assert score1 > 0.2

    # Subsequent interaction increases score further
    score2 = update_interest_score(current_score=score1, last_seen_iso=now.isoformat(), prompt_len=100, now=now)
    assert score2 > score1
    assert score2 <= 1.0
