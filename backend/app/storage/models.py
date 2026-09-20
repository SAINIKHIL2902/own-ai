from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==============================================================================
# RAW DATA MODELS
# ==============================================================================
@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class MessageRecord:
    message_id: str
    conversation_id: str
    role: str
    content: str
    model: str
    provider: str = "local"
    user_id: str = "local_user"
    latency_ms: float = 0.0
    id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self):
        if not self.id:
            self.id = self.message_id


@dataclass
class RawEventRecord:
    event_id: str
    event_type: str
    timestamp: str
    user_id: str
    conversation_id: str
    message_id: str
    payload_json: str
    processed: int = 0


@dataclass
class FeedbackRecord:
    feedback_id: str
    message_id: str
    conversation_id: Optional[str]
    feedback_type: str  # thumbs_up, thumbs_down, positive, negative, correction
    feedback_value: Optional[str] = None  # e.g. "positive", "negative", "ambiguous"
    user_id: str = "local_user"
    rating: Optional[int] = None
    comment: Optional[str] = None
    metadata_json: Optional[str] = None  # JSON string of detected style tags & metrics
    id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self):
        if not self.id:
            self.id = self.feedback_id


# ==============================================================================
# DERIVED PROFILE MODELS
# ==============================================================================
@dataclass
class ResponseStyleRecord:
    style_key: str
    style_name: str
    description: str
    liked_count: int = 0
    disliked_count: int = 0
    affinity_score: float = 0.5
    updated_at: str = field(default_factory=utc_now)


@dataclass
class InterestRecord:
    topic: str
    score: float
    interaction_count: int
    recent_interactions: int
    first_seen: str
    last_seen: str
    updated_at: str = field(default_factory=utc_now)


@dataclass
class PreferenceRecord:
    preference: str
    value: str  # e.g. "true", "false", "step_by_step"
    confidence: float
    evidence_count: int
    updated_at: str = field(default_factory=utc_now)


@dataclass
class ContextRecord:
    id: str
    topic: str
    activity: str
    context: str
    status: str = "current"  # "current", "expired"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class BehaviorStatRecord:
    stat_key: str
    stat_value: str
    updated_at: str = field(default_factory=utc_now)
