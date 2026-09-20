from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field


def generate_event_id(prefix: str = "evt") -> str:
    return f"{prefix}_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:8]}"


class InteractionEvent(BaseModel):
    event_id: str = Field(default_factory=generate_event_id)
    event_type: str = "chat_interaction"
    event_schema_version: int = 2
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str = "local-user"
    conversation_id: str
    message_id: str
    prompt: str
    response: str
    model: str
    provider: str = "local"
    routing: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


class FeedbackEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: generate_event_id("fb"))
    event_type: str = "user_feedback"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str = "local-user"
    conversation_id: Optional[str] = None
    message_id: str
    feedback_type: str  # thumbs_up, thumbs_down, positive, negative, correction
    feedback_value: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
