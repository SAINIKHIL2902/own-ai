from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    INTEREST = "interest"
    GOAL = "goal"
    FACT = "fact"
    CONTEXT = "context"
    INSTRUCTION = "instruction"


class MemorySource(str, Enum):
    EXPLICIT_USER = "explicit_user"
    CONVERSATION = "conversation"
    BEHAVIOR = "behavior"
    FEEDBACK = "feedback"
    SYSTEM = "system"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


@dataclass
class MemoryRecord:
    """Internal dataclass representing a row in SQLite memories table."""
    memory_id: str
    user_id: str
    memory_type: str
    content: str
    source_type: str
    source_reference: Optional[str] = None
    confidence: float = 0.95
    importance: float = 0.80
    evidence_count: int = 1
    created_at: str = ""
    updated_at: str = ""
    last_confirmed_at: Optional[str] = None
    expires_at: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryCandidate(BaseModel):
    """Temporary representation of a detected memory candidate before storage."""
    content: str
    memory_type: MemoryType
    source_type: MemorySource = MemorySource.EXPLICIT_USER
    source_reference: Optional[str] = None
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    importance: float = Field(default=0.80, ge=0.0, le=1.0)
    expires_at: Optional[str] = None


class MemoryCreateRequest(BaseModel):
    """User request model for manually creating a memory via API."""
    content: str = Field(..., min_length=2, max_length=1000)
    memory_type: MemoryType = MemoryType.PREFERENCE
    confidence: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    importance: Optional[float] = Field(default=0.80, ge=0.0, le=1.0)
    expires_at: Optional[str] = None


class MemoryUpdateRequest(BaseModel):
    """User request model for updating an existing memory."""
    content: Optional[str] = Field(default=None, min_length=2, max_length=1000)
    memory_type: Optional[MemoryType] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: Optional[MemoryStatus] = None
    expires_at: Optional[str] = None


class MemoryResponse(BaseModel):
    """API response model representing a memory record."""
    memory_id: str
    user_id: str
    memory_type: str
    content: str
    source_type: str
    source_reference: Optional[str] = None
    confidence: float
    importance: float
    evidence_count: int
    created_at: str
    updated_at: str
    last_confirmed_at: Optional[str] = None
    expires_at: Optional[str] = None
    status: str
