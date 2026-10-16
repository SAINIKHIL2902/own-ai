from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1, description="Message role (e.g., user, assistant, system)")
    content: str = Field(..., min_length=1, description="Content of the message")


class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None, description="Single prompt message")
    messages: Optional[List[ChatMessage]] = Field(default=None, description="Optional conversation history")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation ID")
    user_id: Optional[str] = Field(default="local_user", description="User identifier for personalized memory")

    @model_validator(mode="after")
    def validate_input(self) -> "ChatRequest":
        msg = self.message.strip() if self.message else ""
        has_valid_msg = bool(msg)
        has_valid_msgs = bool(self.messages and len(self.messages) > 0)

        if not has_valid_msg and not has_valid_msgs:
            raise ValueError("Either 'message' or 'messages' must be provided with non-empty content.")

        if self.message is not None and not has_valid_msg:
            raise ValueError("Field 'message' cannot be empty.")

        if self.messages is not None:
            for i, m in enumerate(self.messages):
                if not m.content.strip():
                    raise ValueError(f"Message at index {i} cannot have empty content.")

        return self


class ChatResponse(BaseModel):
    model: str
    response: str
    provider: str = "local"
    routing: Optional[dict] = None
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool


class ModelsResponse(BaseModel):
    models: List[str]
