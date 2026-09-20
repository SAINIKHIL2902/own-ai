import logging
from typing import Dict, List, Optional
from app.config.settings import settings
from app.ollama.client import OllamaClient
from app.ollama.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process validated chat request and invoke Ollama client."""
        messages_payload: List[Dict[str, str]] = []

        if request.messages:
            for msg in request.messages:
                messages_payload.append({"role": msg.role, "content": msg.content})

        if request.message:
            messages_payload.append({"role": "user", "content": request.message})

        logger.info(f"Dispatching chat request to Ollama model '{settings.OLLAMA_MODEL}' ({len(messages_payload)} messages)")

        result = await self.client.generate_chat(
            messages=messages_payload,
            model=settings.OLLAMA_MODEL,
        )

        return ChatResponse(
            model=result["model"],
            response=result["response"],
        )
