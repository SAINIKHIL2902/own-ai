import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional

from app.config.settings import settings
from app.events.producer import EventProducer, event_producer
from app.events.schemas import InteractionEvent
from app.ollama.client import OllamaClient
from app.ollama.models import ChatRequest, ChatResponse
from app.storage.repositories import RawRepository, raw_repo

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        producer: Optional[EventProducer] = None,
        repository: Optional[RawRepository] = None,
    ):
        self.client = ollama_client or OllamaClient()
        self.producer = producer or event_producer
        self.repo = repository or raw_repo

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process validated chat request, invoke Ollama, and asynchronously record events."""
        messages_payload: List[Dict[str, str]] = []

        if request.messages:
            for msg in request.messages:
                messages_payload.append({"role": msg.role, "content": msg.content})

        if request.message:
            messages_payload.append({"role": "user", "content": request.message})

        user_messages = [m["content"] for m in messages_payload if m.get("role") == "user"]
        latest_prompt = user_messages[-1] if user_messages else (messages_payload[-1]["content"] if messages_payload else "")
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        asst_msg_id = f"msg_{uuid.uuid4().hex[:8]}"

        logger.info(f"Dispatching chat request to Ollama model '{settings.OLLAMA_MODEL}' ({len(messages_payload)} messages)")

        start_time = time.perf_counter()
        result = await self.client.generate_chat(
            messages=messages_payload,
            model=settings.OLLAMA_MODEL,
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        model_name = result.get("model", settings.OLLAMA_MODEL)
        response_text = result.get("response", "")

        # Persist raw conversation history to SQLite
        try:
            self.repo.save_message(
                msg_id=user_msg_id,
                conversation_id=conv_id,
                role="user",
                content=latest_prompt,
                model=model_name,
                latency_ms=0.0,
                user_id="local_user",
            )
            self.repo.save_message(
                msg_id=asst_msg_id,
                conversation_id=conv_id,
                role="assistant",
                content=response_text,
                model=model_name,
                latency_ms=latency_ms,
                user_id="local_user",
            )
        except Exception as exc:
            logger.warning(f"Failed to record messages in raw SQLite: {exc}")

        # Construct and asynchronously emit interaction event (non-blocking)
        event = InteractionEvent(
            conversation_id=conv_id,
            message_id=asst_msg_id,
            prompt=latest_prompt,
            response=response_text,
            model=model_name,
            latency_ms=latency_ms,
        )

        # Trigger background event publish so synchronous response is never delayed
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.producer.publish_interaction(event))
        except RuntimeError:
            asyncio.run(self.producer.publish_interaction(event))

        return ChatResponse(
            model=model_name,
            response=response_text,
            message_id=asst_msg_id,
            conversation_id=conv_id,
        )
