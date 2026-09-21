import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from app.config.settings import settings
from app.events.producer import EventProducer, event_producer
from app.events.schemas import InteractionEvent
from app.llm.base import BaseLLMClient
from app.llm.gemini_client import GeminiClient, GeminiError
from app.llm.ollama_adapter import OllamaAdapter
from app.memory.context_builder import ContextBuilder, context_builder
from app.memory.models import MemoryRecord
from app.memory.service import MemoryService, memory_service
from app.ollama.client import OllamaClient
from app.ollama.models import ChatRequest, ChatResponse
from app.privacy.scanner import PrivacyScanner, privacy_scanner
from app.router.router import ModelRouter, RoutingDecision, model_router
from app.storage.repositories import RawRepository, raw_repo

logger = logging.getLogger(__name__)


class ChatService:
    """
    Orchestrates chat requests:
    1. Retrieves relevant user memories (Phase 4).
    2. Builds personalized model context via ContextBuilder while ensuring current user
       instructions strictly override background memories.
    3. Analyzes prompts and routes intelligently between Local Ollama and Google Gemini (Phase 3).
    4. Enforces strict privacy boundary: Gemini receives only privacy-sanitized personalized context.
    5. Fallbacks gracefully if external model fails.
    6. Persists interaction in SQLite and emits Kafka events (Phase 2).
    7. Asynchronously extracts and updates memories without blocking chat response.
    """

    def __init__(
        self,
        ollama_client: Optional[Any] = None,
        gemini_client: Optional[BaseLLMClient] = None,
        router: Optional[ModelRouter] = None,
        privacy_scanner_inst: Optional[PrivacyScanner] = None,
        producer: Optional[EventProducer] = None,
        repository: Optional[RawRepository] = None,
        memory_svc: Optional[MemoryService] = None,
        ctx_builder: Optional[ContextBuilder] = None,
    ):
        if ollama_client is not None:
            if hasattr(ollama_client, "generate"):
                self.ollama = ollama_client
            else:
                self.ollama = OllamaAdapter(client=ollama_client)
        else:
            self.ollama = OllamaAdapter()

        self.gemini = gemini_client or GeminiClient()
        self.router = router or model_router
        self.privacy = privacy_scanner_inst or privacy_scanner
        self.producer = producer or event_producer
        self.repo = repository or raw_repo

        # Initialize MemoryService and ContextBuilder
        if memory_svc is not None:
            self.memory = memory_svc
        elif repository is not None and hasattr(repository, "manager"):
            from app.memory.repository import MemoryRepository
            from app.memory.retriever import MemoryRetriever
            custom_mem_repo = MemoryRepository(manager=repository.manager)
            custom_retriever = MemoryRetriever(repository=custom_mem_repo)
            self.memory = MemoryService(repository=custom_mem_repo, retriever_inst=custom_retriever)
        else:
            self.memory = memory_service

        self.context_builder = ctx_builder or context_builder

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request with personalized context and intelligent model routing."""
        messages_payload: List[Dict[str, str]] = []

        if request.messages:
            for msg in request.messages:
                messages_payload.append({"role": msg.role, "content": msg.content})

        if request.message:
            messages_payload.append({"role": "user", "content": request.message})

        user_messages = [m["content"] for m in messages_payload if m.get("role") == "user"]
        latest_prompt = user_messages[-1] if user_messages else (messages_payload[-1]["content"] if messages_payload else "")
        history = messages_payload[:-1] if len(messages_payload) > 1 else None

        conv_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        asst_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        user_id = getattr(request, "user_id", None) or "local_user"

        # 1. Retrieve relevant memories for the latest prompt (Phase 4)
        relevant_memories: List[MemoryRecord] = []
        try:
            relevant_memories = self.memory.retrieve_relevant_memories(
                prompt=latest_prompt,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to retrieve memories: {exc}")

        # 2. Build personalized message context via ContextBuilder
        # Enforces: current user request always supersedes remembered preferences
        personalized_messages = self.context_builder.personalize_messages(
            messages=messages_payload,
            memories=relevant_memories,
        )

        # 3. Route request: determine whether Local is sufficiently capable or if Gemini is required (Phase 3)
        decision: RoutingDecision = self.router.route(latest_prompt, history)
        selected_provider = decision.selected_provider
        target_model = decision.selected_model

        actual_provider = selected_provider
        actual_model = target_model
        response_text = ""
        start_time = time.perf_counter()

        # 4. Execute selected model
        if selected_provider == "local":
            logger.info(f"Routing to LOCAL Ollama model '{target_model}' (suitability={decision.local_suitability})")
            # Preserve system memory message when slicing sliding window
            if len(personalized_messages) > 8:
                if personalized_messages[0].get("role") == "system":
                    local_payload = [personalized_messages[0]] + personalized_messages[-7:]
                else:
                    local_payload = personalized_messages[-8:]
            else:
                local_payload = personalized_messages

            result = await self.ollama.generate(messages=local_payload, model=target_model)
            response_text = result.get("response", "")
            actual_model = result.get("model", target_model)
            actual_provider = "local"
        else:
            # Selected provider is GEMINI
            # Step 4a: Enforce Privacy Boundary -> Sanitize personalized messages before transmitting externally
            sanitized_messages, scan_result = self.privacy.sanitize_messages(personalized_messages)
            if scan_result.contains_sensitive_data:
                logger.info(f"Sanitized {len(scan_result.detected_types)} sensitive items before sending to Gemini")

            try:
                logger.info(f"Routing to GEMINI model '{target_model}'")
                # GeminiClient receives strictly sanitized personalized context
                result = await self.gemini.generate(messages=sanitized_messages, model=target_model)
                response_text = result.get("response", "")
                actual_model = result.get("model", target_model)
                actual_provider = "gemini"
            except Exception as gemini_err:
                logger.warning(f"Gemini generation failed: {gemini_err}")

                # Step 4b: Capability-aware fallback
                can_fallback = (
                    settings.ROUTER_FALLBACK_TO_LOCAL
                    and decision.local_suitability >= (settings.LOCAL_MIN_SUITABILITY * 0.70)
                )

                if can_fallback:
                    logger.warning("Failing over to LOCAL Ollama model due to Gemini error (local model deemed capable).")
                    fallback_res = await self.ollama.generate(messages=personalized_messages, model=settings.OLLAMA_MODEL)
                    response_text = fallback_res.get("response", "")
                    actual_model = fallback_res.get("model", settings.OLLAMA_MODEL)
                    actual_provider = "local"
                    decision.reason += f" [Fell back to local Ollama due to Gemini API failure: {gemini_err}]"
                else:
                    logger.error("Gemini failed and local model is incapable of answering this request safely.")
                    raise

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 5. Persist conversation history to SQLite
        try:
            self.repo.save_message(
                msg_id=user_msg_id,
                conversation_id=conv_id,
                role="user",
                content=latest_prompt,
                model=actual_model,
                latency_ms=0.0,
                user_id=user_id,
                provider=actual_provider,
            )
            self.repo.save_message(
                msg_id=asst_msg_id,
                conversation_id=conv_id,
                role="assistant",
                content=response_text,
                model=actual_model,
                latency_ms=latency_ms,
                user_id=user_id,
                provider=actual_provider,
            )
        except Exception as exc:
            logger.warning(f"Failed to record messages in raw SQLite: {exc}")

        # 6. Construct and emit version 2 InteractionEvent
        event = InteractionEvent(
            conversation_id=conv_id,
            message_id=asst_msg_id,
            prompt=latest_prompt,
            response=response_text,
            model=actual_model,
            provider=actual_provider,
            routing=decision.to_dict(),
            latency_ms=latency_ms,
        )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.producer.publish_interaction(event))
        except RuntimeError:
            asyncio.run(self.producer.publish_interaction(event))

        # 7. Asynchronously detect and store memory candidates (non-blocking)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_memory_async(latest_prompt, user_id, user_msg_id))
        except RuntimeError:
            pass

        return ChatResponse(
            model=actual_model,
            provider=actual_provider,
            response=response_text,
            message_id=asst_msg_id,
            conversation_id=conv_id,
            routing=decision.to_dict(),
        )

    async def _process_memory_async(self, prompt: str, user_id: str, message_id: str) -> None:
        """Background memory candidate extraction to ensure zero latency impact on chat responses."""
        try:
            await asyncio.to_thread(
                self.memory.process_prompt_for_memories,
                prompt=prompt,
                user_id=user_id,
                message_id=message_id,
            )
        except Exception as exc:
            logger.warning(f"Async memory extraction error: {exc}")


chat_service = ChatService()

