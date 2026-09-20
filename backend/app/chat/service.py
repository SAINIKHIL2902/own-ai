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
from app.ollama.client import OllamaClient
from app.ollama.models import ChatRequest, ChatResponse
from app.privacy.scanner import PrivacyScanner, privacy_scanner
from app.router.router import ModelRouter, RoutingDecision, model_router
from app.storage.repositories import RawRepository, raw_repo

logger = logging.getLogger(__name__)


class ChatService:
    """
    Orchestrates chat requests: analyzes prompts, routes intelligently between
    Local Ollama and Google Gemini, sanitizes external requests, handles capability-aware
    fallbacks, records history in SQLite, and emits Kafka interaction events.
    """

    def __init__(
        self,
        ollama_client: Optional[Any] = None,
        gemini_client: Optional[BaseLLMClient] = None,
        router: Optional[ModelRouter] = None,
        privacy_scanner_inst: Optional[PrivacyScanner] = None,
        producer: Optional[EventProducer] = None,
        repository: Optional[RawRepository] = None,
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

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat request using intelligent evidence-based model routing."""
        messages_payload: List[Dict[str, str]] = []

        if request.messages:
            for msg in request.messages:
                messages_payload.append({"role": msg.role, "content": msg.content})

        if request.message:
            messages_payload.append({"role": "user", "content": request.message})

        user_messages = [m["content"] for m in messages_payload if m.get("role") == "user"]
        latest_prompt = user_messages[-1] if user_messages else (messages_payload[-1]["content"] if messages_payload else "")
        history = messages_payload[:-1] if len(messages_payload) > 1 else None

        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        asst_msg_id = f"msg_{uuid.uuid4().hex[:8]}"

        # 1. Route request: determine whether Local is sufficiently capable or if Gemini is required
        decision: RoutingDecision = self.router.route(latest_prompt, history)
        selected_provider = decision.selected_provider
        target_model = decision.selected_model

        actual_provider = selected_provider
        actual_model = target_model
        response_text = ""
        start_time = time.perf_counter()

        # 2. Execute selected model (Single-model execution in production)
        if selected_provider == "local":
            logger.info(f"Routing to LOCAL Ollama model '{target_model}' (suitability={decision.local_suitability})")
            result = await self.ollama.generate(messages=messages_payload, model=target_model)
            response_text = result.get("response", "")
            actual_model = result.get("model", target_model)
            actual_provider = "local"
        else:
            # Selected provider is GEMINI
            # Step 2a: Enforce Privacy Boundary -> Sanitize prompt before transmitting externally
            sanitized_messages, scan_result = self.privacy.sanitize_messages(messages_payload)
            if scan_result.contains_sensitive_data:
                logger.info(f"Sanitized {len(scan_result.detected_types)} sensitive items before sending to Gemini")

            try:
                logger.info(f"Routing to GEMINI model '{target_model}'")
                # GeminiClient receives strictly sanitized messages
                result = await self.gemini.generate(messages=sanitized_messages, model=target_model)
                response_text = result.get("response", "")
                actual_model = result.get("model", target_model)
                actual_provider = "gemini"
            except Exception as gemini_err:
                logger.warning(f"Gemini generation failed: {gemini_err}")

                # Step 2b: Capability-aware fallback
                can_fallback = (
                    settings.ROUTER_FALLBACK_TO_LOCAL
                    and decision.local_suitability >= (settings.LOCAL_MIN_SUITABILITY * 0.70)
                )

                if can_fallback:
                    logger.warning("Failing over to LOCAL Ollama model due to Gemini error (local model deemed capable).")
                    fallback_res = await self.ollama.generate(messages=messages_payload, model=settings.OLLAMA_MODEL)
                    response_text = fallback_res.get("response", "")
                    actual_model = fallback_res.get("model", settings.OLLAMA_MODEL)
                    actual_provider = "local"  # MUST reflect reality
                    decision.reason += f" [Fell back to local Ollama due to Gemini API failure: {gemini_err}]"
                else:
                    # Do not silently or falsely fallback if local model is incapable
                    logger.error("Gemini failed and local model is incapable of answering this request safely.")
                    raise

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 3. Persist conversation history to SQLite
        # Preserves original user prompt locally for internal learning while recording actual responding provider
        try:
            self.repo.save_message(
                msg_id=user_msg_id,
                conversation_id=conv_id,
                role="user",
                content=latest_prompt,
                model=actual_model,
                latency_ms=0.0,
                user_id="local_user",
                provider=actual_provider,
            )
            self.repo.save_message(
                msg_id=asst_msg_id,
                conversation_id=conv_id,
                role="assistant",
                content=response_text,
                model=actual_model,
                latency_ms=latency_ms,
                user_id="local_user",
                provider=actual_provider,
            )
        except Exception as exc:
            logger.warning(f"Failed to record messages in raw SQLite: {exc}")

        # 4. Construct and emit version 2 InteractionEvent
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

        return ChatResponse(
            model=actual_model,
            provider=actual_provider,
            response=response_text,
            message_id=asst_msg_id,
            conversation_id=conv_id,
            routing=decision.to_dict(),
        )


chat_service = ChatService()
