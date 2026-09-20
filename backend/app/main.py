import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.behavior.analyzer import behavior_analyzer, run_behavior_consumer
from app.chat.service import ChatService
from app.config.settings import PROJECT_ROOT, settings
from app.events.producer import event_producer
from app.events.schemas import FeedbackEvent
from app.ollama.client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from app.llm.gemini_client import GeminiAuthError, GeminiConnectionError, GeminiError
from app.ollama.models import ChatRequest, ChatResponse, HealthResponse, ModelsResponse
from app.storage.repositories import profile_repo, raw_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("own-ai")

# Base paths
FRONTEND_DIR = PROJECT_ROOT / "frontend"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
SPOOL_DIR = PROJECT_ROOT / "data" / "events"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize Kafka producer and background consumer."""
    logger.info("Initializing Phase 2 event infrastructure...")
    await event_producer.start()

    # Launch background consumer task
    consumer_task = asyncio.create_task(run_behavior_consumer())

    yield

    logger.info("Shutting down Phase 2 event infrastructure...")
    consumer_task.cancel()
    await event_producer.stop()


app = FastAPI(
    title="Personal Local AI - Phase 2",
    description="Event-Driven User Behavior, Interest & Preference Modeling with local Ollama, Kafka, and SQLite.",
    version="2.0.0",
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ollama_client = OllamaClient()


class MainOllamaAdapter:
    async def generate(self, messages, model=None):
        target_model = model or settings.OLLAMA_MODEL
        res = await ollama_client.generate_chat(messages=messages, model=target_model)
        return {
            "model": res.get("model", target_model),
            "response": res.get("response", ""),
            "provider": "local",
        }


chat_service = ChatService(ollama_client=MainOllamaAdapter(), producer=event_producer, repository=raw_repo)


# ==============================================================================
# PHASE 1 ENDPOINTS (Preserved Untouched)
# ==============================================================================
@app.get("/", response_class=FileResponse, summary="Serve Chat UI")
async def get_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found.",
        )
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def get_health():
    is_connected = await ollama_client.check_health()
    return HealthResponse(
        status="ok" if is_connected else "degraded",
        ollama_connected=is_connected,
    )


@app.get("/models", response_model=ModelsResponse, summary="List Local Models")
async def get_models():
    try:
        models = await ollama_client.list_models()
        return ModelsResponse(models=models)
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Error fetching models: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models from local Ollama daemon.",
        )


@app.post("/chat", response_model=ChatResponse, summary="Send Chat Message")
async def post_chat(request: ChatRequest):
    try:
        return await chat_service.chat(request)
    except OllamaConnectionError as exc:
        logger.error(f"Ollama connection failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except OllamaModelNotFoundError as exc:
        logger.error(f"Ollama model not found: {exc}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except OllamaClientError as exc:
        logger.error(f"Ollama generation failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except (GeminiAuthError, GeminiConnectionError, GeminiError) as exc:
        logger.error(f"Gemini generation failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External model provider failure: {exc}",
        )
    except Exception as exc:
        logger.error(f"Unhandled server error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing your request.",
        )


# ==============================================================================
# PHASE 2 ENDPOINTS: FEEDBACK, USER PROFILE, AND ANALYTICS
# ==============================================================================
class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    feedback_type: str = Field(..., description="'thumbs_up', 'thumbs_down', 'positive', 'negative', or 'correction'")
    feedback_value: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/feedback", summary="Submit Message Feedback")
async def post_feedback(request: FeedbackRequest):
    """Record user feedback directly linked to assistant response message_id for behavioral profiling."""
    fb_event = FeedbackEvent(
        message_id=request.message_id,
        conversation_id=request.conversation_id,
        feedback_type=request.feedback_type,
        rating=request.rating,
        comment=request.comment,
    )
    event_dict = fb_event.model_dump()
    if request.feedback_value:
        event_dict["feedback_value"] = request.feedback_value

    await event_producer.publish_feedback(fb_event)

    # Immediately analyze style and update behavioral profile
    behavior_analyzer.process_event_dict(event_dict)

    return {"status": "ok", "event_id": fb_event.event_id, "message_id": request.message_id}


@app.get("/profile", summary="Get Derived User Profile")
async def get_user_profile():
    """Retrieve the derived user profile (interests, preferences, response_styles, contexts, and stats)."""
    behavior_analyzer.drain_unprocessed()
    return profile_repo.get_full_profile()


@app.get("/analytics/events", summary="Analytics Event Status")
async def get_event_stats():
    """Get count of raw events and offline spooled event backlog."""
    spooled_count = len(list(SPOOL_DIR.glob("*.json"))) if SPOOL_DIR.exists() else 0
    total_raw = raw_repo.count_raw_events()
    unprocessed_count = len(raw_repo.get_unprocessed_raw_events(limit=1000))
    return {
        "raw_events_stored": total_raw,
        "unprocessed_events_pending": unprocessed_count,
        "spooled_events_pending": spooled_count,
        "kafka_connected": event_producer._is_connected,
        "schema_version": 1,
    }


@app.post("/behavior/drain", summary="Drain Spooled Offline Events")
async def post_drain_spool():
    """Manually drain and process any events saved during offline operation."""
    drained = behavior_analyzer.drain_unprocessed()
    return {"status": "ok", "events_drained": drained}


@app.post("/analytics/export", summary="Export Versioned Training Dataset")
async def post_export_dataset():
    """Curates and exports a versioned training dataset JSONL file from raw history."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_id = f"dataset_v1_{int(datetime.now(timezone.utc).timestamp())}"
    file_path = EXPORTS_DIR / f"{export_id}.jsonl"

    profile = profile_repo.get_full_profile()
    count = 0

    with open(file_path, "w", encoding="utf-8") as f:
        # Include header metadata record
        meta_record = {
            "type": "metadata",
            "schema_version": 1,
            "export_id": export_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interests_snapshot": profile.get("interests", []),
            "preferences_snapshot": profile.get("preferences", []),
            "response_styles_snapshot": profile.get("response_styles", []),
        }
        f.write(json.dumps(meta_record) + "\n")

        # Include raw interaction records enriched with user feedback
        with raw_repo.manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM raw_events ORDER BY timestamp ASC")
            for row in cursor.fetchall():
                try:
                    payload = json.loads(row["payload_json"])
                    if payload.get("event_type") == "chat_interaction":
                        msg_id = payload.get("message_id")
                        fb = raw_repo.get_feedback_by_message_id(msg_id) if msg_id else None
                        dataset_entry = {
                            "type": "interaction",
                            "event_id": payload.get("event_id"),
                            "message_id": msg_id,
                            "timestamp": payload.get("timestamp"),
                            "prompt": payload.get("prompt"),
                            "response": payload.get("response"),
                            "model": payload.get("model"),
                            "user_feedback": {
                                "feedback_type": fb["feedback_type"],
                                "feedback_value": fb.get("feedback_value"),
                                "metadata": json.loads(fb["metadata_json"]) if fb and fb.get("metadata_json") else None,
                            } if fb else None,
                        }
                        f.write(json.dumps(dataset_entry) + "\n")
                        count += 1
                except Exception:
                    continue

    return {
        "status": "ok",
        "export_id": export_id,
        "records_exported": count,
        "file_path": str(file_path),
    }
