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
from app.memory.models import (
    MemoryCreateRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)
from app.memory.service import memory_service
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
        if not model:
            try:
                from app.training.deployment import model_deployment_manager
                model = model_deployment_manager.get_current_model()
            except Exception:
                model = settings.OLLAMA_MODEL
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


# ==============================================================================
# PHASE 4 ENDPOINTS: MEMORY MANAGEMENT & PERSONALIZATION
# ==============================================================================
@app.get("/memory", response_model=List[MemoryResponse], summary="List Memories")
async def get_memories(
    user_id: str = "local_user",
    type: Optional[str] = None,
    status_filter: Optional[str] = None,
):
    """List stored memories filtered by user, type, or lifecycle status."""
    memories = memory_service.list_memories(user_id=user_id, memory_type=type, status=status_filter)
    return [MemoryResponse(**m.to_dict()) for m in memories]


@app.get("/memory/{memory_id}", response_model=MemoryResponse, summary="Get Single Memory")
async def get_memory_by_id(memory_id: str):
    """Retrieve a specific memory by its unique ID."""
    memory = memory_service.get_memory(memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )
    return MemoryResponse(**memory.to_dict())


@app.post("/memory", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED, summary="Create Memory")
async def create_memory(request: MemoryCreateRequest, user_id: str = "local_user"):
    """Manually add a memory preference, interest, goal, fact, or instruction."""
    try:
        created = memory_service.create_memory_manual(request, user_id=user_id)
        return MemoryResponse(**created.to_dict())
    except Exception as exc:
        logger.error(f"Failed to create memory: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@app.patch("/memory/{memory_id}", response_model=MemoryResponse, summary="Update Memory")
async def update_memory(memory_id: str, request: MemoryUpdateRequest):
    """Update content, type, status, importance, or confidence of an existing memory."""
    updated = memory_service.update_memory(memory_id, request)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )
    return MemoryResponse(**updated.to_dict())


@app.delete("/memory/{memory_id}", summary="Delete or Archive Memory")
async def delete_memory(memory_id: str, soft: bool = True):
    """Delete a memory (soft-archives by default, or hard deletes if soft=False)."""
    success = memory_service.delete_memory(memory_id, soft=soft)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )
    return {"status": "ok", "deleted": True, "memory_id": memory_id, "soft": soft}


# ==============================================================================
# PHASE 5 ENDPOINTS: BEHAVIOR-BASED LOCAL MODEL TRAINING & DEPLOYMENT
# ==============================================================================
from app.training.models import (
    ActivateRunRequest,
    ColabExportRequest,
    ColabImportRequest,
    ConsentRequest,
    DatasetBuildRequest,
    EvaluateRunRequest,
    PackageRunRequest,
    RollbackRunRequest,
    TrainingConfig,
    TrainingRunCreateRequest,
)
from app.training.service import training_service


@app.get("/api/training/status", summary="Get Model Training Pipeline Status")
async def get_training_status():
    """Returns overview of active model, consent status, and recent training runs."""
    return training_service.get_status()


@app.get("/api/training/consent", summary="Get User Training Consent")
async def get_training_consent(user_id: str = "local_user"):
    """Check whether the user has opted in to behavior-based model training."""
    consent = training_service.get_consent(user_id=user_id)
    return consent.to_dict()


@app.post("/api/training/consent", summary="Update User Training Consent")
async def set_training_consent(req: ConsentRequest):
    """Opt-in or opt-out of training data extraction and model fine-tuning."""
    consent = training_service.set_consent(user_id=req.user_id, opted_in=req.opted_in)
    return consent.to_dict()


@app.post("/api/training/dataset", summary="Build Curated Training Dataset")
async def build_training_dataset(req: DatasetBuildRequest):
    """
    Extract sanitized, consent-aware user interactions and build a reproducible
    JSONL train/val dataset.
    """
    cfg = TrainingConfig()
    if req.base_model:
        cfg.base_model = req.base_model
    if req.train_val_split is not None:
        cfg.train_val_split = req.train_val_split

    try:
        manifest = training_service.prepare_dataset(config=cfg, user_id=req.user_id)
        return manifest
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Dataset build failed: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/api/training/runs", summary="Create Training Run & Cloud Bundle")
async def create_training_run(req: TrainingRunCreateRequest):
    """
    Register a training run and generate a self-contained Cloud GPU bundle (.zip)
    ready for execution on Google Colab (Free T4 GPU) or Kaggle.
    """
    cfg = TrainingConfig()
    if req.base_model:
        cfg.base_model = req.base_model
    if req.num_epochs is not None:
        cfg.num_epochs = req.num_epochs
    if req.batch_size is not None:
        cfg.batch_size = req.batch_size
    if req.learning_rate is not None:
        cfg.learning_rate = req.learning_rate
    if req.candidate_tag:
        cfg.candidate_tag = req.candidate_tag

    result = training_service.initiate_training_run(
        dataset_version=req.dataset_version,
        config=cfg,
        user_id=req.user_id,
    )
    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))
    return result


@app.get("/api/training/runs", summary="List All Training Runs")
async def list_training_runs(limit: int = 20):
    """List recent training runs and their current statuses."""
    runs = training_service.repo.list_runs(limit=limit)
    return [r.to_dict() for r in runs]


@app.get("/api/training/runs/{run_id}", summary="Get Training Run Details")
async def get_training_run(run_id: str):
    """Retrieve details for a specific training run."""
    run = training_service.repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found.")
    return run.to_dict()


@app.post("/api/training/evaluate", summary="Evaluate Candidate Model")
async def evaluate_candidate(req: EvaluateRunRequest):
    """Run fixed benchmark suite on a trained candidate adapter."""
    try:
        res = await training_service.evaluate_candidate(
            run_id=req.run_id,
            adapter_path=req.adapter_path,
            mock_eval=req.mock_eval,
        )
        return res
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Candidate evaluation failed: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/api/training/package", summary="Package Candidate into Ollama Modelfile")
async def package_candidate(req: PackageRunRequest):
    """Generate Modelfile and register candidate with Ollama."""
    res = await training_service.package_candidate(run_id=req.run_id)
    if not res.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res


@app.post("/api/training/activate", summary="Activate Candidate Model")
async def activate_candidate(req: ActivateRunRequest):
    """
    Perform pre-flight sanity checks and atomically activate the fine-tuned
    candidate model for local routing and inference.
    """
    res = await training_service.activate_model(run_id=req.run_id)
    if not res.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res


@app.post("/api/training/rollback", summary="Rollback Active Model to Baseline")
async def rollback_model(req: Optional[RollbackRunRequest] = None):
    """Immediately rollback active routing model to base model or previous stable version."""
    res = training_service.rollback_model()
    return res


# ==============================================================================
# GOOGLE COLAB & GOOGLE DRIVE END-TO-END WORKFLOW ENDPOINTS
# ==============================================================================
@app.post("/api/training/colab/export", summary="Gather Data & Export Unified Unit to Google Colab / Drive")
async def export_colab_unit(req: ColabExportRequest):
    """
    Gathers all interactions, behavioral preferences, and memories into one cohesive unit,
    generates the Colab training notebook, and syncs directly to Google Drive.
    """
    cfg = TrainingConfig()
    if req.base_model:
        cfg.base_model = req.base_model
    if req.num_epochs is not None:
        cfg.num_epochs = req.num_epochs
    if req.batch_size is not None:
        cfg.batch_size = req.batch_size

    try:
        res = training_service.export_colab_unit(
            config=cfg,
            user_id=req.user_id,
            sync_to_gdrive=req.sync_to_gdrive,
        )
        return res
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Colab export failed: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/api/training/colab/import", summary="Download and Deploy Versioned Model from Google Drive")
async def import_colab_model(req: ColabImportRequest):
    """
    Retrieves the latest or specified versioned model checkpoint from Google Drive,
    evaluates against the benchmark suite, packages into Ollama, and activates.
    """
    try:
        res = await training_service.import_colab_model(
            version_id=req.version_id,
            auto_evaluate=req.auto_evaluate,
            auto_activate=req.auto_activate,
            mock_eval=req.mock_eval,
        )
        return res
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Colab model import failed: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get("/api/training/colab/models", summary="List Available Versioned Models in Cloud & Local")
async def list_colab_models():
    """Lists all versioned model checkpoints in Google Drive and local SQLite registry."""
    return training_service.list_colab_models()


@app.get("/api/training/colab/notebook/{version_id}", summary="Download Colab Training Notebook")
async def get_colab_notebook(version_id: str):
    """Download the generated Jupyter/Colab notebook (.ipynb) for a specific training unit."""
    nb_file = PROJECT_ROOT / "data" / "training" / "units" / version_id / "colab_train.ipynb"
    if not nb_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Notebook for '{version_id}' not found.")
    return FileResponse(
        path=str(nb_file),
        filename=f"colab_train_{version_id}.ipynb",
        media_type="application/x-ipynb+json",
    )



