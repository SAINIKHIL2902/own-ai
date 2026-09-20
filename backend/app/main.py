import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import settings
from app.ollama.client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from app.ollama.models import ChatRequest, ChatResponse, HealthResponse, ModelsResponse
from app.chat.service import ChatService

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("own-ai")

app = FastAPI(
    title="Personal Local AI - Phase 1",
    description="Minimal, strictly local offline AI application communicating directly with Ollama.",
    version="1.0.0",
)

# Base paths
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

ollama_client = OllamaClient()
chat_service = ChatService(ollama_client=ollama_client)


@app.get("/", response_class=FileResponse, summary="Serve Chat UI")
async def get_index():
    """Serve the local single-page Chat UI directly."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found.",
        )
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def get_health():
    """Verify backend health and local Ollama daemon connectivity."""
    is_connected = await ollama_client.check_health()
    return HealthResponse(
        status="ok" if is_connected else "degraded",
        ollama_connected=is_connected,
    )


@app.get("/models", response_model=ModelsResponse, summary="List Local Models")
async def get_models():
    """Retrieve list of locally installed Ollama models."""
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
    """Execute real local inference via Ollama and return model response."""
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
    except Exception as exc:
        logger.error(f"Unhandled server error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing your request.",
        )
