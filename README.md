# Personal Local AI — Phase 1

Welcome to **Phase 1** of your Personal Local AI project. This project enables you to run a modern Large Language Model (LLM) completely locally on your Mac, accessible through a clean, ChatGPT-style chat interface in your browser, with **zero external cloud API calls** and **complete offline capability**.

---

## 1. What Phase 1 Is

Phase 1 is the minimal, reliable, testable foundation of your personal AI stack. Its single objective is:

> **"I can open my local application, type a message, and a local LLM running on my Mac generates the response even when the internet is completely disconnected."**

Key characteristics:
- **100% Local Inference**: Runs directly on Apple Silicon via Ollama.
- **Strictly Offline**: No network packets leave your machine during generation.
- **Single-Origin Application**: FastAPI serves both the API endpoints (`/chat`, `/health`, `/models`) and the front-end user interface at `http://127.0.0.1:8000/`.
- **Completely Self-Contained**: All Python dependencies live in `backend/venv` and Ollama model weights reside inside `models/`.

---

## 2. What Phase 1 Deliberately Does NOT Contain

To maintain stability and keep Phase 1 beginner-friendly, the following features are intentionally out of scope:
- **No Cloud APIs**: Zero calls to OpenAI, Google Gemini, Anthropic, or Azure.
- **No Mock or Fake Responses**: All AI responses come from genuine model inference.
- **No Vector Databases or RAG**: No embeddings, Chroma, Pinecone, or document ingestion.
- **No Persistent Memory**: Conversations exist only in browser memory for the active session.
- **No Complex Telemetry**: No tracking, metrics collectors, or external loggers.
- **No Multi-Model Routing / Task Classification**: Strictly one configured model (`llama3.2:1b`).
- **No Streaming (SSE)**: Synchronous non-streamed inference (`stream: false`) for maximum initial reliability.
- **No Heavy Frontend Frameworks**: Built using vanilla HTML5, CSS3, and JavaScript — no React, Next.js, NPM, or CDN dependencies.

---

## 3. Architecture Diagram

```
User Browser
   │
   │  Loads http://127.0.0.1:8000/
   ▼
Simple Chat UI (`frontend/index.html`)
   │
   │  HTTP POST /chat  {"messages": [...]}
   ▼
FastAPI Application (`backend/app/main.py`)
   │
   ▼
ChatService (`backend/app/chat/service.py`)
   │
   ▼
OllamaClient (`backend/app/ollama/client.py`)
   │
   │  HTTP POST http://127.0.0.1:11434/api/chat (stream: false, timeout: 120s)
   ▼
Ollama Daemon (`ollama serve`)
   │  Storage: OLLAMA_MODELS=own-ai/models
   ▼
Local LLM (`llama3.2:1b`)
   │
   │  (Tokens generated locally via Apple Silicon Metal/GPU)
   ▼
Response returned back up the stack
```

---

## 4. Project Repository Tree

```
own-ai/
├── README.md                           # Comprehensive documentation & guide
├── .gitignore                          # Ignores venv, .env, models, and cache
├── .env.example                        # Template for environment configuration
├── .env                                # Local configuration (not committed)
├── models/                             # Local model storage directory (OLLAMA_MODELS)
├── backend/
│   ├── venv/                           # Isolated Python virtual environment
│   ├── requirements.txt                # Pinned backend dependencies
│   ├── app/
│   │   ├── __init__.py                 # Package marker
│   │   ├── main.py                     # FastAPI application & route definitions
│   │   ├── config/
│   │   │   ├── __init__.py             # Package marker
│   │   │   └── settings.py             # Pydantic Settings reading .env
│   │   ├── ollama/
│   │   │   ├── __init__.py             # Package marker
│   │   │   ├── client.py               # Asynchronous HTTP client for Ollama
│   │   │   └── models.py               # Pydantic request/response schemas
│   │   └── chat/
│   │       ├── __init__.py             # Package marker
│   │       └── service.py              # Thin chat orchestration layer
│   └── tests/
│       ├── __init__.py                 # Package marker
│       ├── test_config.py              # Unit tests for settings & overrides
│       ├── test_chat.py                # Unit tests for ChatService & validation
│       ├── test_api.py                 # API tests for /health, /models, /chat
│       └── integration/
│           ├── __init__.py             # Package marker
│           └── test_ollama_live.py     # Live integration test against real Ollama
└── frontend/
    ├── index.html                      # Single-page ChatGPT-like layout
    ├── style.css                       # Modern dark-mode UI stylesheet
    └── app.js                          # Client-side interaction & API caller
```

---

## 5. File Explanations (Beginner-Friendly)

| File | Purpose | Why It Exists |
|---|---|---|
| `backend/app/config/settings.py` | Reads `.env` using Pydantic | Centralizes URLs, model names, and timeouts into one type-safe object. |
| `backend/app/ollama/models.py` | Request/Response schemas | Enforces data formats, prevents empty messages, and serializes responses. |
| `backend/app/ollama/client.py` | Low-level Ollama communicator | Handles HTTP calls (`/api/chat`, `/api/tags`), timeouts, and converts network failures into clear errors. |
| `backend/app/chat/service.py` | Chat business logic | Thin orchestrator that formats conversation messages and requests inference from the client. |
| `backend/app/main.py` | FastAPI web server | Defines endpoints (`GET /`, `GET /health`, `GET /models`, `POST /chat`) and maps error codes (503, 404, 422, 500). |
| `frontend/index.html` | Chat UI layout | Clean markup for header, system status card, message list, and text input. |
| `frontend/style.css` | UI styling | Dark mode ChatGPT aesthetic with responsive bubbles and status indicators. |
| `frontend/app.js` | UI logic | Captures Enter key, displays loading bubbles, sends messages to `/chat`, and keeps in-memory history. |
| `backend/tests/` | Automated tests | Ensures configuration, request validation, API routes, and live model inference work properly. |

---

## 6. Prerequisites

- **macOS**: Apple Silicon (M1/M2/M3/M4) recommended.
- **Python 3.10+**: Available in terminal (`python3 --version`).
- **Homebrew**: Package manager for macOS (`brew --version`).

---

## 7. Configuration: Application vs. Daemon (Important Distinction!)

It is essential to understand the difference between:

1. **Application Configuration (`.env`)**:
   Controls how the **FastAPI backend** communicates with Ollama:
   ```env
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2:1b
   OLLAMA_TIMEOUT=120
   ```
   *Note: Putting `OLLAMA_MODELS` in `.env` does NOT configure Ollama itself.*

2. **Ollama Daemon Configuration (`OLLAMA_MODELS`)**:
   Controls where the **Ollama server process** stores model weights (so they do not fill up your user root `~/.ollama`):
   ```bash
   export OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models"
   ```

---

## 8. Step-by-Step Setup & Installation

### Step 1: Create the Local Python Virtual Environment
Navigate to `backend/` and initialize an isolated virtual environment:
```bash
cd "own-ai/backend"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Install Ollama
If not already installed:
```bash
brew install ollama
```

### Step 3: Start the Ollama Daemon with Isolated Storage
Launch Ollama in a dedicated terminal (or as a background service) with the storage path set to `own-ai/models`:
```bash
export OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models"
ollama serve
```

### Step 4: Download the Local Model
In another terminal, download `llama3.2:1b` (requires internet access once):
```bash
export OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models"
ollama pull llama3.2:1b
```

Verify the model is installed:
```bash
ollama list
```

Verify the model weights are stored in `own-ai/models`:
```bash
ls -lh "own-ai/models/manifests/registry.ollama.ai/library/llama3.2"
```

---

## 9. Starting the Application

Activate your virtual environment and start the FastAPI server:
```bash
cd "own-ai/backend"
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 10. Opening the Chat UI

Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
You will see:
- **System Status**: Shows "Ollama Online" with a green indicator.
- **Active Model**: Displays `llama3.2:1b`.
- **Chat Workspace**: Allows you to type prompts, press **Enter** to send, or press **Shift+Enter** for a new line.

---

## 11. API Reference & curl Examples

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "ollama_connected": true
}
```

### 2. List Installed Models
```bash
curl -X GET http://127.0.0.1:8000/models
```
**Response:**
```json
{
  "models": [
    "llama3.2:1b"
  ]
}
```

### 3. Chat Inference (Single Prompt)
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'
```
**Response:**
```json
{
  "model": "llama3.2:1b",
  "response": "Python is a high-level, interpreted programming language known for its readability and simplicity..."
}
```

### 4. Chat Inference with Conversation History
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language."},
      {"role": "user", "content": "What is it mainly used for?"}
    ]
  }'
```

---

## 12. Testing

### Run Unit and API Tests (No Ollama required)
```bash
cd "own-ai/backend"
venv/bin/pytest tests/test_config.py tests/test_chat.py tests/test_api.py -v
```

### Run Live Ollama Integration Tests (Requires running Ollama daemon)
```bash
cd "own-ai/backend"
venv/bin/pytest tests/integration/test_ollama_live.py -v -s
```

---

## 13. Complete Offline Verification Procedure

To definitively prove Phase 1 works without any internet connection:

1. **Ensure prerequisites are met**:
   - Model `llama3.2:1b` is downloaded.
   - Ollama daemon is running (`ollama serve`).
   - FastAPI backend is running on `http://127.0.0.1:8000`.

2. **Disconnect All Internet Connections**:
   - Turn off Wi-Fi on your Mac (`networksetup -setairportpower en0 off` or via the Control Center).
   - Disconnect any Ethernet cable.
   - Disconnect any VPN or hotspot.

3. **Verify CLI Inference Offline**:
   ```bash
   ollama run llama3.2:1b "Explain gravity in one short sentence."
   ```
   *Expected: A real answer generated with zero internet access.*

4. **Verify Backend Health Offline**:
   ```bash
   curl -s http://127.0.0.1:8000/health
   ```
   *Expected: `{"status": "ok", "ollama_connected": true}`*

5. **Verify Chat API Offline**:
   ```bash
   curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is 2 + 2?"}'
   ```
   *Expected: Model returns `{"model": "llama3.2:1b", "response": "2 + 2 = 4..."}`.*

6. **Verify Browser UI Offline**:
   - Refresh `http://127.0.0.1:8000/`.
   - Send the message: *"Explain what an offline LLM is."*
   - Observe the response generated directly on your Mac.

---

## 14. Troubleshooting Common Errors

### Error: "Ollama is not running at http://127.0.0.1:11434" (HTTP 503)
- **Cause**: The Ollama background process is not running.
- **Fix**: Open a terminal and run `OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models" ollama serve`.

### Error: "Model 'llama3.2:1b' is not installed" (HTTP 404)
- **Cause**: The model was pulled with a different `OLLAMA_MODELS` path or has not been pulled yet.
- **Fix**: Run `OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models" ollama pull llama3.2:1b`.

### Error: "422 Unprocessable Entity"
- **Cause**: An empty message was submitted (`{"message": ""}`).
- **Fix**: Send at least one non-empty string character.

---

## 15. Phase 1 Definition of Done Checklist

- [x] Ollama installed on Mac.
- [x] Ollama daemon runs locally on port 11434.
- [x] Local LLM (`llama3.2:1b`) downloaded.
- [x] Model storage location verified in `own-ai/models`.
- [x] `ollama list` displays `llama3.2:1b`.
- [x] `ollama run` produces valid local inference.
- [x] Model functions with internet completely disconnected.
- [x] Python virtual environment isolated in `backend/venv`.
- [x] FastAPI application starts and serves `GET /`.
- [x] `GET /health` reports accurate connection status.
- [x] `GET /models` lists installed models.
- [x] `POST /chat` executes genuine inference.
- [x] ChatGPT-like UI functions without any CDN or external assets.
- [x] Unit, API, and Live integration tests pass.
