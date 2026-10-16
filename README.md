# Personal Local AI — Phase 1, Phase 2, Phase 3 & Phase 4

A clean, reliable, modular, privacy-first **Personal AI Assistant** running on macOS Apple Silicon.

- **Phase 1**: Local LLM inference via Ollama (`llama3.2:1b`), FastAPI backend, and ChatGPT-style web interface.
- **Phase 2**: Event-driven user behavior, interest, preference & context modeling using Apache Kafka, local SQLite storage, and an automated background Behavior Analyzer.
- **Phase 3**: Intelligent capability-based model routing between Local (Ollama) and Cloud (Google Gemini), hard capability boundaries, deterministic privacy sanitization, empirical benchmark evaluation, and visual model attribution.
- **Phase 4**: Local-first Memory & Personalization with deterministic detection, confidence reinforcement, contradiction reconciliation, lifecycle management, relevance scoring, `ContextBuilder` prompt injection (current user instructions always win), privacy-sanitized Gemini context, and an interactive Memory Management UI.


---

## Architecture Overview

```
                          User (Browser UI)
                                 │
                                 ▼
                         Chat UI (ChatGPT-Style)
                                 │
                                 ▼
                              FastAPI
                                 │
                                 ▼
                            ChatService
                                 │
                                 ▼
                          Prompt Analyzer
                                 │
                                 ▼
                       Hard Capability Checks
                         ├── Exceeds context window (>6000 chars)?
                         └── Extreme formal reasoning / proof?
                                 │
                   ┌─────────────┴─────────────┐
                  FAIL                        PASS
                   │                           │
                   ▼                           ▼
             Route: GEMINI            Suitability & Confidence
                                      - Capability match (by task & difficulty)
                                      - Reasoning fit & Context fit
                                               │
                                 ┌─────────────┴─────────────┐
                      Suitability >= 0.70 &        Suitability < 0.70 OR
                      Confidence >= 0.65           Confidence < 0.65
                                 │                 (Weak / Uncertain)
                                 ▼                           │
                            Route: LOCAL                     ▼
                                 │                     Route: GEMINI
                                 │                           │
                                 │                    Privacy Scanner
                                 │                     (Sanitize API keys,
                                 │                      passwords, PII)
                                 │                           │
                                 ▼                           ▼
                           Ollama Client               Gemini Client
                           (llama3.2:1b)            (gemini-3.5-flash-lite)
                                 │                           │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                      Model Attribution
                                      (Ollama vs. Gemini)
                                               │
                                               ▼
                                     Local SQLite Storage
                                       (`data/user.db`)
                                               │
                                               ▼
                                 Kafka Producer (Async Background)
                                               │
                                               ▼
                                  Topic: `interaction.events`
                                               │
                                         ┌─────┴──────┐
                                         │            │ (Fallback: local disk spool)
                                         ▼            ▼
                                    Behavior Analyzer Consumer
                                               │
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                            Interests     Preferences    Behavior Stats
```

> [!IMPORTANT]
> **Core Architectural Principles**:
> 1. **Single-Model Production Path**: Every chat request invokes **exactly one model**. Dual generation is strictly reserved for offline benchmarking and calibration.
> 2. **Local-First & Capability-Aware**: Prefers the local model whenever evidence indicates it is capable. Routes to Gemini when local capabilities are insufficient or routing confidence is low.
> 3. **Deterministic Privacy Boundary**: Any credentials, tokens, connection strings, emails, or phone numbers detected in a prompt are redacted before transmitting to Gemini. Raw user messages are preserved exclusively in local SQLite.
> 4. **Resilient Fallback**: If Gemini fails or times out, the system checks whether the local model is capable ($\text{suitability} \ge 0.50$) before falling back. If incapable, it returns a clean error rather than low-quality output.
> 5. **Zero LLM Blocking**: Real-time chat inference never blocks on Kafka or behavioral analysis.

---

## Directory Structure

```
own-ai/
├── README.md                           # Comprehensive documentation & architecture guide
├── .gitignore                          # Ignores venv/, .env, models/, data/*.db, data/events/*
├── .env.example                        # Configuration template for Phase 1, 2, and 3
├── .env                                # Local environment settings (not committed to git)
├── models/                             # Local model storage directory (OLLAMA_MODELS)
├── data/                               # Local persistent storage
│   ├── user.db                         # SQLite database (WAL mode; Raw + Derived tables)
│   ├── events/                         # Local event spool for offline fallback
│   └── exports/                        # Curated training dataset exports
├── evaluation/                         # Phase 3 Benchmark & Evaluation Framework
│   ├── datasets/
│   │   └── routing_eval_v1.json        # Multi-task, multi-difficulty test benchmark
│   ├── evaluator.py                    # Multi-dimensional response quality evaluator
│   ├── runner.py                       # Automated benchmark runner for local & cloud models
│   └── reports/                        # Timestamped evaluation reports (git-ignored)
├── kafka/                              # Kafka infrastructure
│   ├── docker-compose.yml              # Lightweight single-node KRaft Kafka service
│   └── README.md                       # Kafka setup and lifecycle instructions
├── backend/
│   ├── venv/                           # Python 3.14 virtual environment
│   ├── requirements.txt                # FastAPI, httpx, pydantic, aiokafka, pytest
│   ├── app/
│   │   ├── main.py                     # FastAPI application, lifecycle, static mounts
│   │   ├── config/
│   │   │   └── settings.py             # Strongly-typed settings (Phase 1, 2, and 3)
│   │   ├── llm/                        # Unified LLM client layer (Phase 3)
│   │   │   ├── base.py                 # Abstract BaseLLMClient interface
│   │   │   ├── ollama_adapter.py       # Adapter for local Ollama instance
│   │   │   └── gemini_client.py        # Async client for Google Gemini API
│   │   ├── privacy/                    # Privacy & Data Sanitization (Phase 3)
│   │   │   ├── patterns.py             # Regex detectors for keys, tokens, PII
│   │   │   └── scanner.py              # PrivacyScanner with sanitize_messages()
│   │   ├── router/                     # Intelligent Model Router (Phase 3)
│   │   │   ├── prompt_analyzer.py      # Deterministic zero-latency task classifier
│   │   │   ├── hard_checks.py          # Physical & structural constraint checker
│   │   │   ├── capability.py           # Calibrated capability profiles & overrides
│   │   │   ├── scoring.py              # Normalized suitability & confidence calculation
│   │   │   ├── router.py               # ModelRouter with explainable decisions
│   │   │   └── evaluation.py           # CLI benchmark entrypoint (`python -m app.router.evaluation`)
│   │   ├── ollama/
│   │   │   ├── client.py               # Async HTTP client for Ollama
│   │   │   └── models.py               # ChatRequest & ChatResponse schemas
│   │   ├── chat/
│   │   │   └── service.py              # Chat orchestration, routing, & event emission
│   │   ├── events/
│   │   │   ├── schemas.py              # InteractionEvent (v2) & FeedbackEvent definitions
│   │   │   └── producer.py             # Resilient Kafka producer with local disk spooling
│   │   ├── behavior/
│   │   │   ├── interest.py             # Topic extraction
│   │   │   ├── scoring.py              # Recency decay & weighted interest scoring
│   │   │   ├── preferences.py          # Evidence-based preference detection
│   │   │   ├── context.py              # Project & learning context tracking
│   │   │   └── analyzer.py             # Background consumer & spool drainer
│   │   └── storage/
│   │       ├── database.py             # SQLite WAL-mode connection & migration
│   │       ├── models.py               # Dataclass records for raw & derived entities
│   │       └── repositories.py         # RawRepository and ProfileRepository
│   └── tests/
│       ├── test_config.py
│       ├── test_chat.py
│       ├── test_api.py
│       ├── llm/
│       │   └── test_gemini.py          # Gemini client payload, timeout, & error tests
│       ├── privacy/
│       │   └── test_privacy.py         # Secret redaction & boundary isolation tests
│       ├── router/
│       │   ├── test_prompt_analyzer.py # Task type & difficulty classification tests
│       │   ├── test_hard_checks.py     # Hard constraint boundary tests
│       │   ├── test_scoring.py         # Suitability & confidence calculation tests
│       │   └── test_router.py          # Decision matrix & uncertainty handling tests
│       ├── events/
│       │   ├── test_schemas.py
│       │   └── test_producer.py
│       ├── behavior/
│       │   ├── test_interest.py
│       │   ├── test_scoring.py
│       │   ├── test_preferences.py
│       │   ├── test_context.py
│       │   └── test_style.py
│       ├── storage/
│       │   └── test_repositories.py
│       └── integration/
│           ├── test_ollama_live.py     # Live Ollama inference verification
│           ├── test_pipeline_live.py   # End-to-end Phase 2 pipeline test
│           └── test_phase3_pipeline.py # End-to-end Phase 3 routing & privacy test
└── frontend/
    ├── index.html                      # UI with Attribution Badges & Profile Modal
    ├── style.css                       # Modern dark theme with attribution badge styles
    └── app.js                          # Client-side chat, session continuity, feedback
```

---

## Phase 3: Intelligent Model Routing & Capabilities

### 1. Decision Pipeline
1. **Prompt Analysis**: Classifies task type (`coding`, `explanation`, `mathematics`, `reasoning`, etc.), difficulty tier (`easy`, `medium`, `hard`), reasoning demand, and structured output requirements without recursive LLM calls.
2. **Hard Capability Checks**:
   - **Context Window**: Prompts exceeding `LOCAL_MAX_CONTEXT_CHARS` (6,000 chars) immediately divert to **Gemini**.
   - **Extreme Reasoning**: Formal multi-step deductive proofs immediately divert to **Gemini**.
3. **Suitability Scoring**:
   Calculates normalized score $S \in [0.0, 1.0]$:
   $$S = 0.40 \cdot M_{\text{cap}} + 0.20 \cdot F_{\text{diff}} + 0.15 \cdot F_{\text{reas}} + 0.15 \cdot F_{\text{ctx}} + 0.10 \cdot F_{\text{fmt}}$$
4. **Routing Confidence**:
   Calculates confidence $C \in [0.0, 1.0]$ accounting for task clarity and borderline score zones.
5. **Uncertainty State**:
   - $\text{Suitability} \ge 0.70$ AND $\text{Confidence} \ge 0.65 \implies \mathbf{LOCAL}$ (Ollama)
   - Otherwise $\implies \mathbf{GEMINI}$ (Cloud API)

### 2. Privacy Sanitization Boundary
When Gemini is selected:
- Detects API keys (`AIza...`, `sk-...`), bearer tokens, passwords, private keys, database connection strings (`postgresql://`, `mysql://`), email addresses, and phone numbers.
- Redacts secrets in-memory using descriptive placeholders (`[REDACTED_API_KEY]`, `[PASSWORD_REDACTED]`).
- Only sanitized payloads reach `GeminiClient`.
- The raw, untouched message is stored in local SQLite (`data/user.db`).

### 3. Model Attribution Badges
Every assistant response renders a visual badge in the chat UI:
- `⚡ Answered by: Ollama (llama3.2:1b)`
- `✨ Answered by: Gemini (gemini-3.5-flash-lite)`

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /` | `GET` | Serves the web chat interface |
| `GET /health` | `GET` | Health status of backend, Ollama, and model configurations |
| `GET /models` | `GET` | Lists available local Ollama models |
| `POST /chat` | `POST` | Executes intelligent routing, inference, and event emission |
| `POST /feedback` | `POST` | Submits message feedback (thumbs up/down) |
| `GET /profile` | `GET` | Retrieves user profile (interests, preferences, behavior stats) |
| `GET /analytics/events` | `GET` | Returns Kafka event counts and local spool statistics |
| `POST /behavior/drain` | `POST` | Drains and processes offline spooled events |
| `POST /analytics/export` | `POST` | Exports a curated, versioned JSONL training dataset |

---

## Quickstart Guide

### 1. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to provide your Gemini API key:
```dotenv
# Phase 1 - Local Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=120

# Phase 2 - Kafka & Storage
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_INTERACTIONS=interaction.events
KAFKA_ENABLED=true
SQLITE_DB_PATH=data/user.db

# Phase 3 - Gemini Cloud & Router
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TIMEOUT=60
LOCAL_MIN_SUITABILITY=0.70
LOCAL_MIN_CONFIDENCE=0.65
LOCAL_MAX_CONTEXT_CHARS=6000
ROUTER_FALLBACK_TO_LOCAL=true
```

### 2. Start Kafka (Optional for streaming)
```bash
cd "own-ai/kafka"
docker compose up -d
```
*(If Kafka is not running, the application automatically spools events to `data/events/` without interruption).*

### 3. Start Local Ollama
```bash
export OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models"
/opt/homebrew/bin/ollama serve
```

### 4. Start the FastAPI Application
```bash
cd "own-ai/backend"
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 5. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8000
```

---

## Testing & Verification

### 1. Automated Test Suite (96 Tests)
Run the complete regression test suite covering Phase 1, Phase 2, Phase 3, and Phase 4:
```bash
cd "own-ai/backend"
venv/bin/pytest tests/ --ignore=tests/integration/test_ollama_live.py -v
```

**Results:**
```text
94 passed, 2 warnings in 0.81s (100% PASS)
- tests/behavior/ (19 tests) PASSED
- tests/events/ (3 tests) PASSED
- tests/llm/ (4 tests) PASSED
- tests/privacy/ (7 tests) PASSED
- tests/router/ (15 tests) PASSED
- tests/storage/ (4 tests) PASSED
- tests/integration/ (6 tests) PASSED
- tests/memory/ (21 tests) PASSED:
  ├── test_detector.py (5 tests)
  ├── test_confidence_and_contradiction.py (4 tests)
  ├── test_lifecycle_and_retrieval.py (4 tests)
  ├── test_context_builder.py (3 tests)
  ├── test_privacy_boundary.py (1 test)
  ├── test_memory_api.py (2 tests)
  └── test_chat_service_memory.py (2 tests)
- Phase 1 chat & config tests (17 tests) PASSED
```

### 2. Live Ollama Verification (2 Tests)
```bash
venv/bin/pytest tests/integration/test_ollama_live.py -v
```
Output: `2 passed in 1.79s (100% PASS)`

---

## Phase 4: Memory & Personalization

### 1. Memory Lifecycle & Flow
```
User Prompt
    │
    ▼
ChatService
    │
    ├── 1. Retrieve Active Memories (Scored by: 0.35*Topic + 0.25*Importance + 0.25*Confidence + 0.15*Recency)
    ├── 2. ContextBuilder (Prepends system directive: Current request instructions ALWAYS supersede background memory)
    ├── 3. ModelRouter (Phase 3: Evaluates Local vs. Gemini routing)
    ├── 4. If Gemini: PrivacyScanner sanitizes personalized context (API keys, passwords redacted; original preserved locally)
    ├── 5. Single-Model Execution (Local Ollama or Gemini)
    ├── 6. Persist Interaction & Emit Event (Phase 2 Outbox / Kafka)
    └── 7. Async Memory Detection (Zero blocking: extracts preferences, goals, facts, instructions, detects contradictions)
```

### 2. Memory REST API
- `GET /memory`: List memories (filter by `type`, `status_filter`, or `user_id`).
- `GET /memory/{memory_id}`: Retrieve single memory details.
- `POST /memory`: Create memory record manually.
- `PATCH /memory/{memory_id}`: Update memory content, type, importance, or confidence.
- `DELETE /memory/{memory_id}`: Soft-archive or hard-delete memory.

### 3. Frontend Memory Management UI
- **Header & Sidebar Buttons**: Direct access via `🧠 Memories`.
- **Add Memory Form**: Inline creation with category selection (`preference`, `interest`, `goal`, `instruction`, `fact`, `context`).
- **Filter Tabs**: Instant filtering across memory types.
- **Card Controls**: Live in-place edit and soft archive/deletion.

---

## Phase 5: Behavior-Based Model Training & Cloud GPU Pipeline

Phase 5 introduces privacy-first model customization: extracting behavioral patterns, formatting reproducible datasets, preparing cloud training bundles for free GPUs (Google Colab T4 / Kaggle), evaluating candidates against a fixed benchmark, and controlled 1-click activation/rollback into the local Ollama runtime.

### 1. Architecture & Lifecycle

```
Local SQLite Interactions & Feedback
                 │
                 ▼
        User Training Consent Check (Opt-In Required)
                 │
                 ▼
    ConsentAwareExtractor & TrainingSanitizer
    ├── Secret/Credential Exclusion (Zero API keys, private keys, tokens)
    ├── Quality Filtering (Exclude thumbs_down & ratings <= 2)
    └── PII Redaction ([EMAIL_REDACTED], [PHONE_REDACTED])
                 │
                 ▼
          DatasetBuilder
    ├── ChatML JSONL Generation (train.jsonl, val.jsonl)
    └── SHA-256 Content Deduplication & Provenance Manifest
                 │
                 ▼
          TrainingRunner
    └── Exports Cloud GPU Bundle (`cloud_bundle_<run_id>.zip`)
                 │
                 ▼
   ┌──────────────────────────────────────────────┐
   │ Cloud Training (Google Colab / Kaggle Free)  │
   │ - Hardware: NVIDIA T4 / A100 GPU             │
   │ - Method: Hugging Face PEFT LoRA Fine-Tuning │
   │ - Output: `trained_adapter` weights          │
   └──────────────────────────────────────────────┘
                 │
                 ▼
          ModelEvaluator (Fixed Benchmark Suite)
    ├── Instruction Following, Code Style, Explanation Style, Privacy
    └── Minimum Accuracy Threshold: >= 75%
                 │
                 ▼
          OllamaPackager
    └── Generates versioned `Modelfile` & imports to Ollama daemon
                 │
                 ▼
       ModelDeploymentManager
    ├── Pre-flight Health & Smoke Test Verification
    ├── Atomically Activates Candidate Model for Local Routing
    └── 1-Click Rollback to Baseline (`llama3.2:1b`)
```

### 2. Cloud GPU Training Workflow (Google Colab Free T4)

When training without a local discrete GPU (e.g., Apple Silicon or integrated graphics laptops), own-ai offloads only the compute-intensive gradient calculations to cloud GPUs while maintaining full data governance:

1. **Build Dataset & Export Bundle**:
   ```bash
   python -m app.training.cli consent --opt-in
   python -m app.training.cli dataset
   python -m app.training.cli run --dataset <DATASET_VERSION>
   ```
   This generates `data/training/bundles/cloud_bundle_<run_id>.zip`.

2. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
   - Set Hardware Accelerator: **Runtime -> Change runtime type -> T4 GPU**.

3. **Run Training in Colab**:
   ```python
   # 1. Upload and unpack the bundle
   !unzip cloud_bundle_<run_id>.zip

   # 2. Install dependencies
   !pip install -q torch transformers datasets peft accelerate bitsandbytes

   # 3. Train LoRA adapter
   !python train_gpu.py --data_dir . --output_dir ./trained_adapter --epochs 3
   ```

4. **Import & Activate Locally**:
   Download the resulting `trained_adapter` directory and run:
   ```bash
   python -m app.training.cli eval --run-id <run_id>
   python -m app.training.cli package --run-id <run_id>
   python -m app.training.cli activate --run-id <run_id>
   ```

### 3. Training CLI Reference

All Phase 5 pipeline stages can be inspected and triggered via the CLI utility:

```bash
# Check current pipeline status, active model, and consent
python -m app.training.cli status

# Manage user training consent (opt-in / opt-out)
python -m app.training.cli consent --opt-in
python -m app.training.cli consent --opt-out

# Build curated JSONL training dataset
python -m app.training.cli dataset --split 0.85

# Create training run and cloud bundle
python -m app.training.cli run --dataset <version> --epochs 3

# Evaluate model candidate against benchmark
python -m app.training.cli eval --run-id <run_id>

# Package into Ollama Modelfile
python -m app.training.cli package --run-id <run_id>

# Activate candidate model for routing
python -m app.training.cli activate --run-id <run_id>

# Immediate rollback to baseline
python -m app.training.cli rollback
```

### 4. Phase 5 REST API

- `GET /api/training/status`: Get active model, consent status, and recent runs.
- `GET /api/training/consent`: Check consent status.
- `POST /api/training/consent`: Opt-in or opt-out of training data collection.
- `POST /api/training/dataset`: Build curated, sanitized JSONL dataset.
- `POST /api/training/runs`: Create training run and generate `.zip` cloud bundle.
- `GET /api/training/runs`: List all registered runs.
- `GET /api/training/runs/{run_id}`: Get details for a specific run.
- `POST /api/training/evaluate`: Evaluate candidate against benchmark suite.
- `POST /api/training/package`: Generate Modelfile and register with Ollama.
- `POST /api/training/activate`: Perform pre-flight check and activate candidate model.
- `POST /api/training/rollback`: Rollback active model to base `llama3.2:1b`.


