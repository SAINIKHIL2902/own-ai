# Personal Local AI — Phase 1 & Phase 2

A clean, reliable, modular, completely offline **Personal Local AI** running on macOS Apple Silicon.

- **Phase 1**: Local LLM inference via Ollama (`llama3.2:1b`), FastAPI backend, and ChatGPT-style web interface.
- **Phase 2**: Event-Driven User Behavior, Interest, Preference & Context Modeling using Apache Kafka, local SQLite storage, and a Behavior Analysis worker.

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
                        ┌────────┴─────────┐
                        │                  │
                        ▼                  ▼
                    Ollama            Kafka Producer (Async Background)
                        │                  │
                        ▼                  ▼
                    Local LLM       Topic: `interaction.events`
                        │                  │
                        ▼             ┌────┴─────┐
                     Response         │          │
                        │             │          │ (Fallback: local data/events/ spool)
                        └─────────────┘          ▼
                                        Behavior Analyzer Consumer
                                                 │
                                ┌────────────────┼────────────────┐
                                │                │                │
                                ▼                ▼                ▼
                            Interests       Preferences       Behavior Stats
                                │                │                │
                                └────────────────┼────────────────┘
                                                 │
                                                 ▼
                                        Local SQLite Storage (`data/user.db`)
                                        ├── RAW: conversations, messages, events, feedback
                                        └── DERIVED: interests, preferences, contexts, stats
```

> [!IMPORTANT]
> **Core Architectural Principles**:
> 1. **Zero LLM Blocking**: Synchronous chat inference (`FastAPI -> Ollama -> Local LLM -> Response`) never blocks on Kafka.
> 2. **Kafka Fault-Tolerance**: If Kafka is stopped or unreachable, events spool safely to local disk (`data/events/`), logging a warning while user chat succeeds smoothly.
> 3. **Raw vs. Derived Separation**: Raw interaction history is preserved untouched so future algorithms can replay/recalculate features from scratch.
> 4. **No Cloud / No Training in Phase 2**: Strictly local data capture and behavioral modeling.

---

## Directory Structure

```
own-ai/
├── README.md                           # Comprehensive documentation & guide
├── .gitignore                          # Ignores venv/, .env, models/, data/*.db, data/events/*
├── .env.example                        # Template for environment configuration
├── .env                                # Local configuration (not committed)
├── models/                             # Local model storage directory (OLLAMA_MODELS)
├── data/                               # Local persistent storage (Phase 2)
│   ├── user.db                         # SQLite database (Raw + Derived tables)
│   ├── events/                         # Local event spool for offline fallback
│   └── exports/                        # Curated training dataset exports
├── kafka/                              # Kafka infrastructure
│   ├── docker-compose.yml              # Lightweight single-node KRaft Kafka service
│   └── README.md                       # Setup and management instructions
├── backend/
│   ├── venv/                           # Isolated Python virtual environment
│   ├── requirements.txt                # FastAPI, Uvicorn, httpx, Pydantic, aiokafka, pytest
│   ├── app/
│   │   ├── main.py                     # FastAPI routes, lifespan, /feedback, /profile, /export
│   │   ├── config/
│   │   │   └── settings.py             # Strongly-typed Pydantic settings
│   │   ├── ollama/
│   │   │   ├── client.py               # Async Ollama HTTP client
│   │   │   └── models.py               # Request/Response schemas
│   │   ├── chat/
│   │   │   └── service.py              # Chat orchestration & background event publisher
│   │   ├── events/
│   │   │   ├── schemas.py              # InteractionEvent & FeedbackEvent definitions
│   │   │   └── producer.py             # Resilient Kafka producer with local disk spooling
│   │   ├── behavior/
│   │   │   ├── interest.py             # Topic extraction
│   │   │   ├── scoring.py              # Recency decay & weighted interest scoring
│   │   │   ├── preferences.py          # Evidence-based preference detection
│   │   │   ├── context.py              # Project and learning context tracking
│   │   │   └── analyzer.py             # Behavior analyzer consumer & spool drainer
│   │   └── storage/
│   │       ├── database.py             # SQLite WAL-mode connection & schema initialization
│   │       ├── models.py               # Dataclass records for raw & derived data
│   │       └── repositories.py         # RawRepository and ProfileRepository
│   └── tests/
│       ├── test_config.py
│       ├── test_chat.py
│       ├── test_api.py
│       ├── events/
│       │   ├── test_schemas.py
│       │   └── test_producer.py
│       ├── behavior/
│       │   ├── test_interest.py
│       │   ├── test_scoring.py
│       │   ├── test_preferences.py
│       │   └── test_context.py
│       ├── storage/
│       │   └── test_repositories.py
│       └── integration/
│           ├── test_ollama_live.py     # Live Ollama inference test
│           └── test_pipeline_live.py   # Full pipeline integration test
└── frontend/
    ├── index.html                      # ChatGPT-style UI with Profile Modal & Feedback
    ├── style.css                       # Modern dark-mode styling (zero CDNs/fonts)
    └── app.js                          # Client-side chat, feedback, and profile logic
```

---

## Phase 2: Behavior Modeling & Data Principles

### 1. Raw Data (Immutable Source of Truth)
- **`conversations`**: ID, title, timestamps.
- **`messages`**: Message ID, role (`user`/`assistant`), content, model, latency.
- **`raw_events`**: Event ID, timestamp, user ID, full JSON payload, processed flag.
- **`feedback`**: Message ID, feedback type (`positive`, `negative`, `correction`), rating, comment.

### 2. Derived Data (Recalculable Profile)
- **`user_interests`**: Topics (e.g. Python, Kafka, MLOps), frequency count, recent interactions, last seen, calculated score.
- **`user_preferences`**: Detected styles (e.g. `prefers_code`, `prefers_step_by_step`, `prefers_concise_answers`) with evidence count and confidence score.
- **`user_contexts`**: Active learning and project contexts with timestamps and status.
- **`user_behavior_stats`**: Total messages, average prompt length, active hours, feedback counts.

### 3. Scoring & Recency Decay Formula
```text
decay = (INTEREST_DECAY_RATE) ^ (elapsed_days)
decayed_score = old_score * decay
interaction_weight = (frequency_weight * 0.2) + (recency_weight * 0.3) + (engagement_weight * depth) + (feedback_weight * feedback)
new_score = decayed_score + (1.0 - decayed_score) * (interaction_weight * 0.5)
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /` | `GET` | Serves the ChatGPT-like chat interface |
| `GET /health` | `GET` | Checks backend and Ollama connectivity |
| `GET /models` | `GET` | Lists locally installed Ollama models |
| `POST /chat` | `POST` | Executes real local inference and emits background event |
| `POST /feedback` | `POST` | Submits message feedback (thumbs up/down/correction) |
| `GET /profile` | `GET` | Returns derived user profile (interests, preferences, stats) |
| `GET /analytics/events` | `GET` | Returns raw events count, spool status, and Kafka health |
| `POST /behavior/drain` | `POST` | Drains and processes offline spooled events |
| `POST /analytics/export` | `POST` | Generates a versioned training dataset JSONL file |

---

## Running the Application

### 1. Start Kafka (Optional for full streaming)
```bash
cd "own-ai/kafka"
docker compose up -d
```
*(If Kafka is not started, the application automatically falls back to local spooling in `data/events/` without errors).*

### 2. Start Ollama Daemon
```bash
export OLLAMA_MODELS="/Users/sainikhilreddy/Documents/my own files/own-ai/models"
/opt/homebrew/bin/ollama serve
```

### 3. Start the FastAPI Backend
```bash
cd "own-ai/backend"
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 4. Open the Interface
Navigate to:
```
http://127.0.0.1:8000/
```
- Chat with the local AI.
- Rate responses with 👍 or 👎.
- Click **📊 Profile** in the top navigation to view your modeled interests and preferences.
- Click **Export Versioned Dataset** in the modal to create a training dataset in `data/exports/`.

---

## Running the Test Suite

Run all 40 automated tests:
```bash
cd "own-ai/backend"
venv/bin/pytest tests/ -v
```

**Results:**
```
40 passed in 0.49s (100% PASS)
```
