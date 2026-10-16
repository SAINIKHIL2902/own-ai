import logging
from pathlib import Path
import sqlite3
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- =============================================================================
-- RAW DATA TABLES (Immutable historical source of truth)
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    id TEXT,
    conversation_id TEXT NOT NULL,
    user_id TEXT DEFAULT 'local_user',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT DEFAULT 'local',
    latency_ms REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS raw_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    processed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    id TEXT,
    message_id TEXT NOT NULL,
    conversation_id TEXT,
    user_id TEXT DEFAULT 'local_user',
    feedback_type TEXT NOT NULL,
    feedback_value TEXT,
    rating INTEGER,
    comment TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages (message_id) ON DELETE CASCADE
);

-- =============================================================================
-- DERIVED PROFILE TABLES (Recalculable from raw events)
-- =============================================================================
CREATE TABLE IF NOT EXISTS user_interests (
    topic TEXT PRIMARY KEY,
    score REAL NOT NULL,
    interaction_count INTEGER DEFAULT 1,
    recent_interactions INTEGER DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER DEFAULT 1,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_contexts (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    activity TEXT NOT NULL,
    context TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_response_styles (
    style_key TEXT PRIMARY KEY,
    style_name TEXT NOT NULL,
    description TEXT NOT NULL,
    liked_count INTEGER DEFAULT 0,
    disliked_count INTEGER DEFAULT 0,
    affinity_score REAL DEFAULT 0.5,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_behavior_stats (
    stat_key TEXT PRIMARY KEY,
    stat_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- =============================================================================
-- PHASE 4 MEMORY TABLES
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'local_user',
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_reference TEXT,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    evidence_count INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_confirmed_at TEXT,
    expires_at TEXT,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_feedback_msg ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_events_processed ON raw_events(processed);
CREATE INDEX IF NOT EXISTS idx_interests_score ON user_interests(score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
"""


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.resolved_db_path
        self._ensure_directory()
        self.init_schema()

    def _ensure_directory(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        """Initialize all raw, derived, and memory tables with schema versioning and seamless migrations."""
        with self.get_connection() as conn:
            conn.executescript(INIT_SQL)

            # Migration: ensure messages table has message_id, id, and user_id
            msg_cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
            if "message_id" not in msg_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN message_id TEXT")
                conn.execute("UPDATE messages SET message_id = id WHERE message_id IS NULL")
            if "id" not in msg_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN id TEXT")
                conn.execute("UPDATE messages SET id = message_id WHERE id IS NULL")
            if "user_id" not in msg_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT 'local_user'")
            if "provider" not in msg_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN provider TEXT DEFAULT 'local'")

            # Migration: ensure feedback table has feedback_id, id, user_id, feedback_value, and metadata_json
            fb_cols = [r["name"] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()]
            if "feedback_id" not in fb_cols:
                conn.execute("ALTER TABLE feedback ADD COLUMN feedback_id TEXT")
                conn.execute("UPDATE feedback SET feedback_id = id WHERE feedback_id IS NULL")
            if "id" not in fb_cols:
                conn.execute("ALTER TABLE feedback ADD COLUMN id TEXT")
                conn.execute("UPDATE feedback SET id = feedback_id WHERE id IS NULL")
            if "user_id" not in fb_cols:
                conn.execute("ALTER TABLE feedback ADD COLUMN user_id TEXT DEFAULT 'local_user'")
            if "feedback_value" not in fb_cols:
                conn.execute("ALTER TABLE feedback ADD COLUMN feedback_value TEXT")
            if "metadata_json" not in fb_cols:
                conn.execute("ALTER TABLE feedback ADD COLUMN metadata_json TEXT")

            # Indexes for foreign key lookup and message_id
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_msg_id ON messages(message_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_msg_id ON feedback(message_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")

            conn.execute(
                "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        logger.info(f"Initialized SQLite database at {self.db_path} (schema v{SCHEMA_VERSION})")


db_manager = DatabaseManager()
