from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Phase 1 - Ollama configuration
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    OLLAMA_TIMEOUT: int = 120

    # Phase 2 - Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_INTERACTIONS: str = "interaction.events"
    KAFKA_CONSUMER_GROUP: str = "behavior-analyzer"
    KAFKA_ENABLED: bool = True

    # Phase 2 - Local SQLite Database
    SQLITE_DB_PATH: str = "data/user.db"

    # Phase 2 - Scoring Weights & Decay
    INTEREST_FREQUENCY_WEIGHT: float = 0.35
    INTEREST_RECENCY_WEIGHT: float = 0.35
    INTEREST_ENGAGEMENT_WEIGHT: float = 0.20
    INTEREST_FEEDBACK_WEIGHT: float = 0.10
    INTEREST_DECAY_RATE: float = 0.95

    # Load from .env file if present
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_db_path(self) -> Path:
        p = Path(self.SQLITE_DB_PATH)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p


settings = Settings()
