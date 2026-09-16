from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENAI_ORG_ID: Optional[str] = None

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100

    # LLM Models
    LLM_MODEL_PRIMARY: str = "gpt-4o-mini"
    LLM_MODEL_FALLBACK: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000

    # Vector DB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "banking_knowledge"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "banking_knowledge"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_TTL: int = 3600
    REDIS_RATE_LIMIT_WINDOW: int = 60
    REDIS_RATE_LIMIT_MAX_REQUESTS: int = 30

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/bank_chatbot"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 30

    # LangSmith
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "bank-chatbot"
    LANGSMITH_TRACING_V2: bool = True

    # Guardrails
    GUARDRAILS_CONFIG_DIR: str = "./src/bank_chatbot/guardrails/config"

    # Evaluation
    EVAL_GOLDEN_SET_PATH: str = "./data/eval/golden_set.json"
    EVAL_THRESHOLD_FAITHFULNESS: float = 0.70
    EVAL_THRESHOLD_RELEVANCE: float = 0.70
    EVAL_THRESHOLD_PRECISION: float = 0.50

    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", description="Secret key for JWT")
    AUTH_TOKEN: str = Field(default="dev-token-change-in-production", description="API authentication token")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 30
    RATE_LIMIT_BURST: int = 10

    # Feature Flags
    FEATURE_RAG_ENABLED: bool = True
    FEATURE_TOOLS_ENABLED: bool = False
    FEATURE_MULTI_AGENT_ENABLED: bool = False
    FEATURE_HUMAN_HANDOFF_ENABLED: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"


settings = Settings()


def get_settings() -> Settings:
    return settings