"""Application configuration module using Pydantic Settings."""

from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """NyayaSetu AI backend application settings."""

    APP_NAME: str = "NyayaSetu AI — Multilingual Family Law Information System"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # Host & Port configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS configuration - restricted to local Vite dev server by default
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # RAG & Vector Database Settings
    # Pretrained multilingual embedding model with support for English, Hindi, and Kannada
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "nyayasetu_legal_corpus"
    PROCESSED_DATA_DIR: str = "data/processed"
    DEFAULT_TOP_K: int = 5
    # Similarity threshold: queries below this minimum score return insufficient evidence result
    MIN_SIMILARITY_SCORE: float = 0.25

    # Hybrid Retrieval & Reranker Settings
    ENABLE_HYBRID_RETRIEVAL: bool = True
    HYBRID_DENSE_CANDIDATES: int = 20
    HYBRID_SPARSE_CANDIDATES: int = 20
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K: int = 5

    # Step 4: Configurable LLM Provider Settings
    # Providers supported: 'template' (deterministic local fallback with 0 API keys),
    #                      'ollama' (local Ollama instance),
    #                      'openai' (OpenAI or compatible: Groq, DeepSeek, vLLM, OpenRouter),
    #                      'gemini' (Google Gemini API via OpenAI-compatible endpoint or REST)
    LLM_PROVIDER: str = "template"
    LLM_API_KEY: Union[str, None] = None
    LLM_API_BASE: Union[str, None] = None
    LLM_MODEL_NAME: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_REQUEST_TIMEOUT: float = 30.0

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
