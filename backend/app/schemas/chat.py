"""Chat request and response schemas."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SupportedLanguage(str, Enum):
    """Supported Indian languages in NyayaSetu AI."""

    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    AUTO = "auto"

    @classmethod
    def _missing_(cls, value: object):
        # Allow case-insensitive or descriptive names (e.g., 'english', 'Hindi', 'kannada', 'auto')
        if isinstance(value, str):
            val_lower = value.strip().lower()
            mapping = {
                "en": cls.ENGLISH,
                "english": cls.ENGLISH,
                "hi": cls.HINDI,
                "hindi": cls.HINDI,
                "kn": cls.KANNADA,
                "kannada": cls.KANNADA,
                "auto": cls.AUTO,
                "detect": cls.AUTO,
                "automatic": cls.AUTO,
            }
            if val_lower in mapping:
                return mapping[val_lower]
        return None


class ChatRequest(BaseModel):
    """Request payload for NyayaSetu AI chat query."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The family law query submitted by the user",
        json_schema_extra={"example": "What are the grounds for divorce under the Hindu Marriage Act?"},
    )
    language: Optional[SupportedLanguage] = Field(
        default=None,
        description="User preferred language (en: English, hi: Hindi, kn: Kannada, auto: detect). If omitted, auto-detection is applied.",
        json_schema_extra={"example": "en"},
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query message cannot be empty or contain only whitespace.")
        return trimmed


class ChatResponse(BaseModel):
    """Structured response payload for chat endpoint with RAG citations."""

    status: str = Field(default="success", description="Status code or flag")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this response")
    received_message: str = Field(..., description="Echoed input message")
    language: SupportedLanguage = Field(..., description="Processed language")
    reply: str = Field(..., description="Assistant response text")
    is_placeholder: bool = Field(default=True, description="Indicates whether this is a development placeholder")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC response timestamp",
    )
    disclaimer: str = Field(
        default="NyayaSetu AI provides general legal information and decision support for Indian family law. It is not legal advice and does not substitute for consultation with a qualified legal practitioner.",
        description="Mandatory legal disclaimer",
    )
    evidence_status: str = Field(
        default="sufficient",
        description="Evidence status: 'sufficient', 'insufficient', 'empty_corpus', or 'emergency'",
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Verified source citations matching retrieved passages",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Important statutory limitations or caveats",
    )
    retrieved_passages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved statutory passages from the legal knowledge base",
    )
    agent_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Agent execution trace steps (planner, retriever, synthesizer)",
    )
    # Step 5 additions — backward-compatible (default None)
    clarification_question: Optional[str] = Field(
        default=None,
        description="Clarification question to the user when query is materially ambiguous",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Classified legal topic label (e.g. 'Divorce', 'Maintenance')",
    )
    # Step 6 additions — language detection metadata
    detected_language: Optional[SupportedLanguage] = Field(
        default=None,
        description="Language automatically detected from user query",
    )
    language_detection_method: Optional[str] = Field(
        default=None,
        description="Detection method used: script_devanagari, script_kannada, latin_english, etc.",
    )
    language_uncertain: bool = Field(
        default=False,
        description="True if language detection was uncertain and user selection is requested",
    )

