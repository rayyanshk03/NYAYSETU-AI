"""Schemas for RAG legal retrieval requests and responses."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import SupportedLanguage


class RetrievalRequest(BaseModel):
    """Request payload for retrieving relevant legal passages."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The legal question or query to search across the legal corpus",
        json_schema_extra={"example": "What are the grounds for mutual consent divorce under the Hindu Marriage Act?"},
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of relevant legal passages to retrieve",
        json_schema_extra={"example": 5},
    )
    language: Optional[SupportedLanguage] = Field(
        default=SupportedLanguage.ENGLISH,
        description="User query language (en, hi, kn)",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific Act (e.g. 'hma_1955')",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold (0.0 to 1.0)",
    )


class RetrievedPassage(BaseModel):
    """Structured representation of an authoritative legal passage retrieved from the vector store."""

    chunk_id: str = Field(..., description="Unique chunk identifier from Step 2 ingestion")
    document_id: str = Field(..., description="Canonical document ID (e.g. 'hma_1955')")
    document_title: str = Field(..., description="Full formal title of the statute")
    act_name: Optional[str] = Field(default=None, description="Common statute title, e.g. 'Hindu Marriage Act, 1955'")
    section_number: Optional[str] = Field(default=None, description="Section number (e.g. '13B', '125')")
    section_title: Optional[str] = Field(default=None, description="Statutory heading of the section")
    subsection_id: Optional[str] = Field(default=None, description="Subsection or clause identifier")
    page_start: int = Field(default=1, description="Starting 1-indexed page in original bare act PDF")
    page_end: int = Field(default=1, description="Ending 1-indexed page in original bare act PDF")
    page_number: Optional[int] = Field(default=None, description="Page number in original document (alias to page_start)")
    source_url: str = Field(..., description="Official India Code or government repository URL")
    publication_date: Optional[str] = Field(default=None, description="Enactment or gazette publication date")
    effective_date: Optional[str] = Field(default=None, description="Date law came into force, if verified")
    document_version: Optional[str] = Field(default=None, description="Bare act version/amendment status")
    legal_text: str = Field(..., description="Cleaned statutory legal text")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0; not a legal correctness probability)")
    is_split_fallback: bool = Field(default=False, description="Whether section was split across chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete dictionary of indexed metadata")

    def model_post_init(self, __context: Any) -> None:
        if self.page_number is None:
            self.page_number = self.page_start
        if not self.metadata:
            self.metadata = {
                "document_id": self.document_id,
                "document_title": self.document_title,
                "act_name": self.act_name,
                "section_number": self.section_number,
                "section_title": self.section_title,
                "subsection_id": self.subsection_id,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "page_number": self.page_number,
                "source_url": self.source_url,
                "publication_date": self.publication_date,
                "effective_date": self.effective_date,
                "document_version": self.document_version,
            }


class RetrievalResponse(BaseModel):
    """Structured response payload for /api/retrieve endpoint."""

    status: str = Field(default="success", description="Status code")
    query: str = Field(..., description="Echoed input search query")
    language: Optional[SupportedLanguage] = Field(default=SupportedLanguage.ENGLISH)
    total_results: int = Field(..., description="Number of legal passages returned")
    insufficient_evidence: bool = Field(
        default=False,
        description="True if no indexed passages met the relevance threshold or corpus is empty",
    )
    evidence_note: Optional[str] = Field(
        default=None,
        description="Explanation when evidence is insufficient or when indexing status requires attention",
    )
    passages: List[RetrievedPassage] = Field(
        default_factory=list,
        description="List of top-k retrieved legal passages ordered by relevance",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of retrieval",
    )
    disclaimer: str = Field(
        default="Retrieved passages reflect statutory provisions from indexed official sources. Similarity scores measure semantic closeness and must not be interpreted as calibrated probabilities of legal validity or outcomes.",
        description="Mandatory retrieval disclaimer",
    )
