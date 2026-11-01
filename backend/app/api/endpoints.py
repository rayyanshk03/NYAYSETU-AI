"""FastAPI endpoints for NyayaSetu AI — Step 4: Agentic AI Layer."""

from dataclasses import asdict
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.chat_service import chat_service
from app.rag.retrieval import retrieval_service
from app.rag.hybrid_retriever import hybrid_retrieval_service
from app.rag.audit import CorpusAuditor

api_router = APIRouter()


# ── Health ──────────────────────────────────────────────────────────────────

@api_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns the health status and metadata of the NyayaSetu AI backend service.",
)
async def get_health() -> HealthResponse:
    """Return operational health status."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
    )


# ── Chat (Agentic Pipeline) ─────────────────────────────────────────────────

@api_router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Legal Query (Agentic)",
    description=(
        "Runs the full agentic pipeline: query planning → hybrid retrieval → "
        "grounded answer synthesis. Returns a response with statutory citations, "
        "retrieved passages, and the agent execution trace."
    ),
)
async def post_chat(request: ChatRequest) -> ChatResponse:
    """Handle incoming legal question through the agentic pipeline."""
    return chat_service.process_request(request)


# ── Retrieval (Hybrid Retrieval Endpoint) ───────────────────────────────────

@api_router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid Legal Retrieval",
    description=(
        "Production hybrid retrieval layer combining ChromaDB dense vector similarity, "
        "BM25 sparse keyword search, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking. "
        "Returns retrieved passages with similarity scores, provenance, and latency metadata."
    ),
    tags=["Retrieval"],
)
async def post_retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """Execute hybrid retrieval over the indexed legal corpus."""
    return hybrid_retrieval_service.retrieve(
        query=request.query,
        top_k=request.top_k,
        language=request.language,
        document_id=request.document_id,
        min_score=request.min_score,
    )


# ── Corpus Status & Audit ───────────────────────────────────────────────────

@api_router.get(
    "/corpus/status",
    status_code=status.HTTP_200_OK,
    summary="Knowledge Base Status",
    description="Returns the total number of indexed legal provisions and collection metadata.",
    tags=["Corpus"],
)
async def get_corpus_status():
    """Return how many legal chunks are currently indexed in ChromaDB."""
    count = hybrid_retrieval_service.vector_store.count()
    return {
        "indexed_provisions": count,
        "collection_name": hybrid_retrieval_service.vector_store.collection_name,
        "status": "ready" if count > 0 else "empty",
        "hybrid_retrieval_enabled": settings.ENABLE_HYBRID_RETRIEVAL,
        "reranker_enabled": settings.ENABLE_RERANKER,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "message": (
            f"{count} legal provisions indexed and ready for hybrid retrieval."
            if count > 0
            else (
                "Knowledge base is empty. Download official Bare Act PDFs into data/raw/, "
                "run `python -m app.ingestion.pipeline`, then `python -m app.rag.indexer`."
            )
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@api_router.post(
    "/corpus/audit",
    status_code=status.HTTP_200_OK,
    summary="Run Corpus Audit",
    description="Audits raw PDFs, processed JSONLs, metadata quality, and checksum duplicates.",
    tags=["Corpus"],
)
async def run_corpus_audit():
    """Run corpus auditor and return detailed quality audit report."""
    auditor = CorpusAuditor()
    report = auditor.run()
    return asdict(report)
