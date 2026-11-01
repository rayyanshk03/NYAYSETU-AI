"""FastAPI endpoints for NyayaSetu AI — Step 4: Agentic AI Layer."""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.chat_service import chat_service
from app.rag.retrieval import retrieval_service

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
        "Runs the full agentic pipeline: query planning → legal retrieval → "
        "grounded answer synthesis. Returns a response with statutory citations, "
        "retrieved passages, and the agent execution trace."
    ),
)
async def post_chat(request: ChatRequest) -> ChatResponse:
    """Handle incoming legal question through the agentic pipeline."""
    return chat_service.process_request(request)


# ── Retrieval (Development / Inspection Endpoint) ───────────────────────────

@api_router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Raw Legal Retrieval",
    description=(
        "Development-mode direct access to the ChromaDB retrieval layer. "
        "Returns raw retrieved passages with similarity scores and metadata "
        "for inspection without answer synthesis. "
        "Useful for evaluating retrieval quality before and after indexing."
    ),
    tags=["Development"],
)
async def post_retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """Execute raw vector retrieval over the indexed legal corpus."""
    return retrieval_service.retrieve(
        query=request.query,
        top_k=request.top_k,
        language=request.language,
        document_id=request.document_id,
        min_score=request.min_score,
    )


# ── Corpus Status ───────────────────────────────────────────────────────────

@api_router.get(
    "/corpus/status",
    status_code=status.HTTP_200_OK,
    summary="Knowledge Base Status",
    description="Returns the total number of indexed legal provisions and collection metadata.",
    tags=["Development"],
)
async def get_corpus_status():
    """Return how many legal chunks are currently indexed in ChromaDB."""
    count = retrieval_service.vector_store.count()
    return {
        "indexed_provisions": count,
        "collection_name": retrieval_service.vector_store.collection_name,
        "status": "ready" if count > 0 else "empty",
        "message": (
            f"{count} legal provisions indexed and ready for retrieval."
            if count > 0
            else (
                "Knowledge base is empty. Download official Bare Act PDFs into data/raw/, "
                "run `python -m app.ingestion.pipeline`, then `python -m app.rag.indexer`."
            )
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
