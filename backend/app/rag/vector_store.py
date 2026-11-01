"""ChromaDB vector store integration for legal document chunks.

Manages persistent local collection, idempotent batch indexing, metadata serialization,
and vector similarity querying.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.ingestion.chunk import LegalChunk
from app.rag.embeddings import MultilingualEmbedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages persistent ChromaDB collection for the NyayaSetu legal knowledge base."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[MultilingualEmbedder] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.persist_dir = Path(persist_dir) if persist_dir else (base_dir / settings.CHROMA_PERSIST_DIRECTORY)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedder = embedder or MultilingualEmbedder.get_instance()

        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Initialize or return the persistent ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def get_collection(self):
        """Retrieve or create the legal knowledge base Chroma collection."""
        if self._collection is None:
            client = self._get_client()
            # Configure collection with cosine distance
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "NyayaSetu AI Authoritative Indian Family Law Corpus"},
            )
        return self._collection

    def count(self) -> int:
        """Return the total number of indexed legal chunks in the collection."""
        try:
            col = self.get_collection()
            return col.count()
        except Exception as e:
            logger.error("Failed to get count from Chroma collection: %s", e)
            return 0

    def reset_collection(self) -> None:
        """
        Explicitly delete and recreate the collection.
        This is a documented, deliberate operation for rebuilding the index.
        """
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
            logger.info("Successfully deleted Chroma collection '%s'", self.collection_name)
        except Exception as e:
            logger.warning("Could not delete collection '%s' (may not exist): %s", self.collection_name, e)
        self._collection = None
        # Re-create empty collection
        self.get_collection()

    @staticmethod
    def _sanitize_metadata(chunk: LegalChunk) -> Dict[str, Any]:
        """Convert LegalChunk fields into Chroma-compatible primitive metadata."""
        return {
            "document_id": str(chunk.document_id),
            "document_title": str(chunk.document_title),
            "act_name": str(chunk.act_name or ""),
            "section_number": str(chunk.section_number or ""),
            "section_title": str(chunk.section_title or ""),
            "subsection_id": str(chunk.subsection_id or ""),
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "source_url": str(chunk.source_url),
            "publication_date": str(chunk.publication_date or ""),
            "effective_date": str(chunk.effective_date or ""),
            "document_version": str(chunk.document_version or ""),
            "content_hash": str(chunk.content_hash),
            "char_count": int(chunk.char_count),
            "is_split_fallback": bool(chunk.is_split_fallback),
        }

    def index_chunks(
        self,
        chunks: List[LegalChunk],
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> int:
        """
        Idempotently index a list of LegalChunk models into ChromaDB.
        Re-indexes existing IDs safely via upsert.
        """
        if not chunks:
            return 0

        col = self.get_collection()
        total = len(chunks)
        indexed_count = 0

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.legal_text for c in batch]
            metadatas = [self._sanitize_metadata(c) for c in batch]

            # Generate normalized vector embeddings
            embeddings = self.embedder.encode_passages(documents)

            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            indexed_count += len(batch)
            if show_progress:
                logger.info("Indexed %d / %d chunks into ChromaDB", indexed_count, total)

        return indexed_count

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the collection using cosine similarity.
        Returns a list of result dictionaries with similarity scores.
        """
        if self.count() == 0:
            return []

        col = self.get_collection()
        query_vector = self.embedder.encode_text(query_text)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = col.query(**kwargs)

        output: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance: d in [0, 2], where 0 is identical
            # Similarity score: 1 - (d / 2) or standard 1 - d clamped to [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            output.append(
                {
                    "chunk_id": chunk_id,
                    "legal_text": doc_text,
                    "metadata": meta,
                    "score": round(similarity, 4),
                    "distance": float(dist),
                }
            )

        return output
