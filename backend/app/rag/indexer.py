"""Legal knowledge base indexer.

Reads processed JSONL files from Step 2 ingestion, validates chunk structures,
and indexes passages into ChromaDB with vector embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.config import settings
from app.ingestion.chunk import LegalChunk
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class IndexingSummary(BaseModel):
    """Summary statistics for a knowledge base indexing run."""

    files_found: int = Field(default=0)
    total_valid_chunks: int = Field(default=0)
    total_indexed: int = Field(default=0)
    invalid_chunks: int = Field(default=0)
    collection_total_after: int = Field(default=0)
    indexed_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class LegalCorpusIndexer:
    """Orchestrates indexing of processed legal chunks into ChromaDB."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.processed_dir = Path(processed_dir) if processed_dir else (base_dir / settings.PROCESSED_DATA_DIR)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store or ChromaVectorStore()

    def load_chunks_from_file(self, file_path: Path) -> Tuple[List[LegalChunk], int]:
        """
        Parse and validate LegalChunk records from a single JSONL file.
        Returns: (valid_chunks, invalid_count)
        """
        valid_chunks: List[LegalChunk] = []
        invalid_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                    chunk = LegalChunk.model_validate(data)
                    valid_chunks.append(chunk)
                except Exception as e:
                    invalid_count += 1
                    logger.warning("Invalid chunk schema at %s:%d: %s", file_path.name, line_no, e)

        return valid_chunks, invalid_count

    def index(
        self,
        reset: bool = False,
        target_file: Optional[str] = None,
        batch_size: int = 50,
    ) -> IndexingSummary:
        """
        Scan processed directory, validate chunks, and index into ChromaDB.

        Args:
            reset: If True, reset/rebuild the Chroma collection before indexing.
            target_file: Optional specific JSONL filename to index.
            batch_size: Embedding and upsert batch size.
        """
        summary = IndexingSummary()

        if reset:
            print("\n🔄 Resetting ChromaDB collection before re-indexing...")
            self.vector_store.reset_collection()

        # Locate processed JSONL files
        all_jsonl = sorted(list(self.processed_dir.glob("*.jsonl")))

        if target_file:
            all_jsonl = [f for f in all_jsonl if f.name == target_file]
        elif (self.processed_dir / "all_legal_chunks.jsonl").exists():
            # If consolidated file exists, use it exclusively to prevent duplicate chunk loading
            all_jsonl = [self.processed_dir / "all_legal_chunks.jsonl"]

        summary.files_found = len(all_jsonl)

        if not all_jsonl:
            print(f"\n{'='*75}")
            print(" NyayaSetu AI — Legal Knowledge Base Indexer (Step 3)")
            print(f"{'='*75}")
            print(f" Notice: No processed legal chunk files (*.jsonl) found in:")
            print(f"         {self.processed_dir}")
            print("\n To prepare and ingest official legal documents, first run:")
            print("         python -m app.ingestion.pipeline")
            print(f"{'='*75}\n")
            summary.errors.append("No processed JSONL files found in data/processed/.")
            summary.collection_total_after = self.vector_store.count()
            return summary

        print(f"\n{'='*75}")
        print(" NyayaSetu AI — Legal Knowledge Base Indexer")
        print(f"{'='*75}")
        print(f" Processed Dir : {self.processed_dir}")
        print(f" Target Files  : {[f.name for f in all_jsonl]}")
        print(f" Vector Store  : ChromaDB ({self.vector_store.collection_name})")
        print(f" Reset Mode    : {'Yes (Clean Rebuild)' if reset else 'No (Idempotent Upsert)'}")
        print(f"{'-'*75}\n")

        all_valid_chunks: List[LegalChunk] = []

        for jsonl_path in all_jsonl:
            chunks, invalid_count = self.load_chunks_from_file(jsonl_path)
            summary.total_valid_chunks += len(chunks)
            summary.invalid_chunks += invalid_count
            summary.indexed_files.append(jsonl_path.name)
            all_valid_chunks.extend(chunks)
            print(f"📄 Loaded {len(chunks)} valid chunks from {jsonl_path.name} ({invalid_count} errors)")

        if not all_valid_chunks:
            print("\n⚠️ No valid legal chunks found to index.")
            summary.collection_total_after = self.vector_store.count()
            return summary

        # Deduplicate chunks by chunk_id
        unique_chunks_dict = {c.chunk_id: c for c in all_valid_chunks}
        unique_chunks = list(unique_chunks_dict.values())

        print(f"\n🚀 Indexing {len(unique_chunks)} unique legal chunks into ChromaDB...")
        indexed_count = self.vector_store.index_chunks(unique_chunks, batch_size=batch_size)

        summary.total_indexed = indexed_count
        summary.collection_total_after = self.vector_store.count()

        print(f"\n{'='*75}")
        print(" Indexing Complete")
        print(f"{'='*75}")
        print(f" Chunks Read     : {summary.total_valid_chunks}")
        print(f" Chunks Indexed  : {summary.total_indexed}")
        print(f" Collection Size : {summary.collection_total_after} records")
        print(f"{'='*75}\n")

        return summary


def main():
    """Command-line entrypoint for indexing legal chunks."""
    parser = argparse.ArgumentParser(
        description="NyayaSetu AI — Legal Knowledge Base Indexer into ChromaDB"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and rebuild the Chroma collection before indexing",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional specific JSONL file in data/processed/ to index",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Custom path to processed JSONL directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for vector embedding and upsert",
    )

    args = parser.parse_args()

    indexer = LegalCorpusIndexer(
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
    )
    summary = indexer.index(
        reset=args.reset,
        target_file=args.file,
        batch_size=args.batch_size,
    )
    if summary.errors and summary.files_found == 0:
        sys.exit(0)  # Graceful exit for empty corpus


if __name__ == "__main__":
    main()
