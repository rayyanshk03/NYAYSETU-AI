"""Legal document ingestion and knowledge base preparation pipeline.

Scans raw PDFs, performs extraction via PyMuPDF, cleans text conservatively,
generates legal-aware chunks, updates provenance manifests, and exports
structured JSONL records.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from app.ingestion.extract import PDFExtractor
from app.ingestion.clean import TextCleaner
from app.ingestion.chunk import LegalChunker, LegalChunk
from app.ingestion.metadata import (
    ManifestManager,
    DocumentMetadata,
    ProcessingStatus,
    CANONICAL_FAMILY_LAW_CORPUS,
)


class IngestionSummary(BaseModel):
    """Execution summary report for an ingestion pipeline run."""

    total_raw_files_found: int = Field(default=0)
    processed_count: int = Field(default=0)
    skipped_unchanged_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    scanned_ocr_needed_count: int = Field(default=0)
    total_chunks_generated: int = Field(default=0)
    documents_processed: List[str] = Field(default_factory=list)
    documents_skipped: List[str] = Field(default_factory=list)
    documents_failed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class IngestionPipeline:
    """Orchestrates end-to-end ingestion of Indian family law documents."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        manifest_dir: Optional[Path] = None,
        max_chunk_size: int = 1800,
        chunk_overlap: int = 200,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = Path(raw_dir) if raw_dir else (base_dir / "data" / "raw")
        self.processed_dir = Path(processed_dir) if processed_dir else (base_dir / "data" / "processed")
        self.manifest_dir = Path(manifest_dir) if manifest_dir else (base_dir / "data" / "manifests")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = LegalChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.manifest_mgr = ManifestManager(self.manifest_dir)

    def run(self, force: bool = False, target_filename: Optional[str] = None) -> IngestionSummary:
        """
        Execute the ingestion pipeline across all raw PDFs.

        Args:
            force: If True, re-process files even if their SHA-256 hash has not changed.
            target_filename: If provided, only process the specified file.
        """
        summary = IngestionSummary()

        # 1. Discover raw PDF files
        pdf_pattern = "*.pdf"
        raw_files = sorted(list(self.raw_dir.glob(pdf_pattern)) + list(self.raw_dir.glob("*.PDF")))

        if target_filename:
            raw_files = [f for f in raw_files if f.name == target_filename]

        summary.total_raw_files_found = len(raw_files)

        # 2. Check if raw directory is empty
        if not raw_files:
            self._handle_empty_raw_directory(summary)
            self.manifest_mgr.save()
            return summary

        print(f"\n{'='*75}")
        print(f" NyayaSetu AI — Legal Ingestion Pipeline")
        print(f"{'='*75}")
        print(f" Raw Directory       : {self.raw_dir}")
        print(f" Processed Directory : {self.processed_dir}")
        print(f" Manifest Directory  : {self.manifest_dir}")
        print(f" Raw PDFs Found      : {len(raw_files)}")
        print(f" Mode                : {'Force Re-process' if force else 'Incremental / Hash Check'}")
        print(f"{'-'*75}\n")

        all_generated_chunks: List[LegalChunk] = []

        # 3. Process each discovered PDF
        for file_path in raw_files:
            filename = file_path.name
            print(f"▶ Processing: {filename}")

            # Identify or create metadata
            doc_meta = self.manifest_mgr.get_by_filename(filename)
            if not doc_meta:
                # Unregistered PDF: register with custom ID
                doc_id = file_path.stem.lower().replace(" ", "_").replace("-", "_")
                doc_meta = DocumentMetadata(
                    document_id=doc_id,
                    title=f"Legal Document: {file_path.stem.replace('_', ' ').title()}",
                    act_name=file_path.stem.replace('_', ' ').title(),
                    official_source_url="https://www.indiacode.nic.in/",
                    local_filename=filename,
                    source_authority="Authoritative Official Portal",
                    document_version="Unregistered Bare Act PDF",
                )

            # Compute hash for change detection
            current_hash = PDFExtractor.compute_sha256(file_path)

            # Check idempotency
            if not force and doc_meta.document_hash == current_hash and doc_meta.processing_status == ProcessingStatus.PROCESSED:
                print(f"  ⏭ Skipped: File unchanged (SHA-256 match: {current_hash[:12]}...).")
                summary.skipped_unchanged_count += 1
                summary.documents_skipped.append(filename)
                continue

            # Extract text
            extract_result = self.extractor.extract(file_path, doc_meta.document_id)

            if not extract_result.success:
                print(f"  ❌ Extraction failed: {extract_result.error_message}")
                doc_meta.processing_status = ProcessingStatus.FAILED
                doc_meta.extraction_warnings = [extract_result.error_message or "Unknown failure"]
                doc_meta.document_hash = current_hash
                doc_meta.last_processed_at = datetime.now(timezone.utc).isoformat()
                self.manifest_mgr.upsert(doc_meta)
                summary.failed_count += 1
                summary.documents_failed.append(filename)
                continue

            if extract_result.is_scanned_ocr_required:
                print(f"  ⚠️ Scanned PDF detected: requires OCR engine to extract digital text.")
                doc_meta.processing_status = ProcessingStatus.SCANNED_OCR_REQUIRED
                doc_meta.extraction_warnings = extract_result.warnings
                doc_meta.document_hash = current_hash
                doc_meta.page_count = extract_result.page_count
                doc_meta.file_size_bytes = extract_result.file_size_bytes
                doc_meta.last_processed_at = datetime.now(timezone.utc).isoformat()
                self.manifest_mgr.upsert(doc_meta)
                summary.scanned_ocr_needed_count += 1
                summary.warnings.extend(extract_result.warnings)
                continue

            # Clean text
            cleaned_pages = self.cleaner.clean_document_pages(extract_result.pages)

            # Chunk text
            chunks = self.chunker.chunk_document(cleaned_pages, doc_meta)

            if not chunks:
                print(f"  ⚠️ No chunks produced (empty or unextractable content).")
                doc_meta.processing_status = ProcessingStatus.FAILED
                doc_meta.extraction_warnings = ["No chunks produced after cleaning."]
                doc_meta.document_hash = current_hash
                doc_meta.last_processed_at = datetime.now(timezone.utc).isoformat()
                self.manifest_mgr.upsert(doc_meta)
                summary.failed_count += 1
                summary.documents_failed.append(filename)
                continue

            # Write document JSONL
            out_jsonl_path = self.processed_dir / f"{doc_meta.document_id}.jsonl"
            with open(out_jsonl_path, "w", encoding="utf-8") as f:
                for chk in chunks:
                    f.write(chk.model_dump_json() + "\n")

            all_generated_chunks.extend(chunks)

            # Update document metadata
            doc_meta.processing_status = ProcessingStatus.PROCESSED
            doc_meta.document_hash = current_hash
            doc_meta.page_count = extract_result.page_count
            doc_meta.chunk_count = len(chunks)
            doc_meta.file_size_bytes = extract_result.file_size_bytes
            doc_meta.last_processed_at = datetime.now(timezone.utc).isoformat()
            doc_meta.retrieval_date = datetime.now(timezone.utc).isoformat() if not doc_meta.retrieval_date else doc_meta.retrieval_date
            doc_meta.extraction_warnings = extract_result.warnings

            self.manifest_mgr.upsert(doc_meta)

            print(f"  ✅ Completed: {extract_result.page_count} pages -> {len(chunks)} legal chunks saved to {out_jsonl_path.name}")
            summary.processed_count += 1
            summary.total_chunks_generated += len(chunks)
            summary.documents_processed.append(filename)

        # 4. Save updated manifest (JSON and CSV)
        self.manifest_mgr.save()

        # 5. Write combined JSONL for downstream RAG ingestion
        if all_generated_chunks:
            combined_path = self.processed_dir / "all_legal_chunks.jsonl"
            with open(combined_path, "w", encoding="utf-8") as f:
                for chk in all_generated_chunks:
                    f.write(chk.model_dump_json() + "\n")
            print(f"\n📦 Consolidated {len(all_generated_chunks)} chunks into {combined_path.name}")

        self._print_summary(summary)
        return summary

    def _handle_empty_raw_directory(self, summary: IngestionSummary) -> None:
        """Gracefully handle empty raw directory with guidance on adding legal PDFs."""
        print(f"\n{'='*75}")
        print(f" NyayaSetu AI — Legal Ingestion Pipeline (Step 2)")
        print(f"{'='*75}")
        print(f" Notice: No raw legal PDF documents found in:")
        print(f"         {self.raw_dir}")
        print(f"\n To prepare the knowledge base, please download official Indian Bare Acts")
        print(f" from India Code (https://www.indiacode.nic.in/) and place them in data/raw/:")
        print(f"{'-'*75}")

        for doc_id, meta in CANONICAL_FAMILY_LAW_CORPUS.items():
            print(f" • {meta.act_name}:")
            print(f"   Target Filename : {meta.local_filename}")
            print(f"   Official URL    : {meta.official_source_url}")
            # Mark status in manifest
            meta_copy = meta.model_copy()
            meta_copy.processing_status = ProcessingStatus.MISSING_RAW_FILE
            meta_copy.extraction_warnings = [
                f"PDF file '{meta.local_filename}' not yet placed in data/raw/. "
                f"Download from {meta.official_source_url}."
            ]
            self.manifest_mgr.upsert(meta_copy)

        print(f"{'-'*75}")
        print(f" Once files are placed, re-run: python -m app.ingestion.pipeline\n")
        summary.warnings.append("No raw PDF documents found. Registered canonical acts in manifest.")

    def _print_summary(self, summary: IngestionSummary) -> None:
        """Display summary statistics table."""
        print(f"\n{'='*75}")
        print(f" Ingestion Pipeline Summary")
        print(f"{'='*75}")
        print(f" Files Discovered     : {summary.total_raw_files_found}")
        print(f" Successfully Cleaned : {summary.processed_count}")
        print(f" Skipped (Unchanged)  : {summary.skipped_unchanged_count}")
        print(f" Scanned / Need OCR   : {summary.scanned_ocr_needed_count}")
        print(f" Failed               : {summary.failed_count}")
        print(f" Total Chunks Created : {summary.total_chunks_generated}")
        print(f" Manifest Saved To    : {self.manifest_dir / 'source_manifest.json'}")
        print(f" CSV Manifest Saved   : {self.manifest_dir / 'source_manifest.csv'}")
        print(f"{'='*75}\n")


def main():
    """Command-line entrypoint for running legal ingestion."""
    parser = argparse.ArgumentParser(
        description="NyayaSetu AI — Legal Document Ingestion & Knowledge Base Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Custom path to raw PDF directory (defaults to backend/data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Custom path to processed output directory (defaults to backend/data/processed)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Custom path to manifests directory (defaults to backend/data/manifests)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of documents even if SHA-256 hash has not changed",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process only a specific filename in raw_dir",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1800,
        help="Maximum characters per legal chunk before triggering fallback splitting (default: 1800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap when splitting unusually long sections (default: 200)",
    )

    args = parser.parse_args()

    pipeline = IngestionPipeline(
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    summary = pipeline.run(force=args.force, target_filename=args.file)
    if summary.failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
