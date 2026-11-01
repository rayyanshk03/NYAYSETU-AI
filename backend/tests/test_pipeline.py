"""Tests for end-to-end ingestion pipeline execution and idempotency."""

import json
from pathlib import Path
import pymupdf
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.metadata import ProcessingStatus


def create_test_pdf(dest_path: Path, sections: list[tuple[str, str]]) -> None:
    """Helper to create a well-formed PDF with statutory sections."""
    doc = pymupdf.open()
    for title, body in sections:
        page = doc.new_page(width=595, height=842)
        content = f"{title}\n\n{body}"
        page.insert_text((50, 70), content, fontsize=11)
    doc.save(str(dest_path))
    doc.close()


def test_pipeline_empty_raw_directory(tmp_path: Path):
    """Verify pipeline handles empty raw directory gracefully without crashing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"

    pipeline = IngestionPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        manifest_dir=manifest_dir,
    )

    summary = pipeline.run()

    assert summary.total_raw_files_found == 0
    assert summary.processed_count == 0
    assert summary.failed_count == 0
    assert len(summary.warnings) > 0

    # Verify manifest was populated with canonical missing documents
    mgr = pipeline.manifest_mgr
    hma = mgr.get("hma_1955")
    assert hma is not None
    assert hma.processing_status == ProcessingStatus.MISSING_RAW_FILE


def test_pipeline_end_to_end_and_idempotency(tmp_path: Path):
    """Verify full end-to-end extraction, cleaning, chunking, and idempotent re-runs."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Place a canonical synthetic PDF in raw/
    pdf_path = raw_dir / "hindu_marriage_act_1955.pdf"
    sections = [
        (
            "1. Short title and extent.—",
            "(1) This Act may be called the Hindu Marriage Act, 1955.\n(2) It extends to the whole of India.",
        ),
        (
            "13B. Divorce by mutual consent.—",
            "(1) Subject to the provisions of this Act a petition for dissolution of marriage by a decree of divorce may be presented to the district court by both the parties to a marriage together.\n(2) On the motion of both the parties the court shall pass a decree of divorce.",
        ),
    ]
    create_test_pdf(pdf_path, sections)

    pipeline = IngestionPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        manifest_dir=manifest_dir,
    )

    # First Run
    summary1 = pipeline.run(force=False)
    assert summary1.total_raw_files_found == 1
    assert summary1.processed_count == 1
    assert summary1.skipped_unchanged_count == 0
    assert summary1.total_chunks_generated >= 2

    # Check output JSONL exists
    jsonl_file = processed_dir / "hma_1955.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 2
        first_chunk = json.loads(lines[0])
        assert first_chunk["document_id"] == "hma_1955"
        assert first_chunk["act_name"] == "Hindu Marriage Act, 1955"
        assert "content_hash" in first_chunk

    # Check manifest updated
    hma = pipeline.manifest_mgr.get("hma_1955")
    assert hma.processing_status == ProcessingStatus.PROCESSED
    assert hma.chunk_count == len(lines)
    assert hma.document_hash is not None

    # Second Run (Without Force -> must be skipped due to hash match!)
    summary2 = pipeline.run(force=False)
    assert summary2.total_raw_files_found == 1
    assert summary2.processed_count == 0
    assert summary2.skipped_unchanged_count == 1
    assert "hindu_marriage_act_1955.pdf" in summary2.documents_skipped

    # Third Run (With Force -> must re-process)
    summary3 = pipeline.run(force=True)
    assert summary3.total_raw_files_found == 1
    assert summary3.processed_count == 1
    assert summary3.skipped_unchanged_count == 0
