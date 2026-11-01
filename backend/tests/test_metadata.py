"""Tests for source manifest and document metadata management."""

import json
from pathlib import Path
from app.ingestion.metadata import (
    ManifestManager,
    DocumentMetadata,
    ProcessingStatus,
    CANONICAL_FAMILY_LAW_CORPUS,
)


def test_canonical_corpus_registry():
    """Verify core Indian family law statutes are pre-registered with official India Code URLs."""
    expected_ids = {"hma_1955", "sma_1954", "hama_1956", "pwdva_2005", "gwa_1890", "crpc_sec_125", "fca_1984"}
    registered_ids = set(CANONICAL_FAMILY_LAW_CORPUS.keys())
    assert expected_ids.issubset(registered_ids)

    # Check HMA 1955 entry
    hma = CANONICAL_FAMILY_LAW_CORPUS["hma_1955"]
    assert "Hindu Marriage Act, 1955" in hma.title
    assert "https://www.indiacode.nic.in" in hma.official_source_url
    assert hma.publication_date == "1955-05-18"
    assert hma.local_filename == "hindu_marriage_act_1955.pdf"


def test_manifest_manager_save_and_reload(tmp_path: Path):
    """Verify manifest can be saved to both JSON and CSV and loaded back accurately."""
    mgr = ManifestManager(tmp_path)

    # Modify an entry
    hma = mgr.get("hma_1955")
    assert hma is not None
    hma.processing_status = ProcessingStatus.PROCESSED
    hma.page_count = 25
    hma.chunk_count = 38
    hma.document_hash = "abc123hash"
    hma.retrieval_date = "2026-10-30T00:00:00Z"
    mgr.upsert(hma)
    mgr.save()

    # Check files exist on disk
    json_path = tmp_path / "source_manifest.json"
    csv_path = tmp_path / "source_manifest.csv"
    assert json_path.exists()
    assert csv_path.exists()

    # Reload into a fresh manager instance
    mgr_reloaded = ManifestManager(tmp_path)
    reloaded_hma = mgr_reloaded.get("hma_1955")
    assert reloaded_hma.processing_status == ProcessingStatus.PROCESSED
    assert reloaded_hma.page_count == 25
    assert reloaded_hma.chunk_count == 38
    assert reloaded_hma.document_hash == "abc123hash"
    assert reloaded_hma.retrieval_date == "2026-10-30T00:00:00Z"
    # Ensure publication date was kept distinct from retrieval date
    assert reloaded_hma.publication_date == "1955-05-18"


def test_manifest_get_by_filename(tmp_path: Path):
    """Verify case-insensitive lookup by local filename."""
    mgr = ManifestManager(tmp_path)
    doc = mgr.get_by_filename("HINDU_MARRIAGE_ACT_1955.PDF")
    assert doc is not None
    assert doc.document_id == "hma_1955"


def test_missing_metadata_fields_defaults():
    """Verify DocumentMetadata assigns safe defaults when optional metadata is missing."""
    doc = DocumentMetadata(
        document_id="custom_doc",
        title="Custom Act",
        act_name="Custom Act",
        official_source_url="https://www.indiacode.nic.in/handle/123",
        local_filename="custom.pdf",
    )
    assert doc.processing_status == ProcessingStatus.PENDING
    assert doc.retrieval_date is None
    assert doc.publication_date is None
    assert doc.document_hash is None
    assert doc.chunk_count == 0
