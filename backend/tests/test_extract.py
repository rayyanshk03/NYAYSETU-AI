"""Tests for PyMuPDF PDF extraction module."""

from pathlib import Path
import pymupdf  # PyMuPDF
from app.ingestion.extract import PDFExtractor


def create_synthetic_pdf(tmp_path: Path, filename: str, pages_text: list[str]) -> Path:
    """Helper to generate a valid synthetic test PDF with known text."""
    pdf_path = tmp_path / filename
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        if text.strip():
            page.insert_text((50, 50), text, fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_extract_valid_multi_page_pdf(tmp_path):
    """Verify clean extraction across multiple pages with 1-indexed page numbers."""
    pages = [
        "Section 1. Short Title.\nThis is a synthetic legal act page one.",
        "Section 2. Definitions.\nThis is synthetic legal act page two.",
    ]
    pdf_path = create_synthetic_pdf(tmp_path, "synthetic_test.pdf", pages)

    extractor = PDFExtractor()
    result = extractor.extract(pdf_path, document_id="test_act")

    assert result.success is True
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "Section 1" in result.pages[0].raw_text
    assert result.pages[1].page_number == 2
    assert "Section 2" in result.pages[1].raw_text
    assert result.is_scanned_ocr_required is False
    assert len(result.document_hash) == 64


def test_extract_zero_byte_pdf(tmp_path):
    """Verify 0-byte empty file is handled gracefully without crashing."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    extractor = PDFExtractor()
    result = extractor.extract(empty_file, document_id="empty_doc")

    assert result.success is False
    assert "0 bytes" in result.error_message


def test_extract_corrupted_pdf(tmp_path):
    """Verify corrupted non-PDF bytes return success=False with clear error message."""
    corrupted_file = tmp_path / "corrupted.pdf"
    corrupted_file.write_bytes(b"%PDF-1.4\nCorrupted content completely invalid binary data")

    extractor = PDFExtractor()
    result = extractor.extract(corrupted_file, document_id="corrupt_doc")

    assert result.success is False
    assert "PDF extraction failed" in result.error_message


def test_scanned_pdf_detection(tmp_path):
    """Verify PDF with empty/scanned pages triggers is_scanned_ocr_required."""
    # Create 3 pages with no extractable text
    pdf_path = create_synthetic_pdf(tmp_path, "scanned_doc.pdf", ["", "", ""])

    extractor = PDFExtractor(min_chars_per_page=30)
    result = extractor.extract(pdf_path, document_id="scanned_doc")

    assert result.success is True
    assert result.page_count == 3
    assert result.is_scanned_ocr_required is True
    assert any("OCR processing is required" in w for w in result.warnings)


def test_nonexistent_file():
    """Verify missing file path returns success=False without crashing."""
    extractor = PDFExtractor()
    result = extractor.extract(Path("/nonexistent/path/doc.pdf"), document_id="ghost")
    assert result.success is False
    assert "File not found" in result.error_message
