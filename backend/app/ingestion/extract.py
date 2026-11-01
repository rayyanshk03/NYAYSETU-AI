"""PDF text extraction module using PyMuPDF.

Extracts text page-by-page, preserves 1-indexed page numbers, detects
scanned/image-only PDFs, handles corrupted files gracefully, and supports
optional OCR plugin hooks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, List, Optional
import pymupdf  # PyMuPDF
from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    """Raw text and metadata extracted from an individual PDF page."""

    page_number: int = Field(..., description="1-indexed physical page number")
    raw_text: str = Field(..., description="Raw extracted text directly from PyMuPDF")
    char_count: int = Field(default=0, description="Character count on this page")
    word_count: int = Field(default=0, description="Word count on this page")
    has_images: bool = Field(default=False, description="Whether page contains embedded images")
    is_likely_scanned: bool = Field(
        default=False,
        description="True if page contains images but very little or no extractable text",
    )


class ExtractionResult(BaseModel):
    """Complete result of extracting text from a single PDF document."""

    success: bool = Field(..., description="Whether extraction completed without fatal error")
    document_id: str = Field(..., description="Unique document identifier")
    file_path: str = Field(..., description="Path to source file on disk")
    document_hash: str = Field(default="", description="SHA-256 hash of the source file")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    page_count: int = Field(default=0, description="Total number of pages")
    total_chars: int = Field(default=0, description="Total characters extracted across all pages")
    pages: List[PageExtraction] = Field(default_factory=list, description="Extracted pages")
    warnings: List[str] = Field(default_factory=list, description="Extraction warnings and alerts")
    is_scanned_ocr_required: bool = Field(
        default=False,
        description="True if document appears to be a scanned image lacking digital text",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Fatal error message if extraction failed",
    )


class PDFExtractor:
    """Robust PDF text extractor built on PyMuPDF."""

    def __init__(
        self,
        min_chars_per_page: int = 35,
        ocr_engine: Optional[Callable[[pymupdf.Page], str]] = None,
    ):
        """
        Initialize the PDF extractor.

        Args:
            min_chars_per_page: Minimum character threshold below which a page is
                flagged as potentially scanned.
            ocr_engine: Optional callable hook to plug in an OCR engine (e.g. Tesseract)
                for scanned pages in future steps.
        """
        self.min_chars_per_page = min_chars_per_page
        self.ocr_engine = ocr_engine

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute the SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract(self, file_path: Path, document_id: str) -> ExtractionResult:
        """
        Extract text from a PDF file page by page.

        Gracefully handles errors, detects scanned PDFs, and records warnings
        without raising uncaught exceptions.
        """
        path = Path(file_path)
        warnings: List[str] = []

        if not path.exists():
            return ExtractionResult(
                success=False,
                document_id=document_id,
                file_path=str(path),
                error_message=f"File not found: {path}",
            )

        if not path.is_file():
            return ExtractionResult(
                success=False,
                document_id=document_id,
                file_path=str(path),
                error_message=f"Path is not a file: {path}",
            )

        try:
            file_size = path.stat().st_size
            doc_hash = self.compute_sha256(path)
        except Exception as e:
            return ExtractionResult(
                success=False,
                document_id=document_id,
                file_path=str(path),
                error_message=f"Failed reading file stats or computing hash: {e}",
            )

        if file_size == 0:
            return ExtractionResult(
                success=False,
                document_id=document_id,
                file_path=str(path),
                document_hash=doc_hash,
                file_size_bytes=0,
                error_message="PDF file is 0 bytes (empty file).",
            )

        doc = None
        try:
            doc = pymupdf.open(str(path))
            page_count = len(doc)

            if page_count == 0:
                warnings.append("PDF contains 0 pages.")
                return ExtractionResult(
                    success=True,
                    document_id=document_id,
                    file_path=str(path),
                    document_hash=doc_hash,
                    file_size_bytes=file_size,
                    page_count=0,
                    total_chars=0,
                    pages=[],
                    warnings=warnings,
                )

            extracted_pages: List[PageExtraction] = []
            total_chars = 0
            scanned_pages_count = 0

            for page_index in range(page_count):
                page_number = page_index + 1
                try:
                    page = doc[page_index]
                    text = page.get_text("text") or ""
                    char_count = len(text.strip())
                    word_count = len(text.split())

                    image_list = page.get_images(full=True)
                    has_images = len(image_list) > 0

                    is_likely_scanned = False
                    if char_count < self.min_chars_per_page:
                        # Very low text count
                        if has_images or char_count == 0:
                            is_likely_scanned = True
                            scanned_pages_count += 1

                    # Optional OCR hook for future expansion
                    if is_likely_scanned and self.ocr_engine is not None:
                        try:
                            ocr_text = self.ocr_engine(page)
                            if ocr_text:
                                text = ocr_text
                                char_count = len(text.strip())
                                word_count = len(text.split())
                                is_likely_scanned = False
                                warnings.append(f"Page {page_number}: Extracted text via OCR engine.")
                        except Exception as ocr_err:
                            warnings.append(f"Page {page_number}: OCR engine failed: {ocr_err}")

                    total_chars += char_count
                    extracted_pages.append(
                        PageExtraction(
                            page_number=page_number,
                            raw_text=text,
                            char_count=char_count,
                            word_count=word_count,
                            has_images=has_images,
                            is_likely_scanned=is_likely_scanned,
                        )
                    )
                except Exception as page_err:
                    warnings.append(f"Page {page_number}: Extraction error: {page_err}")
                    extracted_pages.append(
                        PageExtraction(
                            page_number=page_number,
                            raw_text="",
                            char_count=0,
                            word_count=0,
                            has_images=False,
                            is_likely_scanned=True,
                        )
                    )

            # Determine whether the document as a whole is scanned or lacking extractable text
            is_scanned_ocr_required = False
            avg_chars = (total_chars / page_count) if page_count > 0 else 0

            if page_count > 0 and (
                avg_chars < self.min_chars_per_page
                or scanned_pages_count == page_count
            ):
                is_scanned_ocr_required = True
                warnings.append(
                    f"Document appears to be scanned or contains non-extractable text "
                    f"(avg {avg_chars:.1f} chars/page across {page_count} pages). "
                    f"OCR processing is required to extract text."
                )
            elif scanned_pages_count > 0:
                warnings.append(
                    f"{scanned_pages_count} of {page_count} pages contain little/no extractable digital text."
                )

            return ExtractionResult(
                success=True,
                document_id=document_id,
                file_path=str(path),
                document_hash=doc_hash,
                file_size_bytes=file_size,
                page_count=page_count,
                total_chars=total_chars,
                pages=extracted_pages,
                warnings=warnings,
                is_scanned_ocr_required=is_scanned_ocr_required,
            )

        except Exception as e:
            # Catch corrupted PDFs, decryption failures, invalid headers
            return ExtractionResult(
                success=False,
                document_id=document_id,
                file_path=str(path),
                document_hash=doc_hash,
                file_size_bytes=file_size,
                error_message=f"PDF extraction failed (possible corrupted file): {e}",
            )
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass
