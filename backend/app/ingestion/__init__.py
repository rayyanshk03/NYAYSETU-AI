"""NyayaSetu AI - Legal Document Ingestion and Knowledge Base Pipeline.

This package provides a clean, modular, legal-aware pipeline for:
- PDF text extraction with PyMuPDF (extract.py)
- Conservative legal text cleaning and header/footer removal (clean.py)
- Section- and subsection-preserving legal chunking (chunk.py)
- Source manifest tracking and metadata management (metadata.py)
- Pipeline orchestration and batch CLI processing (pipeline.py)
"""

from .extract import PDFExtractor, ExtractionResult, PageExtraction
from .clean import TextCleaner, CleanedPage
from .chunk import LegalChunker, LegalChunk
from .metadata import ManifestManager, DocumentMetadata, ProcessingStatus
from .pipeline import IngestionPipeline, IngestionSummary

__all__ = [
    "PDFExtractor",
    "ExtractionResult",
    "PageExtraction",
    "TextCleaner",
    "CleanedPage",
    "LegalChunker",
    "LegalChunk",
    "ManifestManager",
    "DocumentMetadata",
    "ProcessingStatus",
    "IngestionPipeline",
    "IngestionSummary",
]
