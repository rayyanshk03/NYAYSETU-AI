"""Tests for legal-aware chunking boundaries and metadata preservation."""

from app.ingestion.clean import CleanedPage
from app.ingestion.chunk import LegalChunker
from app.ingestion.metadata import DocumentMetadata, ProcessingStatus


def make_metadata(doc_id="test_act_2026"):
    return DocumentMetadata(
        document_id=doc_id,
        title="The Test Family Code Act, 2026",
        act_name="Test Family Code, 2026",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/9999",
        local_filename="test_family_code_2026.pdf",
        source_authority="India Code Test Authority",
        publication_date="2026-01-01",
        effective_date="2026-01-15",
        document_version="Bare Act Edition 1",
        processing_status=ProcessingStatus.PENDING,
    )


def test_section_boundary_detection_and_no_unnecessary_split():
    """Verify short legal sections are kept intact in individual chunks."""
    chunker = LegalChunker(max_chunk_size=1000, chunk_overlap=100)
    meta = make_metadata()

    text_page_1 = (
        "PREAMBLE\nAn Act to amend and codify synthetic family provisions.\n\n"
        "1. Short title and extent.—(1) This Act may be called the Test Family Code, 2026.\n"
        "(2) It extends to the whole of India.\n\n"
        "2. Definitions.—In this Act, unless the context otherwise requires,—\n"
        "(a) 'court' means a principal civil court of original jurisdiction;\n"
        "(b) 'district' means the local limits of the jurisdiction of a principal civil court."
    )

    page = CleanedPage(
        page_number=1,
        raw_text=text_page_1,
        cleaned_text=text_page_1,
        char_count=len(text_page_1),
    )

    chunks = chunker.chunk_document([page], meta)

    # We expect: 1 chunk for preamble, 1 chunk for Section 1, 1 chunk for Section 2
    assert len(chunks) == 3

    sec_nums = [c.section_number for c in chunks]
    assert sec_nums == [None, "1", "2"]

    # Verify Section 1 chunk
    sec1_chunk = chunks[1]
    assert sec1_chunk.section_number == "1"
    assert sec1_chunk.section_title == "Short title and extent"
    assert "Test Family Code, 2026" in sec1_chunk.legal_text
    assert sec1_chunk.is_split_fallback is False
    assert sec1_chunk.content_hash is not None

    # Verify Section 2 chunk
    sec2_chunk = chunks[2]
    assert sec2_chunk.section_number == "2"
    assert sec2_chunk.section_title == "Definitions"
    assert "'court' means" in sec2_chunk.legal_text
    assert sec2_chunk.is_split_fallback is False


def test_long_section_fallback_splitting():
    """Verify unusually long sections exceeding max_chunk_size are split at subsections with overlap."""
    # Low max chunk size to force fallback split
    chunker = LegalChunker(max_chunk_size=200, chunk_overlap=30)
    meta = make_metadata()

    long_section = (
        "13. Divorce.—\n"
        "(1) Any marriage solemnized, whether before or after the commencement of this Act, may, on a petition presented by either the husband or the wife, be dissolved by decree of divorce on the ground that the other party has committed adultery.\n\n"
        "(2) A wife may also present a petition for the dissolution of her marriage on the ground that in the case of any marriage solemnized before the commencement of this Act, the husband had married again before such commencement.\n\n"
        "(3) The court may on being satisfied that the grounds have been proven pass a decree accordingly."
    )

    page = CleanedPage(
        page_number=1,
        raw_text=long_section,
        cleaned_text=long_section,
        char_count=len(long_section),
    )

    chunks = chunker.chunk_document([page], meta)

    # Must be split into multiple chunks
    assert len(chunks) > 1

    for c in chunks:
        assert c.section_number == "13"
        assert c.section_title == "Divorce"
        assert c.is_split_fallback is True
        assert len(c.legal_text) > 0


def test_non_invention_of_section_headings():
    """Verify section headings are None and NOT invented when omitted in text."""
    chunker = LegalChunker(max_chunk_size=1000)
    meta = make_metadata()

    # Section with number but no heading
    text = "25. (1) Any court exercising jurisdiction under this Act may direct payment of maintenance."
    page = CleanedPage(
        page_number=1,
        raw_text=text,
        cleaned_text=text,
        char_count=len(text),
    )

    chunks = chunker.chunk_document([page], meta)
    assert len(chunks) == 1
    assert chunks[0].section_number == "25"
    assert chunks[0].section_title is None  # Heading is not invented!


def test_metadata_preservation_in_chunk():
    """Verify all provenance fields are preserved in chunk."""
    chunker = LegalChunker()
    meta = make_metadata()
    text = "5. Conditions for a marriage.—A marriage may be solemnized between any two parties."
    page = CleanedPage(
        page_number=2,
        raw_text=text,
        cleaned_text=text,
        char_count=len(text),
    )

    chunks = chunker.chunk_document([page], meta)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == meta.document_id
    assert c.document_title == meta.title
    assert c.act_name == meta.act_name
    assert c.source_url == meta.official_source_url
    assert c.publication_date == "2026-01-01"
    assert c.effective_date == "2026-01-15"
    assert c.page_start == 2
    assert c.page_end == 2
    assert len(c.content_hash) == 64  # Valid SHA-256
