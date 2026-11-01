"""Tests for conservative legal text cleaning."""

from app.ingestion.clean import TextCleaner
from app.ingestion.extract import PageExtraction


def test_whitespace_and_newline_normalization():
    """Verify excessive spaces and carriage returns are normalized while structure is kept."""
    cleaner = TextCleaner()
    raw = "The   court   may,\r\n\r\n\r\n\r\nupon   application,\nmake  such  order."
    page = PageExtraction(page_number=1, raw_text=raw, char_count=len(raw), word_count=7)
    results = cleaner.clean_document_pages([page])
    assert len(results) == 1
    cleaned = results[0].cleaned_text
    assert "   " not in cleaned
    assert "\r" not in cleaned
    assert "The court may," in cleaned
    assert "make such order." in cleaned
    assert "\n\n" in cleaned  # Double newline preserved, not 4 newlines


def test_legal_section_numbering_preserved():
    """Verify section numbers, subsections, and marginal notes are intact."""
    cleaner = TextCleaner()
    raw = (
        "Section 12. Voidable marriages.—\n"
        "(1) Any marriage solemnized, whether before or after the commencement of this Act, shall be voidable.\n"
        "(a) that the marriage has not been consummated;\n"
        "Provided that no petition for annulling a marriage—\n"
        "Explanation.—For the purposes of this section, mental disorder means mental illness."
    )
    page = PageExtraction(page_number=1, raw_text=raw, char_count=len(raw), word_count=40)
    results = cleaner.clean_document_pages([page])
    cleaned = results[0].cleaned_text

    assert "Section 12. Voidable marriages.—" in cleaned
    assert "(1)" in cleaned
    assert "(a)" in cleaned
    assert "Provided that" in cleaned
    assert "Explanation.—" in cleaned


def test_dehyphenation_at_linebreaks():
    """Verify broken words are healed while compound terms remain hyphenated."""
    cleaner = TextCleaner()
    raw = (
        "The interim mainte-\nnance order was passed.\n"
        "In accordance with sub-\nsection (2) of the Act."
    )
    page = PageExtraction(page_number=1, raw_text=raw, char_count=len(raw), word_count=15)
    results = cleaner.clean_document_pages([page])
    cleaned = results[0].cleaned_text

    assert "maintenance" in cleaned
    assert "sub-section" in cleaned  # Compound legal term preserved with hyphen


def test_running_header_footer_detection_and_removal():
    """Verify lines repeating across >= 40% and >= 3 pages as header/footer are removed."""
    cleaner = TextCleaner()
    header_text = "THE SYNTHETIC FAMILY ACT, 2026"
    footer_text = "Page 1 of 5"

    pages = []
    for i in range(1, 6):
        raw_page = (
            f"{header_text}\n\n"
            f"Section {i}. Sample Statutory Provision {i}.—\n"
            f"Text for section {i} goes here.\n\n"
            f"{footer_text}"
        )
        pages.append(
            PageExtraction(
                page_number=i,
                raw_text=raw_page,
                char_count=len(raw_page),
                word_count=15,
            )
        )

    results = cleaner.clean_document_pages(pages)
    assert len(results) == 5

    # Headers and footers removed from cleaned text
    for res in results:
        assert header_text not in res.cleaned_text
        assert footer_text not in res.cleaned_text
        assert res.removed_header == header_text
        # Raw text remains unchanged
        assert header_text in res.raw_text
        assert footer_text in res.raw_text


def test_empty_page_handling():
    """Verify empty or whitespace-only pages are handled gracefully without error."""
    cleaner = TextCleaner()
    page = PageExtraction(page_number=1, raw_text="   \n\t  ", char_count=0, word_count=0)
    results = cleaner.clean_document_pages([page])
    assert len(results) == 1
    assert results[0].cleaned_text == ""
    assert "Empty page text" in results[0].warnings
