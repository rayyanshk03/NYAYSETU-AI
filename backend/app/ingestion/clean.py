"""Conservative legal text cleaning module.

Removes repeated running headers and footers across pages only when
reliably identified, normalizes whitespace, repairs word hyphens across linebreaks,
while strictly preserving legal numbering, subsection labels, statutory terminology,
and paragraph structure.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.ingestion.extract import PageExtraction


class CleanedPage(BaseModel):
    """Cleaned representation of a document page preserving raw source text."""

    page_number: int = Field(..., description="1-indexed page number")
    raw_text: str = Field(..., description="Original uncleaned extracted text")
    cleaned_text: str = Field(..., description="Conservatively cleaned text")
    char_count: int = Field(default=0, description="Cleaned character count")
    removed_header: Optional[str] = Field(
        default=None,
        description="Detected running header removed from page",
    )
    removed_footer: Optional[str] = Field(
        default=None,
        description="Detected running footer removed from page",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Cleaning notes or warnings",
    )


class TextCleaner:
    """Conservative legal text cleaner."""

    COMPOUND_LEGAL_TERMS = {
        "sub-section",
        "sub-sections",
        "sub-clause",
        "sub-clauses",
        "cross-examination",
        "ex-parte",
        "cooling-off",
        "quasi-judicial",
        "de-facto",
        "de-jure",
        "bona-fide",
    }

    PAGE_NUMBER_REGEX = re.compile(
        r"^[\s\-_—–]*\[?(?:Page\s+)?(?:\d+|\b[ivxlcdm]+\b)(?:\s*(?:of|/)\s*\d+)?\]?[\s\-_—–]*$",
        re.IGNORECASE,
    )

    def clean_document_pages(
        self,
        pages: List[PageExtraction],
    ) -> List[CleanedPage]:
        """
        Clean text across document pages, removing repeated headers/footers
        and normalizing whitespace while preserving legal structure.
        """
        if not pages:
            return []

        # Step 1: Detect candidate headers and footers across pages
        repeated_headers, repeated_footers = self._detect_repeated_headers_footers(pages)

        cleaned_pages: List[CleanedPage] = []

        for p in pages:
            page_text = p.raw_text
            warnings: List[str] = []
            removed_header = None
            removed_footer = None

            if not page_text.strip():
                cleaned_pages.append(
                    CleanedPage(
                        page_number=p.page_number,
                        raw_text=p.raw_text,
                        cleaned_text="",
                        char_count=0,
                        warnings=["Empty page text"],
                    )
                )
                continue

            raw_lines = page_text.splitlines()

            # Find boundary indices without dropping internal blank lines
            start_idx = 0
            while start_idx < len(raw_lines) and not raw_lines[start_idx].strip():
                start_idx += 1

            end_idx = len(raw_lines)
            while end_idx > start_idx and not raw_lines[end_idx - 1].strip():
                end_idx -= 1

            if start_idx >= end_idx:
                cleaned_pages.append(
                    CleanedPage(
                        page_number=p.page_number,
                        raw_text=p.raw_text,
                        cleaned_text="",
                        char_count=0,
                        warnings=["Empty page text"],
                    )
                )
                continue

            # Remove header if top line matches
            first_line_stripped = raw_lines[start_idx].strip()
            if first_line_stripped in repeated_headers:
                removed_header = first_line_stripped
                start_idx += 1
                while start_idx < end_idx and not raw_lines[start_idx].strip():
                    start_idx += 1

            # Remove footer if bottom line matches or matches page number pattern
            if end_idx > start_idx:
                last_line_stripped = raw_lines[end_idx - 1].strip()
                if last_line_stripped in repeated_footers or self.PAGE_NUMBER_REGEX.match(last_line_stripped):
                    removed_footer = last_line_stripped
                    end_idx -= 1
                    while end_idx > start_idx and not raw_lines[end_idx - 1].strip():
                        end_idx -= 1

            body_text = "\n".join(raw_lines[start_idx:end_idx])
            cleaned = self._clean_text(body_text)

            cleaned_pages.append(
                CleanedPage(
                    page_number=p.page_number,
                    raw_text=p.raw_text,
                    cleaned_text=cleaned,
                    char_count=len(cleaned),
                    removed_header=removed_header,
                    removed_footer=removed_footer,
                    warnings=warnings,
                )
            )

        return cleaned_pages

    def _detect_repeated_headers_footers(
        self,
        pages: List[PageExtraction],
    ) -> Tuple[Set[str], Set[str]]:
        """Identify candidate running headers and footers across pages."""
        num_pages = len(pages)
        if num_pages < 3:
            return set(), set()

        header_candidates = Counter()
        footer_candidates = Counter()

        for p in pages:
            lines = [line.strip() for line in p.raw_text.splitlines() if line.strip()]
            if lines:
                first_line = lines[0]
                # Avoid treating section headings as headers
                if not re.match(r"^(?:Section\s+\d+|\d+[A-Z]?\.)", first_line, re.IGNORECASE):
                    header_candidates[first_line] += 1

                last_line = lines[-1]
                footer_candidates[last_line] += 1

        min_threshold = max(3, int(num_pages * 0.40))

        headers_to_remove = {
            line for line, count in header_candidates.items() if count >= min_threshold
        }
        footers_to_remove = {
            line for line, count in footer_candidates.items() if count >= min_threshold
        }

        return headers_to_remove, footers_to_remove

    def _clean_text(self, text: str) -> str:
        """Apply conservative whitespace normalization and line repair."""
        if not text:
            return ""

        # 1. Normalize unicode spaces and control characters
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. De-hyphenate words broken at line boundaries
        def dehyphenate(match: re.Match) -> str:
            word_part1 = match.group(1)
            word_part2 = match.group(2)
            combined = f"{word_part1}-{word_part2}".lower()
            if combined in self.COMPOUND_LEGAL_TERMS:
                return f"{word_part1}-{word_part2}"
            return f"{word_part1}{word_part2}"

        text = re.sub(r"([A-Za-z]{2,})-\s*\n([A-Za-z]{2,})", dehyphenate, text)

        # 3. Collapse excessive consecutive horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # 4. Normalize paragraph breaks:
        # 3 or more newlines become double newlines (single empty line)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Soft line break healing:
        lines = text.split("\n")
        merged_lines: List[str] = []

        section_start_pattern = re.compile(
            r"^(?:"
            r"(?:Section\s+\d+[A-Z]?)"
            r"|(?:\d+[A-Z]?\.)"
            r"|(?:\([0-9a-zA-Z]+\))"
            r"|(?:Provided\s+(?:that|further))"
            r"|(?:Explanation(?:\s+[0-9IVX]+)?[\.\—\:])"
            r"|(?:Exception[\.\—\:])"
            r"|(?:CHAPTER\s+[0-9IVX]+)"
            r"|(?:PART\s+[0-9IVX]+)"
            r")",
            re.IGNORECASE,
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                # Blank line indicates paragraph boundary
                if merged_lines and merged_lines[-1] != "":
                    merged_lines.append("")
                continue

            if not merged_lines or merged_lines[-1] == "":
                merged_lines.append(line_str)
                continue

            prev_line = merged_lines[-1]

            ends_with_terminal = prev_line[-1] in {".", ";", ":", "?", "!", "—", "–"}
            starts_legal_unit = bool(section_start_pattern.match(line_str))

            if not ends_with_terminal and not starts_legal_unit:
                # Merge into previous line
                merged_lines[-1] = f"{prev_line} {line_str}"
            else:
                merged_lines.append(line_str)

        cleaned_text = "\n".join(merged_lines)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()
