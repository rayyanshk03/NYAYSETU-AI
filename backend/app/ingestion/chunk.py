"""Legal-aware document chunking module.

Preserves complete legal sections and subsections, accurately identifies
statutory numbering and titles without hallucination, and provides configurable
fallback splitting with overlap for unusually long provisions.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from app.ingestion.clean import CleanedPage
from app.ingestion.metadata import DocumentMetadata


class LegalChunk(BaseModel):
    """Structured legal chunk preserving section boundaries and provenance."""

    chunk_id: str = Field(..., description="Unique deterministic identifier for this chunk")
    document_id: str = Field(..., description="Parent document identifier")
    document_title: str = Field(..., description="Full title of the legal statute")
    act_name: Optional[str] = Field(default=None, description="Short statute name")
    section_number: Optional[str] = Field(
        default=None,
        description="Statutory section number, e.g. '13B', '125', or None for preamble",
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Extracted section heading or None if not present in text",
    )
    subsection_id: Optional[str] = Field(
        default=None,
        description="Subsection or paragraph identifier, e.g. '(1)', '(2)(a)'",
    )
    page_start: int = Field(..., description="Starting 1-indexed page number")
    page_end: int = Field(..., description="Ending 1-indexed page number")
    source_url: str = Field(..., description="Authoritative source URL from manifest")
    document_version: Optional[str] = Field(default=None, description="Document version")
    publication_date: Optional[str] = Field(default=None, description="Enactment/publication date")
    effective_date: Optional[str] = Field(default=None, description="Effective date if verified")
    legal_text: str = Field(..., description="Cleaned statutory legal text")
    raw_text: Optional[str] = Field(default=None, description="Original uncleaned text slice")
    content_hash: str = Field(..., description="SHA-256 hash of legal_text for change detection")
    char_count: int = Field(..., description="Total character length")
    token_estimate: int = Field(..., description="Estimated token count")
    is_split_fallback: bool = Field(
        default=False,
        description="True if an unusually long section was split across chunks",
    )


class LegalChunker:
    """Chunker that respects Indian statutory section and subsection structure."""

    # Matches Section headers typical in India Code bare acts:
    # E.g. "13B. Divorce by mutual consent.—"
    #      "Section 125. Order for maintenance of wives, children and parents.—"
    #      "24. Maintenance pendente lite..."
    # Group 1: Section Number
    # Group 2: Section Title (optional)
    SECTION_HEADER_REGEX = re.compile(
        r"(?m)^(?:(?:Section|Sec\.)\s+)?(\d+[A-Z]?)\.\s*(?:([A-Za-z][^\n—\.\:]*?)(?:[—\.\:]|\s*\n))?",
    )

    # Subsection markers: (1), (2), (a), (b), (i), (ii), Provided, Explanation
    SUBSECTION_SPLIT_REGEX = re.compile(
        r"(?m)\n(?=(?:\([0-9a-zA-Z]+\)|\bProvided\s+(?:that|further)|\bExplanation|\bException))"
    )

    def __init__(self, max_chunk_size: int = 1800, chunk_overlap: int = 200):
        """
        Initialize the legal chunker.

        Args:
            max_chunk_size: Maximum characters per chunk before triggering fallback split.
            chunk_overlap: Overlap in characters when fallback splitting long sections.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        cleaned_pages: List[CleanedPage],
        metadata: DocumentMetadata,
    ) -> List[LegalChunk]:
        """
        Convert cleaned document pages into legal-aware chunks.

        Short sections are kept intact in single chunks. Long sections are split
        at subsection boundaries with overlap.
        """
        if not cleaned_pages:
            return []

        # 1. Build document text with page offset mapping
        full_text, page_offsets = self._build_offset_map(cleaned_pages)
        if not full_text.strip():
            return []

        # 2. Locate all section boundaries
        sections = self._find_sections(full_text)

        chunks: List[LegalChunk] = []
        chunk_counter = 0

        if not sections:
            # Document has no identifiable sections (e.g. short preamble, order, or notice)
            doc_chunks = self._chunk_unsectioned_text(
                full_text,
                page_offsets,
                metadata,
                chunk_counter,
            )
            return doc_chunks

        # 3. Handle preamble/text before the first section
        first_sec_start = sections[0]["start"]
        if first_sec_start > 0:
            preamble_text = full_text[:first_sec_start].strip()
            if preamble_text:
                preamble_chunks = self._chunk_arbitrary_text(
                    text=preamble_text,
                    char_offset_in_doc=0,
                    page_offsets=page_offsets,
                    metadata=metadata,
                    section_number=None,
                    section_title="Preamble / Preliminary",
                    chunk_counter_start=chunk_counter,
                )
                chunks.extend(preamble_chunks)
                chunk_counter += len(preamble_chunks)

        # 4. Process each identified statutory section
        for i, sec in enumerate(sections):
            sec_start = sec["start"]
            sec_end = sections[i + 1]["start"] if i + 1 < len(sections) else len(full_text)
            sec_text = full_text[sec_start:sec_end].strip()

            if not sec_text:
                continue

            sec_num = sec["section_number"]
            sec_title = sec["section_title"]

            if len(sec_text) <= self.max_chunk_size:
                # Section fits in a single chunk - keep it whole!
                chunk_counter += 1
                page_start, page_end = self._get_page_range(sec_start, sec_end, page_offsets)
                chunk_id = f"{metadata.document_id}_sec_{sec_num}_{chunk_counter}"
                chunks.append(
                    self._create_chunk(
                        chunk_id=chunk_id,
                        text=sec_text,
                        metadata=metadata,
                        section_number=sec_num,
                        section_title=sec_title,
                        subsection_id=None,
                        page_start=page_start,
                        page_end=page_end,
                        is_split=False,
                    )
                )
            else:
                # Long section: split at subsection boundaries
                split_chunks = self._split_long_section(
                    sec_text=sec_text,
                    sec_start_in_doc=sec_start,
                    page_offsets=page_offsets,
                    metadata=metadata,
                    section_number=sec_num,
                    section_title=sec_title,
                    chunk_counter_start=chunk_counter,
                )
                chunks.extend(split_chunks)
                chunk_counter += len(split_chunks)

        return chunks

    def _find_sections(self, text: str) -> List[dict]:
        """Find all statutory section headers and extract number and title."""
        sections = []
        for match in self.SECTION_HEADER_REGEX.finditer(text):
            sec_num = match.group(1).strip()
            raw_title = match.group(2)
            sec_title = raw_title.strip() if raw_title else None

            # Filter false positives: ignore if title is just a single number or symbol
            if sec_title and len(sec_title) < 2:
                sec_title = None

            sections.append(
                {
                    "start": match.start(),
                    "section_number": sec_num,
                    "section_title": sec_title,
                }
            )
        return sections

    def _split_long_section(
        self,
        sec_text: str,
        sec_start_in_doc: int,
        page_offsets: List[Tuple[int, int, int]],
        metadata: DocumentMetadata,
        section_number: str,
        section_title: Optional[str],
        chunk_counter_start: int,
    ) -> List[LegalChunk]:
        """Split an unusually long section at subsection boundaries with overlap."""
        # Try splitting at subsection boundaries
        subsections = self.SUBSECTION_SPLIT_REGEX.split(sec_text)
        subsections = [s.strip() for s in subsections if s.strip()]

        chunks: List[LegalChunk] = []
        current_chunk_parts: List[str] = []
        current_len = 0
        counter = chunk_counter_start

        # Detect subsection label on first line, e.g. (1), (2)
        def get_sub_id(s: str) -> Optional[str]:
            m = re.match(r"^(\([0-9a-zA-Z]+\))", s)
            return m.group(1) if m else None

        current_sub_id = None

        for part in subsections:
            part_len = len(part)

            if current_len + part_len + 1 > self.max_chunk_size and current_chunk_parts:
                # Emit accumulated chunk
                combined_text = "\n\n".join(current_chunk_parts)
                counter += 1
                chunk_id = f"{metadata.document_id}_sec_{section_number}_part_{counter}"
                # Compute page range
                p_start, p_end = self._get_page_range(
                    sec_start_in_doc,
                    sec_start_in_doc + len(sec_text),
                    page_offsets,
                )
                chunks.append(
                    self._create_chunk(
                        chunk_id=chunk_id,
                        text=combined_text,
                        metadata=metadata,
                        section_number=section_number,
                        section_title=section_title,
                        subsection_id=current_sub_id,
                        page_start=p_start,
                        page_end=p_end,
                        is_split=True,
                    )
                )
                # Overlap: keep the last item if reasonable
                if len(current_chunk_parts) > 1 and len(current_chunk_parts[-1]) < self.chunk_overlap:
                    current_chunk_parts = [current_chunk_parts[-1], part]
                    current_len = sum(len(p) for p in current_chunk_parts) + 1
                else:
                    current_chunk_parts = [part]
                    current_len = part_len
                current_sub_id = get_sub_id(part)
            else:
                if not current_chunk_parts:
                    current_sub_id = get_sub_id(part)
                current_chunk_parts.append(part)
                current_len += part_len + 1

        if current_chunk_parts:
            combined_text = "\n\n".join(current_chunk_parts)
            counter += 1
            chunk_id = f"{metadata.document_id}_sec_{section_number}_part_{counter}"
            p_start, p_end = self._get_page_range(
                sec_start_in_doc,
                sec_start_in_doc + len(sec_text),
                page_offsets,
            )
            chunks.append(
                self._create_chunk(
                    chunk_id=chunk_id,
                    text=combined_text,
                    metadata=metadata,
                    section_number=section_number,
                    section_title=section_title,
                    subsection_id=current_sub_id,
                    page_start=p_start,
                    page_end=p_end,
                    is_split=True,
                )
            )

        return chunks

    def _chunk_arbitrary_text(
        self,
        text: str,
        char_offset_in_doc: int,
        page_offsets: List[Tuple[int, int, int]],
        metadata: DocumentMetadata,
        section_number: Optional[str],
        section_title: Optional[str],
        chunk_counter_start: int,
    ) -> List[LegalChunk]:
        """Chunk arbitrary paragraphs within size limits."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[LegalChunk] = []
        current_parts: List[str] = []
        current_len = 0
        counter = chunk_counter_start

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len + 2 > self.max_chunk_size and current_parts:
                combined = "\n\n".join(current_parts)
                counter += 1
                chunk_id = f"{metadata.document_id}_p{counter}"
                p_start, p_end = self._get_page_range(
                    char_offset_in_doc,
                    char_offset_in_doc + len(combined),
                    page_offsets,
                )
                chunks.append(
                    self._create_chunk(
                        chunk_id=chunk_id,
                        text=combined,
                        metadata=metadata,
                        section_number=section_number,
                        section_title=section_title,
                        subsection_id=None,
                        page_start=p_start,
                        page_end=p_end,
                        is_split=True,
                    )
                )
                current_parts = [para]
                current_len = para_len
            else:
                current_parts.append(para)
                current_len += para_len + 2

        if current_parts:
            combined = "\n\n".join(current_parts)
            counter += 1
            chunk_id = f"{metadata.document_id}_p{counter}"
            p_start, p_end = self._get_page_range(
                char_offset_in_doc,
                char_offset_in_doc + len(combined),
                page_offsets,
            )
            chunks.append(
                self._create_chunk(
                    chunk_id=chunk_id,
                    text=combined,
                    metadata=metadata,
                    section_number=section_number,
                    section_title=section_title,
                    subsection_id=None,
                    page_start=p_start,
                    page_end=p_end,
                    is_split=False,
                )
            )

        return chunks

    def _chunk_unsectioned_text(
        self,
        full_text: str,
        page_offsets: List[Tuple[int, int, int]],
        metadata: DocumentMetadata,
        chunk_counter_start: int,
    ) -> List[LegalChunk]:
        """Fallback chunking when no statutory sections are detected."""
        return self._chunk_arbitrary_text(
            text=full_text,
            char_offset_in_doc=0,
            page_offsets=page_offsets,
            metadata=metadata,
            section_number=None,
            section_title=None,
            chunk_counter_start=chunk_counter_start,
        )

    def _create_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
        section_number: Optional[str],
        section_title: Optional[str],
        subsection_id: Optional[str],
        page_start: int,
        page_end: int,
        is_split: bool,
    ) -> LegalChunk:
        """Create a validated LegalChunk model instance."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        char_count = len(text)
        token_estimate = int(len(text.split()) * 1.33)

        return LegalChunk(
            chunk_id=chunk_id,
            document_id=metadata.document_id,
            document_title=metadata.title,
            act_name=metadata.act_name,
            section_number=section_number,
            section_title=section_title,
            subsection_id=subsection_id,
            page_start=page_start,
            page_end=page_end,
            source_url=metadata.official_source_url,
            document_version=metadata.document_version,
            publication_date=metadata.publication_date,
            effective_date=metadata.effective_date,
            legal_text=text,
            content_hash=content_hash,
            char_count=char_count,
            token_estimate=token_estimate,
            is_split_fallback=is_split,
        )

    def _build_offset_map(
        self,
        cleaned_pages: List[CleanedPage],
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Concatenate page texts and build offset index.

        Returns:
            Tuple of (full_text, [(page_number, start_offset, end_offset), ...])
        """
        full_text_parts = []
        page_offsets = []
        current_offset = 0

        for p in cleaned_pages:
            text = p.cleaned_text
            start = current_offset
            full_text_parts.append(text)
            current_offset += len(text)
            end = current_offset
            page_offsets.append((p.page_number, start, end))

            # Add inter-page separator
            full_text_parts.append("\n\n")
            current_offset += 2

        return "".join(full_text_parts), page_offsets

    def _get_page_range(
        self,
        start_char: int,
        end_char: int,
        page_offsets: List[Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        """Determine page_start and page_end for a character range."""
        if not page_offsets:
            return 1, 1

        p_start = page_offsets[0][0]
        p_end = page_offsets[-1][0]

        for page_num, s, e in page_offsets:
            if start_char >= s and start_char <= e:
                p_start = page_num
                break

        for page_num, s, e in reversed(page_offsets):
            if end_char >= s:
                p_end = page_num
                break

        if p_start > p_end:
            p_end = p_start

        return p_start, p_end
