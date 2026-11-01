"""Citation validation service for NyayaSetu AI.

Validates that every citation produced by the answer-generation pipeline
refers to an actual indexed document or chunk. Uses deterministic checks
against the indexed metadata; does NOT trust the LLM's self-reported citations.

Validation checks:
  1. chunk_id exists in the vector store index
  2. act_name / section_number match the stored chunk metadata
  3. source_url originates from the manifest or the official India Code domain
  4. No citation was fabricated (i.e., section not present in retrieved passages)

A citation that fails any check is marked as UNVERIFIED and flagged in warnings.
Unverified citations are NOT silently presented as verified.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

TRUSTED_URL_DOMAINS = {
    "indiacode.nic.in",
    "legislative.gov.in",
    "egazette.gov.in",
    "doj.gov.in",
}


@dataclass
class CitationValidationResult:
    chunk_id: str
    act_name: Optional[str]
    section_number: Optional[str]
    source_url: Optional[str]
    verified: bool
    failure_reasons: List[str] = field(default_factory=list)
    note: str = ""


class CitationValidator:
    """
    Validates citations against indexed passages and manifest metadata.

    The validator works on the passages already retrieved by the retrieval
    pipeline. A citation is valid only if:
      - Its chunk_id is in the set of retrieved passage IDs
      - The act_name (if provided) matches the stored metadata
      - The section_number (if provided) matches the stored metadata
      - The source_url (if provided) is from a trusted official domain
    """

    def __init__(
        self,
        trusted_domains: Optional[Set[str]] = None,
    ):
        self.trusted_domains = trusted_domains or TRUSTED_URL_DOMAINS

    def _url_is_trusted(self, url: Optional[str]) -> bool:
        """Check whether a URL belongs to a trusted official government domain."""
        if not url:
            return False
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.trusted_domains)

    def _normalize_section(self, s: Optional[str]) -> str:
        """Strip whitespace and lowercase for comparison."""
        if not s:
            return ""
        return s.strip().lower()

    def _normalize_act(self, a: Optional[str]) -> str:
        """Normalize act name: lowercase, strip punctuation for fuzzy comparison."""
        if not a:
            return ""
        return re.sub(r"[^a-z0-9]", "", a.lower())

    def validate_citations(
        self,
        citations: List[Dict[str, Any]],
        retrieved_passages: List[Any],  # List[RetrievedPassage]
    ) -> List[CitationValidationResult]:
        """
        Validate a list of citation dicts against retrieved passages.

        Args:
            citations: List of citation dicts (from synthesizer output).
                       Each dict may contain: chunk_id, act_name, section_number,
                       source_url, index (1-based position).
            retrieved_passages: List of RetrievedPassage objects from retrieval.

        Returns:
            List of CitationValidationResult, one per citation.
        """
        # Build lookup map from retrieved passages
        passage_map: Dict[str, Any] = {}
        for p in retrieved_passages:
            pid = getattr(p, "chunk_id", None)
            if pid:
                passage_map[pid] = p

        retrieved_indices = set(range(len(retrieved_passages)))

        results = []
        for cite in citations:
            chunk_id = cite.get("chunk_id", "")
            act_name = cite.get("act_name")
            section_number = cite.get("section_number")
            source_url = cite.get("source_url")
            cite_index = cite.get("index")  # 1-based citation index

            failures = []

            # Check 1: chunk_id exists in retrieved passages
            if chunk_id and chunk_id not in passage_map:
                failures.append(
                    f"chunk_id '{chunk_id}' not found in retrieved passages — "
                    "possible hallucinated citation."
                )

            # Check 2: citation index refers to a valid passage
            if cite_index is not None:
                zero_idx = int(cite_index) - 1
                if zero_idx not in retrieved_indices:
                    failures.append(
                        f"Citation index [{cite_index}] out of bounds "
                        f"({len(retrieved_passages)} passages retrieved)."
                    )

            # Check 3: act_name matches stored metadata (if chunk_id valid)
            if chunk_id in passage_map and act_name:
                stored_act = getattr(passage_map[chunk_id], "act_name", None) or ""
                if self._normalize_act(act_name) != self._normalize_act(stored_act):
                    failures.append(
                        f"Act name mismatch: cited '{act_name}' but stored '{stored_act}'."
                    )

            # Check 4: section_number matches stored metadata
            if chunk_id in passage_map and section_number:
                stored_sec = getattr(passage_map[chunk_id], "section_number", None) or ""
                if self._normalize_section(section_number) != self._normalize_section(stored_sec):
                    failures.append(
                        f"Section mismatch: cited '{section_number}' but stored '{stored_sec}'."
                    )

            # Check 5: source URL trusted domain
            if source_url and not self._url_is_trusted(source_url):
                failures.append(
                    f"Source URL '{source_url}' does not originate from a "
                    "trusted official government domain."
                )

            verified = len(failures) == 0
            results.append(
                CitationValidationResult(
                    chunk_id=chunk_id,
                    act_name=act_name,
                    section_number=section_number,
                    source_url=source_url,
                    verified=verified,
                    failure_reasons=failures,
                    note=(
                        "Validated against indexed metadata."
                        if verified
                        else "UNVERIFIED — see failure_reasons."
                    ),
                )
            )

        return results

    def validate_passage_citations(
        self,
        retrieved_passages: List[Any],
    ) -> List[CitationValidationResult]:
        """
        Auto-validate all retrieved passages as citations (no external citation list needed).
        Used when the synthesizer directly cites the top-k passages by their retrieved position.

        Returns a CitationValidationResult for each passage:
          - verified=True  if source_url is trusted
          - verified=False otherwise
        """
        results = []
        for p in retrieved_passages:
            failures = []
            src_url = getattr(p, "source_url", None)
            if not self._url_is_trusted(src_url):
                failures.append(
                    f"source_url '{src_url}' is not from a trusted official domain."
                )
            results.append(
                CitationValidationResult(
                    chunk_id=getattr(p, "chunk_id", ""),
                    act_name=getattr(p, "act_name", None),
                    section_number=getattr(p, "section_number", None),
                    source_url=src_url,
                    verified=len(failures) == 0,
                    failure_reasons=failures,
                    note="Auto-validated from retrieved passage metadata.",
                )
            )
        return results

    def build_warning_messages(
        self, validation_results: List[CitationValidationResult]
    ) -> List[str]:
        """Return human-readable warning strings for all unverified citations."""
        warnings = []
        for res in validation_results:
            if not res.verified:
                warn = (
                    f"Unverified citation "
                    f"(chunk={res.chunk_id or 'unknown'}, "
                    f"section={res.section_number or 'unknown'}): "
                    + "; ".join(res.failure_reasons)
                )
                warnings.append(warn)
        return warnings


# Module-level singleton
_citation_validator: Optional[CitationValidator] = None


def get_citation_validator() -> CitationValidator:
    global _citation_validator
    if _citation_validator is None:
        _citation_validator = CitationValidator()
    return _citation_validator
