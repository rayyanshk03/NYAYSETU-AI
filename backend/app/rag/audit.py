"""Corpus audit module for NyayaSetu AI.

Inspects all raw PDFs in data/raw/, all processed JSONL chunks in
data/processed/, and the active ChromaDB collection to produce a
comprehensive audit report covering:

  - Document identification from PDF title pages
  - Text extraction success/failure
  - Legal section number and heading preservation
  - Empty pages and OCR-scan detection
  - Duplicate documents (identical byte checksums)
  - Duplicate chunks (identical content_hash across JSONL files)
  - Missing or uncertain source metadata
  - Manifest registration status
  - Amendment status caveat (all India Code PDFs are snapshots;
    the system cannot independently verify current legal applicability)

IMPORTANT: This audit does not verify that the legal content is
currently applicable or that amendments have been incorporated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Audit data classes ──────────────────────────────────────────────────────

@dataclass
class DocumentAuditRecord:
    filename: str
    file_size_bytes: int
    sha256_hash: Optional[str] = None
    page_count: int = 0
    empty_pages: int = 0
    estimated_text_chars: int = 0
    likely_scanned_ocr: bool = False
    extraction_success: bool = False
    extraction_error: Optional[str] = None
    detected_title: Optional[str] = None
    detected_act_number: Optional[str] = None
    chunk_count: int = 0
    section_numbers_detected: List[str] = field(default_factory=list)
    manifest_registered: bool = False
    manifest_document_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    # Amendment status — always flagged as unverified
    amendment_status_verified: bool = False
    amendment_note: str = (
        "Amendment status NOT independently verified. "
        "Document is a snapshot from India Code at retrieval date. "
        "Confirm applicability against the latest official gazette before legal use."
    )


@dataclass
class CorpusAuditReport:
    audit_timestamp: str = ""
    raw_dir: str = ""
    processed_dir: str = ""

    # Corpus totals
    total_raw_files: int = 0
    total_successfully_extracted: int = 0
    total_failed_or_partial: int = 0
    total_scanned_ocr: int = 0

    # Chunk statistics
    total_chunks_in_jsonl: int = 0
    total_duplicate_chunks: int = 0
    total_unique_chunks: int = 0

    # Metadata quality
    total_missing_section_metadata: int = 0
    total_missing_source_url: int = 0
    total_duplicate_documents: int = 0  # by byte-identical checksum

    # Documents needing review
    documents_requiring_review: List[str] = field(default_factory=list)

    # Indexed corpus
    indexed_chunk_count: int = 0

    # Per-document records
    document_records: List[DocumentAuditRecord] = field(default_factory=list)

    # Cross-document duplicate groups
    duplicate_hash_groups: Dict[str, List[str]] = field(default_factory=dict)

    # Notices
    notices: List[str] = field(default_factory=list)


# ── Audit engine ────────────────────────────────────────────────────────────

class CorpusAuditor:
    """Audits the legal corpus against extraction, quality, and metadata criteria."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        base = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = Path(raw_dir) if raw_dir else (base / "data" / "raw")
        self.processed_dir = Path(processed_dir) if processed_dir else (base / "data" / "processed")
        self.manifest_path = (
            Path(manifest_path) if manifest_path
            else (base / "data" / "manifests" / "source_manifest.json")
        )

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load source_manifest.json keyed by local_filename."""
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return {
                doc["local_filename"]: doc
                for doc in data.get("documents", [])
            }
        except Exception:
            return {}

    def _audit_pdf(self, path: Path, manifest: Dict[str, Any]) -> DocumentAuditRecord:
        """Inspect a single PDF file."""
        rec = DocumentAuditRecord(
            filename=path.name,
            file_size_bytes=path.stat().st_size,
        )

        # Compute SHA-256
        try:
            rec.sha256_hash = self._sha256(path)
        except Exception as e:
            rec.warnings.append(f"SHA-256 failed: {e}")

        # Check manifest
        manifest_entry = manifest.get(path.name)
        if manifest_entry:
            rec.manifest_registered = True
            rec.manifest_document_id = manifest_entry.get("document_id")
        else:
            rec.warnings.append("Not registered in source_manifest.json — metadata unverified.")

        # PyMuPDF extraction
        try:
            import fitz  # type: ignore  (pymupdf)
            doc = fitz.open(str(path))
            rec.page_count = len(doc)

            full_text_chars = 0
            empty_count = 0
            all_text = ""
            for pg_idx in range(rec.page_count):
                pg_text = doc[pg_idx].get_text()
                full_text_chars += len(pg_text.strip())
                if len(pg_text.strip()) < 10:
                    empty_count += 1
                if pg_idx < 5:  # first 5 pages for metadata heuristics
                    all_text += pg_text

            rec.empty_pages = empty_count
            rec.estimated_text_chars = full_text_chars
            rec.likely_scanned_ocr = full_text_chars < 200
            rec.extraction_success = full_text_chars > 50

            # Detect title from first page
            first_page_text = doc[0].get_text() if rec.page_count > 0 else ""
            title_match = re.search(
                r'The\s+[A-Z][A-Za-z\s,]+(?:Act|Code|Sanhita),\s+\d{4}',
                first_page_text,
                re.IGNORECASE,
            )
            if title_match:
                rec.detected_title = title_match.group(0).strip()

            # Detect act number
            act_num_match = re.search(r'ACT\s+NO\.\s+(\d+\s+OF\s+\d{4})', all_text, re.IGNORECASE)
            if act_num_match:
                rec.detected_act_number = act_num_match.group(1).strip()

            # Detect section numbers in first 5 pages
            section_matches = re.findall(r'\b(\d+[A-Z]?)\.\s+[A-Z]', all_text)
            rec.section_numbers_detected = list(set(section_matches[:20]))

            doc.close()

        except Exception as e:
            rec.extraction_success = False
            rec.extraction_error = str(e)
            rec.warnings.append(f"PDF extraction error: {e}")

        # Likely scanned flag
        if rec.likely_scanned_ocr:
            rec.warnings.append(
                "Likely scanned PDF (< 200 chars extracted). OCR may be needed for full text."
            )

        # Amendment caveat always applies
        rec.amendment_status_verified = False

        return rec

    def _load_chunk_stats(self, document_id: str) -> int:
        """Count chunks in the per-document JSONL file."""
        for suffix in ["", "_", "(1)"]:
            jsonl_path = self.processed_dir / f"{document_id}{suffix}.jsonl"
            if jsonl_path.exists():
                count = 0
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
                return count
        return 0

    def _audit_duplicate_chunks(self) -> tuple[int, int]:
        """
        Scan all JSONL files and count duplicate content hashes.
        Returns (total_chunks, duplicate_count).
        """
        all_hashes: List[str] = []
        jsonl_files = list(self.processed_dir.glob("*.jsonl"))
        for jf in jsonl_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            h = data.get("content_hash", "")
                            if h:
                                all_hashes.append(h)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass

        hash_counts = Counter(all_hashes)
        duplicates = sum(c - 1 for c in hash_counts.values() if c > 1)
        return len(all_hashes), duplicates

    def _count_missing_metadata(self) -> tuple[int, int]:
        """
        Scan all_legal_chunks.jsonl for missing section and source URL metadata.
        Returns (missing_section, missing_url).
        """
        combined = self.processed_dir / "all_legal_chunks.jsonl"
        if not combined.exists():
            return 0, 0

        missing_sec = 0
        missing_url = 0
        with open(combined, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not data.get("section_number"):
                        missing_sec += 1
                    src = data.get("source_url", "")
                    if not src or src == "https://www.indiacode.nic.in/":
                        missing_url += 1
                except json.JSONDecodeError:
                    pass
        return missing_sec, missing_url

    def run(self) -> CorpusAuditReport:
        """Execute the full corpus audit and return the report."""
        from datetime import datetime, timezone
        report = CorpusAuditReport(
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            raw_dir=str(self.raw_dir),
            processed_dir=str(self.processed_dir),
        )

        manifest = self._load_manifest()

        # 1. Discover PDFs
        pdf_paths = sorted(
            list(self.raw_dir.glob("*.pdf")) + list(self.raw_dir.glob("*.PDF"))
        )
        report.total_raw_files = len(pdf_paths)

        # 2. Audit each PDF
        sha256_to_files: Dict[str, List[str]] = defaultdict(list)
        for path in pdf_paths:
            rec = self._audit_pdf(path, manifest)

            # Look up chunk count from processed JSONL
            doc_id_guess = path.stem.lower().replace(" ", "_").replace("-", "_")
            if rec.manifest_document_id:
                doc_id_guess = rec.manifest_document_id
            rec.chunk_count = self._load_chunk_stats(doc_id_guess)

            if rec.sha256_hash:
                sha256_to_files[rec.sha256_hash].append(path.name)

            if rec.likely_scanned_ocr:
                report.total_scanned_ocr += 1
            if rec.extraction_success:
                report.total_successfully_extracted += 1
            else:
                report.total_failed_or_partial += 1

            # Flag for manual review
            needs_review = (
                not rec.manifest_registered
                or rec.likely_scanned_ocr
                or not rec.extraction_success
                or rec.chunk_count == 0
            )
            if needs_review:
                report.documents_requiring_review.append(path.name)

            report.document_records.append(rec)

        # 3. Duplicate documents by SHA-256
        for sha, files in sha256_to_files.items():
            if len(files) > 1:
                report.total_duplicate_documents += 1
                report.duplicate_hash_groups[sha[:16]] = files

        # 4. Duplicate chunks
        total_chunks, dup_chunks = self._audit_duplicate_chunks()
        report.total_chunks_in_jsonl = total_chunks
        report.total_duplicate_chunks = dup_chunks
        report.total_unique_chunks = total_chunks - dup_chunks

        # 5. Missing metadata
        miss_sec, miss_url = self._count_missing_metadata()
        report.total_missing_section_metadata = miss_sec
        report.total_missing_source_url = miss_url

        # 6. ChromaDB indexed count
        try:
            from app.rag.vector_store import ChromaVectorStore
            vs = ChromaVectorStore()
            report.indexed_chunk_count = vs.count()
        except Exception:
            report.indexed_chunk_count = -1

        # 7. Global notices
        report.notices.append(
            "AMENDMENT STATUS NOTICE: All documents are snapshots from India Code as of "
            "their retrieval date. The system cannot independently verify whether any "
            "subsequent amendments have been incorporated. Do not rely on these documents "
            "as authoritative current law without cross-referencing the official India Code "
            "portal (https://www.indiacode.nic.in/)."
        )
        if report.total_duplicate_documents:
            report.notices.append(
                f"{report.total_duplicate_documents} duplicate file(s) detected by SHA-256 checksum. "
                "Duplicate PDFs will produce duplicate chunks; consider removing duplicates."
            )

        return report

    def print_report(self, report: CorpusAuditReport) -> None:
        """Print the audit report to stdout in human-readable format."""
        print(f"\n{'='*75}")
        print(" NyayaSetu AI — Corpus Audit Report")
        print(f"{'='*75}")
        print(f" Audit Timestamp     : {report.audit_timestamp}")
        print(f" Raw Directory       : {report.raw_dir}")
        print(f" Processed Directory : {report.processed_dir}")
        print(f"{'-'*75}")
        print(f" Total Raw PDFs      : {report.total_raw_files}")
        print(f" Successfully Extracted: {report.total_successfully_extracted}")
        print(f" Failed / Partial    : {report.total_failed_or_partial}")
        print(f" Scanned / OCR Needed: {report.total_scanned_ocr}")
        print(f" Duplicate Documents : {report.total_duplicate_documents}")
        print(f"{'-'*75}")
        print(f" Total Chunks (JSONL): {report.total_chunks_in_jsonl}")
        print(f" Duplicate Chunks    : {report.total_duplicate_chunks}")
        print(f" Unique Chunks       : {report.total_unique_chunks}")
        print(f" Indexed in ChromaDB : {report.indexed_chunk_count}")
        print(f" Missing Sec Metadata: {report.total_missing_section_metadata}")
        print(f" Missing Source URLs : {report.total_missing_source_url}")
        print(f"{'-'*75}")

        if report.duplicate_hash_groups:
            print(" Duplicate File Groups:")
            for sha_prefix, files in report.duplicate_hash_groups.items():
                print(f"   SHA256[:{sha_prefix}…] → {files}")

        if report.documents_requiring_review:
            print(f"\n Documents Requiring Manual Review ({len(report.documents_requiring_review)}):")
            for fname in report.documents_requiring_review:
                print(f"   ⚠  {fname}")

        print(f"\n Per-Document Summary:")
        for rec in report.document_records:
            status = "✅" if rec.extraction_success and not rec.likely_scanned_ocr else "⚠️ "
            print(
                f"   {status} {rec.filename:<35} "
                f"pages={rec.page_count:>4}  chunks={rec.chunk_count:>4}  "
                f"{'SCANNED' if rec.likely_scanned_ocr else 'text'}"
            )
            if rec.detected_title:
                print(f"      → {rec.detected_title}")
            for w in rec.warnings[:2]:
                print(f"      ⚠  {w}")

        print(f"\n{'='*75}")
        print(" IMPORTANT NOTICES:")
        for notice in report.notices:
            print(f"  • {notice}")
        print(f"{'='*75}\n")


def main() -> None:
    """CLI entry point for corpus audit."""
    import argparse
    parser = argparse.ArgumentParser(
        description="NyayaSetu AI — Legal Corpus Audit"
    )
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to write JSON audit report")
    args = parser.parse_args()

    auditor = CorpusAuditor(
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    report = auditor.run()
    auditor.print_report(report)

    if args.json_out:
        import dataclasses
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(report), f, indent=2, ensure_ascii=False)
        print(f"JSON report saved to {args.json_out}")


if __name__ == "__main__":
    main()
