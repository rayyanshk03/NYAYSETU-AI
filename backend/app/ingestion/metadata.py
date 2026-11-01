"""Source manifest and document metadata management.

Tracks provenance, official India Code URLs, legal publication dates,
retrieval dates, processing status, and document hashes.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Processing lifecycle states for legal documents."""

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    SCANNED_OCR_REQUIRED = "scanned_ocr_required"
    MISSING_RAW_FILE = "missing_raw_file"


class DocumentMetadata(BaseModel):
    """Structured legal document metadata tracking provenance and processing status."""

    document_id: str = Field(
        ...,
        description="Unique identifier for the statute, e.g. 'hma_1955'",
    )
    title: str = Field(
        ...,
        description="Full formal title of the statute or provision",
    )
    act_name: str = Field(
        ...,
        description="Short title or common act name, e.g. 'Hindu Marriage Act, 1955'",
    )
    official_source_url: str = Field(
        ...,
        description="Authoritative India Code or official government portal URL",
    )
    local_filename: str = Field(
        ...,
        description="Expected or actual local filename in data/raw/",
    )
    source_authority: str = Field(
        default="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        description="Official publishing or issuing authority",
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Official enactment or gazette publication date (YYYY-MM-DD)",
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Date the act or amendment came into force, if verified",
    )
    retrieval_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp when document was downloaded",
    )
    document_version: Optional[str] = Field(
        default="Official Bare Act",
        description="Document version or amendment edition",
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current ingestion status",
    )
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings encountered during extraction, cleaning, or chunking",
    )
    document_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the raw source file for change detection",
    )
    page_count: int = Field(default=0, description="Total pages in the source document")
    chunk_count: int = Field(default=0, description="Total chunks generated")
    file_size_bytes: int = Field(default=0, description="Raw file size in bytes")
    last_processed_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of last pipeline run",
    )


# Authoritative pre-registered Indian family law corpus registry
# Verified canonical entries on India Code (https://www.indiacode.nic.in)
CANONICAL_FAMILY_LAW_CORPUS: Dict[str, DocumentMetadata] = {
    "hma_1955": DocumentMetadata(
        document_id="hma_1955",
        title="The Hindu Marriage Act, 1955 (Act No. 25 of 1955)",
        act_name="Hindu Marriage Act, 1955",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/1567",
        local_filename="hindu_marriage_act_1955.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1955-05-18",
        effective_date="1955-05-18",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "sma_1954": DocumentMetadata(
        document_id="sma_1954",
        title="The Special Marriage Act, 1954 (Act No. 43 of 1954)",
        act_name="Special Marriage Act, 1954",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/1547",
        local_filename="special_marriage_act_1954.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1954-10-09",
        effective_date="1955-01-01",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "hama_1956": DocumentMetadata(
        document_id="hama_1956",
        title="The Hindu Adoptions and Maintenance Act, 1956 (Act No. 78 of 1956)",
        act_name="Hindu Adoptions and Maintenance Act, 1956",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/1640",
        local_filename="hindu_adoptions_and_maintenance_act_1956.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1956-12-21",
        effective_date="1956-12-21",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "pwdva_2005": DocumentMetadata(
        document_id="pwdva_2005",
        title="The Protection of Women from Domestic Violence Act, 2005 (Act No. 43 of 2005)",
        act_name="Protection of Women from Domestic Violence Act, 2005",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/2085",
        local_filename="protection_of_women_from_domestic_violence_act_2005.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="2005-09-13",
        effective_date="2006-10-26",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "gwa_1890": DocumentMetadata(
        document_id="gwa_1890",
        title="The Guardians and Wards Act, 1890 (Act No. 8 of 1890)",
        act_name="Guardians and Wards Act, 1890",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/2296",
        local_filename="guardians_and_wards_act_1890.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1890-03-21",
        effective_date="1890-07-01",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "crpc_sec_125": DocumentMetadata(
        document_id="crpc_sec_125",
        title="Code of Criminal Procedure, 1973 — Order for Maintenance of Wives, Children and Parents (Sections 125-128)",
        act_name="CrPC 1973 / BNSS 2023 (Maintenance Provisions)",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/1611",
        local_filename="crpc_section_125_maintenance.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1974-01-25",
        effective_date="1974-04-01",
        document_version="Relevant Extract (Sections 125-128)",
        processing_status=ProcessingStatus.PENDING,
    ),
    "fca_1984": DocumentMetadata(
        document_id="fca_1984",
        title="The Family Courts Act, 1984 (Act No. 66 of 1984)",
        act_name="Family Courts Act, 1984",
        official_source_url="https://www.indiacode.nic.in/handle/123456789/1944",
        local_filename="family_courts_act_1984.pdf",
        source_authority="Legislative Department, Ministry of Law and Justice, Government of India (India Code)",
        publication_date="1984-09-14",
        effective_date="1984-09-14",
        document_version="As modified up to date (India Code)",
        processing_status=ProcessingStatus.PENDING,
    ),
}


class ManifestManager:
    """Manages reading, updating, and exporting source manifests in JSON and CSV formats."""

    def __init__(self, manifest_dir: Path):
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.manifest_dir / "source_manifest.json"
        self.csv_path = self.manifest_dir / "source_manifest.csv"
        self._entries: Dict[str, DocumentMetadata] = {}
        self.load()

    def load(self) -> None:
        """Load manifest from JSON if available, seeding from canonical corpus."""
        # Start with canonical registry
        self._entries = {k: v.model_copy(deep=True) for k, v in CANONICAL_FAMILY_LAW_CORPUS.items()}

        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("documents", []):
                        doc = DocumentMetadata.model_validate(item)
                        self._entries[doc.document_id] = doc
            except Exception as e:
                print(f"[ManifestManager] Warning: Could not parse existing manifest at {self.json_path}: {e}")

    def save(self) -> None:
        """Persist manifest to both JSON and CSV files."""
        # 1. Save JSON
        data = {
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(self._entries),
            "documents": [doc.model_dump() for doc in self._entries.values()],
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 2. Save CSV
        fields = [
            "document_id",
            "title",
            "act_name",
            "official_source_url",
            "local_filename",
            "source_authority",
            "publication_date",
            "effective_date",
            "retrieval_date",
            "document_version",
            "processing_status",
            "page_count",
            "chunk_count",
            "file_size_bytes",
            "document_hash",
            "last_processed_at",
            "extraction_warnings",
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for doc in self._entries.values():
                row = doc.model_dump()
                row["extraction_warnings"] = " | ".join(row["extraction_warnings"])
                writer.writerow(row)

    def get(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve metadata for a specific document ID."""
        return self._entries.get(document_id)

    def get_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        """Find metadata matching a local filename."""
        clean_name = Path(filename).name.lower()
        for doc in self._entries.values():
            if Path(doc.local_filename).name.lower() == clean_name:
                return doc
        return None

    def upsert(self, metadata: DocumentMetadata) -> None:
        """Insert or update a document metadata record."""
        self._entries[metadata.document_id] = metadata

    def all_documents(self) -> List[DocumentMetadata]:
        """Return all tracked documents in the manifest."""
        return list(self._entries.values())
