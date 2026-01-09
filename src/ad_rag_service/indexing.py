from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss

from ad_rag_service.types import ChunkRecord

logger = logging.getLogger(__name__)


class IndexStore:
    """
    Read-only store for the FAISS index and metadata lookup.
    Loads artifacts into memory and ensures consistency.
    """

    def __init__(
        self,
        index_path: Path,
        lookup_path: Path,
        meta_path: Path,
    ) -> None:
        self.index_path = index_path
        self.lookup_path = lookup_path
        self.meta_path = meta_path

        self.index: faiss.Index | None = None
        self.lookup: list[ChunkRecord] = []
        self.meta: dict[str, Any] = {}

    def load(self) -> None:
        """
        Load artifacts from disk.
        Raises FileNotFoundError or ValueError if artifacts are missing or inconsistent.
        """
        logger.info(f"Loading index from {self.index_path.parent}")

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.lookup_path.exists():
            raise FileNotFoundError(f"Lookup file not found: {self.lookup_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        # 1. Load Metadata
        try:
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.meta = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse meta JSON: {e}") from e

        # 2. Load FAISS Index
        try:
            self.index = faiss.read_index(str(self.index_path))
        except RuntimeError as e:
            raise ValueError(f"Failed to load FAISS index: {e}") from e

        # 3. Load Lookup
        self.lookup = []
        try:
            with self.lookup_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_dict = json.loads(line)
                        # We map the dict to ChunkRecord.
                        # We filter strict fields or just pass kwargs?
                        # Using explicit fields for safety, but lookup might have extras.
                        # We'll rely on explicit args that match our pipeline output.

                        # Pipeline output: row_id, text, pmcid, pmid, section_title,
                        # chunk_index_in_section, chunk_id, source_xml, + base_md (journal, doi...)

                        # We map essential fields to ChunkRecord.
                        rec = ChunkRecord(
                            row_id=record_dict["row_id"],
                            text=record_dict["text"],
                            pmcid=record_dict["pmcid"],
                            pmid=record_dict.get("pmid"),
                            section_title=record_dict["section_title"],
                            chunk_index_in_section=record_dict["chunk_index_in_section"],
                            source_xml=record_dict["source_xml"],
                            chunk_id=record_dict.get("chunk_id"),
                        )
                        self.lookup.append(rec)
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValueError(f"Invalid lookup record at line {line_num}: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Failed to read lookup file: {e}") from e

        self._validate()
        logger.info("IndexStore loaded successfully.")

    def _validate(self) -> None:
        """Ensure index and lookup are consistent."""
        if self.index is None:
            raise ValueError("Index not loaded.")

        # Check count consistency
        if self.index.ntotal != len(self.lookup):
            raise ValueError(
                f"Consistency error: Index has {self.index.ntotal} vectors, "
                f"but lookup has {len(self.lookup)} records."
            )

        # Check dimension consistency if available in meta
        expected_dim = self.meta.get("embedding_dim")
        if expected_dim is not None and self.index.d != expected_dim:
            raise ValueError(
                f"Dimension mismatch: Meta says {expected_dim}, Index has {self.index.d}"
            )
