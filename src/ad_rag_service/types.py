from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkRecord:
    """Represents a single chunk record from lookup.jsonl."""

    row_id: int
    text: str
    pmcid: str
    pmid: str | None
    section_title: str
    chunk_index_in_section: int
    source_xml: str
    chunk_id: int | None = None
    # Additional metadata fields can be in **kwargs if we used a dict,
    # but explicit fields are better for type safety.
    # We'll allow extra kwargs for flexibility if needed,
    # but primarily we care about the core fields for citations.


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the index with its relevance score."""

    record: ChunkRecord
    score: float


@dataclass
class Citation:
    """Citation pointing to a specific chunk."""

    chunk_id: int
    pmcid: str
    text_snippet: str  # Short snippet to verify grounding


@dataclass
class AnswerWithCitations:
    """Final answer from the RAG service."""

    answer: str
    citations: list[Citation]
    context_used: list[RetrievedChunk]
