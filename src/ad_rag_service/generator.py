from __future__ import annotations

import logging
import re
from typing import Protocol

from ad_rag_service.types import AnswerWithCitations, Citation, RetrievedChunk

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Abstract interface for an LLM provider."""

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str: ...


class AnswerGenerator:
    """
    Generates grounded answers using retrieved context and an LLM.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context_str_parts = []
        for i, chunk in enumerate(chunks, start=1):
            # Format: [i] (PMCID, Section): text
            # We truncate text slightly if needed? For now, assume it fits
            # or is handled by max tokens logic elsewhere.
            # config.MAX_CONTEXT_TOKENS handles retrieval limit, here we just use what we got.
            header = f"[{i}] ({chunk.record.pmcid}, {chunk.record.section_title})"
            context_str_parts.append(f"{header}: {chunk.record.text}")

        context_block = "\n\n".join(context_str_parts)

        return f"""You are an expert Alzheimer's Disease researcher. 
Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer, say "I don't know based on the provided context."
Cite the context chunks you use by their ID, e.g. [1], [2].
Every factual statement must be cited.

Context:
{context_block}

Question: {query}

Answer:"""

    def _parse_citations(self, answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        """
        Extract citations like [1], [2] from the answer and map to chunks.
        """
        # Regex to find [1], [2], etc.
        # We look for [digits]
        matches = re.findall(r"\[(\d+)\]", answer)

        citations = []
        seen_indices = set()

        for match in matches:
            try:
                idx = int(match) - 1  # Convert [1] -> 0 index
                if idx < 0 or idx >= len(chunks):
                    continue

                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                chunk = chunks[idx]
                # snippet: take first 50 chars as snippet for verification
                snippet = (
                    chunk.record.text[:50] + "..."
                    if len(chunk.record.text) > 50
                    else chunk.record.text
                )

                citations.append(
                    Citation(
                        chunk_id=chunk.record.chunk_id
                        if chunk.record.chunk_id is not None
                        else -1,  # Fallback if chunk_id missing
                        pmcid=chunk.record.pmcid,
                        text_snippet=snippet,
                    )
                )
            except ValueError:
                continue

        return citations

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> AnswerWithCitations:
        if not chunks:
            return AnswerWithCitations(
                answer="I found no relevant documents to answer your question.",
                citations=[],
                context_used=[],
            )

        prompt = self._build_prompt(query, chunks)

        # We assume config defaults are handled by the caller or we can inject them.
        # For now, use defaults in Protocol or pass explicit?
        # The class doesn't hold config. We'll use defaults.
        raw_answer = self.llm_client.complete(prompt)

        citations = self._parse_citations(raw_answer, chunks)

        return AnswerWithCitations(
            answer=raw_answer.strip(), citations=citations, context_used=chunks
        )
