from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from ad_rag_service import config
from ad_rag_service.indexing import IndexStore
from ad_rag_service.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding text."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class Retriever:
    """
    Executes dense retrieval against the IndexStore.
    """

    def __init__(
        self,
        index_store: IndexStore,
        model_id: str,
        device: str = "cpu",
        embedder: Embedder | None = None,
    ) -> None:
        """
        Initialize Retriever.

        Args:
            index_store: Loaded IndexStore instance.
            model_id: HuggingFace model ID for query embedding.
            device: Device for inference ('cpu', 'cuda').
            embedder: Optional pre-initialized embedder (for testing or sharing).
                      If None, loads SentenceTransformer(model_id).
        """
        self.index_store = index_store
        self.model_id = model_id
        self.device = device

        if embedder:
            self.embedder = embedder
        else:
            logger.info(f"Loading embedding model: {model_id} on {device}")
            self.embedder = SentenceTransformer(model_id, device=device)

    def retrieve(self, query: str, k: config.TOP_K) -> list[RetrievedChunk]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects, sorted by score (descending).
        """
        if not query.strip():
            return []

        # 1. Embed query
        # normalize_embeddings=True because index is cosine (Inner Product on normalized vectors)
        # encode returns numpy array if convert_to_numpy=True (default in recent versions)
        # We pass as list [query]
        embeddings = self.embedder.encode([query], normalize_embeddings=True)

        # Ensure float32 for FAISS
        query_vector = embeddings.astype(np.float32)

        # 2. Search Index
        if self.index_store.index is None:
            raise RuntimeError("Index not loaded in IndexStore.")

        # D: Distances (scores), indices: Row IDs
        D, indices = self.index_store.index.search(query_vector, k)

        # 3. Map results
        results: list[RetrievedChunk] = []

        # D and indices are shape (1, k)
        row_ids = indices[0]
        scores = D[0]

        for row_id, score in zip(row_ids, scores, strict=False):
            # FAISS returns -1 if fewer than k results found
            if row_id == -1:
                continue

            if row_id < 0 or row_id >= len(self.index_store.lookup):
                logger.error(f"Index returned row_id {row_id} out of bounds.")
                continue

            record = self.index_store.lookup[row_id]
            results.append(RetrievedChunk(record=record, score=float(score)))

        return results
