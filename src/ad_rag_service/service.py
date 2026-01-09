from __future__ import annotations

import logging

from ad_rag_service.config import TOP_K
from ad_rag_service.generator import AnswerGenerator
from ad_rag_service.indexing import IndexStore
from ad_rag_service.retrieval import Retriever
from ad_rag_service.types import AnswerWithCitations

logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestrates the RAG pipeline: Retrieval -> Generation.
    """

    def __init__(
        self,
        index_store: IndexStore,
        retriever: Retriever,
        generator: AnswerGenerator,
    ) -> None:
        self.index_store = index_store
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, k: int = TOP_K) -> AnswerWithCitations:
        """
        Answer a user query using RAG.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve (default: config.TOP_K).

        Returns:
            AnswerWithCitations object.
        """
        logger.info(f"Processing query: {query}")

        # 1. Retrieve
        chunks = self.retriever.retrieve(query, k=k)
        logger.info(f"Retrieved {len(chunks)} chunks.")

        # 2. Generate
        answer = self.generator.generate(query, chunks)
        logger.info("Generated answer.")

        return answer
