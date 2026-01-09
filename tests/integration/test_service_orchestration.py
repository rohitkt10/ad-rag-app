from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from ad_rag_service.generator import AnswerGenerator, LLMClient
from ad_rag_service.indexing import IndexStore
from ad_rag_service.retrieval import Retriever
from ad_rag_service.service import RAGService
from ad_rag_service.types import ChunkRecord


@pytest.fixture
def mock_index_store():
    # Tiny index (dim=2)
    d = 2
    index = faiss.IndexFlatIP(d)
    # Vec A: [1, 0]
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))

    store = MagicMock(spec=IndexStore)
    store.index = index
    store.lookup = [
        ChunkRecord(
            row_id=0,
            text="Doc A content",
            pmcid="PMC1",
            pmid=None,
            section_title="S1",
            chunk_index_in_section=0,
            source_xml="x",
            chunk_id=99,
        )
    ]
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Query matches Doc A perfectly
    embedder.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    return embedder


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.complete.return_value = "Answer based on [1]."
    return llm


def test_rag_service_flow(mock_index_store, mock_embedder, mock_llm):
    # Assemble components
    retriever = Retriever(mock_index_store, model_id="dummy", embedder=mock_embedder)
    generator = AnswerGenerator(mock_llm)
    service = RAGService(mock_index_store, retriever, generator)

    # Execute
    result = service.answer("test query")

    # Assert Retrieval happened
    mock_embedder.encode.assert_called()
    assert len(result.context_used) == 1
    assert result.context_used[0].record.text == "Doc A content"

    # Assert Generation happened
    mock_llm.complete.assert_called()
    assert result.answer == "Answer based on [1]."

    # Assert Citations parsed
    assert len(result.citations) == 1
    assert result.citations[0].pmcid == "PMC1"
    assert result.citations[0].chunk_id == 99
