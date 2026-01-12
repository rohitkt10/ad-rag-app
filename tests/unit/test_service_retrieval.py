from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from ad_rag_service.indexing import IndexStore
from ad_rag_service.retrieval import Retriever
from ad_rag_service.types import ChunkRecord


@pytest.fixture
def mock_index_store():
    # Create a tiny index with 3 vectors of dim 2
    # vec0=[1, 0], vec1=[0, 1], vec2=[0.7, 0.7] (approx)
    d = 2
    index = faiss.IndexFlatIP(d)
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], dtype=np.float32)
    index.add(vecs)

    store = MagicMock(spec=IndexStore)
    store.index = index
    store.lookup = [
        ChunkRecord(
            row_id=0,
            text="doc A",
            pmcid="1",
            pmid=None,
            section_title="s",
            chunk_index_in_section=0,
            source_xml="x",
        ),
        ChunkRecord(
            row_id=1,
            text="doc B",
            pmcid="2",
            pmid=None,
            section_title="s",
            chunk_index_in_section=0,
            source_xml="x",
        ),
        ChunkRecord(
            row_id=2,
            text="doc C",
            pmcid="3",
            pmid=None,
            section_title="s",
            chunk_index_in_section=0,
            source_xml="x",
        ),
    ]
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Mock behavior: return [1, 0] for query "A", [0, 1] for "B", etc.
    # We'll just generic return based on side_effect or simple logic if needed.
    # For now, just return a fixed vector [1, 0] to match doc A.
    embedder.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    return embedder


def test_retrieve_top_k(mock_index_store, mock_embedder):
    retriever = Retriever(mock_index_store, model_id="dummy", embedder=mock_embedder)

    # Query vector is [1, 0]. Should match doc A (score 1.0) best.
    results = retriever.retrieve("query", k=2)

    assert len(results) == 2
    # Top result
    assert results[0].record.text == "doc A"
    assert results[0].score >= 0.99

    # Second result should be doc C (dot product ~0.707) or doc B (0.0)
    # 1*0.7 + 0*0.7 = 0.7. 1*0 + 0*1 = 0.
    # So doc C is second.
    assert results[1].record.text == "doc C"
    assert 0.7 < results[1].score < 0.8


def test_retrieve_empty_query(mock_index_store, mock_embedder):
    retriever = Retriever(mock_index_store, model_id="dummy", embedder=mock_embedder)
    results = retriever.retrieve("", k=5)
    assert results == []
    mock_embedder.encode.assert_not_called()


def test_retrieve_index_not_loaded(mock_embedder):
    store = MagicMock(spec=IndexStore)
    store.index = None
    retriever = Retriever(store, model_id="dummy", embedder=mock_embedder)

    with pytest.raises(RuntimeError, match="Index not loaded"):
        retriever.retrieve("foo", k=5)
