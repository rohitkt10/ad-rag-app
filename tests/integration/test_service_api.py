from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from ad_rag_service.types import AnswerWithCitations, ChunkRecord, Citation, RetrievedChunk


# Mock the LLMClientImpl to avoid actual LLM calls
class MockLLMClientImpl:
    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        return "Mocked answer for test [1]."


# Mock the entire RAGService for API integration tests
# We don't need to test the RAGService logic here, just the API wiring
def mock_rag_service_instance():
    mock_service = MagicMock()
    mock_service.answer.return_value = AnswerWithCitations(
        answer="Mocked answer for test [1].",
        citations=[Citation(chunk_id=1, pmcid="PMC123", text_snippet="Mock snippet...")],
        context_used=[
            RetrievedChunk(record=MagicMock(pmcid="PMC123", section_title="Intro"), score=0.9)
        ],
    )
    mock_service.index_store = MagicMock()
    mock_service.index_store.index = MagicMock()
    mock_service.index_store.index.is_trained = True  # For health check

    # Mock retriever for the /retrieve endpoint
    mock_service.retriever = MagicMock()
    mock_service.retriever.retrieve.return_value = [
        RetrievedChunk(
            record=ChunkRecord(
                row_id=0,
                text="Mock chunk 1",
                pmcid="PMC456",
                pmid="1",
                section_title="Methods",
                chunk_index_in_section=0,
                source_xml="test.xml",
                chunk_id=101,
            ),
            score=0.85,
        ),
        RetrievedChunk(
            record=ChunkRecord(
                row_id=1,
                text="Mock chunk 2",
                pmcid="PMC789",
                pmid="2",
                section_title="Results",
                chunk_index_in_section=0,
                source_xml="test.xml",
                chunk_id=102,
            ),
            score=0.75,
        ),
    ]

    return mock_service


@pytest.fixture(scope="module")
def client():
    # Patch the global rag_service instance at the module level for the test client
    with patch("ad_rag_service.main.rag_service", new_callable=mock_rag_service_instance):
        # Patch the factory function to return a mock client
        with patch("ad_rag_service.main.get_llm_client", return_value=MockLLMClientImpl()):
            # Patch _file_info and _read_json_if_exists for metadata endpoint
            with (
                patch("ad_rag_service.main._file_info") as mock_file_info,
                patch("ad_rag_service.main._read_json_if_exists") as mock_read_json,
            ):
                mock_file_info.return_value = {"path": "/mock/path", "exists": False, "bytes": 0}
                mock_read_json.return_value = {"mock_key": "mock_value"}

                from ad_rag_service.main import app

                yield TestClient(app)


def test_health_check_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok", "index_loaded": True}


def test_query_rag_service_success(client: TestClient):
    response = client.post("/query", json={"question": "What is Alzheimer's?"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == "Mocked answer for test [1]."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["pmcid"] == "PMC123"


def test_query_rag_service_empty_question(client: TestClient):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Question cannot be empty."


def test_query_rag_service_uninitialized_service(client: TestClient):
    # Temporarily set rag_service to None for this test
    with patch("ad_rag_service.main.rag_service", None):
        response = client.post("/query", json={"question": "dummy"})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json()["detail"] == "RAG service not initialized."


def test_health_check_uninitialized_service(client: TestClient):
    with patch("ad_rag_service.main.rag_service", None):
        response = client.get("/health")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json()["detail"] == "RAG service not initialized or index not loaded."


def test_metadata_endpoint(client: TestClient):
    response = client.get("/metadata")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "llm_provider" in data
    assert "llm_model_name" in data
    assert "embedding_model_id" in data
    assert "embedding_device" in data
    assert "top_k_default" in data
    assert "artifacts" in data
    assert "manifest" in data

    assert data["artifacts"]["faiss_index"]["exists"] is False  # Now mocked
    assert data["manifest"] == {"mock_key": "mock_value"}  # Now mocked


def test_retrieve_endpoint_success(client: TestClient):
    response = client.post("/retrieve", json={"query": "some query", "k": 2})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert len(data) == 2
    assert data[0]["record"]["pmcid"] == "PMC456"
    assert data[0]["score"] == 0.85
    assert data[1]["record"]["pmcid"] == "PMC789"
    assert data[1]["score"] == 0.75


def test_retrieve_endpoint_empty_query(client: TestClient):
    response = client.post("/retrieve", json={"query": "   ", "k": 1})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Query cannot be empty."


def test_retrieve_endpoint_uninitialized_service(client: TestClient):
    with patch("ad_rag_service.main.rag_service", None):
        response = client.post("/retrieve", json={"query": "dummy", "k": 1})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json()["detail"] == "RAG service not initialized."
