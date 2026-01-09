from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from ad_rag_service.types import AnswerWithCitations, Citation, RetrievedChunk


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
    return mock_service


@pytest.fixture(scope="module")
def client():
    # Patch the global rag_service instance at the module level for the test client
    with patch("ad_rag_service.main.rag_service", new_callable=mock_rag_service_instance):
        # Patch the actual LLMClientImpl during app startup for a clean mock
        with patch("ad_rag_service.main.LLMClientImpl", new=MockLLMClientImpl):
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
