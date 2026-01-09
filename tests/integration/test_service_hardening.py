from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# We can reuse the mock client fixture setup but with variations
from ad_rag_service.types import AnswerWithCitations


def mock_rag_service_instance():
    # Helper to create a functional mock service
    mock_service = MagicMock()
    mock_service.answer.return_value = AnswerWithCitations(
        answer="Mock.", citations=[], context_used=[]
    )
    mock_service.index_store.index.is_trained = True
    return mock_service


@pytest.fixture
def client():
    # Base client with functional mock
    with patch("ad_rag_service.main.rag_service", new_callable=mock_rag_service_instance):
        from ad_rag_service.main import app

        yield TestClient(app)


def test_query_validation_max_length(client: TestClient):
    long_question = "a" * 1001
    response = client.post("/query", json={"question": long_question})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    # Pydantic validation error details usually in body
    assert "String should have at most 1000 characters" in response.text


def test_startup_failure_missing_index():
    # This test needs to mock IndexStore to raise exception,
    # AND we need to run lifespan logic which TestClient does automatically
    # if using 'with TestClient(app) as client:'
    # BUT our rag_service is global.
    # We need to ensure we don't break other tests by leaving rag_service
    # in a bad state or relying on side effects.

    # We'll mock config to point to non-existent files
    with patch("ad_rag_service.config.FAISS_INDEX_PATH", "/non/existent/path"):
        # We need to re-import or use a fresh app instance to trigger lifespan?
        # The lifespan is attached to the app.
        # TestClient(app) triggers lifespan startup.

        from ad_rag_service.main import app

        # We expect a RuntimeError during startup
        with pytest.raises(RuntimeError, match="Service initialization failed"):
            with TestClient(app):
                pass


def test_query_error_handling(client: TestClient):
    # Mock rag_service.answer to raise an exception
    with patch(
        "ad_rag_service.main.rag_service.answer",
        side_effect=ValueError("Simulated retrieval error"),
    ):
        response = client.post("/query", json={"question": "Valid question"})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Simulated retrieval error" in response.json()["detail"]
