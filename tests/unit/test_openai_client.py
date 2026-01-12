from unittest.mock import MagicMock, patch

import pytest

from ad_rag_service import config


# We need to ensure config is in a clean state compatible with OpenAI
@pytest.fixture
def setup_openai_env(monkeypatch):
    # Patch load_dotenv to prevent it from reading .env and overriding our monkeypatch
    with patch("dotenv.load_dotenv"):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        monkeypatch.setenv(
            "OPENAI_MODEL_NAME", "gpt-5.1-test"
        )  # Explicitly set model to verify usage

        # Reload config to apply LLM_PROVIDER changes (which sets LLM_MODEL_NAME)
        import importlib

        importlib.reload(config)

        yield

        # Teardown logic
        monkeypatch.setenv("LLM_PROVIDER", "dummy")
        importlib.reload(config)


def test_openai_client_init(setup_openai_env):
    from ad_rag_service.llm.openai_client import OpenAIClient

    with patch("ad_rag_service.llm.openai_client.OpenAI") as mock_openai_cls:
        client = OpenAIClient()

        # Verify OpenAI client initialized with correct key
        mock_openai_cls.assert_called_once_with(api_key="sk-test-key-123")
        # Verify model was picked up from config
        assert client.model == "gpt-5.1-test"


def test_openai_client_init_missing_key(monkeypatch):
    # Patch load_dotenv here as well
    with patch("dotenv.load_dotenv"):
        # Set provider to openai but unset key
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import importlib

        importlib.reload(config)

        from ad_rag_service.llm.openai_client import OpenAIClient

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIClient()


def test_openai_client_complete(setup_openai_env):
    from ad_rag_service.llm.openai_client import OpenAIClient

    with patch("ad_rag_service.llm.openai_client.OpenAI") as mock_openai_cls:
        # Mock the chain: client.chat.completions.create
        mock_instance = mock_openai_cls.return_value
        mock_create = mock_instance.chat.completions.create

        # Mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test generated answer."
        mock_create.return_value = mock_response

        client = OpenAIClient()
        response = client.complete("Test prompt", temperature=0.5, max_tokens=100)

        assert response == "Test generated answer."

    mock_create.assert_called_once_with(
            model="gpt-5.1-test",
            messages=[
                {
                    "role": "user",
                    "content": "Test prompt\n\nNote: This request has an upper limit on number of output tokens. Please keep your answer to within approximately 50 words.",
                }
            ],
            temperature=0.5,
            max_completion_tokens=100,
            reasoning_effort="none",
        )


def test_openai_client_api_error(setup_openai_env):
    from openai import OpenAIError

    from ad_rag_service.llm.openai_client import OpenAIClient

    with patch("ad_rag_service.llm.openai_client.OpenAI") as mock_openai_cls:
        mock_instance = mock_openai_cls.return_value
        mock_create = mock_instance.chat.completions.create

        # Simulate API Error
        mock_create.side_effect = OpenAIError("Rate limit exceeded")

        client = OpenAIClient()

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            client.complete("Test prompt")
