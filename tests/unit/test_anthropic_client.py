from unittest.mock import MagicMock, patch

import pytest

from ad_rag_service import config


@pytest.fixture
def setup_anthropic_env(monkeypatch):
    with patch("dotenv.load_dotenv"):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123")
        monkeypatch.setenv("ANTHROPIC_MODEL_NAME", "claude-3-test")

        import importlib

        importlib.reload(config)

        yield

        monkeypatch.setenv("LLM_PROVIDER", "dummy")
        importlib.reload(config)


def test_anthropic_client_init(setup_anthropic_env):
    from ad_rag_service.llm.anthropic_client import AnthropicClient

    with patch("ad_rag_service.llm.anthropic_client.Anthropic") as mock_ant_cls:
        client = AnthropicClient()
        mock_ant_cls.assert_called_once_with(api_key="sk-ant-test-key-123")
        assert client.model == "claude-3-test"


def test_anthropic_client_init_missing_key(monkeypatch):
    with patch("dotenv.load_dotenv"):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        import importlib

        importlib.reload(config)

        from ad_rag_service.llm.anthropic_client import AnthropicClient

        with pytest.raises(ValueError, match="Anthropic API key not found"):
            AnthropicClient()


def test_anthropic_client_complete(setup_anthropic_env):
    from ad_rag_service.llm.anthropic_client import AnthropicClient

    with patch("ad_rag_service.llm.anthropic_client.Anthropic") as mock_ant_cls:
        mock_instance = mock_ant_cls.return_value
        mock_create = mock_instance.messages.create

        # Mock response structure for Anthropic
        # Response has a .content attribute which is a list of blocks
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Claude answer."

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_create.return_value = mock_response

        client = AnthropicClient()
        response = client.complete("Prompt", temperature=0.7, max_tokens=200)

        assert response == "Claude answer."

        mock_create.assert_called_once_with(
            model="claude-3-test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Prompt\n\nNote: This request has an upper limit on number of "
                        "output tokens. Please keep your answer to within approximately 100 words."
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=200,
        )


def test_anthropic_client_api_error(setup_anthropic_env):
    from anthropic import APIStatusError

    from ad_rag_service.llm.anthropic_client import AnthropicClient

    with patch("ad_rag_service.llm.anthropic_client.Anthropic") as mock_ant_cls:
        mock_instance = mock_ant_cls.return_value
        mock_create = mock_instance.messages.create

        # Simulate API Error. APIStatusError requires msg, response, body
        # We'll mock a simple version or just ensure the class handles exceptions broadly
        # APIStatusError needs init args. We'll mock it roughly.

        err_response = MagicMock()
        err_response.status_code = 500
        mock_create.side_effect = APIStatusError(
            message="Server error", response=err_response, body={}
        )

        client = AnthropicClient()

        with pytest.raises(RuntimeError, match="Anthropic API error"):
            client.complete("Prompt")
