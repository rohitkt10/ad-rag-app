import os
from unittest.mock import patch

import pytest

from ad_rag_service import config
from ad_rag_service.llm.dummy_client import LLMClientImpl
from ad_rag_service.llm.factory import get_llm_client
from ad_rag_service.llm.interface import LLMClient


# Fixture to temporarily set and unset environment variables
@pytest.fixture
def set_env_vars():
    # Patch load_dotenv to avoid reading .env during tests
    with patch("dotenv.load_dotenv"):
        original_llm_provider = os.getenv("LLM_PROVIDER")
        original_openai_model = os.getenv("OPENAI_MODEL_NAME")
        original_anthropic_model = os.getenv("ANTHROPIC_MODEL_NAME")

        # Clear env vars to ensure clean state for each test
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        if "OPENAI_MODEL_NAME" in os.environ:
            del os.environ["OPENAI_MODEL_NAME"]
        if "ANTHROPIC_MODEL_NAME" in os.environ:
            del os.environ["ANTHROPIC_MODEL_NAME"]

        yield  # Run the test

        # Restore original env vars
        if original_llm_provider is not None:
            os.environ["LLM_PROVIDER"] = original_llm_provider
        else:
            if "LLM_PROVIDER" in os.environ:
                del os.environ["LLM_PROVIDER"]

        if original_openai_model is not None:
            os.environ["OPENAI_MODEL_NAME"] = original_openai_model
        else:
            if "OPENAI_MODEL_NAME" in os.environ:
                del os.environ["OPENAI_MODEL_NAME"]

        if original_anthropic_model is not None:
            os.environ["ANTHROPIC_MODEL_NAME"] = original_anthropic_model
        else:
            if "ANTHROPIC_MODEL_NAME" in os.environ:
                del os.environ["ANTHROPIC_MODEL_NAME"]


def test_get_llm_client_dummy_default(set_env_vars):
    """
    Test that get_llm_client returns LLMClientImpl when LLM_PROVIDER is unset (defaults to dummy).
    """
    # Ensure config reloads its values after env var changes
    import importlib

    importlib.reload(config)

    client = get_llm_client()
    assert isinstance(client, LLMClientImpl)
    assert isinstance(client, LLMClient)
    assert config.LLM_PROVIDER == "dummy"
    assert config.LLM_MODEL_NAME == "dummy-model"


def test_get_llm_client_dummy_explicit(set_env_vars):
    """
    Test that get_llm_client returns LLMClientImpl when LLM_PROVIDER is explicitly set to dummy.
    """
    os.environ["LLM_PROVIDER"] = "dummy"
    import importlib

    importlib.reload(config)

    client = get_llm_client()
    assert isinstance(client, LLMClientImpl)
    assert isinstance(client, LLMClient)
    assert config.LLM_PROVIDER == "dummy"
    assert config.LLM_MODEL_NAME == "dummy-model"


def test_get_llm_client_unsupported_provider(set_env_vars):
    """
    Test that get_llm_client raises ValueError for an unsupported provider.
    """
    os.environ["LLM_PROVIDER"] = "unsupported"
    with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
        import importlib

        importlib.reload(config)  # Config reload will trigger the validation
        _ = config.LLM_PROVIDER  # Accessing it will raise the error


def test_get_llm_client_openai_success(set_env_vars):
    """
    Test that get_llm_client returns OpenAIClient when LLM_PROVIDER is openai.
    """
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-fake"
    import importlib

    importlib.reload(config)

    with patch("ad_rag_service.llm.openai_client.OpenAIClient") as MockClient:
        client = get_llm_client()
        assert client == MockClient.return_value

    assert config.LLM_MODEL_NAME == "gpt-5.1"


def test_get_llm_client_anthropic_success(set_env_vars):
    """
    Test that get_llm_client returns AnthropicClient when LLM_PROVIDER is anthropic.
    """
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake"
    import importlib

    importlib.reload(config)

    with patch("ad_rag_service.llm.anthropic_client.AnthropicClient") as MockClient:
        client = get_llm_client()
        assert client == MockClient.return_value

    assert config.LLM_MODEL_NAME == "claude-3-5-sonnet"


def test_config_model_name_override_openai(set_env_vars):
    """
    Test that OPENAI_MODEL_NAME env var overrides default.
    """
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_MODEL_NAME"] = "gpt-alpha"
    import importlib

    importlib.reload(config)

    assert config.LLM_MODEL_NAME == "gpt-alpha"


def test_config_model_name_override_anthropic(set_env_vars):
    """
    Test that ANTHROPIC_MODEL_NAME env var overrides default.
    """
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_MODEL_NAME"] = "claude-next"
    import importlib

    importlib.reload(config)

    assert config.LLM_MODEL_NAME == "claude-next"
