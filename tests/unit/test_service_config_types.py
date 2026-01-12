from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ad_rag_service import config
from ad_rag_service.types import AnswerWithCitations, ChunkRecord, Citation, RetrievedChunk


# Fixture to clear and reset environment variables related to config for each test
@pytest.fixture(autouse=True)
def reset_config_env_vars():
    with patch("dotenv.load_dotenv"):  # Patch load_dotenv for all tests using this fixture
        original_llm_provider = os.getenv("LLM_PROVIDER")
        original_openai_model = os.getenv("OPENAI_MODEL_NAME")
        original_anthropic_model = os.getenv("ANTHROPIC_MODEL_NAME")
        original_llm_temperature = os.getenv("LLM_TEMPERATURE")
        original_llm_max_tokens = os.getenv("LLM_MAX_TOKENS")

        keys_to_clear = [
            "LLM_PROVIDER",
            "OPENAI_MODEL_NAME",
            "ANTHROPIC_MODEL_NAME",
            "LLM_TEMPERATURE",  # Added to clear for testing defaults
            "LLM_MAX_TOKENS",  # Added to clear for testing defaults
        ]
        for key in keys_to_clear:
            if key in os.environ:
                del os.environ[key]

        # Reload config to ensure a clean state
        import importlib

        importlib.reload(config)

        yield

        # Restore original env vars
        for key in keys_to_clear:
            if key == "LLM_PROVIDER":
                if original_llm_provider is not None:
                    os.environ["LLM_PROVIDER"] = original_llm_provider
                else:
                    if "LLM_PROVIDER" in os.environ:
                        del os.environ["LLM_PROVIDER"]
            elif key == "OPENAI_MODEL_NAME":
                if original_openai_model is not None:
                    os.environ["OPENAI_MODEL_NAME"] = original_openai_model
                else:
                    if "OPENAI_MODEL_NAME" in os.environ:
                        del os.environ["OPENAI_MODEL_NAME"]
            elif key == "ANTHROPIC_MODEL_NAME":
                if original_anthropic_model is not None:
                    os.environ["ANTHROPIC_MODEL_NAME"] = original_anthropic_model
                else:
                    if "ANTHROPIC_MODEL_NAME" in os.environ:
                        del os.environ["ANTHROPIC_MODEL_NAME"]
            elif key == "LLM_TEMPERATURE":
                if original_llm_temperature is not None:
                    os.environ["LLM_TEMPERATURE"] = original_llm_temperature
                else:
                    if "LLM_TEMPERATURE" in os.environ:
                        del os.environ["LLM_TEMPERATURE"]
            elif key == "LLM_MAX_TOKENS":
                if original_llm_max_tokens is not None:
                    os.environ["LLM_MAX_TOKENS"] = original_llm_max_tokens
                else:
                    if "LLM_MAX_TOKENS" in os.environ:
                        del os.environ["LLM_MAX_TOKENS"]

        # Reload config again after restoring env vars
        importlib.reload(config)


def test_config_paths():
    assert isinstance(config.REPO_ROOT, Path)
    assert config.REPO_ROOT.exists()

    assert isinstance(config.INDEX_DIR, Path)
    assert str(config.INDEX_DIR).endswith("artifacts/index")


def test_config_defaults_dummy():
    # When no LLM_PROVIDER is set, it defaults to 'dummy'
    # We enforce 'dummy' here because .env might be present in the dev environment
    os.environ["LLM_PROVIDER"] = "dummy"

    # With load_dotenv patched by the fixture, direct os.environ manipulation
    # and config defaults will be used reliably.
    import importlib

    importlib.reload(config)

    assert config.LLM_PROVIDER == "dummy"
    assert config.LLM_MODEL_NAME == "dummy-model"
    assert config.LLM_TEMPERATURE == 0.3
    assert config.LLM_MAX_TOKENS == 1000
    assert config.OPENAI_API_KEY_ENV == "OPENAI_API_KEY"
    assert config.ANTHROPIC_API_KEY_ENV == "ANTHROPIC_API_KEY"


def test_config_allowed_providers():
    assert "openai" in config.ALLOWED_PROVIDERS
    assert "anthropic" in config.ALLOWED_PROVIDERS
    assert "dummy" in config.ALLOWED_PROVIDERS


def test_config_invalid_provider_raises_error():
    os.environ["LLM_PROVIDER"] = "invalid_provider"
    with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
        import importlib

        importlib.reload(config)


def test_config_openai_defaults():
    os.environ["LLM_PROVIDER"] = "openai"
    import importlib

    importlib.reload(config)
    assert config.LLM_PROVIDER == "openai"
    assert config.LLM_MODEL_NAME == "gpt-5.1"


def test_config_anthropic_defaults():
    os.environ["LLM_PROVIDER"] = "anthropic"
    import importlib

    importlib.reload(config)
    assert config.LLM_PROVIDER == "anthropic"
    assert config.LLM_MODEL_NAME == "claude-sonnet-4-5"


def test_config_openai_model_override():
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_MODEL_NAME"] = "gpt-custom"
    import importlib

    importlib.reload(config)
    assert config.LLM_PROVIDER == "openai"
    assert config.LLM_MODEL_NAME == "gpt-custom"


def test_config_anthropic_model_override():
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_MODEL_NAME"] = "claude-custom"
    import importlib

    importlib.reload(config)
    assert config.LLM_PROVIDER == "anthropic"
    assert config.LLM_MODEL_NAME == "claude-custom"


def test_config_generic_llm_model_name_ignored_if_provider_specific_exists():
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL_NAME"] = "ignored-model"  # This should be ignored by the new logic
    os.environ["OPENAI_MODEL_NAME"] = "gpt-specific"
    import importlib

    importlib.reload(config)
    assert config.LLM_PROVIDER == "openai"
    assert config.LLM_MODEL_NAME == "gpt-specific"


def test_chunk_record_creation():
    rec = ChunkRecord(
        row_id=0,
        text="some text",
        pmcid="PMC123",
        pmid="12345",
        section_title="Intro",
        chunk_index_in_section=0,
        source_xml="/a/b/c.xml",
        chunk_id=100,
    )
    assert rec.row_id == 0
    assert rec.text == "some text"


def test_retrieved_chunk_creation():
    rec = ChunkRecord(
        row_id=0,
        text="t",
        pmcid="p",
        pmid="p",
        section_title="s",
        chunk_index_in_section=0,
        source_xml="x",
    )
    r_chunk = RetrievedChunk(record=rec, score=0.99)
    assert r_chunk.record.pmcid == "p"
    assert r_chunk.score == 0.99


def test_citation_creation():
    cite = Citation(chunk_id=100, pmcid="PMC123", text_snippet="snippet")
    assert cite.chunk_id == 100
    assert cite.pmcid == "PMC123"


def test_answer_with_citations_creation():
    rec = ChunkRecord(
        row_id=0,
        text="t",
        pmcid="p",
        pmid="p",
        section_title="s",
        chunk_index_in_section=0,
        source_xml="x",
    )
    r_chunk = RetrievedChunk(record=rec, score=0.99)
    cite = Citation(chunk_id=0, pmcid="PMC123", text_snippet="snippet")

    ans = AnswerWithCitations(answer="This is an answer.", citations=[cite], context_used=[r_chunk])
    assert ans.answer == "This is an answer."
    assert len(ans.citations) == 1
    assert ans.context_used[0].score == 0.99
