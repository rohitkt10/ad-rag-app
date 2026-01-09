from __future__ import annotations

from pathlib import Path

from ad_rag_service import config
from ad_rag_service.types import AnswerWithCitations, ChunkRecord, Citation, RetrievedChunk


def test_config_paths():
    assert isinstance(config.REPO_ROOT, Path)
    assert config.REPO_ROOT.exists()

    assert isinstance(config.INDEX_DIR, Path)
    assert str(config.INDEX_DIR).endswith("artifacts/index")


def test_config_defaults():
    assert config.TOP_K == 5
    assert config.MAX_CONTEXT_TOKENS == 3000
    assert config.LLM_MODEL_NAME == "gemini-1.5-flash"
    assert config.LLM_TEMPERATURE == 0.0
    assert config.LLM_MAX_TOKENS == 512


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("TOP_K", "10")  # Should not be overridden as it's not from os.getenv

    # Reload config to pick up env vars
    # This is a bit hacky but needed for testing module-level constants
    import importlib

    importlib.reload(config)

    assert config.LLM_MODEL_NAME == "gpt-4"
    assert config.LLM_TEMPERATURE == 0.5
    # TOP_K should not change as it's not loaded from env var
    assert config.TOP_K == 5


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
