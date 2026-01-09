from unittest.mock import MagicMock

import pytest

from ad_rag_service.generator import AnswerGenerator, LLMClient
from ad_rag_service.types import ChunkRecord, RetrievedChunk


@pytest.fixture
def mock_llm():
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def chunks():
    r1 = ChunkRecord(
        row_id=0,
        text="APOE4 increases risk.",
        pmcid="PMC1",
        pmid=None,
        section_title="Intro",
        chunk_index_in_section=0,
        source_xml="x",
        chunk_id=101,
    )
    r2 = ChunkRecord(
        row_id=1,
        text="Tau tangles correlate with decline.",
        pmcid="PMC2",
        pmid=None,
        section_title="Results",
        chunk_index_in_section=0,
        source_xml="x",
        chunk_id=102,
    )
    return [RetrievedChunk(record=r1, score=0.9), RetrievedChunk(record=r2, score=0.8)]


def test_generate_answer_with_citations(mock_llm, chunks):
    # Mock LLM response containing citations
    mock_llm.complete.return_value = "APOE4 is a risk factor [1]. Tau also matters [2]."

    gen = AnswerGenerator(mock_llm)
    result = gen.generate("What are the risk factors?", chunks)

    assert result.answer == "APOE4 is a risk factor [1]. Tau also matters [2]."
    assert len(result.citations) == 2

    c1 = result.citations[0]
    assert c1.chunk_id == 101
    assert c1.pmcid == "PMC1"

    c2 = result.citations[1]
    assert c2.chunk_id == 102
    assert c2.pmcid == "PMC2"

    # Check context used is passed through
    assert result.context_used == chunks


def test_generate_prompt_structure(mock_llm, chunks):
    mock_llm.complete.return_value = "Answer."
    gen = AnswerGenerator(mock_llm)
    gen.generate("Query", chunks)

    # Verify prompt construction
    args = mock_llm.complete.call_args[0]
    prompt = args[0]

    assert "Question: Query" in prompt
    assert "[1] (PMC1, Intro): APOE4 increases risk." in prompt
    assert "[2] (PMC2, Results): Tau tangles" in prompt
    assert "Answer the user's question using ONLY" in prompt


def test_generate_no_chunks(mock_llm):
    gen = AnswerGenerator(mock_llm)
    result = gen.generate("Query", [])

    assert "no relevant documents" in result.answer
    assert result.citations == []
    mock_llm.complete.assert_not_called()


def test_parse_citations_invalid_index(mock_llm, chunks):
    # LLM hallucinates [3] which doesn't exist
    mock_llm.complete.return_value = "Something [1] and [3]."

    gen = AnswerGenerator(mock_llm)
    result = gen.generate("Q", chunks)

    assert len(result.citations) == 1
    assert result.citations[0].pmcid == "PMC1"
    # [3] is ignored gracefully
