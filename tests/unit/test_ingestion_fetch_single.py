import json
from unittest.mock import patch

from ad_rag_pipeline import ingestion


@patch("ad_rag_pipeline.ingestion.search_pubmed")
@patch("ad_rag_pipeline.ingestion.get_pmcid_from_pmid")
@patch("ad_rag_pipeline.ingestion.fetch_pmc_xml")
def test_fetch_pmc_corpus_single(mock_fetch, mock_get_pmcid, mock_search, tmp_path):
    # Setup mocks
    mock_search.return_value = ["111", "222"]

    # First PMID maps to PMCID 999, second is ignored because n=1
    mock_get_pmcid.side_effect = lambda pmid: "999" if pmid == "111" else None

    # Mock fetch writing file side effect?
    # Actually fetch_pmc_xml implementation writes file.
    # But here we mocked it. So we must simulate the file write if we want to check it exists?
    # No, the logic in fetch_pmc_corpus assumes fetch_pmc_xml returns True/False.
    # It constructs the path and adds to manifest. It doesn't check file existence after fetch
    # (except for resume check which is before).
    mock_fetch.return_value = True

    out_dir = tmp_path / "raw"
    manifest_path = out_dir / "manifest.jsonl"

    counts = ingestion.fetch_pmc_corpus(
        query="test",
        out_dir=out_dir,
        email="test@example.com",
        target_n=1,
        sleep_s=0,
        manifest_path=manifest_path,
    )

    assert counts["downloaded"] == 1
    # Only called for the first one
    mock_get_pmcid.assert_called_with("111")
    mock_fetch.assert_called_once()

    # Check manifest
    assert manifest_path.exists()
    lines = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(lines) == 2  # run record + 1 article
    assert lines[0]["type"] == "run"
    assert lines[1]["type"] == "article"
    assert lines[1]["pmid"] == "111"
    assert lines[1]["pmcid"] == "PMC999"
    assert lines[1]["ok"] is True
