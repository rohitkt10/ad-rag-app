from unittest.mock import patch

from ad_rag_pipeline import ingestion


@patch("ad_rag_pipeline.ingestion.search_pubmed")
@patch("ad_rag_pipeline.ingestion.get_pmcid_from_pmid")
@patch("ad_rag_pipeline.ingestion.fetch_pmc_xml")
def test_ingestion_resume(mock_fetch, mock_get_pmcid, mock_search, tmp_path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()

    # Pre-create PMC123.xml
    (out_dir / "PMC123.xml").write_text("existing")

    mock_search.return_value = ["111"]
    mock_get_pmcid.return_value = "123"  # PMID 111 -> PMC123

    counts = ingestion.fetch_pmc_corpus(
        query="test", out_dir=out_dir, email="test@example.com", target_n=1, sleep_s=0, resume=True
    )

    assert counts["skipped"] == 1
    assert counts["downloaded"] == 0

    # Should NOT have called fetch
    mock_fetch.assert_not_called()
