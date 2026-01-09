from unittest.mock import MagicMock, patch

from ad_rag_pipeline import ingestion


@patch("Bio.Entrez.esearch")
@patch("Bio.Entrez.read")
def test_search_pubmed(mock_read, mock_esearch):
    mock_read.return_value = {"IdList": ["123", "456"]}
    pmids = ingestion.search_pubmed("query", 10)
    assert pmids == ["123", "456"]
    mock_esearch.assert_called_once_with(db="pubmed", term="query", retmax=10, sort="relevance")


@patch("Bio.Entrez.elink")
@patch("Bio.Entrez.read")
def test_get_pmcid_from_pmid_success(mock_read, mock_elink):
    # Entrez.read returns a list of dicts for elink
    mock_read.return_value = [{"LinkSetDb": [{"Link": [{"Id": "999"}]}]}]
    pmcid = ingestion.get_pmcid_from_pmid("123")
    assert pmcid == "999"
    mock_elink.assert_called_once_with(dbfrom="pubmed", id="123", linkname="pubmed_pmc")


@patch("Bio.Entrez.elink")
@patch("Bio.Entrez.read")
def test_get_pmcid_from_pmid_no_link(mock_read, mock_elink):
    mock_read.return_value = [{"LinkSetDb": []}]
    pmcid = ingestion.get_pmcid_from_pmid("123")
    assert pmcid is None


@patch("Bio.Entrez.efetch")
def test_fetch_pmc_xml_success(mock_efetch, tmp_path):
    mock_handle = MagicMock()
    mock_handle.read.return_value = b"<xml>content</xml>"
    mock_efetch.return_value = mock_handle

    out_file = tmp_path / "test.xml"
    success = ingestion.fetch_pmc_xml("999", out_file)

    assert success is True
    assert out_file.read_bytes() == b"<xml>content</xml>"
    mock_efetch.assert_called_once_with(db="pmc", id="999", rettype="full", retmode="xml")


@patch("Bio.Entrez.efetch")
def test_fetch_pmc_xml_failure(mock_efetch, tmp_path):
    mock_efetch.side_effect = Exception("Fetch error")
    out_file = tmp_path / "fail.xml"
    success = ingestion.fetch_pmc_xml("999", out_file)
    assert success is False
    assert not out_file.exists()
