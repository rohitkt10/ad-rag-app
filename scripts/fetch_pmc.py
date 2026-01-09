#!/usr/bin/env python3
"""
Download PMC full-text XMLs for an AD biomarker corpus (via PubMed -> PMC link).

Writes:
- data/raw/PMC<pmcid>.xml
- data/raw/manifest.jsonl  (run record + per-article records)

Notes:
- Uses PubMed search (query includes date range), then maps PMID -> PMCID via Entrez elink.
- Keeps logic intentionally simple for MVP.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from Bio import Entrez

# ----------------------------
# Paths (repo-relative)
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MANIFEST_PATH = RAW_DIR / "manifest.jsonl"


# ----------------------------
# Entrez helpers
# ----------------------------
def _init_entrez() -> None:
    """
    Initialize NCBI Entrez global configuration with email and API key from environment variables.

    Raises:
        RuntimeError: If NCBI_EMAIL (or ENTREZ_EMAIL) is not set in the environment.
    """
    email = os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL")
    if not email:
        raise RuntimeError("Set NCBI_EMAIL (or ENTREZ_EMAIL) in your environment for NCBI Entrez.")
    Entrez.email = email

    api_key = os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def search_pubmed(query: str, retmax: int) -> list[str]:
    """
    Search PubMed for a given query and return a list of matching PMIDs.

    Args:
        query (str): The PubMed search string.
        retmax (int): The maximum number of IDs to retrieve.

    Returns:
        list[str]: A list of PubMed IDs (PMIDs) matching the query.
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=retmax,
        sort="relevance",
    )
    res = Entrez.read(handle)
    handle.close()
    return list(res.get("IdList", []))


def get_pmcid_from_pmid(pmid: str) -> str | None:
    """
    Retrieve the PMC ID corresponding to a given PubMed ID.

    Uses Entrez ELink to find the 'pubmed_pmc' link.

    Args:
        pmid (str): The PubMed ID to query.

    Returns:
        Optional[str]: The numeric part of the PMC ID (e.g., '123456') if found, else None.
    """
    h = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc")
    res = Entrez.read(h)
    h.close()

    # Typical structure: res[0]["LinkSetDb"][0]["Link"][0]["Id"] -> PMCID numeric
    if not res:
        return None
    linksetdb = res[0].get("LinkSetDb")
    if not linksetdb:
        return None
    links = linksetdb[0].get("Link", [])
    if not links:
        return None
    return links[0].get("Id")


def fetch_or_load_xml(pmcid: str) -> bytes:
    """
    Fetch the full-text XML for a given PMC ID from NCBI or load it from disk if cached.

    Args:
        pmcid (str): The numeric PMC ID (without 'PMC' prefix) to fetch.

    Returns:
        bytes: The raw XML content of the article.
    """
    xml_path = RAW_DIR / f"PMC{pmcid}.xml"
    if xml_path.exists():
        return xml_path.read_bytes()

    handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    xml_bytes = handle.read()
    handle.close()

    xml_path.write_bytes(xml_bytes)
    return xml_bytes


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------
def default_query(start_year: int, end_year: int) -> str:
    base = (
        '("Alzheimer Disease"[MeSH] OR "Alzheimer*"[Title/Abstract]) AND '
        '("Biomarkers"[MeSH] OR "biomarker*"[Title/Abstract])'
    )
    # PubMed date range on Date of Publication [dp]
    date_filter = f' AND ("{start_year}"[dp] : "{end_year}"[dp])'
    return base + date_filter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of PMC XMLs to download (default=50).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year for PubMed [dp] filter (default=2021).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year for PubMed [dp] filter (default=current year).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Override the default PubMed query entirely.",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=3,
        help="PubMed retmax multiplier to find enough PMC-linked PMIDs (default=3).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds to sleep between Entrez requests (default=0.35).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to JSONL manifest (default=data/raw/manifest.jsonl).",
    )
    args = parser.parse_args()

    _init_entrez()

    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    q = args.query or default_query(args.start_year, args.end_year)
    target_n = args.n
    query_n = max(target_n * args.oversample, target_n)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_jsonl(
        manifest_path,
        {
            "type": "run",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": q,
            "target_n": target_n,
            "query_retmax": query_n,
            "raw_dir": str(RAW_DIR),
        },
    )

    print(f"PubMed query (retmax={query_n}): {q}")
    pmids = search_pubmed(q, query_n)
    print(f"Found {len(pmids)} PMIDs")

    downloaded = 0

    for pmid in pmids:
        if downloaded >= target_n:
            break

        rec = {
            "type": "article",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pmid": pmid,
            "pmcid": None,
            "xml_path": None,
            "ok": False,
            "error": None,
        }

        try:
            pmcid = get_pmcid_from_pmid(pmid)
            time.sleep(args.sleep)

            if not pmcid:
                rec["error"] = "no_pmc_link"
                write_jsonl(manifest_path, rec)
                continue

            rec["pmcid"] = f"PMC{pmcid}"

            # Download (or load if already present)
            fetch_or_load_xml(pmcid)
            time.sleep(args.sleep)

            rec["xml_path"] = str((RAW_DIR / f"PMC{pmcid}.xml").resolve())
            rec["ok"] = True
            write_jsonl(manifest_path, rec)

            downloaded += 1
            print(f"[{downloaded}/{target_n}] Downloaded PMC{pmcid} (from PMID {pmid})")

        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            write_jsonl(manifest_path, rec)
            print(f"Failed PMID {pmid}: {rec['error']}")
            time.sleep(args.sleep)

    print(f"Done. Downloaded {downloaded} PMC XML files.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
