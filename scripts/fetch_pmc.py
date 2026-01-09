#!/usr/bin/env python3
"""
Download PMC full-text XMLs for an AD biomarker corpus (via PubMed -> PMC link).
CLI wrapper around ad_rag_pipeline.ingestion.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from ad_rag_pipeline import config
from ad_rag_pipeline.ingestion import fetch_pmc_corpus

# Configure basic logging for CLI
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
        default=str(config.RAW_DIR / "manifest.jsonl"),
        help="Path to JSONL manifest (default=data/raw/manifest.jsonl).",
    )
    args = parser.parse_args()

    email = os.environ.get("NCBI_EMAIL") or os.environ.get("ENTREZ_EMAIL")
    if not email:
        print("Error: Set NCBI_EMAIL (or ENTREZ_EMAIL) in your environment.", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("NCBI_API_KEY") or os.environ.get("ENTREZ_API_KEY")
    query = args.query or default_query(args.start_year, args.end_year)
    out_dir = config.RAW_DIR
    manifest_path = Path(args.manifest).resolve()

    try:
        counts = fetch_pmc_corpus(
            query=query,
            out_dir=out_dir,
            email=email,
            target_n=args.n,
            oversample=args.oversample,
            sleep_s=args.sleep,
            api_key=api_key,
            resume=True,
            manifest_path=manifest_path,
        )
        print(f"Done. Summary: {counts}")
        print(f"Manifest: {manifest_path}")

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
