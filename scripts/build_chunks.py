#!/usr/bin/env python3
"""
Build section-aware text chunks from PMC full-text XMLs.
CLI wrapper around ad_rag_pipeline.chunking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ad_rag_pipeline import config
from ad_rag_pipeline.chunking import build_chunks_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text chunks from raw PMC XMLs.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.RAW_DIR / "manifest.jsonl",
        help="Path to manifest.jsonl (to map PMCID -> PMID).",
    )
    parser.add_argument("--out-dir", type=Path, default=config.CHUNKS_DIR)
    parser.add_argument("--chunk-size-words", type=int, default=config.CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap-words", type=int, default=config.CHUNK_OVERLAP_WORDS)
    parser.add_argument("--min-words", type=int, default=config.MIN_WORDS)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")

    args = parser.parse_args()

    # Minimal validation
    if not args.raw_dir.exists():
        print(f"Error: Raw dir not found: {args.raw_dir}", file=sys.stderr)
        sys.exit(2)

    chunks_file = args.out_dir / "chunks.jsonl"
    if chunks_file.exists() and not args.force:
        print(
            f"Error: Output file exists: {chunks_file}. Use --force to overwrite.", file=sys.stderr
        )
        sys.exit(2)

    try:
        out_path, meta_path = build_chunks_dataset(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            chunk_size_words=args.chunk_size_words,
            overlap_words=args.overlap_words,
            min_words=args.min_words,
            manifest_path=args.manifest,
        )
        print(f"Done. Chunks written to: {out_path}")
        print(f"Metadata written to: {meta_path}")

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
