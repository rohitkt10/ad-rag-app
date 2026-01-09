#!/usr/bin/env python3
"""
CLI script to build a FAISS index from text chunks.

Input:
  - JSONL file containing text chunks (default: data/chunks/chunks.jsonl)

Output:
  - artifacts/index/faiss.index
  - artifacts/index/lookup.jsonl
  - artifacts/index/index.meta.json
"""

import argparse
import sys
from pathlib import Path

from ad_rag_pipeline import config
from ad_rag_pipeline.indexing import build_faiss_index_from_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from text chunks.")

    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=config.CHUNKS_DIR / "chunks.jsonl",
        help="Path to input chunks JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.INDEX_DIR,
        help="Directory to save output artifacts.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=config.EMBEDDING_MODEL_ID,
        help="Hugging Face embedding model ID.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.EMBEDDING_DEVICE,
        help="Device to use (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine"],
        help="Similarity metric (only 'cosine' supported currently).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index artifacts if present.",
    )

    args = parser.parse_args()

    try:
        faiss_path, lookup_path, meta_path = build_faiss_index_from_chunks(
            chunks_path=args.chunks_path,
            out_dir=args.out_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            device=args.device,
            metric=args.metric,
            force=args.force,
        )

        print("\nIndex built successfully!")
        print(f"  Index:   {faiss_path}")
        print(f"  Lookup:  {lookup_path}")
        print(f"  Meta:    {meta_path}")

    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
