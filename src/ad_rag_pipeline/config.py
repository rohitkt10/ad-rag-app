from __future__ import annotations

import os
from pathlib import Path

# Resolve repo root as: <repo>/src/ad_rag_pipeline/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load repo-level .env early (best-effort; no hard dependency)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# Data dirs
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

# Artifacts (production outputs)
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"

# Default chunking params (word-based)
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
MIN_WORDS = 1

# Embeddings
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
_env_device = os.getenv("EMBEDDING_DEVICE")
if _env_device:
    EMBEDDING_DEVICE = _env_device
else:
    try:
        import torch  # type: ignore

        EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        EMBEDDING_DEVICE = "cpu"

# Index artifact filenames
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
LOOKUP_JSONL_PATH = INDEX_DIR / "lookup.jsonl"
MANIFEST_JSON_PATH = INDEX_DIR / "index.meta.json"
