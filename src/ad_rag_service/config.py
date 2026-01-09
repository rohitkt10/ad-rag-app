from __future__ import annotations

import os
from pathlib import Path

# Resolve repo root relative to this file
REPO_ROOT = Path(__file__).resolve().parents[2]

# Try to load .env
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Artifacts
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"

# Index Paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
LOOKUP_JSONL_PATH = INDEX_DIR / "lookup.jsonl"
MANIFEST_JSON_PATH = INDEX_DIR / "index.meta.json"

# Retrieval Defaults
TOP_K = 5
MAX_CONTEXT_TOKENS = 3000  # Conservative limit for context window

# LLM Config
# Using google/flan-t5-base or similar as default if not specified,
# but for the "service", we might assume an API-based LLM or local.
# The plan mentions "LLM settings".
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")  # Example default
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Embedding Config (must match pipeline)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
