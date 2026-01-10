from __future__ import annotations

import os
from pathlib import Path

# Resolve repo root relative to this file
REPO_ROOT = Path(__file__).resolve().parents[2]

# Try to load .env
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
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
# --- LLM Provider and Model Selection ---
# To add a new LLM provider:
# 1. Add the provider name to ALLOWED_PROVIDERS.
# 2. Add an 'elif' block below for the new provider to set its default model
#    (e.g., using os.getenv("<NEW_PROVIDER>_MODEL_NAME", "default_model_id")).
# 3. Define the API key environment variable name for the new provider.

ALLOWED_PROVIDERS = {"openai", "anthropic", "dummy"}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dummy")

if LLM_PROVIDER not in ALLOWED_PROVIDERS:
    raise ValueError(
        f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of {list(ALLOWED_PROVIDERS)}"
    )

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

if LLM_PROVIDER == "openai":
    LLM_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.1")
elif LLM_PROVIDER == "anthropic":
    LLM_MODEL_NAME = os.getenv(
        "ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet"
    )  # Updated to reflect latest known
else:  # LLM_PROVIDER == "dummy"
    LLM_MODEL_NAME = "dummy-model"

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Embedding Config (must match pipeline)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
