from __future__ import annotations

import os
from pathlib import Path

import torch

# Resolve repo root relative to this file
REPO_ROOT = Path(__file__).resolve().parents[2]

# Try to load .env
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=True)
except ImportError:
    pass

# Artifacts
# Allow overriding for Cloud Run (e.g. /tmp/artifacts)
_env_artifacts_dir = os.getenv("ARTIFACTS_DIR")
if _env_artifacts_dir:
    ARTIFACTS_DIR = Path(_env_artifacts_dir)
else:
    ARTIFACTS_DIR = REPO_ROOT / "artifacts"

INDEX_DIR = ARTIFACTS_DIR / "index"

# Index Paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
LOOKUP_JSONL_PATH = INDEX_DIR / "lookup.jsonl"
MANIFEST_JSON_PATH = INDEX_DIR / "index.meta.json"

# GCS Config
GCS_BUCKET = os.getenv("GCS_BUCKET")  # If set, artifacts will be downloaded from here

# LLM Config
# --- LLM Provider and Model Selection ---
# To add a new LLM provider:
# 1. Add the provider name to ALLOWED_PROVIDERS.
# 2. Add an 'elif' block below for the new provider to set its default model
#    (e.g., using os.getenv("<NEW_PROVIDER>_MODEL_NAME", "default_model_id")).
# 3. Define the API key environment variable name for the new provider.

ALLOWED_PROVIDERS = {"openai", "anthropic", "dummy"}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

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
        "ANTHROPIC_MODEL_NAME", "claude-sonnet-4-5"
    )  # Updated to reflect latest known
else:  # LLM_PROVIDER == "dummy"
    LLM_MODEL_NAME = "dummy-model"

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
TOP_K = int(os.getenv("TOP_K", "5"))

# Embedding Config (must match pipeline)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
