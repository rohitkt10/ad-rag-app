FROM python:3.12-slim

# Install system dependencies
# libgomp1 is often required for FAISS/PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (excluding the project itself to allow caching)
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download embedding model to bake it into the image
ENV HF_HOME=/models
# Use the virtual environment's python directly to avoid uv trying to sync the project
# (which would fail because src is missing at this stage)
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Copy source code
COPY src/ ./src

# Install the project itself
RUN uv sync --frozen --no-dev

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Entrypoint
# We use shell form to allow variable expansion for $PORT (Cloud Run requirement)
CMD uv run uvicorn ad_rag_service.main:app --host 0.0.0.0 --port ${PORT:-8080}
