# Alzheimer’s Disease Biomarkers Q&A RAG Microservice

[![CI](https://github.com/rohitkt10/llm-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/rohitkt10/llm-profiler/actions/workflows/ci.yml)

A specialized Retrieval-Augmented Generation (RAG) system for Alzheimer’s Disease biomarkers research. This application ingests scientific literature (PMC XML), builds a searchable index, and provides a question-answering interface grounded in biomedical evidence.

## Features

*   **Data Pipeline**: Ingests and chunks PMC XML articles.
*   **Vector Search**: Uses FAISS for efficient similarity search with embeddings (Sentence Transformers).
*   **RAG Service**: Orchestrates retrieval and generation using OpenAI or Anthropic LLMs.
*   **Interactive UI**: A Streamlit-based web interface for querying and exploring citations.
*   **Citations**: Provides grounded answers with links to specific source chunks and relevance scores.

## Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rohitpc/ad-rag-app.git
    cd ad-rag-app
    ```

2.  **Set up environment variables:**
    Copy the example environment file and configure your API keys (OpenAI or Anthropic).
    ```bash
    cp .env.example .env
    # Edit .env with your keys
    ```

3.  **Install dependencies:**
    ```bash
    make sync
    # OR using uv directly:
    uv sync
    ```

## Usage

### Running the Web Interface

Launch the Streamlit app to interact with the RAG service:

```bash
uv run streamlit run src/ad_rag_ui/app.py
```

Open your browser at `http://localhost:8501`.

### Development Commands

The project includes a `Makefile` for common tasks:

*   **Run Tests**:
    ```bash
    make test
    ```
*   **Lint Code**:
    ```bash
    make lint
    ```
*   **Format Code**:
    ```bash
    make fmt
    ```

## Project Structure

```
.
├── .github/workflows/  # CI/CD configurations
├── artifacts/          # Generated FAISS indexes and lookup tables
├── data/               # Raw XML data and processed chunks
├── docs/               # Documentation and plans
├── scripts/            # Utility scripts (ingestion, integration testing)
├── src/
│   ├── ad_rag_pipeline/ # Logic for data ingestion, chunking, and indexing
│   ├── ad_rag_service/  # Core RAG service (Retrieval, Generation, LLM adapters)
│   └── ad_rag_ui/       # Streamlit web application
├── tests/              # Unit and integration tests
├── .env.example        # Template for environment variables
├── Makefile            # Command shortcuts
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # Project documentation
```

## Configuration

The application is configured via environment variables (loaded from `.env`). Key settings include:

*   **LLM_PROVIDER**: `openai` or `anthropic`.
*   **OPENAI_API_KEY** / **ANTHROPIC_API_KEY**: Credentials for the chosen provider.
*   **EMBEDDING_MODEL_ID**: HuggingFace model ID for embeddings (default: `BAAI/bge-base-en-v1.5`).
*   **LLM_MODEL_NAME**: Specific model version (e.g., `gpt-5.1` or `claude-sonnet-4-5`).
*   **LLM_TEMPERATURE**: Controls the randomness of LLM outputs (default: `0.3`).
*   **LLM_MAX_TOKENS**: Maximum number of tokens for LLM generation (default: `500`).
*   **REASONING_EFFORT**: (e.g., `none`, `low`, `medium`, `high`) Influences the LLM's reasoning process (default: `none`).
*   **TOP_K**: Number of top similar chunks to retrieve from the index (default: `3`).

See `src/ad_rag_service/config.py` for all available configuration options.
