from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ad_rag_pipeline import config as pipeline_config
from ad_rag_service import config
from ad_rag_service.generator import AnswerGenerator
from ad_rag_service.indexing import IndexStore
from ad_rag_service.llm.factory import get_llm_client
from ad_rag_service.retrieval import Retriever
from ad_rag_service.service import RAGService
from ad_rag_service.types import AnswerWithCitations, RetrievedChunk

# Configure basic logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


# Store the RAGService instance globally
rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    global rag_service

    logger.info("Service startup: Initializing RAG components...")
    try:
        # Phase 1: IndexStore
        index_store = IndexStore(
            index_path=config.FAISS_INDEX_PATH,
            lookup_path=config.LOOKUP_JSONL_PATH,
            meta_path=config.MANIFEST_JSON_PATH,
        )
        index_store.load()
        logger.info("IndexStore loaded.")

        # Phase 2: Retriever
        # Using pipeline config for embedding model defaults for consistency
        retriever = Retriever(
            index_store=index_store,
            model_id=pipeline_config.EMBEDDING_MODEL_ID,
            device=pipeline_config.EMBEDDING_DEVICE,
        )
        logger.info("Retriever initialized.")

        # Phase 3: Generator
        llm_client = get_llm_client()
        generator = AnswerGenerator(llm_client)
        logger.info("Generator initialized.")

        # Phase 4: RAGService Orchestrator
        rag_service = RAGService(
            index_store=index_store,
            retriever=retriever,
            generator=generator,
        )
        logger.info("RAGService fully initialized.")

    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # In a real app, you might want to exit or provide a degraded service.
        # For this MVP, we raise, which will prevent the app from starting.
        raise RuntimeError(f"Service initialization failed: {e}") from e

    yield  # Application runs

    logger.info("Service shutdown: Cleaning up resources (if any)...")
    # No explicit cleanup needed for FAISS or SentenceTransformer;
    # they manage their own resources.


app = FastAPI(
    title="AD Biomarker RAG Service",
    description=(
        "A microservice for Retrieval-Augmented Generation on Alzheimer's Disease"
        " biomarker literature."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.
    """
    if rag_service is None or rag_service.index_store.index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized or index not loaded.",
        )
    return {"status": "ok", "index_loaded": rag_service.index_store.index.is_trained}


def _file_info(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_unix": stat.st_mtime,
    }


def _read_json_if_exists(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/metadata", status_code=status.HTTP_200_OK)
async def metadata():
    """
    Deployment/debug metadata: artifact presence + (optional) manifest content.
    """
    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_model_name": config.LLM_MODEL_NAME,
        "embedding_model_id": pipeline_config.EMBEDDING_MODEL_ID,
        "embedding_device": pipeline_config.EMBEDDING_DEVICE,
        "top_k_default": config.TOP_K,
        "artifacts": {
            "faiss_index": _file_info(str(config.FAISS_INDEX_PATH)),
            "lookup_jsonl": _file_info(str(config.LOOKUP_JSONL_PATH)),
            "manifest_json": _file_info(str(config.MANIFEST_JSON_PATH)),
        },
        "manifest": _read_json_if_exists(str(config.MANIFEST_JSON_PATH)),
    }


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=1000)


@app.post("/query", response_model=AnswerWithCitations)
async def query_rag_service(request: QueryRequest):
    """
    Query the RAG service with a question about AD biomarker literature.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized.",
        )
    if not request.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty."
        )

    try:
        answer_with_citations = rag_service.answer(request.question)
        return answer_with_citations
    except Exception as e:
        logger.exception("Error processing query.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}",
        ) from e


class RetrieveRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    k: int = Field(default=config.TOP_K, ge=1)


@app.post("/retrieve", response_model=list[RetrievedChunk])
async def retrieve_only(request: RetrieveRequest):
    """
    Retrieval-only endpoint (no LLM). Returns top-k chunks with scores + metadata.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized.",
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        return rag_service.retriever.retrieve(request.query, request.k)
    except Exception as e:
        logger.exception("Error processing retrieve.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}",
        ) from e
