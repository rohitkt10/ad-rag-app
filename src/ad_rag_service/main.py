from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ad_rag_pipeline import config as pipeline_config  # For EMBEDDING_MODEL_ID and DEVICE
from ad_rag_service import config
from ad_rag_service.generator import AnswerGenerator
from ad_rag_service.indexing import IndexStore  # Added import
from ad_rag_service.llm.factory import get_llm_client
from ad_rag_service.retrieval import Retriever
from ad_rag_service.service import RAGService
from ad_rag_service.types import AnswerWithCitations

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
