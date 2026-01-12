import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ad_rag_service import config

# Config overrides for test
config.LLM_PROVIDER = "openai"
config.LLM_MODEL_NAME = "gpt-5.1"

from ad_rag_service.indexing import IndexStore
from ad_rag_service.retrieval import Retriever
from ad_rag_service.generator import AnswerGenerator
from ad_rag_service.llm.factory import get_llm_client
from ad_rag_service.service import RAGService

QUERY = "What are the most predictive CSF biomarkers of AD progression?"

# Setup
index_store = IndexStore(
    index_path=config.FAISS_INDEX_PATH,
    lookup_path=config.LOOKUP_JSONL_PATH,
    meta_path=config.MANIFEST_JSON_PATH,
)
index_store.load()

# check that retrieval works correctly
retriever = Retriever(
    index_store=index_store, model_id=config.EMBEDDING_MODEL_ID, device=config.EMBEDDING_DEVICE
)

config.LLM_TEMPERATURE = 0.1
config.LLM_MAX_TOKENS = 500
config.TOP_K = 3
llm_client = get_llm_client()
generator = AnswerGenerator(llm_client)
service = RAGService(index_store, retriever, generator)

res = service.answer(query=QUERY, k=config.TOP_K)
print(res)
