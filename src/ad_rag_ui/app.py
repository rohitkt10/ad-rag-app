import streamlit as st

from ad_rag_service import config
from ad_rag_service.generator import AnswerGenerator
from ad_rag_service.indexing import IndexStore
from ad_rag_service.llm.factory import get_llm_client
from ad_rag_service.retrieval import Retriever
from ad_rag_service.service import RAGService

st.set_page_config(page_title="AD Biomarker RAG", page_icon="🧠")

st.title("AD Biomarker RAG")


@st.cache_resource(show_spinner="Initializing...")
def get_rag_service() -> RAGService:
    """Initialize RAG Service once and cache it."""
    if not config.FAISS_INDEX_PATH.exists():
        st.error(
            f"Index not found at {config.FAISS_INDEX_PATH}. Please run the indexing pipeline first."
        )
        st.stop()

    index_store = IndexStore(
        index_path=config.FAISS_INDEX_PATH,
        lookup_path=config.LOOKUP_JSONL_PATH,
        meta_path=config.MANIFEST_JSON_PATH,
    )
    index_store.load()

    retriever = Retriever(
        index_store=index_store,
        model_id=config.EMBEDDING_MODEL_ID,
        device=config.EMBEDDING_DEVICE,
    )

    llm_client = get_llm_client()
    generator = AnswerGenerator(llm_client)

    return RAGService(index_store, retriever, generator)


def init_session_state() -> None:
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "answer_result" not in st.session_state:
        st.session_state.answer_result = None


def clear_output() -> None:
    st.session_state.query = ""
    st.session_state.answer_result = None


init_session_state()

try:
    service = get_rag_service()
except Exception as e:
    st.error(f"Failed to initialize service: {e}")
    st.stop()

with st.form("rag_form"):
    query_input = st.text_input(
        "Enter your query here:",
        value=st.session_state.query,
        key="query_input_field",
    )

    col_left, col_mid, col_right = st.columns([3, 2, 3])
    with col_mid:
        btn_col1, btn_col2 = st.columns([1, 1], gap="small")
        with btn_col1:
            submitted = st.form_submit_button("Ask", use_container_width=True)
        with btn_col2:
            clear_button = st.form_submit_button("Clear", use_container_width=True)

if clear_button:
    clear_output()
    st.rerun()

if submitted:
    st.session_state.query = query_input
    if st.session_state.query:
        with st.spinner("Analyzing documents..."):
            try:
                st.session_state.answer_result = service.answer(st.session_state.query)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.answer_result = None
    else:
        st.warning("Please enter a query.")
        st.session_state.answer_result = None

if st.session_state.answer_result:
    result = st.session_state.answer_result

    st.markdown("### Answer")
    st.markdown(result.answer)

    with st.expander("📚 Sources Used"):
        if not result.context_used:
            st.info("No specific documents were retrieved.")
        else:
            for i, chunk in enumerate(result.context_used, start=1):
                st.markdown(f"**[{i}] {chunk.record.pmcid}** ({chunk.record.section_title})")
                st.caption(f"Relevance Score: {chunk.score:.4f}")
                with st.expander("Show Text"):
                    st.text(chunk.record.text)
                st.divider()
