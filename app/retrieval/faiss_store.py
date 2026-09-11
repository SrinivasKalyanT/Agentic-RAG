# services/retrieval/faiss_store.py

from pathlib import Path

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.ingestion.embedder import get_embedding_model
from app.ingestion.indexer import load_bm25_docs
from app.retrieval.cache import compute_cache_key, get_cached, set_cached

FAISS_ROOT = Path("data/faiss")

# Fuses dense (FAISS) and sparse (BM25) retrieval via weighted Reciprocal
# Rank Fusion instead of dense-only similarity search - helps keyword/exact
# -term queries (IDs, acronyms, rare proper nouns) that embeddings alone
# tend to miss. The BM25 corpus is already built and persisted on every
# reindex (see app.ingestion.indexer.save_bm25_docs / pipeline.py); this was
# previously written but never read back. Knobs live on the central
# Settings (app/core/config.py), same as GROQ_API_KEY etc.
_settings = get_settings()
HYBRID_SEARCH_ENABLED = _settings.hybrid_search_enabled
HYBRID_DENSE_WEIGHT = _settings.hybrid_dense_weight
HYBRID_SPARSE_WEIGHT = _settings.hybrid_sparse_weight


def load_vectorstore(knowledge_base: str):
    path = FAISS_ROOT / knowledge_base
    try:
        embedding_model = get_embedding_model()
    except Exception:
        # If embeddings or heavy ML libs are unavailable, fail gracefully
        # and let retrieve_chunks return an empty list instead of crashing.
        return None

    return FAISS.load_local(
        str(path), embedding_model, allow_dangerous_deserialization=True
    )


def _build_hybrid_retriever(knowledge_base: str, vectorstore, k: int):
    """Return an EnsembleRetriever fusing dense (FAISS) + sparse (BM25)
    results via weighted Reciprocal Rank Fusion, or None if no BM25 corpus
    is available yet (e.g. a knowledge base indexed before this feature
    existed and not yet reindexed) - caller should fall back to dense-only.

    Dedup key is `chunk_id`, which app.ingestion.pipeline.reindex_knowledge_base
    stamps onto every chunk's metadata before it's saved to *both* FAISS and
    the BM25 pickle, so it's guaranteed present on docs from either side.
    """
    bm25_docs = load_bm25_docs(knowledge_base)
    if not bm25_docs:
        return None

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(bm25_docs, k=k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT],
        id_key="chunk_id",
    )


def retrieve_chunks(knowledge_base: str, question: str, k: int = 5):
    cache_key = compute_cache_key(f"{knowledge_base}:{question}:{k}")
    cached = get_cached("retrieval_results", cache_key)
    if cached is not None:
        # cached is a list of serializable dicts; reconstruct lightweight doc objects
        return [_serializable_to_doc(d) for d in cached]

    vectorstore = load_vectorstore(knowledge_base)
    if vectorstore is None:
        return []

    hybrid_retriever = (
        _build_hybrid_retriever(knowledge_base, vectorstore, k)
        if HYBRID_SEARCH_ENABLED
        else None
    )
    if hybrid_retriever is not None:
        # EnsembleRetriever returns the full deduped/fused set, not capped
        # at k, so slice it down to match the dense-only behavior below.
        docs = hybrid_retriever.invoke(question)[:k]
    else:
        docs = vectorstore.similarity_search(question, k=k)

    # store a JSON-serializable representation only
    serializable = [_doc_to_serializable(d) for d in docs]
    set_cached("retrieval_results", cache_key, serializable)

    return docs


def _doc_to_serializable(doc) -> dict:
    return {
        "page_content": getattr(doc, "page_content", None)
        or getattr(doc, "content", None)
        or getattr(doc, "text", None)
        or str(doc),
        "metadata": getattr(doc, "metadata", {}) or {},
    }


def _serializable_to_doc(d: dict):
    # A real Document (not SimpleNamespace) - LangGraph's checkpointer needs
    # to msgpack-serialize `docs` as part of the graph state, and
    # SimpleNamespace isn't serializable that way, unlike Document.
    return Document(
        page_content=d.get("page_content") or "", metadata=d.get("metadata", {})
    )


def build_context(docs):

    contexts = []

    for i, doc in enumerate(docs):

        contexts.append(f"""
Source: {doc.metadata['source']}
Page: {doc.metadata['page']}

{doc.page_content}
""")

    return "\n\n".join(contexts)


def extract_citations(docs: list):

    citations = []

    for doc in docs:
        citations.append(
            {"document": doc.metadata["source"], "page": doc.metadata["page"]}
        )

    return citations
