# services/retrieval/faiss_store.py

from pathlib import Path
from types import SimpleNamespace

from langchain_community.vectorstores import FAISS

from app.ingestion.embedder import get_embedding_model
from app.retrieval.cache import compute_cache_key, get_cached, set_cached

FAISS_ROOT = Path("data/faiss")


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


def retrieve_chunks(knowledge_base: str, question: str, k: int = 5):
    cache_key = compute_cache_key(f"{knowledge_base}:{question}:{k}")
    cached = get_cached("retrieval_results", cache_key)
    if cached is not None:
        # cached is a list of serializable dicts; reconstruct lightweight doc objects
        return [_serializable_to_doc(d) for d in cached]

    vectorstore = load_vectorstore(knowledge_base)
    if vectorstore is None:
        return []

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
    return SimpleNamespace(
        page_content=d.get("page_content"), metadata=d.get("metadata", {})
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
