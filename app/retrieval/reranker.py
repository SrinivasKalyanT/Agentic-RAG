import os
from typing import List, Tuple

from sentence_transformers import CrossEncoder

MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INITIAL_RETRIEVE_K = int(os.getenv("INITIAL_RETRIEVE_K", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))


def _get_text(doc: object) -> str:
    return (
        getattr(doc, "page_content", None)
        or getattr(doc, "content", None)
        or getattr(doc, "text", None)
        or str(doc)
    )


def _load_reranker() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank_docs(
    docs: List,
    question: str,
    top_k: int = TOP_K_RERANK,
) -> List[Tuple[float, object]]:
    """Score and rerank docs using a cross-encoder relevance model.

    Returns a list of docs ordered by descending score. Each item is a tuple
    (score, doc).
    """
    if not docs:
        return []

    reranker = _load_reranker()
    pairs = [(question, _get_text(doc)) for doc in docs]
    scores = reranker.predict(pairs, convert_to_numpy=True).tolist()

    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
