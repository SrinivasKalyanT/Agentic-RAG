import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.ingestion.embedder import get_embedding_model

FAISS_ROOT = Path("data/faiss")
BM25_ROOT = Path("data/bm25")


def build_faiss(chunks):

    return FAISS.from_documents(chunks, get_embedding_model())


def save_faiss(vector_store, knowledge_base):

    save_path = FAISS_ROOT / knowledge_base

    save_path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(save_path))


def save_bm25_docs(chunks, knowledge_base):

    save_dir = BM25_ROOT / knowledge_base

    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "documents.pkl", "wb") as f:

        pickle.dump(chunks, f)


def load_faiss(knowledge_base: str):
    index_path = FAISS_ROOT / knowledge_base
    if not index_path.exists():
        return None
    return FAISS.load_local(
        str(index_path), get_embedding_model(), allow_dangerous_deserialization=True
    )


def load_bm25_docs(knowledge_base: str):
    bm25_path = BM25_ROOT / knowledge_base / "documents.pkl"
    if not bm25_path.exists():
        return []
    with open(bm25_path, "rb") as f:
        return pickle.load(f)
