_embedding_model = None


def get_embedding_model():
    """Lazily import and return the embedding model.

    This avoids importing heavy dependencies at module import time (useful
    for fast startup of web servers when embeddings aren't immediately needed).
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from langchain_huggingface import HuggingFaceEmbeddings

    _embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True}
    )
    return _embedding_model
