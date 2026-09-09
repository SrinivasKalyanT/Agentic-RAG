from pathlib import Path

from app.ingestion.factory import get_strategy
from app.ingestion.indexer import (
    build_faiss,
    load_bm25_docs,
    load_faiss,
    save_bm25_docs,
    save_faiss,
)
from app.ingestion.registry import DocumentRegistry

UPLOAD_ROOT = Path("data/uploads")


def reindex_knowledge_base(knowledge_base: str):

    upload_dir = UPLOAD_ROOT / knowledge_base
    registry = DocumentRegistry(knowledge_base)
    new_files, modified_files = registry.get_changed_documents(upload_dir)
    files_to_process = new_files + modified_files

    strategy = get_strategy()

    all_chunks = []
    processed_docs = 0

    for pdf_file, file_hash in files_to_process:
        chunks = strategy.load_and_chunk(pdf_file)
        document_id = (
            registry.get_document_id(pdf_file.name) or registry.generate_document_id()
        )
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "source": pdf_file.name,
                    "chunk_id": f"{document_id}_chunk_{idx}",
                    "knowledge_base": knowledge_base,
                    "file_hash": file_hash,
                }
            )
        all_chunks.extend(chunks)
        registry.register_document(
            filename=pdf_file.name, file_hash=file_hash, document_id=document_id
        )
        processed_docs += 1

    if len(all_chunks) == 0:
        return {"documents": 0, "chunks": 0}

    # Load existing index if it exists and merge, don't overwrite
    existing_index = load_faiss(knowledge_base)  # returns None if not found
    if existing_index is not None:
        existing_index.add_documents(all_chunks)
        save_faiss(existing_index, knowledge_base)
    else:
        vector_store = build_faiss(all_chunks)
        save_faiss(vector_store, knowledge_base)

    # Load existing BM25 docs and merge
    existing_bm25_docs = load_bm25_docs(knowledge_base)  # returns [] if not found
    save_bm25_docs(existing_bm25_docs + all_chunks, knowledge_base)

    return {"documents": processed_docs, "chunks": len(all_chunks)}
