# api/upload.py
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.ingestion.indexer import BM25_ROOT, FAISS_ROOT
from app.ingestion.registry import REGISTRY_ROOT
from app.retrieval.cache import CACHE_ROOT, load_cache, save_cache
from app.schemas.upload import (
    DeleteKnowledgeBaseResponse,
    FileInfo,
    FileListResponse,
    KnowledgeBaseListResponse,
)

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_documents(
    knowledge_base: str = Form(...),
    files: list[UploadFile] = File(...),
):
    safe_kb = Path(knowledge_base).name

    kb_dir = UPLOAD_DIR / safe_kb
    kb_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    failed_files = []

    for file in files:
        try:
            if not file.filename:
                failed_files.append("unknown")
                continue

            safe_filename = Path(file.filename).name

            file_path = kb_dir / safe_filename

            contents = await file.read()

            with open(file_path, "wb") as f:
                f.write(contents)

            uploaded_files.append(safe_filename)

        except Exception as e:
            print(f"Upload failed: {e}")
            failed_files.append(file.filename)  # type: ignore

    return {
        "success": len(failed_files) == 0,
        "message": "Files processed",
        "uploaded_files": uploaded_files,
        "failed_files": failed_files,
    }


@router.get("/files", response_model=FileListResponse)
async def list_files(knowledge_name: str):
    safe_kb = Path(knowledge_name).name
    kb_dir = UPLOAD_DIR / safe_kb

    if not kb_dir.exists():
        return FileListResponse(
            success=False,
            knowledge_base=safe_kb,
            files=[],
        )

    files = []

    for file_path in kb_dir.iterdir():
        if file_path.is_file():
            files.append(
                FileInfo(
                    filename=file_path.name,
                    size=file_path.stat().st_size,
                )
            )

    return FileListResponse(
        success=True,
        knowledge_base=safe_kb,
        files=files,
    )


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases():
    if not UPLOAD_DIR.exists():
        return KnowledgeBaseListResponse(success=True, knowledge_bases=[])

    knowledge_bases = [
        directory.name for directory in UPLOAD_DIR.iterdir() if directory.is_dir()
    ]

    return KnowledgeBaseListResponse(
        success=True, knowledge_bases=sorted(knowledge_bases)
    )


def _semantic_cache_name(knowledge_base: str) -> str:
    # Mirrors app.retrieval.generate_answer._final_answers_cache_name -
    # duplicated (rather than imported) to keep this module free of the
    # heavy LLM/graph import chain that module pulls in.
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", knowledge_base)
    return f"final_answers_semantic__{safe}"


@router.delete("/{knowledge_base}", response_model=DeleteKnowledgeBaseResponse)
async def delete_knowledge_base(knowledge_base: str):
    safe_kb = Path(knowledge_base).name

    target_dirs = {
        "uploads": UPLOAD_DIR / safe_kb,
        "faiss": FAISS_ROOT / safe_kb,
        "bm25": BM25_ROOT / safe_kb,
        "registry": REGISTRY_ROOT / safe_kb,
    }

    if not any(path.exists() for path in target_dirs.values()):
        raise HTTPException(
            status_code=404, detail=f"Knowledge base '{safe_kb}' not found"
        )

    removed_paths = []
    for path in target_dirs.values():
        if path.exists():
            shutil.rmtree(path)
            removed_paths.append(str(path))

    # Drop the per-KB semantic answer cache too, so a future knowledge base
    # created with the same name is never served a stale cached answer
    # generated from the content being deleted here.
    semantic_cache_file = CACHE_ROOT / f"{_semantic_cache_name(safe_kb)}.json"
    if semantic_cache_file.exists():
        semantic_cache_file.unlink()
        removed_paths.append(str(semantic_cache_file))

    # The retrieval-result cache is keyed by a hash of the question text, so
    # entries for this KB can't be targeted directly - filter by what the
    # cached chunks themselves say they belong to.
    pruned_cache_entries = 0
    retrieval_cache = load_cache("retrieval_results")
    if retrieval_cache:
        kept = {}
        for key, cached_docs in retrieval_cache.items():
            belongs_to_kb = any(
                doc.get("metadata", {}).get("knowledge_base") == safe_kb
                for doc in cached_docs
            )
            if belongs_to_kb:
                pruned_cache_entries += 1
                continue
            kept[key] = cached_docs
        if pruned_cache_entries:
            save_cache("retrieval_results", kept)

    return DeleteKnowledgeBaseResponse(
        success=True,
        knowledge_base=safe_kb,
        removed_paths=removed_paths,
        pruned_cache_entries=pruned_cache_entries,
    )
