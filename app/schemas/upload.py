# schemas/upload.py

from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    message: str
    # List of filenames that were uploaded successfully
    uploaded_files: list[str]
    # List of filenames that failed to upload
    failed_files: Optional[list[str]] = []


class FileInfo(BaseModel):
    filename: str
    size: int


class FileListResponse(BaseModel):
    success: bool
    knowledge_base: str
    files: list[FileInfo]


class KnowledgeBaseListResponse(BaseModel):
    success: bool
    knowledge_bases: list[str]


class DeleteKnowledgeBaseResponse(BaseModel):
    success: bool
    knowledge_base: str
    # Directories/files actually removed (uploads, FAISS index, BM25 docs,
    # registry, per-KB semantic answer cache).
    removed_paths: list[str]
    # Retrieval-result cache entries dropped because they held chunks from
    # this knowledge base.
    pruned_cache_entries: int
