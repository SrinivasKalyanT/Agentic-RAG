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
