# api/upload.py
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.upload import FileInfo, FileListResponse, KnowledgeBaseListResponse

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
