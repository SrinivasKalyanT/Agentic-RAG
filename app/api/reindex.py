from fastapi import APIRouter

from app.ingestion.pipeline import reindex_knowledge_base
from app.schemas.reindex import ReindexRequest

router = APIRouter()


@router.post("/reindex")
async def reindex(request: ReindexRequest):

    stats = reindex_knowledge_base(request.knowledge_base)

    return {"success": True, "knowledge_base": request.knowledge_base, **stats}
