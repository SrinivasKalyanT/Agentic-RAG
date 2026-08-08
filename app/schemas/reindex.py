from pydantic import BaseModel


class ReindexRequest(BaseModel):
    knowledge_base: str


class ReindexResponse(BaseModel):
    success: bool
    knowledge_base: str
    documents: int
    chunks: int
