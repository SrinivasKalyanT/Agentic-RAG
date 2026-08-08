# schemas/query.py
from operator import add
from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import BaseModel


class QueryRequest(BaseModel):
    knowledge_base: Any
    question: str
    chat_id: str


class Citation(BaseModel):
    document: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class RAGState(TypedDict):
    # ---- core fields ----
    question: str
    knowledge_base: Any

    chat_history: NotRequired[Annotated[list[dict], add]]

    # ---- pipeline fields ----
    docs: NotRequired[list]
    context: NotRequired[str]  # raw merged context
    summarized_context: NotRequired[str]  # compressed by summarize node
    citations: NotRequired[list]
    relevance_score: NotRequired[float]
    verification_score: NotRequired[float]
    answer_supported: NotRequired[bool]
    verification_reasoning: NotRequired[str]
    retrieval_attempts: NotRequired[int]
    answer: NotRequired[str]
    next_action: NotRequired[str]
    agent_reasoning: NotRequired[str]  # free-text scratchpad (useful for debugging)
