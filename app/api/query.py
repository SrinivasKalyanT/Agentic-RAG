# api/query.py
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.retrieval.generate_answer import RAGState, graph
from app.schemas.query import QueryRequest

router = APIRouter()


@router.post("/")
async def query(request: QueryRequest):

    state: RAGState = {
        "knowledge_base": request.knowledge_base,
        "question": request.question,
    }
    config = {"configurable": {"thread_id": request.chat_id}}

    result = graph.invoke(state, config=config)  # type: ignore

    response = {
        "answer": result["answer"],
        "citations": result["citations"],
    }
    if "reranker_scores" in result:
        response["reranker_scores"] = result["reranker_scores"]

    return response


@router.post("/stream")
async def query_stream(request: QueryRequest):
    state: RAGState = {
        "knowledge_base": request.knowledge_base,
        "question": request.question,
    }
    config = {"configurable": {"thread_id": request.chat_id}}

    result = graph.invoke(state, config=config)  # type: ignore

    async def event_generator():
        yield "event: answer\n"
        answer = result.get("answer", "")
        for chunk in answer.splitlines(keepends=True):
            yield f"data: {chunk.rstrip()}\n\n"
        if "citations" in result:
            citations = result["citations"]
            yield "event: citations\n"
            yield f"data: {json.dumps(citations)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
