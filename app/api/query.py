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

    result = await graph.ainvoke(state, config=config)  # type: ignore

    response = {
        "answer": result["answer"],
        "citations": result["citations"],
        "cached_answer": result.get("cached_answer", False),
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

    async def event_generator():
        # graph.ainvoke (not the blocking .invoke) so this request doesn't
        # freeze the whole event loop - other requests keep being served
        # while this one runs. Note this still has to run the pipeline to
        # completion before emitting anything: verify_answer_node can
        # reject/replace the generated answer after the fact, so there's
        # nothing safe to stream token-by-token as it's produced.
        try:
            result = await graph.ainvoke(state, config=config)  # type: ignore
        except Exception as exc:
            yield "event: error\n"
            yield f"data: {json.dumps({'message': str(exc)})}\n\n"
            return

        answer = result.get("answer", "")
        for chunk in answer.splitlines(keepends=True):
            # Repeat "event: answer" before every chunk - SSE only applies
            # the last-seen event name to the data line(s) that follow it,
            # so without this only the first chunk would ever be dispatched
            # as an "answer" event and the rest would arrive as anonymous
            # "message" events instead.
            yield "event: answer\n"
            yield f"data: {chunk.rstrip()}\n\n"

        if "citations" in result:
            citations = result["citations"]
            yield "event: citations\n"
            yield f"data: {json.dumps(citations)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
