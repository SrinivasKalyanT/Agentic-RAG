import json
import re
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.prompts.agentic_rag_prompts import _DECIDE_SYSTEM
from app.retrieval.cache import (
    compute_cache_key,
    get_cached,
    get_semantic_cached,
    set_cached,
    set_semantic_cached,
)
from app.retrieval.faiss_store import build_context, extract_citations, retrieve_chunks
from app.retrieval.llm import (
    evaluate_context,
    generate_answer,
    generate_llm_response,
    verify_answer,
)
from app.retrieval.reranker import INITIAL_RETRIEVE_K, rerank_docs
from app.schemas.query import RAGState


# Per-turn scratch fields that must never leak from one turn to the next.
# The graph is compiled with a checkpointer keyed by chat_id (thread_id),
# and app/api/query.py's input to graph.invoke/ainvoke only ever supplies
# `question` and `knowledge_base` - LangGraph reloads the *entire* previous
# turn's final state first and only overwrites keys present in each node's
# return value, so anything not reset here keeps last turn's value. That
# previously let a cache *miss* on `check_cached_answer_node` (which
# returns {} on a miss, touching nothing) fall through to
# `route_from_cached_answer` seeing the *previous* turn's leftover
# `answer` and treating it as this turn's answer, skipping retrieval
# entirely and returning a stale, unrelated answer. `chat_history` is
# deliberately excluded - it uses an `add` reducer and is meant to
# accumulate across turns.
_TURN_RESET: dict = {
    "answer": "",
    "citations": [],
    "cached_answer": False,
    "docs": [],
    "context": "",
    "summarized_context": "",
    "relevance_score": 0.0,
    "verification_score": 0.0,
    "answer_supported": False,
    "verification_reasoning": "",
    "retrieval_attempts": 0,
    "next_action": "",
    "agent_reasoning": "",
    "reranker_scores": [],
}


def memory_inject_node(state: RAGState) -> dict:
    """
    Reset the per-turn scratch fields listed in `_TURN_RESET` (see its
    comment), then prepend the last 3 turns of chat history to the
    question so the LLM has conversational context when generating the
    final answer.

    This writes `augmented_question`, not `question`. If it overwrote
    `question`, every downstream node (reformulate, check_cached_answer,
    retrieve, agent_decide, cache_answer, verify_answer) would operate on a
    string dominated by the prior turn's (often much longer) answer text -
    a topic switch within the same chat_id would then match unrelated
    cache entries and retrieve unrelated docs, because the embedding of
    "prior Q + prior long A + new question" is mostly about the prior
    topic. Only generate_node reads `augmented_question`, so history only
    ever influences how the final answer is phrased, never what gets
    retrieved or cache-matched.
    """
    updates: dict = dict(_TURN_RESET)

    history = state.get("chat_history") or []
    if not history:
        return updates

    recent = history[-3:]
    history_text = "\n".join(
        f"User: {h['question']}\nAssistant: {h['answer']}" for h in recent
    )
    updates["augmented_question"] = (
        f"Conversation so far:\n{history_text}\n\n" f"New question: {state['question']}"
    )
    return updates


def reformulate_node(state: RAGState) -> dict:
    prompt = (
        f"Rewrite the following question to improve document retrieval.\n"
        f"Return only the rewritten question, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    cache_key = compute_cache_key(f"rewrite:{state['question']}")
    cached = get_cached("rewrite_results", cache_key)
    if cached is not None:
        return {"question": cached}

    reformulated = generate_llm_response(prompt)
    rewritten = reformulated.strip()  # type: ignore
    set_cached("rewrite_results", cache_key, rewritten)
    return {"question": rewritten}  # type: ignore


# ---------------------------------------------------------------------------
# Node 3 — retrieve
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    attempts = state.get("retrieval_attempts") or 0
    docs = retrieve_chunks(
        state["knowledge_base"], state["question"], k=INITIAL_RETRIEVE_K
    )
    return {
        "docs": docs,
        "retrieval_attempts": attempts + 1,
        "summarized_context": "",
    }


def _final_answers_cache_name(knowledge_base) -> str:
    """Per-knowledge-base cache name so semantic matches never cross KBs."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(knowledge_base))
    return f"final_answers_semantic__{safe}"


def check_cached_answer_node(state: RAGState) -> dict:
    question = state.get("question")
    knowledge_base = state.get("knowledge_base")
    if question is None or knowledge_base is None:
        return {}

    cache_name = _final_answers_cache_name(knowledge_base)
    cached = get_semantic_cached(cache_name, question)
    if not cached:
        return {}

    return {
        "answer": cached.get("answer"),
        "citations": cached.get("citations", []),
        "answer_supported": True,
        "verification_score": float(cached.get("verification_score", 1.0)),
        "verification_reasoning": cached.get(
            "verification_reasoning", "Reused cached supported answer."
        ),
        "cached_answer": True,
    }


def rerank_node(state: RAGState) -> dict:
    """Re-ranks the `docs` list using the reranker and returns the top-k docs."""
    docs = state.get("docs") or []
    question = state.get("question", "")
    if not docs:
        return {}

    scored = rerank_docs(docs, question)
    new_docs = [d for _, d in scored]
    scores = [float(s) for s, _ in scored]
    return {"docs": new_docs, "reranker_scores": scores}


# ---------------------------------------------------------------------------
# Node 4 — build context
# ---------------------------------------------------------------------------


def context_node(state: RAGState) -> dict:
    docs = state.get("docs")
    if not docs:
        raise ValueError("docs missing before build_context")
    context = build_context(docs)
    citations = extract_citations(docs)
    reranker_scores = state.get("reranker_scores") or []
    return {
        "context": context,
        "citations": citations,
        "reranker_scores": reranker_scores,
    }


def summarize_node(state: RAGState) -> dict:
    """
    Compress the raw retrieved context into a dense passage before evaluation.
    This prevents the LLM from drowning in boilerplate and keeps token usage low.
    """
    context = state.get("context", "")
    question = state.get("question", "")

    if not context:
        return {"summarized_context": ""}

    prompt = (
        f"Summarize the following document excerpts, keeping only information "
        f"relevant to answering: '{question}'\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Return a concise summary (max 300 words). Preserve key facts and numbers."
    )
    summary = generate_llm_response(prompt)
    return {"summarized_context": summary.strip()}  # type: ignore


def agent_decide_node(state: RAGState) -> dict:
    """
    LLM reads the full pipeline state and picks the next action.
    This replaces the hard-coded relevance threshold.
    """
    # Build a compact state snapshot for the LLM
    state_snapshot = {
        "question": state.get("question", ""),
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        "has_docs": bool(state.get("docs")),
        "docs_count": len(state.get("docs", [])),
        "context_length": len(state.get("context", "")),
        "summarized_context_length": len(state.get("summarized_context", "")),
        "summarized_context_preview": (state.get("summarized_context") or "")[:300],
        "relevance_score": state.get("relevance_score"),
    }

    prompt = (
        f"{_DECIDE_SYSTEM}\n\n"
        f"Current pipeline state:\n{json.dumps(state_snapshot, indent=2)}"
    )

    raw = generate_llm_response(prompt)

    try:
        parsed = json.loads(raw.strip())  # type: ignore
        next_action = parsed.get("next_action", "no_answer")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback: safe default
        next_action = "no_answer"
        reasoning = f"JSON parse failed on: {raw[:100]}"  # type: ignore

    # Hard guard: never loop forever
    if next_action == "retrieve" and (state.get("retrieval_attempts") or 0) >= 3:
        next_action = "no_answer"
        reasoning = "Max retrieval attempts reached, overriding to no_answer"

    return {"next_action": next_action, "agent_reasoning": reasoning}


def evaluate_context_citations_node(state: RAGState):

    question = state.get("question")
    if question is None:
        raise ValueError("Missing question in state before evaluating")

    context = state.get("context")
    if context is None:
        raise ValueError("Missing context in state before evaluating")

    relevance_score = evaluate_context(question, context)
    score = float(relevance_score.strip())  # type: ignore
    return {"relevance_score": score}


# After — validate membership then cast
_VALID_ACTIONS = frozenset({"retrieve", "summarize", "generate", "no_answer"})


def route_from_decision(
    state: RAGState,
) -> Literal["retrieve", "summarize", "generate", "no_answer"]:
    from typing import cast

    action = state.get("next_action") or "no_answer"
    if action not in _VALID_ACTIONS:
        action = "no_answer"
    return cast(Literal["retrieve", "summarize", "generate", "no_answer"], action)


def route_from_cached_answer(state: RAGState) -> Literal["retrieve", "update_history"]:
    return "update_history" if state.get("answer") else "retrieve"


# ---------------------------------------------------------------------------
# Node 7 — generate
# ---------------------------------------------------------------------------


def generate_node(state: RAGState):
    context = state.get("summarized_context") or state.get("context")
    if not context:
        raise ValueError("Missing context in state before generating answer")

    # Use the history-augmented question (if any) only here, so the answer
    # can be phrased with conversational context, without that history
    # having influenced which docs were retrieved or which cache entry
    # matched - see memory_inject_node.
    question = state.get("augmented_question") or state["question"]
    answer = generate_answer(question=question, context=context)
    return {"answer": answer}


def verify_answer_node(state: RAGState) -> dict:
    question = state.get("question")
    context = state.get("context")
    answer = state.get("answer")

    if question is None or context is None or answer is None:
        raise ValueError("Missing question, context, or answer before verification")

    assert isinstance(question, str)
    assert isinstance(context, str)
    assert isinstance(answer, str)

    raw = verify_answer(question=question, context=context, answer=answer)
    assert isinstance(raw, str)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        parsed = {}

    supported = parsed.get("supported", False)
    confidence = float(parsed.get("confidence", 0.0))
    reasoning = parsed.get("reasoning", "Could not parse verification result.")

    return {
        "answer_supported": bool(supported),
        "verification_score": confidence,
        "verification_reasoning": reasoning,
    }


def route_after_verify(state: RAGState) -> Literal["cache_answer", "no_answer"]:
    return "cache_answer" if state.get("answer_supported") else "no_answer"


def cache_answer_node(state: RAGState) -> dict:
    question = state.get("question")
    knowledge_base = state.get("knowledge_base")
    answer = state.get("answer")
    citations = state.get("citations", [])
    verification_score = state.get("verification_score", 1.0)
    verification_reasoning = state.get(
        "verification_reasoning", "Cached after successful verification."
    )

    if (
        question is None
        or knowledge_base is None
        or answer is None
        or not state.get("answer_supported")
    ):
        return {}

    cache_name = _final_answers_cache_name(knowledge_base)
    set_semantic_cached(
        cache_name,
        question,
        {
            "answer": answer,
            "citations": citations,
            "verification_score": verification_score,
            "verification_reasoning": verification_reasoning,
        },
    )
    return {}


# ---------------------------------------------------------------------------
# Node 8 — no_answer
# ---------------------------------------------------------------------------


def no_answer_node(state: RAGState) -> dict:
    return {
        "answer": "I could not find a relevant answer in the available documents.",
        "citations": [],
    }


# ---------------------------------------------------------------------------
# Node 9 — update history  (writes back to Scope-2 memory)
# ---------------------------------------------------------------------------


def update_history_node(state: RAGState) -> dict:
    """
    Appends the current Q+A pair to chat_history.
    Because chat_history uses Annotated[list, add], LangGraph merges this
    across checkpointed turns automatically — no manual dedup needed.
    """
    entry = {
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
    }
    return {"chat_history": [entry]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> CompiledStateGraph[RAGState, None, RAGState, RAGState]:
    graph_builder = StateGraph(RAGState)

    # Register nodes
    graph_builder.add_node("memory_inject", memory_inject_node)
    graph_builder.add_node("reformulate", reformulate_node)
    graph_builder.add_node("check_cached_answer", check_cached_answer_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("build_context", context_node)
    graph_builder.add_node("evaluate_context", evaluate_context_citations_node)
    graph_builder.add_node("summarize", summarize_node)  # NEW
    graph_builder.add_node("agent_decide", agent_decide_node)  # NEW
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("verify_answer", verify_answer_node)
    graph_builder.add_node("cache_answer", cache_answer_node)
    graph_builder.add_node("no_answer", no_answer_node)
    graph_builder.add_node("update_history", update_history_node)  # NEW

    # Linear entry path
    graph_builder.add_edge(START, "memory_inject")
    graph_builder.add_edge("memory_inject", "reformulate")
    graph_builder.add_edge("reformulate", "check_cached_answer")
    graph_builder.add_edge("retrieve", "rerank")
    graph_builder.add_edge("rerank", "build_context")
    graph_builder.add_edge("build_context", "evaluate_context")
    graph_builder.add_edge("evaluate_context", "agent_decide")  # first decision point

    graph_builder.add_conditional_edges(
        "check_cached_answer",
        route_from_cached_answer,
        {
            "retrieve": "retrieve",
            "update_history": "update_history",
        },
    )

    # Agentic conditional routing
    graph_builder.add_conditional_edges(
        "agent_decide",
        route_from_decision,
        {
            "retrieve": "retrieve",  # loop back for better docs
            "summarize": "summarize",  # compress context first
            "generate": "generate",  # good enough, answer now
            "no_answer": "no_answer",  # give up gracefully
        },
    )

    # After summarize, re-evaluate
    graph_builder.add_edge("summarize", "agent_decide")

    # After generation, verify the answer before committing to history
    graph_builder.add_edge("generate", "verify_answer")
    graph_builder.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {
            "cache_answer": "cache_answer",
            "no_answer": "no_answer",
        },
    )
    # after caching a supported answer, continue to history
    graph_builder.add_edge("cache_answer", "update_history")

    # Terminal paths both flow through update_history before END
    graph_builder.add_edge("no_answer", "update_history")
    graph_builder.add_edge("update_history", END)

    # Compile with MemorySaver for Scope-2 session memory
    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)  # type: ignore


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

graph = build_graph()
