"""
Agentic RAG Graph
=================
Upgrades from the original pipeline:
  1. LLM-driven routing  — an `agent_decide` node reads the full state and
     chooses the next step instead of hard-coded threshold logic.
  2. Context summarization — a `summarize` node compresses retrieved chunks
     before evaluation, keeping the LLM context window clean.
  3. Scope-2 session memory — MemorySaver checkpointer + chat_history in
     RAGState lets every turn see prior Q&A. Pass a `thread_id` in config.

Usage
-----
    graph = build_graph()

    config = {"configurable": {"thread_id": "user-123"}}

    # Turn 1
    result = graph.invoke({"question": "What is our refund policy?"}, config)

    # Turn 2 — agent sees the prior exchange automatically
    result = graph.invoke({"question": "What about digital products?"}, config)
"""

from __future__ import annotations

import json
from operator import add
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Stubs — replace with your real implementations
# ---------------------------------------------------------------------------
from app.retrieval.faiss_store import build_context, extract_citations, retrieve_chunks
from app.retrieval.llm import evaluate_context, generate_answer


def _llm_call(prompt: str) -> str:
    """Thin wrapper around your LLM. Replace with your actual call."""
    raise NotImplementedError("Wire up your LLM here")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    # ---- core fields ----
    question: str
    knowledge_base: Any

    # ---- session memory (Scope 2) ----
    # Annotated[list, add] means LangGraph *appends* across checkpointed turns
    # rather than overwriting.
    chat_history: Annotated[list[dict], add]

    # ---- pipeline fields ----
    docs: list
    context: str  # raw merged context
    summarized_context: str  # compressed by summarize node
    citations: list
    relevance_score: float
    retrieval_attempts: int
    answer: str

    # ---- agentic decision ----
    # The LLM decision node writes one of:
    #   "retrieve" | "summarize" | "generate" | "no_answer"
    next_action: str
    agent_reasoning: str  # free-text scratchpad (useful for debugging)


# ---------------------------------------------------------------------------
# Node 1 — inject chat history into the question
# ---------------------------------------------------------------------------


def memory_inject_node(state: RAGState) -> dict:
    """
    Prepend the last 3 turns of chat history to the question so the LLM has
    conversational context when reformulating and generating.
    """
    history = state.get("chat_history") or []
    if not history:
        return {}

    recent = history[-3:]
    history_text = "\n".join(
        f"User: {h['question']}\nAssistant: {h['answer']}" for h in recent
    )
    rewritten = (
        f"Conversation so far:\n{history_text}\n\n" f"New question: {state['question']}"
    )
    return {"question": rewritten}


# ---------------------------------------------------------------------------
# Node 2 — reformulate
# ---------------------------------------------------------------------------


def reformulate_node(state: RAGState) -> dict:
    prompt = (
        f"Rewrite the following question to improve document retrieval.\n"
        f"Return only the rewritten question, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    reformulated = _llm_call(prompt)
    return {"question": reformulated.strip()}


# ---------------------------------------------------------------------------
# Node 3 — retrieve
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    attempts = state.get("retrieval_attempts") or 0
    docs = retrieve_chunks(state["knowledge_base"], state["question"])
    return {
        "docs": docs,
        "retrieval_attempts": attempts + 1,
    }


# ---------------------------------------------------------------------------
# Node 4 — build context
# ---------------------------------------------------------------------------


def context_node(state: RAGState) -> dict:
    docs = state.get("docs")
    if not docs:
        raise ValueError("docs missing before build_context")
    context = build_context(docs)
    citations = extract_citations(docs)
    return {"context": context, "citations": citations}


# ---------------------------------------------------------------------------
# Node 5 — summarize  (NEW)
# ---------------------------------------------------------------------------


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
    summary = _llm_call(prompt)
    return {"summarized_context": summary.strip()}


# ---------------------------------------------------------------------------
# Node 6 — AGENT DECIDE  (replaces hard-coded route_relevance)
# ---------------------------------------------------------------------------

_DECIDE_SYSTEM = """
You are the routing brain of a RAG pipeline. Given the current state, decide
the single best next action.

Respond with a JSON object ONLY — no markdown, no explanation:
{
  "next_action": "<one of: retrieve | summarize | generate | no_answer>",
  "reasoning": "<one sentence>"
}

Rules:
- "retrieve"   — context is missing, empty, or clearly off-topic; AND
                  retrieval_attempts < 3
- "summarize"  — docs retrieved but context not yet summarized; always run
                  summarize before generate on the first pass
- "generate"   — summarized_context looks relevant and sufficient
- "no_answer"  — retrieval_attempts >= 3 and context still inadequate,
                  or question is unanswerable from available docs
"""


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
        "context_length": len(state.get("context", "")),
        "summarized_context_length": len(state.get("summarized_context", "")),
        "summarized_context_preview": (state.get("summarized_context") or "")[:300],
        "relevance_score": state.get("relevance_score"),
    }

    prompt = (
        f"{_DECIDE_SYSTEM}\n\n"
        f"Current pipeline state:\n{json.dumps(state_snapshot, indent=2)}"
    )

    raw = _llm_call(prompt)

    try:
        parsed = json.loads(raw.strip())
        next_action = parsed.get("next_action", "no_answer")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback: safe default
        next_action = "no_answer"
        reasoning = f"JSON parse failed on: {raw[:100]}"

    # Hard guard: never loop forever
    if next_action == "retrieve" and (state.get("retrieval_attempts") or 0) >= 3:
        next_action = "no_answer"
        reasoning = "Max retrieval attempts reached, overriding to no_answer"

    return {"next_action": next_action, "agent_reasoning": reasoning}


def route_from_decision(
    state: RAGState,
) -> Literal["retrieve", "summarize", "generate", "no_answer"]:
    """Conditional edge: reads next_action set by agent_decide_node."""
    return state.get("next_action", "no_answer")


# ---------------------------------------------------------------------------
# Node 7 — generate
# ---------------------------------------------------------------------------


def generate_node(state: RAGState) -> dict:
    context = state.get("summarized_context") or state.get("context")
    if not context:
        raise ValueError("context missing before generate")
    answer = generate_answer(question=state["question"], context=context)
    return {"answer": answer}


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


def build_graph() -> StateGraph:
    graph_builder = StateGraph(RAGState)

    # Register nodes
    graph_builder.add_node("memory_inject", memory_inject_node)
    graph_builder.add_node("reformulate", reformulate_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("build_context", context_node)
    graph_builder.add_node("summarize", summarize_node)  # NEW
    graph_builder.add_node("agent_decide", agent_decide_node)  # NEW
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("no_answer", no_answer_node)
    graph_builder.add_node("update_history", update_history_node)  # NEW

    # Linear entry path
    graph_builder.add_edge(START, "memory_inject")
    graph_builder.add_edge("memory_inject", "reformulate")
    graph_builder.add_edge("reformulate", "retrieve")
    graph_builder.add_edge("retrieve", "build_context")
    graph_builder.add_edge("build_context", "agent_decide")  # first decision point

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

    # Terminal paths both flow through update_history before END
    graph_builder.add_edge("generate", "update_history")
    graph_builder.add_edge("no_answer", "update_history")
    graph_builder.add_edge("update_history", END)

    # Compile with MemorySaver for Scope-2 session memory
    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

graph = build_graph()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-session-1"}}

    from IPython.display import Image, display

    display(Image(graph.get_graph().draw_mermaid_png()))
    # Turn 1
    # r1 = graph.invoke(
    #     {"question": "What is the refund policy?", "knowledge_base": None},
    #     config=config,
    # )
    # print("Turn 1:", r1["answer"])
    # print("Reasoning:", r1.get("agent_reasoning"))

    # # Turn 2 — chat_history is loaded automatically from checkpoint
    # r2 = graph.invoke(
    #     {
    #         "question": "Does that apply to digital products too?",
    #         "knowledge_base": None,
    #     },
    #     config=config,
    # )
    # print("Turn 2:", r2["answer"])
    # print("Reasoning:", r2.get("agent_reasoning"))
