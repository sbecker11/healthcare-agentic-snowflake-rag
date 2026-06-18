"""The agentic-RAG graph.

This is the LangGraph core. Unlike a linear LCEL chain (a DAG), this graph has
**cycles**: a self-correcting retrieval loop (retrieve -> grade -> rewrite ->
retrieve) and an answer-regeneration loop (generate -> grade -> generate). The
loops are bounded by hyperparameters (`max_rewrites`, `max_regenerations`) that
HPO tunes. Conditional edges route on graded state, which is exactly the control
flow that a plain chain cannot express.

    START
      v
   retrieve  <-------------------+
      v                          |
  grade_retrieval                |
      v                          |
  (retrieval_ok or budget?) --no-> rewrite_query --+
      | yes
      v
   generate  <-----------------+
      v                        |
  grade_answer                 |
      v                        |
  (answer_ok or budget?) --no--+
      | yes
      v
     END
"""
from __future__ import annotations

import re
from typing import Optional

from langgraph.graph import StateGraph, START, END

from .config import RagConfig
from .knowledge import Retriever, RetrievedDoc
from .llm import LLM
from .state import MemberNavState


def _keyword_overlap(question: str, doc: RetrievedDoc) -> float:
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    d = set(re.findall(r"[a-z0-9]+", f"{doc.title} {doc.text}".lower()))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def build_graph(
    config: RagConfig,
    retriever: Retriever,
    llm: Optional[LLM] = None,
    checkpointer=None,
):
    """Compile and return the executable graph for a *fixed* config.

    The config is captured by closure into the nodes: within a single run the
    hyperparameters are constants (the inner loop), while HPO varies them across
    runs (the outer loop).
    """
    llm = llm or LLM()

    # -- nodes ---------------------------------------------------------------

    def retrieve(state: MemberNavState) -> dict:
        query = state.get("query") or state["question"]
        docs = retriever.search(query, top_k=config.top_k)
        return {
            "retrieved": docs,
            "context_docs": docs,
            "trace": [f"retrieve(query={query!r}) -> {len(docs)} docs, "
                      f"top_score={docs[0].score:.3f}" if docs else "retrieve -> 0 docs"],
        }

    def grade_retrieval(state: MemberNavState) -> dict:
        docs = state.get("context_docs") or []
        if not docs:
            return {"retrieval_ok": False, "trace": ["grade_retrieval -> no docs"]}
        top = docs[0]
        overlap = _keyword_overlap(state["question"], top)
        ok = (top.score >= config.relevance_threshold) or (
            overlap >= config.min_keyword_overlap
        )
        return {
            "retrieval_ok": ok,
            "trace": [f"grade_retrieval -> ok={ok} "
                      f"(score={top.score:.3f} thr={config.relevance_threshold}, "
                      f"overlap={overlap:.3f} min={config.min_keyword_overlap})"],
        }

    def rewrite_query(state: MemberNavState) -> dict:
        docs = state.get("context_docs") or []
        weak = docs[0].text if docs else ""
        new_query = llm.rewrite_query(state["question"], weak)
        return {
            "query": new_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "trace": [f"rewrite_query -> {new_query!r}"],
        }

    def generate(state: MemberNavState) -> dict:
        docs = state.get("context_docs") or []
        context = "\n\n".join(f"[{d.title}] {d.text}" for d in docs)
        answer = llm.generate_answer(
            state["question"], context, config.prompt_version
        )
        return {"answer": answer, "trace": [f"generate -> {len(answer)} chars"]}

    def grade_answer(state: MemberNavState) -> dict:
        answer = (state.get("answer") or "").strip()
        docs = state.get("context_docs") or []
        ctx_terms = set(
            re.findall(r"[a-z0-9]+", " ".join(d.text for d in docs).lower())
        )
        ans_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
        grounded = len(ctx_terms & ans_terms) >= 3
        is_fallback = "front desk" in answer.lower()
        ok = bool(answer) and grounded and not is_fallback
        out = {
            "answer_ok": ok,
            "trace": [f"grade_answer -> ok={ok} (grounded={grounded}, "
                      f"fallback={is_fallback})"],
        }
        if not ok:
            out["regen_count"] = state.get("regen_count", 0) + 1
        return out

    # -- conditional edge routers -------------------------------------------

    def after_retrieval(state: MemberNavState) -> str:
        if state.get("retrieval_ok"):
            return "generate"
        if state.get("rewrite_count", 0) >= config.max_rewrites:
            return "generate"  # out of rewrite budget; proceed with best effort
        return "rewrite_query"

    def after_answer(state: MemberNavState) -> str:
        if state.get("answer_ok"):
            return END
        if state.get("regen_count", 0) > config.max_regenerations:
            return END  # out of regen budget
        return "generate"

    # -- wire the graph ------------------------------------------------------

    g = StateGraph(MemberNavState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_retrieval", grade_retrieval)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("grade_answer", grade_answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_retrieval")
    g.add_conditional_edges(
        "grade_retrieval", after_retrieval,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    g.add_edge("rewrite_query", "retrieve")  # the retrieval cycle
    g.add_edge("generate", "grade_answer")
    g.add_conditional_edges(
        "grade_answer", after_answer,
        {"generate": "generate", END: END},
    )

    return g.compile(checkpointer=checkpointer)


def run_once(
    question: str,
    config: RagConfig,
    retriever: Retriever,
    llm: Optional[LLM] = None,
) -> MemberNavState:
    """Convenience: execute the graph once (no persistence) and return state."""
    app = build_graph(config, retriever, llm)
    init: MemberNavState = {
        "question": question,
        "query": question,
        "retrieved": [],
        "rewrite_count": 0,
        "regen_count": 0,
        "trace": [],
    }
    return app.invoke(init)
