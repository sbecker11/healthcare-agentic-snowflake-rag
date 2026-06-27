"""End-to-end demo: run example questions through the agentic-RAG graph and show
the execution trace, then demonstrate durable persistence via the SQLite
checkpointer (pause/resume + state history).

Run:  python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from member_benefits_assistant import RagConfig, TfidfRetriever, load_documents, build_graph, LLM
from member_benefits_assistant.state import MemberBenefitsAssistantState

ROOT = Path(__file__).resolve().parents[1]


def banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def main() -> None:
    docs = load_documents(ROOT / "data" / "knowledge_base.json")
    retriever = TfidfRetriever(docs)
    llm = LLM()  # offline mock unless ANTHROPIC_API_KEY is set
    config = RagConfig()

    print(f"LLM mode: {'CLAUDE (online)' if llm.online else 'deterministic mock (offline)'}")

    questions = [
        "When is open enrollment and when do plan changes take effect?",
        "Do I need prior authorization for an MRI?",
        "What is the copay for a specialist visit?",
    ]

    banner("AGENTIC-RAG TRACES")
    app = build_graph(config, retriever, llm)
    for q in questions:
        init: MemberBenefitsAssistantState = {
            "question": q, "query": q, "retrieved": [],
            "rewrite_count": 0, "regen_count": 0, "trace": [],
        }
        state = app.invoke(init)
        print(f"\nQ: {q}")
        for step in state["trace"]:
            print(f"   . {step}")
        print(f"A: {state['answer']}")

    # --- durable persistence via SQLite checkpointer ----------------------
    banner("DURABLE CHECKPOINTING (SqliteSaver)")
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = ROOT / "checkpoints.sqlite"
    # Allow our Pydantic RetrievedDoc to round-trip through the checkpoint store.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("member_benefits_assistant.knowledge", "RetrievedDoc")]
    )
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        cp = SqliteSaver(conn, serde=serde)
        durable = build_graph(config, retriever, llm, checkpointer=cp)
        thread = {"configurable": {"thread_id": "member-1138"}}
        q = "What is the phone number for the 24/7 nurse advice line?"
        init = {"question": q, "query": q, "retrieved": [],
                "rewrite_count": 0, "regen_count": 0, "trace": []}
        result = durable.invoke(init, config=thread)
        print(f"\nQ: {q}\nA: {result['answer']}")

        # The run is now persisted; we can reload its state from the store.
        snapshot = durable.get_state(thread)
        n_checkpoints = len(list(durable.get_state_history(thread)))
        print(f"\nPersisted thread 'member-1138': {n_checkpoints} checkpoints saved")
        print(f"Steps executed: {snapshot.metadata.get('step')}; "
              f"state recoverable from disk at {db.name}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
