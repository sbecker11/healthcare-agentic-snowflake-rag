"""Offline evaluation and the promotion gate.

Defines the metrics HPO optimizes and the gate the registry uses to decide
whether a candidate config is promotable to Production. Metrics:

  * recall_at_k       - fraction of eval examples whose relevant doc appears in
                        the retrieved set (retrieval quality).
  * answer_accuracy   - fraction of answers containing the required ground-truth
                        substring(s) (end-to-end quality).
  * composite         - weighted blend; the single objective HPO maximizes.

The gate is a hard quality bar: a candidate must beat the incumbent on the
composite *and* clear absolute floors. This is the MLOps analog of a CI test
gate, except it gates on model metrics rather than unit tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import RagConfig
from ..graph import run_once
from ..knowledge import Retriever
from ..llm import LLM


@dataclass
class EvalResult:
    recall_at_k: float
    answer_accuracy: float
    composite: float
    n: int
    avg_top_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_eval_set(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())["examples"]


def evaluate(
    config: RagConfig,
    retriever: Retriever,
    eval_set: list[dict],
    llm: LLM | None = None,
    recall_weight: float = 0.5,
) -> EvalResult:
    llm = llm or LLM()
    hits = 0
    correct = 0
    top_scores = []
    for ex in eval_set:
        state = run_once(ex["question"], config, retriever, llm)
        retrieved_ids = {d.id for d in state.get("context_docs", [])}
        if set(ex["relevant_ids"]) & retrieved_ids:
            hits += 1
        if state.get("context_docs"):
            top_scores.append(state["context_docs"][0].score)
        answer = (state.get("answer") or "").lower()
        if all(token.lower() in answer for token in ex["must_include"]):
            correct += 1
    n = len(eval_set)
    recall = hits / n if n else 0.0
    acc = correct / n if n else 0.0
    composite = recall_weight * recall + (1 - recall_weight) * acc
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return EvalResult(
        recall_at_k=round(recall, 4),
        answer_accuracy=round(acc, 4),
        composite=round(composite, 4),
        n=n,
        avg_top_score=round(avg_top, 4),
    )


# Absolute floors a candidate must clear regardless of the incumbent.
PROMOTION_FLOORS = {"recall_at_k": 0.75, "answer_accuracy": 0.50}


def passes_gate(
    candidate: EvalResult, incumbent: EvalResult | None
) -> tuple[bool, str]:
    """Return (promote?, human-readable reason)."""
    if candidate.recall_at_k < PROMOTION_FLOORS["recall_at_k"]:
        return False, (
            f"recall {candidate.recall_at_k} below floor "
            f"{PROMOTION_FLOORS['recall_at_k']}"
        )
    if candidate.answer_accuracy < PROMOTION_FLOORS["answer_accuracy"]:
        return False, (
            f"accuracy {candidate.answer_accuracy} below floor "
            f"{PROMOTION_FLOORS['answer_accuracy']}"
        )
    if incumbent is None:
        return True, "no incumbent; candidate clears floors"
    if candidate.composite > incumbent.composite:
        return True, (
            f"composite {candidate.composite} beats incumbent "
            f"{incumbent.composite}"
        )
    return False, (
        f"composite {candidate.composite} does not beat incumbent "
        f"{incumbent.composite}"
    )
