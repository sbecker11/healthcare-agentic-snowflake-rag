"""Test suite for the Concierge agentic-RAG + MLOps pipeline.

Runs fully offline (deterministic mock LLM, TF-IDF retriever). Exercises the
graph's cyclic control flow, the eval gate, drift detection, the registry's
promotion semantics, and a tiny HPO sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concierge import RagConfig, TfidfRetriever, load_documents, run_once, build_graph, LLM
from concierge.knowledge import RetrievedDoc, Retriever
from concierge.mlops import (
    evaluate, load_eval_set, passes_gate, EvalResult,
    population_stability_index, ConfigRegistry, run_sweep, Tracker,
)


@pytest.fixture(scope="module")
def docs():
    return load_documents(ROOT / "data" / "knowledge_base.json")


@pytest.fixture(scope="module")
def retriever(docs):
    return TfidfRetriever(docs)


@pytest.fixture(scope="module")
def eval_set():
    return load_eval_set(ROOT / "data" / "eval_set.json")


# --- retriever -------------------------------------------------------------

def test_retriever_returns_top_k(retriever):
    out = retriever.search("late check-out time", top_k=3)
    assert len(out) == 3
    assert all(isinstance(d, RetrievedDoc) for d in out)
    # scores are sorted descending
    assert out[0].score >= out[1].score >= out[2].score


def test_retriever_is_relevant(retriever):
    out = retriever.search("are dogs allowed", top_k=1)
    assert out[0].id == "policy-pets"


def test_retriever_satisfies_protocol(retriever):
    assert isinstance(retriever, Retriever)


# --- graph -----------------------------------------------------------------

def test_graph_runs_and_answers(retriever):
    state = run_once("What time is check-out?", RagConfig(), retriever)
    assert state["answer"]
    assert state["answer_ok"] is True
    assert state["context_docs"]


def test_rewrite_loop_is_bounded(retriever):
    # Impossibly strict grading forces rewrites; loop must respect the budget.
    cfg = RagConfig(relevance_threshold=0.99, min_keyword_overlap=0.99, max_rewrites=2)
    state = run_once("dog fee", cfg, retriever)
    assert state["rewrite_count"] == 2  # exactly the budget, no infinite loop
    # retrieve runs max_rewrites + 1 times
    retrieves = [t for t in state["trace"] if t.startswith("retrieve(")]
    assert len(retrieves) == 3


def test_no_rewrite_when_retrieval_is_good(retriever):
    state = run_once("How much is valet parking?", RagConfig(), retriever)
    assert state.get("rewrite_count", 0) == 0


# --- evaluation & gate -----------------------------------------------------

def test_evaluate_default_config(retriever, eval_set):
    res = evaluate(RagConfig(), retriever, eval_set)
    assert res.n == len(eval_set)
    assert 0.0 <= res.recall_at_k <= 1.0
    assert res.recall_at_k >= 0.75  # default config should clear the recall floor


def test_gate_rejects_below_floor():
    weak = EvalResult(recall_at_k=0.5, answer_accuracy=0.4, composite=0.45,
                      n=12, avg_top_score=0.1)
    ok, _ = passes_gate(weak, None)
    assert ok is False


def test_gate_requires_beating_incumbent():
    incumbent = EvalResult(0.9, 0.8, 0.85, 12, 0.2)
    tie = EvalResult(0.9, 0.8, 0.85, 12, 0.2)
    ok, _ = passes_gate(tie, incumbent)
    assert ok is False  # must strictly beat


# --- monitoring ------------------------------------------------------------

def test_psi_stable_for_same_distribution():
    import numpy as np
    rng = np.random.default_rng(0)
    ref = rng.normal(0.25, 0.05, 1000).tolist()
    cur = rng.normal(0.25, 0.05, 1000).tolist()
    assert population_stability_index(ref, cur).severity == "stable"


def test_psi_flags_significant_shift():
    import numpy as np
    rng = np.random.default_rng(0)
    ref = rng.normal(0.25, 0.05, 1000).tolist()
    cur = rng.normal(0.60, 0.05, 1000).tolist()
    report = population_stability_index(ref, cur)
    assert report.severity == "significant"
    assert report.psi > 0.25


# --- registry --------------------------------------------------------------

def test_registry_register_and_promote(tmp_path):
    reg = ConfigRegistry(tmp_path / "reg.json")
    v1 = reg.register(RagConfig(), {"composite": 0.8}, note="first")
    assert v1.stage == "Staging"
    reg.promote(v1.version)
    assert reg.production().version == 1
    v2 = reg.register(RagConfig(top_k=5), {"composite": 0.9})
    reg.promote(v2.version)
    assert reg.production().version == 2
    # prior production is archived
    assert [v for v in reg.all() if v.version == 1][0].stage == "Archived"


def test_registry_persists(tmp_path):
    path = tmp_path / "reg.json"
    reg = ConfigRegistry(path)
    reg.register(RagConfig(), {"composite": 0.8})
    reloaded = ConfigRegistry(path)
    assert reloaded.latest().version == 1


# --- HPO sweep (integration) ----------------------------------------------

def test_sweep_registers_and_promotes(retriever, eval_set, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep mlflow.db / artifacts in tmp
    reg = ConfigRegistry(tmp_path / "registry.json")
    result = run_sweep(retriever, eval_set, n_trials=5,
                       tracker=Tracker(), registry=reg)
    assert result.n_trials == 5
    assert result.registered_version == 1
    assert 0.0 <= result.best_metrics.composite <= 1.0
    assert reg.latest() is not None
