"""Run the hyperparameter sweep (the outer search loop), log every trial to
MLflow, register the best config, and promote it through the gate if it wins.

Run:  python scripts/run_sweep.py [n_trials]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from concierge import TfidfRetriever, load_documents, LLM
from concierge.mlops import run_sweep, load_eval_set, ConfigRegistry, Tracker

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    docs = load_documents(ROOT / "data" / "knowledge_base.json")
    retriever = TfidfRetriever(docs)
    eval_set = load_eval_set(ROOT / "data" / "eval_set.json")

    tracker = Tracker(experiment="concierge-rag")
    registry = ConfigRegistry(ROOT / "registry.json")

    print(f"Running {n_trials}-trial TPE sweep "
          f"(tracking backend: {tracker.backend})...")
    result = run_sweep(retriever, eval_set, n_trials=n_trials,
                       llm=LLM(), tracker=tracker, registry=registry)

    print("-" * 60)
    print(f"Best composite : {result.best_metrics.composite}")
    print(f"  recall@k     : {result.best_metrics.recall_at_k}")
    print(f"  answer acc   : {result.best_metrics.answer_accuracy}")
    print(f"Best config    : {result.best_config.to_dict()}")
    print(f"Registered     : v{result.registered_version}")
    print(f"Promotion gate : {'PROMOTED' if result.promoted else 'HELD'} "
          f"({result.gate_reason})")

    prod = registry.production()
    print(f"Current Production: v{prod.version}" if prod else "No Production version")
    print("\nInspect all trials with:  mlflow ui --backend-store-uri "
          f"sqlite:///{(ROOT / 'mlflow.db')}")


if __name__ == "__main__":
    main()
