"""Evaluate the current Production config (or the default if none is registered)
against the offline eval set and print the metric breakdown.

Run:  python scripts/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from member_nav import RagConfig, TfidfRetriever, load_documents, LLM
from member_nav.mlops import evaluate, load_eval_set, ConfigRegistry, passes_gate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    docs = load_documents(ROOT / "data" / "knowledge_base.json")
    retriever = TfidfRetriever(docs)
    eval_set = load_eval_set(ROOT / "data" / "eval_set.json")

    registry = ConfigRegistry(ROOT / "registry.json")
    prod = registry.production()
    if prod is not None:
        config = RagConfig.from_dict(prod.config)
        source = f"Production config v{prod.version}"
    else:
        config = RagConfig()
        source = "default config (no Production version registered)"

    result = evaluate(config, retriever, eval_set, LLM())
    print(f"Evaluating: {source}")
    print(f"Config    : {config.to_dict()}")
    print("-" * 60)
    for k, v in result.to_dict().items():
        print(f"  {k:<16}: {v}")
    ok, reason = passes_gate(result, None)
    print("-" * 60)
    print(f"  clears absolute floors: {ok} ({reason})")


if __name__ == "__main__":
    main()
