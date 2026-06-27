"""Hyperparameter configuration for the Member Benefits Assistant agentic-RAG pipeline.

This is the object that hyperparameter optimization searches over. Keeping the
search space in one typed place is deliberate: it is the boundary between the
*training/tuning* outer loop (which varies these) and the *graph execution*
inner loop (which treats them as fixed constants for a single run).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RagConfig:
    """A single fixed point in hyperparameter space.

    Attributes:
        top_k: number of documents the retriever returns per query.
        relevance_threshold: minimum top document score below which the graph
            decides retrieval was weak and enters the query-rewrite loop.
        max_rewrites: maximum number of query-rewrite iterations (loop budget).
        max_regenerations: maximum answer-regeneration iterations.
        min_keyword_overlap: lexical floor used by the document grader.
        prompt_version: which generation prompt template to use (registry-tracked).
    """

    top_k: int = 3
    relevance_threshold: float = 0.10
    max_rewrites: int = 1
    max_regenerations: int = 1
    min_keyword_overlap: float = 0.04
    prompt_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RagConfig":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


# The search space consumed by the Optuna HPO sweep. Each entry is
# (suggest_kind, *args) interpreted in mlops/tuning.py. Centralizing it here
# keeps the tunable surface auditable and version-controlled.
SEARCH_SPACE: dict[str, tuple] = {
    "top_k": ("int", 1, 6),
    "relevance_threshold": ("float", 0.02, 0.30),
    "max_rewrites": ("int", 0, 3),
    "max_regenerations": ("int", 0, 2),
    "min_keyword_overlap": ("float", 0.0, 0.12),
    "prompt_version": ("categorical", ["v1", "v2"]),
}
