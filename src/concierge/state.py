"""LangGraph state schema.

The state is the contract every node reads and writes. The `Annotated` reducers
are the detail that matters in production: they define *how* a field merges when
a node returns an update. `retrieved` accumulates across rewrite loops (append),
while scalar fields overwrite. Getting reducers wrong is the single most common
source of LangGraph production incidents, so they are explicit here.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .knowledge import RetrievedDoc


def _keep_last(_old, new):
    """Reducer for scalars: last write wins (explicit for readability)."""
    return new


class ConciergeState(TypedDict, total=False):
    # Inputs
    question: str

    # Working query (may be rewritten in the retrieval loop)
    query: Annotated[str, _keep_last]

    # Accumulates across rewrite iterations so we can inspect the full trail.
    retrieved: Annotated[list[RetrievedDoc], operator.add]

    # The docs selected for generation after grading (overwrite each pass).
    context_docs: Annotated[list[RetrievedDoc], _keep_last]

    # Loop bookkeeping
    rewrite_count: Annotated[int, _keep_last]
    regen_count: Annotated[int, _keep_last]

    # Grades / decisions (for tracing and eval)
    retrieval_ok: Annotated[bool, _keep_last]
    answer_ok: Annotated[bool, _keep_last]

    # Output
    answer: Annotated[str, _keep_last]

    # Execution trace for observability (append-only).
    trace: Annotated[list[str], operator.add]
