"""LangGraph state for the Member Benefits Assistant agent.

Fields use explicit reducers: `retrieved` accumulates across rewrite iterations
(operator.add); scalar fields are last-write-wins.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .knowledge import RetrievedDoc


class MemberBenefitsAssistantState(TypedDict, total=False):
    question: str
    query: str
    retrieved: Annotated[list[RetrievedDoc], operator.add]
    context_docs: list[RetrievedDoc]
    retrieval_ok: bool
    rewrite_count: int
    regen_count: int
    answer: str
    answer_ok: bool
    trace: Annotated[list[str], operator.add]
