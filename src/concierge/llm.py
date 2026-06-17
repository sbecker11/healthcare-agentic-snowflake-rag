"""LLM access for the graph's reasoning nodes.

Design goal: the repo must run end-to-end with **no** API key (for CI, tests,
and reviewers) while making the swap to a real Claude call a single environment
variable away. When ANTHROPIC_API_KEY is set, `LLM.complete` calls Claude via
langchain-anthropic; otherwise it falls back to a deterministic, dependency-free
mock that is good enough to exercise every branch of the graph reproducibly.
"""
from __future__ import annotations

import os
import re
from typing import Optional


class LLM:
    """Thin wrapper exposing two task-specific methods the graph needs."""

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = None
        self.online = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.online:
            try:
                from langchain_anthropic import ChatAnthropic

                self._client = ChatAnthropic(model=model, temperature=temperature)
            except Exception:
                # Any import/auth problem degrades gracefully to offline mode.
                self.online = False

    # -- public task methods -------------------------------------------------

    def rewrite_query(self, query: str, weak_context: str) -> str:
        """Produce a re-phrased retrieval query when the first pass was weak."""
        prompt = (
            "You are improving a search query for a hotel knowledge base. "
            "The previous query retrieved weak results. Rewrite it to be more "
            "specific and keyword-rich. Return ONLY the rewritten query.\n\n"
            f"Original query: {query}\n"
            f"Weak context found: {weak_context[:300]}"
        )
        if self.online:
            return self._chat(prompt).strip()
        return self._mock_rewrite(query)

    def generate_answer(self, query: str, context: str, prompt_version: str) -> str:
        """Generate a grounded answer from retrieved context."""
        style = (
            "Answer concisely in one or two sentences."
            if prompt_version == "v1"
            else "Answer warmly as a concierge, citing the specific policy detail."
        )
        prompt = (
            "You are a hotel concierge assistant. Answer the guest's question "
            f"using ONLY the context below. {style} If the context does not "
            "contain the answer, say you will check with the front desk.\n\n"
            f"Context:\n{context}\n\nGuest question: {query}\n\nAnswer:"
        )
        if self.online:
            return self._chat(prompt).strip()
        return self._mock_answer(query, context)

    # -- internals -----------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        msg = self._client.invoke(prompt)
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    @staticmethod
    def _mock_rewrite(query: str) -> str:
        # Deterministic expansion: append domain keywords to nudge retrieval.
        ql = query.lower()
        hints = []
        for kw, expand in [
            ("late", "check-out time policy"),
            ("dog", "pet policy fee"),
            ("park", "valet self-parking nightly rate"),
            ("point", "loyalty rewards redemption"),
            ("spa", "massage booking discount"),
            ("wifi", "internet streaming bandwidth fee"),
        ]:
            if kw in ql:
                hints.append(expand)
        suffix = (" " + " ".join(hints)) if hints else " hotel policy details"
        return (query + suffix).strip()

    @staticmethod
    def _mock_answer(query: str, context: str) -> str:
        """Extractive mock: return the sentence(s) most relevant to the query.

        This is intentionally simple and deterministic so tests and the eval
        gate behave identically with or without a real model. A real Claude
        call produces far better prose; the architecture is unchanged.
        """
        if not context.strip():
            return "I'll check with the front desk and follow up shortly."
        q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
        scored = []
        for s in sentences:
            s_terms = set(re.findall(r"[a-z0-9]+", s.lower()))
            overlap = len(q_terms & s_terms)
            if overlap:
                scored.append((overlap, s.strip()))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return sentences[0].strip()
        return " ".join(s for _, s in scored[:2])
