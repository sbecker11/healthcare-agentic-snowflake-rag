"""Retrieval layer.

A `Retriever` protocol with two interchangeable adapters:

  * `TfidfRetriever`     - local, offline, scikit-learn TF-IDF + cosine. This is
                           the default and the one that runs in tests / CI / this
                           sandbox with zero external dependencies.
  * `CortexRetriever`    - Snowflake Cortex Search adapter (managed hybrid
                           vector + keyword + reranking). Real, runnable code,
                           gated behind credentials. This is the production swap
                           and the path you would demo against a Snowflake account.

The graph depends only on the protocol, so swapping retrieval backends is a
one-line change and HPO can tune whichever backend is active.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievedDoc(BaseModel):
    """A retrieved document. A Pydantic model (not a bare dataclass) so the
    LangGraph checkpointer serializes/deserializes it cleanly across pauses."""

    id: str
    title: str
    text: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedDoc]: ...


def load_documents(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["documents"]


class TfidfRetriever:
    """Offline cosine-similarity retriever over a TF-IDF document matrix.

    Stands in for a production vector store (pgvector / OpenSearch / Cortex).
    The interface and scores are what the graph and HPO consume, so the rest of
    the system is agnostic to this being TF-IDF rather than dense embeddings.
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._corpus = [f"{d['title']}. {d['text']}" for d in documents]
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    def search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        order = np.argsort(-sims)[:top_k]
        out: list[RetrievedDoc] = []
        for idx in order:
            d = self.documents[idx]
            out.append(
                RetrievedDoc(
                    id=d["id"], title=d["title"], text=d["text"], score=float(sims[idx])
                )
            )
        return out


class CortexRetriever:
    """Snowflake Cortex Search adapter (production path).

    Cortex Search is a managed hybrid retrieval service: it embeds and indexes
    your documents, then serves each query through vector search + keyword search
    + semantic reranking. You provision the service once with SQL (see
    ``CORTEX_SEARCH_DDL`` below), then query it via the Snowpark / snowflake.core
    Python API. This adapter is intentionally lazy-connected so importing the
    module never requires Snowflake to be installed or reachable.

    Usage:
        from snowflake.snowpark import Session
        session = Session.builder.configs(conn_params).create()
        retriever = CortexRetriever(session, service_name="CONCIERGE_KB_SEARCH")
    """

    def __init__(self, session, service_name: str, database: str = "AI",
                 schema: str = "CONCIERGE"):
        self._session = session
        self._service_name = service_name
        self._database = database
        self._schema = schema
        self._svc = None  # resolved on first search

    def _service(self):
        if self._svc is None:
            from snowflake.core import Root  # imported lazily

            root = Root(self._session)
            self._svc = (
                root.databases[self._database]
                .schemas[self._schema]
                .cortex_search_services[self._service_name]
            )
        return self._svc

    def search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        resp = self._service().search(
            query=query,
            columns=["id", "title", "text"],
            limit=top_k,
        )
        out: list[RetrievedDoc] = []
        # Cortex returns reranked rows in descending relevance; we synthesize a
        # rank-derived score so the graph's threshold logic stays backend-neutral.
        rows = resp.results if hasattr(resp, "results") else resp
        n = max(len(rows), 1)
        for rank, row in enumerate(rows):
            out.append(
                RetrievedDoc(
                    id=row["id"],
                    title=row["title"],
                    text=row["text"],
                    score=float((n - rank) / n),
                )
            )
        return out


# DDL to provision the managed service once, against your own Snowflake account.
# Cortex Search handles embedding (snowflake-arctic-embed), indexing, hybrid
# retrieval, and reranking; no external vector DB or ETL is required.
CORTEX_SEARCH_DDL = """
CREATE DATABASE IF NOT EXISTS AI;
CREATE SCHEMA IF NOT EXISTS AI.CONCIERGE;

CREATE OR REPLACE TABLE AI.CONCIERGE.KNOWLEDGE_BASE (
    id      STRING,
    title   STRING,
    category STRING,
    text    STRING
);
-- (load rows from data/knowledge_base.json via COPY INTO or Snowpark write)

CREATE OR REPLACE CORTEX SEARCH SERVICE AI.CONCIERGE.CONCIERGE_KB_SEARCH
    ON text
    ATTRIBUTES id, title, category
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 hour'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
    AS (
        SELECT id, title, category, text
        FROM AI.CONCIERGE.KNOWLEDGE_BASE
    );
"""
