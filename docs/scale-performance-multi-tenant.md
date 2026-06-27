# Scale, Performance, and Multi-Tenant Design

This document describes **production-scale considerations** for agentic member
navigation on Snowflake — beyond what the current portfolio implements (12
synthetic KB documents, single-tenant DDL, offline TF-IDF). The patterns here
extend the medallion → semantic layer → Cortex → LangGraph architecture in this
repo; they are **design guidance**, not deployed infrastructure.

---

## Scope of this repo vs. production

| Concern | This portfolio | Production target |
|---------|----------------|-------------------|
| Knowledge corpus | ~12 policy documents | Thousands of policies + code reference data |
| Diagnosis / procedure codes | Not modeled | ICD-10-CM, CPT, HCPCS (millions of rows) |
| Members | Synthetic, no member grain | Millions of members with eligibility / claims |
| Tenancy | Single `AI` database | Multi-plan, multi-region, or multi-payer isolation |
| Retrieval | In-memory TF-IDF; Cortex adapter stub | Partitioned Cortex Search, warehouse tuning |
| Latency SLO | Best-effort (dev/CI) | Sub-second retrieval; bounded agent loops |

---

## Millions of diagnostic and treatment codes

**Do not** treat millions of ICD/CPT/HCPCS codes as unstructured RAG documents in
`MEMBER_KB`. Code sets are **structured reference data** with hierarchy, effective
dates, and crosswalks — a poor fit for naive full-text embedding of every code row.

### Recommended data model

```
BRONZE: raw code feeds (CMS, AMA, internal crosswalks)
   ↓
SILVER: conformed CODE_REFERENCE (code, system, description, parent_code,
        effective_start, effective_end, tenant_id)
   ↓
GOLD:   CODE_LOOKUP (current codes only)
        CODE_HIERARCHY (chapter → category → code)
        CODE_SYNONYM (lay terms → code, e.g. "heart attack" → I21.*)
   ↓
SEMANTIC: semantic views for NL analytics ("procedure cost by code family")
   ↓
CORTEX (optional): search over *descriptions + synonyms*, not one doc per code
```

### Retrieval strategy for codes

1. **Structured first** — If the question maps to a code lookup ("Is 99213
   covered?"), route to semantic views or SQL, not vector search over the full
   code table.

2. **Block then rank** — For NL queries ("MRI brain without contrast"):
   - filter by code system, effective date, plan, tenant
   - narrow candidates by prefix, specialty, or taxonomy branch
   - run Cortex hybrid search only on the reduced set (or on synonym/description
     tables, not 2M raw code rows)

3. **Separate search services** — Policy prose (`MEMBER_KB_SEARCH`) and code
   intelligence (`CODE_SEARCH`) are different corpora with different refresh
   cadences and relevance profiles. The agent graph can choose the retriever or
   call both and merge.

4. **Hierarchy in the semantic layer** — Semantic views expose `code_family`,
   `chapter`, `short_description` so analysts and Cortex Analyst map business
   language to columns without scanning every leaf code.

5. **Lineage and versioning** — Code sets change quarterly. `lineage_sk` and
   `effective_date` on gold rows ensure answers cite the code version in effect
   for the member's date of service.

### What breaks at code scale

| Anti-pattern | Why it fails |
|--------------|--------------|
| One Cortex doc per code × millions of codes | Index cost, stale embeddings, noisy retrieval |
| Single flat `MEMBER_KB` mixing policy + all codes | Wrong top_k hits; conflates "copay policy" with "CPT definition" |
| TF-IDF over full code corpus | Memory and latency explode; no managed reranking |
| Ignoring effective dates | Wrong code version → compliance and billing errors |

---

## Multi-tenant issues

Health plans often serve **multiple lines of business**, **regions**, or **payer
contracts** from one platform. Tenancy must be enforced at **data**, **search**,
**agent runtime**, and **governance** layers — not only in application code.

### Tenancy dimensions

| Dimension | Example | Isolation mechanism |
|-----------|---------|---------------------|
| Payer / plan sponsor | Regional Blues, ASO clients | `tenant_id`, separate schemas or RAP |
| Plan product | PPO vs HMO vs Medicare Advantage | `plan_code` filters on gold + Cortex attributes |
| Environment | dev / staging / prod | Separate databases or account-level isolation |
| Role | agent vs analyst vs admin | Snowflake roles + masking + row access policies |

### Snowflake patterns

1. **`tenant_id` on every medallion table** — bronze through gold; never join
   across tenants without an explicit, audited cross-tenant role.

2. **Row access policies (RAP)** — Extend `06_governance.sql` so
   `MEMBER_BENEFITS_ASSISTANT_AGENT_ROLE` sees only rows where
   `tenant_id = CURRENT_SESSION_TENANT()` (session context set by the app).

3. **Cortex Search attributes** — Index with `tenant_id`, `plan_code`, `category`
   as filterable attributes; every query passes tenant context from the member
   session so retrieval cannot leak another tenant's policies.

4. **Semantic views per tenant or parameterized views** — Shared DDL with
   `tenant_id` dimension, or tenant-specific views over shared gold where
   contracts allow.

5. **Warehouse isolation** — Large tenants may get dedicated warehouses for
   Cortex Search and agent workloads to avoid noisy-neighbor credit burn.

### Agent / LangGraph tenancy

- **Thread ID** = `{tenant_id}:{member_session_id}` for checkpointer isolation.
- **Retriever closure** — Inject tenant-scoped filters when constructing
  `CortexRetriever.search()` (extend adapter to pass Cortex filter JSON).
- **Prompt context** — Include plan/tenant in state; never rely on the LLM to
  infer tenant from free text alone.
- **Eval sets per tenant** — Promotion gates run on tenant-representative eval
  slices so a global config doesn't underperform on a regional plan.

### Cross-tenant failure modes

| Risk | Mitigation |
|------|------------|
| Retrieval returns another tenant's policy | Cortex attribute filters + RAP on gold |
| Shared KB row without tenant_id | Reject at silver QA; block promotion |
| Analyst SQL without tenant predicate | Secure views; query tags; policy audits |
| Checkpoint replay across tenants | Tenant-prefixed thread IDs; separate DB/schema |

---

## Millions of members

Benefits navigation mixes **plan-wide policy** (shared KB) with **member-specific
facts** (eligibility, accumulators, prior auths, PCP assignment). At millions of
members, **do not embed every member in the vector index**.

### Split shared vs. member-specific data

| Data type | Scale | Serving pattern |
|-----------|-------|-----------------|
| Plan policies, FAQs, code reference | Thousands–millions of rows (codes) | Gold + Cortex Search + semantic views |
| Member eligibility, benefits, claims summary | Millions of rows | Keyed SQL / semantic views filtered by `member_id_hash` |
| Session / conversation state | Per active session | Postgres checkpointer; TTL and encryption |

The agent workflow becomes **retrieve shared policy** + **lookup member context**
(structured tool call or semantic view query) + **generate grounded answer**.

### Snowflake performance at member scale

1. **Clustering** — `CLUSTER BY (tenant_id, member_id_hash)` on large member
   fact tables; avoid full scans for single-member lookups.

2. **Bronze partitioning** — `ingest_batch_id`, date partitions on event streams;
   micro-batch silver merges instead of full-table rewrites.

3. **Gold materialization** — Incremental `MERGE` into `MEMBER_KB` and member
   summary tables; `is_active` and SCD patterns for policy changes.

4. **PII / PHI** — Keep hashed identifiers in agent-accessible layers; reveal
   only under break-glass roles. Masking policies in bronze (see
   `06_governance.sql`) extend to member tables at scale.

5. **Caching** — Plan-level policy retrieval is highly repetitive; cache Cortex
   results or hot documents at the API edge with tenant-scoped cache keys.

---

## Performance considerations

### Retrieval latency

| Layer | Bottleneck | Tuning |
|-------|------------|--------|
| TF-IDF (offline) | In-memory matrix size | Dev/CI only; not for production corpus |
| Cortex Search | Index lag, warehouse queue | `TARGET_LAG`, dedicated WH, right-sized S/M/L |
| Agent graph | Rewrite + regen loops | `max_rewrites`, `max_regenerations` (HPO-tuned) |
| LLM generation | Token count | Cap context docs; summarize long policy text in silver |

The graph's **bounded loops** are intentional latency guards: worst case is
`(1 + max_rewrites) × retrieve + (1 + max_regenerations) × generate`.

### Hyperparameter tuning (already in repo)

Optuna sweeps over `top_k`, thresholds, and loop budgets against an eval set —
trade recall for latency by lowering `top_k` and tightening rewrite triggers.
Production adds **p95 latency** and **cost per query** to the promotion gate,
not just recall@k and accuracy.

### Index and pipeline refresh

| Asset | Refresh cadence | Performance note |
|-------|-----------------|------------------|
| `MEMBER_KB` / Cortex Search | Hourly (`TARGET_LAG`) or event-driven | Stale index → PSI drift alerts |
| Code reference gold | Quarterly + emergency patches | Versioned indexes; blue/green Cortex service |
| Semantic views | On gold refresh | Lightweight; no re-embed |
| Member facts | Near real-time or daily | Not in Cortex; SQL lookup |

### Observability at scale

- **PSI on retrieval scores** (see `mlops/monitoring.py`) — shift in question
  or corpus difficulty without code deploys.
- **Query tags** on Snowflake — tenant, plan, retriever backend, graph path.
- **MLflow** — config version on every production trace for rollback.
- **SLO dashboards** — retrieve p95, end-to-end p95, rewrite rate, fallback rate.

### Production swaps (extended)

| Concern | Portfolio | At scale |
|---------|-----------|----------|
| Checkpointer | SqliteSaver | Postgres / Redis with tenant isolation |
| Experiment store | SQLite MLflow | Managed MLflow / Databricks |
| Retriever | TF-IDF | Cortex Search + structured SQL tools |
| Eval gate | recall / accuracy | + latency p95, cost, tenant slice coverage |
| Governance | Illustrative RAP | Tenant RAP + audit logging + break-glass |

---

## Suggested evolution from this repo

A credible path from the current portfolio to production scale:

1. **Add gold `CODE_REFERENCE` + semantic views** — structured code lookups without
   bloating `MEMBER_KB`.
2. **Add `tenant_id` + member session filters** — extend DDL and `CortexRetriever`
   with attribute filters.
3. **Split agent tools** — `search_policy_kb`, `lookup_member_benefits`,
   `lookup_code` instead of one monolithic retriever.
4. **Harden promotion gate** — latency and per-tenant eval slices alongside recall.
5. **Replace SqliteSaver** — Postgres checkpointer for concurrent member sessions.

---

## Related docs

- [agentic-ai-engineering-primer.md](agentic-ai-engineering-primer.md) — role mapping and architecture
- [../README.md](../README.md) — agentic RAG workflow and Cortex cooperation
- [../snowflake/README.md](../snowflake/README.md) — medallion deploy order
- [../snowflake/06_governance.sql](../snowflake/06_governance.sql) — baseline roles and RAP
