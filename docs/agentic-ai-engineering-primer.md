# Agentic AI Engineering Primer

Portfolio mapping for **Snowflake-based semantic data assets** and **agent-driven
analytics** — aligned with agentic AI engineering roles in health plans (e.g.
Cambia Health). All member data in this repo is **synthetic**; no real PHI.

## Role requirements ↔ repository artifacts

| Requirement | Artifact | Status |
|-------------|----------|--------|
| Snowflake semantic data assets for NL Q&A and AI insights | `AI.GOLD.MEMBER_KB` + `MEMBER_KB_SEARCH` Cortex service | Implemented |
| Medallion bronze / silver / gold layers | `snowflake/01_bronze.sql` → `03_gold.sql` | Implemented (DDL + lineage comments) |
| Semantic views and business abstractions | `snowflake/04_semantic_views.sql` (`MEMBER_COST_SHARING`, `V_MEMBER_COPAY`) | Implemented |
| Reusable data products from cross-team requirements | `data/knowledge_base.json`, `data/eval_set.json` (product/analytics ground truth) | Implemented |
| Operationalize Snowflake Cortex for agent workloads | `CortexRetriever` in `src/member_benefits_assistant/knowledge.py`, `snowflake/05_cortex_search.sql` | Implemented |
| Data quality validation | `src/member_benefits_assistant/mlops/evaluation.py` (recall@k, accuracy, promotion gate) | Implemented |
| Performance tuning | `src/member_benefits_assistant/mlops/tuning.py` (Optuna TPE over `RagConfig`) | Implemented |
| Lineage-aware transformations | Bronze → silver → gold `lineage_sk` / `ingest_batch_id`; MLflow trial lineage | Implemented |
| Governed access patterns | `snowflake/06_governance.sql` (roles, masking, row access policies) | Implemented (illustrative) |
| Agentic workloads at scale | LangGraph cyclic graph, bounded loops, SqliteSaver checkpointing, PSI drift | Implemented |

## Architecture

```mermaid
flowchart TB
    subgraph BRONZE["Bronze"]
        RAW["RAW_POLICY_FEED"]
    end
    subgraph SILVER["Silver"]
        BP["BENEFIT_POLICY"]
    end
    subgraph GOLD["Gold"]
        KB["MEMBER_KB"]
        PCS["PLAN_COST_SHARING"]
    end
    subgraph SEM["Semantic"]
        SV["MEMBER_COST_SHARING view"]
    end
    subgraph CORTEX["Cortex"]
        CS["MEMBER_KB_SEARCH"]
    end
    subgraph AGENT["Agent (LangGraph)"]
        G["Member Benefits Assistant graph"]
    end

    RAW --> BP --> KB --> CS --> G
    PCS --> SV
```

## Data lineage

| From | To | Transform |
|------|-----|-----------|
| Source JSON / feeds | `AI.BRONZE.RAW_POLICY_FEED` | Append-only ingest with `file_hash` dedupe key |
| Bronze | `AI.SILVER.BENEFIT_POLICY` | Trim, category normalize, latest-wins dedupe |
| Silver | `AI.GOLD.MEMBER_KB` | Curated agent-ready documents with `lineage_sk` |
| Silver structured | `AI.GOLD.PLAN_COST_SHARING` | Plan cost-sharing reference rows |
| Gold KB | `AI.MEMBER_BENEFITS_ASSISTANT.MEMBER_KB_SEARCH` | Cortex Search indexing + hybrid retrieval |
| Gold structured | `AI.SEMANTIC.*` | Business-friendly semantic and SQL views |

## Agent layer

The `member_benefits_assistant` package implements **agentic RAG** — not a linear chain:

1. **Retrieve** from `Retriever` (TF-IDF offline, Cortex Search production)
2. **Grade retrieval** → rewrite query if weak (bounded loop)
3. **Generate** grounded answer
4. **Grade answer** → regenerate if ungrounded (bounded loop)

Hyperparameters (`top_k`, relevance threshold, loop budgets, prompt version) are
tuned by Optuna, tracked in MLflow, and promoted through a quality gate.

## MLOps lifecycle

| Stage | Component | Purpose |
|-------|-----------|---------|
| Search | `SEARCH_SPACE` + Optuna TPE | Find optimal `RagConfig` |
| Track | MLflow (`member-benefits-assistant-rag` experiment) | Audit params and metrics per trial |
| Gate | `passes_gate()` | Floors: recall ≥ 0.75, accuracy ≥ 0.50; beat incumbent |
| Register | `ConfigRegistry` | Versioned Staging → Production promotion |
| Monitor | PSI on retrieval scores | Detect distribution drift post-deploy |

**HPO** (hyperparameter optimization) is that Search → Track → Gate → Register
outer loop. **TPE** (Tree-structured Parzen Estimator) is the Optuna sampler
driving Search — see below.

## TPE vs Simplex

**Simplex search** here means **Nelder–Mead** (downhill simplex): a local,
derivative-free optimizer that moves a simplex of \(n+1\) points in \(n\)
dimensions via reflect / expand / contract / shrink until it settles near a
nearby minimum.

| | Nelder–Mead (simplex) | Optuna TPE (this repo) |
|--|----------------------|-------------------------|
| Scope | Local from a start point | Global-ish coverage of the search space |
| Mechanism | Geometry of the simplex | Probabilistic “good vs bad” regions from past trials |
| Parameter types | Best for continuous, smooth-ish objectives | Strong on mixed / discrete / conditional spaces (`top_k`, loop budgets, prompt version) |
| Expensive trials | Can spend steps in poor local basins | Built for costly black-box evals (sample → full eval → update) |
| Global optimum | Not guaranteed | Not guaranteed — better coverage via exploration/exploitation |

For this project's RAG HPO — integer `top_k`, thresholds, rewrite/regen budgets,
optional categorical prompt versions, each trial = full eval set — **TPE is the
better default**. Nelder–Mead is a better fit when the space is mostly continuous
and a good starting region is already known (local polish). They can be combined
(TPE or random to find a basin, then local refine); this repo uses TPE alone for
the outer sweep (`src/member_benefits_assistant/mlops/tuning.py`).

## Governance model

`snowflake/06_governance.sql` defines:

- **`MEMBER_BENEFITS_ASSISTANT_AGENT_ROLE`** — read `MEMBER_KB`, use Cortex Search warehouse
- **`MEMBER_BENEFITS_ASSISTANT_ANALYST_ROLE`** — read semantic views and structured gold
- **Masking policy** on bronze `member_id_hash`
- **Row access policy** on gold `MEMBER_KB` (`is_active = TRUE`)

## Truthful resume language

> Portfolio: Snowflake medallion pipeline (bronze/silver/gold) feeding a Cortex
> Search semantic asset and semantic views over plan cost-sharing; LangGraph
> agentic RAG Member Benefits Assistant with MLflow tracking, Optuna HPO, promotion
> gates, and PSI drift monitoring. Synthetic health-plan data only.

## Local development (no Snowflake required)

```bash
pip install -e ".[dev]"
pytest -q
python scripts/demo.py
python scripts/run_sweep.py 20
```

## Production Snowflake path

Deploy `snowflake/*.sql` in order, load `data/knowledge_base.json` into bronze,
refresh silver/gold, then:

```python
from snowflake.snowpark import Session
from member_benefits_assistant import CortexRetriever, build_graph, RagConfig

session = Session.builder.configs(conn_params).create()
retriever = CortexRetriever(session, service_name="MEMBER_KB_SEARCH")
app = build_graph(RagConfig(), retriever)
```

## Scale and production considerations

The portfolio implements the **architecture and MLOps patterns** at demo scale.
For conversation on **millions of diagnosis/procedure codes**, **multi-tenant
isolation**, **millions of members**, and **performance tuning** at production
scale, see [scale-performance-multi-tenant.md](scale-performance-multi-tenant.md).

## Related docs

- [README](../README.md) — quickstart and technical deep dive
- [scale-performance-multi-tenant.md](scale-performance-multi-tenant.md) — codes, tenancy, members, latency
- [snowflake/README.md](../snowflake/README.md) — DDL deploy order and load examples
