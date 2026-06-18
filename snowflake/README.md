# Snowflake medallion + Cortex assets

Illustrative DDL for a **health-plan member navigation** portfolio. All data is
synthetic; no real PHI. Deploy in order:

| Script | Layer | Purpose |
|--------|-------|---------|
| `01_bronze.sql` | Bronze | Raw policy feeds and member events (landing) |
| `02_silver.sql` | Silver | Cleansed `BENEFIT_POLICY` with dedupe |
| `03_gold.sql` | Gold | `MEMBER_KB` (agent retrieval) + `PLAN_COST_SHARING` (analytics) |
| `04_semantic_views.sql` | Semantic | `MEMBER_COST_SHARING` semantic view + `V_MEMBER_COPAY` |
| `05_cortex_search.sql` | Cortex | `MEMBER_KB_SEARCH` hybrid search service |
| `06_governance.sql` | Governance | Roles, masking, row access policies |

## Lineage flow

```
source systems
    → BRONZE.RAW_POLICY_FEED
    → SILVER.BENEFIT_POLICY
    → GOLD.MEMBER_KB ──────────→ CORTEX SEARCH (MEMBER_KB_SEARCH)
    → GOLD.PLAN_COST_SHARING ──→ SEMANTIC views
                                      ↓
                              member_nav LangGraph agent
```

## Load local JSON into bronze (example)

```sql
-- After creating a stage and uploading data/knowledge_base.json documents:
INSERT INTO AI.BRONZE.RAW_POLICY_FEED (source_id, source_system, ingest_batch_id,
    raw_title, raw_category, raw_body, file_hash)
SELECT
    d.value:id::STRING,
    'portfolio_json',
    'batch_001',
    d.value:title::STRING,
    d.value:category::STRING,
    d.value:text::STRING,
    MD5(d.value:text::STRING)
FROM @AI.BRONZE.MEMBER_STAGE/knowledge_base.json
     (FILE_FORMAT => 'json_format'),
     LATERAL FLATTEN(input => $1:documents) d;
```

Then run `02_silver.sql` → `03_gold.sql` → `05_cortex_search.sql`.

## Python adapter

```python
from snowflake.snowpark import Session
from member_nav import CortexRetriever, build_graph, RagConfig

session = Session.builder.configs(conn_params).create()
retriever = CortexRetriever(session, service_name="MEMBER_KB_SEARCH")
app = build_graph(RagConfig(), retriever)
```
