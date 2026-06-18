-- Cortex Search: operationalize semantic intelligence on gold MEMBER_KB.
-- Feeds the member_nav CortexRetriever adapter (hybrid vector + keyword + rerank).

CREATE SCHEMA IF NOT EXISTS AI.MEMBER_NAV;

CREATE OR REPLACE CORTEX SEARCH SERVICE AI.MEMBER_NAV.MEMBER_KB_SEARCH
    ON text
    ATTRIBUTES id, title, category
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 hour'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
    AS (
        SELECT id, title, category, text
        FROM AI.GOLD.MEMBER_KB
        WHERE is_active = TRUE
    );

COMMENT ON CORTEX SEARCH SERVICE AI.MEMBER_NAV.MEMBER_KB_SEARCH IS
    'Managed hybrid search over member benefits knowledge. Production retrieval for agentic RAG.';
