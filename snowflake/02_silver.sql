-- Silver layer: cleansed, conformed, deduplicated policy documents.
-- Lineage: bronze.RAW_POLICY_FEED -> silver.BENEFIT_POLICY (transform + QA).

CREATE SCHEMA IF NOT EXISTS AI.SILVER;

CREATE OR REPLACE TABLE AI.SILVER.BENEFIT_POLICY AS
SELECT
    MD5(COALESCE(file_hash, source_id))           AS policy_sk,
    source_id                                     AS policy_id,
    TRIM(raw_title)                               AS title,
    LOWER(TRIM(raw_category))                     AS category,
    TRIM(raw_body)                                AS body_text,
    ingest_batch_id,
    ingested_at,
    CURRENT_TIMESTAMP()                           AS refined_at
FROM AI.BRONZE.RAW_POLICY_FEED
WHERE raw_body IS NOT NULL
  AND LENGTH(TRIM(raw_body)) > 0
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY COALESCE(file_hash, source_id)
    ORDER BY ingested_at DESC
) = 1;

-- Data quality checks (run after each load)
-- SELECT COUNT(*) AS null_bodies FROM AI.SILVER.BENEFIT_POLICY WHERE body_text IS NULL;
-- SELECT policy_id, COUNT(*) FROM AI.SILVER.BENEFIT_POLICY GROUP BY 1 HAVING COUNT(*) > 1;

COMMENT ON TABLE AI.SILVER.BENEFIT_POLICY IS
    'Silver conformed benefit policies. One row per policy_id; latest ingest wins.';
