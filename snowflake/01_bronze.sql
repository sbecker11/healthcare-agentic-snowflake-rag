-- Bronze layer: raw policy and benefits content from source systems.
-- Lineage: source_system -> bronze (append-only landing).

CREATE DATABASE IF NOT EXISTS AI;
CREATE SCHEMA IF NOT EXISTS AI.BRONZE;

CREATE OR REPLACE TABLE AI.BRONZE.RAW_POLICY_FEED (
    source_id           STRING      NOT NULL,
    source_system       STRING      NOT NULL,   -- e.g. 'cms_guidance', 'plan_admin', 'vendor_pdf'
    ingest_batch_id     STRING      NOT NULL,
    ingested_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    raw_title           STRING,
    raw_category        STRING,
    raw_body            STRING,
    raw_metadata        VARIANT,                -- original JSON / PDF extract fields
    file_hash           STRING                  -- dedupe key across re-ingests
);

CREATE OR REPLACE TABLE AI.BRONZE.RAW_MEMBER_EVENTS (
    event_id            STRING      NOT NULL,
    event_type          STRING      NOT NULL,   -- enrollment, claim, auth_request
    member_id_hash      STRING,                 -- hashed identifier at bronze
    payload             VARIANT,
    ingested_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

COMMENT ON TABLE AI.BRONZE.RAW_POLICY_FEED IS
    'Bronze landing for unstructured benefits/policy content. Immutable ingest audit trail.';
