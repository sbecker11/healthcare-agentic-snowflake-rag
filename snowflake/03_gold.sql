-- Gold layer: agent- and analytics-ready semantic assets.
-- Lineage: silver.BENEFIT_POLICY -> gold.MEMBER_KB.

CREATE SCHEMA IF NOT EXISTS AI.GOLD;

CREATE OR REPLACE TABLE AI.GOLD.MEMBER_KB (
    id          STRING      NOT NULL,
    title       STRING      NOT NULL,
    category    STRING      NOT NULL,
    text        STRING      NOT NULL,
    effective_date DATE     DEFAULT CURRENT_DATE(),
    is_active   BOOLEAN     DEFAULT TRUE,
    lineage_sk  STRING,     -- FK to silver.BENEFIT_POLICY.policy_sk
    updated_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Populate from silver (production: incremental MERGE on id)
INSERT INTO AI.GOLD.MEMBER_KB (id, title, category, text, lineage_sk)
SELECT
    policy_id   AS id,
    title,
    category,
    body_text   AS text,
    policy_sk   AS lineage_sk
FROM AI.SILVER.BENEFIT_POLICY;

-- Structured gold: plan cost-sharing summary for analytics + semantic views
CREATE OR REPLACE TABLE AI.GOLD.PLAN_COST_SHARING (
    plan_code       STRING      NOT NULL,
    plan_name       STRING      NOT NULL,
    service_type    STRING      NOT NULL,
    copay_usd       NUMBER(10,2),
    coinsurance_pct NUMBER(5,2),
    deductible_applies BOOLEAN DEFAULT FALSE,
    effective_date  DATE        DEFAULT CURRENT_DATE()
);

INSERT INTO AI.GOLD.PLAN_COST_SHARING (plan_code, plan_name, service_type, copay_usd, coinsurance_pct, deductible_applies)
VALUES
    ('PPO', 'PPO Plan', 'primary_care', 25.00, NULL, FALSE),
    ('PPO', 'PPO Plan', 'specialist', 50.00, NULL, FALSE),
    ('PPO', 'PPO Plan', 'urgent_care', 75.00, NULL, FALSE),
    ('PPO', 'PPO Plan', 'emergency_room', 250.00, NULL, FALSE),
    ('PPO', 'PPO Plan', 'telehealth_primary', 15.00, NULL, FALSE),
    ('HMO', 'HMO Plan', 'primary_care', 25.00, NULL, FALSE),
    ('HMO', 'HMO Plan', 'specialist', 50.00, NULL, TRUE);

COMMENT ON TABLE AI.GOLD.MEMBER_KB IS
    'Gold knowledge base for Cortex Search and agentic RAG. Curated member-facing policy text.';
COMMENT ON TABLE AI.GOLD.PLAN_COST_SHARING IS
    'Gold structured cost-sharing reference for trusted analytics and semantic views.';
