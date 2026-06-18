-- Semantic views: business-friendly abstractions over gold structured data.
-- Enables trusted analytics and natural-language querying (Cortex Analyst-ready).

CREATE SCHEMA IF NOT EXISTS AI.SEMANTIC;

CREATE OR REPLACE SEMANTIC VIEW AI.SEMANTIC.MEMBER_COST_SHARING
  TABLES (
    plan_cost_sharing AS AI.GOLD.PLAN_COST_SHARING
      PRIMARY KEY (plan_code, service_type)
      WITH SYNONYMS ('cost sharing', 'copays', 'member cost')
  )
  FACTS (
    plan_cost_sharing.copay_usd AS copay_usd
      COMMENT = 'Fixed dollar copay for the service',
    plan_cost_sharing.coinsurance_pct AS coinsurance_pct
      COMMENT = 'Percent member pays after deductible'
  )
  DIMENSIONS (
    plan_cost_sharing.plan_code AS plan_code
      COMMENT = 'Plan identifier (PPO, HMO)',
    plan_cost_sharing.plan_name AS plan_name,
    plan_cost_sharing.service_type AS service_type
      WITH SYNONYMS ('visit type', 'care setting'),
    plan_cost_sharing.deductible_applies AS deductible_applies
  );

-- Lightweight relational view for SQL consumers without semantic view support
CREATE OR REPLACE VIEW AI.SEMANTIC.V_MEMBER_COPAY AS
SELECT
    plan_code,
    plan_name,
    service_type,
    copay_usd,
    coinsurance_pct,
    deductible_applies
FROM AI.GOLD.PLAN_COST_SHARING
WHERE effective_date <= CURRENT_DATE();

COMMENT ON VIEW AI.SEMANTIC.V_MEMBER_COPAY IS
    'Business-friendly copay lookup by plan and service type.';
