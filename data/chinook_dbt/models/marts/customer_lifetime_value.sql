-- models/marts/customer_lifetime_value.sql
-- Customer lifetime value (CLV) calculation.
-- Refreshed daily; used by the retention team.
{{ config(
    materialized='table',
    tags=['retention', 'daily'],
    meta={'owner': 'retention-team', 'freshness': 'daily'}
) }}

WITH customer_orders AS (
    SELECT
        i.customer_id,
        MIN(i.invoice_date) AS first_purchase,
        MAX(i.invoice_date) AS last_purchase,
        COUNT(i.invoice_id) AS total_orders,
        SUM(i.total)        AS total_spent
    FROM {{ ref('stg_invoice') }} i
    GROUP BY i.customer_id
)

SELECT
    c.customer_id,
    c.full_name,
    c.country,
    co.first_purchase,
    co.last_purchase,
    co.total_orders,
    co.total_spent                                  AS lifetime_value,
    co.total_spent / NULLIF(co.total_orders, 0)     AS avg_order_value,
    CURRENT_DATE - co.last_purchase::date           AS days_since_last_order,
    CASE
        WHEN CURRENT_DATE - co.last_purchase::date <= 90
            THEN 'active'
        ELSE 'churned'
    END                                             AS status
FROM {{ ref('stg_customer') }} c
JOIN customer_orders co
  ON c.customer_id = co.customer_id
