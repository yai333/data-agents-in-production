-- models/marts/monthly_revenue.sql
-- Monthly revenue aggregation by country.
-- Refreshed daily; used by the finance dashboard.
{{ config(
    materialized='table',
    tags=['finance', 'daily'],
    meta={'owner': 'finance-team', 'freshness': 'daily'}
) }}

SELECT
    i.invoice_month                        AS month,
    i.fiscal_quarter,
    i.fiscal_year,
    c.country,
    SUM(i.total)                           AS revenue,
    COUNT(DISTINCT i.invoice_id)           AS order_count,
    COUNT(DISTINCT i.customer_id)          AS unique_customers,
    SUM(i.total) / NULLIF(COUNT(DISTINCT i.customer_id), 0)
                                           AS revenue_per_customer
FROM {{ ref('stg_invoice') }} i
JOIN {{ ref('stg_customer') }} c
  ON i.customer_id = c.customer_id
GROUP BY 1, 2, 3, 4
