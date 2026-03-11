-- models/staging/stg_invoice.sql
-- Staged invoices: filters out pre-migration data, adds fiscal helpers.
{{ config(materialized='view', tags=['finance', 'hourly']) }}

SELECT
    invoice_id,
    customer_id,
    invoice_date,
    DATE_TRUNC('month', invoice_date)  AS invoice_month,
    EXTRACT(QUARTER FROM invoice_date) AS fiscal_quarter,
    EXTRACT(YEAR FROM invoice_date)    AS fiscal_year,
    billing_address,
    billing_city,
    billing_state,
    billing_country,
    billing_postal_code,
    total
FROM {{ source('chinook', 'invoice') }}
WHERE invoice_date >= '2009-01-01'   -- Exclude pre-migration test rows
  AND total > 0                      -- Exclude zero-dollar adjustments
