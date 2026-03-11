-- models/staging/stg_customer.sql
-- Staged customers: excludes test accounts and internal employees.
{{ config(materialized='view', tags=['core', 'hourly']) }}

SELECT
    customer_id,
    first_name,
    last_name,
    first_name || ' ' || last_name AS full_name,
    company,
    city,
    state,
    country,
    email,
    support_rep_id
FROM {{ source('chinook', 'customer') }}
WHERE email NOT LIKE '%@chinookcorp.com'  -- Exclude internal test accounts
  AND customer_id > 0                      -- Exclude placeholder row
