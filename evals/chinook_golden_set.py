"""Starter golden set for Chinook database.

The Chinook database models a digital media store with:
- Artists, Albums, Tracks (music catalog)
- Customers, Invoices, InvoiceItems (sales)
- Employees (staff hierarchy)
- Genres, MediaTypes, Playlists (categorization)

This golden set provides 65 queries across semantic/business domain
categories (sales, catalog, playlist, employee, negative) and
difficulties to establish a baseline, enable meaningful comparisons,
and provide enough data for APO optimization (Chapter 4A) with
stratified train/val/test splits.
"""

from evals.golden_set import GoldenQuery


GOLDEN_SET: list[GoldenQuery] = [
    # ==========================================================================
    # COUNT QUERIES (Basic aggregation)
    # ==========================================================================
    GoldenQuery(
        id="count_001",
        question="How many artists are in the database?",
        sql="SELECT COUNT(*) FROM artist",
        expected_result=275,
        category="catalog",
        difficulty="easy",
        tables_used=["artist"],
    ),
    GoldenQuery(
        id="count_002",
        question="How many tracks are longer than 5 minutes?",
        sql="SELECT COUNT(*) FROM track WHERE milliseconds > 300000",
        expected_result=1069,
        category="catalog",
        difficulty="easy",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="count_003",
        question="How many customers are from the USA?",
        sql="SELECT COUNT(*) FROM customer WHERE country = 'USA'",
        expected_result=13,
        category="sales",
        difficulty="easy",
        tables_used=["customer"],
    ),
    GoldenQuery(
        id="count_004",
        question="How many albums are in the database?",
        sql="SELECT COUNT(*) FROM album",
        expected_result=347,
        category="catalog",
        difficulty="easy",
        tables_used=["album"],
    ),
    GoldenQuery(
        id="count_005",
        question="How many employees work at the company?",
        sql="SELECT COUNT(*) FROM employee",
        expected_result=8,
        category="employee",
        difficulty="easy",
        tables_used=["employee"],
    ),

    # ==========================================================================
    # FILTER QUERIES (WHERE clauses)
    # ==========================================================================
    GoldenQuery(
        id="filter_001",
        question="List all customers from Canada",
        sql="SELECT * FROM customer WHERE country = 'Canada'",
        expected_result=8,  # Count of rows
        category="sales",
        difficulty="easy",
        tables_used=["customer"],
    ),
    GoldenQuery(
        id="filter_002",
        question="Find all tracks in the 'Rock' genre",
        sql="""
            SELECT t.* FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            WHERE g.name = 'Rock'
        """,
        expected_result=1297,  # Count of rock tracks
        category="catalog",
        difficulty="medium",
        tables_used=["track", "genre"],
    ),
    GoldenQuery(
        id="filter_003",
        question="Show employees who report to Andrew Adams",
        sql="""
            SELECT e.* FROM employee e
            JOIN employee m ON e.reports_to = m.employee_id
            WHERE m.first_name = 'Andrew' AND m.last_name = 'Adams'
        """,
        expected_result=2,  # Nancy and Jane
        category="employee",
        difficulty="medium",
        tables_used=["employee"],
    ),

    # ==========================================================================
    # JOIN QUERIES (Table relationships)
    # ==========================================================================
    GoldenQuery(
        id="join_001",
        question="List all albums with their artist names",
        sql="""
            SELECT album.title, artist.name
            FROM album
            JOIN artist ON album.artist_id = artist.artist_id
        """,
        expected_result=347,  # Same as album count
        category="catalog",
        difficulty="easy",
        tables_used=["album", "artist"],
    ),
    GoldenQuery(
        id="join_002",
        question="Show all tracks with their album and artist names",
        sql="""
            SELECT t.name as track_name, al.title as album_title, ar.name as artist_name
            FROM track t
            JOIN album al ON t.album_id = al.album_id
            JOIN artist ar ON al.artist_id = ar.artist_id
        """,
        expected_result=3503,  # Total tracks
        category="catalog",
        difficulty="medium",
        tables_used=["track", "album", "artist"],
    ),
    GoldenQuery(
        id="join_003",
        question="List all invoices with customer names and countries",
        sql="""
            SELECT i.invoice_id, c.first_name, c.last_name, c.country, i.total
            FROM invoice i
            JOIN customer c ON i.customer_id = c.customer_id
        """,
        expected_result=412,  # Total invoices
        category="sales",
        difficulty="easy",
        tables_used=["invoice", "customer"],
    ),

    # ==========================================================================
    # GROUP QUERIES (Aggregation with GROUP BY)
    # ==========================================================================
    GoldenQuery(
        id="group_001",
        question="How many tracks are in each genre?",
        sql="""
            SELECT g.name, COUNT(*) as track_count
            FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY g.genre_id, g.name
            ORDER BY track_count DESC
        """,
        expected_result=25,  # Number of genres
        category="catalog",
        difficulty="medium",
        tables_used=["track", "genre"],
    ),
    GoldenQuery(
        id="group_002",
        question="What is the total sales amount by country?",
        sql="""
            SELECT c.country, SUM(i.total) as total_sales
            FROM invoice i
            JOIN customer c ON i.customer_id = c.customer_id
            GROUP BY c.country
            ORDER BY total_sales DESC
        """,
        expected_result=24,  # Number of countries
        category="sales",
        difficulty="medium",
        tables_used=["invoice", "customer"],
    ),
    GoldenQuery(
        id="group_003",
        question="How many albums does each artist have?",
        sql="""
            SELECT ar.name, COUNT(*) as album_count
            FROM album al
            JOIN artist ar ON al.artist_id = ar.artist_id
            GROUP BY ar.artist_id, ar.name
            ORDER BY album_count DESC
        """,
        expected_result=204,  # Artists with at least one album
        category="catalog",
        difficulty="medium",
        tables_used=["album", "artist"],
    ),

    # ==========================================================================
    # ORDER QUERIES (Sorting and limiting)
    # ==========================================================================
    GoldenQuery(
        id="order_001",
        question="What are the 10 longest tracks?",
        sql="""
            SELECT name, milliseconds
            FROM track
            ORDER BY milliseconds DESC
            LIMIT 10
        """,
        expected_result=10,
        category="catalog",
        difficulty="easy",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="order_002",
        question="Who are the top 5 customers by total purchase amount?",
        sql="""
            SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
            FROM customer c
            JOIN invoice i ON c.customer_id = i.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY total_spent DESC
            LIMIT 5
        """,
        expected_result=5,
        category="sales",
        difficulty="medium",
        tables_used=["customer", "invoice"],
    ),
    GoldenQuery(
        id="order_003",
        question="What are the 5 most expensive tracks?",
        sql="""
            SELECT name, unit_price
            FROM track
            ORDER BY unit_price DESC
            LIMIT 5
        """,
        expected_result=5,
        category="catalog",
        difficulty="easy",
        tables_used=["track"],
    ),

    # ==========================================================================
    # SUM QUERIES (Numeric aggregation)
    # ==========================================================================
    GoldenQuery(
        id="sum_001",
        question="What is the total revenue from all invoices?",
        sql="SELECT SUM(total) FROM invoice",
        expected_result=2328.60,  # Approximate
        category="sales",
        difficulty="easy",
        tables_used=["invoice"],
    ),
    GoldenQuery(
        id="sum_002",
        question="What is the total duration of all tracks in minutes?",
        sql="SELECT SUM(milliseconds) / 60000.0 as total_minutes FROM track",
        expected_result=22979.63,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),

    # ==========================================================================
    # DATE QUERIES (Temporal filtering)
    # ==========================================================================
    GoldenQuery(
        id="date_001",
        question="How many invoices were created in 2023?",
        sql="""
            SELECT COUNT(*) FROM invoice
            WHERE invoice_date >= '2023-01-01' AND invoice_date < '2024-01-01'
        """,
        expected_result=83,
        category="sales",
        difficulty="medium",
        tables_used=["invoice"],
    ),
    GoldenQuery(
        id="date_002",
        question="What is the total revenue for 2022?",
        sql="""
            SELECT SUM(total) FROM invoice
            WHERE invoice_date >= '2022-01-01' AND invoice_date < '2023-01-01'
        """,
        expected_result=481.45,
        category="sales",
        difficulty="medium",
        tables_used=["invoice"],
    ),

    # ==========================================================================
    # COMPLEX QUERIES (Multi-step reasoning)
    # ==========================================================================
    GoldenQuery(
        id="complex_001",
        question="Which customers spent the most in 2023?",
        sql="""
            SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
            FROM customer c
            JOIN invoice i ON c.customer_id = i.customer_id
            WHERE i.invoice_date >= '2023-01-01' AND i.invoice_date < '2024-01-01'
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY total_spent DESC
            LIMIT 10
        """,
        expected_result=10,
        category="sales",
        difficulty="hard",
        tables_used=["customer", "invoice"],
    ),
    GoldenQuery(
        id="complex_002",
        question="What are the most popular genres by number of tracks sold?",
        sql="""
            SELECT g.name, COUNT(il.track_id) as tracks_sold
            FROM invoice_line il
            JOIN track t ON il.track_id = t.track_id
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY g.genre_id, g.name
            ORDER BY tracks_sold DESC
            LIMIT 10
        """,
        expected_result=10,
        category="catalog",
        difficulty="hard",
        tables_used=["invoice_line", "track", "genre"],
    ),
    GoldenQuery(
        id="complex_003",
        question="Which artists have the highest total sales revenue?",
        sql="""
            SELECT ar.name, SUM(il.unit_price * il.quantity) as revenue
            FROM invoice_line il
            JOIN track t ON il.track_id = t.track_id
            JOIN album al ON t.album_id = al.album_id
            JOIN artist ar ON al.artist_id = ar.artist_id
            GROUP BY ar.artist_id, ar.name
            ORDER BY revenue DESC
            LIMIT 10
        """,
        expected_result=10,
        category="catalog",
        difficulty="hard",
        tables_used=["invoice_line", "track", "album", "artist"],
    ),
    GoldenQuery(
        id="complex_004",
        question="What is the average invoice total by customer country, for countries with more than 5 customers?",
        sql="""
            SELECT c.country, AVG(i.total) as avg_invoice
            FROM invoice i
            JOIN customer c ON i.customer_id = c.customer_id
            WHERE c.country IN (
                SELECT country FROM customer
                GROUP BY country
                HAVING COUNT(*) > 5
            )
            GROUP BY c.country
            ORDER BY avg_invoice DESC
        """,
        expected_result=2,  # Countries with >5 customers (USA, Canada)
        category="sales",
        difficulty="hard",
        tables_used=["invoice", "customer"],
    ),

    # ==========================================================================
    # DISTINCT & NULL HANDLING
    # ==========================================================================
    GoldenQuery(
        id="distinct_001",
        question="How many distinct composers are in the track database?",
        sql="SELECT COUNT(DISTINCT composer) FROM track WHERE composer IS NOT NULL",
        expected_result=853,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="distinct_002",
        question="List all tracks that have no composer listed",
        sql="SELECT name FROM track WHERE composer IS NULL",
        expected_result=977,
        category="catalog",
        difficulty="easy",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="distinct_003",
        question="How many unique cities do our customers come from?",
        sql="SELECT COUNT(DISTINCT city) FROM customer",
        expected_result=53,
        category="sales",
        difficulty="easy",
        tables_used=["customer"],
    ),

    # ==========================================================================
    # ADDITIONAL FILTER QUERIES
    # ==========================================================================
    GoldenQuery(
        id="filter_004",
        question="Find all tracks between 3 and 5 minutes long",
        sql="""
            SELECT name, milliseconds
            FROM track
            WHERE milliseconds BETWEEN 180000 AND 300000
            ORDER BY milliseconds
        """,
        expected_result=1954,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="filter_005",
        question="List all employees hired before 2003",
        sql="""
            SELECT first_name, last_name, hire_date
            FROM employee
            WHERE hire_date < '2003-01-01'
        """,
        expected_result=3,
        category="employee",
        difficulty="medium",
        tables_used=["employee"],
    ),

    # ==========================================================================
    # ADDITIONAL JOIN QUERIES
    # ==========================================================================
    GoldenQuery(
        id="join_004",
        question="Show all tracks with their genre name and media type",
        sql="""
            SELECT t.name as track, g.name as genre, mt.name as media_type
            FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            JOIN media_type mt ON t.media_type_id = mt.media_type_id
        """,
        expected_result=3503,
        category="catalog",
        difficulty="medium",
        tables_used=["track", "genre", "media_type"],
    ),
    GoldenQuery(
        id="join_005",
        question="List all invoice items with track name and customer name",
        sql="""
            SELECT
                c.first_name || ' ' || c.last_name as customer,
                t.name as track,
                il.unit_price,
                il.quantity
            FROM invoice_line il
            JOIN invoice i ON il.invoice_id = i.invoice_id
            JOIN customer c ON i.customer_id = c.customer_id
            JOIN track t ON il.track_id = t.track_id
        """,
        expected_result=2240,
        category="sales",
        difficulty="medium",
        tables_used=["invoice_line", "invoice", "customer", "track"],
    ),

    # ==========================================================================
    # ADDITIONAL GROUP BY QUERIES
    # ==========================================================================
    GoldenQuery(
        id="group_004",
        question="What is the total revenue per genre?",
        sql="""
            SELECT g.name, SUM(il.unit_price * il.quantity) as revenue
            FROM invoice_line il
            JOIN track t ON il.track_id = t.track_id
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY g.genre_id, g.name
            ORDER BY revenue DESC
        """,
        expected_result=24,
        category="catalog",
        difficulty="medium",
        tables_used=["invoice_line", "track", "genre"],
    ),

    # ==========================================================================
    # ADDITIONAL DATE QUERIES
    # ==========================================================================
    GoldenQuery(
        id="date_003",
        question="How many invoices were created each month in 2022?",
        sql="""
            SELECT
                EXTRACT(MONTH FROM invoice_date) as month,
                COUNT(*) as invoice_count
            FROM invoice
            WHERE invoice_date >= '2022-01-01' AND invoice_date < '2023-01-01'
            GROUP BY EXTRACT(MONTH FROM invoice_date)
            ORDER BY month
        """,
        expected_result=12,
        category="sales",
        difficulty="medium",
        tables_used=["invoice"],
    ),

    # ==========================================================================
    # HAVING QUERIES (Filtered aggregation)
    # ==========================================================================
    GoldenQuery(
        id="having_001",
        question="Which genres have more than 100 tracks?",
        sql="""
            SELECT g.name, COUNT(*) as track_count
            FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY g.genre_id, g.name
            HAVING COUNT(*) > 100
            ORDER BY track_count DESC
        """,
        expected_result=5,
        category="catalog",
        difficulty="medium",
        tables_used=["track", "genre"],
    ),
    GoldenQuery(
        id="having_002",
        question="Which countries have more than 5 customers?",
        sql="""
            SELECT country, COUNT(*) as customer_count
            FROM customer
            GROUP BY country
            HAVING COUNT(*) > 5
            ORDER BY customer_count DESC
        """,
        expected_result=2,
        category="sales",
        difficulty="medium",
        tables_used=["customer"],
    ),
    GoldenQuery(
        id="having_003",
        question="Which artists have more than 10 albums?",
        sql="""
            SELECT ar.name, COUNT(*) as album_count
            FROM album al
            JOIN artist ar ON al.artist_id = ar.artist_id
            GROUP BY ar.artist_id, ar.name
            HAVING COUNT(*) > 10
            ORDER BY album_count DESC
        """,
        expected_result=3,
        category="catalog",
        difficulty="medium",
        tables_used=["album", "artist"],
    ),
    GoldenQuery(
        id="having_004",
        question="Which playlists contain more than 100 tracks?",
        sql="""
            SELECT p.name, COUNT(*) as track_count
            FROM playlist p
            JOIN playlist_track pt ON p.playlist_id = pt.playlist_id
            GROUP BY p.playlist_id, p.name
            HAVING COUNT(*) > 100
            ORDER BY track_count DESC
        """,
        expected_result=5,
        category="playlist",
        difficulty="medium",
        tables_used=["playlist", "playlist_track"],
    ),

    # ==========================================================================
    # PATTERN MATCHING
    # ==========================================================================
    GoldenQuery(
        id="pattern_001",
        question="Find all customers whose last name starts with 'S'",
        sql="""
            SELECT first_name, last_name
            FROM customer
            WHERE last_name LIKE 'S%'
            ORDER BY last_name
        """,
        expected_result=8,
        category="sales",
        difficulty="easy",
        tables_used=["customer"],
    ),
    GoldenQuery(
        id="pattern_002",
        question="Find all tracks with 'love' in the title",
        sql="SELECT name FROM track WHERE LOWER(name) LIKE '%love%'",
        expected_result=114,
        category="catalog",
        difficulty="easy",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="pattern_003",
        question="List all artists whose name contains 'The'",
        sql="SELECT name FROM artist WHERE name LIKE '%The%' ORDER BY name",
        expected_result=17,
        category="catalog",
        difficulty="easy",
        tables_used=["artist"],
    ),

    # ==========================================================================
    # CASE & CONDITIONAL LOGIC
    # ==========================================================================
    GoldenQuery(
        id="case_001",
        question="Categorize tracks as Short (under 3 min), Medium (3-5 min), or Long (over 5 min) and count each",
        sql="""
            SELECT
                CASE
                    WHEN milliseconds < 180000 THEN 'Short'
                    WHEN milliseconds <= 300000 THEN 'Medium'
                    ELSE 'Long'
                END as length_category,
                COUNT(*) as track_count
            FROM track
            GROUP BY length_category
            ORDER BY track_count DESC
        """,
        expected_result=3,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="case_002",
        question="Classify customers by total spending: High (over $40), Medium ($20-$40), Low (under $20) and count each tier",
        sql="""
            SELECT customer_tier, COUNT(*) as customer_count FROM (
                SELECT
                    CASE
                        WHEN SUM(i.total) > 40 THEN 'High'
                        WHEN SUM(i.total) >= 20 THEN 'Medium'
                        ELSE 'Low'
                    END as customer_tier
                FROM customer c
                JOIN invoice i ON c.customer_id = i.customer_id
                GROUP BY c.customer_id
            ) tiers
            GROUP BY customer_tier
            ORDER BY customer_count DESC
        """,
        expected_result=2,
        category="sales",
        difficulty="hard",
        tables_used=["customer", "invoice"],
    ),

    # ==========================================================================
    # SUBQUERY QUERIES (Nested logic)
    # ==========================================================================
    GoldenQuery(
        id="subquery_001",
        question="Find customers who have never made a purchase",
        sql="""
            SELECT first_name, last_name
            FROM customer
            WHERE customer_id NOT IN (
                SELECT DISTINCT customer_id FROM invoice
            )
        """,
        expected_result=0,
        category="sales",
        difficulty="hard",
        tables_used=["customer", "invoice"],
    ),
    GoldenQuery(
        id="subquery_002",
        question="Which tracks are priced above the average track price?",
        sql="""
            SELECT name, unit_price
            FROM track
            WHERE unit_price > (SELECT AVG(unit_price) FROM track)
            ORDER BY unit_price DESC
        """,
        expected_result=213,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="subquery_003",
        question="Which genres have above-average track counts?",
        sql="""
            SELECT g.name, COUNT(*) as track_count
            FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY g.genre_id, g.name
            HAVING COUNT(*) > (
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) as cnt
                    FROM track
                    GROUP BY genre_id
                ) sub
            )
            ORDER BY track_count DESC
        """,
        expected_result=4,
        category="catalog",
        difficulty="hard",
        tables_used=["track", "genre"],
    ),
    GoldenQuery(
        id="subquery_004",
        question="What percentage of all tracks are in the Rock genre?",
        sql="""
            SELECT
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM track), 2) as rock_percentage
            FROM track t
            JOIN genre g ON t.genre_id = g.genre_id
            WHERE g.name = 'Rock'
        """,
        expected_result=37.03,
        category="catalog",
        difficulty="hard",
        tables_used=["track", "genre"],
    ),
    GoldenQuery(
        id="subquery_005",
        question="Find artists who have tracks in more than 3 different genres",
        sql="""
            SELECT ar.name, COUNT(DISTINCT g.genre_id) as genre_count
            FROM artist ar
            JOIN album al ON ar.artist_id = al.artist_id
            JOIN track t ON al.album_id = t.album_id
            JOIN genre g ON t.genre_id = g.genre_id
            GROUP BY ar.artist_id, ar.name
            HAVING COUNT(DISTINCT g.genre_id) > 3
            ORDER BY genre_count DESC
        """,
        expected_result=1,
        category="catalog",
        difficulty="hard",
        tables_used=["artist", "album", "track", "genre"],
    ),

    # ==========================================================================
    # WINDOW FUNCTION QUERIES
    # ==========================================================================
    GoldenQuery(
        id="window_001",
        question="Rank customers by their total spending",
        sql="""
            SELECT
                c.first_name, c.last_name,
                SUM(i.total) as total_spent,
                RANK() OVER (ORDER BY SUM(i.total) DESC) as spending_rank
            FROM customer c
            JOIN invoice i ON c.customer_id = i.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY spending_rank
        """,
        expected_result=59,
        category="sales",
        difficulty="hard",
        tables_used=["customer", "invoice"],
    ),
    GoldenQuery(
        id="window_002",
        question="Show each invoice with a running total of revenue",
        sql="""
            SELECT
                invoice_id,
                invoice_date,
                total,
                SUM(total) OVER (ORDER BY invoice_date, invoice_id) as running_total
            FROM invoice
            ORDER BY invoice_date, invoice_id
        """,
        expected_result=412,
        category="sales",
        difficulty="hard",
        tables_used=["invoice"],
    ),
    GoldenQuery(
        id="window_003",
        question="Find the top 3 best-selling tracks in each genre",
        sql="""
            SELECT genre_name, track_name, tracks_sold FROM (
                SELECT
                    g.name as genre_name,
                    t.name as track_name,
                    COUNT(il.invoice_line_id) as tracks_sold,
                    ROW_NUMBER() OVER (
                        PARTITION BY g.genre_id
                        ORDER BY COUNT(il.invoice_line_id) DESC
                    ) as rn
                FROM genre g
                JOIN track t ON g.genre_id = t.genre_id
                JOIN invoice_line il ON t.track_id = il.track_id
                GROUP BY g.genre_id, g.name, t.track_id, t.name
            ) ranked
            WHERE rn <= 3
            ORDER BY genre_name, tracks_sold DESC
        """,
        expected_result=72,
        category="catalog",
        difficulty="hard",
        tables_used=["genre", "track", "invoice_line"],
    ),
    GoldenQuery(
        id="window_004",
        question="For each customer, show their most recent invoice",
        sql="""
            SELECT first_name, last_name, invoice_date, total FROM (
                SELECT
                    c.first_name, c.last_name,
                    i.invoice_date, i.total,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.customer_id
                        ORDER BY i.invoice_date DESC
                    ) as rn
                FROM customer c
                JOIN invoice i ON c.customer_id = i.customer_id
            ) latest
            WHERE rn = 1
            ORDER BY last_name, first_name
        """,
        expected_result=59,
        category="sales",
        difficulty="hard",
        tables_used=["customer", "invoice"],
    ),

    # ==========================================================================
    # BUSINESS QUESTIONS (Multi-table reasoning)
    # ==========================================================================
    GoldenQuery(
        id="business_001",
        question="What is each employee's total sales as a support representative?",
        sql="""
            SELECT
                e.first_name, e.last_name,
                SUM(i.total) as total_sales
            FROM employee e
            JOIN customer c ON e.employee_id = c.support_rep_id
            JOIN invoice i ON c.customer_id = i.customer_id
            GROUP BY e.employee_id, e.first_name, e.last_name
            ORDER BY total_sales DESC
        """,
        expected_result=3,
        category="sales",
        difficulty="hard",
        tables_used=["employee", "customer", "invoice"],
    ),
    GoldenQuery(
        id="business_002",
        question="Which tracks appear in the most playlists?",
        sql="""
            SELECT t.name, COUNT(pt.playlist_id) as playlist_count
            FROM track t
            JOIN playlist_track pt ON t.track_id = pt.track_id
            GROUP BY t.track_id, t.name
            ORDER BY playlist_count DESC
            LIMIT 10
        """,
        expected_result=10,
        category="catalog",
        difficulty="medium",
        tables_used=["track", "playlist_track"],
    ),
    GoldenQuery(
        id="business_003",
        question="What is the average number of tracks per album?",
        sql="""
            SELECT ROUND(AVG(track_count), 2) as avg_tracks_per_album
            FROM (
                SELECT album_id, COUNT(*) as track_count
                FROM track
                GROUP BY album_id
            ) album_counts
        """,
        expected_result=10.09,
        category="catalog",
        difficulty="medium",
        tables_used=["track"],
    ),
    GoldenQuery(
        id="business_004",
        question="Which media type generates the most revenue?",
        sql="""
            SELECT mt.name, SUM(il.unit_price * il.quantity) as revenue
            FROM invoice_line il
            JOIN track t ON il.track_id = t.track_id
            JOIN media_type mt ON t.media_type_id = mt.media_type_id
            GROUP BY mt.media_type_id, mt.name
            ORDER BY revenue DESC
        """,
        expected_result=5,
        category="catalog",
        difficulty="medium",
        tables_used=["invoice_line", "track", "media_type"],
    ),
    GoldenQuery(
        id="business_005",
        question="What is the average invoice total by billing country?",
        sql="""
            SELECT billing_country, ROUND(AVG(total), 2) as avg_total
            FROM invoice
            GROUP BY billing_country
            ORDER BY avg_total DESC
        """,
        expected_result=24,
        category="sales",
        difficulty="medium",
        tables_used=["invoice"],
    ),

    # ==========================================================================
    # ADDITIONAL COMPLEX QUERIES
    # ==========================================================================
    GoldenQuery(
        id="complex_005",
        question="Which city has the highest total invoice amount?",
        sql="""
            SELECT billing_city, SUM(total) as city_total
            FROM invoice
            GROUP BY billing_city
            ORDER BY city_total DESC
            LIMIT 1
        """,
        expected_result=1,
        category="sales",
        difficulty="medium",
        tables_used=["invoice"],
    ),
    GoldenQuery(
        id="complex_006",
        question="How many tracks have been purchased more than once?",
        sql="""
            SELECT COUNT(*) FROM (
                SELECT track_id
                FROM invoice_line
                GROUP BY track_id
                HAVING COUNT(*) > 1
            ) multi_purchase
        """,
        expected_result=256,
        category="sales",
        difficulty="hard",
        tables_used=["invoice_line"],
    ),
    GoldenQuery(
        id="complex_007",
        question="What is the most popular artist in each country by sales?",
        sql="""
            SELECT country, artist_name, total_sales FROM (
                SELECT
                    c.country,
                    ar.name as artist_name,
                    SUM(il.unit_price * il.quantity) as total_sales,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.country
                        ORDER BY SUM(il.unit_price * il.quantity) DESC
                    ) as rn
                FROM invoice_line il
                JOIN invoice i ON il.invoice_id = i.invoice_id
                JOIN customer c ON i.customer_id = c.customer_id
                JOIN track t ON il.track_id = t.track_id
                JOIN album al ON t.album_id = al.album_id
                JOIN artist ar ON al.artist_id = ar.artist_id
                GROUP BY c.country, ar.artist_id, ar.name
            ) ranked
            WHERE rn = 1
            ORDER BY country
        """,
        expected_result=24,
        category="sales",
        difficulty="hard",
        tables_used=["invoice_line", "invoice", "customer", "track", "album", "artist"],
    ),

    # ==========================================================================
    # NEGATIVE CASES (Agent should decline or ask clarification)
    # ==========================================================================
    GoldenQuery(
        id="negative_001",
        question="What's the weather in Seattle?",
        sql="",  # Should decline - out of scope
        expected_result="decline",
        category="negative",
        difficulty="easy",
        tables_used=[],
    ),
    GoldenQuery(
        id="negative_002",
        question="Delete all customers from the database",
        sql="",  # Should refuse - dangerous operation
        expected_result="refuse",
        category="negative",
        difficulty="easy",
        tables_used=[],
    ),
    GoldenQuery(
        id="negative_003",
        question="Show me the data",
        sql="",  # Should ask clarification - too ambiguous
        expected_result="clarify",
        category="negative",
        difficulty="easy",
        tables_used=[],
    ),
    GoldenQuery(
        id="negative_004",
        question="What were our sales in 2025?",
        sql="",  # Should explain - data doesn't exist (Chinook ends in 2013)
        expected_result="no_data",
        category="negative",
        difficulty="medium",
        tables_used=["invoice"],
    ),
    GoldenQuery(
        id="negative_005",
        question="Show large orders",
        sql="",  # Should ask clarification - "large" is undefined
        expected_result="clarify",
        category="negative",
        difficulty="medium",
        tables_used=[],
    ),
]


def get_queries_by_category(category: str) -> list[GoldenQuery]:
    """Get all queries for a specific category."""
    return [q for q in GOLDEN_SET if q.category == category]


def get_queries_by_difficulty(difficulty: str) -> list[GoldenQuery]:
    """Get all queries for a specific difficulty level."""
    return [q for q in GOLDEN_SET if q.difficulty == difficulty]


def get_categories() -> list[str]:
    """Get all unique categories in the golden set."""
    return sorted(set(q.category for q in GOLDEN_SET))


def get_summary() -> dict:
    """Get a summary of the golden set composition."""
    by_category = {}
    by_difficulty = {}

    for q in GOLDEN_SET:
        by_category[q.category] = by_category.get(q.category, 0) + 1
        by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1

    return {
        "total": len(GOLDEN_SET),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }
