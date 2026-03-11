"""Few-shot example library for Chinook database.

This library contains 40+ examples categorized by semantic/business domain:
- sales: customer, invoice, invoice_line queries
- catalog: artist, album, track, genre, media_type queries
- employee: employee queries
- playlist: playlist, playlist_track queries

Each example has:
- Core fields (shown to LLM): question, sql, explanation (optional)
- Metadata (for retrieval/eval only): id, tables_used, category, difficulty

The render_examples() function outputs ONLY the core fields to the prompt.
"""
from __future__ import annotations

from src.retrieval.models import FewShotExample


CHINOOK_EXAMPLES: list[FewShotExample] = [
    # ==========================================================================
    # COUNT QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_count_001",
        question="How many artists are there?",
        sql="SELECT COUNT(*) FROM artist",
        tables_used=["artist"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_count_002",
        question="How many tracks have a duration longer than 5 minutes?",
        sql="SELECT COUNT(*) FROM track WHERE milliseconds > 300000",
        tables_used=["track"],
        category="catalog",
        difficulty="easy",
        explanation="5 minutes = 300,000 milliseconds",
    ),
    FewShotExample(
        id="ex_count_003",
        question="How many customers are from Brazil?",
        sql="SELECT COUNT(*) FROM customer WHERE country = 'Brazil'",
        tables_used=["customer"],
        category="sales",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_count_004",
        question="How many albums does the artist 'AC/DC' have?",
        sql="""SELECT COUNT(*) FROM album al
JOIN artist ar ON al.artist_id = ar.artist_id
WHERE ar.name = 'AC/DC'""",
        tables_used=["album", "artist"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_count_005",
        question="How many distinct genres are in the database?",
        sql="SELECT COUNT(DISTINCT genre_id) FROM genre",
        tables_used=["genre"],
        category="catalog",
        difficulty="easy",
    ),

    # ==========================================================================
    # FILTER QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_filter_001",
        question="List all customers from Canada",
        sql="SELECT * FROM customer WHERE country = 'Canada'",
        tables_used=["customer"],
        category="sales",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_filter_002",
        question="Find all tracks that cost more than $0.99",
        sql="SELECT * FROM track WHERE unit_price > 0.99",
        tables_used=["track"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_filter_003",
        question="Show employees who work in Calgary",
        sql="SELECT * FROM employee WHERE city = 'Calgary'",
        tables_used=["employee"],
        category="employee",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_filter_004",
        question="Find tracks in the 'Rock' genre",
        sql="""SELECT t.* FROM track t
JOIN genre g ON t.genre_id = g.genre_id
WHERE g.name = 'Rock'""",
        tables_used=["track", "genre"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_filter_005",
        question="List invoices with a total greater than $10",
        sql="SELECT * FROM invoice WHERE total > 10",
        tables_used=["invoice"],
        category="sales",
        difficulty="easy",
    ),

    # ==========================================================================
    # JOIN QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_join_001",
        question="List all albums with their artist names",
        sql="""SELECT al.title, ar.name as artist_name
FROM album al
JOIN artist ar ON al.artist_id = ar.artist_id""",
        tables_used=["album", "artist"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_join_002",
        question="Show all tracks with their album and artist",
        sql="""SELECT t.name as track_name, al.title as album, ar.name as artist
FROM track t
JOIN album al ON t.album_id = al.album_id
JOIN artist ar ON al.artist_id = ar.artist_id""",
        tables_used=["track", "album", "artist"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_join_003",
        question="List all invoices with customer names",
        sql="""SELECT i.invoice_id, c.first_name, c.last_name, i.total
FROM invoice i
JOIN customer c ON i.customer_id = c.customer_id""",
        tables_used=["invoice", "customer"],
        category="sales",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_join_004",
        question="Show each track with its genre name",
        sql="""SELECT t.name as track_name, g.name as genre
FROM track t
JOIN genre g ON t.genre_id = g.genre_id""",
        tables_used=["track", "genre"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_join_005",
        question="List employees and their managers",
        sql="""SELECT e.first_name || ' ' || e.last_name as employee,
       m.first_name || ' ' || m.last_name as manager
FROM employee e
LEFT JOIN employee m ON e.reports_to = m.employee_id""",
        tables_used=["employee"],
        category="employee",
        difficulty="medium",
        explanation="Self-join to get manager names",
    ),

    # ==========================================================================
    # GROUP QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_group_001",
        question="How many tracks are in each genre?",
        sql="""SELECT g.name as genre, COUNT(*) as track_count
FROM track t
JOIN genre g ON t.genre_id = g.genre_id
GROUP BY g.genre_id, g.name
ORDER BY track_count DESC""",
        tables_used=["track", "genre"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_group_002",
        question="Show the number of albums per artist",
        sql="""SELECT ar.name as artist, COUNT(*) as album_count
FROM album al
JOIN artist ar ON al.artist_id = ar.artist_id
GROUP BY ar.artist_id, ar.name
ORDER BY album_count DESC""",
        tables_used=["album", "artist"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_group_003",
        question="How many customers are in each country?",
        sql="""SELECT country, COUNT(*) as customer_count
FROM customer
GROUP BY country
ORDER BY customer_count DESC""",
        tables_used=["customer"],
        category="sales",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_group_004",
        question="What is the average track length by genre?",
        sql="""SELECT g.name as genre, AVG(t.milliseconds/1000.0) as avg_seconds
FROM track t
JOIN genre g ON t.genre_id = g.genre_id
GROUP BY g.genre_id, g.name
ORDER BY avg_seconds DESC""",
        tables_used=["track", "genre"],
        category="catalog",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_group_005",
        question="How many invoices per customer?",
        sql="""SELECT c.first_name, c.last_name, COUNT(*) as invoice_count
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY invoice_count DESC""",
        tables_used=["customer", "invoice"],
        category="sales",
        difficulty="medium",
    ),

    # ==========================================================================
    # ORDER/LIMIT QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_order_001",
        question="What are the 10 longest tracks?",
        sql="""SELECT name, milliseconds/1000 as seconds
FROM track
ORDER BY milliseconds DESC
LIMIT 10""",
        tables_used=["track"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_order_002",
        question="Show the 5 most expensive tracks",
        sql="""SELECT name, unit_price
FROM track
ORDER BY unit_price DESC
LIMIT 5""",
        tables_used=["track"],
        category="catalog",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_order_003",
        question="List the top 10 customers by total spending",
        sql="""SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC
LIMIT 10""",
        tables_used=["customer", "invoice"],
        category="sales",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_order_004",
        question="What are the bottom 5 selling genres?",
        sql="""SELECT g.name as genre, COUNT(il.track_id) as tracks_sold
FROM genre g
JOIN track t ON g.genre_id = t.genre_id
LEFT JOIN invoice_line il ON t.track_id = il.track_id
GROUP BY g.genre_id, g.name
ORDER BY tracks_sold ASC
LIMIT 5""",
        tables_used=["genre", "track", "invoice_line"],
        category="catalog",
        difficulty="hard",
    ),

    # ==========================================================================
    # SUM/REVENUE QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_sum_001",
        question="What is the total revenue from all invoices?",
        sql="SELECT SUM(total) as total_revenue FROM invoice",
        tables_used=["invoice"],
        category="sales",
        difficulty="easy",
    ),
    FewShotExample(
        id="ex_sum_002",
        question="What is the total revenue by country?",
        sql="""SELECT c.country, SUM(i.total) as revenue
FROM invoice i
JOIN customer c ON i.customer_id = c.customer_id
GROUP BY c.country
ORDER BY revenue DESC""",
        tables_used=["invoice", "customer"],
        category="sales",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_sum_003",
        question="What is the total sales by employee?",
        sql="""SELECT e.first_name, e.last_name, SUM(i.total) as total_sales
FROM employee e
JOIN customer c ON e.employee_id = c.support_rep_id
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY e.employee_id, e.first_name, e.last_name
ORDER BY total_sales DESC""",
        tables_used=["employee", "customer", "invoice"],
        category="sales",
        difficulty="hard",
    ),
    FewShotExample(
        id="ex_sum_004",
        question="Calculate the total duration of all tracks in hours",
        sql="SELECT SUM(milliseconds) / 3600000.0 as total_hours FROM track",
        tables_used=["track"],
        category="catalog",
        difficulty="easy",
        explanation="milliseconds to hours: divide by 3,600,000",
    ),

    # ==========================================================================
    # DATE QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_date_001",
        question="How many invoices were created in 2013?",
        sql="""SELECT COUNT(*) FROM invoice
WHERE invoice_date >= '2013-01-01' AND invoice_date < '2014-01-01'""",
        tables_used=["invoice"],
        category="sales",
        difficulty="medium",
        explanation="Using date range for better index usage than YEAR()",
    ),
    FewShotExample(
        id="ex_date_002",
        question="What was the total revenue in 2012?",
        sql="""SELECT SUM(total) as revenue_2012 FROM invoice
WHERE invoice_date >= '2012-01-01' AND invoice_date < '2013-01-01'""",
        tables_used=["invoice"],
        category="sales",
        difficulty="medium",
    ),
    FewShotExample(
        id="ex_date_003",
        question="Show monthly revenue for 2013",
        sql="""SELECT DATE_TRUNC('month', invoice_date) as month, SUM(total) as revenue
FROM invoice
WHERE invoice_date >= '2013-01-01' AND invoice_date < '2014-01-01'
GROUP BY DATE_TRUNC('month', invoice_date)
ORDER BY month""",
        tables_used=["invoice"],
        category="sales",
        difficulty="hard",
    ),
    FewShotExample(
        id="ex_date_004",
        question="List employees hired before 2003",
        sql="SELECT * FROM employee WHERE hire_date < '2003-01-01'",
        tables_used=["employee"],
        category="employee",
        difficulty="easy",
    ),

    # ==========================================================================
    # COMPLEX QUERIES
    # ==========================================================================
    FewShotExample(
        id="ex_complex_001",
        question="What are the most popular genres by number of tracks sold?",
        sql="""SELECT g.name as genre, COUNT(il.invoice_line_id) as tracks_sold
FROM genre g
JOIN track t ON g.genre_id = t.genre_id
JOIN invoice_line il ON t.track_id = il.track_id
GROUP BY g.genre_id, g.name
ORDER BY tracks_sold DESC""",
        tables_used=["genre", "track", "invoice_line"],
        category="catalog",
        difficulty="hard",
    ),
    FewShotExample(
        id="ex_complex_002",
        question="Which artists have generated the most revenue?",
        sql="""SELECT ar.name as artist, SUM(il.unit_price * il.quantity) as revenue
FROM artist ar
JOIN album al ON ar.artist_id = al.artist_id
JOIN track t ON al.album_id = t.album_id
JOIN invoice_line il ON t.track_id = il.track_id
GROUP BY ar.artist_id, ar.name
ORDER BY revenue DESC
LIMIT 10""",
        tables_used=["artist", "album", "track", "invoice_line"],
        category="catalog",
        difficulty="hard",
    ),
    FewShotExample(
        id="ex_complex_003",
        question="Find customers who have spent more than $40 total",
        sql="""SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING SUM(i.total) > 40
ORDER BY total_spent DESC""",
        tables_used=["customer", "invoice"],
        category="sales",
        difficulty="hard",
        explanation="HAVING filters after GROUP BY",
    ),
    FewShotExample(
        id="ex_complex_004",
        question="What is the average invoice total by country, only for countries with at least 10 invoices?",
        sql="""SELECT c.country, COUNT(*) as invoice_count, AVG(i.total) as avg_total
FROM invoice i
JOIN customer c ON i.customer_id = c.customer_id
GROUP BY c.country
HAVING COUNT(*) >= 10
ORDER BY avg_total DESC""",
        tables_used=["invoice", "customer"],
        category="sales",
        difficulty="hard",
    ),
    FewShotExample(
        id="ex_complex_005",
        question="List tracks that have never been sold",
        sql="""SELECT t.name FROM track t
WHERE t.track_id NOT IN (
    SELECT track_id FROM invoice_line
)""",
        tables_used=["track", "invoice_line"],
        category="sales",
        difficulty="hard",
        explanation="Using NOT IN subquery to find unsold tracks",
    ),
]


def get_examples_by_category(category: str) -> list[FewShotExample]:
    """Get all examples for a specific category."""
    return [ex for ex in CHINOOK_EXAMPLES if ex.category == category]


def get_examples_by_difficulty(difficulty: str) -> list[FewShotExample]:
    """Get all examples for a specific difficulty."""
    return [ex for ex in CHINOOK_EXAMPLES if ex.difficulty == difficulty]


def get_examples_by_tables(tables: list[str]) -> list[FewShotExample]:
    """Get examples that use any of the specified tables."""
    table_set = set(tables)
    return [ex for ex in CHINOOK_EXAMPLES if set(ex.tables_used) & table_set]


def get_example_summary() -> dict:
    """Get a summary of the example library."""
    by_category = {}
    by_difficulty = {}

    for ex in CHINOOK_EXAMPLES:
        by_category[ex.category] = by_category.get(ex.category, 0) + 1
        by_difficulty[ex.difficulty] = by_difficulty.get(ex.difficulty, 0) + 1

    return {
        "total": len(CHINOOK_EXAMPLES),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }
