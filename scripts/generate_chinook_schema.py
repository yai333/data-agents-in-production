#!/usr/bin/env python3
"""Generate Schema Cards for the Chinook database.

Three sources, each providing what it does best:
1. Database (information_schema): column names, types, nullable, PKs
2. LLM (enrich_column): descriptions and examples from column name + samples
3. Enrichments (CHINOOK_ENRICHMENTS): business rules LLM can't infer

Usage:
    # Make sure database is running
    make db-up

    # Generate schema cards
    python scripts/generate_chinook_schema.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.schema.models import TableCard, ColumnCard
from src.schema.enrichment import enrich_column, extract_sample_values
from src.adapters import create_adapter
from src.utils.config import load_config


# =============================================================================
# Manual Enrichments (Chapter 2.2: capturing business knowledge)
# =============================================================================

CHINOOK_ENRICHMENTS = {
    "invoice": {
        "description": "Sales invoices for customer purchases. Each invoice has line items.",
        "relationships": [
            {
                "name": "invoice_customer",
                "models": ["invoice", "customer"],
                "join_type": "MANY_TO_ONE",
                "condition": "invoice.customer_id = customer.customer_id",
            },
            {
                "name": "invoice_invoice_line",
                "models": ["invoice", "invoice_line"],
                "join_type": "ONE_TO_MANY",
                "condition": "invoice.invoice_id = invoice_line.invoice_id",
            },
        ],
        "metrics": [
            {
                "name": "revenue",
                "description": "Total sales revenue",
                "sql_pattern": "SUM(total)",
                "default_aggregation": "GROUP BY period",
            },
            {
                "name": "order_count",
                "description": "Number of invoices",
                "sql_pattern": "COUNT(invoice_id)",
            },
        ],
        "columns": {
            "invoice_date": {
                "business_rules": "Dates range from 2009 to 2013. Fiscal year is Jan-Dec.",
            },
            "total": {
                "business_rules": "Sum of (unit_price * quantity) from invoice_line items",
            },
        },
    },
    "track": {
        "description": "Individual songs/tracks.",
        "relationships": [
            {
                "name": "track_album",
                "models": ["track", "album"],
                "join_type": "MANY_TO_ONE",
                "condition": "track.album_id = album.album_id",
            },
            {
                "name": "track_genre",
                "models": ["track", "genre"],
                "join_type": "MANY_TO_ONE",
                "condition": "track.genre_id = genre.genre_id",
            },
            {
                "name": "track_invoice_line",
                "models": ["track", "invoice_line"],
                "join_type": "ONE_TO_MANY",
                "condition": "track.track_id = invoice_line.track_id",
            },
        ],
        "metrics": [
            {
                "name": "track_count",
                "description": "Number of tracks",
                "sql_pattern": "COUNT(track_id)",
            },
        ],
        "columns": {
            "unit_price": {
                "business_rules": "Standard price 0.99, premium/long tracks 1.99",
            },
            "milliseconds": {
                "business_rules": "5 minutes = 300,000 milliseconds",
            },
        },
    },
    "album": {
        "description": "Music albums.",
        "relationships": [
            {
                "name": "album_artist",
                "models": ["album", "artist"],
                "join_type": "MANY_TO_ONE",
                "condition": "album.artist_id = artist.artist_id",
            },
            {
                "name": "album_track",
                "models": ["album", "track"],
                "join_type": "ONE_TO_MANY",
                "condition": "album.album_id = track.album_id",
            },
        ],
    },
    "customer": {
        "description": "Customers who purchase music. Can be assigned to a support rep.",
        "relationships": [
            {
                "name": "customer_invoice",
                "models": ["customer", "invoice"],
                "join_type": "ONE_TO_MANY",
                "condition": "customer.customer_id = invoice.customer_id",
            },
            {
                "name": "customer_employee",
                "models": ["customer", "employee"],
                "join_type": "MANY_TO_ONE",
                "condition": "customer.support_rep_id = employee.employee_id",
            },
        ],
        "columns": {
            "support_rep_id": {
                "business_rules": "Internal customers have support_rep_id IS NOT NULL",
            },
        },
    },
    "artist": {
        "description": "Musical artists and bands in the catalog.",
        "relationships": [
            {
                "name": "artist_album",
                "models": ["artist", "album"],
                "join_type": "ONE_TO_MANY",
                "condition": "artist.artist_id = album.artist_id",
            },
        ],
    },
    "genre": {
        "description": "Music genre classifications (Rock, Jazz, Classical, etc.).",
        "relationships": [
            {
                "name": "genre_track",
                "models": ["genre", "track"],
                "join_type": "ONE_TO_MANY",
                "condition": "genre.genre_id = track.genre_id",
            },
        ],
    },
    "media_type": {
        "description": "Audio file formats (MPEG, AAC, etc.).",
        "relationships": [
            {
                "name": "media_type_track",
                "models": ["media_type", "track"],
                "join_type": "ONE_TO_MANY",
                "condition": "media_type.media_type_id = track.media_type_id",
            },
        ],
    },
    "invoice_line": {
        "description": "Line items for invoices. Each line is a track purchase.",
        "relationships": [
            {
                "name": "invoice_line_invoice",
                "models": ["invoice_line", "invoice"],
                "join_type": "MANY_TO_ONE",
                "condition": "invoice_line.invoice_id = invoice.invoice_id",
            },
            {
                "name": "invoice_line_track",
                "models": ["invoice_line", "track"],
                "join_type": "MANY_TO_ONE",
                "condition": "invoice_line.track_id = track.track_id",
            },
        ],
    },
    "playlist": {
        "description": "User-created playlists grouping tracks together.",
        "relationships": [
            {
                "name": "playlist_playlist_track",
                "models": ["playlist", "playlist_track"],
                "join_type": "ONE_TO_MANY",
                "condition": "playlist.playlist_id = playlist_track.playlist_id",
            },
        ],
    },
    "playlist_track": {
        "description": "Junction table linking playlists to tracks.",
        "relationships": [
            {
                "name": "playlist_track_playlist",
                "models": ["playlist_track", "playlist"],
                "join_type": "MANY_TO_ONE",
                "condition": "playlist_track.playlist_id = playlist.playlist_id",
            },
            {
                "name": "playlist_track_track",
                "models": ["playlist_track", "track"],
                "join_type": "MANY_TO_ONE",
                "condition": "playlist_track.track_id = track.track_id",
            },
        ],
    },
    "employee": {
        "description": "Company employees. Some are support reps assigned to customers.",
        "relationships": [
            {
                "name": "employee_manager",
                "models": ["employee", "employee"],
                "join_type": "MANY_TO_ONE",
                "condition": "employee.reports_to = manager.employee_id",
            },
            {
                "name": "employee_customer",
                "models": ["employee", "customer"],
                "join_type": "ONE_TO_MANY",
                "condition": "employee.employee_id = customer.support_rep_id",
            },
        ],
        "columns": {
            "reports_to": {
                "business_rules": "NULL for top-level managers",
            },
        },
    },
}

# Global glossary (database-level)
CHINOOK_GLOSSARY = {
    "churn": "Customer with no invoice in last 90 days",
    "active": "Customer with at least one invoice in last 90 days",
    "revenue": "Sum of invoice totals (calculated from invoice.total)",
}

# Global business context (database-level)
CHINOOK_ADDITIONAL_DESCRIPTIONS = [
    "Fiscal year runs January to December. Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec.",
    "'Recent' means last 30 days. 'Last month' means previous calendar month.",
    "'Top' without a number means top 10. 'Top performers' uses revenue ranking.",
    "A customer is 'active' if they have an invoice in the last 90 days.",
    "Sales regions are based on billing_country in the invoice table.",
]


async def extract_schema(conn, adapter) -> list[TableCard]:
    """Build Schema Cards from DB structure + LLM descriptions + manual rules."""
    tables = []

    table_names = await conn.fetch("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)

    for row in table_names:
        table_name = row["table_name"]
        enrichment = CHINOOK_ENRICHMENTS.get(table_name, {})
        col_enrichments = enrichment.get("columns", {})

        # Get columns with types from information_schema
        columns_data = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table_name)

        # Get primary key columns
        pk_data = await conn.fetch("""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
                AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = $1::regclass AND i.indisprimary
        """, table_name)
        pk_columns = {row["attname"] for row in pk_data}

        # Get foreign keys
        fk_data = await conn.fetch("""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.key_column_usage kcu
            JOIN information_schema.constraint_column_usage ccu
                ON kcu.constraint_name = ccu.constraint_name
            JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND kcu.table_name = $1
        """, table_name)
        fk_map = {row["column_name"]: f"{row['foreign_table']}.{row['foreign_column']}" for row in fk_data}

        # Build columns: structure from DB, descriptions from LLM, rules from enrichments
        columns = []
        for col in columns_data:
            col_name = col["column_name"]
            col_enrich = col_enrichments.get(col_name, {})

            # Format data type
            data_type = col["data_type"].upper()
            if col["character_maximum_length"]:
                data_type = f"{data_type}({col['character_maximum_length']})"

            # Extract sample values from database
            samples = await extract_sample_values(conn, table_name, col_name)

            # Generate description via LLM (uses column name + samples)
            col_card = await enrich_column(
                table_name=table_name,
                column_name=col_name,
                data_type=data_type,
                sample_values=samples,
                adapter=adapter,
            )

            # Apply schema info from database
            col_card.nullable = col["is_nullable"] == "YES"
            col_card.is_primary_key = col_name in pk_columns
            col_card.is_foreign_key = col_name in fk_map
            col_card.references = fk_map.get(col_name)

            # Add business rules from enrichments (what LLM can't infer)
            if "business_rules" in col_enrich:
                col_card.business_rules = col_enrich["business_rules"]

            columns.append(col_card)

        # Build TableCard with structured relationships and metrics
        table = TableCard(
            name=table_name,
            description=enrichment.get("description", f"Table {table_name}"),
            columns=columns,
            primary_key=list(pk_columns),
            relationships=enrichment.get("relationships", []),
            metrics=enrichment.get("metrics", []),
        )
        tables.append(table)
        print(f"  Processed: {table_name} ({len(columns)} columns)")

    return tables


async def main():
    """Generate and save schema cards."""
    print("=" * 60)
    print("Generating Chinook Schema Cards")
    print("=" * 60)
    print()

    settings = load_config()
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model}")
    print(f"Database: {settings.database.url.split('@')[-1]}")
    print()

    conn = await asyncpg.connect(settings.database.url)
    adapter = create_adapter()  # For LLM-based enrichment

    try:
        print("Extracting schema and generating descriptions...")
        tables = await extract_schema(conn, adapter)

        # Save complete semantic layer: tables + global context
        output = {
            "tables": [t.model_dump() for t in tables],
            "glossary": CHINOOK_GLOSSARY,
            "additional_descriptions": CHINOOK_ADDITIONAL_DESCRIPTIONS,
        }
        output_path = project_root / "config" / "chinook_schema.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print()
        print(f"Generated schema cards for {len(tables)} tables")
        print(f"Saved to: {output_path}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
