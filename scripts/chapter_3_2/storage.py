"""SQLite-backed PII mapping storage."""

import sqlite3
import uuid
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "pii_mappings.db"


class PIIMappingStore:
    """SQLite-backed PII mapping storage with session isolation."""

    def __init__(self, db_path: str | Path | None = None, session_id: str | None = None):
        self.db_path = str(db_path) if db_path else str(DEFAULT_DB_PATH)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.conn = sqlite3.connect(self.db_path)
        self._init_tables()

    def _init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pii_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                placeholder TEXT NOT NULL,
                original_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, placeholder)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pii_counters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                counter INTEGER NOT NULL DEFAULT 0,
                UNIQUE(session_id, entity_type)
            )
        """)
        self.conn.commit()

    def add(self, entity_type: str, placeholder: str, real_value: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pii_mappings (session_id, entity_type, placeholder, original_value)
            VALUES (?, ?, ?, ?)
        """, (self.session_id, entity_type, placeholder, real_value))
        self.conn.commit()

    def get_original(self, placeholder: str) -> str | None:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT original_value FROM pii_mappings
            WHERE session_id = ? AND placeholder = ?
        """, (self.session_id, placeholder))
        row = cursor.fetchone()
        return row[0] if row else None

    def find_placeholder(self, entity_type: str, value: str) -> str | None:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT placeholder FROM pii_mappings
            WHERE session_id = ? AND entity_type = ? AND LOWER(original_value) = LOWER(?)
        """, (self.session_id, entity_type, value))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_next_counter(self, entity_type: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO pii_counters (session_id, entity_type, counter)
            VALUES (?, ?, 1)
            ON CONFLICT(session_id, entity_type)
            DO UPDATE SET counter = counter + 1
        """, (self.session_id, entity_type))
        cursor.execute("""
            SELECT counter FROM pii_counters
            WHERE session_id = ? AND entity_type = ?
        """, (self.session_id, entity_type))
        self.conn.commit()
        row = cursor.fetchone()
        return row[0] if row else 1

    @property
    def mappings(self) -> dict[str, dict[str, str]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT entity_type, placeholder, original_value
            FROM pii_mappings WHERE session_id = ?
            ORDER BY entity_type, placeholder
        """, (self.session_id,))

        result: dict[str, dict[str, str]] = {}
        for entity_type, placeholder, original_value in cursor.fetchall():
            if entity_type not in result:
                result[entity_type] = {}
            result[entity_type][placeholder] = original_value
        return result

    def cleanup(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM pii_mappings WHERE session_id = ?", (self.session_id,))
        cursor.execute("DELETE FROM pii_counters WHERE session_id = ?", (self.session_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
