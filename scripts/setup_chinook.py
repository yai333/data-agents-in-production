#!/usr/bin/env python3
"""Download and prepare the Chinook database for PostgreSQL.

This script downloads the Chinook SQL file and places it in the
data/chinook directory for Docker to load on first startup.

Usage:
    python scripts/setup_chinook.py
"""

import urllib.request
from pathlib import Path


CHINOOK_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "chinook"
OUTPUT_FILE = OUTPUT_DIR / "01_chinook.sql"


def main() -> None:
    """Download Chinook database SQL file."""
    print(f"Downloading Chinook database from {CHINOOK_URL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"File already exists: {OUTPUT_FILE}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Skipping download.")
            return

    urllib.request.urlretrieve(CHINOOK_URL, OUTPUT_FILE)
    print(f"Downloaded to: {OUTPUT_FILE}")

    print("\nNext steps:")
    print("  1. Run: docker compose up -d postgres")
    print("  2. Wait for database to initialize")
    print("  3. Test: docker compose exec postgres psql -U postgres -d chinook -c '\\dt'")


if __name__ == "__main__":
    main()
