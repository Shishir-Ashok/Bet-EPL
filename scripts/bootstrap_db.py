"""
scripts/bootstrap_db.py
-----------------------
Run this ONCE after you've executed the SQL schema in Supabase.

What it does:
  1. Verifies your Supabase connection
  2. Seeds all 20 Premier League teams for the 2024-25 season
  3. Initialises the wallet (€10 starting balance)
  4. Prints a summary of what was inserted

Usage:
  pip install supabase python-dotenv
  python scripts/bootstrap_db.py
"""

import os
import sys
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # reads from .env in the project root

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")  # NOT the anon key

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    print("ERROR: Missing SUPABASE_URL or SUPABASE_SECRET_KEY in .env")
    sys.exit(1)


# =============================================================================
# Premier League 2024-25 teams
# tla = three-letter abbreviation used by football-data.org
# =============================================================================
PL_TEAMS = [
    {"name": "Arsenal FC",              "short_name": "Arsenal",    "tla": "ARS"},
    {"name": "Aston Villa FC",          "short_name": "Aston Villa","tla": "AVL"},
    {"name": "AFC Bournemouth",         "short_name": "Bournemouth","tla": "BOU"},
    {"name": "Brentford FC",            "short_name": "Brentford",  "tla": "BRE"},
    {"name": "Brighton & Hove Albion FC","short_name": "Brighton",  "tla": "BHA"},
    {"name": "Chelsea FC",              "short_name": "Chelsea",    "tla": "CHE"},
    {"name": "Crystal Palace FC",       "short_name": "Crystal Palace","tla": "CRY"},
    {"name": "Everton FC",              "short_name": "Everton",    "tla": "EVE"},
    {"name": "Fulham FC",               "short_name": "Fulham",     "tla": "FUL"},
    {"name": "Ipswich Town FC",         "short_name": "Ipswich",    "tla": "IPS"},
    {"name": "Leicester City FC",       "short_name": "Leicester",  "tla": "LEI"},
    {"name": "Liverpool FC",            "short_name": "Liverpool",  "tla": "LIV"},
    {"name": "Manchester City FC",      "short_name": "Man City",   "tla": "MCI"},
    {"name": "Manchester United FC",    "short_name": "Man United", "tla": "MUN"},
    {"name": "Newcastle United FC",     "short_name": "Newcastle",  "tla": "NEW"},
    {"name": "Nottingham Forest FC",    "short_name": "Nott'm Forest","tla": "NFO"},
    {"name": "Southampton FC",          "short_name": "Southampton","tla": "SOU"},
    {"name": "Tottenham Hotspur FC",    "short_name": "Spurs",      "tla": "TOT"},
    {"name": "West Ham United FC",      "short_name": "West Ham",   "tla": "WHU"},
    {"name": "Wolverhampton Wanderers FC","short_name": "Wolves",   "tla": "WOL"},
]


def main():
    print("=" * 55)
    print("  PL Betting Bot — Database Bootstrap")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Connect
    # ------------------------------------------------------------------
    print("\n[1/3] Connecting to Supabase...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        # Quick ping — list tables via a simple select
        supabase.table("teams").select("id").limit(1).execute()
        print("      ✓ Connected successfully")
    except Exception as e:
        print(f"      ✗ Connection failed: {e}")
        print("\n  Make sure you've run 001_initial_schema.sql first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Seed teams (upsert — safe to re-run)
    # ------------------------------------------------------------------
    print("\n[2/3] Seeding Premier League teams...")
    try:
        result = (
            supabase.table("teams")
            .upsert(PL_TEAMS, on_conflict="tla")  # tla is the unique key
            .execute()
        )
        print(f"      ✓ {len(PL_TEAMS)} teams upserted")
    except Exception as e:
        print(f"      ✗ Failed to seed teams: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Verify wallet exists (already seeded by SQL, but confirm)
    # ------------------------------------------------------------------
    print("\n[3/3] Checking wallet...")
    try:
        wallet = supabase.table("wallet").select("*").eq("id", 1).single().execute()
        w = wallet.data
        print(f"      ✓ Wallet found")
        print(f"        Balance:       €{w['balance']:.2f}")
        print(f"        Inception:     {w['inception_date']}")
    except Exception as e:
        print(f"      ✗ Wallet check failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Bootstrap complete. Your database is ready.")
    print("=" * 55)
    print("""
  Next steps:
  ─────────────────────────────────────────────────────
  • Phase 2: Run the data pipeline to fetch historical
    fixtures from football-data.org (2020–24 seasons)
    to train the initial XGBoost model.

  • Your Supabase dashboard:
    https://app.supabase.com → Table Editor
    You should see 20 rows in the 'teams' table.
""")


if __name__ == "__main__":
    main()
