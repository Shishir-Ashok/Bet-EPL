"""
scripts/seed_historical_data.py
---------------------------------
Run this ONCE before training the initial model.

This is the "cold start" script. It orchestrates the full historical
data collection pipeline in the correct order:

  1. Fetch 4 seasons of fixtures (2020-21 through 2023-24)
  2. Scrape xG for all 4 seasons from Understat
  3. Calculate ELO ratings for all historical matches

After this script completes successfully, your DB will have:
  ~1,520 historical matches with results
  ~1,400 xG records (Understat coverage is ~92% of matches)
  ~3,040 ELO rating records

That gives the XGBoost model enough data for a meaningful train/val split
and gives the DQN agent enough history to bootstrap its replay buffer.

Runtime: ~12-15 minutes (mostly waiting for Understat scraping delays)

Usage:
  python scripts/seed_historical_data.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_pipeline.fetch_fixtures import run_historical
from backend.data_pipeline.scrape_stats   import run as run_stats
from backend.data_pipeline.update_elo     import run as run_elo

HISTORICAL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]


def main():
    print("=" * 55)
    print("  PL Betting Bot — Historical Data Seeding")
    print("  This will take ~12-15 minutes. Don't interrupt it.")
    print("=" * 55)

    # Step 1: Fixtures
    print("\n\n──── STEP 1/3: Fetch fixtures ────────────────────────")
    run_historical()

    # Brief pause between API sources
    time.sleep(5)

    # Step 2: xG data per season
    print("\n\n──── STEP 2/3: Scrape xG from Understat ─────────────")
    for season in HISTORICAL_SEASONS:
        print(f"\n  Season: {season}")
        run_stats(season=season)
        time.sleep(3)   # polite delay between seasons

    # Step 3: ELO ratings
    print("\n\n──── STEP 3/3: Calculate ELO ratings ────────────────")
    run_elo(all_seasons=True)   # recalculate from scratch across all seasons

    print("\n\n" + "=" * 55)
    print("  Historical seeding complete!")
    print("=" * 55)
    print("""
  Your database now has 4 seasons of training data.

  Next steps:
  ─────────────────────────────────────────────────────
  • Phase 3: Train the initial XGBoost outcome predictor
      python -m backend.model.train --mode xgboost

  • Check data quality in Supabase Table Editor:
      matches      → ~1,520 rows, all with results
      match_stats  → ~1,400 rows with home_xg / away_xg
      elo_ratings  → ~3,040 rows (2 per match)
""")


if __name__ == "__main__":
    main()
