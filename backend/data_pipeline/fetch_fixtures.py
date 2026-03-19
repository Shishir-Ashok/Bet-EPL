"""
backend/data_pipeline/fetch_fixtures.py
----------------------------------------
Fetches Premier League fixtures and results from football-data.org
and upserts them into the `matches` table.

Two modes:
  --historical   Fetches 2020-21 through 2023-24 (training data for XGBoost)
  --current      Fetches the current season (2024-25) — used by the daily cron

Why football-data.org?
  Free tier covers the full PL with no rate-limit issues for our volume.
  The API returns structured JSON — no scraping needed for fixtures/results.

Rate limit: 10 calls/minute on free tier. We sleep between requests.

Usage:
  python -m backend.data_pipeline.fetch_fixtures --historical
  python -m backend.data_pipeline.fetch_fixtures --current
"""

import os
import sys
import time
import argparse
import requests
from datetime import datetime, timezone

# Allow running as a module from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Constants ────────────────────────────────────────────────────────────────

API_KEY         = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL        = "https://api.football-data.org/v4"
PL_CODE         = "PL"

# Seasons to fetch for historical training data.
# football-data.org season codes use the start year.
HISTORICAL_SEASONS = ["2020", "2021", "2022", "2023"]
CURRENT_SEASON     = "2025"   # 2025-26 season — update each August when the new season starts

# Status codes returned by the API
FINISHED_STATUSES  = {"FINISHED"}
SCHEDULED_STATUSES = {"SCHEDULED", "TIMED"}
IN_PLAY_STATUSES   = {"IN_PLAY", "PAUSED", "SUSPENDED"}

# How long to sleep between API calls (free tier = 10/min = 6s minimum)
RATE_LIMIT_SLEEP = 7  # slightly above 6s to be safe


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_headers() -> dict:
    if not API_KEY:
        raise EnvironmentError("FOOTBALL_DATA_API_KEY not set in environment.")
    return {"X-Auth-Token": API_KEY}


def get_or_create_team(team_data: dict) -> int:
    """
    Looks up a team by their football-data.org TLA.
    If it doesn't exist yet (e.g. a promoted club), inserts it.
    Returns the internal DB team id.

    Why upsert here instead of pre-seeding?
    The bootstrap script seeds the current 20 PL teams, but historical
    seasons include relegated clubs (e.g. Norwich, Burnley) that aren't
    in the current squad. Rather than pre-seeding all of them, we create
    them on the fly as we encounter them.
    """
    tla = team_data.get("tla") or team_data.get("shortName", "UNK")[:3].upper()

    existing = (
        supabase.table("teams")
        .select("id")
        .eq("tla", tla)
        .execute()
    )

    if existing.data:
        return existing.data[0]["id"]

    # Team not found — insert it
    new_team = {
        "name":       team_data.get("name", tla),
        "short_name": team_data.get("shortName", tla),
        "tla":        tla,
        "crest_url":  team_data.get("crest"),
    }
    result = supabase.table("teams").insert(new_team).execute()
    team_id = result.data[0]["id"]
    print(f"  → Created new team: {new_team['name']} (id={team_id})")
    return team_id


def normalise_result(home_goals: int, away_goals: int) -> str | None:
    """Converts scoreline to our HOME/DRAW/AWAY label."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "HOME"
    if home_goals < away_goals:
        return "AWAY"
    return "DRAW"


def normalise_status(api_status: str) -> str:
    """Maps football-data.org status codes to our simplified set."""
    if api_status in FINISHED_STATUSES:
        return "FINISHED"
    if api_status in SCHEDULED_STATUSES:
        return "SCHEDULED"
    if api_status in IN_PLAY_STATUSES:
        return "IN_PLAY"
    return api_status  # POSTPONED, CANCELLED — pass through


# ─── Core fetch logic ─────────────────────────────────────────────────────────

def fetch_season(season_code: str) -> list[dict]:
    """
    Fetches all matches for a given PL season from the API.

    football-data.org season parameter uses the start year:
      "2024" → 2024-25 season
      "2023" → 2023-24 season
    """
    url = f"{BASE_URL}/competitions/{PL_CODE}/matches"
    params = {"season": season_code}

    print(f"\nFetching {PL_CODE} season {season_code}...")
    response = requests.get(url, headers=get_headers(), params=params, timeout=30)

    if response.status_code == 429:
        print("  Rate limited — sleeping 60s...")
        time.sleep(60)
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)

    response.raise_for_status()
    data = response.json()

    matches = data.get("matches", [])
    print(f"  API returned {len(matches)} matches for season {season_code}")
    return matches


def upsert_matches(matches: list[dict], season_label: str) -> int:
    """
    Transforms raw API match objects and upserts them into the DB.
    Returns the number of records upserted.

    We use upsert (on_conflict="external_id") so this is safe to re-run
    — it updates scores/status for matches that have since finished.
    """
    rows = []

    for m in matches:
        # Skip matches without both teams resolved (very rare edge case)
        if not m.get("homeTeam", {}).get("id") or not m.get("awayTeam", {}).get("id"):
            continue

        home_id = get_or_create_team(m["homeTeam"])
        away_id = get_or_create_team(m["awayTeam"])

        # Score — only present when status is FINISHED
        score      = m.get("score", {})
        full_time  = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        rows.append({
            "external_id":   m["id"],
            "home_team_id":  home_id,
            "away_team_id":  away_id,
            "kickoff_time":  m["utcDate"],
            "matchday":      m.get("matchday"),
            "season":        season_label,
            "status":        normalise_status(m["status"]),
            "home_goals":    home_goals,
            "away_goals":    away_goals,
            "result":        normalise_result(home_goals, away_goals),
        })

    if not rows:
        print("  No rows to upsert.")
        return 0

    # Upsert in batches of 50 to avoid hitting Supabase's request size limit
    batch_size = 50
    total_upserted = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        supabase.table("matches").upsert(
            batch,
            on_conflict="external_id"
        ).execute()
        total_upserted += len(batch)

    return total_upserted


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_historical():
    """Fetch 4 complete past seasons for training data."""
    print("=" * 55)
    print("  Fetching historical PL fixtures (training data)")
    print("=" * 55)

    total = 0
    for season in HISTORICAL_SEASONS:
        matches = fetch_season(season)

        # Season label stored in DB: "2020-21", "2021-22" etc.
        label   = f"{season}-{str(int(season)+1)[2:]}"
        count   = upsert_matches(matches, label)
        total  += count
        print(f"  ✓ Season {label}: {count} matches upserted")

        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\n✓ Historical fetch complete. Total matches upserted: {total}")


def run_current():
    """Fetch the current season — used by the daily cron job."""
    print("=" * 55)
    print("  Fetching current PL season fixtures")
    print("=" * 55)

    season  = CURRENT_SEASON
    label   = f"{season}-{str(int(season)+1)[2:]}"
    matches = fetch_season(season)
    count   = upsert_matches(matches, label)
    print(f"\n✓ Current season ({label}): {count} matches upserted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PL fixtures from football-data.org")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--historical", action="store_true", help="Fetch 4 past seasons")
    group.add_argument("--current",    action="store_true", help="Fetch current season")
    args = parser.parse_args()

    if args.historical:
        run_historical()
    else:
        run_current()