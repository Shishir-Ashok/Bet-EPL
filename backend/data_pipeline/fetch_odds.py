"""
backend/data_pipeline/fetch_odds.py
-------------------------------------
Fetches pre-match odds for upcoming PL fixtures from The Odds API
and upserts them into the `odds` table.

Why The Odds API?
  It aggregates odds from 40+ bookmakers in one clean JSON response.
  The free tier gives 500 requests/month. Since we're only fetching
  odds for games within the next 7 days (up to ~10 games/matchday),
  we use roughly 2-4 requests per matchday × 38 matchdays ≈ 150/season.
  Well within the free tier.

Key design decisions:
  - We fetch odds for the NEXT 7 DAYS only (not all upcoming fixtures)
    to conserve the monthly request quota.
  - We store every bookmaker's line, not just an average. This lets us
    find the best available odds when placing a virtual bet.
  - Odds are fetched as DECIMAL (e.g. 2.10) not fractional (11/10).

Usage:
  python -m backend.data_pipeline.fetch_odds
"""

import os
import sys
import requests
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.data_pipeline.odds_validator import validate_odds_row

# ─── Constants ────────────────────────────────────────────────────────────────

API_KEY     = os.environ.get("THE_ODDS_API_KEY")
BASE_URL    = "https://api.the-odds-api.com/v4"
SPORT       = "soccer_epl"
REGIONS     = "eu"
MARKETS     = "h2h"
ODDS_FORMAT = "decimal"

# Bookmaker priority: Pinnacle is the sharpest market and our primary source.
# Marathonbet is the fallback if Pinnacle doesn't return odds for a fixture.
PRIMARY_BOOKMAKER  = "pinnacle"
FALLBACK_BOOKMAKER = "marathonbet"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_headers() -> dict:
    return {}  # The Odds API uses query params, not headers


def get_params() -> dict:
    if not API_KEY:
        raise EnvironmentError("THE_ODDS_API_KEY not set in environment.")
    return {
        "apiKey":      API_KEY,
        "regions":     REGIONS,
        "markets":     MARKETS,
        "oddsFormat":  ODDS_FORMAT,
    }


def match_event_to_db_match(event: dict) -> int | None:
    """
    The Odds API uses its own event IDs and team names that don't
    match football-data.org. We match events to our DB by comparing
    kickoff time (±60 minutes) and fuzzy team name matching.

    Returns the internal match ID if found, None otherwise.
    """
    event_time = datetime.fromisoformat(
        event["commence_time"].replace("Z", "+00:00")
    )

    # Window: kickoff ± 90 minutes (handles slight time discrepancies)
    window_start = (event_time - timedelta(minutes=90)).isoformat()
    window_end   = (event_time + timedelta(minutes=90)).isoformat()

    candidates = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, teams!matches_home_team_id_fkey(name, short_name, tla), away:teams!matches_away_team_id_fkey(name, short_name, tla)")
        .gte("kickoff_time", window_start)
        .lte("kickoff_time", window_end)
        .eq("status", "SCHEDULED")
        .execute()
    )

    if not candidates.data:
        return None

    # The Odds API home_team / away_team are strings like "Arsenal"
    api_home = event.get("home_team", "").lower()
    api_away = event.get("away_team", "").lower()

    for match in candidates.data:
        db_home_names = [
            match["teams"]["name"].lower(),
            match["teams"]["short_name"].lower(),
            match["teams"]["tla"].lower(),
        ]
        db_away_names = [
            match["away"]["name"].lower(),
            match["away"]["short_name"].lower(),
            match["away"]["tla"].lower(),
        ]

        # Check if any known name for the team is a substring of the API name
        home_match = any(n in api_home or api_home in n for n in db_home_names)
        away_match = any(n in api_away or api_away in n for n in db_away_names)

        if home_match and away_match:
            return match["id"]

    return None


def extract_h2h_odds(
    bookmaker:  dict,
    home_team:  str,   # event["home_team"] — the actual home side
    away_team:  str,   # event["away_team"]
) -> tuple[float, float, float] | None:
    """
    Pulls home/draw/away odds matched by team name, not by list position.

    The Odds API does not guarantee outcome ordering. We match each
    outcome name against the known home/away team strings using the
    same fuzzy logic as match_event_to_db_match().

    Returns (home_odds, draw_odds, away_odds) or None.
    """
    for market in bookmaker.get("markets", []):
        if market.get("key") != "h2h":
            continue

        outcomes = market.get("outcomes", [])
        if len(outcomes) != 3:
            return None

        draw_odds = None
        home_odds = None
        away_odds = None

        home_lower = home_team.lower()
        away_lower = away_team.lower()

        for o in outcomes:
            name  = o["name"].lower()
            price = o["price"]

            if name == "draw":
                draw_odds = price
            elif name in home_lower or home_lower in name:
                home_odds = price
            elif name in away_lower or away_lower in name:
                away_odds = price

        if home_odds and draw_odds and away_odds:
            return home_odds, draw_odds, away_odds

    return None


# ─── Core fetch logic ─────────────────────────────────────────────────────────

def fetch_upcoming_odds() -> list[dict]:
    """Fetches all upcoming EPL events from The Odds API."""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = get_params()

    print("Fetching odds from The Odds API...")
    response = requests.get(url, params=params, timeout=30)

    # Log remaining quota so we can monitor free tier usage
    remaining = response.headers.get("x-requests-remaining", "?")
    used       = response.headers.get("x-requests-used", "?")
    print(f"  API quota: {used} used, {remaining} remaining this month")

    response.raise_for_status()
    return response.json()


ODDS_CHANGE_THRESHOLD = 0.02  # only update DB if any line moved by more than this

def upsert_odds(events: list[dict]) -> tuple[int, int]:
    """
    For each event:
      1. Match to a DB match record
      2. Extract Pinnacle (primary) or Marathonbet (fallback) odds
      3. Check if existing odds already match — skip DB write if unchanged
      4. Upsert (single row per match) if new or changed

    Returns (events_matched, odds_rows_upserted).
    """
    events_matched = 0
    odds_upserted  = 0

    # Fetch all existing odds rows for upcoming matches in one query
    # so we can compare without per-match DB calls
    existing_odds: dict[int, dict] = {}
    try:
        rows = (
            supabase.table("odds")
            .select("match_id, home_odds, draw_odds, away_odds")
            .execute()
            .data
        )
        existing_odds = {r["match_id"]: r for r in rows}
    except Exception as e:
        print(f"  ⚠ Could not prefetch existing odds: {e}")

    for event in events:
        match_id = match_event_to_db_match(event)
        if not match_id:
            print(f"  ⚠ No DB match for: {event.get('home_team')} vs "
                  f"{event.get('away_team')} ({event.get('commence_time')})")
            continue

        events_matched += 1
        bookmakers      = {bk["key"]: bk for bk in event.get("bookmakers", [])}
        selected_odds   = None
        selected_name   = None

        for bk_key in (PRIMARY_BOOKMAKER, FALLBACK_BOOKMAKER):
            bk = bookmakers.get(bk_key)
            if bk:
                h2h = extract_h2h_odds(bk, event["home_team"], event["away_team"])
                if h2h:
                    selected_odds = h2h
                    selected_name = bk_key
                    break

        if not selected_odds:
            print(f"  ⚠ No usable odds for: {event.get('home_team')} vs "
                  f"{event.get('away_team')}")
            continue

        new_home, new_draw, new_away = selected_odds

        # Compare against stored odds — skip write if nothing meaningful changed
        existing = existing_odds.get(match_id)
        if existing:
            moved = (
                abs(float(existing["home_odds"]) - new_home) > ODDS_CHANGE_THRESHOLD or
                abs(float(existing["draw_odds"]) - new_draw) > ODDS_CHANGE_THRESHOLD or
                abs(float(existing["away_odds"]) - new_away) > ODDS_CHANGE_THRESHOLD
            )
            if not moved:
                print(f"  – {event['home_team']} vs {event['away_team']}: "
                      f"odds unchanged, skipping")
                continue
        
        match = (
            supabase.table("matches")
            .select("home_team_id, away_team_id")
            .eq("id", match_id)
            .single()
            .execute()
            .data
        )
        if not match:
            continue

        validated = validate_odds_row(
            home_team_id = match["home_team_id"],
            away_team_id = match["away_team_id"],
            home_odds    = new_home,
            draw_odds    = new_draw,
            away_odds    = new_away,
        )
        if validated is None:
            print(f"  ✗ Odds rejected for match {match_id} — implausible values")
            continue

        supabase.table("odds").upsert(
            {
                "match_id":   match_id,
                "home_odds":  validated["home_odds"],
                "draw_odds":  validated["draw_odds"],
                "away_odds":  validated["away_odds"],
                "bookmaker":  selected_name,
            }, 
            on_conflict="match_id"
        ).execute()

        odds_upserted += 1
        status = "updated" if existing else "inserted"
        print(f"  ✓ {event['home_team']} vs {event['away_team']}: "
              f"{selected_name} {new_home}/{new_draw}/{new_away} ({status})")

    return events_matched, odds_upserted


def get_best_odds(match_id: int) -> dict | None:
    """
    Returns the stored odds for a match.
    One row per match (Pinnacle, or Marathonbet fallback).
    """
    result = (
        supabase.table("odds")
        .select("bookmaker, home_odds, draw_odds, away_odds")
        .eq("match_id", match_id)
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    row = result.data[0]
    return {
        "home":       row["home_odds"],
        "draw":       row["draw_odds"],
        "away":       row["away_odds"],
        "bookmaker":  row["bookmaker"],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Fetching pre-match PL odds (Pinnacle / Marathonbet)")
    print("=" * 55)

    events = fetch_upcoming_odds()
    matched, upserted = upsert_odds(events)

    print(f"\n✓ Done. {matched}/{len(events)} events matched, {upserted} odds rows upserted.")