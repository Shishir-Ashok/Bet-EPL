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

# ─── Constants ────────────────────────────────────────────────────────────────

API_KEY     = os.environ.get("THE_ODDS_API_KEY")
BASE_URL    = "https://api.the-odds-api.com/v4"
SPORT       = "soccer_epl"
REGIONS     = "eu"         # European decimal odds
MARKETS     = "h2h"        # Head-to-head = 1X2 (home/draw/away)
ODDS_FORMAT = "decimal"

# Bookmakers we care about — prioritise well-known ones for realistic odds
PREFERRED_BOOKMAKERS = {
    "bet365", "betfair", "williamhill", "paddypower",
    "unibet", "skybet", "ladbrokes", "coral"
}


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


def extract_h2h_odds(bookmaker: dict) -> tuple[float, float, float] | None:
    """
    Pulls home/draw/away decimal odds from a bookmaker's h2h market.
    Returns None if the market data is malformed.
    """
    for market in bookmaker.get("markets", []):
        if market.get("key") != "h2h":
            continue
        outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
        if len(outcomes) == 3:
            # h2h has: home team name, away team name, "Draw"
            prices = list(outcomes.values())
            # Identify which is Draw, which is home, which is away
            draw_price = outcomes.get("Draw")
            if not draw_price:
                return None
            team_prices = [v for k, v in outcomes.items() if k != "Draw"]
            if len(team_prices) != 2:
                return None
            # The API orders outcomes: home team first, away team second
            outcome_list = [o for o in market["outcomes"] if o["name"] != "Draw"]
            if len(outcome_list) != 2:
                return None
            home_price = outcome_list[0]["price"]
            away_price = outcome_list[1]["price"]
            return home_price, draw_price, away_price
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


def upsert_odds(events: list[dict]) -> tuple[int, int]:
    """
    For each event, find the matching DB match and insert odds rows
    for every bookmaker.

    Returns (events_matched, odds_rows_inserted).
    """
    events_matched   = 0
    odds_rows_total  = 0

    for event in events:
        match_id = match_event_to_db_match(event)
        if not match_id:
            print(f"  ⚠ No DB match found for: {event.get('home_team')} vs {event.get('away_team')} ({event.get('commence_time')})")
            continue

        events_matched += 1
        odds_rows = []

        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker.get("key", "unknown")
            h2h = extract_h2h_odds(bookmaker)
            if not h2h:
                continue
            home_odds, draw_odds, away_odds = h2h

            odds_rows.append({
                "match_id":   match_id,
                "bookmaker":  bk_name,
                "home_odds":  home_odds,
                "draw_odds":  draw_odds,
                "away_odds":  away_odds,
            })

        if odds_rows:
            supabase.table("odds").insert(odds_rows).execute()
            odds_rows_total += len(odds_rows)
            print(f"  ✓ {event['home_team']} vs {event['away_team']}: {len(odds_rows)} bookmakers")

    return events_matched, odds_rows_total


def get_best_odds(match_id: int) -> dict | None:
    """
    Utility: returns the best available decimal odds across all bookmakers
    for a given match. Used by the bet engine to maximise expected value.

    Best = highest odds for the predicted outcome (most favourable payout).
    """
    result = (
        supabase.table("odds")
        .select("bookmaker, home_odds, draw_odds, away_odds, fetched_at")
        .eq("match_id", match_id)
        .order("fetched_at", desc=True)
        .limit(20)  # last 20 snapshots across bookmakers
        .execute()
    )

    if not result.data:
        return None

    # Find the max odds for each outcome
    best = {
        "home": max(r["home_odds"] for r in result.data),
        "draw": max(r["draw_odds"] for r in result.data),
        "away": max(r["away_odds"] for r in result.data),
    }
    return best


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Fetching pre-match PL odds")
    print("=" * 55)

    events             = fetch_upcoming_odds()
    matched, inserted  = upsert_odds(events)

    print(f"\n✓ Done. {matched}/{len(events)} events matched, {inserted} odds rows inserted.")
