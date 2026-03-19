"""
backend/data_pipeline/fetch_fixtures.py
----------------------------------------
Two data sources, two modes:

  --historical   Downloads free CSVs from football-data.co.uk
                 (NOT football-data.org — different site entirely)
                 Covers 2020-21 through 2023-24. No API key needed.

  --current      Fetches the current season (2024-25) from football-data.org
                 The free tier supports the current season only — that's enough.

Why the split?
  football-data.org's free tier returns a 403 for any historical season.
  football-data.co.uk publishes free CSVs for every PL season since 1993.
  Each CSV has one row per match with date, teams, score, and bookmaker odds.

Usage:
  python -m backend.data_pipeline.fetch_fixtures --historical
  python -m backend.data_pipeline.fetch_fixtures --current
"""

import os
import sys
import io
import time
import hashlib
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Constants ────────────────────────────────────────────────────────────────

# football-data.co.uk CSV URLs — free, no auth, direct download
# Format: https://www.football-data.co.uk/mmz4281/{YYYY}/{league}.csv
# where YYYY is e.g. "2021" for the 2020-21 season, league "E0" = PL
FDCO_BASE      = "https://www.football-data.co.uk/mmz4281"
FDCO_SEASONS   = {
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
}

# football-data.org — current season only
API_KEY        = os.environ.get("FOOTBALL_DATA_API_KEY")
FDOG_BASE      = "https://api.football-data.org/v4"
CURRENT_SEASON = "2024"
CURRENT_LABEL  = "2024-25"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PLBettingBot/1.0; research)"
}

# football-data.co.uk uses full team names — map to our short_name in DB
FDCO_TEAM_MAP = {
    "Arsenal":             "Arsenal",
    "Aston Villa":         "Aston Villa",
    "Bournemouth":         "Bournemouth",
    "Brentford":           "Brentford",
    "Brighton":            "Brighton",
    "Chelsea":             "Chelsea",
    "Crystal Palace":      "Crystal Palace",
    "Everton":             "Everton",
    "Fulham":              "Fulham",
    "Ipswich":             "Ipswich",
    "Leicester":           "Leicester",
    "Liverpool":           "Liverpool",
    "Man City":            "Man City",
    "Man United":          "Man United",
    "Newcastle":           "Newcastle",
    "Nott'm Forest":       "Nott'm Forest",
    "Nottingham Forest":   "Nott'm Forest",
    "Southampton":         "Southampton",
    "Tottenham":           "Spurs",
    "West Ham":            "West Ham",
    "Wolves":              "Wolves",
    # Historical clubs
    "Leeds":               "Leeds",
    "Norwich":             "Norwich",
    "Burnley":             "Burnley",
    "Watford":             "Watford",
    "Sheffield United":    "Sheffield Utd",
    "West Brom":           "West Brom",
    "Cardiff":             "Cardiff",
    "Huddersfield":        "Huddersfield",
    "Swansea":             "Swansea",
}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def normalise_result(home_goals, away_goals) -> str | None:
    try:
        hg, ag = int(home_goals), int(away_goals)
    except (TypeError, ValueError):
        return None
    if hg > ag:
        return "HOME"
    if hg < ag:
        return "AWAY"
    return "DRAW"


def get_or_create_team(short_name: str) -> int | None:
    """
    Looks up a team by short_name. Creates it if it doesn't exist yet
    (handles historically promoted/relegated clubs cleanly).
    Returns DB team id, or None if name is blank.
    """
    if not short_name:
        return None

    existing = (
        supabase.table("teams")
        .select("id")
        .eq("short_name", short_name)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    # Not found — create a minimal record
    tla = "".join(w[0] for w in short_name.split()[:3]).upper()[:3]
    result = supabase.table("teams").insert({
        "name":       short_name + " FC",
        "short_name": short_name,
        "tla":        tla,
    }).execute()
    team_id = result.data[0]["id"]
    print(f"    → Created team: {short_name} (id={team_id})")
    return team_id


def upsert_match_rows(rows: list[dict]) -> int:
    """Batch upserts match rows. Returns count inserted."""
    if not rows:
        return 0
    batch_size = 50
    total = 0
    for i in range(0, len(rows), batch_size):
        supabase.table("matches").upsert(
            rows[i:i + batch_size],
            on_conflict="external_id"
        ).execute()
        total += len(rows[i:i + batch_size])
    return total


# ─── Historical: football-data.co.uk CSV ─────────────────────────────────────

def fetch_fdco_csv(season_label: str, season_code: str) -> pd.DataFrame | None:
    """
    Downloads the CSV for a given PL season from football-data.co.uk.
    Returns a DataFrame or None on failure.

    CSV URL example:
      https://www.football-data.co.uk/mmz4281/2021/E0.csv  (2020-21 season)
    """
    url = f"{FDCO_BASE}/{season_code}/E0.csv"
    print(f"  Downloading: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print(f"    → {len(df)} rows, columns: {list(df.columns[:8])}...")
        return df
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return None


def parse_fdco_date(date_str: str) -> str | None:
    """
    football-data.co.uk dates are in DD/MM/YY or DD/MM/YYYY format.
    Converts to ISO 8601 for Postgres.
    """
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            # Assume 15:00 UTC as default kickoff (most PL 3pm Saturday games)
            return dt.replace(hour=15, tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def process_fdco_season(df: pd.DataFrame, season_label: str) -> list[dict]:
    """
    Converts a football-data.co.uk DataFrame into match rows ready for DB.

    Key CSV columns we use:
      Date, HomeTeam, AwayTeam, FTHG (full-time home goals),
      FTAG (full-time away goals), FTR (full-time result: H/D/A)
    """
    rows = []

    for _, row in df.iterrows():
        try:
            home_name = FDCO_TEAM_MAP.get(str(row.get("HomeTeam", "")).strip())
            away_name = FDCO_TEAM_MAP.get(str(row.get("AwayTeam", "")).strip())

            if not home_name or not away_name:
                continue

            home_id = get_or_create_team(home_name)
            away_id = get_or_create_team(away_name)

            if not home_id or not away_id:
                continue

            kickoff = parse_fdco_date(row.get("Date", ""))
            if not kickoff:
                continue

            # Goals — present for all finished matches in the CSV
            home_goals = row.get("FTHG") or row.get("HG")
            away_goals = row.get("FTAG") or row.get("AG")

            # FTR: "H" = home win, "D" = draw, "A" = away win
            ftr_map = {"H": "HOME", "D": "DRAW", "A": "AWAY"}
            ftr     = str(row.get("FTR", "")).strip()
            result  = ftr_map.get(ftr) or normalise_result(home_goals, away_goals)

            # Use a synthetic external_id since FDCO has no match IDs
            # Format: season_homeId_awayId_date  (unique per match)
            raw = f"fdco-{season_label}-{home_id}-{away_id}-{kickoff[:10]}"
            external_id = int(hashlib.md5(raw.encode()).hexdigest()[:12], 16) % (10**15)

            rows.append({
                "external_id":  external_id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "kickoff_time": kickoff,
                "matchday":     None,   # FDCO CSVs don't include matchday
                "season":       season_label,
                "status":       "FINISHED",
                "home_goals":   int(home_goals) if pd.notna(home_goals) else None,
                "away_goals":   int(away_goals) if pd.notna(away_goals) else None,
                "result":       result,
            })

        except Exception as e:
            print(f"    ⚠ Skipping row: {e}")
            continue

    return rows


def run_historical():
    """Fetch 4 seasons of historical PL data from football-data.co.uk."""
    print("=" * 55)
    print("  Fetching historical PL data (football-data.co.uk)")
    print("=" * 55)

    total = 0
    for season_label, season_code in FDCO_SEASONS.items():
        print(f"\nSeason {season_label}...")
        df = fetch_fdco_csv(season_label, season_code)
        if df is None:
            print(f"  ✗ Skipping {season_label}")
            continue

        rows  = process_fdco_season(df, season_label)
        count = upsert_match_rows(rows)
        total += count
        print(f"  ✓ {season_label}: {count} matches upserted")
        time.sleep(1)   # polite delay

    print(f"\n✓ Historical import complete. Total: {total} matches")


# ─── Current season: football-data.org API ───────────────────────────────────

def run_current():
    """
    Fetch the current 2024-25 season from football-data.org.
    The free tier supports the current season — this is all we need here.
    """
    print("=" * 55)
    print("  Fetching current season (football-data.org API)")
    print("=" * 55)

    if not API_KEY:
        raise EnvironmentError("FOOTBALL_DATA_API_KEY not set in environment.")

    url     = f"{FDOG_BASE}/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    params  = {"season": CURRENT_SEASON}

    print(f"\nFetching season {CURRENT_LABEL}...")
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 429:
        print("  Rate limited — sleeping 65s...")
        time.sleep(65)
        response = requests.get(url, headers=headers, params=params, timeout=30)

    response.raise_for_status()
    matches = response.json().get("matches", [])
    print(f"  API returned {len(matches)} matches")

    rows = []
    status_map = {
        "FINISHED":  "FINISHED",
        "SCHEDULED": "SCHEDULED",
        "TIMED":     "SCHEDULED",
        "IN_PLAY":   "IN_PLAY",
        "PAUSED":    "IN_PLAY",
    }

    for m in matches:
        ht = m.get("homeTeam", {})
        at = m.get("awayTeam", {})

        home_name = FDCO_TEAM_MAP.get(ht.get("shortName", ""))
        away_name = FDCO_TEAM_MAP.get(at.get("shortName", ""))

        # Fall back to tla lookup if shortName not in map
        if not home_name:
            home_name = FDCO_TEAM_MAP.get(ht.get("name", ""))
        if not away_name:
            away_name = FDCO_TEAM_MAP.get(at.get("name", ""))

        home_id = get_or_create_team(home_name) if home_name else None
        away_id = get_or_create_team(away_name) if away_name else None

        if not home_id or not away_id:
            continue

        ft         = m.get("score", {}).get("fullTime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")

        rows.append({
            "external_id":  m["id"],
            "home_team_id": home_id,
            "away_team_id": away_id,
            "kickoff_time": m["utcDate"],
            "matchday":     m.get("matchday"),
            "season":       CURRENT_LABEL,
            "status":       status_map.get(m.get("status", ""), m.get("status", "")),
            "home_goals":   home_goals,
            "away_goals":   away_goals,
            "result":       normalise_result(home_goals, away_goals),
        })

    count = upsert_match_rows(rows)
    print(f"\n✓ Current season ({CURRENT_LABEL}): {count} matches upserted")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch PL fixtures. Historical = football-data.co.uk CSVs. "
                    "Current = football-data.org API."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--historical", action="store_true", help="Fetch 4 past seasons (no API key needed)")
    group.add_argument("--current",    action="store_true", help="Fetch current season (requires API key)")
    args = parser.parse_args()

    if args.historical:
        run_historical()
    else:
        run_current()