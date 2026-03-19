"""
backend/data_pipeline/scrape_stats.py
---------------------------------------
Extracts match stats from football-data.co.uk CSVs — the same files
used for fixtures in fetch_fixtures.py.

xG proxy:
  FDCO CSVs don't include xG. We compute a shots-on-target proxy:
    xG_proxy = shots_on_target × 0.30
  This is the league-average conversion rate across PL history and is
  substantially better than a neutral fallback.

  Real xG values can be layered on top by running:
    uv run python scripts/fetch_fbref_xg.py

Stats extracted from CSV columns:
  HST / AST  → shots on target  (used as xG proxy)
  HS  / AS   → total shots
  HC  / AC   → corners
  HF  / AF   → fouls
  HY  / AY   → yellow cards
  HR  / AR   → red cards

Usage:
  python -m backend.data_pipeline.scrape_stats --all-historical
  python -m backend.data_pipeline.scrape_stats --season 2023-24
"""

import os
import sys
import io
import time
import argparse
from datetime import datetime, timedelta

import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Constants ────────────────────────────────────────────────────────────────

FDCO_BASE             = "https://www.football-data.co.uk/mmz4281"
XG_PER_SHOT_ON_TARGET = 0.30

FDCO_SEASONS = {
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
}

HISTORICAL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PLBettingBot/1.0; research)"
}

FDCO_TEAM_MAP = {
    "Arsenal":           "Arsenal",
    "Aston Villa":       "Aston Villa",
    "Bournemouth":       "Bournemouth",
    "Brentford":         "Brentford",
    "Brighton":          "Brighton",
    "Chelsea":           "Chelsea",
    "Crystal Palace":    "Crystal Palace",
    "Everton":           "Everton",
    "Fulham":            "Fulham",
    "Ipswich":           "Ipswich",
    "Leicester":         "Leicester",
    "Liverpool":         "Liverpool",
    "Man City":          "Man City",
    "Man United":        "Man United",
    "Newcastle":         "Newcastle",
    "Nott'm Forest":     "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Southampton":       "Southampton",
    "Tottenham":         "Spurs",
    "West Ham":          "West Ham",
    "Wolves":            "Wolves",
    "Leeds":             "Leeds",
    "Norwich":           "Norwich",
    "Burnley":           "Burnley",
    "Watford":           "Watford",
    "Sheffield United":  "Sheffield Utd",
    "West Brom":         "West Brom",
}


# ─── CSV download + extraction ────────────────────────────────────────────────

def fetch_stats_from_csv(season_label: str) -> dict:
    """
    Downloads the FDCO CSV and extracts per-match stats.

    Returns a lookup dict keyed by (home_short, away_short, date_str).
    """
    season_code = FDCO_SEASONS.get(season_label)
    if not season_code:
        print(f"  ✗ No season code for {season_label}")
        return {}

    url = f"{FDCO_BASE}/{season_code}/E0.csv"
    print(f"  Downloading: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return {}

    required = ["HomeTeam", "AwayTeam", "Date", "HST", "AST"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        return {}

    print(f"  → {len(df)} rows, extracting stats...")
    lookup = {}

    for _, row in df.iterrows():
        try:
            home = FDCO_TEAM_MAP.get(str(row.get("HomeTeam", "")).strip())
            away = FDCO_TEAM_MAP.get(str(row.get("AwayTeam", "")).strip())
            if not home or not away:
                continue

            date_raw = str(row.get("Date", "")).strip()
            date_str = None
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    date_str = datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            if not date_str:
                continue

            hst = _safe_int(row.get("HST"))
            ast = _safe_int(row.get("AST"))
            if hst is None or ast is None:
                continue

            lookup[(home, away, date_str)] = {
                "home_xg":              round(hst * XG_PER_SHOT_ON_TARGET, 2),
                "away_xg":              round(ast * XG_PER_SHOT_ON_TARGET, 2),
                "home_shots":           _safe_int(row.get("HS")),
                "away_shots":           _safe_int(row.get("AS")),
                "home_shots_on_target": hst,
                "away_shots_on_target": ast,
                "home_corners":         _safe_int(row.get("HC")),
                "away_corners":         _safe_int(row.get("AC")),
                "home_fouls":           _safe_int(row.get("HF")),
                "away_fouls":           _safe_int(row.get("AF")),
                "home_yellow_cards":    _safe_int(row.get("HY")),
                "away_yellow_cards":    _safe_int(row.get("AY")),
                "home_red_cards":       _safe_int(row.get("HR")),
                "away_red_cards":       _safe_int(row.get("AR")),
            }
        except (TypeError, ValueError, KeyError):
            continue

    print(f"  → {len(lookup)} rows extracted")
    return lookup


def _safe_int(val) -> int | None:
    try:
        f = float(val)
        return int(f) if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


# ─── DB helpers ───────────────────────────────────────────────────────────────

def get_finished_matches_without_stats(season: str) -> list[dict]:
    matches = (
        supabase.table("matches")
        .select(
            "id, kickoff_time, "
            "home:teams!matches_home_team_id_fkey(short_name), "
            "away:teams!matches_away_team_id_fkey(short_name)"
        )
        .eq("season", season)
        .eq("status", "FINISHED")
        .execute()
    )
    if not matches.data:
        return []

    match_ids    = [m["id"] for m in matches.data]
    existing_ids = {
        r["match_id"] for r in
        supabase.table("match_stats")
        .select("match_id")
        .in_("match_id", match_ids)
        .execute()
        .data
    }
    return [m for m in matches.data if m["id"] not in existing_ids]


def upsert_stats(match_id: int, stats: dict) -> None:
    row = {"match_id": match_id, **{k: v for k, v in stats.items() if v is not None}}
    supabase.table("match_stats").upsert(row, on_conflict="match_id").execute()


# ─── Matching + writing ───────────────────────────────────────────────────────

def match_and_write(lookup: dict, matches: list[dict]) -> tuple[int, int]:
    """
    Matches CSV rows to DB matches by team name + date (±1 day tolerance).
    Returns (updated, skipped).
    """
    updated = skipped = 0

    for match in matches:
        home  = match["home"]["short_name"]
        away  = match["away"]["short_name"]
        date  = match["kickoff_time"][:10]
        dt    = datetime.fromisoformat(date)

        stats = (
            lookup.get((home, away, date)) or
            lookup.get((home, away, (dt - timedelta(days=1)).strftime("%Y-%m-%d"))) or
            lookup.get((home, away, (dt + timedelta(days=1)).strftime("%Y-%m-%d")))
        )

        if stats:
            upsert_stats(match["id"], stats)
            updated += 1
        else:
            print(f"  ⚠ No stats: {home} vs {away} ({date})")
            skipped += 1

    return updated, skipped


# ─── Entry points ─────────────────────────────────────────────────────────────

def run(season: str):
    print("=" * 55)
    print(f"  Match stats — season: {season}")
    print("=" * 55)

    lookup  = fetch_stats_from_csv(season)
    matches = get_finished_matches_without_stats(season)

    if not lookup:
        print("  No data — exiting.")
        return

    print(f"\n  Matching to {len(matches)} DB matches...")
    updated, skipped = match_and_write(lookup, matches)
    print(f"\n✓ Done. Updated: {updated}, Skipped: {skipped}")


def run_all_historical():
    print("=" * 55)
    print("  Match stats — all historical seasons")
    print("=" * 55)

    total_updated = total_skipped = 0

    for season in HISTORICAL_SEASONS:
        print(f"\n── {season} ──────────────────────────────────")
        lookup  = fetch_stats_from_csv(season)
        matches = get_finished_matches_without_stats(season)

        if not lookup:
            print(f"  Skipping — no data")
            continue

        updated, skipped  = match_and_write(lookup, matches)
        total_updated    += updated
        total_skipped    += skipped
        print(f"  ✓ {updated} updated, {skipped} skipped")
        time.sleep(1)

    total = total_updated + total_skipped
    print(f"\n✓ All done. Updated: {total_updated}, Skipped: {total_skipped}")
    if total:
        print(f"  Skip rate: {total_skipped/total*100:.1f}%  (under 10% is normal)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract match stats from FDCO CSVs")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season",         help="Single season e.g. 2023-24")
    group.add_argument("--all-historical", action="store_true")
    args = parser.parse_args()

    if args.all_historical:
        run_all_historical()
    else:
        run(season=args.season)