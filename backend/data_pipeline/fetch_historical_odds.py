"""
backend/data_pipeline/fetch_historical_odds.py
-------------------------------------------------
Fetches real pre-match odds from football-data.co.uk (100% free,
no API key, no rate limits).

Why opening odds, not closing odds
------------------------------------
Closing Pinnacle odds (PSCH/PSCD/PSCA) reflect everything the market
knew right before kickoff — including late team news and sharp money.
You can never actually bet at closing odds.

Opening Pinnacle odds (PSH/PSD/PSA) are collected by football-data.co.uk
on Friday afternoons for weekend games and Tuesday afternoons for midweek.
These are the realistic prices available when a pre-match betting decision
is made — which is what this system simulates.

Using closing odds as the price you bet at is lookahead bias. It makes
the backtest look better than reality and teaches Kelly to expect prices
you will never receive on future games.

Fallback order (opening lines only — closing deliberately excluded):
  1. PSH/PSD/PSA     Pinnacle opening    ← sharpest available opening line
  2. B365H/B365D/B365A Bet365 opening
  3. MaxH/MaxD/MaxA  Market maximum opening
  4. AvgH/AvgD/AvgA  Market average opening

Source
------
https://www.football-data.co.uk
URL: /mmz4281/{YYYY}/E0.csv  e.g. 2024-25 → /mmz4281/2425/E0.csv

Usage
-----
  python -m backend.data_pipeline.fetch_historical_odds
  python -m backend.data_pipeline.fetch_historical_odds --seasons 2024-25 2025-26
  python -m backend.data_pipeline.fetch_historical_odds --dry-run
"""

import os
import sys
import io
import csv
import time
import argparse
import requests
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.data_pipeline.odds_validator import validate_odds_row

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL        = "https://www.football-data.co.uk/mmz4281"
DEFAULT_SEASONS = ["2024-25", "2025-26"]

# Opening odds only — closing columns (PSCH/PSCD/PSCA) are intentionally absent
ODDS_PRIORITY = [
    ("PSH",   "PSD",   "PSA",   "pinnacle_opening"),
    ("B365H", "B365D", "B365A", "bet365"),
    ("MaxH",  "MaxD",  "MaxA",  "market_max"),
    ("AvgH",  "AvgD",  "AvgA",  "market_avg"),
]

# football-data.co.uk abbreviated names → normalised key
FDC_NAME_MAP = {
    "Man United":     "manchester united",
    "Manchester Utd": "manchester united",
    "Man City":       "manchester city",
    "Tottenham":      "tottenham",
    "Spurs":          "tottenham",
    "Newcastle":      "newcastle",
    "Newcastle Utd":  "newcastle",
    "Nottm Forest":   "nottingham forest",
    "Nott'm Forest":  "nottingham forest",
    "Sheffield Utd":  "sheffield united",
    "Wolves":         "wolverhampton",
    "West Brom":      "west bromwich",
    "QPR":            "queens park rangers",
    "Aston Villa":    "aston villa",
    "Crystal Palace": "crystal palace",
    "West Ham":       "west ham",
    "Luton":          "luton",
    "Ipswich":        "ipswich",
    "Leicester":      "leicester",
    "Brighton":       "brighton",
    "Bournemouth":    "bournemouth",
    "Brentford":      "brentford",
    "Fulham":         "fulham",
    "Arsenal":        "arsenal",
    "Chelsea":        "chelsea",
    "Liverpool":      "liverpool",
    "Everton":        "everton",
    "Burnley":        "burnley",
    "Sunderland":     "sunderland",
    "Southampton":    "southampton",
    "Middlesbrough":  "middlesbrough",
    "Stoke":          "stoke",
    "Watford":        "watford",
    "Huddersfield":   "huddersfield",
    "Cardiff":        "cardiff",
    "Swansea":        "swansea",
    "Hull":           "hull",
    "Norwich":        "norwich",
    "Leeds":          "leeds",
}


def _season_to_url_segment(label: str) -> str:
    """'2024-25' → '2425'"""
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected season label: {label!r}")
    return parts[0][-2:] + parts[1][-2:]


# ─── Normalisation ────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalise a team name for fuzzy matching."""
    if not name:
        return ""
    n = name.lower().strip()
    for k, v in FDC_NAME_MAP.items():
        if k.lower() == n:
            return v
    for suffix in [" fc", " afc", " city", " united", " hotspur",
                   " wanderers", " rovers", " athletic", " albion",
                   " town", " county", " palace"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    return n


# ─── CSV fetching ─────────────────────────────────────────────────────────────

def _fetch_csv(season_label: str) -> list[dict] | None:
    seg = _season_to_url_segment(season_label)
    url = f"{BASE_URL}/{seg}/E0.csv"
    print(f"  Fetching: {url}")
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 404:
            print(f"    ✗ 404 — {season_label} not yet published on football-data.co.uk")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ✗ Download failed: {e}")
        return None
    content = resp.text.lstrip("\ufeff")
    rows    = [r for r in csv.DictReader(io.StringIO(content)) if r.get("HomeTeam", "").strip()]
    print(f"    {len(rows)} rows parsed")
    return rows


def _parse_date(s: str) -> date | None:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _extract_odds(row: dict) -> tuple[float, float, float, str] | None:
    """Returns (home, draw, away, bookmaker) using the first valid opening column set."""
    for hc, dc, ac, label in ODDS_PRIORITY:
        try:
            h = float(row.get(hc, "") or 0)
            d = float(row.get(dc, "") or 0)
            a = float(row.get(ac, "") or 0)
            if h > 1.01 and d > 1.01 and a > 1.01:
                return h, d, a, label
        except (ValueError, TypeError):
            continue
    return None


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _load_db_matches(season_label: str) -> dict[date, list[dict]]:
    matches, offset, page_size = [], 0, 500
    while True:
        page = (
            supabase.table("matches")
            .select(
                "id, home_team_id, away_team_id, kickoff_time, "
                "home:teams!matches_home_team_id_fkey(name), "
                "away:teams!matches_away_team_id_fkey(name)"
            )
            .eq("season", season_label)
            .not_.is_("result", "null")
            .order("kickoff_time")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if not page:
            break
        matches.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.05)

    by_date: dict[date, list[dict]] = defaultdict(list)
    for m in matches:
        ko = datetime.fromisoformat(m["kickoff_time"].replace("Z", "+00:00"))
        by_date[ko.date()].append(m)
    return by_date


def _existing_odds_ids(season_label: str) -> set[int]:
    rows = supabase.table("matches").select("id").eq("season", season_label).execute().data
    if not rows:
        return set()
    all_ids  = [r["id"] for r in rows]
    existing: set[int] = set()
    for i in range(0, len(all_ids), 200):
        chunk = all_ids[i : i + 200]
        odds  = supabase.table("odds").select("match_id").in_("match_id", chunk).execute().data
        existing.update(r["match_id"] for r in odds)
    return existing


def _find_db_match(
    csv_home: str,
    csv_away: str,
    match_date: date,
    db_by_date: dict[date, list[dict]],
) -> dict | None:
    ch = _norm(csv_home)
    ca = _norm(csv_away)
    for delta in (0, 1, -1):
        for m in db_by_date.get(match_date + timedelta(days=delta), []):
            dh = _norm(m["home"]["name"])
            da = _norm(m["away"]["name"])
            if (ch in dh or dh in ch) and (ca in da or da in ca):
                return m
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(seasons: list[str] = None, dry_run: bool = False) -> dict:
    seasons = seasons or DEFAULT_SEASONS
    print("=" * 65)
    print("  Odds fetcher — football-data.co.uk (opening odds, no API key)")
    print(f"  Seasons: {', '.join(seasons)}")
    print(f"  Mode:    {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 65)

    total = {"matched": 0, "missed": 0, "skipped": 0, "swapped": 0}

    for season_label in seasons:
        print(f"\n── {season_label} ─────────────────────────────────────")

        csv_rows = _fetch_csv(season_label)
        if not csv_rows:
            continue

        db_by_date   = _load_db_matches(season_label)
        existing_ids = _existing_odds_ids(season_label)
        db_total     = sum(len(v) for v in db_by_date.values())
        print(f"  DB: {db_total} matches  |  {len(existing_ids)} already have odds")

        s = {"matched": 0, "missed": 0, "skipped": 0, "swapped": 0}
        unmatched_names: set[str] = set()

        for row in csv_rows:
            home     = row.get("HomeTeam", "").strip()
            away     = row.get("AwayTeam", "").strip()
            date_str = row.get("Date", "").strip()
            if not home or not away or not date_str:
                continue

            ko_date = _parse_date(date_str)
            if not ko_date:
                continue

            db_match = _find_db_match(home, away, ko_date, db_by_date)
            if not db_match:
                s["missed"] += 1
                unmatched_names.add(f"{home} vs {away}")
                continue

            if db_match["id"] in existing_ids:
                s["skipped"] += 1
                continue

            odds_result = _extract_odds(row)
            if not odds_result:
                s["missed"] += 1
                continue

            raw_h, raw_d, raw_a, bookmaker = odds_result

            validated = validate_odds_row(
                home_team_id = db_match["home_team_id"],
                away_team_id = db_match["away_team_id"],
                home_odds    = raw_h,
                draw_odds    = raw_d,
                away_odds    = raw_a,
                match_id     = db_match["id"],
            )
            if not validated:
                s["missed"] += 1
                continue

            if validated["swapped"]:
                s["swapped"] += 1

            if not dry_run:
                supabase.table("odds").upsert({
                    "match_id":  db_match["id"],
                    "home_odds": validated["home_odds"],
                    "draw_odds": validated["draw_odds"],
                    "away_odds": validated["away_odds"],
                    "bookmaker": bookmaker,
                }, on_conflict="match_id").execute()
                existing_ids.add(db_match["id"])

            swap_flag = " [SWAPPED]" if validated["swapped"] else ""
            print(f"  ✓ {home} vs {away} ({date_str})  "
                  f"H:{validated['home_odds']}  D:{validated['draw_odds']}  "
                  f"A:{validated['away_odds']}  [{bookmaker}]{swap_flag}")
            s["matched"] += 1

        print(f"\n  {season_label} summary: "
              f"matched={s['matched']}  missed={s['missed']}  "
              f"skipped={s['skipped']}  swapped={s['swapped']}")

        if unmatched_names:
            print(f"  Unmatched team names (add to FDC_NAME_MAP if miss rate is high):")
            for name in sorted(unmatched_names)[:10]:
                print(f"    {name}")

        for k in total:
            total[k] += s[k]

    print(f"\n{'='*65}")
    print(f"  Total matched:  {total['matched']}")
    print(f"  Total missed:   {total['missed']}")
    print(f"  Total skipped:  {total['skipped']}")
    print(f"  Total swapped:  {total['swapped']}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch PL opening odds from football-data.co.uk (free)"
    )
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(seasons=args.seasons, dry_run=args.dry_run)