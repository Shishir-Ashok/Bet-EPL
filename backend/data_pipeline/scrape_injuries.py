"""
backend/data_pipeline/scrape_injuries.py
------------------------------------------
Scrapes current injury and suspension lists from the BBC Sport
Premier League pages and upserts them into `team_injuries`.

Why BBC Sport?
  - No auth required, no JavaScript rendering needed
  - Updated daily by editorial staff
  - Clean, consistent HTML structure
  - More reliable than club official sites (which vary wildly)

How this feeds the model:
  The injury table is used to compute an "injury impact score" for
  each team before a match. Key players missing = lower expected
  performance. This is one of the 16 features in the DQN state vector.

Strategy:
  - We clear old injury records for a team before inserting new ones
    (injuries resolve, new ones occur — we want the current snapshot)
  - We run this the morning of a matchday, so the model has fresh news

Usage:
  python -m backend.data_pipeline.scrape_injuries
  python -m backend.data_pipeline.scrape_injuries --team ARS
"""

import os
import sys
import re
import time
import argparse

import requests
from bs4 import BeautifulSoup
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Constants ────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PLBettingBot/1.0; research project)",
    "Accept-Language": "en-GB,en;q=0.9",
}

SCRAPE_DELAY = 3  # seconds between team pages

# BBC Sport team slug → DB TLA mapping.
# Covers all clubs in the PL since 2020-21, including promoted/relegated.
# Add new clubs here whenever a team outside this list is promoted.
BBC_TEAM_SLUGS = {
    # 2025-26 current season
    "ARS": "arsenal",
    "AVL": "aston-villa",
    "BOU": "bournemouth",
    "BRE": "brentford",
    "BHA": "brighton-and-hove-albion",
    "BUR": "burnley",          # promoted 2025-26
    "CHE": "chelsea",
    "CRY": "crystal-palace",
    "EVE": "everton",
    "FUL": "fulham",
    "LEE": "leeds-united",     # promoted 2025-26
    "LIV": "liverpool",
    "MCI": "manchester-city",
    "MUN": "manchester-united",
    "NEW": "newcastle-united",
    "NFO": "nottingham-forest",
    "SUN": "sunderland",       # promoted 2025-26
    "TOT": "tottenham-hotspur",
    "WHU": "west-ham-united",
    "WOL": "wolverhampton-wanderers",
    # Recently relegated — keep so historical scrapes still work
    "IPS": "ipswich-town",
    "LEI": "leicester-city",
    "SOU": "southampton",
    "LUT": "luton-town",
    "SHU": "sheffield-united",
    "NOR": "norwich-city",
    "WAT": "watford",
    "WBA": "west-bromwich-albion",
}

BBC_BASE = "https://www.bbc.com/sport/football"

# Player importance weights — used to compute injury_impact score
# Position weights approximate the impact of losing a player in that role
POSITION_WEIGHTS = {
    "GK":  0.15,
    "DEF": 0.10,
    "MID": 0.10,
    "FWD": 0.12,
}
DEFAULT_WEIGHT = 0.10


# ─── Scraping logic ───────────────────────────────────────────────────────────

def scrape_team_injuries(tla: str, team_id: int) -> list[dict]:
    """
    Scrapes the BBC Sport injury page for a single team.
    Returns a list of injury record dicts ready to insert into the DB.

    BBC Sport injury page URL format:
      https://www.bbc.com/sport/football/teams/{slug}/squad
    """
    slug = BBC_TEAM_SLUGS.get(tla)
    if not slug:
        print(f"  ⚠ No BBC slug found for {tla}, skipping")
        return []

    url = f"{BBC_BASE}/teams/{slug}/injuries-and-suspensions"
    response = _get_with_retry(url)
    if not response:
        return []

    soup    = BeautifulSoup(response.text, "lxml")
    records = []

    # BBC renders injury data in a list format with player name, status, and
    # expected return. The exact HTML structure varies slightly so we handle
    # multiple patterns.

    # Pattern 1: Table-based layout (most common)
    injury_table = soup.find("table", class_=re.compile("gs-o-table"))
    if injury_table:
        rows = injury_table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            player_name = cells[0].get_text(strip=True)
            status_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            return_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""

            if not player_name or player_name.lower() in ("player", "name"):
                continue  # Skip header rows

            status        = _classify_status(status_text)
            return_date   = _parse_return_date(return_text)
            records.append({
                "team_id":     team_id,
                "player_name": player_name,
                "status":      status,
                "return_date": return_date,
                "source_url":  url,
            })

    # Pattern 2: Article/list layout (BBC sometimes uses this for suspensions)
    if not records:
        injury_items = soup.find_all("li", class_=re.compile("sp-c-squad__item"))
        for item in injury_items:
            name_el   = item.find(class_=re.compile("sp-c-squad__name"))
            status_el = item.find(class_=re.compile("sp-c-squad__status"))
            return_el = item.find(class_=re.compile("sp-c-squad__return"))

            if not name_el:
                continue

            records.append({
                "team_id":     team_id,
                "player_name": name_el.get_text(strip=True),
                "status":      _classify_status(status_el.get_text(strip=True) if status_el else ""),
                "return_date": _parse_return_date(return_el.get_text(strip=True) if return_el else ""),
                "source_url":  url,
            })

    return records


# ─── Utilities ────────────────────────────────────────────────────────────────

def _classify_status(text: str) -> str:
    """Maps freeform status text to our three categories."""
    text_lower = text.lower()
    if "suspend" in text_lower:
        return "Suspended"
    if "doubt" in text_lower or "50" in text_lower:
        return "Doubt"
    return "Injured"  # Default


def _parse_return_date(text: str) -> str | None:
    """
    Tries to parse a return date from text like:
      "24 Oct", "October 2024", "Unknown", "Season"

    Returns an ISO date string or None.
    """
    if not text or "unknown" in text.lower() or "season" in text.lower():
        return None

    # Try common formats
    formats = ["%d %b", "%d %B", "%B %Y", "%d %b %Y"]
    current_year = date.today().year

    for fmt in formats:
        try:
            parsed = None
            if fmt in ("%d %b", "%d %B"):
                # No year — assume current or next year
                parsed = date(current_year, 1, 1)  # placeholder, re-parse below
                parsed = date.fromisoformat(
                    f"{current_year}-{time.strptime(text, fmt).tm_mon:02d}-{time.strptime(text, fmt).tm_mday:02d}"
                )
                # If the date has already passed, it's probably next year
                if parsed < date.today():
                    parsed = date(current_year + 1, parsed.month, parsed.day)
            else:
                import datetime as dt
                parsed = dt.datetime.strptime(text, fmt).date()

            return parsed.isoformat() if parsed else None
        except (ValueError, OverflowError):
            continue

    return None


def compute_injury_impact(team_id: int) -> float:
    """
    Returns a 0.0–1.0 score representing how much a team is affected
    by current injuries and suspensions.

    Used as one of the 16 features in the DQN state vector:
      0.0 = full squad available
      1.0 = heavily depleted (unrealistic in practice, max ~0.4–0.5)

    Formula: sum(weight for each unavailable player) / normalisation_factor
    The normalisation factor (2.0) is empirically set so a team missing
    3 key players scores roughly 0.5.
    """
    injuries = (
        supabase.table("team_injuries")
        .select("status, player_id, players(position)")
        .eq("team_id", team_id)
        .execute()
    )

    if not injuries.data:
        return 0.0

    total_impact = 0.0
    for record in injuries.data:
        position = None
        if record.get("players"):
            position = record["players"].get("position")
        weight = POSITION_WEIGHTS.get(position, DEFAULT_WEIGHT)
        # Doubts count as half-impact (they might play)
        if record["status"] == "Doubt":
            weight *= 0.5
        total_impact += weight

    return round(min(total_impact / 2.0, 1.0), 4)


def _get_with_retry(url: str, max_retries: int = 3) -> requests.Response | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                return response
            if response.status_code == 404:
                return None
            print(f"  ⚠ HTTP {response.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  ⚠ Error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(tla_filter: str | None = None):
    print("=" * 55)
    print("  Scraping injury & suspension news")
    print("=" * 55)

    # Fetch all teams from DB
    teams = supabase.table("teams").select("id, name, tla").execute().data

    if tla_filter:
        teams = [t for t in teams if t["tla"] == tla_filter.upper()]
        if not teams:
            print(f"ERROR: Team with TLA '{tla_filter}' not found in DB")
            sys.exit(1)

    updated_teams  = 0
    total_injuries = 0

    for team in teams:
        tla     = team["tla"]
        team_id = team["id"]
        name    = team["name"]

        print(f"\n  {name} ({tla})...")

        records = scrape_team_injuries(tla, team_id)

        if not records:
            print(f"    → No injury data found (squad may be fully fit)")
            continue

        # Clear old records for this team before inserting fresh ones
        supabase.table("team_injuries").delete().eq("team_id", team_id).execute()

        # Insert new records
        supabase.table("team_injuries").insert(records).execute()

        impact = compute_injury_impact(team_id)
        print(f"    → {len(records)} players unavailable | injury impact: {impact:.2f}")

        updated_teams  += 1
        total_injuries += len(records)

        time.sleep(SCRAPE_DELAY)

    print(f"\n✓ Done. {updated_teams} teams updated, {total_injuries} injury records inserted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PL injury news from BBC Sport")
    parser.add_argument("--team", help="Only scrape a specific team (e.g. ARS)", default=None)
    args = parser.parse_args()

    run(tla_filter=args.team)
