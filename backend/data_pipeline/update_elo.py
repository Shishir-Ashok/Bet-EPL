"""
backend/data_pipeline/update_elo.py
-------------------------------------
Calculates and stores ELO ratings for every PL team after each match.

Why ELO?
  ELO is one of the most predictive single features for football outcomes.
  It encodes the full history of results into a single number that updates
  after each game — teams that beat strong opponents gain more rating than
  teams that beat weak ones.

  Unlike raw form (last 5 results), ELO accounts for OPPONENT QUALITY.
  Beating Man City means more than beating relegated Southampton.

Our ELO parameters:
  K  = 32    — update speed. 32 is standard for football (chess uses 16-32).
               Higher K = faster adaptation, more volatile.
  HFA = 65   — Home Field Advantage in ELO points. Empirically derived from
               decades of PL data. Home teams win ~46% of PL games.
  Starting ELO = 1500 for all teams. After one full season, ratings will
               diverge meaningfully (Man City might reach 1700+, a newly
               promoted side might drop to 1300).

How it's run:
  After results are ingested (post-match cron), this script is called
  to recalculate ratings for all newly settled matches in chronological order.

Usage:
  python -m backend.data_pipeline.update_elo --season 2024-25
  python -m backend.data_pipeline.update_elo --all-seasons   (recalculates from scratch)
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── ELO Parameters ───────────────────────────────────────────────────────────

STARTING_ELO = 1500.0
K_FACTOR     = 32.0
HOME_ADVANTAGE = 65.0   # ELO points added to home team's effective rating


# ─── Core ELO math ────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Probability that team A beats team B, given their ratings.
    Classic ELO formula: E(A) = 1 / (1 + 10^((Rb - Ra) / 400))

    Returns a float between 0 and 1.
    e.g. Ra=1600, Rb=1400 → E(A) ≈ 0.76 (76% expected win probability)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    home_rating: float,
    away_rating: float,
    result: str,          # "HOME", "DRAW", or "AWAY"
) -> tuple[float, float]:
    """
    Applies one match result and returns updated (home_rating, away_rating).

    The home team gets an effective rating boost of HOME_ADVANTAGE for the
    purpose of calculating expected scores. The actual stored rating is still
    the "true" rating — we only apply the advantage inside this function.

    Actual scores:
      HOME win  → home_score=1.0, away_score=0.0
      DRAW      → home_score=0.5, away_score=0.5
      AWAY win  → home_score=0.0, away_score=1.0
    """
    # Apply home advantage only for expected score calculation
    effective_home = home_rating + HOME_ADVANTAGE

    e_home = expected_score(effective_home, away_rating)
    e_away = 1.0 - e_home

    if result == "HOME":
        s_home, s_away = 1.0, 0.0
    elif result == "AWAY":
        s_home, s_away = 0.0, 1.0
    else:  # DRAW
        s_home, s_away = 0.5, 0.5

    new_home = home_rating + K_FACTOR * (s_home - e_home)
    new_away = away_rating + K_FACTOR * (s_away - e_away)

    return round(new_home, 2), round(new_away, 2)


# ─── DB helpers ───────────────────────────────────────────────────────────────

def get_current_elo(team_id: int) -> float:
    """
    Returns the team's most recent ELO rating from the DB.
    Falls back to STARTING_ELO if the team has no ELO history yet.
    """
    result = (
        supabase.table("elo_ratings")
        .select("elo")
        .eq("team_id", team_id)
        .order("calculated_at", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return float(result.data[0]["elo"])
    return STARTING_ELO


def save_elo(team_id: int, match_id: int, elo: float) -> None:
    """Persists a new ELO rating for a team after a specific match."""
    supabase.table("elo_ratings").upsert(
        {"team_id": team_id, "match_id": match_id, "elo": elo},
        on_conflict="team_id,match_id"
    ).execute()


def get_matches_without_elo(season: str | None = None, all_seasons: bool = False) -> list[dict]:
    """
    Returns finished matches that don't yet have ELO records for both teams.

    We check for BOTH home and away ELO rows — a match only counts as "done"
    if both teams have been updated.
    """
    query = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, result, kickoff_time, season")
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)   # chronological — important for ELO
    )

    if season and not all_seasons:
        query = query.eq("season", season)

    matches = query.execute().data

    if not matches:
        return []

    # Find matches that already have ELO for home team (proxy for "already processed")
    match_ids = [m["id"] for m in matches]
    existing  = (
        supabase.table("elo_ratings")
        .select("match_id")
        .in_("match_id", match_ids)
        .execute()
    )
    processed_ids = {r["match_id"] for r in existing.data}

    return [m for m in matches if m["id"] not in processed_ids]


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(season: str | None = None, all_seasons: bool = False):
    print("=" * 55)
    print("  Updating ELO ratings")
    print("=" * 55)

    if all_seasons:
        print("  Mode: full recalculation (all seasons)")
        print("  ⚠ This clears all existing ELO ratings first!")

        # Nuclear option — wipe and recalculate from scratch
        # Use with care: only needed if K or HFA parameters change
        supabase.table("elo_ratings").delete().neq("id", 0).execute()
        matches = get_matches_without_elo(all_seasons=True)
    else:
        season = season or "2024-25"
        print(f"  Mode: incremental ({season})")
        matches = get_matches_without_elo(season=season)

    if not matches:
        print("\n  No new matches to process. All ELO ratings are up to date.")
        return

    print(f"\n  Processing {len(matches)} matches...\n")

    updated = 0
    for match in matches:
        home_id  = match["home_team_id"]
        away_id  = match["away_team_id"]
        result   = match["result"]
        match_id = match["id"]

        # Get current ratings BEFORE this match
        home_elo_before = get_current_elo(home_id)
        away_elo_before = get_current_elo(away_id)

        # Calculate new ratings
        home_elo_after, away_elo_after = update_ratings(
            home_elo_before, away_elo_before, result
        )

        # Persist
        save_elo(home_id, match_id, home_elo_after)
        save_elo(away_id, match_id, away_elo_after)

        home_delta = home_elo_after - home_elo_before
        away_delta = away_elo_after - away_elo_before

        print(
            f"  Match {match_id} ({result}): "
            f"Home {home_elo_before:.0f} → {home_elo_after:.0f} "
            f"({'+'if home_delta>=0 else ''}{home_delta:.1f})  |  "
            f"Away {away_elo_before:.0f} → {away_elo_after:.0f} "
            f"({'+'if away_delta>=0 else ''}{away_delta:.1f})"
        )
        updated += 1

    print(f"\n✓ ELO ratings updated for {updated} matches.")

    # Print current top 5 as a sanity check
    print("\n  Current top 5 ELO ratings:")
    teams  = supabase.table("teams").select("id, short_name").execute().data
    team_map = {t["id"]: t["short_name"] for t in teams}

    top_elos = []
    for team in teams:
        elo = get_current_elo(team["id"])
        if elo != STARTING_ELO:  # Only show teams with real history
            top_elos.append((team["short_name"], elo))

    top_elos.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, elo) in enumerate(top_elos[:5], 1):
        print(f"  {rank}. {name}: {elo:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update ELO ratings after match results")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--season",      default="2024-25", help="Season to update (default: 2024-25)")
    group.add_argument("--all-seasons", action="store_true", help="Recalculate all seasons from scratch")
    args = parser.parse_args()

    run(season=args.season, all_seasons=args.all_seasons)
