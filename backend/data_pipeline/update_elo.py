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
  HFA = 65   — Home Field Advantage in ELO points.
  Starting ELO = 1500 for all teams.

Fixes vs original:
  1. Supabase silently caps .execute() at 1000 rows — all queries now paginate.
  2. .in_() with 1000+ IDs also breaks — chunked into batches of 200.
  3. get_current_elo() was called per-team per-match (~4000 DB round trips
     for a full recalculation). Now uses an in-memory dict that updates as
     matches are processed chronologically — zero extra queries in the loop.
  4. calculated_at stored as match kickoff_time, not now(). This is critical
     for features.py._get_elo() temporal filtering to work correctly.

Usage:
  python -m backend.data_pipeline.update_elo --season 2024-25
  python -m backend.data_pipeline.update_elo --all-seasons
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.data_pipeline.season_utils import get_current_season
from backend.db import supabase

# ─── ELO Parameters ───────────────────────────────────────────────────────────

STARTING_ELO   = 1500.0
K_FACTOR       = 32.0
HOME_ADVANTAGE = 65.0


# ─── Pagination helper ────────────────────────────────────────────────────────

def _paginate(query, page_size: int = 1000) -> list:
    """
    Pages through Supabase results to overcome the silent 1000-row cap.
    Without this, .execute() silently truncates — no error, no warning.
    """
    rows, offset = [], 0
    while True:
        for attempt in range(3):
            try:
                page = query.range(offset, offset + page_size - 1).execute().data
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"  Supabase error (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                time.sleep(wait)

        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.15)
    return rows


# ─── Core ELO math ────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    home_rating: float,
    away_rating: float,
    result: str,
) -> tuple[float, float]:
    effective_home = home_rating + HOME_ADVANTAGE
    e_home = expected_score(effective_home, away_rating)
    e_away = 1.0 - e_home

    if result == "HOME":
        s_home, s_away = 1.0, 0.0
    elif result == "AWAY":
        s_home, s_away = 0.0, 1.0
    else:
        s_home, s_away = 0.5, 0.5

    new_home = home_rating + K_FACTOR * (s_home - e_home)
    new_away = away_rating + K_FACTOR * (s_away - e_away)
    return round(new_home, 2), round(new_away, 2)


# ─── DB helpers ───────────────────────────────────────────────────────────────

def get_current_elo(team_id: int) -> float:
    """
    Returns the team's most recent ELO from DB.
    Used only for the top-5 display at the end and for live inference.
    NOT called during the main processing loop — we use in-memory state instead.
    """
    result = (
        supabase.table("elo_ratings")
        .select("elo")
        .eq("team_id", team_id)
        .order("calculated_at", desc=True)
        .limit(1)
        .execute()
    )
    return float(result.data[0]["elo"]) if result.data else STARTING_ELO


def _save_elo_batch(rows: list[dict]) -> None:
    """
    Bulk-upserts a batch of ELO rows.
    Each row: {team_id, match_id, elo, calculated_at}.

    CRITICAL: calculated_at = match kickoff_time (not now()).
    features.py._get_elo() filters with `if calc_at < before_time`.
    If calculated_at were the script-run timestamp (Supabase default),
    that filter would always be False for historical matches and every
    team would return 1500 — killing ELO as a feature.
    """
    if not rows:
        return
    supabase.table("elo_ratings").upsert(
        rows,
        on_conflict="team_id,match_id"
    ).execute()


def _load_all_finished_matches(season: str | None = None) -> list[dict]:
    """
    Returns all finished matches with results, in strict chronological order.
    Paginates to overcome Supabase's 1000-row cap.
    """
    query = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, result, kickoff_time, season")
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )
    if season:
        query = query.eq("season", season)

    matches = _paginate(query)
    print(f"  Loaded {len(matches)} finished matches")
    return matches


def _get_already_processed_ids(match_ids: list[int]) -> set[int]:
    """
    Returns the set of match IDs that already have ELO rows.
    Chunked into batches of 200 to avoid .in_() query size limits.
    """
    processed: set[int] = set()
    for i in range(0, len(match_ids), 200):
        chunk = match_ids[i : i + 200]
        rows  = (
            supabase.table("elo_ratings")
            .select("match_id")
            .in_("match_id", chunk)
            .execute()
            .data
        )
        processed.update(r["match_id"] for r in rows)
        time.sleep(0.05)
    return processed


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(season: str | None = None, all_seasons: bool = False):
    print("=" * 55)
    print("  Updating ELO ratings")
    print("=" * 55)

    if all_seasons:
        print("  Mode: full recalculation (all seasons)")
        print("  Clearing all existing ELO ratings...")
        supabase.table("elo_ratings").delete().neq("id", 0).execute()
        matches_to_process = _load_all_finished_matches()
    else:
        if season is None:
            season = get_current_season()["label"]
            print(f"  No season specified — auto-detected: {season}")
        print(f"  Mode: incremental ({season})")

        # Load all matches for this season first, then filter to unprocessed
        all_season_matches = _load_all_finished_matches(season=season)
        if not all_season_matches:
            print("\n  No finished matches found for this season.")
            return

        match_ids      = [m["id"] for m in all_season_matches]
        processed_ids  = _get_already_processed_ids(match_ids)
        matches_to_process = [m for m in all_season_matches if m["id"] not in processed_ids]
        print(f"  Already processed: {len(processed_ids)}  |  Remaining: {len(matches_to_process)}")

    if not matches_to_process:
        print("\n  No new matches to process. All ELO ratings are up to date.")
        return

    print(f"\n  Processing {len(matches_to_process)} matches...\n")

    # ── In-memory ELO state ───────────────────────────────────────────────────
    # Instead of querying get_current_elo() for every team on every match
    # (~4000 DB round trips for a full recalculation), we:
    #   1. Seed the dict with each team's current DB rating once
    #   2. Update it in memory after every match
    #   3. Flush to DB in batches of 50
    #
    # This reduces DB calls from O(matches) to O(1) seed + batched writes.

    current_elo: dict[int, float] = {}

    def _get_elo_for(team_id: int) -> float:
        if team_id not in current_elo:
            current_elo[team_id] = get_current_elo(team_id)
        return current_elo[team_id]

    # ── Process matches chronologically ──────────────────────────────────────
    BATCH_SIZE  = 50
    pending_rows: list[dict] = []
    updated = 0

    for match in matches_to_process:
        home_id      = match["home_team_id"]
        away_id      = match["away_team_id"]
        result       = match["result"]
        match_id     = match["id"]
        kickoff_time = match["kickoff_time"]

        home_before = _get_elo_for(home_id)
        away_before = _get_elo_for(away_id)

        home_after, away_after = update_ratings(home_before, away_before, result)

        # Update in-memory state immediately so the next match sees correct ELO
        current_elo[home_id] = home_after
        current_elo[away_id] = away_after

        # Queue for batch write — calculated_at = kickoff_time (not now())
        pending_rows.append({
            "team_id":       home_id,
            "match_id":      match_id,
            "elo":           home_after,
            "calculated_at": kickoff_time,
        })
        pending_rows.append({
            "team_id":       away_id,
            "match_id":      match_id,
            "elo":           away_after,
            "calculated_at": kickoff_time,
        })

        home_delta = home_after - home_before
        away_delta = away_after - away_before
        print(
            f"  [{updated+1:4d}/{len(matches_to_process)}] "
            f"Match {match_id} [{match['season']}] ({result}): "
            f"Home {home_before:.0f}→{home_after:.0f} "
            f"({'+'if home_delta>=0 else ''}{home_delta:.1f})  |  "
            f"Away {away_before:.0f}→{away_after:.0f} "
            f"({'+'if away_delta>=0 else ''}{away_delta:.1f})"
        )
        updated += 1

        # Flush batch to DB
        if len(pending_rows) >= BATCH_SIZE * 2:
            _save_elo_batch(pending_rows)
            pending_rows = []

    # Flush remaining
    if pending_rows:
        _save_elo_batch(pending_rows)

    print(f"\n  ✓ ELO ratings updated for {updated} matches.")

    # ── Top 5 current ratings ─────────────────────────────────────────────────
    # Use in-memory state if available (much faster than re-querying)
    if current_elo:
        # Get team names for display
        teams     = supabase.table("teams").select("id, short_name").execute().data
        name_map  = {t["id"]: t["short_name"] for t in teams}
        top_elos  = sorted(
            [(name_map.get(tid, str(tid)), elo) for tid, elo in current_elo.items()],
            key=lambda x: x[1], reverse=True,
        )
        print("\n  Current top 5 ELO ratings:")
        for rank, (name, elo) in enumerate(top_elos[:5], 1):
            print(f"  {rank}. {name}: {elo:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update ELO ratings after match results")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--season", default=None,
        help="Season label e.g. 2024-25 (default: auto-detect current season)"
    )
    group.add_argument(
        "--all-seasons", action="store_true",
        help="Wipe and recalculate all seasons from scratch"
    )
    args = parser.parse_args()
    run(season=args.season, all_seasons=args.all_seasons)