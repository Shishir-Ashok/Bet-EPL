"""
backend/data_pipeline/odds_validator.py
-----------------------------------------
Validates that home_odds and away_odds are not swapped before
writing to the database.

The Odds API occasionally returns lines where the home and away
columns are transposed — the symptom is a heavy favourite appearing
at 6.0 when their ELO strongly predicts they should be ~1.4.

Strategy
--------
1. Sanity checks: all odds are decimal > 1.0 and total implied
   probability (vig check) sits in a realistic range (102%–115%).
2. Favourite alignment: the team with the higher ELO rating should
   generally have the lower (odds-on or close-to-even) odds.
   If ELO difference is large (≥ 150 points) we check that the
   odds direction agrees. If it doesn't, we swap home/away odds.
3. Extreme inversion guard: if home_odds > 5.0 AND away_odds < 1.5
   (or vice versa) with a moderate ELO edge, that's almost certainly
   a swap regardless of the ELO threshold.

How to use
----------
In fetch_odds.py, replace the direct supabase insert with:

    from backend.data_pipeline.odds_validator import validate_odds_row

    validated = validate_odds_row(
        home_team_id = match["home_team_id"],
        away_team_id = match["away_team_id"],
        home_odds    = raw_home_odds,
        draw_odds    = raw_draw_odds,
        away_odds    = raw_away_odds,
    )
    if validated is None:
        print(f"  ✗ Odds rejected for match {match_id} — implausible values")
        continue

    supabase.table("odds").upsert({
        "match_id":   match_id,
        "home_odds":  validated["home_odds"],
        "draw_odds":  validated["draw_odds"],
        "away_odds":  validated["away_odds"],
        "bookmaker":  bookmaker,
    }, on_conflict="match_id").execute()
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Thresholds ───────────────────────────────────────────────────────────────

# Minimum / maximum realistic decimal odds
ODDS_MIN = 1.01
ODDS_MAX = 30.0

# Total implied probability must sit between these bounds (covers vig)
VIG_LOW  = 1.01   # below 101% = no vig = suspicious
VIG_HIGH = 1.20   # above 120% = massive overround = suspicious

# ELO difference above which we enforce favourite direction
ELO_STRONG_THRESHOLD = 150   # ≥150 ELO pts difference = meaningful edge

# If home_odds / away_odds ratio exceeds this, we call it "extreme"
# even with a smaller ELO gap (e.g. 6.0 vs 1.4 = ratio 4.3)
EXTREME_RATIO = 3.5

# Pairs swapped in the last run — printed for diagnostics
_SWAPS_THIS_SESSION: list[dict] = []


# ─── ELO helper ───────────────────────────────────────────────────────────────

def _get_elos(home_team_id: int, away_team_id: int) -> tuple[float, float]:
    """
    Returns (home_elo, away_elo). Falls back to 1500/1500 if not found.
    Uses the most recent elo_ratings row for each team.
    """
    default = 1500.0

    def _fetch(team_id: int) -> float:
        try:
            row = (
                supabase.table("elo_ratings")
                .select("elo")
                .eq("team_id", team_id)
                .order("calculated_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
            return float(row[0]["elo"]) if row else default
        except Exception:
            return default

    return _fetch(home_team_id), _fetch(away_team_id)


# ─── Core validator ───────────────────────────────────────────────────────────

def validate_odds_row(
    home_team_id: int,
    away_team_id: int,
    home_odds:    float,
    draw_odds:    float,
    away_odds:    float,
    match_id:     Optional[int] = None,   # for logging only
) -> Optional[dict]:
    """
    Validates and (if needed) corrects a set of three decimal odds.

    Returns a dict {"home_odds", "draw_odds", "away_odds", "swapped": bool}
    or None if the odds are fundamentally implausible and should be rejected.

    Parameters
    ----------
    home_team_id / away_team_id : used to fetch ELO ratings from the DB.
    match_id : optional, used only in log messages.
    """
    tag = f"match_id={match_id}" if match_id else f"teams {home_team_id} vs {away_team_id}"

    try:
        ho = float(home_odds)
        do = float(draw_odds)
        ao = float(away_odds)
    except (TypeError, ValueError):
        print(f"  [OddsValidator] ✗ Non-numeric odds for {tag}")
        return None

    # ── 1. Hard sanity checks ─────────────────────────────────────────────────
    if not (ODDS_MIN <= ho <= ODDS_MAX and
            ODDS_MIN <= do <= ODDS_MAX and
            ODDS_MIN <= ao <= ODDS_MAX):
        print(f"  [OddsValidator] ✗ Out-of-range odds for {tag}: "
              f"H={ho} D={do} A={ao}")
        return None

    total_implied = (1 / ho) + (1 / do) + (1 / ao)
    if not (VIG_LOW <= total_implied <= VIG_HIGH):
        print(f"  [OddsValidator] ✗ Implausible total implied prob {total_implied:.3f} "
              f"for {tag}")
        return None

    # Draw odds must be reasonable (draw probability is ~25-35% in PL)
    draw_implied = 1 / do
    if not (0.18 <= draw_implied <= 0.50):
        print(f"  [OddsValidator] ✗ Implausible draw implied prob {draw_implied:.3f} "
              f"for {tag}")
        return None

    # ── 2. Fetch ELO ratings ──────────────────────────────────────────────────
    home_elo, away_elo = _get_elos(home_team_id, away_team_id)
    elo_diff = home_elo - away_elo   # positive = home is stronger

    # ── 3. Detect swap ───────────────────────────────────────────────────────
    # home_favoured_by_elo  : home ELO is meaningfully higher
    # home_favoured_by_odds : home odds are meaningfully lower
    home_elo_edge  = elo_diff >= ELO_STRONG_THRESHOLD
    away_elo_edge  = elo_diff <= -ELO_STRONG_THRESHOLD
    odds_ratio_ha  = ho / ao   # > 1 means away is the odds-on favourite
    odds_ratio_ah  = ao / ho   # > 1 means home is the odds-on favourite

    # Extreme inversion: one side is heavily odds-on while ELO says the
    # other is the stronger team — almost certainly a swap.
    extreme_home_swap = (odds_ratio_ha >= EXTREME_RATIO and home_elo_edge)
    extreme_away_swap = (odds_ratio_ah >= EXTREME_RATIO and away_elo_edge)

    # Strong-ELO disagreement: ELO says home is clear favourite but
    # away odds are lower (and the ratio is notable).
    elo_home_odds_away = home_elo_edge and (ao < ho) and (odds_ratio_ha >= 1.5)
    elo_away_odds_home = away_elo_edge and (ho < ao) and (odds_ratio_ah >= 1.5)

    should_swap = (
        extreme_home_swap or
        extreme_away_swap or
        elo_home_odds_away or
        elo_away_odds_home
    )

    if should_swap:
        ho, ao = ao, ho   # swap home ↔ away; draw is symmetric, stays the same
        _SWAPS_THIS_SESSION.append({
            "tag":      tag,
            "elo_diff": round(elo_diff, 1),
            "original": {"H": float(home_odds), "D": do, "A": float(away_odds)},
            "fixed":    {"H": ho, "D": do, "A": ao},
        })
        print(f"  [OddsValidator] ⚠ Swapped home/away odds for {tag}  "
              f"ELO_diff={elo_diff:+.0f}  "
              f"H:{float(home_odds):.2f}→{ho:.2f}  "
              f"A:{float(away_odds):.2f}→{ao:.2f}")

    return {
        "home_odds": round(ho, 4),
        "draw_odds": round(do, 4),
        "away_odds": round(ao, 4),
        "swapped":   should_swap,
    }


def get_session_swaps() -> list[dict]:
    """Returns all swaps detected in the current session (for audit logging)."""
    return list(_SWAPS_THIS_SESSION)


# ─── Bulk re-validator for existing odds rows ─────────────────────────────────

def revalidate_existing_odds() -> dict:
    """
    Runs the validator over every row currently in the odds table and
    fixes any swaps in-place.

    Returns a summary dict: {"total", "swapped", "rejected"}.
    Call this once after the DB reset if you re-fetch historical odds
    before running the backtest.
    """
    rows = supabase.table("odds").select("id, match_id, home_odds, draw_odds, away_odds").execute().data
    total = len(rows)
    swapped = 0
    rejected = 0

    for row in rows:
        # Get team IDs from the joined match
        match = (
            supabase.table("matches")
            .select("home_team_id, away_team_id")
            .eq("id", row["match_id"])
            .single()
            .execute()
            .data
        )
        if not match:
            continue

        validated = validate_odds_row(
            home_team_id = match["home_team_id"],
            away_team_id = match["away_team_id"],
            home_odds    = row["home_odds"],
            draw_odds    = row["draw_odds"],
            away_odds    = row["away_odds"],
            match_id     = row["match_id"],
        )

        if validated is None:
            supabase.table("odds").delete().eq("id", row["id"]).execute()
            rejected += 1
            continue

        if validated["swapped"]:
            supabase.table("odds").update({
                "home_odds": validated["home_odds"],
                "away_odds": validated["away_odds"],
            }).eq("id", row["id"]).execute()
            swapped += 1

    print(f"\n  [OddsValidator] Revalidation complete: "
          f"{total} rows  |  {swapped} swapped  |  {rejected} rejected")
    return {"total": total, "swapped": swapped, "rejected": rejected}