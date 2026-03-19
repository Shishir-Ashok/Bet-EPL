"""
scripts/simulate_historical_bets.py
-------------------------------------
Runs the full bet engine (XGBoost + DQN + Kelly) on every finished
historical match and stores the results in the bets, predictions,
and wallet tables.

This does two things:
  1. Populates the dashboard with real historical data immediately
  2. Proves the exact same code path used on live matchdays works end-to-end

The simulation is honest — it only uses data that was available BEFORE
each match (no lookahead). ELO ratings, form, and xG are all computed
using only matches prior to the one being evaluated.

Usage:
  uv run python scripts/simulate_historical_bets.py
  uv run python scripts/simulate_historical_bets.py --seasons 2023-24 2024-25
  uv run python scripts/simulate_historical_bets.py --dry-run   # no DB writes
"""

import os
import sys
import argparse
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.db import supabase
from backend.model.xgboost_model import load_model as load_xgb, predict_probabilities
from backend.model.dqn_agent     import DQNAgent
from backend.model.features      import bulk_fetch, build_state_vector
from backend.engine.kelly        import best_bet, remove_vig

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_SEASONS   = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
STARTING_BALANCE  = 10.0
DQN_GATE          = 0.3     # Q(bet) - Q(pass) threshold

# Implied odds used when no bookmaker odds exist for historical matches.
# We invert the model's own probabilities and add a synthetic 5% vig
# to simulate realistic bookmaker pricing.
SYNTHETIC_VIG     = 0.05


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_wallet_balance() -> float:
    r = supabase.table("wallet").select("balance").eq("id", 1).single().execute()
    return float(r.data["balance"])


def set_wallet(balance: float, staked_delta: float = 0, returned_delta: float = 0):
    current = supabase.table("wallet").select("*").eq("id", 1).single().execute().data
    supabase.table("wallet").update({
        "balance":        round(balance, 2),
        "total_staked":   round(float(current["total_staked"])   + staked_delta,   2),
        "total_returned": round(float(current["total_returned"]) + returned_delta, 2),
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


def load_matches_paginated(seasons: list[str]) -> list[dict]:
    """Loads all finished matches for given seasons, paginating past 1000-row cap."""
    matches, offset, page_size = [], 0, 1000
    base = (
        supabase.table("matches")
        .select(
            "id, home_team_id, away_team_id, kickoff_time, result, season, "
            "home:teams!matches_home_team_id_fkey(short_name, tla), "
            "away:teams!matches_away_team_id_fkey(short_name, tla)"
        )
        .in_("season", seasons)
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )
    while True:
        page = base.range(offset, offset + page_size - 1).execute().data
        if not page:
            break
        matches.extend(page)
        if len(page) < page_size:
            break
        offset    += page_size
        time.sleep(0.15)
    return matches


def get_historical_odds(match_id: int) -> dict | None:
    """Returns best stored odds for a match. None if not in DB."""
    r = (
        supabase.table("odds")
        .select("home_odds, draw_odds, away_odds")
        .eq("match_id", match_id)
        .order("fetched_at", desc=True)
        .limit(10)
        .execute()
    )
    if not r.data:
        return None
    return {
        "home_odds": max(float(x["home_odds"]) for x in r.data),
        "draw_odds": max(float(x["draw_odds"]) for x in r.data),
        "away_odds": max(float(x["away_odds"]) for x in r.data),
    }


def synthetic_odds(probs: dict) -> dict:
    """
    When no bookmaker odds exist, generate synthetic odds by inverting
    model probabilities and applying a synthetic vig.
    """
    factor = 1 + SYNTHETIC_VIG
    return {
        "home_odds": round(factor / max(probs["HOME"], 0.05), 2),
        "draw_odds": round(factor / max(probs["DRAW"], 0.05), 2),
        "away_odds": round(factor / max(probs["AWAY"], 0.05), 2),
    }


def already_simulated(match_id: int) -> bool:
    """Skip matches that already have a bet recorded."""
    r = supabase.table("bets").select("id").eq("match_id", match_id).limit(1).execute()
    return len(r.data) > 0


def get_model_version() -> str:
    r = (
        supabase.table("model_versions")
        .select("version_tag")
        .eq("model_type", "xgboost")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    return r.data[0]["version_tag"] if r.data else "unknown"


# ─── Main simulation ──────────────────────────────────────────────────────────

def run(seasons: list[str], dry_run: bool = False):
    print("=" * 60)
    print("  Historical bet simulation")
    print(f"  Seasons: {', '.join(seasons)}")
    print(f"  Mode:    {'DRY RUN — no DB writes' if dry_run else 'LIVE — writing to DB'}")
    print("=" * 60)

    # Load models
    print("\n  Loading models...")
    try:
        xgb_model = load_xgb()
        print("  ✓ XGBoost loaded")
    except FileNotFoundError as e:
        print(f"  ✗ XGBoost not found: {e}")
        print("  Run: uv run python -m backend.model.train --mode full")
        sys.exit(1)

    agent = None
    try:
        agent = DQNAgent.load_active()
        print("  ✓ DQN loaded")
    except FileNotFoundError:
        print("  ⚠ No DQN — using XGBoost + Kelly only")

    # Load matches
    print(f"\n  Loading finished matches for {seasons}...")
    matches = load_matches_paginated(seasons)
    print(f"  {len(matches)} finished matches found")

    if not matches:
        print("  No matches found. Run seed_historical_data.py first.")
        sys.exit(1)

    # Bulk fetch features once
    print("\n  Bulk fetching features (4 DB queries)...")
    feature_cache = bulk_fetch([m["id"] for m in matches])
    model_version = get_model_version()

    # Simulation loop
    balance        = get_wallet_balance()
    print(f"\n  Starting balance: €{balance:.2f}")
    print(f"  Processing {len(matches)} matches...\n")

    bets_placed = bets_won = bets_lost = 0
    total_pnl   = 0.0
    skipped     = 0

    for i, match in enumerate(matches):
        match_id = match["id"]
        result   = match["result"]
        home     = match["home"]["tla"]
        away     = match["away"]["tla"]

        # Skip if already has a bet
        if not dry_run and already_simulated(match_id):
            skipped += 1
            continue

        # Build state vector using only pre-match data
        try:
            state = build_state_vector(
                match_id       = match_id,
                home_team_id   = match["home_team_id"],
                away_team_id   = match["away_team_id"],
                kickoff_time   = match["kickoff_time"],
                wallet_balance = balance,
                before_date    = match["kickoff_time"],
                cache          = feature_cache,
            )
        except Exception as e:
            skipped += 1
            continue

        # XGBoost probabilities
        try:
            probs = predict_probabilities(xgb_model, state)
            state[7] = float(probs["HOME"])
            state[8] = float(probs["DRAW"])
            state[9] = float(probs["AWAY"])
        except Exception:
            skipped += 1
            continue

        # Get odds — use real if available, synthetic otherwise
        odds_data   = get_historical_odds(match_id) or synthetic_odds(probs)
        odds_source = "real" if get_historical_odds(match_id) else "synthetic"

        # Kelly sizing
        bet = best_bet(
            model_probs = probs,
            home_odds   = odds_data["home_odds"],
            draw_odds   = odds_data["draw_odds"],
            away_odds   = odds_data["away_odds"],
            balance     = balance,
        )

        # DQN confirmation gate
        action     = "PASS"
        confidence = 0.0

        if bet and agent:
            q_vals    = agent.get_q_values(state)
            action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
            dqn_action = action_map[bet["outcome"]]
            if q_vals[dqn_action] - q_vals["PASS"] >= DQN_GATE:
                action     = dqn_action
                confidence = agent.get_confidence(state)
        elif bet and not agent:
            action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
            action = action_map[bet["outcome"]]

        # Determine outcome
        if action != "PASS" and bet:
            outcome_map = {"BET_HOME": "HOME", "BET_DRAW": "DRAW", "BET_AWAY": "AWAY"}
            bet_on      = outcome_map[action]
            outcome     = "WIN" if bet_on == result else "LOSS"
            stake       = bet["stake"]
            odds        = bet["odds"]
            pnl         = round(stake * (odds - 1.0) if outcome == "WIN" else -stake, 2)
            new_balance = round(max(0.0, balance + pnl), 2)

            if not dry_run:
                # Save prediction
                pred = supabase.table("predictions").insert({
                    "match_id":           match_id,
                    "model_version":      model_version,
                    "prob_home":          probs["HOME"],
                    "prob_draw":          probs["DRAW"],
                    "prob_away":          probs["AWAY"],
                    "recommended_action": action,
                    "confidence":         confidence,
                    "was_correct":        (outcome == "WIN"),
                }).execute()
                pred_id = pred.data[0]["id"]

                # Save bet (already settled since match is finished)
                supabase.table("bets").insert({
                    "match_id":       match_id,
                    "prediction_id":  pred_id,
                    "action":         action,
                    "stake":          stake,
                    "odds":           odds,
                    "balance_before": balance,
                    "outcome":        outcome,
                    "pnl":            pnl,
                    "balance_after":  new_balance,
                    "placed_at":      match["kickoff_time"],
                    "settled_at":     match["kickoff_time"],
                }).execute()

                returned = (stake + pnl) if outcome == "WIN" else 0.0
                set_wallet(new_balance, staked_delta=stake, returned_delta=returned)

            balance    = new_balance
            total_pnl += pnl
            bets_placed += 1
            if outcome == "WIN":
                bets_won  += 1
            else:
                bets_lost += 1

            if not dry_run:
                status = "✓ WIN" if outcome == "WIN" else "✗ LOSS"
                print(f"  [{i+1:4d}/{len(matches)}] {home} vs {away:3s} "
                      f"| {action:8s} @ {odds:.2f} "
                      f"| €{stake:.2f} stake "
                      f"| {status} {'+' if pnl >= 0 else ''}€{pnl:.2f} "
                      f"| Balance: €{balance:.2f} "
                      f"| ({odds_source})")

    # Summary
    win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0
    roi      = (total_pnl / (bets_placed * 1.0) * 100) if bets_placed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {'DRY RUN ' if dry_run else ''}Simulation complete")
    print(f"  Matches processed : {len(matches) - skipped}")
    print(f"  Skipped           : {skipped} (already had bets or no features)")
    print(f"  Bets placed       : {bets_placed}")
    print(f"  Won / Lost        : {bets_won} / {bets_lost}")
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Total P&L         : {'+' if total_pnl >= 0 else ''}€{total_pnl:.2f}")
    print(f"  Final balance     : €{balance:.2f}")
    if dry_run:
        print(f"\n  ↑ DRY RUN — nothing was written to the database.")
        print(f"  Run without --dry-run to commit results.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate historical bets")
    parser.add_argument(
        "--seasons", nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to simulate e.g. 2023-24 2024-25"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing to DB"
    )
    args = parser.parse_args()
    run(seasons=args.seasons, dry_run=args.dry_run)