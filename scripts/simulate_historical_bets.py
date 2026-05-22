"""
scripts/simulate_historical_bets.py
-------------------------------------
Runs the full bet engine (XGBoost + DQN + Kelly) on finished matches
for the simulation seasons (2024-25 and 2025-26 only).

Why NOT 2020-24:
  Those are XGBoost training seasons. Simulating bets on training data
  inflates win rate and is not an honest backtest.

Fixes applied:
  1. bulk_fetch() called with no arguments (was passing match ID list)
  2. build_state_vector called with correct args (no match_id/wallet_balance)
  3. xgb probabilities not written back into xgb_vec (was corrupting vector)
  4. DQN receives dqn_state (24-dim), not xgb_vec (16-dim)
  5. Odds loaded in bulk before the loop (not N individual queries)
  6. Synthetic odds use blended priors, not self-pricing
  7. Training seasons excluded from DEFAULT_SEASONS
  8. --before-date flag added for monthly 2025-26 chunks
  9. replay_buffer.push() called after each settled bet so DQN trains
     on real transitions (was missing entirely — DQN trained on nothing)

Usage:
  python scripts/simulate_historical_bets.py
  python scripts/simulate_historical_bets.py --seasons 2024-25
  python scripts/simulate_historical_bets.py --seasons 2025-26 --before-date 2026-02-01
  python scripts/simulate_historical_bets.py --dry-run
  python scripts/simulate_historical_bets.py --no-dqn
"""

import os
import sys
import argparse
import math
import time
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.db import supabase
from backend.model.xgboost_model import load_model as load_xgb, predict_probabilities
from backend.model.dqn_agent     import DQNAgent
from backend.model.features      import bulk_fetch, build_state_vector, build_dqn_state
from backend.model               import replay_buffer
from backend.engine.kelly        import best_bet, remove_vig
from backend.engine.config       import DQN_CONFIDENCE_GATE

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_SEASONS  = ["2024-25", "2025-26"]
STARTING_BALANCE = 100.0

SYNTHETIC_MARGIN = 0.08
PL_BASE_RATES    = {"HOME": 0.46, "DRAW": 0.26, "AWAY": 0.28}
BLEND_FACTOR     = 0.35

MAX_SYNTHETIC_WARNINGS = 5

# Reward normalisation for replay buffer — matches settle_bets.py convention
REPLAY_STAKE_NORM = 1.0   # treat each bet as per-unit-staked for RL reward scale


# ─── Wallet helpers ───────────────────────────────────────────────────────────

def _get_wallet() -> dict:
    return supabase.table("wallet").select("*").eq("id", 1).single().execute().data


def _update_wallet(balance: float, staked: float, returned: float) -> None:
    w = _get_wallet()
    supabase.table("wallet").update({
        "balance":        round(balance, 2),
        "total_staked":   round(float(w["total_staked"])   + staked,   2),
        "total_returned": round(float(w["total_returned"]) + returned, 2),
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_matches(seasons: list[str], before_date: str = None) -> list[dict]:
    matches, offset, page_size = [], 0, 500
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
    if before_date:
        base = base.lt("kickoff_time", before_date)
    while True:
        page = base.range(offset, offset + page_size - 1).execute().data
        if not page:
            break
        matches.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.1)
    return matches


def _load_odds_cache(match_ids: list[int]) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    for i in range(0, len(match_ids), 300):
        chunk = match_ids[i : i + 300]
        rows  = (
            supabase.table("odds")
            .select("match_id, home_odds, draw_odds, away_odds, bookmaker")
            .in_("match_id", chunk)
            .execute()
            .data
        )
        for r in rows:
            cache[r["match_id"]] = {
                "home_odds": float(r["home_odds"]),
                "draw_odds": float(r["draw_odds"]),
                "away_odds": float(r["away_odds"]),
                "bookmaker": r.get("bookmaker", "stored"),
            }
        time.sleep(0.05)
    return cache


def _load_simulated_ids(match_ids: list[int]) -> set[int]:
    simulated: set[int] = set()
    for i in range(0, len(match_ids), 300):
        chunk = match_ids[i : i + 300]
        rows  = (
            supabase.table("bets")
            .select("match_id")
            .in_("match_id", chunk)
            .execute()
            .data
        )
        simulated.update(r["match_id"] for r in rows)
    return simulated


def _get_model_version() -> str:
    r = (
        supabase.table("model_versions")
        .select("version_tag")
        .eq("model_type", "xgboost")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    return r.data[0]["version_tag"] if r.data else "unknown"


# ─── Odds / probability helpers ───────────────────────────────────────────────

def _synthetic_odds(probs: dict) -> dict:
    blended = {
        k: BLEND_FACTOR * PL_BASE_RATES[k] + (1 - BLEND_FACTOR) * probs[k]
        for k in ("HOME", "DRAW", "AWAY")
    }
    total = sum(blended.values())
    return {
        "home_odds": round(1.0 / ((blended["HOME"] / total) * (1 + SYNTHETIC_MARGIN)), 3),
        "draw_odds": round(1.0 / ((blended["DRAW"] / total) * (1 + SYNTHETIC_MARGIN)), 3),
        "away_odds": round(1.0 / ((blended["AWAY"] / total) * (1 + SYNTHETIC_MARGIN)), 3),
        "bookmaker": "synthetic",
    }


def _compute_log_loss(probs: dict, result: str) -> float:
    true_prob = max(float(probs.get(result, 0.001)), 1e-7)
    return round(-math.log(true_prob), 6)


# ─── Replay buffer helper ─────────────────────────────────────────────────────

def _push_to_replay_buffer(
    match_id:  int,
    dqn_state: np.ndarray,
    action:    str,
    pnl:       float,
    stake:     float,
) -> None:
    """
    Stores a settled bet as an RL transition in the replay buffer.

    Without this, DQN trains on an empty buffer and produces random
    Q-values — effectively disabling the confirmation gate.

    Reward is normalised to per-unit-staked (same scale as live settle_bets.py)
    so the DQN learns consistent reward magnitudes across both simulation
    and live trading.
    """
    ACTION_MAP = {"BET_HOME": 0, "BET_DRAW": 1, "BET_AWAY": 2, "PASS": 3}
    action_idx = ACTION_MAP.get(action, 3)

    # Normalise reward to per-unit-staked, clipped to [-2, +2]
    normalised_reward = float(np.clip(pnl / max(stake, 1e-6), -2.0, 2.0))

    try:
        replay_buffer.push(
            match_id   = match_id,
            state      = dqn_state,
            action     = action_idx,
            reward     = normalised_reward,
            next_state = None,
            done       = False,
        )
    except Exception as e:
        # Non-fatal — simulation continues if replay buffer write fails
        pass


# ─── Main simulation ──────────────────────────────────────────────────────────

def run(
    seasons:     list[str],
    dry_run:     bool = False,
    no_dqn:      bool = False,
    before_date: str  = None,
) -> dict:
    date_label = f" before {before_date}" if before_date else ""
    print("=" * 65)
    print("  Historical bet simulation")
    print(f"  Seasons:  {', '.join(seasons)}{date_label}")
    print(f"  Mode:     {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  DQN:      {'disabled' if no_dqn else 'enabled if available'}")
    print("=" * 65)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading models...")
    try:
        xgb_model = load_xgb()
        print("  ✓ XGBoost loaded")
    except FileNotFoundError as e:
        print(f"  ✗ XGBoost not found: {e}")
        sys.exit(1)

    agent = None
    if not no_dqn:
        try:
            agent = DQNAgent.load_active()
            print("  ✓ DQN loaded")
        except FileNotFoundError:
            print("  ⚠ No active DQN — using XGBoost + Kelly only")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading finished matches...")
    matches = _load_matches(seasons, before_date=before_date)
    if not matches:
        print("  No finished matches found.")
        return {"bets_placed": 0, "synthetic": 0}

    print(f"  {len(matches)} finished matches")
    all_ids = [m["id"] for m in matches]

    print("\n  Pre-loading odds...")
    odds_cache    = _load_odds_cache(all_ids)
    print(f"  Real odds: {len(odds_cache)}/{len(matches)}")

    print("  Checking already-simulated matches...")
    simulated_ids = _load_simulated_ids(all_ids) if not dry_run else set()
    print(f"  Already done: {len(simulated_ids)}")

    # bulk_fetch() with no before_date — simulation uses per-match temporal
    # filtering (before_date = match kickoff_time) inside build_state_vector.
    # The cache carries all historical data; individual feature helpers filter
    # correctly via e[0] < before_time.
    print("  Bulk-fetching features...")
    feature_cache = bulk_fetch()
    print("  ✓ Features loaded")

    model_version = _get_model_version()
    balance       = float(_get_wallet()["balance"])
    print(f"\n  Starting balance: €{balance:.2f}")
    print(f"  Processing {len(matches)} matches...\n")

    # ── Counters ──────────────────────────────────────────────────────────────
    bets_placed = bets_won = bets_lost = 0
    pass_count  = skipped_simulated = skipped_no_state = 0
    synthetic_count  = 0
    synthetic_warned = 0
    total_pnl        = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, match in enumerate(matches):
        match_id = match["id"]
        result   = match["result"]
        home_tla = match["home"]["tla"]
        away_tla = match["away"]["tla"]

        if match_id in simulated_ids:
            skipped_simulated += 1
            continue

        # 1. XGBoost feature vector (16-dim, no lookahead)
        try:
            xgb_vec = build_state_vector(
                home_team_id = match["home_team_id"],
                away_team_id = match["away_team_id"],
                kickoff_time = match["kickoff_time"],
                before_date  = match["kickoff_time"],
                cache        = feature_cache,
            )
        except Exception as e:
            skipped_no_state += 1
            if skipped_no_state <= 5:
                print(f"  [{i+1:4d}] ✗ State vector failed ({home_tla} vs {away_tla}): {e}")
            continue

        # 2. XGBoost probabilities (NOT written back into xgb_vec)
        try:
            probs = predict_probabilities(xgb_model, xgb_vec)
        except Exception:
            skipped_no_state += 1
            continue

        # 3. Odds
        is_synthetic = match_id not in odds_cache
        if is_synthetic:
            odds_data = _synthetic_odds(probs)
            synthetic_count += 1
            if synthetic_warned < MAX_SYNTHETIC_WARNINGS:
                print(f"  [{i+1:4d}] ⚠ No odds ({home_tla} vs {away_tla}) — synthetic")
                synthetic_warned += 1
            elif synthetic_warned == MAX_SYNTHETIC_WARNINGS:
                print(f"  (further no-odds warnings suppressed...)")
                synthetic_warned += 1
        else:
            odds_data = odds_cache[match_id]

        # 4. De-vig for implied probabilities
        try:
            fair    = remove_vig(odds_data["home_odds"], odds_data["draw_odds"], odds_data["away_odds"])
            implied = {"HOME": fair["HOME"], "DRAW": fair["DRAW"], "AWAY": fair["AWAY"]}
        except Exception:
            implied = None

        # 5. DQN state (24-dim, separate from xgb_vec)
        try:
            dqn_state = build_dqn_state(
                xgb_vector     = xgb_vec,
                xgb_probs      = probs,
                implied_probs  = implied,
                wallet_balance = balance,
            )
        except Exception:
            dqn_state = None

        # 6. Kelly sizing
        bet = best_bet(
            model_probs = probs,
            home_odds   = odds_data["home_odds"],
            draw_odds   = odds_data["draw_odds"],
            away_odds   = odds_data["away_odds"],
            balance     = balance,
        )

        # 7. DQN confirmation gate
        action     = "PASS"
        confidence = 0.0

        if bet and agent and dqn_state is not None:
            try:
                q_vals     = agent.get_q_values(dqn_state)
                action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
                dqn_action = action_map.get(bet["outcome"], "PASS")
                q_bet      = q_vals.get(dqn_action, 0.0)
                q_pass     = q_vals.get("PASS", 0.0)
                confidence = agent.get_confidence(dqn_state)
                if q_bet - q_pass >= DQN_CONFIDENCE_GATE:
                    action = dqn_action
            except Exception:
                pass
        elif bet and (no_dqn or not agent):
            action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
            action     = action_map.get(bet["outcome"], "PASS")

        # 8. Save prediction (always — populates frontend)
        log_loss_val = _compute_log_loss(probs, result)
        pred_id      = None

        if not dry_run:
            try:
                pred_result = supabase.table("predictions").upsert({
                    "match_id":           match_id,
                    "model_version":      model_version,
                    "prob_home":          probs["HOME"],
                    "prob_draw":          probs["DRAW"],
                    "prob_away":          probs["AWAY"],
                    "recommended_action": action,
                    "confidence":         round(confidence, 4),
                    "was_correct":        None,
                    "log_loss":           log_loss_val,
                }, on_conflict="match_id,model_version").execute()
                pred_id = pred_result.data[0]["id"] if pred_result.data else None
            except Exception as e:
                print(f"  [{i+1:4d}] ✗ Prediction save failed: {e}")

        # 9. Place bet
        if action == "PASS" or not bet:
            pass_count += 1
            # Push PASS transition to replay buffer so DQN learns when NOT to bet
            if not dry_run and dqn_state is not None:
                _push_to_replay_buffer(match_id, dqn_state, "PASS", 0.0, 1.0)
            continue

        outcome_map = {"BET_HOME": "HOME", "BET_DRAW": "DRAW", "BET_AWAY": "AWAY"}
        bet_on  = outcome_map[action]
        outcome = "WIN" if bet_on == result else "LOSS"
        stake   = float(bet["stake"])
        odds    = float(bet["odds"])

        if outcome == "WIN":
            pnl      = round(stake * (odds - 1.0), 2)
            returned = stake + pnl
        else:
            pnl      = round(-stake, 2)
            returned = 0.0

        new_balance = round(max(0.0, balance + pnl), 2)

        if not dry_run:
            if pred_id:
                try:
                    supabase.table("predictions").update({
                        "was_correct": (outcome == "WIN"),
                    }).eq("id", pred_id).execute()
                except Exception:
                    pass

            try:
                supabase.table("bets").insert({
                    "match_id":       match_id,
                    "prediction_id":  pred_id,
                    "action":         action,
                    "stake":          round(stake, 2),
                    "odds":           round(odds, 4),
                    "balance_before": round(balance, 2),
                    "outcome":        outcome,
                    "pnl":            pnl,
                    "balance_after":  new_balance,
                    "placed_at":      match["kickoff_time"],
                    "settled_at":     match["kickoff_time"],
                }).execute()
            except Exception as e:
                print(f"  [{i+1:4d}] ✗ Bet insert failed: {e}")
                continue

            _update_wallet(new_balance, staked=stake, returned=returned)

            # Push settled bet to replay buffer for DQN training
            # This was missing entirely — DQN was training on nothing
            if dqn_state is not None:
                _push_to_replay_buffer(match_id, dqn_state, action, pnl, stake)

        balance    = new_balance
        total_pnl += pnl
        bets_placed += 1
        bets_won    += (outcome == "WIN")
        bets_lost   += (outcome == "LOSS")

        status   = "✓ WIN " if outcome == "WIN" else "✗ LOSS"
        syn_flag = "(syn)" if is_synthetic else "     "
        print(f"  [{i+1:4d}/{len(matches)}] "
              f"{home_tla:3s} vs {away_tla:3s}  [{result:4s}] "
              f"| {action:8s} @ {odds:.2f} "
              f"| €{stake:.2f} "
              f"| {status} {pnl:+.2f} "
              f"| €{balance:.2f} "
              f"| {syn_flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    win_rate = bets_won / bets_placed * 100 if bets_placed else 0.0

    print(f"\n{'='*65}")
    print(f"  {'DRY RUN — ' if dry_run else ''}Simulation complete")
    print(f"  Seasons:            {', '.join(seasons)}{date_label}")
    print(f"  Total matches:      {len(matches)}")
    print(f"  Skipped (done):     {skipped_simulated}")
    print(f"  Skipped (no state): {skipped_no_state}")
    print(f"  Bets placed:        {bets_placed}")
    print(f"  Passed (no edge):   {pass_count}")
    print(f"  Won / Lost:         {bets_won} / {bets_lost}")
    print(f"  Win rate:           {win_rate:.1f}%")
    print(f"  Total P&L:          {total_pnl:+.2f}")
    print(f"  Final balance:      €{balance:.2f}")
    print(f"  Synthetic odds:     {synthetic_count} matches")

    if synthetic_count > len(matches) * 0.3:
        print(f"\n  ⚠ >30% synthetic odds. Run fetch_historical_odds.py.")
    if dry_run:
        print(f"\n  ↑ DRY RUN — nothing written to DB.")

    return {
        "bets_placed":   bets_placed,
        "bets_won":      bets_won,
        "bets_lost":     bets_lost,
        "win_rate":      round(win_rate, 1),
        "total_pnl":     round(total_pnl, 2),
        "final_balance": round(balance, 2),
        "synthetic":     synthetic_count,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate historical bets")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--before-date", default=None, dest="before_date")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-dqn", action="store_true", dest="no_dqn")
    args = parser.parse_args()
    run(
        seasons     = args.seasons,
        dry_run     = args.dry_run,
        no_dqn      = args.no_dqn,
        before_date = args.before_date,
    )