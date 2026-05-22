"""
scripts/backtest_season.py
--------------------------
Replays the current season chronologically, month by month.

For each month:
  1. For every finished match in that month — simulate what the model
     would have predicted and bet AT kickoff time (no lookahead).
  2. Settle all bets from that month (backdated to day after kickoff).
  3. On the 1st of the following month — retrain XGBoost (all seasons)
     and DQN (warm start, incremental).

After the backtest, the system transitions to live mode automatically.
Scheduled workflows (fetch_prematch, ingest_results, monthly_retrain)
continue from where the backtest leaves off.

Usage:
    python scripts/backtest_season.py
    python scripts/backtest_season.py --epochs 20
    python scripts/backtest_season.py --xgb-only   # just retrain XGBoost, skip DQN
"""

import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.db import supabase
from backend.model.xgboost_model import load_model as load_xgb, predict_probabilities
from backend.model.dqn_agent import DQNAgent
from backend.model.features import build_state_vector, build_dqn_state
from backend.engine.kelly import best_bet, remove_vig
from backend.engine.config import DQN_CONFIDENCE_GATE
from backend.engine import settle_bets as settler
from backend.model.train import (
    train_xgboost_only,
    train_dqn_only,
    CURRENT_SEASON,
    TRAIN_SEASONS,
    VAL_SEASON,
)

USE_DQN_FILTER   = True
STARTING_BALANCE = 100.0


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_current_season_matches() -> list[dict]:
    """All finished current-season matches, oldest first. No row limit."""
    rows, offset, page_size = [], 0, 1000
    while True:
        page = (
            supabase.table("matches")
            .select(
                "id, home_team_id, away_team_id, kickoff_time, result, season, "
                "home:teams!matches_home_team_id_fkey(name, short_name), "
                "away:teams!matches_away_team_id_fkey(name, short_name)"
            )
            .eq("season", CURRENT_SEASON)
            .eq("status", "FINISHED")
            .not_.is_("result", "null")
            .order("kickoff_time", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def group_by_month(matches: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for m in matches:
        groups[m["kickoff_time"][:7]].append(m)   # "2025-08", "2025-09", ...
    return dict(sorted(groups.items()))


def get_wallet() -> dict:
    return supabase.table("wallet").select("*").eq("id", 1).single().execute().data


def update_wallet(new_balance: float, stake: float) -> None:
    current = get_wallet()
    supabase.table("wallet").update({
        "balance":      round(new_balance, 2),
        "total_staked": round(float(current["total_staked"]) + stake, 2),
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


def get_active_model_version(model_type: str) -> str:
    result = (
        supabase.table("model_versions")
        .select("version_tag")
        .eq("model_type", model_type)
        .eq("is_active", True)
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return result[0]["version_tag"] if result else "unknown"


def get_odds(match_id: int) -> dict | None:
    result = (
        supabase.table("odds")
        .select("home_odds, draw_odds, away_odds, bookmaker")
        .eq("match_id", match_id)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not result:
        return None
    r = result[0]
    if not all([r.get("home_odds"), r.get("draw_odds"), r.get("away_odds")]):
        return None
    return r


# ─── Prediction + bet simulation ──────────────────────────────────────────────

def simulate_match(
    match:         dict,
    xgb_model,
    agent:         DQNAgent | None,
    balance:       float,
    model_version: str,
) -> dict:
    """
    Simulates bet_placer for a single match as if we were at kickoff time.
    Uses only data that was available before the match kicked off.
    Timestamps bets to kickoff so frontend history is accurate.
    """
    home     = match["home"]["short_name"]
    away     = match["away"]["short_name"]
    match_id = match["id"]
    kickoff  = match["kickoff_time"]

    # Idempotent — skip if already predicted with this model version
    existing = (
        supabase.table("predictions")
        .select("id")
        .eq("match_id", match_id)
        .eq("model_version", model_version)
        .limit(1)
        .execute()
        .data
    )
    if existing:
        return {"match": f"{home} vs {away}", "action": "SKIP"}

    # 1. XGBoost state (before_date = kickoff → no lookahead)
    try:
        xgb_vec = build_state_vector(
            home_team_id = match["home_team_id"],
            away_team_id = match["away_team_id"],
            kickoff_time = kickoff,
            before_date  = kickoff,
        )
    except Exception as e:
        print(f"    ✗ State error: {e}")
        return {"match": f"{home} vs {away}", "action": "PASS", "reason": "state error"}

    # 2. XGBoost probabilities
    try:
        probs = predict_probabilities(xgb_model, xgb_vec)
    except Exception as e:
        print(f"    ✗ XGBoost error: {e}")
        return {"match": f"{home} vs {away}", "action": "PASS", "reason": "xgb error"}

    # 3. Odds + de-vig
    odds_data = get_odds(match_id)
    implied   = None
    if odds_data:
        fair    = remove_vig(odds_data["home_odds"], odds_data["draw_odds"], odds_data["away_odds"])
        implied = {"HOME": fair["HOME"], "DRAW": fair["DRAW"], "AWAY": fair["AWAY"]}

    # 4. Full DQN state (24-dim, wallet-aware)
    dqn_state = build_dqn_state(
        xgb_vector     = xgb_vec,
        xgb_probs      = probs,
        implied_probs  = implied,
        wallet_balance = balance,
    )

    # 5. Kelly sizing (requires real odds)
    bet    = None
    action = "PASS"
    confidence = 0.0

    if odds_data:
        bet = best_bet(
            model_probs = probs,
            home_odds   = odds_data["home_odds"],
            draw_odds   = odds_data["draw_odds"],
            away_odds   = odds_data["away_odds"],
            balance     = balance,
        )

    # 6. DQN confirmation
    if bet and agent and USE_DQN_FILTER:
        q_vals     = agent.get_q_values(dqn_state)
        action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
        dqn_action = action_map.get(bet["outcome"], "PASS")
        q_bet      = q_vals.get(dqn_action, 0.0)
        q_pass     = q_vals.get("PASS", 0.0)
        confidence = agent.get_confidence(dqn_state)

        if q_bet - q_pass >= DQN_CONFIDENCE_GATE:
            action = dqn_action
        else:
            bet = None

    elif bet and not USE_DQN_FILTER:
        action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
        action = action_map.get(bet["outcome"], "PASS")

    # 7. Save prediction
    pred = supabase.table("predictions").upsert({
        "match_id":           match_id,
        "model_version":      model_version,
        "prob_home":          probs["HOME"],
        "prob_draw":          probs["DRAW"],
        "prob_away":          probs["AWAY"],
        "recommended_action": action,
        "confidence":         confidence,
    }, on_conflict="match_id,model_version").execute()
    pred_id = pred.data[0]["id"]

    # 8. Place bet — backdated to kickoff for accurate frontend timeline
    if bet and action != "PASS":
        if balance < bet["stake"]:
            print(f"    ⚠ Insufficient balance")
            return {"match": f"{home} vs {away}", "action": "PASS", "reason": "insufficient balance"}

        supabase.table("bets").insert({
            "match_id":       match_id,
            "prediction_id":  pred_id,
            "action":         action,
            "stake":          bet["stake"],
            "odds":           bet["odds"],
            "balance_before": balance,
            "placed_at":      kickoff,   # backdated — correct timeline on frontend
        }).execute()

        update_wallet(round(balance - bet["stake"], 2), bet["stake"])
        print(f"    ✓ {action}: €{bet['stake']:.2f} @ {bet['odds']}  edge {bet['edge']:.1%}")
        return {
            "match":  f"{home} vs {away}",
            "action": action,
            "stake":  bet["stake"],
            "odds":   bet["odds"],
            "edge":   bet["edge"],
        }

    return {"match": f"{home} vs {away}", "action": "PASS"}


# ─── Settlement ───────────────────────────────────────────────────────────────

def settle_month(month_matches: list[dict]) -> float:
    """
    Settles all open bets from this month's matches.
    Backdates settled_at to the day after kickoff so daily_pnl view
    shows the correct historical P&L spread on the frontend.
    """
    match_ids = [m["id"] for m in month_matches]
    # Build a kickoff lookup for backdating
    kickoff_by_id = {m["id"]: m["kickoff_time"] for m in month_matches}

    open_bets = (
        supabase.table("bets")
        .select(
            "id, match_id, action, stake, odds, balance_before, "
            "matches(status, result, home_goals, away_goals, "
            "home:teams!matches_home_team_id_fkey(short_name), "
            "away:teams!matches_away_team_id_fkey(short_name))"
        )
        .in_("match_id", match_ids)
        .is_("outcome", "null")
        .execute()
        .data
    )

    if not open_bets:
        return 0.0

    total_pnl = 0.0
    for bet in open_bets:
        match   = bet.get("matches", {})
        result  = match.get("result")
        outcome = settler.determine_outcome(bet["action"], result)

        if not outcome:
            continue

        stake = float(bet["stake"])
        odds  = float(bet["odds"])
        pnl   = round(stake * (odds - 1.0), 2) if outcome == "WIN" else round(-stake, 2)
        returned      = stake + pnl if outcome == "WIN" else 0.0
        balance_after = round(float(bet["balance_before"]) + pnl, 2)

        # Backdate to day after kickoff → correct daily_pnl spread on frontend
        kickoff_str = kickoff_by_id.get(bet["match_id"], "")
        if kickoff_str:
            kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
            settled_ts = (kickoff_dt + timedelta(days=1)).isoformat()
        else:
            settled_ts = None

        settler.settle_bet(bet["id"], outcome, pnl, balance_after, settled_at=settled_ts)
        settler.update_wallet_after_settlement(pnl, returned)
        settler.update_prediction_accuracy(bet["match_id"], result)
        settler.store_rl_transition(bet, pnl)

        total_pnl += pnl
        home = match.get("home", {}).get("short_name", "?")
        away = match.get("away", {}).get("short_name", "?")
        symbol = "✓" if outcome == "WIN" else "✗"
        print(f"    {symbol} {home} vs {away}  [{result}]  "
              f"{outcome}  {'+'if pnl>=0 else ''}€{pnl:.2f}")

    return round(total_pnl, 2)


# ─── Monthly retrain ──────────────────────────────────────────────────────────

def run_monthly_retrain(month_key: str, epochs: int, xgb_only: bool = False) -> None:
    """
    Retrains after a month closes:
    - XGBoost: ALL finished matches across all seasons including current
    - DQN: warm start incremental (learns from newly settled live bets in buffer)
    """
    print(f"\n  ── Monthly retrain after {month_key} ──")
    train_xgboost_only(include_current_season=True)   # every game, no limit

    if not xgb_only:
        train_dqn_only(epochs=epochs, incremental=True, warm_start=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(epochs: int = 30, xgb_only: bool = False) -> None:
    print("=" * 60)
    print(f"  Current Season Backtest — {CURRENT_SEASON}")
    print("=" * 60)

    # ── Phase 0: Initial full training ───────────────────────────────────────
    print(f"\n[Phase 0] Initial training — {', '.join(TRAIN_SEASONS + [VAL_SEASON])}")
    print("  (All historical seasons, no game limit)\n")
    train_xgboost_only()                                          # no limit
    if not xgb_only:
        train_dqn_only(epochs=epochs, clear_buffer=True, warm_start=False)

    # ── Phase 1: Load + group current season ─────────────────────────────────
    print(f"\n[Phase 1] Loading {CURRENT_SEASON} finished matches...")
    all_matches = load_current_season_matches()
    print(f"  Found {len(all_matches)} finished matches")

    if not all_matches:
        print("  No finished current-season matches. Backtest complete.")
        return

    by_month = group_by_month(all_matches)
    months   = sorted(by_month.keys())
    print(f"  Months: {', '.join(months)}")

    # ── Phase 2: Simulate month by month ─────────────────────────────────────
    total_bets = 0
    total_pnl  = 0.0

    for i, month_key in enumerate(months):
        month_matches = by_month[month_key]
        print(f"\n{'─'*60}")
        print(f"  {month_key}  ({len(month_matches)} matches)")
        print(f"{'─'*60}")

        # Load active models for this month
        try:
            xgb_model = load_xgb()
        except FileNotFoundError as e:
            print(f"  ✗ No XGBoost model: {e}")
            continue

        agent = None
        if USE_DQN_FILTER and not xgb_only:
            try:
                agent = DQNAgent.load_active()
            except FileNotFoundError:
                print("  ⚠ No active DQN — using XGBoost + Kelly only")

        xgb_version = get_active_model_version("xgboost")
        wallet      = get_wallet()
        balance     = float(wallet["balance"])

        print(f"\n  Predictions & Bets  (balance: €{balance:.2f})")
        month_bets = 0

        for match in sorted(month_matches, key=lambda m: m["kickoff_time"]):
            home = match["home"]["short_name"]
            away = match["away"]["short_name"]
            print(f"\n  {match['kickoff_time'][:10]}  {home} vs {away}")

            result = simulate_match(match, xgb_model, agent, balance, xgb_version)

            if result["action"] not in ("PASS", "SKIP", "PASS"):
                if result.get("stake"):
                    balance = round(balance - result["stake"], 2)
                    month_bets += 1

        # Settle bets
        print(f"\n  Settlement")
        month_pnl = settle_month(month_matches)
        total_bets += month_bets
        total_pnl  += month_pnl

        wallet  = get_wallet()
        balance = float(wallet["balance"])
        print(f"\n  {month_key}: {month_bets} bets | "
              f"P&L {'+'if month_pnl>=0 else ''}€{month_pnl:.2f} | "
              f"Balance €{balance:.2f}")

        # Retrain at end of each month (simulates "1st of next month" trigger)
        run_monthly_retrain(month_key, epochs=epochs, xgb_only=xgb_only)

    # ── Summary ───────────────────────────────────────────────────────────────
    wallet = get_wallet()
    print(f"\n{'='*60}")
    print(f"  Backtest complete — {CURRENT_SEASON}")
    print(f"  Months simulated:  {len(months)}")
    print(f"  Total bets placed: {total_bets}")
    print(f"  Total P&L:         {'+'if total_pnl>=0 else ''}€{total_pnl:.2f}")
    print(f"  Final balance:     €{float(wallet['balance']):.2f}")
    print(f"  ROI:               {((float(wallet['total_returned']) - float(wallet['total_staked'])) / max(float(wallet['total_staked']), 1)) * 100:.1f}%")
    print(f"{'='*60}")
    print(f"\n  System is now in live mode.")
    print(f"  Scheduled workflows will continue from this point forward.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest current season")
    parser.add_argument("--epochs",   type=int,  default=30,    help="DQN training epochs per monthly retrain")
    parser.add_argument("--xgb-only", action="store_true",      help="Skip DQN training (XGBoost + Kelly only)")
    args = parser.parse_args()
    run(epochs=args.epochs, xgb_only=args.xgb_only)