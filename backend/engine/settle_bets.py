"""
backend/engine/settle_bets.py
------------------------------
Runs the morning after each matchday to:
  1. Find all open (unsettled) bets
  2. Check if the match result is now available
  3. Determine WIN or LOSS
  4. Calculate P&L
  5. Update the bet record (outcome, pnl, balance_after, settled_at)
  6. Update the wallet (balance, total_returned)
  7. Store the RL transition in the replay buffer for future DQN training

This is the feedback loop that closes the learning cycle:
  Place bet → Match played → Settle → Store reward → Retrain DQN

Called by the FastAPI endpoint POST /trigger/settle
which is invoked by the ingest_results GitHub Actions workflow.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model import replay_buffer
from backend.model.features import build_state_vector

# Reward normalisation for RL replay buffer.
# We store pnl / STARTING_BALANCE so rewards are in a consistent range
# regardless of the actual wallet size at time of bet.
STARTING_BALANCE = 100.0


def get_open_bets() -> list[dict]:
    """Returns all bets that have been placed but not yet settled."""
    result = (
        supabase.table("bets")
        .select(
            "id, match_id, action, stake, odds, balance_before, placed_at, "
            "matches(status, result, home_goals, away_goals, "
            "home:teams!matches_home_team_id_fkey(short_name), "
            "away:teams!matches_away_team_id_fkey(short_name))"
        )
        .is_("outcome", "null")
        .execute()
        .data
    )
    return result or []


def determine_outcome(action: str, result: str) -> str | None:
    """
    Returns "WIN" or "LOSS" based on the bet action and match result.
    Returns None if the match isn't finished yet.
    """
    ACTION_TO_OUTCOME = {
        "BET_HOME": "HOME",
        "BET_DRAW": "DRAW",
        "BET_AWAY": "AWAY",
    }
    bet_on = ACTION_TO_OUTCOME.get(action)
    if not bet_on or not result:
        return None
    return "WIN" if bet_on == result else "LOSS"


def get_wallet() -> dict:
    return supabase.table("wallet").select("*").eq("id", 1).single().execute().data


def settle_bet(
    bet_id:      str,
    outcome:     str,
    pnl:         float,
    balance_after: float,
) -> None:
    """Updates a bet record with settlement information."""
    supabase.table("bets").update({
        "outcome":       outcome,
        "pnl":           round(pnl, 2),
        "balance_after": round(balance_after, 2),
        "settled_at":    datetime.now(timezone.utc).isoformat(),
    }).eq("id", bet_id).execute()


def update_wallet_after_settlement(pnl: float, returned: float) -> None:
    """
    Updates wallet balance and total_returned after a bet settles.
    total_returned tracks cumulative returns from winning bets only.
    """
    wallet      = get_wallet()
    new_balance = round(float(wallet["balance"]) + pnl, 2)
    new_returned = round(float(wallet["total_returned"]) + max(returned, 0), 2)

    supabase.table("wallet").update({
        "balance":        max(new_balance, 0.0),
        "total_returned": new_returned,
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


def update_prediction_accuracy(match_id: int, result: str) -> None:
    """
    Marks predictions as correct or incorrect after the match finishes.
    Used for model evaluation tracking.
    """
    predictions = (
        supabase.table("predictions")
        .select("id, recommended_action, prob_home, prob_draw, prob_away")
        .eq("match_id", match_id)
        .execute()
        .data
    )

    for pred in predictions:
        action = pred.get("recommended_action", "PASS")
        ACTION_TO_OUTCOME = {
            "BET_HOME": "HOME", "BET_DRAW": "DRAW", "BET_AWAY": "AWAY"
        }
        bet_on      = ACTION_TO_OUTCOME.get(action)
        was_correct = (bet_on == result) if bet_on else None

        # Calculate log-loss for this prediction
        prob_map  = {"HOME": pred["prob_home"], "DRAW": pred["prob_draw"], "AWAY": pred["prob_away"]}
        true_prob = prob_map.get(result, 0.0001)
        import math
        log_loss  = round(-math.log(max(float(true_prob), 1e-7)), 6)

        supabase.table("predictions").update({
            "was_correct": was_correct,
            "log_loss":    log_loss,
        }).eq("id", pred["id"]).execute()


def store_rl_transition(bet: dict, outcome: str, pnl: float) -> None:
    """
    Stores the settled bet as a replay buffer transition.

    Reward is normalised to match the training reward scale:
      training uses TRAINING_STAKE=1.0, rewards clipped to [-2, +2]
      live rewards must use the same normalisation so the DQN learns
      from real transitions at the same magnitude.
    """
    match = bet.get("matches", {})

    # Resolve team IDs from the match join — do not pass None
    home_team_id = None
    away_team_id = None
    try:
        match_row = (
            supabase.table("matches")
            .select("home_team_id, away_team_id")
            .eq("id", bet["match_id"])
            .single()
            .execute()
        )
        home_team_id = match_row.data["home_team_id"]
        away_team_id = match_row.data["away_team_id"]
    except Exception:
        return   # can't build state without team IDs

    try:
        state = build_state_vector(
            match_id       = bet["match_id"],
            home_team_id   = home_team_id,
            away_team_id   = away_team_id,
            kickoff_time   = "",
            wallet_balance = float(bet["balance_before"]),
        )
    except Exception:
        return

    ACTION_MAP = {"BET_HOME": 0, "BET_DRAW": 1, "BET_AWAY": 2, "PASS": 3}
    action     = ACTION_MAP.get(bet["action"], 3)

    # Normalise reward to match training scale (TRAINING_STAKE=1.0, clipped ±2)
    stake = float(bet["stake"])
    if stake > 0:
        normalised_pnl = pnl / stake   # convert to per-unit-staked (same as TRAINING_STAKE)
    else:
        normalised_pnl = 0.0
    reward = float(np.clip(normalised_pnl, -2.0, 2.0))

    replay_buffer.push(
        match_id    = bet["match_id"],
        state       = state,
        action      = action,
        reward      = reward,
        next_state  = None,
        done        = False,
    )


def run() -> dict:
    """
    Main entry point — settles all open bets that have a finished result.

    Returns a summary dict for the FastAPI response.
    """
    print("=" * 55)
    print("  Bet Settler — settling open bets")
    print("=" * 55)

    open_bets = get_open_bets()
    print(f"\n  Open bets found: {len(open_bets)}")

    if not open_bets:
        wallet = get_wallet()
        return {
            "settled": 0,
            "pending": 0,
            "balance": float(wallet["balance"]),
        }

    settled_count  = 0
    pending_count  = 0
    total_pnl      = 0.0
    results        = []

    for bet in open_bets:
        match = bet.get("matches", {})

        # Skip if match isn't finished yet
        if match.get("status") != "FINISHED":
            pending_count += 1
            continue

        result  = match.get("result")
        outcome = determine_outcome(bet["action"], result)

        if not outcome:
            pending_count += 1
            continue

        # Calculate P&L
        stake = float(bet["stake"])
        odds  = float(bet["odds"])

        if outcome == "WIN":
            pnl      = round(stake * (odds - 1.0), 2)
            returned = stake + pnl
        else:
            pnl      = round(-stake, 2)
            returned = 0.0

        balance_after = round(float(bet["balance_before"]) + pnl, 2)

        home = match.get("home", {}).get("short_name", "?")
        away = match.get("away", {}).get("short_name", "?")

        print(f"\n  {home} vs {away}  [{result}]")
        print(f"    Bet: {bet['action']} @ {odds}  Stake: €{stake:.2f}")
        print(f"    Outcome: {outcome}  P&L: {'+'if pnl>=0 else ''}€{pnl:.2f}")

        # Update bet record
        settle_bet(
            bet_id        = bet["id"],
            outcome       = outcome,
            pnl           = pnl,
            balance_after = balance_after,
        )

        # Update wallet
        update_wallet_after_settlement(pnl, returned)

        # Mark prediction accuracy
        update_prediction_accuracy(bet["match_id"], result)

        # Store RL transition for future retraining
        store_rl_transition(bet, outcome, pnl)

        total_pnl     += pnl
        settled_count += 1
        results.append({
            "match":   f"{home} vs {away}",
            "action":  bet["action"],
            "outcome": outcome,
            "pnl":     pnl,
        })

    wallet      = get_wallet()
    new_balance = float(wallet["balance"])

    print(f"\n{'='*55}")
    print(f"  Settled: {settled_count}  Pending: {pending_count}")
    print(f"  Session P&L: {'+'if total_pnl>=0 else ''}€{total_pnl:.2f}")
    print(f"  New balance: €{new_balance:.2f}")

    return {
        "settled":     settled_count,
        "pending":     pending_count,
        "session_pnl": round(total_pnl, 2),
        "balance":     new_balance,
        "results":     results,
    }


if __name__ == "__main__":
    run()