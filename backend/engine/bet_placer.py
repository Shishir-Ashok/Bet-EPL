"""
backend/engine/bet_placer.py
------------------------------
Runs before each matchday to:
  1. Find upcoming scheduled matches
  2. Build state vectors for each
  3. Run XGBoost → get outcome probabilities
  4. Run DQN agent → confirm whether to bet
  5. Apply Kelly sizing with vig removal
  6. Write bets to the `bets` table
  7. Update the wallet balance

Called by the FastAPI endpoint POST /trigger/predict
which is invoked by the fetch_prematch GitHub Actions workflow.

Decision pipeline per match:
  XGBoost says "54% home win"
       ↓
  DQN confirms "Q(BET_HOME) - Q(PASS) >= threshold → bet"
       ↓
  Kelly sizes the bet using de-vigged market odds
       ↓
  Bet written to DB with balance_before recorded
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model.xgboost_model import load_model as load_xgb, predict_probabilities
from backend.model.dqn_agent     import DQNAgent
from backend.model.features      import build_state_vector
from backend.engine.kelly        import best_bet, remove_vig

# Minimum Q-value advantage over PASS for the DQN to confirm a bet.
# If the DQN isn't sure enough, Kelly alone decides (pure XGBoost + Kelly mode).
DQN_CONFIDENCE_GATE = 0.3

# Use DQN as a filter. If False, uses XGBoost + Kelly only (simpler but less adaptive).
USE_DQN_FILTER = True


def get_wallet() -> dict:
    """Returns the current wallet state."""
    result = supabase.table("wallet").select("*").eq("id", 1).single().execute()
    return result.data


def update_wallet(new_balance: float, stake: float) -> None:
    """Deducts stake from wallet when a bet is placed."""
    current = get_wallet()
    supabase.table("wallet").update({
        "balance":      round(new_balance, 2),
        "total_staked": round(current["total_staked"] + stake, 2),
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


def get_upcoming_matches() -> list[dict]:
    """Returns scheduled matches that don't yet have an open bet."""
    scheduled = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, season, "
                "home:teams!matches_home_team_id_fkey(name, short_name), "
                "away:teams!matches_away_team_id_fkey(name, short_name)")
        .eq("status", "SCHEDULED")
        .order("kickoff_time", desc=False)
        .limit(20)
        .execute()
        .data
    )

    if not scheduled:
        return []

    # Exclude matches that already have an open (unsettled) bet
    match_ids    = [m["id"] for m in scheduled]
    existing     = (
        supabase.table("bets")
        .select("match_id")
        .in_("match_id", match_ids)
        .is_("outcome", "null")   # open bets only
        .execute()
        .data
    )
    existing_ids = {r["match_id"] for r in existing}

    return [m for m in scheduled if m["id"] not in existing_ids]


def get_best_odds(match_id: int) -> dict | None:
    """
    Returns the best available decimal odds for a match across all bookmakers.
    Returns None if no odds have been fetched yet for this match.
    """
    result = (
        supabase.table("odds")
        .select("home_odds, draw_odds, away_odds, bookmaker, fetched_at")
        .eq("match_id", match_id)
        .order("fetched_at", desc=True)
        .limit(20)
        .execute()
        .data
    )

    if not result:
        return None

    # Best odds per outcome across all bookmakers
    return {
        "home_odds": max(r["home_odds"] for r in result),
        "draw_odds": max(r["draw_odds"] for r in result),
        "away_odds": max(r["away_odds"] for r in result),
        "bookmaker": "best_available",
    }


def save_prediction(
    match_id:     int,
    probs:        dict,
    action:       str,
    confidence:   float,
    model_version: str,
) -> int:
    """Saves the model's prediction to the predictions table. Returns prediction id."""
    # Upsert so daily refreshes update the prediction rather than duplicate it
    result = supabase.table("predictions").upsert({
        "match_id":             match_id,
        "model_version":        model_version,
        "prob_home":            probs["HOME"],
        "prob_draw":            probs["DRAW"],
        "prob_away":            probs["AWAY"],
        "recommended_action":   action,
        "confidence":           confidence,
    }, on_conflict="match_id,model_version").execute()
    return result.data[0]["id"]


def place_bet(
    match_id:      int,
    prediction_id: int,
    action:        str,
    stake:         float,
    odds:          float,
    balance:       float,
) -> None:
    """Writes a bet to the bets table and deducts stake from wallet."""
    supabase.table("bets").insert({
        "match_id":       match_id,
        "prediction_id":  prediction_id,
        "action":         action,
        "stake":          stake,
        "odds":           odds,
        "balance_before": balance,
        "placed_at":      datetime.now(timezone.utc).isoformat(),
    }).execute()

    new_balance = round(balance - stake, 2)
    update_wallet(new_balance, stake)


def run() -> dict:
    """
    Main entry point — runs the full prediction and betting pipeline
    for all upcoming scheduled matches.

    Returns a summary dict for the FastAPI response.
    """
    print("=" * 55)
    print("  Bet Placer — running prediction pipeline")
    print("=" * 55)

    # Load models
    print("\n  Loading models...")
    try:
        xgb_model = load_xgb()
        print("  ✓ XGBoost loaded")
    except FileNotFoundError as e:
        return {"error": str(e), "bets_placed": 0}

    agent = None
    if USE_DQN_FILTER:
        try:
            agent = DQNAgent.load_active()
            print("  ✓ DQN loaded")
        except FileNotFoundError:
            print("  ⚠ No active DQN — using XGBoost + Kelly only")

    # Get wallet and upcoming matches
    wallet  = get_wallet()
    balance = float(wallet["balance"])
    print(f"\n  Current balance: €{balance:.2f}")

    matches = get_upcoming_matches()
    print(f"  Upcoming matches needing predictions: {len(matches)}")

    if not matches:
        print("\n  No upcoming matches found.")
        return {"bets_placed": 0, "balance": balance, "matches_processed": 0}

    # Get active model version tag for prediction records
    xgb_version = (
        supabase.table("model_versions")
        .select("version_tag")
        .eq("model_type", "xgboost")
        .eq("is_active", True)
        .limit(1)
        .execute()
        .data
    )
    model_version = xgb_version[0]["version_tag"] if xgb_version else "unknown"

    results = []

    for match in matches:
        home = match["home"]["short_name"]
        away = match["away"]["short_name"]
        print(f"\n  {home} vs {away}  ({match['kickoff_time'][:10]})")

        # ── 1. Build state vector ─────────────────────────────────────────────
        try:
            state = build_state_vector(
                match_id       = match["id"],
                home_team_id   = match["home_team_id"],
                away_team_id   = match["away_team_id"],
                kickoff_time   = match["kickoff_time"],
                wallet_balance = balance,
            )
        except Exception as e:
            print(f"    ✗ State vector failed: {e}")
            continue

        # ── 2. XGBoost probabilities ──────────────────────────────────────────
        try:
            probs = predict_probabilities(xgb_model, state)
            state[7] = float(probs["HOME"])
            state[8] = float(probs["DRAW"])
            state[9] = float(probs["AWAY"])
            print(f"    XGBoost: H {probs['HOME']:.1%}  D {probs['DRAW']:.1%}  A {probs['AWAY']:.1%}")
        except Exception as e:
            print(f"    ✗ XGBoost prediction failed: {e}")
            continue

        # ── 3. Get bookmaker odds ─────────────────────────────────────────────
        odds_data = get_best_odds(match["id"])
        if not odds_data:
            print(f"    ⚠ No odds available yet — saving prediction only")
            pred_id = save_prediction(match["id"], probs, "PASS", 0.0, model_version)
            results.append({"match": f"{home} vs {away}", "action": "PASS", "reason": "no odds"})
            continue

        fair = remove_vig(odds_data["home_odds"], odds_data["draw_odds"], odds_data["away_odds"])
        print(f"    Odds:   H {odds_data['home_odds']}  D {odds_data['draw_odds']}  A {odds_data['away_odds']}  (vig {fair['vig_pct']}%)")
        print(f"    Fair:   H {fair['HOME']:.1%}  D {fair['DRAW']:.1%}  A {fair['AWAY']:.1%}")

        # ── 4. Kelly sizing ───────────────────────────────────────────────────
        bet = best_bet(
            model_probs = probs,
            home_odds   = odds_data["home_odds"],
            draw_odds   = odds_data["draw_odds"],
            away_odds   = odds_data["away_odds"],
            balance     = balance,
        )

        # ── 5. DQN confirmation filter ────────────────────────────────────────
        action     = "PASS"
        confidence = 0.0

        if bet and agent and USE_DQN_FILTER:
            q_vals     = agent.get_q_values(state)
            action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
            dqn_action = action_map.get(bet["outcome"], "PASS")
            q_bet      = q_vals.get(dqn_action, 0.0)
            q_pass     = q_vals.get("PASS", 0.0)
            confidence = agent.get_confidence(state)

            if q_bet - q_pass >= DQN_CONFIDENCE_GATE:
                action = dqn_action
                print(f"    DQN:    Q({dqn_action})={q_bet:.3f} Q(PASS)={q_pass:.3f} → confirmed")
            else:
                print(f"    DQN:    Q({dqn_action})={q_bet:.3f} Q(PASS)={q_pass:.3f} → below gate, passing")
                bet = None

        elif bet and not USE_DQN_FILTER:
            action_map = {"HOME": "BET_HOME", "DRAW": "BET_DRAW", "AWAY": "BET_AWAY"}
            action = action_map.get(bet["outcome"], "PASS")

        # ── 6. Save prediction ────────────────────────────────────────────────
        pred_id = save_prediction(match["id"], probs, action, confidence, model_version)

        # ── 7. Place bet if confirmed ─────────────────────────────────────────
        if bet and action != "PASS":
            if balance < bet["stake"]:
                print(f"    ⚠ Insufficient balance (€{balance:.2f} < €{bet['stake']:.2f})")
                results.append({"match": f"{home} vs {away}", "action": "PASS", "reason": "insufficient balance"})
                continue

            place_bet(
                match_id      = match["id"],
                prediction_id = pred_id,
                action        = action,
                stake         = bet["stake"],
                odds          = bet["odds"],
                balance       = balance,
            )
            balance = round(balance - bet["stake"], 2)

            print(f"    ✓ BET {action}: €{bet['stake']:.2f} @ {bet['odds']}  edge {bet['edge']:.1%}")
            print(f"      {bet['reasoning']}")
            results.append({
                "match":  f"{home} vs {away}",
                "action": action,
                "stake":  bet["stake"],
                "odds":   bet["odds"],
                "edge":   bet["edge"],
            })
        else:
            print(f"    → PASS")
            results.append({"match": f"{home} vs {away}", "action": "PASS"})

    print(f"\n{'='*55}")
    bets_placed = sum(1 for r in results if r["action"] != "PASS")
    print(f"  Done. Bets placed: {bets_placed}/{len(matches)}")
    print(f"  New balance: €{balance:.2f}")

    return {
        "bets_placed":        bets_placed,
        "matches_processed":  len(matches),
        "balance":            balance,
        "results":            results,
    }


if __name__ == "__main__":
    run()