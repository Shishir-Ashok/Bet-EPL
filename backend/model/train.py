"""
backend/model/train.py
-----------------------
DQN training orchestrator.

Overfitting fixes applied:
  1. Train/val split  — train on 2020-23, validate on 2023-24.
     The agent never trains on 2023-24 matches but is evaluated on
     them after each epoch. Real generalisation is the val_reward metric.

  2. Shuffle each epoch — matches are reshuffled before every epoch so
     the agent can't memorise the chronological sequence.

  3. Confidence gate — the agent only executes a bet if its Q-value for
     that action exceeds Q(PASS) by at least MIN_BET_CONFIDENCE. This
     forces genuine selectivity rather than "bet on everything with Q>0".

  4. Duplicate key fix — version_tag uses seconds-precision timestamp
     and _register_dqn uses upsert, so reruns never fail.

Modes:
  --mode xgboost   Train XGBoost only
  --mode dqn       Train DQN only (requires XGBoost to exist)
  --mode full      Train both

Flags:
  --test           100 matches / 3 epochs
  --clear-buffer   Wipe stale replay buffer before training
  --epochs N       Number of training epochs (default 30)
"""

import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model.features      import bulk_fetch, build_state_vector
from backend.model.xgboost_model import train as train_xgboost, load_model as load_xgb, predict_probabilities
from backend.model.dqn_agent     import DQNAgent, _register_dqn
from backend.model import replay_buffer

# ─── Config ───────────────────────────────────────────────────────────────────

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23"]   # agent trains on these
VAL_SEASON    = "2023-24"                            # agent NEVER trains on this

DQN_EPOCHS         = 30
STARTING_BALANCE   = 10.0
MIN_BET            = 0.10

# The agent only bets if Q(best_action) exceeds Q(PASS) by this amount.
# This prevents the "bet on everything because Q > 0" collapse.
# 0.5 means the agent must be half a EUR more confident in betting than passing.
MIN_BET_CONFIDENCE = 0.5

# Flat training stake — variable Kelly requires real bookmaker odds
# which don't exist in historical data
TRAINING_STAKE = 1.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def kelly_stake(prob: float, odds: float, balance: float, fraction: float = 0.25) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (prob * (b + 1) - 1) / b
    return round(min(max(0.0, kelly * fraction * balance), balance * 0.2), 2)


def get_best_odds_for_action(match_id: int, action: int) -> Optional[float]:
    if action == 3:
        return None
    col = {0: "home_odds", 1: "draw_odds", 2: "away_odds"}[action]
    try:
        result = (
            supabase.table("odds")
            .select(col)
            .eq("match_id", match_id)
            .order("fetched_at", desc=True)
            .limit(10)
            .execute()
        )
        values = [float(r[col]) for r in result.data if r.get(col)]
        return max(values) if values else None
    except Exception:
        return None


def compute_reward(action: int, result: str, stake: float, odds: float, balance: float):
    """
    Uses flat TRAINING_STAKE for the reward signal so the agent receives
    a meaningful +1/-1 range regardless of Kelly sizing or odds source.
    Balance is updated with the real Kelly stake for accurate wallet simulation.
    """
    RESULT_MAP = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    if action == 3:
        return 0.0, balance

    won         = (action == RESULT_MAP.get(result))
    train_pnl   = TRAINING_STAKE * (odds - 1.0) if won else -TRAINING_STAKE
    reward      = float(np.clip(train_pnl, -2.0, 2.0))

    real_pnl    = stake * (odds - 1.0) if won else -stake
    new_balance = round(max(0.0, balance + real_pnl), 2)
    return reward, new_balance


# ─── Match loading ────────────────────────────────────────────────────────────

def load_matches(seasons: list[str], test: bool = False) -> list:
    """Loads matches for given seasons with pagination past Supabase 1000-row cap."""
    base_query = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, result, season")
        .in_("season", seasons)
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )

    if test:
        return base_query.limit(100).execute().data

    import time
    matches, offset, page_size = [], 0, 1000
    while True:
        for attempt in range(3):
            try:
                page = base_query.range(offset, offset + page_size - 1).execute().data
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        if not page:
            break
        matches.extend(page)
        if len(page) < page_size:
            break
        offset    += page_size
        time.sleep(0.2)

    return matches


# ─── State pre-computation ────────────────────────────────────────────────────

def precompute_states(matches: list, xgb_model, feature_cache: dict) -> dict:
    """Builds all state vectors from in-memory cache. Zero DB calls."""
    print(f"\n  Building {len(matches)} state vectors from cache...")
    cache = {}
    for i, m in enumerate(matches):
        if (i + 1) % 200 == 0 or i == len(matches) - 1:
            print(f"  {i+1}/{len(matches)}", end="\r")
        try:
            state = build_state_vector(
                match_id       = m["id"],
                home_team_id   = m["home_team_id"],
                away_team_id   = m["away_team_id"],
                kickoff_time   = m["kickoff_time"],
                wallet_balance = STARTING_BALANCE,
                before_date    = m["kickoff_time"],
                cache          = feature_cache,
            )
            try:
                p = predict_probabilities(xgb_model, state)
                state[7] = float(p["HOME"])
                state[8] = float(p["DRAW"])
                state[9] = float(p["AWAY"])
            except Exception:
                pass
            cache[m["id"]] = state
        except Exception as e:
            print(f"\n  Warning: skipping match {m['id']}: {e}")
    print(f"\n  ✓ {len(cache)} state vectors ready\n")
    return cache


def get_state(cache: dict, match_id: int, balance: float) -> np.ndarray:
    state     = cache[match_id].copy()
    state[15] = float(np.clip(balance / STARTING_BALANCE, 0.0, 3.0))
    return state


# ─── Epoch ────────────────────────────────────────────────────────────────────

def run_epoch(
    agent:        DQNAgent,
    matches:      list,
    state_cache:  dict,
    epoch:        int,
    train:        bool = True,
) -> dict:
    """
    Runs one pass through a list of matches.

    train=True  → agent explores (epsilon-greedy), learns from batches,
                  matches are pre-shuffled by the caller.
    train=False → agent exploits (greedy), no learning, reports val metrics.
    """
    balance      = STARTING_BALANCE
    total_reward = 0.0
    bets_placed  = bets_won = 0
    losses_list  = []
    transitions  = []
    match_ids    = []

    for i, match in enumerate(matches):
        result   = match.get("result")
        match_id = match["id"]

        if not result or match_id not in state_cache:
            continue

        state = get_state(state_cache, match_id, balance)

        if train:
            # Epsilon-greedy with confidence gate
            if np.random.random() < agent.epsilon:
                action = np.random.randint(4)
            else:
                q_vals  = agent.get_q_values(state)
                best    = max(q_vals, key=q_vals.get)
                action  = list(q_vals.keys()).index(best)
                # Confidence gate: only bet if meaningfully better than passing
                if action != 3:
                    q_bet  = q_vals[best]
                    q_pass = q_vals["PASS"]
                    if q_bet - q_pass < MIN_BET_CONFIDENCE:
                        action = 3  # not confident enough — pass
        else:
            # Validation: always exploit, always apply confidence gate
            q_vals  = agent.get_q_values(state)
            best    = max(q_vals, key=q_vals.get)
            action  = list(q_vals.keys()).index(best)
            if action != 3:
                if q_vals[best] - q_vals["PASS"] < MIN_BET_CONFIDENCE:
                    action = 3

        odds = get_best_odds_for_action(match_id, action)
        if odds is None:
            probs = [state[7], state[8], state[9]]
            p     = probs[action] if action < 3 else 1.0
            odds  = round(1.0 / max(p, 0.05), 2)

        stake = 0.0
        if action < 3:
            p     = [state[7], state[8], state[9]][action]
            stake = kelly_stake(p, odds, balance)
            stake = max(stake, MIN_BET) if stake > 0 else 0.0

        reward, new_balance = compute_reward(action, result, stake, odds, balance)

        if train:
            is_last    = (i == len(matches) - 1)
            done       = is_last or new_balance <= 0
            next_match = matches[i + 1] if not done else None
            next_state = (
                get_state(state_cache, next_match["id"], new_balance)
                if next_match and next_match["id"] in state_cache
                else None
            )

            replay_buffer.push_memory(state, action, reward, next_state, done)
            transitions.append({"state": state, "action": action, "reward": reward,
                                 "next_state": next_state, "done": done})
            match_ids.append(match_id)

            if replay_buffer.is_ready():
                batch = replay_buffer.sample()
                if batch:
                    losses_list.append(agent.train_step(batch))

        total_reward += reward
        if action < 3:
            bets_placed += 1
            if (action == 0 and result == "HOME") or \
               (action == 1 and result == "DRAW") or \
               (action == 2 and result == "AWAY"):
                bets_won += 1

        balance = new_balance
        if balance <= 0 and train:
            print(f"  ⚠ Bankrupt at match {i+1}/{len(matches)}")
            break

    if train:
        agent.decay_epsilon()

    return {
        "epoch":           epoch,
        "total_reward":    round(float(total_reward), 4),
        "final_balance":   round(float(balance), 2),
        "bets_placed":     bets_placed,
        "win_rate":        round(bets_won / bets_placed, 3) if bets_placed else 0.0,
        "avg_loss":        round(float(np.mean(losses_list)), 6) if losses_list else 0.0,
        "epsilon":         round(float(agent.epsilon), 3),
        "new_transitions": transitions,
        "new_match_ids":   match_ids,
    }


# ─── Entry points ─────────────────────────────────────────────────────────────

def train_xgboost_only(test: bool = False):
    print("\n" + "=" * 55)
    print("  XGBoost outcome predictor")
    if test:
        print("  *** TEST MODE — 100 samples ***")
    print("=" * 55)

    version_tag    = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model, metrics = train_xgboost(
        seasons     = TRAIN_SEASONS,
        version_tag = version_tag,
        limit       = 100 if test else None,
    )
    print(f"\n✓ XGBoost done. Log-loss: {metrics['val_log_loss']}  Version: {version_tag}")
    return model


def train_dqn_only(epochs: int = DQN_EPOCHS, test: bool = False, clear_buffer: bool = False):
    print("\n" + "=" * 55)
    print("  DQN betting agent")
    if test:
        print("  *** TEST MODE — 100 matches, 3 epochs ***")
    print("=" * 55)

    xgb_model = load_xgb()
    n_epochs  = 3 if test else epochs

    # Load train and val matches separately
    print(f"\n  Loading training matches ({', '.join(TRAIN_SEASONS)})...")
    train_matches = load_matches(TRAIN_SEASONS, test=test)

    print(f"  Loading validation matches ({VAL_SEASON})...")
    val_matches   = load_matches([VAL_SEASON], test=test)
    print(f"  Train: {len(train_matches)}  |  Val: {len(val_matches)}")

    all_matches = train_matches + val_matches

    print("\n  Bulk fetching features (4 queries)...")
    feature_cache = bulk_fetch([m["id"] for m in all_matches])

    print("\n  Pre-computing state vectors...")
    train_cache = precompute_states(train_matches, xgb_model, feature_cache)
    val_cache   = precompute_states(val_matches,   xgb_model, feature_cache)

    if clear_buffer:
        print("  Clearing replay buffer from DB...")
        try:
            supabase.table("rl_episodes").delete().neq("id", 0).execute()
            print("  ✓ Cleared")
        except Exception as e:
            print(f"  ⚠ Clear failed: {e}")

    replay_buffer.load_from_db()

    agent       = DQNAgent()
    # Seconds-precision tag prevents duplicate key on same-minute reruns
    version_tag = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_rewards = []
    best_val_reward = float("-inf")

    print(f"\n  Training {n_epochs} epochs")
    print(f"  Train seasons: {TRAIN_SEASONS}")
    print(f"  Val season:    {VAL_SEASON}  (never trained on)")
    print(f"  Confidence gate: Q(bet) - Q(pass) >= {MIN_BET_CONFIDENCE}\n")
    print(f"  {'Epoch':>5}  {'Train reward':>13}  {'Val reward':>11}  {'Val bets':>9}  {'Val win%':>9}  {'Balance':>9}  {'ε':>6}  {'Loss':>8}")
    print("  " + "-" * 85)

    for epoch in range(1, n_epochs + 1):
        # Shuffle training matches — prevents sequence memorisation
        shuffled_train = train_matches.copy()
        random.shuffle(shuffled_train)

        train_m = run_epoch(agent, shuffled_train, train_cache, epoch, train=True)
        val_m   = run_epoch(agent, val_matches,    val_cache,   epoch, train=False)

        all_rewards.append(train_m["total_reward"])

        # Track best validation performance
        if val_m["total_reward"] > best_val_reward:
            best_val_reward = val_m["total_reward"]
            agent.save(f"{version_tag}_best")

        print(
            f"  {epoch:5d}  "
            f"{train_m['total_reward']:>+13.2f}  "
            f"{val_m['total_reward']:>+11.2f}  "
            f"{val_m['bets_placed']:>9d}  "
            f"{val_m['win_rate']:>9.1%}  "
            f"€{train_m['final_balance']:>8.2f}  "
            f"{train_m['epsilon']:>6.3f}  "
            f"{train_m['avg_loss']:>8.5f}"
        )

        replay_buffer.flush_to_db(
            match_ids       = train_m["new_match_ids"],
            episode_num     = epoch,
            new_transitions = train_m["new_transitions"],
        )

        if epoch % 10 == 0:
            agent.save(f"{version_tag}_ep{epoch}")

    save_path  = agent.save(version_tag)
    avg_reward = float(np.mean(all_rewards[-min(10, len(all_rewards)):]))

    _register_dqn(
        version_tag = version_tag,
        save_path   = save_path,
        metrics     = {
            "epochs":           int(n_epochs),
            "final_train_reward": float(train_m["total_reward"]),
            "best_val_reward":  float(best_val_reward),
            "train_seasons":    TRAIN_SEASONS,
            "val_season":       VAL_SEASON,
        },
        avg_reward = avg_reward,
    )
    replay_buffer.prune()

    print(f"\n✓ DQN done.")
    print(f"  Best val reward: {best_val_reward:+.2f}  (saved as {version_tag}_best)")
    print(f"  Final version:   {version_tag}")
    return agent


def train_full(epochs: int = DQN_EPOCHS, test: bool = False, clear_buffer: bool = False):
    print("\n" + "=" * 55)
    print("  Full training: XGBoost + DQN")
    print("=" * 55)
    train_xgboost_only(test=test)
    train_dqn_only(epochs=epochs, test=test, clear_buffer=clear_buffer)
    print("\n✓ Both models trained and registered.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PL betting bot models")
    parser.add_argument("--mode",         choices=["full", "xgboost", "dqn"], default="full")
    parser.add_argument("--epochs",       type=int, default=DQN_EPOCHS)
    parser.add_argument("--test",         action="store_true")
    parser.add_argument("--clear-buffer", action="store_true",
                        help="Wipe replay buffer before training")
    args = parser.parse_args()

    if args.mode == "full":
        train_full(epochs=args.epochs, test=args.test, clear_buffer=args.clear_buffer)
    elif args.mode == "xgboost":
        train_xgboost_only(test=args.test)
    elif args.mode == "dqn":
        train_dqn_only(epochs=args.epochs, test=args.test, clear_buffer=args.clear_buffer)