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
import json
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

from backend.data_pipeline.season_utils import (
    get_historical_seasons, get_current_season, get_all_seasons
)

def _get_train_val_seasons() -> tuple[list[str], str]:
    """
    Train on all completed seasons except the last (which is validation).
    In April 2026 (2025-26 season in progress):
      completed = [2020-21, 2021-22, 2022-23, 2023-24, 2024-25]
      val   = 2024-25   (most recently completed — has full labels)
      train = [2020-21 … 2023-24]

    The CURRENT season (2025-26) is included in DQN incremental training
    via the replay buffer (live settled bets), not in XGBoost batch training
    since most of its matches may not yet be finished.
    """
    completed = get_historical_seasons(from_year=2020)
    if len(completed) < 2:
        raise ValueError("Need at least 2 completed seasons.")
    val_season    = completed[-1]["label"]
    train_seasons = [s["label"] for s in completed[:-1]]
    return train_seasons, val_season


TRAIN_SEASONS, VAL_SEASON = _get_train_val_seasons()
CURRENT_SEASON = get_current_season()["label"]   # "2025-26" — used for incremental DQN

DQN_EPOCHS         = 30
STARTING_BALANCE   = 100.0
MIN_BET            = 0.10
from backend.engine.config import DQN_CONFIDENCE_GATE as MIN_BET_CONFIDENCE
TRAINING_STAKE     = 1.0



def get_last_retrain_episode() -> int:
    """
    Returns the episode_num of the most recent DQN training run,
    used to fetch only NEW replay buffer entries since then.
    Returns 0 if no prior training exists (full retrain needed).
    """
    result = (
        supabase.table("model_versions")
        .select("notes")
        .eq("model_type", "dqn")
        .eq("is_active", True)
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return 0
    try:
        notes = json.loads(result.data[0]["notes"] or "{}")
        return int(notes.get("last_episode_num", 0))
    except Exception:
        return 0
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
                state[16] = float(p["HOME"])
                state[17] = float(p["DRAW"])
                state[18] = float(p["AWAY"])
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
            probs = [state[16], state[17], state[18]]
            p     = probs[action] if action < 3 else 1.0
            odds  = round(1.0 / max(p, 0.05), 2)

        stake = 0.0
        if action < 3:
            p     = [state[16], state[17], state[18]][action]
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

def train_xgboost_only(test: bool = False, recent_only: bool = False):
    """
    Trains XGBoost on historical data.

    recent_only=True: trains on the last 2 completed seasons only.
    Useful for a faster mid-season retrain that emphasises recent patterns.
    Full retrain (recent_only=False) is recommended at season start.
    """
    print("\n" + "=" * 55)
    print("  XGBoost outcome predictor")
    if recent_only:
        completed = get_historical_seasons(from_year=2020)
        # Take last 2 completed seasons as training data
        seasons = [s["label"] for s in completed[-2:]]
        print(f"  Mode: recent-only ({', '.join(seasons)})")
    else:
        seasons = TRAIN_SEASONS
        print(f"  Mode: full ({', '.join(seasons)})")
    if test:
        print("  *** TEST MODE — 100 samples ***")
    print("=" * 55)

    version_tag    = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model, metrics = train_xgboost(
        seasons     = seasons,
        version_tag = version_tag,
        limit       = 100 if test else None,
    )
    print(f"\n✓ XGBoost done. Log-loss: {metrics['val_log_loss']}  Version: {version_tag}")
    return model


def train_dqn_only(
    epochs:        int  = DQN_EPOCHS,
    test:          bool = False,
    clear_buffer:  bool = False,
    incremental:   bool = False,   # NEW: only train on new replay buffer entries
    warm_start:    bool = False,   # NEW: continue from existing DQN checkpoint
):
    """
    Train the DQN agent.

    incremental=True:
      Only processes matches from the current season and new rl_episodes
      entries added since the last retrain. The agent picks up from where
      it left off — no need to re-run all historical data every Monday.
      Use this for the weekly_retrain.yml workflow.

    warm_start=True:
      Loads the active DQN checkpoint before training instead of starting
      fresh. Should always be True when incremental=True.

    Full retrain (default):
      Loads all historical seasons + full replay buffer. Use this when
      STATE_DIM changes, parameters change, or at start of new season.
    """
    print("\n" + "=" * 55)
    print("  DQN betting agent")
    mode_label = "incremental" if incremental else "full retrain"
    print(f"  Mode: {mode_label}{'  [warm start]' if warm_start else ''}")
    if test:
        print("  *** TEST MODE ***")
    print("=" * 55)

    xgb_model = load_xgb()
    n_epochs  = 3 if test else epochs

    if incremental:
        # Only use current season's finished matches + val season
        # Most of the learning comes from the replay buffer (live bets)
        print(f"\n  Loading current season matches ({CURRENT_SEASON})...")
        current_matches = load_matches([CURRENT_SEASON], test=test)
        print(f"  Loading val season matches ({VAL_SEASON})...")
        val_matches = load_matches([VAL_SEASON], test=test)
        train_matches = current_matches
        print(f"  Incremental train: {len(train_matches)} | Val: {len(val_matches)}")
    else:
        print(f"\n  Loading training matches ({', '.join(TRAIN_SEASONS)})...")
        train_matches = load_matches(TRAIN_SEASONS, test=test)
        print(f"  Loading val matches ({VAL_SEASON})...")
        val_matches = load_matches([VAL_SEASON], test=test)
        print(f"  Train: {len(train_matches)} | Val: {len(val_matches)}")

    all_matches   = train_matches + val_matches
    feature_cache = bulk_fetch([m["id"] for m in all_matches])
    train_cache   = precompute_states(train_matches, xgb_model, feature_cache)
    val_cache     = precompute_states(val_matches, xgb_model, feature_cache)

    if clear_buffer:
        print("  Clearing replay buffer from DB...")
        supabase.table("rl_episodes").delete().neq("id", 0).execute()
        print("  ✓ Cleared")

    n_loaded = replay_buffer.load_from_db()

    if warm_start:
        try:
            agent = DQNAgent.load_active()
            print(f"  ✓ Warm start: loaded active DQN (ε={agent.epsilon:.3f})")
            # Don't reset epsilon fully — decay slightly to signal new data phase
            agent.epsilon = max(agent.epsilon, 0.20)
        except FileNotFoundError:
            print("  ⚠ No active DQN found — starting fresh")
            agent = DQNAgent()
    else:
        agent = DQNAgent()

    version_tag     = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_rewards     = []
    best_val_reward = float("-inf")

    # Track highest episode_num in buffer so we can record it in model notes
    last_ep = (
        supabase.table("rl_episodes")
        .select("episode_num")
        .order("id", desc=True)
        .limit(1)
        .execute()
        .data
    )
    last_episode_num = int(last_ep[0]["episode_num"] or 0) if last_ep else 0

    print(f"\n  Replay buffer: {n_loaded} transitions")
    print(f"  Training {n_epochs} epochs | Current season in training: {CURRENT_SEASON}")
    print(f"  {'Epoch':>5}  {'Train reward':>13}  {'Val reward':>11}  "
          f"{'Val bets':>9}  {'Val win%':>9}  {'Balance':>9}  {'ε':>6}  {'Loss':>8}")
    print("  " + "-" * 85)

    for epoch in range(1, n_epochs + 1):
        shuffled_train = train_matches.copy()
        random.shuffle(shuffled_train)

        train_m = run_epoch(agent, shuffled_train, train_cache, epoch, train=True)
        val_m   = run_epoch(agent, val_matches,    val_cache,   epoch, train=False)

        all_rewards.append(train_m["total_reward"])

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
            "epochs":               int(n_epochs),
            "mode":                 mode_label,
            "final_train_reward":   float(train_m["total_reward"]),
            "best_val_reward":      float(best_val_reward),
            "train_seasons":        [CURRENT_SEASON] if incremental else TRAIN_SEASONS,
            "val_season":           VAL_SEASON,
            "current_season":       CURRENT_SEASON,
            "last_episode_num":     last_episode_num,   # for next incremental run
            "replay_buffer_size":   n_loaded,
        },
        avg_reward = avg_reward,
    )
    replay_buffer.prune()

    print(f"\n✓ DQN done.")
    print(f"  Best val reward: {best_val_reward:+.2f}")
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
    parser.add_argument("--mode",          choices=["full", "xgboost", "dqn"], default="full")
    parser.add_argument("--epochs",        type=int, default=DQN_EPOCHS)
    parser.add_argument("--test",          action="store_true")
    parser.add_argument("--clear-buffer",  action="store_true")
    parser.add_argument("--incremental",   action="store_true",
                        help="DQN: train on current season + replay buffer only (weekly mode)")
    parser.add_argument("--warm-start",    action="store_true",
                        help="DQN: continue from existing checkpoint instead of fresh init")
    parser.add_argument("--recent-only",   action="store_true",
                        help="XGBoost: train on last 2 seasons only (faster mid-season retrain)")
    args = parser.parse_args()

    if args.mode == "full":
        train_xgboost_only(test=args.test, recent_only=args.recent_only)
        train_dqn_only(
            epochs=args.epochs, test=args.test,
            clear_buffer=args.clear_buffer,
            incremental=args.incremental,
            warm_start=args.warm_start,
        )
    elif args.mode == "xgboost":
        train_xgboost_only(test=args.test, recent_only=args.recent_only)
    elif args.mode == "dqn":
        train_dqn_only(
            epochs=args.epochs, test=args.test,
            clear_buffer=args.clear_buffer,
            incremental=args.incremental,
            warm_start=args.warm_start,
        )