"""
backend/model/replay_buffer.py
--------------------------------
Experience replay buffer for the DQN agent.

Storage strategy:
  - During training: transitions are accumulated in a Python list (in-memory).
    No DB calls happen inside the epoch loop — this was causing HTTP/2 stream
    exhaustion (ConnectionTerminated last_stream_id:19999) because individual
    INSERT + SELECT calls were fired ~3,000 times per epoch.
  - After each epoch: all new transitions are batch-inserted to Supabase in
    one request. The DB is the persistent store across restarts; memory is
    the fast working buffer.
  - At training start: the full existing buffer is loaded from DB once into
    memory, so the agent benefits from all prior experience.

In-memory buffer size is capped at MAX_BUFFER_SIZE. DB is pruned to match.
"""

import os
import sys
import json
import random
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model.features import STATE_DIM

# ─── Constants ────────────────────────────────────────────────────────────────

BATCH_SIZE       = 64
MIN_BUFFER_SIZE  = 100
MAX_BUFFER_SIZE  = 5000

# ─── In-memory buffer (module-level, shared across calls) ─────────────────────

_memory: list[dict] = []   # holds dicts with keys: state, action, reward, next_state, done


# ─── Load from DB on startup ──────────────────────────────────────────────────

def load_from_db() -> int:
    """
    Loads existing transitions from Supabase into _memory at training start.
    Call this once before the training loop begins.
    Returns the number of transitions loaded.
    """
    global _memory
    try:
        rows = (
            supabase.table("rl_episodes")
            .select("state, action, reward, next_state, done")
            .order("id", desc=True)
            .limit(MAX_BUFFER_SIZE)
            .execute()
            .data
        )
    except Exception as e:
        print(f"  ⚠ Could not load replay buffer from DB: {e}")
        return 0

    _memory = []
    for r in rows:
        try:
            _memory.append({
                "state":      np.array(json.loads(r["state"]),      dtype=np.float32),
                "action":     int(r["action"]),
                "reward":     float(r["reward"]),
                "next_state": np.array(json.loads(r["next_state"]), dtype=np.float32)
                              if r["next_state"] else None,
                "done":       bool(r["done"]),
            })
        except Exception:
            continue

    print(f"  Loaded {len(_memory)} transitions from DB into replay buffer")
    return len(_memory)


# ─── In-memory push (called per transition during epoch) ─────────────────────

def push_memory(
    state:      np.ndarray,
    action:     int,
    reward:     float,
    next_state: Optional[np.ndarray],
    done:       bool,
) -> None:
    """
    Adds a transition to the in-memory buffer only.
    No DB call — safe to call once per match inside the training loop.
    """
    global _memory
    _memory.append({
        "state":      state,
        "action":     int(action),
        "reward":     float(reward),
        "next_state": next_state,
        "done":       bool(done),
    })
    # Keep memory bounded
    if len(_memory) > MAX_BUFFER_SIZE:
        _memory = _memory[-MAX_BUFFER_SIZE:]


# ─── Batch flush to DB (called once per epoch) ────────────────────────────────

def flush_to_db(match_ids: list[int], episode_num: int, new_transitions: list[dict]) -> int:
    """
    Batch-inserts all new transitions from one epoch into Supabase.
    Call this once at the END of each epoch, not inside the match loop.

    Returns number of rows inserted.
    """
    if not new_transitions:
        return 0

    rows = []
    for i, t in enumerate(new_transitions):
        match_id = match_ids[i] if i < len(match_ids) else None
        rows.append({
            "match_id":    match_id,
            "state":       json.dumps([float(v) for v in t["state"]]),
            "action":      int(t["action"]),
            "reward":      float(round(t["reward"], 6)),
            "next_state":  json.dumps([float(v) for v in t["next_state"]])
                           if t["next_state"] is not None else None,
            "done":        bool(t["done"]),
            "episode_num": int(episode_num),
        })

    # Insert in batches of 200 to stay within Supabase request size limits
    inserted = 0
    for i in range(0, len(rows), 200):
        try:
            supabase.table("rl_episodes").insert(rows[i:i+200]).execute()
            inserted += len(rows[i:i+200])
        except Exception as e:
            print(f"  ⚠ Batch insert failed (rows {i}-{i+200}): {e}")

    return inserted


# ─── Sample from memory ───────────────────────────────────────────────────────

def sample(batch_size: int = BATCH_SIZE) -> Optional[dict]:
    """
    Samples a random batch from the in-memory buffer.
    Returns None if the buffer is too small to train on yet.
    """
    if len(_memory) < MIN_BUFFER_SIZE:
        return None

    batch = random.sample(_memory, min(batch_size, len(_memory)))

    states      = np.array([t["state"] for t in batch],      dtype=np.float32)
    actions     = np.array([t["action"] for t in batch],     dtype=np.int64)
    rewards     = np.array([t["reward"] for t in batch],     dtype=np.float32)
    dones       = np.array([t["done"] for t in batch],       dtype=np.float32)
    next_states = np.array([
        t["next_state"] if t["next_state"] is not None else np.zeros(STATE_DIM)
        for t in batch
    ], dtype=np.float32)

    return {
        "states":      states,
        "actions":     actions,
        "rewards":     rewards,
        "next_states": next_states,
        "dones":       dones,
    }


# ─── Utility ──────────────────────────────────────────────────────────────────

def size() -> int:
    return len(_memory)


def is_ready() -> bool:
    return len(_memory) >= MIN_BUFFER_SIZE


def prune(keep_latest: int = MAX_BUFFER_SIZE) -> int:
    """
    Removes oldest rows from the DB to keep it bounded.
    The in-memory buffer is already bounded by push_memory().
    """
    try:
        current = supabase.table("rl_episodes").select("id", count="exact").execute()
        total   = current.count or 0
        if total <= keep_latest:
            return 0

        cutoff_row = (
            supabase.table("rl_episodes")
            .select("id")
            .order("id", desc=True)
            .limit(1)
            .offset(keep_latest - 1)
            .execute()
        )
        if not cutoff_row.data:
            return 0

        cutoff_id = cutoff_row.data[0]["id"]
        supabase.table("rl_episodes").delete().lt("id", cutoff_id).execute()
        deleted = total - keep_latest
        print(f"  Replay buffer pruned: removed {deleted} old transitions from DB")
        return deleted
    except Exception as e:
        print(f"  ⚠ Prune failed: {e}")
        return 0


# ─── Legacy push (kept for inference-time single insertions) ─────────────────

def push(
    match_id:    Optional[int],
    state:       np.ndarray,
    action:      int,
    reward:      float,
    next_state:  Optional[np.ndarray],
    done:        bool,
    episode_num: Optional[int] = None,
) -> None:
    """
    Single-transition insert — used outside training (e.g. live bet settlement).
    During training, use push_memory() + flush_to_db() instead.
    """
    push_memory(state, action, reward, next_state, done)
    try:
        supabase.table("rl_episodes").insert({
            "match_id":    match_id,
            "state":       json.dumps([float(v) for v in state]),
            "action":      int(action),
            "reward":      float(round(reward, 6)),
            "next_state":  json.dumps([float(v) for v in next_state])
                           if next_state is not None else None,
            "done":        bool(done),
            "episode_num": episode_num,
        }).execute()
    except Exception as e:
        print(f"  ⚠ Single push to DB failed: {e}")