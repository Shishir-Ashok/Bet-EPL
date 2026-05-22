"""
backend/scripts/run_backtest.py
--------------------------------
Phases
------
1. Train XGBoost on base seasons 2020-21 → 2023-24.
2. Simulate 2024-25 matchday by matchday (10 matches per round).
   - Rounds 1-19  : XGBoost + Kelly only (no DQN yet).
   - After round 19: retrain XGBoost (base seasons only, no leakage)
                     + first DQN training on accumulated replay buffer.
   - Rounds 20-38 : XGBoost + DQN + Kelly.
   - After round 38: retrain XGBoost on base + full 2024-25
                     + full DQN retrain.
3. Simulate 2025-26 month by month up to today.
   - Each month: place bets → settle → monthly retrain (XGBoost + DQN).
4. Leave remaining SCHEDULED/future matches for the live pipeline.

Usage
-----
    python -m backend.scripts.run_backtest
    python -m backend.scripts.run_backtest --dry-run   # no DB writes
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, date, timezone
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.db import supabase
import backend.engine.bet_placer  as bet_placer_mod
import backend.engine.settle_bets as settle_bets_mod
from backend.model.xgboost_model import train as _train_xgb
from backend.model.train         import train_dqn_only, DQN_EPOCHS

# ─── Season constants ─────────────────────────────────────────────────────────

SEASONS_BASE = ["2020-21", "2021-22", "2022-23", "2023-24"]
SEASON_2425  = "2024-25"
SEASON_2526  = "2025-26"
TODAY        = date.today()

# ─── Helpers ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="run_backtest.log",
    filemode="a",           # 'a' = append, 'w' = overwrite each run
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    logging.info(msg)
    print(f"[{ts}] {msg}")


def _set_status(ids: list[int], status: str, dry_run: bool) -> None:
    """Batch-update match statuses in chunks of 100."""
    if dry_run:
        print(f"    [DRY-RUN] Would set {len(ids)} matches → {status}")
        return
    for i in range(0, len(ids), 100):
        batch = ids[i : i + 100]
        supabase.table("matches").update({"status": status}).in_("id", batch).execute()
    time.sleep(0.15)


def _load_season_matches(season: str) -> list[dict]:
    """
    Returns all matches for a season that have a result, ordered by
    kickoff_time. Paginates past Supabase's 1000-row cap.
    """
    matches, offset, page_size = [], 0, 500
    while True:
        page = (
            supabase.table("matches")
            .select("id, kickoff_time, result, season")
            .eq("season", season)
            .not_.is_("result", "null")
            .order("kickoff_time", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if not page:
            break
        matches.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.1)
    return matches


def _group_into_rounds(matches: list[dict]) -> list[list[dict]]:
    """
    Groups matches into matchday rounds by kickoff proximity.
    A new round begins when there is a gap of > 6 days from the
    previous match — handles double/blank gameweeks.
    """
    if not matches:
        return []
    sorted_m = sorted(matches, key=lambda m: m["kickoff_time"])
    rounds, current = [], [sorted_m[0]]
    for m in sorted_m[1:]:
        prev_dt = datetime.fromisoformat(current[-1]["kickoff_time"].replace("Z", "+00:00"))
        curr_dt = datetime.fromisoformat(m["kickoff_time"].replace("Z", "+00:00"))
        if (curr_dt - prev_dt).days > 6:
            rounds.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        rounds.append(current)
    return rounds


def _train_xgboost(
    label:   str,
    seasons: list[str],
    dry_run: bool,
) -> None:
    """Train and register XGBoost on the given seasons."""
    _log(f"XGBoost retrain: {label}  seasons={seasons}")
    if dry_run:
        _log("  [DRY-RUN] Skipping XGBoost training")
        return
    version_tag = f"xgb_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _, metrics = _train_xgb(seasons=seasons, version_tag=version_tag)
    _log(f"  ✓ XGBoost done. log-loss={metrics['val_log_loss']}  tag={version_tag}")


def _train_dqn(
    label:       str,
    dry_run:     bool,
    epochs:      int  = DQN_EPOCHS,
    incremental: bool = False,
    warm_start:  bool = False,
) -> None:
    """Train and register the DQN agent."""
    mode = "incremental" if incremental else "full"
    _log(f"DQN retrain: {label}  mode={mode}  epochs={epochs}")
    if dry_run:
        _log("  [DRY-RUN] Skipping DQN training")
        return
    train_dqn_only(
        epochs      = epochs,
        incremental = incremental,
        warm_start  = warm_start,
    )
    _log("  ✓ DQN done.")


# ─── Round simulation ─────────────────────────────────────────────────────────

def _simulate_round(
    round_matches: list[dict],
    round_num:     int,
    season_label:  str,
    dry_run:       bool,
) -> None:
    """
    Simulates one matchday round:
      1. Set matches SCHEDULED → bet_placer places bets.
      2. Set matches FINISHED  → settle_bets settles them + stores RL transitions.
    """
    ids = [m["id"] for m in round_matches]
    ko  = round_matches[0]["kickoff_time"][:10]
    _log(f"  Round {round_num:2d} | {season_label} | {ko} | {len(ids)} matches")

    # Step 1: place bets
    _set_status(ids, "SCHEDULED", dry_run)
    if not dry_run:
        try:
            bet_result = bet_placer_mod.run(place_bets=True)
            print(f"    bets placed: {bet_result.get('bets_placed', 0)}"
                  f"  balance: €{bet_result.get('balance', '?')}")
        except Exception as e:
            print(f"    ✗ bet_placer error: {e}")

    # Step 2: settle bets
    _set_status(ids, "FINISHED", dry_run)
    if not dry_run:
        try:
            settle_result = settle_bets_mod.run()
            print(f"    settled: {settle_result.get('settled', 0)}"
                  f"  pnl: {settle_result.get('session_pnl', 0):+.2f}"
                  f"  balance: €{settle_result.get('balance', '?'):.2f}")
        except Exception as e:
            print(f"    ✗ settle_bets error: {e}")

    time.sleep(0.3)   # rate-limit buffer


# ─── Phase 2: 2024-25 ─────────────────────────────────────────────────────────

def run_phase_2(dry_run: bool) -> None:
    _log("=" * 55)
    _log("PHASE 2 — Simulate 2024-25")
    _log("=" * 55)

    matches = _load_season_matches(SEASON_2425)
    if not matches:
        _log("  ✗ No 2024-25 matches found. Stopping phase 2.")
        return

    rounds = _group_into_rounds(matches)
    _log(f"  {len(matches)} matches grouped into {len(rounds)} rounds")

    for i, round_matches in enumerate(rounds, start=1):
        _simulate_round(round_matches, i, SEASON_2425, dry_run)

        # ── Mid-season retrain: after matchday 19 ────────────────────────────
        if i == 19:
            _log("\n  ── MID-SEASON RETRAIN (after round 19) ──")
            # XGBoost: base seasons only — 2024-25 is only half-done,
            # including partial season risks noise leakage.
            _train_xgboost("2425_mid", SEASONS_BASE, dry_run)
            # DQN: first training — uses replay buffer from rounds 1-19.
            # train_dqn_only will see 2024-25 MD1-19 (now FINISHED) as current
            # season matches via CURRENT_SEASON / incremental mode.
            # Since no DQN checkpoint exists yet, warm_start silently falls
            # back to a fresh init.
            _train_dqn(
                "2425_mid",
                dry_run,
                epochs      = DQN_EPOCHS,
                incremental = True,
                warm_start  = True,
            )
            _log("  ── Mid-season retrain complete ──\n")

    # ── End-of-season retrain ─────────────────────────────────────────────────
    _log("\n  ── END-OF-SEASON RETRAIN (after round 38) ──")
    _train_xgboost(
        "2425_end",
        SEASONS_BASE + [SEASON_2425],   # now include complete 2024-25
        dry_run,
    )
    _train_dqn(
        "2425_end",
        dry_run,
        epochs      = DQN_EPOCHS,
        incremental = False,   # full retrain — flush everything into the net
        warm_start  = False,
    )
    _log("  ── End-of-season retrain complete ──\n")


# ─── Phase 3: 2025-26 month by month ─────────────────────────────────────────

def run_phase_3(dry_run: bool) -> None:
    _log("=" * 55)
    _log("PHASE 3 — Simulate 2025-26 (monthly)")
    _log("=" * 55)

    matches = _load_season_matches(SEASON_2526)
    if not matches:
        _log("  No 2025-26 finished matches found.")
        return

    # Filter to matches that are in the past (kickoff <= today)
    past_matches = [
        m for m in matches
        if datetime.fromisoformat(m["kickoff_time"].replace("Z", "+00:00")).date() <= TODAY
    ]
    _log(f"  {len(past_matches)} past 2025-26 matches to simulate (of {len(matches)} total)")

    # Group by calendar month
    by_month: dict[tuple, list] = defaultdict(list)
    for m in past_matches:
        dt  = datetime.fromisoformat(m["kickoff_time"].replace("Z", "+00:00"))
        key = (dt.year, dt.month)
        by_month[key].append(m)

    months = sorted(by_month.keys())
    _log(f"  Months to process: {[f'{y}-{mo:02d}' for y, mo in months]}")

    for month_idx, (year, month) in enumerate(months, start=1):
        month_label = f"{year}-{month:02d}"
        month_matches = sorted(by_month[(year, month)], key=lambda m: m["kickoff_time"])
        _log(f"\n  [{month_label}] {len(month_matches)} matches")

        # Process each round within the month
        month_rounds = _group_into_rounds(month_matches)
        for j, round_m in enumerate(month_rounds, start=1):
            _simulate_round(round_m, j, f"{SEASON_2526} {month_label}", dry_run)

        # Monthly retrain after every month
        _log(f"\n  ── Monthly retrain after {month_label} ──")

        # XGBoost: all base + 2024-25 + 2025-26 (everything now FINISHED)
        # Because 2025-26 matches up to this month are now FINISHED after
        # simulation, the training function will include them automatically.
        _train_xgboost(
            f"2526_{month_label.replace('-','')}_monthly",
            SEASONS_BASE + [SEASON_2425, SEASON_2526],
            dry_run,
        )
        # DQN: incremental warm-start — learns from new bets this month
        _train_dqn(
            f"2526_{month_label.replace('-','')}_monthly",
            dry_run,
            epochs      = 15,   # lighter monthly update
            incremental = True,
            warm_start  = True,
        )
        _log(f"  ── Monthly retrain complete ──\n")
        time.sleep(1)

    # Remaining 2025-26 future matches stay in BACKTEST (or SCHEDULED if
    # they were already SCHEDULED). Reset any still-BACKTEST future ones
    # back to SCHEDULED so the live pipeline can pick them up.
    _log("\n  Restoring remaining future 2025-26 matches to SCHEDULED...")
    future = [
        m for m in matches
        if datetime.fromisoformat(m["kickoff_time"].replace("Z", "+00:00")).date() > TODAY
    ]
    if future:
        _set_status([m["id"] for m in future], "SCHEDULED", dry_run)
        _log(f"  {len(future)} future matches set to SCHEDULED")

    # Also restore any BACKTEST matches that still have no result
    # (defensive cleanup in case of partial runs)
    if not dry_run:
        leftover = (
            supabase.table("matches")
            .select("id")
            .eq("status", "BACKTEST")
            .is_("result", "null")
            .execute()
            .data
        )
        if leftover:
            _set_status([r["id"] for r in leftover], "SCHEDULED", dry_run)
            _log(f"  {len(leftover)} resultless BACKTEST matches restored to SCHEDULED")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    start = datetime.now()
    _log("=" * 55)
    _log("HISTORICAL BACKTEST RUNNER")
    _log(f"Today: {TODAY}  |  dry_run={dry_run}")
    _log("=" * 55)

    # ── Confirm DB reset was run ──────────────────────────────────────────────
    backtest_count = (
        supabase.table("matches")
        .select("id", count="exact")
        .eq("status", "BACKTEST")
        .execute()
        .count
    )
    if backtest_count == 0:
        print("\n⚠  No BACKTEST-status matches found.")
        print("   Did you run 01_db_reset.sql first?")
        print("   Aborting to avoid overwriting live data.\n")
        return

    _log(f"Found {backtest_count} BACKTEST matches — ready to simulate.")

    # ── Phase 1: Base XGBoost ─────────────────────────────────────────────────
    _log("\n" + "=" * 55)
    _log("PHASE 1 — Train base XGBoost (2020-21 to 2023-24)")
    _log("=" * 55)
    _train_xgboost("base", SEASONS_BASE, dry_run)

    # ── Phase 2: 2024-25 simulation ───────────────────────────────────────────
    run_phase_2(dry_run)

    # ── Phase 3: 2025-26 monthly simulation ──────────────────────────────────
    run_phase_3(dry_run)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds() / 60
    wallet  = supabase.table("wallet").select("balance, total_staked").eq("id", 1).single().execute().data
    bets_n  = supabase.table("bets").select("id", count="exact").execute().count

    _log("\n" + "=" * 55)
    _log("BACKTEST COMPLETE")
    _log(f"  Elapsed:       {elapsed:.1f} min")
    _log(f"  Bets placed:   {bets_n}")
    _log(f"  Final balance: €{float(wallet['balance']):.2f}")
    _log(f"  Total staked:  €{float(wallet['total_staked']):.2f}")
    _log("=" * 55)
    _log("The live pipeline (bet_placer / settle_bets) can now run normally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run historical backtest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to the DB or training models",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)