"""
scripts/run_simulation.py
--------------------------
Full orchestrator for the historical backfill simulation.

Retraining schedule — mirrors live deployment behaviour:
  2024-25  MD 1-19   → base model (XGBoost + Kelly only, no DQN)
           MD 19     → mid-season retrain (XGBoost on base seasons, first DQN)
           MD 20-38  → XGBoost + DQN + Kelly
           MD 38     → end-of-season retrain (include full 2024-25)

  2025-26  MD 1-19   → uses 2024-25 end-of-season model
           MD 19     → mid-season retrain (XGBoost on completed seasons, DQN update)
           MD 20-38  → updated XGBoost + DQN + Kelly

Why mid-season only (not monthly) for 2025-26:
  Monthly retraining of XGBoost on partial in-progress seasons creates
  subtle distributional instability — form tables, ELO, and standings
  are atypical early in a season. Two retrains per season (mid + end)
  matches the live deployment schedule and keeps the training set clean.
  DQN learns continuously via the replay buffer regardless.

Why 2025-26 is never included in XGBoost training:
  XGBoost only trains on COMPLETED seasons. A partial season has
  survivorship patterns (teams still fighting relegation, title race
  compressing odds) that don't represent the full distribution.
  The DQN handles current-season adaptation via the replay buffer.

Prerequisites (run in this order):
  1. Run 01_db_reset.sql in Supabase SQL editor (full reset)
  2. python -m backend.data_pipeline.update_elo --all-seasons
  3. python -m backend.data_pipeline.fetch_historical_odds
  4. python -m backend.model.train --mode xgboost

Then:
  python scripts/run_simulation.py
  python scripts/run_simulation.py --dry-run
  python scripts/run_simulation.py --phase 1
  python scripts/run_simulation.py --phase 2
  python scripts/run_simulation.py --phase 3
  python scripts/run_simulation.py --phase 3 --from-second-half   (resume after mid-retrain)
"""

import os
import sys
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.simulate_historical_bets import run as simulate
from backend.model.xgboost_model import train as _train_xgb_direct
from backend.model.train import (
    train_dqn_only,
    TRAIN_SEASONS,
    VAL_SEASON,
    DQN_EPOCHS,
)

# ─── Season config ────────────────────────────────────────────────────────────

SEASON_2425 = "2024-25"
SEASON_2526 = "2025-26"

# Mid-season cutoff for both seasons.
# PL runs Aug→May. MD19 falls ~late December.
# Jan 1 is a clean calendar boundary that reliably splits the season.
MID_SEASON_CUTOFF_2425 = "2025-01-01"
MID_SEASON_CUTOFF_2526 = "2026-01-01"

# Completed seasons — the only ones XGBoost ever trains on.
# 2025-26 deliberately excluded (partial season — see module docstring).
COMPLETED_SEASONS = list(dict.fromkeys(TRAIN_SEASONS + [VAL_SEASON]))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _retrain_xgb(
    label:       str,
    seasons:     list[str],
    before_date: str | None,
    dry_run:     bool,
) -> None:
    """
    Train and register XGBoost on the given completed seasons.

    before_date caps both the label rows AND the bulk_fetch() cache,
    closing all distributional leakage paths. Pass None only when every
    listed season is fully complete.
    """
    date_note = f" before {before_date}" if before_date else " (no cap — seasons complete)"
    _log(f"XGBoost retrain [{label}]")
    _log(f"  seasons: {seasons}{date_note}")
    if dry_run:
        _log("  [DRY-RUN] Skipping")
        return
    version_tag = f"xgb_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _, metrics  = _train_xgb_direct(
        seasons     = seasons,
        version_tag = version_tag,
        before_date = before_date,
    )
    _log(f"  ✓ log-loss={metrics['val_log_loss']}  tag={version_tag}")


def _retrain_dqn(
    label:       str,
    dry_run:     bool,
    epochs:      int  = DQN_EPOCHS,
    incremental: bool = False,
    warm_start:  bool = False,
) -> None:
    mode = "incremental" if incremental else "full"
    _log(f"DQN retrain [{label}]: {mode}  epochs={epochs}")
    if dry_run:
        _log("  [DRY-RUN] Skipping")
        return
    train_dqn_only(epochs=epochs, incremental=incremental, warm_start=warm_start)
    _log("  ✓ DQN done.")


# ─── Phase 1 — 2024-25 first half ────────────────────────────────────────────

def phase_1(dry_run: bool) -> None:
    """
    2024-25 MD 1-19.
    Base XGBoost + Kelly only. DQN doesn't exist yet.
    """
    _log("=" * 60)
    _log("PHASE 1 — 2024-25 MD 1-19  (base model, no DQN)")
    _log("=" * 60)

    simulate(
        seasons     = [SEASON_2425],
        before_date = MID_SEASON_CUTOFF_2425,
        no_dqn      = True,
        dry_run     = dry_run,
    )

    _log("\n── Mid-season retrain (2024-25 MD 19) ──")
    # Base seasons only — 2024-25 is half-done.
    # before_date caps feature cache so Jan-May 2024-25 data is excluded.
    _retrain_xgb(
        label       = "2425_mid",
        seasons     = COMPLETED_SEASONS,
        before_date = MID_SEASON_CUTOFF_2425,
        dry_run     = dry_run,
    )
    # First DQN training — learns from MD 1-19 replay buffer.
    # warm_start=True is a no-op when no checkpoint exists yet.
    _retrain_dqn(
        label       = "2425_mid",
        dry_run     = dry_run,
        epochs      = DQN_EPOCHS,
        incremental = True,
        warm_start  = True,
    )


# ─── Phase 2 — 2024-25 second half ───────────────────────────────────────────

def phase_2(dry_run: bool) -> None:
    """
    2024-25 MD 20-38.
    Updated XGBoost + DQN + Kelly.
    already_simulated() skips MD 1-19 automatically.
    """
    _log("=" * 60)
    _log("PHASE 2 — 2024-25 MD 20-38  (XGBoost + DQN + Kelly)")
    _log("=" * 60)

    simulate(
        seasons = [SEASON_2425],
        no_dqn  = False,
        dry_run = dry_run,
    )

    _log("\n── End-of-season retrain (2024-25 MD 38) ──")
    # Include full 2024-25 — season is complete, no date cap needed.
    xgb_seasons = list(dict.fromkeys(COMPLETED_SEASONS + [SEASON_2425]))
    _retrain_xgb(
        label       = "2425_end",
        seasons     = xgb_seasons,
        before_date = None,   # whole season is done — no cap needed
        dry_run     = dry_run,
    )
    # Full DQN retrain — flush everything accumulated over 2024-25.
    _retrain_dqn(
        label       = "2425_end",
        dry_run     = dry_run,
        epochs      = DQN_EPOCHS,
        incremental = False,
        warm_start  = False,
    )


# ─── Phase 3 — 2025-26 (same mid-season pattern as 2024-25) ──────────────────

def phase_3(dry_run: bool, from_second_half: bool = False) -> None:
    """
    2025-26, split at MD 19 (≈ Jan 1 2026).

    First half  : XGBoost + DQN + Kelly (using 2024-25 end-of-season model)
    Mid-season  : XGBoost retrain on completed seasons, DQN incremental update
    Second half : updated XGBoost + DQN + Kelly

    from_second_half: skip to the second half if the first half + retrain
                      already completed (useful for resuming after interruption).
    """
    _log("=" * 60)
    _log("PHASE 3 — 2025-26  (same mid-season pattern as 2024-25)")
    _log("=" * 60)

    # Seasons available to XGBoost at this point — never includes 2025-26
    xgb_seasons = list(dict.fromkeys(COMPLETED_SEASONS + [SEASON_2425]))

    if not from_second_half:
        # ── First half: MD 1-19 ───────────────────────────────────────────────
        _log("── 2025-26 MD 1-19  (before Jan 2026) ──")
        simulate(
            seasons     = [SEASON_2526],
            before_date = MID_SEASON_CUTOFF_2526,
            no_dqn      = False,   # DQN exists from 2024-25 end-of-season retrain
            dry_run     = dry_run,
        )

        _log("\n── Mid-season retrain (2025-26 MD 19) ──")
        # Completed seasons only. before_date=None because every listed season
        # is fully complete (2024-25 ended before Jan 2026).
        _retrain_xgb(
            label       = "2526_mid",
            seasons     = xgb_seasons,
            before_date = None,
            dry_run     = dry_run,
        )
        # Incremental DQN — adds first-half 2025-26 transitions to existing knowledge
        _retrain_dqn(
            label       = "2526_mid",
            dry_run     = dry_run,
            epochs      = DQN_EPOCHS,
            incremental = True,
            warm_start  = True,
        )

    # ── Second half: MD 20 onwards ────────────────────────────────────────────
    _log("── 2025-26 MD 20 onwards  (from Jan 2026) ──")
    # No before_date — simulate the rest of the season.
    # already_simulated() skips MD 1-19 automatically.
    simulate(
        seasons = [SEASON_2526],
        no_dqn  = False,
        dry_run = dry_run,
    )

    # End-of-season note:
    # If 2025-26 is now complete, an optional final retrain can be run manually:
    #   python -m backend.model.train --mode xgboost --include-current-season
    #   python -m backend.model.train --mode dqn
    _log("\n── 2025-26 simulation complete ──")
    _log("If the season is finished, run a final retrain manually:")
    _log("  python -m backend.model.train --mode xgboost --include-current-season")
    _log("  python -m backend.model.train --mode dqn")


# ─── Entry point ─────────────────────────────────────────────────────────────

def run(
    dry_run:          bool = False,
    phase:            int  = 0,
    from_second_half: bool = False,
) -> None:
    start = datetime.now()
    _log("=" * 60)
    _log("BACKFILL SIMULATION ORCHESTRATOR")
    _log(f"dry_run={dry_run}  phase={phase or 'all'}")
    _log("=" * 60)

    run_all = (phase == 0)

    if run_all or phase == 1:
        phase_1(dry_run)

    if run_all or phase == 2:
        phase_2(dry_run)

    if run_all or phase == 3:
        phase_3(dry_run, from_second_half=from_second_half)

    elapsed = (datetime.now() - start).total_seconds() / 60
    _log(f"\nAll done. Elapsed: {elapsed:.1f} min")
    _log("Live pipeline (bet_placer / settle_bets) can now run normally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full backfill simulation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--from-second-half", action="store_true",
        help="Phase 3 only: skip first half + mid-season retrain, "
             "resume from MD 20 onwards (use if first half already done)"
    )
    args = parser.parse_args()
    run(
        dry_run          = args.dry_run,
        phase            = args.phase,
        from_second_half = args.from_second_half,
    )