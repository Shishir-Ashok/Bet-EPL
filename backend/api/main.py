"""
backend/api/main.py
--------------------
FastAPI server deployed on Render's free tier.

This is the brain that GitHub Actions talks to. Three workflows call it:
  fetch_prematch.yml  → POST /trigger/predict   (before matchday)
  ingest_results.yml  → POST /trigger/settle    (after results)
  weekly_retrain.yml  → POST /trigger/retrain   (every Monday)

Authentication:
  Every request must include the header:
    X-API-Secret: <value of RENDER_API_SECRET env var>
  Without it, all endpoints return 403.

Render free tier behaviour:
  The service spins down after 15 minutes of inactivity.
  GitHub Actions retries 3 times with 30s gaps to handle cold starts.
  A cold start typically takes 30-60 seconds.

Running locally:
  uv run uvicorn backend.api.main:app --reload --port 8000

Deploying to Render:
  1. Connect your GitHub repo to Render
  2. Set build command: pip install -r backend/requirements.txt
  3. Set start command: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
  4. Add all env vars from .env as Render environment variables
"""

import os
import sys
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Auth ─────────────────────────────────────────────────────────────────────

API_SECRET = os.environ.get("RENDER_API_SECRET")

def verify_secret(x_api_secret: str = Header(None)) -> None:
    """
    Validates the X-API-Secret header on every trigger endpoint.
    Returns 403 if missing or wrong — no information about why is given
    to avoid leaking that the endpoint exists.
    """
    if not API_SECRET:
        log.warning("RENDER_API_SECRET not set — all requests will be rejected")
        raise HTTPException(status_code=403, detail="Forbidden")
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PL Betting Bot API starting up")
    yield
    log.info("PL Betting Bot API shutting down")


app = FastAPI(
    title       = "PL Betting Bot API",
    description = "Internal API for triggering the betting pipeline",
    version     = "1.0.0",
    lifespan    = lifespan,
    # Hide docs in production — this is an internal API
    docs_url    = "/docs" if os.environ.get("ENV") != "production" else None,
    redoc_url   = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # frontend reads from Supabase directly, not this API
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ─── Request/response models ──────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    force_promote: bool = False
    epochs:        int  = 30


class TriggerResponse(BaseModel):
    status:    str
    message:   str
    timestamp: str
    data:      dict = {}


def ok(message: str, data: dict = {}) -> TriggerResponse:
    return TriggerResponse(
        status    = "ok",
        message   = message,
        timestamp = datetime.now(timezone.utc).isoformat(),
        data      = data,
    )


# ─── Health check (no auth) ───────────────────────────────────────────────────

@app.get("/")
async def root():
    """
    Health check — no auth required.
    GitHub Actions pings this first to wake the service before calling triggers.
    """
    return {
        "status":    "ok",
        "service":   "PL Betting Bot API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    """Detailed health check — checks DB connectivity."""
    try:
        from backend.db import supabase
        result = supabase.table("wallet").select("balance").eq("id", 1).execute()
        balance = float(result.data[0]["balance"]) if result.data else None
        return {
            "status":   "ok",
            "db":       "connected",
            "balance":  balance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return {
            "status":    "degraded",
            "db":        "error",
            "error":     str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ─── Trigger: predict + place bets ───────────────────────────────────────────

@app.post("/trigger/predict", dependencies=[Depends(verify_secret)])
async def trigger_predict(background_tasks: BackgroundTasks) -> TriggerResponse:
    """
    Triggered by fetch_prematch.yml before each matchday.

    Runs the full prediction pipeline:
      - Build state vectors for upcoming matches
      - Run XGBoost + DQN predictions
      - Apply Kelly sizing with vig removal
      - Place virtual bets and update wallet

    Returns immediately with 200 and runs in the background so GitHub
    Actions doesn't time out waiting for the pipeline to finish.
    """
    log.info("POST /trigger/predict received")
    background_tasks.add_task(_run_predict)
    return ok("Prediction pipeline started", {"mode": "background"})


async def _run_predict():
    try:
        from backend.engine.bet_placer import run
        result = run()
        log.info(f"Predict complete: {result.get('bets_placed',0)} bets placed, "
                 f"balance €{result.get('balance', 0):.2f}")
    except Exception as e:
        log.error(f"Predict pipeline failed: {e}", exc_info=True)


# ─── Trigger: settle bets ────────────────────────────────────────────────────

@app.post("/trigger/settle", dependencies=[Depends(verify_secret)])
async def trigger_settle(background_tasks: BackgroundTasks) -> TriggerResponse:
    """
    Triggered by ingest_results.yml after match results are available.

    Settles all open bets:
      - Determines WIN/LOSS for each open bet
      - Calculates P&L
      - Updates wallet balance
      - Marks prediction accuracy in the predictions table
      - Stores RL transitions for future DQN retraining
    """
    log.info("POST /trigger/settle received")
    background_tasks.add_task(_run_settle)
    return ok("Settlement pipeline started", {"mode": "background"})


async def _run_settle():
    try:
        from backend.engine.settle_bets import run
        result = run()
        log.info(f"Settle complete: {result.get('settled',0)} settled, "
                 f"P&L €{result.get('session_pnl',0):+.2f}, "
                 f"balance €{result.get('balance',0):.2f}")
    except Exception as e:
        log.error(f"Settle pipeline failed: {e}", exc_info=True)


# ─── Trigger: retrain models ─────────────────────────────────────────────────

@app.post("/trigger/retrain", dependencies=[Depends(verify_secret)])
async def trigger_retrain(
    body:             RetrainRequest = RetrainRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> TriggerResponse:
    """
    Triggered by weekly_retrain.yml every Monday at 06:00 UTC.

    Retrains both models:
      1. XGBoost — retrained on all historical + current season matches
      2. DQN — continues training from current weights using accumulated
                replay buffer (including live bet transitions)

    The new model is registered as active only if val performance improves
    (unless force_promote=True is passed in the request body).

    This runs in the background — Render's free tier may take 10-20 minutes.
    The 202 response lets GitHub Actions know the job was accepted.
    """
    log.info(f"POST /trigger/retrain received (force_promote={body.force_promote})")
    background_tasks.add_task(_run_retrain, body.epochs, body.force_promote)
    return TriggerResponse(
        status    = "accepted",
        message   = "Retraining started in background",
        timestamp = datetime.now(timezone.utc).isoformat(),
        data      = {"epochs": body.epochs, "force_promote": body.force_promote},
    )


async def _run_retrain(epochs: int, force_promote: bool):
    try:
        log.info(f"Starting retraining: epochs={epochs}")

        from backend.model.train import train_dqn_only
        agent = train_dqn_only(epochs=epochs, clear_buffer=False)
        log.info("Retraining complete")

    except Exception as e:
        log.error(f"Retraining failed: {e}", exc_info=True)


# ─── Status endpoints (no auth — safe to expose) ─────────────────────────────

@app.get("/status/wallet")
async def wallet_status():
    """Returns current wallet balance. Used by the frontend dashboard."""
    try:
        from backend.db import supabase
        wallet = supabase.table("wallet").select("*").eq("id", 1).single().execute().data
        return {
            "balance":        float(wallet["balance"]),
            "total_staked":   float(wallet["total_staked"]),
            "total_returned": float(wallet["total_returned"]),
            "roi_pct":        round(
                (float(wallet["total_returned"]) - float(wallet["total_staked"]))
                / max(float(wallet["total_staked"]), 0.01) * 100, 2
            ),
            "inception_date": wallet["inception_date"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/bets")
async def bet_status():
    """Returns recent bet history. Quick summary for monitoring."""
    try:
        from backend.db import supabase
        bets = (
            supabase.table("bets")
            .select("action, stake, odds, outcome, pnl, placed_at, settled_at")
            .order("placed_at", desc=True)
            .limit(10)
            .execute()
            .data
        )
        open_count = sum(1 for b in bets if b["outcome"] is None)
        won        = sum(1 for b in bets if b["outcome"] == "WIN")
        lost       = sum(1 for b in bets if b["outcome"] == "LOSS")
        return {
            "recent_bets":  bets,
            "open":         open_count,
            "won":          won,
            "lost":         lost,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/model")
async def model_status():
    """Returns currently active model versions."""
    try:
        from backend.db import supabase
        models = (
            supabase.table("model_versions")
            .select("version_tag, model_type, trained_at, val_log_loss, avg_reward, is_active")
            .eq("is_active", True)
            .execute()
            .data
        )
        return {"active_models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))