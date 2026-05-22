"""
backend/model/xgboost_model.py
--------------------------------
XGBoost multi-class classifier: given a match's 16-feature state vector,
predicts the probability of HOME win, DRAW, or AWAY win.

Fix: before_date now flows all the way through:
  train(before_date)
    → load_training_data(before_date)
      → build_feature_matrix(matches, before_date)   ← WAS MISSING
        → bulk_fetch(before_date)                     ← WAS MISSING

Without passing before_date into build_feature_matrix, load_training_data
correctly filtered which matches became training LABELS, but the cache fed
into feature computation still contained the full future dataset. This caused
distributional leakage — XGBoost implicitly learned from future ELO spreads
and form distributions even though individual future match entries were
blocked by the per-match temporal filter.
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Optional

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model.features import build_feature_matrix

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
os.makedirs(MODEL_DIR, exist_ok=True)

XGB_PARAMS = {
    "objective":             "multi:softprob",
    "num_class":             3,
    "n_estimators":          300,
    "max_depth":             4,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "min_child_weight":      5,
    "gamma":                 0.1,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "eval_metric":           "mlogloss",
    "tree_method":           "hist",
    "device":                "cuda",
    "random_state":          42,
    "verbosity":             1,
    "early_stopping_rounds": 30,
}

LABEL_MAP   = {0: "HOME", 1: "DRAW", 2: "AWAY"}
LABEL_MAP_R = {"HOME": 0, "DRAW": 1, "AWAY": 2}


# ─── Calibrated model wrapper ─────────────────────────────────────────────────

class CalibratedModel:
    """
    XGBoost classifier wrapped with per-class isotonic probability calibration.
    Ensures that "60% home win" actually means home wins 60% of the time.
    """

    def __init__(self, base: xgb.XGBClassifier, calibrators: list):
        self.base        = base
        self.calibrators = calibrators

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)
        cal = np.stack(
            [iso.predict(raw[:, i]) for i, iso in enumerate(self.calibrators)],
            axis=1,
        )
        return cal / np.maximum(cal.sum(axis=1, keepdims=True), 1e-8)

    @property
    def estimator(self) -> xgb.XGBClassifier:
        return self.base


# ─── Training ─────────────────────────────────────────────────────────────────

def load_training_data(
    seasons:     Optional[list[str]] = None,
    limit:       Optional[int]       = None,
    before_date: Optional[str]       = None,
) -> tuple:
    """
    Loads finished historical matches and builds the feature matrix.

    Parameters
    ----------
    seasons     : filter to specific seasons. None = all seasons.
    limit       : cap rows (use 100 for test runs).
    before_date : ISO date string e.g. "2025-10-01".
                  Caps BOTH which matches become training labels AND
                  the data loaded into the feature cache via
                  build_feature_matrix(before_date=before_date).
                  Prevents future match outcomes leaking into either
                  labels or feature distributions.
    """
    import time

    query = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, result, season")
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )

    if seasons:
        query = query.in_("season", seasons)

    if before_date:
        query = query.lt("kickoff_time", before_date)

    if limit:
        query   = query.limit(limit)
        matches = query.execute().data
    else:
        matches   = []
        page_size = 1000
        offset    = 0
        while True:
            for attempt in range(3):
                try:
                    page = query.range(offset, offset + page_size - 1).execute().data
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
            offset += page_size
            time.sleep(0.2)

    if not matches:
        raise ValueError(
            "No finished matches in DB. Run seed_historical_data.py first."
        )

    date_note = f" (before {before_date})" if before_date else ""
    print(f"  Loaded {len(matches)} finished matches{date_note}")
    print(f"  Seasons: {sorted(set(m['season'] for m in matches))}")

    # THE KEY FIX: pass before_date into build_feature_matrix so the cache
    # loaded by bulk_fetch() is also date-capped. Previously before_date
    # filtered the label rows but bulk_fetch() inside build_feature_matrix
    # still loaded the entire DB into the feature cache.
    X, y, ids = build_feature_matrix(matches, before_date=before_date)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Classes — HOME: {sum(y==0)}, DRAW: {sum(y==1)}, AWAY: {sum(y==2)}")

    return X, y, ids


def train(
    seasons:     Optional[list[str]] = None,
    version_tag: Optional[str]       = None,
    val_split:   float               = 0.15,
    limit:       Optional[int]       = None,
    before_date: Optional[str]       = None,
) -> tuple:
    """
    Trains the XGBoost model on historical match data.

    Parameters
    ----------
    before_date : passed through to load_training_data and then into
                  build_feature_matrix so both labels and features are
                  capped to data before this date.
    """
    print("\n[XGBoost] Loading training data...")
    X, y, ids = load_training_data(seasons, limit=limit, before_date=before_date)

    if len(X) < 50:
        raise ValueError(f"Only {len(X)} samples — need at least 50.")

    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}")

    params = XGB_PARAMS.copy()
    try:
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            params["device"] = "cpu"
    except Exception:
        params["device"] = "cpu"

    print(f"  Device: {params['device']}")
    print("\n[XGBoost] Training...")

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    print("\n[XGBoost] Calibrating probabilities...")
    raw_val     = model.predict_proba(X_val)
    calibrators = []
    cal_probs   = np.zeros_like(raw_val)

    for cls in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val[:, cls], (y_val == cls).astype(int))
        cal_probs[:, cls] = iso.predict(raw_val[:, cls])
        calibrators.append(iso)

    cal_probs  = cal_probs / np.maximum(cal_probs.sum(axis=1, keepdims=True), 1e-8)
    calibrated = CalibratedModel(model, calibrators)

    val_preds   = np.argmax(cal_probs, axis=1)
    val_logloss = log_loss(y_val, cal_probs)
    val_acc     = accuracy_score(y_val, val_preds)

    for cls, name in LABEL_MAP.items():
        mask    = y_val == cls
        cls_acc = accuracy_score(y_val[mask], val_preds[mask]) if mask.sum() > 0 else 0
        print(f"  {name} accuracy: {cls_acc:.3f}  ({mask.sum()} samples)")

    print(f"\n  Validation log-loss : {val_logloss:.4f}")
    print(f"  Validation accuracy : {val_acc:.3f}")
    print(f"  Baseline (always HOME): {(y_val==0).mean():.3f}")

    metrics = {
        "val_log_loss":  round(val_logloss, 6),
        "val_accuracy":  round(float(val_acc), 4),
        "train_samples": int(len(X_train)),
        "val_samples":   int(len(X_val)),
    }

    version_tag = version_tag or f"xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_path   = os.path.join(MODEL_DIR, f"{version_tag}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(calibrated, f)

    print(f"\n  Saved: {save_path}")
    _register_model(version_tag, metrics, save_path, len(X_train))

    return calibrated, metrics


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(version_tag: Optional[str] = None) -> CalibratedModel:
    from backend.model.model_store import ensure_local

    if version_tag:
        local_path   = os.path.join(MODEL_DIR, f"{version_tag}.pkl")
        storage_path = f"checkpoints/{version_tag}.pkl"
    else:
        result = (
            supabase.table("model_versions")
            .select("version_tag, storage_path")
            .eq("model_type", "xgboost")
            .eq("is_active", True)
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise FileNotFoundError(
                "No active XGBoost model. Run: "
                "python -m backend.model.train --mode xgboost"
            )
        storage_path = result.data[0]["storage_path"]
        filename     = storage_path.split("/")[-1]
        local_path   = os.path.join(MODEL_DIR, filename)

    local_path = ensure_local(storage_path, local_path)

    with open(local_path, "rb") as f:
        return pickle.load(f)


def predict_probabilities(model: CalibratedModel, state_vector: np.ndarray) -> dict:
    probs = model.predict_proba(state_vector.reshape(1, -1))[0]
    return {
        "HOME": round(float(probs[0]), 4),
        "DRAW": round(float(probs[1]), 4),
        "AWAY": round(float(probs[2]), 4),
    }


def get_feature_importance(model: CalibratedModel) -> list[dict]:
    FEATURE_NAMES = [
        "home_elo", "away_elo", "elo_diff",
        "home_form_5", "away_form_5",
        "home_form_5_home", "away_form_5_away",
        "home_xg_scored", "away_xg_scored",
        "home_xg_conceded", "away_xg_conceded",
        "home_goals_avg", "away_goals_avg",
        "h2h_home_winrate",
        "injury_home", "injury_away",
    ]
    base = getattr(model, "estimator", model)
    if hasattr(base, "feature_importances_"):
        pairs = sorted(
            zip(FEATURE_NAMES, base.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]
    return []


# ─── DB ───────────────────────────────────────────────────────────────────────

def _register_model(version_tag, metrics, local_save_path, training_games):
    from backend.model.model_store import upload

    storage_path = upload(local_save_path)

    supabase.table("model_versions").update(
        {"is_active": False}
    ).eq("model_type", "xgboost").eq("is_active", True).execute()

    supabase.table("model_versions").insert({
        "version_tag":    version_tag,
        "model_type":     "xgboost",
        "training_games": training_games,
        "val_log_loss":   metrics["val_log_loss"],
        "is_active":      True,
        "storage_path":   storage_path,
        "notes":          json.dumps(metrics),
    }).execute()

    print(f"  Registered: {version_tag} (is_active=True)")