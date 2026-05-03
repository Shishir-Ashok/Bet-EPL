"""
backend/model/features.py
--------------------------
Builds the 16-dimensional state vector for every match.

Performance design:
  The naive approach calls build_state_vector() per match, which makes
  5-6 Supabase queries each time. For 1,520 matches that is ~9,000 network
  round trips — the GPU sits idle waiting for IO the entire time.

  This module instead uses bulk_fetch() to load ALL required data in
  4 queries upfront, builds in-memory lookup dicts, then computes every
  state vector with pure numpy. Zero DB calls inside the feature loop.

  Benchmark: ~9,000 queries (old) → 4 queries + numpy (new).

The 16 features per match:
  [0]  home_elo              ELO rating of home team (normalised 0-1)
  [1]  away_elo              ELO rating of away team (normalised 0-1)
  [2]  elo_diff              home_elo - away_elo (normalised -1 to 1)
  [3]  home_form_5           home PPG last 5 matches (0-1)
  [4]  away_form_5           away PPG last 5 matches (0-1)
  [5]  home_form_10          home PPG last 10 matches (0-1)
  [6]  away_form_10          away PPG last 10 matches (0-1)
  [7]  home_xg_avg           home avg xG scored last 5 (0-1)
  [8]  away_xg_avg           away avg xG scored last 5 (0-1)
  [9]  home_xg_conceded      home avg xG conceded last 5 (0-1)
  [10] away_xg_conceded      away avg xG conceded last 5 (0-1)
  [11] home_goals_avg        home avg goals scored last 5 (0-1)
  [12] away_goals_avg        away avg goals scored last 5 (0-1)
  [13] injury_impact_home    squad availability score 0-1
  [14] injury_impact_away    squad availability score 0-1
  [15] wallet_fraction       balance / starting_balance (capped 0-3)
  [16] prob_home             XGBoost HOME probability (appended by caller)
  [17] prob_draw             XGBoost DRAW probability (appended by caller)
  [18] prob_away             XGBoost AWAY probability (appended by caller)
"""

import os
import sys
import numpy as np
from typing import Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

# ─── Constants ────────────────────────────────────────────────────────────────

ELO_MIN          = 1200.0
ELO_MAX          = 1900.0
XG_MAX           = 4.0
GOALS_MAX        = 5.0
STARTING_BALANCE = 100.0
STATE_DIM        = 19

POSITION_WEIGHTS = {"GK": 0.15, "DEF": 0.10, "MID": 0.10, "FWD": 0.12}
DEFAULT_WEIGHT   = 0.10


# ─── Pagination helper ────────────────────────────────────────────────────────

def _paginate(query) -> list:
    """
    Pages through Supabase results 1000 rows at a time.

    Uses retries + a small inter-page delay to avoid triggering Supabase's
    HTTP/2 stream limit (error: ConnectionTerminated last_stream_id:19999).
    That error means we fired too many requests too fast on one connection.
    """
    import time

    rows, offset, page_size = [], 0, 1000
    max_retries = 3

    while True:
        for attempt in range(max_retries):
            try:
                page = query.range(offset, offset + page_size - 1).execute().data
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"\n  Supabase connection error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)

        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.2)  # 200ms between pages — prevents HTTP/2 stream exhaustion

    return rows


# ─── Bulk data loader ─────────────────────────────────────────────────────────

def bulk_fetch(match_ids: list) -> dict:
    """Loads all data in 4 queries. Returns cache dict."""
    print("  Bulk fetching training data (4 queries)...")

    # 1. ELO ratings
    elo_rows = _paginate(
        supabase.table("elo_ratings")
        .select("team_id, match_id, elo, calculated_at")
        .order("calculated_at", desc=False)
    )
    elo_by_team = defaultdict(list)
    for r in elo_rows:
        elo_by_team[r["team_id"]].append(
            (r["match_id"], float(r["elo"]), r["calculated_at"])
        )

    # 2. Results — NOW includes match id in each tuple for xG lookup
    result_rows = _paginate(
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, result, home_goals, away_goals")
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )
    results_by_team = defaultdict(list)
    for r in result_rows:
        kt = r["kickoff_time"]
        mid = r["id"]
        # Tuple now: (kickoff_time, result, is_home, home_goals, away_goals, match_id)
        results_by_team[r["home_team_id"]].append(
            (kt, r["result"], True,  r["home_goals"], r["away_goals"], mid)
        )
        results_by_team[r["away_team_id"]].append(
            (kt, r["result"], False, r["home_goals"], r["away_goals"], mid)
        )

    # 3. Match stats for real xG
    stats_rows = _paginate(
        supabase.table("match_stats")
        .select("match_id, home_xg, away_xg")
    )
    stats = {
        r["match_id"]: {
            "home_xg": float(r["home_xg"]) if r["home_xg"] is not None else None,
            "away_xg": float(r["away_xg"]) if r["away_xg"] is not None else None,
        }
        for r in stats_rows
    }

    # 4. Injury impact per team (unchanged)
    inj_rows = _paginate(
        supabase.table("team_injuries")
        .select("team_id, status, players(position)")
    )
    inj_by_team = defaultdict(list)
    for r in inj_rows:
        inj_by_team[r["team_id"]].append(r)

    injuries = {}
    for team_id, records in inj_by_team.items():
        total = 0.0
        for rec in records:
            pos    = (rec.get("players") or {}).get("position")
            weight = POSITION_WEIGHTS.get(pos, DEFAULT_WEIGHT)
            if rec["status"] == "Doubt":
                weight *= 0.5
            total += weight
        injuries[team_id] = round(min(total / 2.0, 1.0), 4)

    print(f"  Fetched: {len(elo_rows)} ELO, {len(result_rows)} results, "
          f"{len(stats)} stats, {len(injuries)} injury records")

    return {
        "elo":      dict(elo_by_team),
        "results":  dict(results_by_team),
        "stats":    stats,
        "injuries": injuries,
    }


# ─── In-memory feature helpers ────────────────────────────────────────────────

def _get_elo(elo_by_team: dict, team_id: int, before_time: str) -> float:
    entries = elo_by_team.get(team_id, [])
    elo     = 1500.0
    for _, rating, calc_at in entries:
        if calc_at < before_time:
            elo = rating
        else:
            break
    return elo


def _get_form(
    results_by_team: dict,
    team_id: int,
    n: int,
    before_time: str,
    venue: str = "all",   # "all", "home", "away"
) -> float:
    """
    Returns PPG (0–1) for last n matches, optionally filtered by venue.
    venue="home" → only home matches; venue="away" → only away matches.
    """
    all_entries = [e for e in results_by_team.get(team_id, []) if e[0] < before_time]

    if venue == "home":
        filtered = [e for e in all_entries if e[2]]      # is_home = True
    elif venue == "away":
        filtered = [e for e in all_entries if not e[2]]  # is_home = False
    else:
        filtered = all_entries

    recent = filtered[-n:]
    if not recent:
        return 0.5

    pts = 0
    for _, result, is_home, _, _, _ in recent:
        if (result == "HOME" and is_home) or (result == "AWAY" and not is_home):
            pts += 3
        elif result == "DRAW":
            pts += 1

    return round(pts / (len(recent) * 3), 4)


def _get_xg_stats(
    results_by_team: dict,
    team_id: int,
    n: int,
    before_time: str,
    stats_cache: dict | None = None,    # NEW: pass cache["stats"] for real xG
) -> tuple[float, float]:
    """
    Returns (avg_xg_scored, avg_xg_conceded) for the last n matches.

    Priority:
      1. Real xG from match_stats (most accurate — actual shot quality)
      2. Goals as xG proxy (fallback when stats not yet available)

    The stats_cache is keyed by match_id → {"home_xg", "away_xg"}.
    We look up each match in the results list and check stats first.
    """
    recent = [e for e in results_by_team.get(team_id, []) if e[0] < before_time][-n:]
    s, c   = [], []

    for kt, result, is_home, hg, ag, match_id in recent:
        # Try real xG first
        real = (stats_cache or {}).get(match_id)
        if real and real.get("home_xg") is not None and real.get("away_xg") is not None:
            xg_s = float(real["home_xg"] if is_home else real["away_xg"])
            xg_c = float(real["away_xg"] if is_home else real["home_xg"])
        elif hg is not None and ag is not None:
            # Goals fallback — consistent with the proxy used in scrape_stats
            xg_s = float(hg if is_home else ag) * 0.30  # same conversion as scrape_stats
            xg_c = float(ag if is_home else hg) * 0.30
        else:
            continue

        s.append(min(xg_s / XG_MAX, 1.0))
        c.append(min(xg_c / XG_MAX, 1.0))

    avg_s = round(float(np.mean(s)), 4) if s else 0.3
    avg_c = round(float(np.mean(c)), 4) if c else 0.3
    return avg_s, avg_c


def _get_goals_avg(results_by_team: dict, team_id: int, n: int, before_time: str) -> float:
    recent = [e for e in results_by_team.get(team_id, []) if e[0] < before_time][-n:]
    goals  = [
        min(float(hg if is_home else ag) / GOALS_MAX, 1.0)
        for _, _, is_home, hg, ag in recent
        if (hg if is_home else ag) is not None
    ]
    return round(float(np.mean(goals)), 4) if goals else 0.3


def _norm_elo(elo: float) -> float:
    return float(np.clip((elo - ELO_MIN) / (ELO_MAX - ELO_MIN), 0.0, 1.0))


# ─── State vector ─────────────────────────────────────────────────────────────

def build_state_vector(
    match_id:       int,
    home_team_id:   int,
    away_team_id:   int,
    kickoff_time:   str,
    wallet_balance: float,
    before_date:    Optional[str] = None,
    cache:          Optional[dict] = None,
) -> np.ndarray:
    """
    Returns a 19-dim state vector. Indices [16][17][18] are prob_home/draw/away
    and are initialised to 0.0 here — the caller APPENDS XGBoost probs after
    calling predict_probabilities(), it no longer overwrites existing indices.
    """
    cutoff = before_date or kickoff_time

    if cache:
        sc = cache["stats"]  # stats_cache for real xG
        home_elo_raw    = _get_elo(cache["elo"], home_team_id, cutoff)
        away_elo_raw    = _get_elo(cache["elo"], away_team_id, cutoff)
        home_form_5     = _get_form(cache["results"], home_team_id,  5, cutoff, "all")
        away_form_5     = _get_form(cache["results"], away_team_id,  5, cutoff, "all")
        home_form_home  = _get_form(cache["results"], home_team_id,  5, cutoff, "home")
        away_form_away  = _get_form(cache["results"], away_team_id,  5, cutoff, "away")
        home_xg_s, home_xg_c = _get_xg_stats(cache["results"], home_team_id, 5, cutoff, sc)
        away_xg_s, away_xg_c = _get_xg_stats(cache["results"], away_team_id, 5, cutoff, sc)
        home_goals      = _get_goals_avg(cache["results"], home_team_id, 5, cutoff)
        away_goals      = _get_goals_avg(cache["results"], away_team_id, 5, cutoff)
        injury_home     = cache["injuries"].get(home_team_id, 0.0)
        injury_away     = cache["injuries"].get(away_team_id, 0.0)
    else:
        home_elo_raw    = _db_get_elo(home_team_id)
        away_elo_raw    = _db_get_elo(away_team_id)
        home_form_5     = _db_get_form(home_team_id,  5, cutoff, "all")
        away_form_5     = _db_get_form(away_team_id,  5, cutoff, "all")
        home_form_home  = _db_get_form(home_team_id,  5, cutoff, "home")
        away_form_away  = _db_get_form(away_team_id,  5, cutoff, "away")
        home_xg_s, home_xg_c = _db_get_xg(home_team_id, 5, cutoff)
        away_xg_s, away_xg_c = _db_get_xg(away_team_id, 5, cutoff)
        home_goals      = _db_get_goals(home_team_id, 5, cutoff)
        away_goals      = _db_get_goals(away_team_id, 5, cutoff)
        injury_home     = _db_injury_impact(home_team_id)
        injury_away     = _db_injury_impact(away_team_id)

    home_elo    = _norm_elo(home_elo_raw)
    away_elo    = _norm_elo(away_elo_raw)
    elo_diff    = float(np.clip(
        (home_elo_raw - away_elo_raw) / (ELO_MAX - ELO_MIN), -1.0, 1.0
    ))
    wallet_frac = float(np.clip(wallet_balance / STARTING_BALANCE, 0.0, 3.0))

    return np.array([
        home_elo, away_elo, elo_diff,           # [0][1][2]
        home_form_5, away_form_5,               # [3][4]
        home_form_home, away_form_away,         # [5][6]  NEW — replaces form_10
        home_xg_s, away_xg_s,                  # [7][8]
        home_xg_c, away_xg_c,                  # [9][10]
        home_goals, away_goals,                 # [11][12]
        injury_home, injury_away,               # [13][14]
        wallet_frac,                            # [15]
        0.0, 0.0, 0.0,                          # [16][17][18] — XGB probs, filled by caller
    ], dtype=np.float32)


# ─── Bulk matrix builder (XGBoost training) ───────────────────────────────────

def build_feature_matrix(matches: list, wallet_balance: float = 10.0):
    """
    Builds the full feature matrix for all historical matches.
    Calls bulk_fetch() once, then computes everything in memory.

    Returns (X, y, match_ids).
    """
    LABEL_MAP = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    cache     = bulk_fetch([m["id"] for m in matches])
    X, y, ids = [], [], []
    total     = len(matches)

    print(f"  Computing {total} state vectors in-memory...")
    for i, m in enumerate(matches):
        if m.get("result") not in LABEL_MAP:
            continue
        if (i + 1) % 200 == 0 or i == total - 1:
            print(f"  {i+1}/{total}", end="\r")
        try:
            state = build_state_vector(
                match_id       = m["id"],
                home_team_id   = m["home_team_id"],
                away_team_id   = m["away_team_id"],
                kickoff_time   = m["kickoff_time"],
                wallet_balance = wallet_balance,
                before_date    = m["kickoff_time"],
                cache          = cache,
            )
            X.append(state)
            y.append(LABEL_MAP[m["result"]])
            ids.append(m["id"])
        except Exception as e:
            print(f"\n  Warning: skipping match {m['id']}: {e}")

    print(f"\n  Built {len(X)} feature vectors")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), ids


# ─── Slow-path DB fallbacks (inference only) ──────────────────────────────────

def _db_get_elo(team_id: int) -> float:
    r = supabase.table("elo_ratings").select("elo").eq("team_id", team_id)\
        .order("calculated_at", desc=True).limit(1).execute()
    return float(r.data[0]["elo"]) if r.data else 1500.0


def _db_get_form(team_id: int, n: int, before: str, venue: str = "all") -> float:
    rows = supabase.table("matches")\
        .select("id, home_team_id, away_team_id, result")\
        .eq("status", "FINISHED")\
        .or_(f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}")\
        .lt("kickoff_time", before)\
        .order("kickoff_time", desc=True)\
        .limit(n * 3)\
        .execute().data

    if venue == "home":
        rows = [r for r in rows if r["home_team_id"] == team_id]
    elif venue == "away":
        rows = [r for r in rows if r["away_team_id"] == team_id]

    rows = rows[:n]
    if not rows:
        return 0.5

    pts = sum(
        3 if (r["result"] == "HOME" and r["home_team_id"] == team_id) or
             (r["result"] == "AWAY" and r["away_team_id"] == team_id)
        else 1 if r["result"] == "DRAW" else 0
        for r in rows
    )
    return round(pts / (len(rows) * 3), 4)


def _db_get_xg(team_id: int, n: int, before: str) -> tuple:
    rows = supabase.table("matches")\
        .select("home_team_id, away_team_id, match_stats(home_xg, away_xg)")\
        .eq("status", "FINISHED").or_(f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}")\
        .lt("kickoff_time", before).order("kickoff_time", desc=True).limit(n).execute().data
    s, c = [], []
    for r in rows:
        st      = r.get("match_stats") or {}
        is_home = r["home_team_id"] == team_id
        xg_s    = st.get("home_xg" if is_home else "away_xg")
        xg_c    = st.get("away_xg" if is_home else "home_xg")
        if xg_s is not None:
            s.append(min(float(xg_s) / XG_MAX, 1.0))
        if xg_c is not None:
            c.append(min(float(xg_c) / XG_MAX, 1.0))
    return (round(float(np.mean(s)), 4) if s else 0.3,
            round(float(np.mean(c)), 4) if c else 0.3)


def _db_get_goals(team_id: int, n: int, before: str) -> float:
    rows = supabase.table("matches")\
        .select("home_team_id, home_goals, away_goals")\
        .eq("status", "FINISHED").or_(f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}")\
        .lt("kickoff_time", before).order("kickoff_time", desc=True).limit(n).execute().data
    goals = [
        min(float(r["home_goals"] if r["home_team_id"] == team_id else r["away_goals"]) / GOALS_MAX, 1.0)
        for r in rows if r.get("home_goals") is not None
    ]
    return round(float(np.mean(goals)), 4) if goals else 0.3


def _db_injury_impact(team_id: int) -> float:
    rows = supabase.table("team_injuries")\
        .select("status, players(position)").eq("team_id", team_id).execute().data
    if not rows:
        return 0.0
    total = sum(
        POSITION_WEIGHTS.get((r.get("players") or {}).get("position"), DEFAULT_WEIGHT)
        * (0.5 if r["status"] == "Doubt" else 1.0)
        for r in rows
    )
    return round(min(total / 2.0, 1.0), 4)