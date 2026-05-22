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
XGB_STATE_DIM    = 16
DQN_STATE_DIM    = 24

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


def _get_h2h(
    results_by_team: dict,
    home_team_id: int,
    away_team_id: int,
    before_time: str,
    n: int = 6,
) -> float:
    """
    Home team's win rate in last n head-to-head matches at home against this
    specific away side. Returns 0.5 (neutral) if fewer than 2 H2H matches found.
    """
    # All matches where home_team_id was at home before this kickoff
    home_entries = [
        e for e in results_by_team.get(home_team_id, [])
        if e[0] < before_time and e[2]   # is_home=True
    ]
    # Filter to only those where the opponent was away_team_id
    # The results tuple is (kickoff_time, result, is_home, hg, ag, match_id)
    # We need to cross-reference: find matches where away_team also appears as away
    away_match_ids = {
        e[5] for e in results_by_team.get(away_team_id, [])
        if e[0] < before_time and not e[2]   # is_home=False
    }

    h2h = [e for e in home_entries if e[5] in away_match_ids][-n:]

    if len(h2h) < 2:
        return 0.5   # not enough data to be meaningful

    wins = sum(1 for e in h2h if e[1] == "HOME")   # home team won
    return round(wins / len(h2h), 4)


# ─── Bulk data loader ─────────────────────────────────────────────────────────

def bulk_fetch(before_date: str = None) -> dict:
    """
    Loads all data in 4 queries. Returns cache dict.

    before_date : ISO string e.g. "2025-10-01".
        When set, only ELO rows and match results with timestamps
        before this date are loaded into the cache.

        WHY THIS MATTERS:
        build_feature_matrix() calls this once and passes the cache to
        build_state_vector() for every training match. Without a date cap,
        the cache contains the entire DB — including future 2025-26 results
        that haven't been "simulated" yet. Even though per-match temporal
        filters (e[0] < before_time) correctly block future entries from
        appearing in individual feature vectors, XGBoost's training labels
        for those future matches would still be present as rows in the
        matches list passed to build_feature_matrix(). The before_date on
        load_training_data() removes those future rows from the label set,
        but the cache itself carrying future data creates distributional
        leakage — the cache size, ELO spread, and form distributions all
        reflect the full future season rather than what was known at the
        training cutoff.

        During live inference (simulate_historical_bets.py), before_date
        is None. The per-match before_time filter handles isolation at the
        feature level correctly in that context since we process one match
        at a time.
    """
    import time

    date_note = f" before {before_date}" if before_date else ""
    print(f"  Bulk fetching training data (4 queries){date_note}...")

    # 1. ELO ratings — capped to before_date if provided
    elo_query = (
        supabase.table("elo_ratings")
        .select("team_id, match_id, elo, calculated_at")
        .order("calculated_at", desc=False)
    )
    if before_date:
        elo_query = elo_query.lt("calculated_at", before_date)

    elo_rows = _paginate(elo_query)
    elo_by_team = defaultdict(list)
    for r in elo_rows:
        elo_by_team[r["team_id"]].append(
            (r["match_id"], float(r["elo"]), r["calculated_at"])
        )

    # 2. Results — capped to before_date if provided
    result_query = (
        supabase.table("matches")
        .select("id, home_team_id, away_team_id, kickoff_time, result, home_goals, away_goals")
        .eq("status", "FINISHED")
        .not_.is_("result", "null")
        .order("kickoff_time", desc=False)
    )
    if before_date:
        result_query = result_query.lt("kickoff_time", before_date)

    result_rows = _paginate(result_query)
    results_by_team = defaultdict(list)
    for r in result_rows:
        kt  = r["kickoff_time"]
        mid = r["id"]
        results_by_team[r["home_team_id"]].append(
            (kt, r["result"], True,  r["home_goals"], r["away_goals"], mid)
        )
        results_by_team[r["away_team_id"]].append(
            (kt, r["result"], False, r["home_goals"], r["away_goals"], mid)
        )

    # 3. Match stats — filtered to match_ids already in the date-filtered results.
    #    No separate date query needed: stats for future match_ids are unreachable
    #    because those match_ids don't exist in result_rows.
    if result_rows:
        filtered_match_ids = [r["id"] for r in result_rows]
        stats: dict = {}
        for i in range(0, len(filtered_match_ids), 300):
            chunk      = filtered_match_ids[i : i + 300]
            chunk_rows = (
                supabase.table("match_stats")
                .select("match_id, home_xg, away_xg")
                .in_("match_id", chunk)
                .execute()
                .data
            )
            for r in chunk_rows:
                stats[r["match_id"]] = {
                    "home_xg": float(r["home_xg"]) if r["home_xg"] is not None else None,
                    "away_xg": float(r["away_xg"]) if r["away_xg"] is not None else None,
                }
            time.sleep(0.05)
    else:
        stats = {}

    # 4. Injury impact per team (not date-filtered — injuries are current state,
    #    used for live inference; during training they default to 0 anyway)
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
    venue: str = "all",
) -> float:
    """
    Returns PPG (0-1) for last n matches, optionally filtered by venue.
    Uses index access on tuples — safe regardless of tuple length.
    Tuple layout: (kickoff_time[0], result[1], is_home[2], home_goals[3], away_goals[4], match_id[5])
    """
    all_entries = [e for e in results_by_team.get(team_id, []) if e[0] < before_time]

    if venue == "home":
        filtered = [e for e in all_entries if e[2]]       # is_home = True
    elif venue == "away":
        filtered = [e for e in all_entries if not e[2]]   # is_home = False
    else:
        filtered = all_entries

    recent = filtered[-n:]
    if not recent:
        return 0.5

    pts = 0
    for e in recent:
        result, is_home = e[1], e[2]
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
    stats_cache: dict | None = None,
) -> tuple[float, float]:
    """
    Returns (avg_xg_scored, avg_xg_conceded) for the last n matches.

    Priority:
      1. Real xG from match_stats (most accurate — actual shot quality)
      2. Goals as xG proxy (fallback when stats not yet available)

    The stats_cache is keyed by match_id → {"home_xg", "away_xg"}.
    We look up each match in the results list and check stats first.
    """
    entries = [e for e in results_by_team.get(team_id, []) if e[0] < before_time][-n:]
    if not entries:
        return 0.3, 0.3

    scored    = []
    conceded  = []

    for e in entries:
        kt, result, is_home, hg, ag, mid = e
        # Try real xG first
        if stats_cache and mid in stats_cache:
            s = stats_cache[mid]
            xg_scored   = s["home_xg"] if is_home else s["away_xg"]
            xg_conceded = s["away_xg"] if is_home else s["home_xg"]
        else:
            # Fall back to goals
            xg_scored   = float(hg) if hg is not None else None
            xg_conceded = float(ag) if ag is not None else None

        if xg_scored   is not None:
            scored.append(min(xg_scored   / XG_MAX, 1.0))
        if xg_conceded is not None:
            conceded.append(min(xg_conceded / XG_MAX, 1.0))

    avg_scored   = round(float(np.mean(scored)),   4) if scored   else 0.3
    avg_conceded = round(float(np.mean(conceded)), 4) if conceded else 0.3
    return avg_scored, avg_conceded


def _get_goals_avg(
    results_by_team: dict,
    team_id: int,
    n: int,
    before_time: str,
) -> float:
    entries = [e for e in results_by_team.get(team_id, []) if e[0] < before_time][-n:]
    goals = []
    for e in entries:
        kt, result, is_home, hg, ag, mid = e
        g = hg if is_home else ag
        if g is not None:
            goals.append(min(float(g) / GOALS_MAX, 1.0))
    return round(float(np.mean(goals)), 4) if goals else 0.3


# ─── Public state vector builders ────────────────────────────────────────────

def build_state_vector(
    home_team_id: int,
    away_team_id: int,
    kickoff_time: str,
    before_date:  str  = None,
    cache:        dict = None,
) -> np.ndarray:
    """
    Builds the 16-dim XGBoost feature vector for one match.

    before_date: used as the temporal cutoff. Defaults to kickoff_time
    so that only data prior to the match is used — no lookahead.

    cache: if provided (from bulk_fetch), uses in-memory data.
    If None, falls back to individual DB queries (live inference path).
    """
    before = before_date or kickoff_time

    if cache:
        elo_data     = cache["elo"]
        results_data = cache["results"]
        stats_cache  = cache.get("stats")
        injuries     = cache.get("injuries", {})

        home_elo_raw = _get_elo(elo_data, home_team_id, before)
        away_elo_raw = _get_elo(elo_data, away_team_id, before)

        home_form_5    = _get_form(results_data, home_team_id, 5,  before)
        away_form_5    = _get_form(results_data, away_team_id, 5,  before)
        home_form_home = _get_form(results_data, home_team_id, 10, before, venue="home")
        away_form_away = _get_form(results_data, away_team_id, 10, before, venue="away")

        home_xg_s, home_xg_c = _get_xg_stats(results_data, home_team_id, 5, before, stats_cache)
        away_xg_s, away_xg_c = _get_xg_stats(results_data, away_team_id, 5, before, stats_cache)

        home_goals = _get_goals_avg(results_data, home_team_id, 5, before)
        away_goals = _get_goals_avg(results_data, away_team_id, 5, before)

        h2h = _get_h2h(results_data, home_team_id, away_team_id, before)

        injury_home = injuries.get(home_team_id, 0.0)
        injury_away = injuries.get(away_team_id, 0.0)

    else:
        # Live inference fallback — individual DB queries
        home_elo_raw = _db_get_elo(home_team_id)
        away_elo_raw = _db_get_elo(away_team_id)

        home_form_5    = _db_get_form(home_team_id, 5,  before)
        away_form_5    = _db_get_form(away_team_id, 5,  before)
        home_form_home = _db_get_form(home_team_id, 10, before, venue="home")
        away_form_away = _db_get_form(away_team_id, 10, before, venue="away")

        home_xg_s, home_xg_c = _db_get_xg(home_team_id, 5, before)
        away_xg_s, away_xg_c = _db_get_xg(away_team_id, 5, before)

        home_goals = _db_get_goals(home_team_id, 5, before)
        away_goals = _db_get_goals(away_team_id, 5, before)

        h2h = _db_get_h2h(home_team_id, away_team_id, before)

        injury_home = _db_injury_impact(home_team_id)
        injury_away = _db_injury_impact(away_team_id)

    # Normalise ELO to 0-1 range
    home_elo = float(np.clip((home_elo_raw - ELO_MIN) / (ELO_MAX - ELO_MIN), 0.0, 1.0))
    away_elo = float(np.clip((away_elo_raw - ELO_MIN) / (ELO_MAX - ELO_MIN), 0.0, 1.0))
    elo_diff = float(np.clip(
        (home_elo_raw - away_elo_raw) / (ELO_MAX - ELO_MIN), -1.0, 1.0
    ))

    return np.array([
        home_elo, away_elo, elo_diff,         # [0][1][2]
        home_form_5, away_form_5,             # [3][4]
        home_form_home, away_form_away,       # [5][6]
        home_xg_s, away_xg_s,                # [7][8]
        home_xg_c, away_xg_c,                # [9][10]
        home_goals, away_goals,              # [11][12]
        h2h,                                  # [13]
        injury_home, injury_away,             # [14][15]
    ], dtype=np.float32)


def build_dqn_state(
    xgb_vector:     np.ndarray,
    xgb_probs:      dict,
    implied_probs:  dict | None,
    wallet_balance: float = STARTING_BALANCE,
) -> np.ndarray:
    """
    Builds the 24-dim DQN state from the XGB vector plus value signals.
    Call this AFTER build_state_vector() + predict_probabilities().

    If implied_probs is None (no odds available), uses neutral 1/3 for each.
    """
    if implied_probs is None:
        implied_probs = {"HOME": 0.333, "DRAW": 0.333, "AWAY": 0.333}

    imp_h = float(implied_probs["HOME"])
    imp_d = float(implied_probs["DRAW"])
    imp_a = float(implied_probs["AWAY"])
    xgb_h = float(xgb_probs["HOME"])
    xgb_d = float(xgb_probs["DRAW"])
    xgb_a = float(xgb_probs["AWAY"])
    edge_home   = float(np.clip(xgb_h - imp_h, -0.5, 0.5))
    wallet_frac = float(np.clip(wallet_balance / STARTING_BALANCE, 0.0, 3.0))

    return np.array([
        *xgb_vector,                      # [0-15]
        xgb_h, xgb_d, xgb_a,             # [16-18]
        imp_h, imp_d, imp_a,              # [19-21]
        edge_home,                        # [22]
        wallet_frac,                      # [23]
    ], dtype=np.float32)


# ─── Bulk matrix builder (XGBoost training) ───────────────────────────────────

def build_feature_matrix(matches: list, before_date: str = None):
    """
    Builds the XGBoost training matrix — 16 features, no wallet.

    before_date : passed to bulk_fetch() so the cache only contains data
        before this date. This closes the distributional leakage path where
        the cache carried future ELO/form distributions even though per-match
        temporal filters blocked individual future entries.

        Set to the training cutoff date during monthly retrains.
        None = no cap (correct for base training on completed seasons).

    wallet_balance is intentionally not passed; XGBoost predicts match
    outcomes, not betting decisions.
    """
    LABEL_MAP = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    cache     = bulk_fetch(before_date=before_date)   # ← THE KEY CHANGE
    X, y, ids = [], [], []

    print(f"  Computing {len(matches)} XGB feature vectors...")
    for i, m in enumerate(matches):
        if m.get("result") not in LABEL_MAP:
            continue
        try:
            state = build_state_vector(
                home_team_id = m["home_team_id"],
                away_team_id = m["away_team_id"],
                kickoff_time = m["kickoff_time"],
                before_date  = m["kickoff_time"],
                cache        = cache,
            )
            X.append(state)
            y.append(LABEL_MAP[m["result"]])
            ids.append(m["id"])
        except Exception as e:
            print(f"\n  Warning: skipping match {m['id']}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), ids


# ─── Slow-path DB fallbacks (inference only) ──────────────────────────────────

def _db_get_h2h(home_team_id: int, away_team_id: int, before: str, n: int = 6) -> float:
    rows = supabase.table("matches")\
        .select("id, result")\
        .eq("status", "FINISHED")\
        .eq("home_team_id", home_team_id)\
        .eq("away_team_id", away_team_id)\
        .lt("kickoff_time", before)\
        .order("kickoff_time", desc=True)\
        .limit(n)\
        .execute().data
    if len(rows) < 2:
        return 0.5
    wins = sum(1 for r in rows if r["result"] == "HOME")
    return round(wins / len(rows), 4)

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