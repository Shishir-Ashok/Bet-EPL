# PL Betting Bot

A self-running machine learning system that watches every Premier League match,
predicts outcomes, places virtual bets, and learns from wins and losses — all
on a free cloud stack with a public live dashboard.

**Starting bankroll:** €100.00 &nbsp;|&nbsp; **Status:** [Live dashboard →](https://bet-epl-phi.vercel.app/)

---

## How it works

```
football-data.org        ──┐
football-data.co.uk      ──┼──▶  GitHub Actions (cron)
FDCO CSVs (xG proxy)     ──┘           │
                                       ▼
                               Supabase PostgreSQL
                                       │
                           ┌───────────┴───────────┐
                           ▼                       ▼
                   XGBoost predictor         ELO ratings
                   (3-class softprob)        (K=32, HFA=65)
                           │
                           ▼
                    DQN betting agent
                    (Q-gate: bet or pass?)
                           │
                           ▼
                    Kelly Criterion
                    (33% fractional, 20% cap)
                           │
                           ▼
                    Virtual bet logger
                           │
                           ▼
                Next.js dashboard on Vercel
```

The model never bets real money. The €100 is a virtual bankroll. All bets are
hypothetical and logged for analysis.

---

## Model architecture

### XGBoost (outcome predictor)

- **Input:** 16-dim feature vector per match
- **Output:** calibrated probabilities for HOME / DRAW / AWAY
- **Features:** ELO ratings, form (5 + 10 games), xG scored/conceded, H2H win rate, injury impact
- **Training:** Chronological 85/15 split, validates on most recent season
- **Calibration:** Per-class isotonic regression on validation set
- **Leakage guard:** `before_date` parameter caps both training labels and the feature cache

### DQN agent (betting decision)

- **State:** 24-dim (16 XGB features + 3 XGBoost probs + 3 implied probs + edge + wallet fraction)
- **Actions:** `BET_HOME`, `BET_DRAW`, `BET_AWAY`, `PASS`
- **Reward:** normalised P&L per unit staked, clipped ±2
- **Gate:** `Q(bet) − Q(PASS) ≥ threshold` to confirm a Kelly suggestion

### Kelly Criterion (stake sizing)

- Bookmaker overround removed via proportional de-vig
- Minimum edge threshold: 2% after de-vig
- Fractional Kelly: 33% of full Kelly fraction
- Hard cap: 20% of bankroll per bet

---

## Odds source

Historical and live odds come from **football-data.co.uk** (free, no API key).
Pinnacle **opening** lines (`PSH/PSD/PSA`) are used — not closing lines — because
closing odds reflect late sharp money you can never actually bet at. Bet365
and market-max lines serve as fallbacks.

---

## Retraining schedule

| Season                     | Trigger       | What retrains                              |
| -------------------------- | ------------- | ------------------------------------------ |
| 2024-25 MD 19 (≈ Jan 2025) | Mid-season    | XGBoost (base seasons) + first DQN         |
| 2024-25 MD 38              | End of season | XGBoost (incl. 2024-25) + full DQN retrain |
| 2025-26 MD 19 (≈ Jan 2026) | Mid-season    | XGBoost (completed seasons) + DQN update   |
| 2025-26 MD 38              | End of season | XGBoost (incl. 2025-26) + full DQN retrain |

XGBoost only ever trains on **completed** seasons to avoid partial-season distributional instability. The DQN handles current-season adaptation continuously via the replay buffer.

---

## Project structure

```
pl-betting-bot/
│
├── .github/
│   └── workflows/
│       ├── matchday_orchestrator.yml   # Runs 3h before each PL matchday
│       ├── ingest_results.yml          # Runs the morning after matches finish
│       └── monthly_retrain.yml         # Mid-season and end-of-season retrains
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql      # ← Run this first in Supabase SQL editor
│
├── backend/
│   ├── requirements.txt
│   ├── data_pipeline/
│   │   ├── fetch_fixtures.py           # football-data.org → matches table
│   │   ├── fetch_historical_odds.py    # football-data.co.uk → odds table (Pinnacle opening)
│   │   ├── fetch_odds.py               # Live pre-match odds → odds table
│   │   ├── odds_validator.py           # Detects and corrects swapped home/away odds
│   │   ├── scrape_stats.py             # FDCO CSVs → match_stats (xG proxy: 0.30 per SOT)
│   │   ├── scrape_injuries.py          # Squad availability → team_injuries
│   │   └── update_elo.py               # ELO ratings (K=32, HFA=65) with kickoff timestamps
│   ├── model/
│   │   ├── features.py                 # Feature engineering — bulk_fetch with before_date guard
│   │   ├── xgboost_model.py            # Train / predict — before_date flows to bulk_fetch
│   │   ├── dqn_agent.py                # Deep Q-Network definition + training loop
│   │   ├── replay_buffer.py            # Experience replay (loads from rl_episodes)
│   │   └── train.py                    # Orchestrates full training pipeline
│   ├── engine/
│   │   ├── kelly.py                    # Kelly Criterion stake sizing (33% fractional)
│   │   ├── bet_placer.py               # Writes bets to DB, updates wallet
│   │   └── settle_bets.py              # Settles open bets, pushes RL transitions
│   └── api/
│       └── main.py                     # FastAPI server — triggered by GitHub Actions
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                    # Overview: live balance + upcoming fixtures
│   │   ├── dashboard/page.tsx          # Charts and filters
│   │   ├── history/page.tsx            # Full bet history
│   │   └── docs/page.tsx               # Technical documentation
│   ├── components/
│   │   ├── ClubBadge.tsx
│   │   ├── Navbar.tsx
│   │   └── SortTh.tsx
│   └── lib/
│       └── supabase.ts                 # Supabase client (anon key, read-only)
│
├── scripts/
│   ├── seed_historical_data.py         # One-time: fetch fixtures + stats + ELO for 2020-24
│   ├── simulate_historical_bets.py     # Backfill bets for 2024-25 and 2025-26
│   ├── run_simulation.py               # Orchestrator with mid/end-season retrains
│   └── bootstrap_db.py                 # Seeds teams and wallet row
│
└── .env.example
```

---

## Setup

### Step 1 — Database

1. Create a [Supabase](https://supabase.com) project
2. **SQL Editor → New query** → paste `supabase/migrations/001_initial_schema.sql` → Run
3. Copy your `Project URL`, `anon key`, and `service_role key` from **Settings → API**

```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY
```

### Step 2 — Bootstrap

```bash
pip install -r backend/requirements.txt
python scripts/bootstrap_db.py
```

### Step 3 — Historical data

```bash
# Fetch fixtures + stats + ELO for 2020-24 (training data)
python scripts/seed_historical_data.py

# Recalculate ELO with correct timestamps (critical — must run before training)
python -m backend.data_pipeline.update_elo --all-seasons

# Fetch Pinnacle opening odds for 2024-25 and 2025-26 (free, no API key)
python -m backend.data_pipeline.fetch_historical_odds
```

### Step 4 — Train and simulate

```bash
# Train base XGBoost on 2020-24
python -m backend.model.train --mode xgboost

# Run full simulation (2024-25 + 2025-26 with mid/end-season retrains)
python scripts/run_simulation.py
```

### Step 5 — Deploy

1. Deploy the FastAPI backend to [Render](https://render.com) (free tier)
2. Deploy the Next.js frontend to [Vercel](https://vercel.com) (free tier)
3. Add GitHub Actions secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `RENDER_DEPLOY_HOOK`

---

## Free tier usage

| Service             | Free limit            | Our usage                |
| ------------------- | --------------------- | ------------------------ |
| Supabase            | 500MB DB, 1GB storage | ~5MB/season              |
| Vercel              | Unlimited deploys     | ~1GB/month bandwidth     |
| Render              | 750h/month            | ~2h/month                |
| GitHub Actions      | 2,000 min/month       | ~1500 min/month          |
| football-data.org   | 10 calls/min          | ~500/season              |
| football-data.co.uk | Unlimited             | ~10 CSV downloads/season |
| the-odds-api.com    | 500 calls/month       | ~60-85 calls/month       |

All comfortably within free limits. No credit card required.

---

## Important implementation notes

**ELO timestamps matter.** `elo_ratings.calculated_at` must be set to the match
`kickoff_time`, not the script run time. `features.py` filters ELO with
`calc_at < before_time` — if `calculated_at` is a 2026 timestamp, every team
returns 1500.0 for every historical match, killing the strongest predictor.

**Opening odds, not closing.** Using `PSCH/PSCD/PSCA` (Pinnacle closing) as the
bet price is lookahead bias — you can never actually get closing odds. The fetcher
uses `PSH/PSD/PSA` (opening) with Bet365 as fallback.

**before_date flows all the way down.** The leakage guard must propagate:
`run_simulation → train → load_training_data → build_feature_matrix → bulk_fetch`.
If any link in this chain calls `bulk_fetch()` without `before_date`, the feature
cache loads future data and XGBoost memorises it.
