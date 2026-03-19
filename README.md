# PL Betting Bot

A self-running machine learning system that watches every Premier League match,
predicts outcomes, places virtual bets, and learns from wins and losses — all
on a free cloud stack with a public live dashboard.

**Starting bankroll:** €100.00 &nbsp;|&nbsp; **Status:** [Live website →](https://bet-epl-phi.vercel.app/)

---

## How it works

```
football-data.org  ──┐
The Odds API       ──┼──▶  GitHub Actions (cron)
FBref / Understat  ──┘         │
                               ▼
                       Supabase PostgreSQL
                               │
                               ▼
                     XGBoost (outcome probs)
                               │
                               ▼
                     DQN Agent (bet or pass?)
                               │
                               ▼
                       Virtual Bet Logger
                               │
                               ▼
                   Next.js Dashboard on Vercel
```

The model never bets real money. The €100 is a virtual bankroll. All bets are
hypothetical and logged for analysis.

---

## Project structure

```
pl-betting-bot/
│
├── .github/
│   └── workflows/
│       ├── fetch_prematch.yml      # Runs 3h before each PL matchday
│       ├── ingest_results.yml      # Runs the morning after matches finish
│       └── weekly_retrain.yml      # Retrains XGBoost + DQN every Monday
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql  # ← Run this first in Supabase SQL editor
│
├── backend/
│   ├── requirements.txt
│   ├── data_pipeline/
│   │   ├── fetch_fixtures.py       # football-data.org → matches table
│   │   ├── fetch_odds.py           # The Odds API → odds table
│   │   ├── scrape_stats.py         # FBref + Understat → match_stats
│   │   ├── scrape_injuries.py      # BBC Sport → team_injuries
│   │   └── update_elo.py           # Calculate ELO after each result
│   ├── model/
│   │   ├── features.py             # Feature engineering (16-dim state vector)
│   │   ├── xgboost_model.py        # Train / predict with XGBoost
│   │   ├── dqn_agent.py            # Deep Q-Network definition + training loop
│   │   ├── replay_buffer.py        # Experience replay (loads from rl_episodes)
│   │   └── train.py                # Orchestrates full training pipeline
│   ├── engine/
│   │   ├── kelly.py                # Kelly Criterion stake sizing
│   │   ├── bet_placer.py           # Writes bets to DB, updates wallet
│   │   └── settle_bets.py          # Settles open bets after results come in
│   └── api/
│       └── main.py                 # FastAPI server — triggered by GitHub Actions
│
├── frontend/                       # Next.js app (Phase 7)
│   ├── app/
│   │   ├── page.tsx                # Home: live balance + last 5 bets
│   │   └── dashboard/page.tsx      # Charts, filters, history
│   ├── components/
│   └── lib/
│       └── supabase.ts             # Supabase client (anon key, read-only)
│
├── scripts/
│   └── bootstrap_db.py             # ← Run this after the SQL migration
│
├── .env.example                    # Copy to .env and fill in keys
└── .gitignore
```

---

## Setup — Phase 1: Database

### Step 1 — Create a Supabase project

1. Go to [supabase.com](https://supabase.com) → New project
2. Choose a name (e.g. `pl-betting-bot`) and a strong DB password
3. Pick a region (closest to UK for PL data latency)
4. Wait ~2 minutes for provisioning

### Step 2 — Run the schema migration

1. In your Supabase dashboard: **SQL Editor → New query**
2. Paste the entire contents of `supabase/migrations/001_initial_schema.sql`
3. Click **Run** — you should see "Success. No rows returned"
4. Go to **Table Editor** — you should see 13 tables

### Step 3 — Get your API keys

From Supabase: **Settings → API**

- Copy `Project URL` → `SUPABASE_URL`
- Copy `anon / public` key → `SUPABASE_ANON_KEY`
- Copy `service_role` key → `SUPABASE_SERVICE_KEY`

```bash
cp .env.example .env
# Edit .env with your values
```

### Step 4 — Bootstrap the database

```bash
cd pl-betting-bot
pip install supabase python-dotenv
python scripts/bootstrap_db.py
```

Expected output:

```
=======================================================
  PL Betting Bot — Database Bootstrap
=======================================================

[1/3] Connecting to Supabase...
      ✓ Connected successfully
[2/3] Seeding Premier League teams...
      ✓ 20 teams upserted
[3/3] Checking wallet...
      ✓ Wallet found
        Balance:       €100.00
        Inception:     2024-xx-xx
```

### Step 5 — Verify in Supabase Table Editor

| Table           | Expected rows after bootstrap |
| --------------- | ----------------------------- |
| `teams`         | 20                            |
| `wallet`        | 1 (€100.00 balance)           |
| `matches`       | 0 (filled by Phase 2)         |
| `bets`          | 0 (filled by Phase 3+)        |
| everything else | 0                             |

---

## Phases

| Phase | What gets built                                 |
| ----- | ----------------------------------------------- |
| 1     | DB schema + Supabase setup                      |
| 2     | Data pipeline (fixtures, odds, stats, injuries) |
| 3     | XGBoost outcome predictor                       |
| 4     | DQN betting agent                               |
| 5     | Bet engine + virtual wallet                     |
| 6     | FastAPI on Render                               |
| 7     | Next.js frontend on Vercel                      |
| 8     | GitHub Actions automation                       |

---

## Free tier limits

| Service           | Free limit                         | Our usage     |
| ----------------- | ---------------------------------- | ------------- |
| Supabase          | 500MB DB, 1GB storage              | ~5MB/season   |
| Vercel            | Unlimited deploys, 100GB bandwidth | ~1GB/month    |
| Render            | 750h/month (spins down)            | ~2h/month     |
| GitHub Actions    | 2,000 min/month                    | ~60 min/month |
| football-data.org | 10 calls/min, all PL data          | ~500/season   |
| The Odds API      | 500 requests/month                 | ~380/season   |

All comfortably within free limits.

---
