# Data dictionary

Complete reference for every table, view, and column in the PL Betting Bot database.

---

## Tables

### `teams`

One row per club. Seeded by `bootstrap_db.py`. Historical promoted/relegated clubs are added automatically when fixtures are imported.

| Column       | Type        | Description                                                   |
| ------------ | ----------- | ------------------------------------------------------------- |
| `id`         | serial PK   | Internal team identifier                                      |
| `name`       | text        | Full official name — e.g. `Arsenal FC`                        |
| `short_name` | text        | Display name used throughout the app — e.g. `Arsenal`         |
| `tla`        | text        | Three-letter abbreviation from football-data.org — e.g. `ARS` |
| `crest_url`  | text        | URL to the club badge image                                   |
| `created_at` | timestamptz | Row creation timestamp                                        |

---

### `players`

Basic player reference. Stats per match live in `player_match_stats`. Populated by scraping when player-level data is available.

| Column          | Type           | Description                              |
| --------------- | -------------- | ---------------------------------------- |
| `id`            | serial PK      | Internal player identifier               |
| `team_id`       | int FK → teams | Current club (set null if player leaves) |
| `name`          | text           | Full player name                         |
| `position`      | text           | `GK`, `DEF`, `MID`, or `FWD`             |
| `nationality`   | text           | Country of nationality                   |
| `date_of_birth` | date           | Player's date of birth                   |
| `created_at`    | timestamptz    | Row creation timestamp                   |
| `updated_at`    | timestamptz    | Last update timestamp                    |

---

### `matches`

One row per Premier League fixture. The central table everything else joins to.

| Column         | Type           | Description                                                                                                        |
| -------------- | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `id`           | serial PK      | Internal match identifier                                                                                          |
| `external_id`  | bigint         | Source system match ID. football-data.org integer for current season; md5-derived hash for FDCO historical imports |
| `home_team_id` | int FK → teams | Home side                                                                                                          |
| `away_team_id` | int FK → teams | Away side                                                                                                          |
| `kickoff_time` | timestamptz    | Scheduled kickoff in UTC                                                                                           |
| `matchday`     | int            | Gameweek number (1–38). Null for FDCO historical imports which don't include this                                  |
| `season`       | text           | Season label — e.g. `2024-25`                                                                                      |
| `status`       | text           | `SCHEDULED`, `IN_PLAY`, `FINISHED`, or `POSTPONED`                                                                 |
| `home_goals`   | int            | Full-time home goals. Null until match finishes                                                                    |
| `away_goals`   | int            | Full-time away goals. Null until match finishes                                                                    |
| `result`       | text           | `HOME`, `DRAW`, or `AWAY`. Null until match finishes                                                               |
| `created_at`   | timestamptz    | Row creation timestamp                                                                                             |
| `updated_at`   | timestamptz    | Last update timestamp                                                                                              |

---

### `match_stats`

Aggregated per-match team statistics. One row per match, populated after the match finishes by `scrape_stats.py`. xG values are shots-on-target × 0.30 (proxy) unless upgraded by `fetch_fbref_xg.py`.

| Column                 | Type             | Description                              |
| ---------------------- | ---------------- | ---------------------------------------- |
| `id`                   | serial PK        |                                          |
| `match_id`             | int FK → matches | The match these stats belong to          |
| `home_xg`              | numeric(5,2)     | Home team expected goals                 |
| `away_xg`              | numeric(5,2)     | Away team expected goals                 |
| `home_shots`           | int              | Total home shots                         |
| `away_shots`           | int              | Total away shots                         |
| `home_shots_on_target` | int              | Home shots on target                     |
| `away_shots_on_target` | int              | Away shots on target                     |
| `home_possession`      | numeric(4,1)     | Home possession percentage               |
| `away_possession`      | numeric(4,1)     | Away possession percentage               |
| `home_passes`          | int              | Total home passes                        |
| `away_passes`          | int              | Total away passes                        |
| `home_pass_accuracy`   | numeric(4,1)     | Home pass accuracy percentage            |
| `away_pass_accuracy`   | numeric(4,1)     | Away pass accuracy percentage            |
| `home_tackles`         | int              | Home tackles                             |
| `away_tackles`         | int              | Away tackles                             |
| `home_interceptions`   | int              | Home interceptions                       |
| `away_interceptions`   | int              | Away interceptions                       |
| `home_pressures`       | int              | Home pressing actions (FBref definition) |
| `away_pressures`       | int              | Away pressing actions                    |
| `home_corners`         | int              | Home corners                             |
| `away_corners`         | int              | Away corners                             |
| `home_fouls`           | int              | Home fouls committed                     |
| `away_fouls`           | int              | Away fouls committed                     |
| `home_yellow_cards`    | int              | Home yellow cards                        |
| `away_yellow_cards`    | int              | Away yellow cards                        |
| `home_red_cards`       | int              | Home red cards                           |
| `away_red_cards`       | int              | Away red cards                           |
| `created_at`           | timestamptz      | Row creation timestamp                   |

---

### `player_match_stats`

Individual player performance per match. One row per player per game. Used for player-level form features in future model iterations.

| Column       | Type             | Description                                 |
| ------------ | ---------------- | ------------------------------------------- |
| `id`         | serial PK        |                                             |
| `match_id`   | int FK → matches |                                             |
| `player_id`  | int FK → players |                                             |
| `team_id`    | int FK → teams   |                                             |
| `minutes`    | int              | Minutes played                              |
| `goals`      | int              | Goals scored                                |
| `assists`    | int              | Assists                                     |
| `xg`         | numeric(5,2)     | Player-level expected goals                 |
| `xa`         | numeric(5,2)     | Player-level expected assists               |
| `shots`      | int              | Total shots                                 |
| `key_passes` | int              | Key passes                                  |
| `rating`     | numeric(3,1)     | Match rating (0.0–10.0) from scraped source |
| `created_at` | timestamptz      | Row creation timestamp                      |

---

### `team_injuries`

Current injury and suspension list per team. Cleared and re-inserted before each matchday by `scrape_injuries.py`. Used by the model to compute `injury_impact` (feature [13] and [14] in the state vector).

| Column        | Type             | Description                                       |
| ------------- | ---------------- | ------------------------------------------------- |
| `id`          | serial PK        |                                                   |
| `team_id`     | int FK → teams   |                                                   |
| `player_id`   | int FK → players | Null if player not yet in the `players` table     |
| `player_name` | text             | Denormalised name for easy display without a join |
| `status`      | text             | `Injured`, `Suspended`, or `Doubt`                |
| `return_date` | date             | Estimated return date. Null if unknown            |
| `source_url`  | text             | Page the data was scraped from                    |
| `scraped_at`  | timestamptz      | When this record was last fetched                 |

---

### `odds`

Pre-match bookmaker odds. One row per bookmaker per match per fetch. Multiple rows exist per match — use `fetched_at DESC LIMIT 1` to get the most recent line for each bookmaker.

| Column       | Type             | Description                                                        |
| ------------ | ---------------- | ------------------------------------------------------------------ |
| `id`         | serial PK        |                                                                    |
| `match_id`   | int FK → matches |                                                                    |
| `bookmaker`  | text             | Bookmaker slug from The Odds API — e.g. `bet365`, `williamhill`    |
| `home_odds`  | numeric(6,2)     | Decimal odds for home win — e.g. `2.10` means £1 bet returns £2.10 |
| `draw_odds`  | numeric(6,2)     | Decimal odds for draw                                              |
| `away_odds`  | numeric(6,2)     | Decimal odds for away win                                          |
| `fetched_at` | timestamptz      | When these odds were fetched                                       |

---

### `elo_ratings`

ELO rating for each team after each match. One row per team per match — two rows inserted per completed match. Used as features [0], [1], and [2] in the state vector.

| Column          | Type             | Description                                       |
| --------------- | ---------------- | ------------------------------------------------- |
| `id`            | serial PK        |                                                   |
| `team_id`       | int FK → teams   |                                                   |
| `match_id`      | int FK → matches | The match that triggered this rating update       |
| `elo`           | numeric(7,2)     | ELO rating after this match. Starting value: 1500 |
| `calculated_at` | timestamptz      | Calculation timestamp                             |

**ELO parameters:** K=32 (update speed), home advantage=65 points, starting rating=1500. Higher rating = stronger team. Man City at peak ≈ 1750, newly promoted side ≈ 1350.

---

### `predictions`

Model output for each match. One row per match per model version. Filled before the match by the prediction engine; `was_correct` and `log_loss` are filled after results come in.

| Column               | Type             | Description                                                    |
| -------------------- | ---------------- | -------------------------------------------------------------- |
| `id`                 | serial PK        |                                                                |
| `match_id`           | int FK → matches |                                                                |
| `model_version`      | text             | Version tag — e.g. `xgb_20241201_1430`                         |
| `prob_home`          | numeric(6,4)     | XGBoost probability of home win (0–1)                          |
| `prob_draw`          | numeric(6,4)     | XGBoost probability of draw (0–1)                              |
| `prob_away`          | numeric(6,4)     | XGBoost probability of away win (0–1)                          |
| `recommended_action` | text             | DQN decision: `BET_HOME`, `BET_DRAW`, `BET_AWAY`, or `PASS`    |
| `confidence`         | numeric(5,4)     | DQN confidence score (0–1), derived from softmax of Q-values   |
| `was_correct`        | boolean          | Whether the recommended action was correct. Null until settled |
| `log_loss`           | numeric(8,6)     | Per-prediction log loss. Null until settled                    |
| `created_at`         | timestamptz      | When the prediction was made                                   |

---

### `wallet`

Singleton table — always exactly one row (`id = 1`). Tracks the virtual bankroll.

| Column           | Type          | Description                                      |
| ---------------- | ------------- | ------------------------------------------------ |
| `id`             | int PK        | Always 1                                         |
| `balance`        | numeric(10,2) | Current virtual balance in EUR. Starts at €10.00 |
| `total_staked`   | numeric(10,2) | Cumulative amount wagered across all bets        |
| `total_returned` | numeric(10,2) | Cumulative returns from winning bets             |
| `inception_date` | date          | Date the project started                         |
| `updated_at`     | timestamptz   | Last update timestamp                            |

**Derived:** ROI = `(total_returned - total_staked) / total_staked × 100`

---

### `bets`

Every virtual bet placed. The core P&L log. Settled after results come in.

| Column           | Type                 | Description                                                     |
| ---------------- | -------------------- | --------------------------------------------------------------- |
| `id`             | uuid PK              |                                                                 |
| `match_id`       | int FK → matches     |                                                                 |
| `prediction_id`  | int FK → predictions | The prediction that triggered this bet                          |
| `action`         | text                 | `BET_HOME`, `BET_DRAW`, or `BET_AWAY`                           |
| `stake`          | numeric(8,2)         | Amount wagered in EUR, sized by Kelly Criterion                 |
| `odds`           | numeric(6,2)         | Decimal odds at time of bet                                     |
| `balance_before` | numeric(10,2)        | Wallet balance before this bet was placed                       |
| `outcome`        | text                 | `WIN` or `LOSS`. Null until match settles                       |
| `pnl`            | numeric(8,2)         | Profit (positive) or loss (negative) in EUR. Null until settled |
| `balance_after`  | numeric(10,2)        | Wallet balance after settlement. Null until settled             |
| `placed_at`      | timestamptz          | When the bet was placed                                         |
| `settled_at`     | timestamptz          | When the bet was settled. Null until settled                    |

---

### `rl_episodes`

DQN replay buffer stored in Postgres. Each row is one `(state, action, reward, next_state, done)` transition. The DQN training loop samples random batches from this table.

| Column        | Type             | Description                                                     |
| ------------- | ---------------- | --------------------------------------------------------------- |
| `id`          | bigserial PK     | Monotonically increasing — used for recency-based pruning       |
| `match_id`    | int FK → matches | Match this transition came from                                 |
| `state`       | jsonb            | 24-element float array — the DQN state vector before the action |
| `next_state`  | jsonb            | 24-element float array for the next match. `Null` if done=true  |
| `action`      | int              | 0=BET_HOME, 1=BET_DRAW, 2=BET_AWAY, 3=PASS                      |
| `reward`      | numeric(8,4)     | P&L normalised by starting balance (`pnl / 10.0`)               |
| `done`        | boolean          | True at end of a season or when the bankroll hits zero          |
| `episode_num` | int              | Training epoch number — used to track learning progress         |
| `created_at`  | timestamptz      | Insertion timestamp                                             |

**Note:** The buffer is capped at 5,000 rows. Older transitions are pruned after each weekly retraining run.

---

### `model_versions`

Registry of every saved model checkpoint. Only one XGBoost and one DQN model have `is_active=true` at any time.

| Column           | Type         | Description                                                                  |
| ---------------- | ------------ | ---------------------------------------------------------------------------- |
| `id`             | serial PK    |                                                                              |
| `version_tag`    | text         | Unique identifier — e.g. `xgb_20241201_1430`, `dqn_20241201_1502`            |
| `model_type`     | text         | `xgboost` or `dqn`                                                           |
| `trained_at`     | timestamptz  | When training completed                                                      |
| `training_games` | int          | Number of matches used in training                                           |
| `val_log_loss`   | numeric(8,6) | XGBoost validation log-loss (lower is better; ~1.00 is typical for football) |
| `avg_reward`     | numeric(8,4) | DQN average episode reward over the last 10 epochs                           |
| `is_active`      | boolean      | Whether this version is currently being used for predictions                 |
| `storage_path`   | text         | Absolute path to the serialised model file on disk                           |
| `notes`          | text         | JSON blob with additional training metrics                                   |

---

## Views

Views are pre-built queries exposed directly via Supabase's auto-REST API. The Next.js frontend reads from these — no custom backend endpoint needed.

---

### `daily_pnl`

Running P&L aggregated by calendar day. Used for the main trend chart on the dashboard.

| Column         | Type    | Description                                      |
| -------------- | ------- | ------------------------------------------------ |
| `bet_date`     | date    | Calendar date (UTC)                              |
| `bets_placed`  | bigint  | Number of bets settled on this day               |
| `total_staked` | numeric | Total EUR staked across all bets on this day     |
| `gross_profit` | numeric | Sum of winnings (losses not included)            |
| `net_pnl`      | numeric | Net profit or loss — negative means a losing day |
| `wins`         | bigint  | Number of winning bets                           |
| `losses`       | bigint  | Number of losing bets                            |

---

### `team_betting_record`

Per-club betting performance since inception. Used for the club filter on the dashboard.

| Column         | Type    | Description                                      |
| -------------- | ------- | ------------------------------------------------ |
| `team_id`      | int     |                                                  |
| `team_name`    | text    | Full club name                                   |
| `tla`          | text    | Three-letter abbreviation                        |
| `total_bets`   | bigint  | Total bets placed on matches involving this club |
| `total_pnl`    | numeric | Net P&L from bets on this club's matches         |
| `wins`         | bigint  | Winning bets                                     |
| `losses`       | bigint  | Losing bets                                      |
| `win_rate_pct` | numeric | Win percentage (0–100)                           |

---

### `upcoming_matches_view`

All scheduled upcoming fixtures with the latest odds and model prediction attached. Used by the frontend home page to show what the model is currently thinking.

| Column               | Type        | Description                                                       |
| -------------------- | ----------- | ----------------------------------------------------------------- |
| `match_id`           | int         |                                                                   |
| `kickoff_time`       | timestamptz | Scheduled kickoff                                                 |
| `matchday`           | int         | Gameweek number                                                   |
| `season`             | text        | Season label                                                      |
| `home_team`          | text        | Home club full name                                               |
| `home_tla`           | text        | Home club TLA                                                     |
| `away_team`          | text        | Away club full name                                               |
| `away_tla`           | text        | Away club TLA                                                     |
| `home_odds`          | numeric     | Best available home odds                                          |
| `draw_odds`          | numeric     | Best available draw odds                                          |
| `away_odds`          | numeric     | Best available away odds                                          |
| `bookmaker`          | text        | Source bookmaker for these odds                                   |
| `prob_home`          | numeric     | Model probability of home win                                     |
| `prob_draw`          | numeric     | Model probability of draw                                         |
| `prob_away`          | numeric     | Model probability of away win                                     |
| `recommended_action` | text        | DQN recommendation: `BET_HOME`, `BET_DRAW`, `BET_AWAY`, or `PASS` |
| `confidence`         | numeric     | Model confidence in the recommendation (0–1)                      |

---

## Row level security

| Role                        | Access                              |
| --------------------------- | ----------------------------------- |
| `anon` (publishable key)    | SELECT only on all tables and views |
| `service_role` (secret key) | Full access — bypasses RLS entirely |

The frontend uses the publishable key and can only read. All writes go through the backend using the secret key.

---

## Key relationships

```
teams ──< matches (home_team_id, away_team_id)
matches ──< odds
matches ──< match_stats          (one per match)
matches ──< elo_ratings          (two per match, one per team)
matches ──< predictions          (one per match per model version)
matches ──< bets                 (one per match if DQN says bet)
predictions ──< bets
wallet ←── updated by settle_bets.py after each result
rl_episodes ←── written during DQN training loop
```
