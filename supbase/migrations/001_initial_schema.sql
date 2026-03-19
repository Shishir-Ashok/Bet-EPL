-- =============================================================================
-- PL Betting Bot — Initial Schema
-- Run this in your Supabase SQL editor (Dashboard → SQL Editor → New query)
-- =============================================================================

-- Enable UUID generation (already on in Supabase, but kept for clarity)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- =============================================================================
-- TEAMS
-- One row per Premier League club. Seeded once, rarely updated.
-- =============================================================================
CREATE TABLE teams (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- e.g. "Arsenal FC"
    short_name  TEXT NOT NULL,                 -- e.g. "Arsenal"
    tla         TEXT NOT NULL UNIQUE,          -- e.g. "ARS" (football-data.org code)
    crest_url   TEXT,                          -- badge image URL
    created_at  TIMESTAMPTZ DEFAULT NOW()
);


-- =============================================================================
-- PLAYERS
-- Basic player reference table. Stats live in player_match_stats.
-- =============================================================================
CREATE TABLE players (
    id            SERIAL PRIMARY KEY,
    team_id       INT REFERENCES teams(id) ON DELETE SET NULL,
    name          TEXT NOT NULL,
    position      TEXT,                        -- GK, DEF, MID, FWD
    nationality   TEXT,
    date_of_birth DATE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_players_team ON players(team_id);


-- =============================================================================
-- MATCHES
-- One row per PL fixture. Populated by the data-fetch cron job.
-- result is NULL until the match is played.
-- =============================================================================
CREATE TABLE matches (
    id              SERIAL PRIMARY KEY,
    external_id     BIGINT UNIQUE,                -- football-data.org match ID
    home_team_id    INT NOT NULL REFERENCES teams(id),
    away_team_id    INT NOT NULL REFERENCES teams(id),
    kickoff_time    TIMESTAMPTZ NOT NULL,
    matchday        INT,                       -- GW1–GW38
    season          TEXT NOT NULL,             -- e.g. "2024-25"
    status          TEXT DEFAULT 'SCHEDULED',  -- SCHEDULED | IN_PLAY | FINISHED | POSTPONED

    -- Filled after the match finishes
    home_goals      INT,
    away_goals      INT,
    result          TEXT,                      -- "HOME" | "DRAW" | "AWAY"

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_matches_kickoff  ON matches(kickoff_time);
CREATE INDEX idx_matches_season   ON matches(season);
CREATE INDEX idx_matches_home     ON matches(home_team_id);
CREATE INDEX idx_matches_away     ON matches(away_team_id);


-- =============================================================================
-- MATCH_STATS
-- Aggregated per-match team stats. Populated after a match finishes.
-- Both home and away are stored in the same row to keep joins simple.
-- =============================================================================
CREATE TABLE match_stats (
    id                  SERIAL PRIMARY KEY,
    match_id            INT NOT NULL UNIQUE REFERENCES matches(id) ON DELETE CASCADE,

    -- Expected goals (from Understat or FBref scraping)
    home_xg             NUMERIC(5, 2),
    away_xg             NUMERIC(5, 2),

    -- Shots
    home_shots          INT,
    away_shots          INT,
    home_shots_on_target INT,
    away_shots_on_target INT,

    -- Possession & passing
    home_possession     NUMERIC(4, 1),        -- percentage, e.g. 54.3
    away_possession     NUMERIC(4, 1),
    home_passes         INT,
    away_passes         INT,
    home_pass_accuracy  NUMERIC(4, 1),
    away_pass_accuracy  NUMERIC(4, 1),

    -- Defensive actions
    home_tackles        INT,
    away_tackles        INT,
    home_interceptions  INT,
    away_interceptions  INT,
    home_pressures      INT,                  -- FBref: pressing actions
    away_pressures      INT,

    -- Set pieces & cards
    home_corners        INT,
    away_corners        INT,
    home_fouls          INT,
    away_fouls          INT,
    home_yellow_cards   INT,
    away_yellow_cards   INT,
    home_red_cards      INT,
    away_red_cards      INT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- =============================================================================
-- PLAYER_MATCH_STATS
-- Individual player performance per match. Used as ML features for form.
-- =============================================================================
CREATE TABLE player_match_stats (
    id          SERIAL PRIMARY KEY,
    match_id    INT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id   INT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    team_id     INT NOT NULL REFERENCES teams(id),
    minutes     INT DEFAULT 0,
    goals       INT DEFAULT 0,
    assists     INT DEFAULT 0,
    xg          NUMERIC(5, 2),               -- player-level xG
    xa          NUMERIC(5, 2),               -- xAssist
    shots       INT DEFAULT 0,
    key_passes  INT DEFAULT 0,
    rating      NUMERIC(3, 1),               -- 0.0–10.0 from scraped source
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(match_id, player_id)
);

CREATE INDEX idx_pms_match  ON player_match_stats(match_id);
CREATE INDEX idx_pms_player ON player_match_stats(player_id);


-- =============================================================================
-- TEAM_INJURIES
-- Current injury/suspension list per team.
-- Refreshed by the news-scraping cron before each matchday.
-- =============================================================================
CREATE TABLE team_injuries (
    id          SERIAL PRIMARY KEY,
    team_id     INT NOT NULL REFERENCES teams(id),
    player_id   INT REFERENCES players(id) ON DELETE SET NULL,
    player_name TEXT NOT NULL,               -- denormalised for easy display
    status      TEXT NOT NULL,               -- "Injured" | "Suspended" | "Doubt"
    return_date DATE,                        -- estimated return, can be NULL
    source_url  TEXT,
    scraped_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_injuries_team ON team_injuries(team_id);


-- =============================================================================
-- ODDS
-- Pre-match bookmaker odds. Fetched from The Odds API.
-- Multiple rows per match (one per snapshot — we store the last fetch only,
-- but keeping history lets us see line movement).
-- =============================================================================
CREATE TABLE odds (
    id              SERIAL PRIMARY KEY,
    match_id        INT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker       TEXT NOT NULL,           -- e.g. "bet365"
    home_odds       NUMERIC(6, 2) NOT NULL,  -- decimal odds, e.g. 2.10
    draw_odds       NUMERIC(6, 2) NOT NULL,
    away_odds       NUMERIC(6, 2) NOT NULL,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_odds_match   ON odds(match_id);
CREATE INDEX idx_odds_fetched ON odds(fetched_at DESC);


-- =============================================================================
-- ELO_RATINGS
-- Calculated ELO per team after each match.
-- Maintained by the results-ingestion job, not raw scraped data.
-- =============================================================================
CREATE TABLE elo_ratings (
    id          SERIAL PRIMARY KEY,
    team_id     INT NOT NULL REFERENCES teams(id),
    match_id    INT NOT NULL REFERENCES matches(id),   -- the match that caused this update
    elo         NUMERIC(7, 2) NOT NULL,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(team_id, match_id)
);

CREATE INDEX idx_elo_team ON elo_ratings(team_id);


-- =============================================================================
-- PREDICTIONS
-- The ML model's output for each match.
-- One row per match per model version (so we can compare versions).
-- =============================================================================
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    match_id        INT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    model_version   TEXT NOT NULL,           -- e.g. "xgb_v1", "dqn_v3"

    -- XGBoost layer outputs (raw probabilities, always sum to ~1.0)
    prob_home       NUMERIC(6, 4) NOT NULL,
    prob_draw       NUMERIC(6, 4) NOT NULL,
    prob_away       NUMERIC(6, 4) NOT NULL,

    -- DQN agent decision
    recommended_action  TEXT,               -- "BET_HOME" | "BET_DRAW" | "BET_AWAY" | "PASS"
    confidence          NUMERIC(5, 4),      -- Q-value magnitude, normalised 0–1

    -- Post-match evaluation (filled in by results job)
    was_correct     BOOLEAN,                -- did the recommended action win?
    log_loss        NUMERIC(8, 6),          -- per-prediction log loss

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(match_id, model_version)
);

CREATE INDEX idx_predictions_match   ON predictions(match_id);
CREATE INDEX idx_predictions_version ON predictions(model_version);


-- =============================================================================
-- WALLET
-- Single-row table for the virtual bankroll.
-- Updated atomically by the bet-settlement job.
-- =============================================================================
CREATE TABLE wallet (
    id              INT PRIMARY KEY DEFAULT 1,       -- always row 1
    balance         NUMERIC(10, 2) NOT NULL DEFAULT 10.00,  -- starting €10
    total_staked    NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    total_returned  NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    inception_date  DATE NOT NULL DEFAULT CURRENT_DATE,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    CHECK (id = 1)   -- enforce singleton
);

-- Seed the wallet immediately
INSERT INTO wallet (id, balance, inception_date)
VALUES (1, 10.00, CURRENT_DATE)
ON CONFLICT (id) DO NOTHING;


-- =============================================================================
-- BETS
-- Every virtual bet the agent places. The core P&L log.
-- =============================================================================
CREATE TABLE bets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id        INT NOT NULL REFERENCES matches(id),
    prediction_id   INT REFERENCES predictions(id),

    -- What the agent decided
    action          TEXT NOT NULL,           -- "BET_HOME" | "BET_DRAW" | "BET_AWAY"
    stake           NUMERIC(8, 2) NOT NULL,  -- amount wagered in EUR
    odds            NUMERIC(6, 2) NOT NULL,  -- decimal odds at time of bet

    -- Wallet state at time of bet
    balance_before  NUMERIC(10, 2) NOT NULL,

    -- Settled after match (NULL until match finishes)
    outcome         TEXT,                    -- "WIN" | "LOSS"
    pnl             NUMERIC(8, 2),           -- profit (positive) or loss (negative)
    balance_after   NUMERIC(10, 2),

    placed_at       TIMESTAMPTZ DEFAULT NOW(),
    settled_at      TIMESTAMPTZ
);

CREATE INDEX idx_bets_match     ON bets(match_id);
CREATE INDEX idx_bets_placed_at ON bets(placed_at DESC);
CREATE INDEX idx_bets_outcome   ON bets(outcome);


-- =============================================================================
-- RL_EPISODES
-- Training data for the DQN. Each row is one (state, action, reward, next_state)
-- transition stored in the replay buffer. The Python side samples from this.
-- Keeping it in Postgres (rather than just in memory) means training survives
-- restarts and we can analyse what the agent learned.
-- =============================================================================
CREATE TABLE rl_episodes (
    id              BIGSERIAL PRIMARY KEY,
    match_id        INT REFERENCES matches(id),

    -- State vector stored as a JSON array of floats (easier than 30+ columns)
    -- Shape: [home_elo, away_elo, home_form_5, away_form_5,
    --         home_xg_avg, away_xg_avg, home_goals_avg, away_goals_avg,
    --         home_shots_avg, away_shots_avg, injury_impact_home,
    --         injury_impact_away, prob_home, prob_draw, prob_away,
    --         wallet_fraction]   (16 features — see model/features.py)
    state           JSONB NOT NULL,
    next_state      JSONB,

    action          INT NOT NULL,            -- 0=BET_HOME, 1=BET_DRAW, 2=BET_AWAY, 3=PASS
    reward          NUMERIC(8, 4) NOT NULL,  -- P&L normalised by initial balance
    done            BOOLEAN DEFAULT FALSE,   -- True at end of a season
    episode_num     INT,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rl_match   ON rl_episodes(match_id);
CREATE INDEX idx_rl_episode ON rl_episodes(episode_num);


-- =============================================================================
-- MODEL_VERSIONS
-- Metadata for every saved model checkpoint.
-- The actual weights file is stored in Supabase Storage or the repo.
-- =============================================================================
CREATE TABLE model_versions (
    id              SERIAL PRIMARY KEY,
    version_tag     TEXT NOT NULL UNIQUE,    -- e.g. "xgb_v1", "dqn_v3"
    model_type      TEXT NOT NULL,           -- "xgboost" | "dqn"
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    training_games  INT,                     -- number of matches used
    val_log_loss    NUMERIC(8, 6),           -- XGBoost validation metric
    avg_reward      NUMERIC(8, 4),           -- DQN average episode reward
    is_active       BOOLEAN DEFAULT FALSE,   -- only one DQN active at a time
    storage_path    TEXT,                    -- path in Supabase Storage
    notes           TEXT
);


-- =============================================================================
-- VIEWS
-- Pre-built queries the frontend will use directly via Supabase's auto REST API
-- =============================================================================

-- Running P&L by day — used for the main chart
CREATE VIEW daily_pnl AS
SELECT
    DATE(placed_at)                         AS bet_date,
    COUNT(*)                                AS bets_placed,
    SUM(stake)                              AS total_staked,
    SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0 END)  AS gross_profit,
    SUM(pnl)                                AS net_pnl,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses
FROM bets
WHERE outcome IS NOT NULL
GROUP BY DATE(placed_at)
ORDER BY bet_date;


-- Per-team betting record — used for the club filter on the dashboard
CREATE VIEW team_betting_record AS
SELECT
    t.id AS team_id,
    t.name AS team_name,
    t.tla,
    COUNT(b.id)                              AS total_bets,
    SUM(b.pnl)                               AS total_pnl,
    COUNT(*) FILTER (WHERE b.outcome = 'WIN') AS wins,
    COUNT(*) FILTER (WHERE b.outcome = 'LOSS') AS losses,
    ROUND(
        COUNT(*) FILTER (WHERE b.outcome = 'WIN')::NUMERIC
        / NULLIF(COUNT(b.id), 0) * 100, 1
    )                                        AS win_rate_pct
FROM teams t
LEFT JOIN matches m  ON (m.home_team_id = t.id OR m.away_team_id = t.id)
LEFT JOIN bets b     ON b.match_id = m.id AND b.outcome IS NOT NULL
GROUP BY t.id, t.name, t.tla
ORDER BY total_pnl DESC NULLS LAST;


-- Latest snapshot of every upcoming match with odds + prediction
CREATE VIEW upcoming_matches_view AS
SELECT
    m.id AS match_id,
    m.kickoff_time,
    m.matchday,
    m.season,
    ht.name  AS home_team,
    ht.tla   AS home_tla,
    at.name  AS away_team,
    at.tla   AS away_tla,
    -- Latest odds for this match
    o.home_odds,
    o.draw_odds,
    o.away_odds,
    o.bookmaker,
    -- Latest prediction
    p.prob_home,
    p.prob_draw,
    p.prob_away,
    p.recommended_action,
    p.confidence
FROM matches m
JOIN teams ht  ON ht.id = m.home_team_id
JOIN teams at  ON at.id = m.away_team_id
LEFT JOIN LATERAL (
    SELECT * FROM odds
    WHERE match_id = m.id
    ORDER BY fetched_at DESC LIMIT 1
) o ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM predictions
    WHERE match_id = m.id
    ORDER BY created_at DESC LIMIT 1
) p ON TRUE
WHERE m.status = 'SCHEDULED'
ORDER BY m.kickoff_time;


-- =============================================================================
-- ROW LEVEL SECURITY
-- The frontend reads data anonymously (public read).
-- Only the backend service role can write.
-- =============================================================================
ALTER TABLE teams              ENABLE ROW LEVEL SECURITY;
ALTER TABLE players            ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches            ENABLE ROW LEVEL SECURITY;
ALTER TABLE match_stats        ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_match_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_injuries      ENABLE ROW LEVEL SECURITY;
ALTER TABLE odds               ENABLE ROW LEVEL SECURITY;
ALTER TABLE elo_ratings        ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet             ENABLE ROW LEVEL SECURITY;
ALTER TABLE bets               ENABLE ROW LEVEL SECURITY;
ALTER TABLE rl_episodes        ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions     ENABLE ROW LEVEL SECURITY;

-- Public can read everything (anonymous frontend access via Supabase anon key)
CREATE POLICY "public_read_teams"       ON teams       FOR SELECT USING (true);
CREATE POLICY "public_read_players"     ON players     FOR SELECT USING (true);
CREATE POLICY "public_read_matches"     ON matches     FOR SELECT USING (true);
CREATE POLICY "public_read_stats"       ON match_stats FOR SELECT USING (true);
CREATE POLICY "public_read_pms"         ON player_match_stats FOR SELECT USING (true);
CREATE POLICY "public_read_injuries"    ON team_injuries FOR SELECT USING (true);
CREATE POLICY "public_read_odds"        ON odds        FOR SELECT USING (true);
CREATE POLICY "public_read_elo"         ON elo_ratings FOR SELECT USING (true);
CREATE POLICY "public_read_predictions" ON predictions FOR SELECT USING (true);
CREATE POLICY "public_read_wallet"      ON wallet      FOR SELECT USING (true);
CREATE POLICY "public_read_bets"        ON bets        FOR SELECT USING (true);
CREATE POLICY "public_read_rl"          ON rl_episodes FOR SELECT USING (true);
CREATE POLICY "public_read_models"      ON model_versions FOR SELECT USING (true);

-- Only service_role (backend) can write — no policy = deny for anon/authenticated
-- The backend uses SUPABASE_SERVICE_KEY which bypasses RLS entirely
