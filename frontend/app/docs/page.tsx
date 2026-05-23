import Link from "next/link";

// ─── Static data ──────────────────────────────────────────────────────────────

const GITHUB_URL = "https://github.com/Shishir-Ashok/Bet-EPL";

const pipeline = [
  {
    step: "01",
    title: "Data Collection",
    color: "text-accent",
    border: "border-accent/20",
    bg: "bg-accent-light",
    items: [
      { name: "football-data.org", detail: "Fixtures, results, match status" },
      {
        name: "football-data.co.uk",
        detail: "Pre-match opening odds (Pinnacle, Bet365)",
      },
      {
        name: "Understat / FDCO CSVs",
        detail: "Shots on target ➜ xG proxy (0.30 per SOT)",
      },
      { name: "ELO ratings", detail: "Computed in-house, K=32, HFA=65pts" },
    ],
  },
  {
    step: "02",
    title: "Feature Engineering",
    color: "text-profit",
    border: "border-profit/20",
    bg: "bg-profit-bg",
    items: [
      {
        name: "home_elo / away_elo",
        detail: "Normalised ELO (1200-1900 ➜ 0-1)",
      },
      { name: "elo_diff", detail: "Signed difference, clipped ±1" },
      { name: "form_5 / form_10", detail: "PPG over last 5 and 10 matches" },
      { name: "xG scored / conceded", detail: "Avg over last 5 matches" },
      {
        name: "h2h_home_winrate",
        detail: "Last 6 head-to-heads at home venue",
      },
      { name: "injury_impact", detail: "Weighted squad availability score" },
    ],
  },
  {
    step: "03",
    title: "XGBoost Predictor",
    color: "text-draw",
    border: "border-draw/20",
    bg: "bg-draw-bg",
    items: [
      {
        name: "Objective",
        detail: "multi:softprob — 3 classes (HOME / DRAW / AWAY)",
      },
      { name: "Architecture", detail: "300 estimators, max_depth=4, lr=0.05" },
      {
        name: "Calibration",
        detail: "Per-class isotonic regression on validation set",
      },
      {
        name: "Train split",
        detail: "Chronological 85/15 — validates on most recent season",
      },
      {
        name: "Leakage guard",
        detail: "before_date cap on features AND labels",
      },
    ],
  },
  {
    step: "04",
    title: "DQN Betting Agent",
    color: "text-accent",
    border: "border-accent/20",
    bg: "bg-accent-light",
    items: [
      {
        name: "State",
        detail: "24-dim: 16 XGB features + 3 probs + 3 implied + edge + wallet",
      },
      { name: "Actions", detail: "BET_HOME · BET_DRAW · BET_AWAY · PASS" },
      { name: "Reward", detail: "Normalised P&L per unit staked, clipped ±2" },
      {
        name: "Architecture",
        detail: "MLP with experience replay + target network",
      },
      {
        name: "Gate",
        detail: "Q(bet) - Q(PASS) ≥ threshold to confirm Kelly suggestion",
      },
    ],
  },
  {
    step: "05",
    title: "Kelly Sizing",
    color: "text-profit",
    border: "border-profit/20",
    bg: "bg-profit-bg",
    items: [
      {
        name: "De-vig",
        detail: "Implied probs normalised to remove bookmaker overround",
      },
      { name: "Edge", detail: "model_prob - fair_prob > 2% minimum threshold" },
      {
        name: "Fraction",
        detail: "33% Kelly to reduce variance from model uncertainty",
      },
      {
        name: "Cap",
        detail: "Max 20% of bankroll per bet regardless of Kelly output",
      },
    ],
  },
  {
    step: "06",
    title: "Retraining Schedule",
    color: "text-draw",
    border: "border-draw/20",
    bg: "bg-draw-bg",
    items: [
      {
        name: "2020-24",
        detail: "Base XGBoost training data (5 completed seasons)",
      },
      {
        name: "Mid-season",
        detail: "After matchday 19 (≈ Jan 1) — XGBoost + first DQN",
      },
      {
        name: "End-of-season",
        detail: "After MD 38 — XGBoost includes full season, DQN full retrain",
      },
      {
        name: "2025-26+",
        detail: "Same mid/end pattern — XGBoost on completed seasons only",
      },
    ],
  },
];

const infra = [
  {
    service: "Supabase",
    role: "PostgreSQL database + storage for model checkpoints",
    free: "500 MB",
  },
  {
    service: "Vercel",
    role: "Next.js frontend hosting",
    free: "Unlimited deploys",
  },
  {
    service: "Render",
    role: "FastAPI server triggered by GitHub Actions",
    free: "750h/month",
  },
  {
    service: "GitHub Actions",
    role: "Cron jobs — fetch odds, ingest results, retrain",
    free: "2,000 min/month",
  },
  {
    service: "football-data.org",
    role: "Fixtures and results API",
    free: "10 req/min",
  },
  {
    service: "football-data.co.uk",
    role: "Historical odds CSVs (Pinnacle opening lines)",
    free: "Unlimited",
  },
  {
    service: "the-odds-api.com",
    role: "Odds of upcoming fixtures",
    free: "500 API calls/month",
  },
];

const schema = [
  {
    table: "teams",
    description: "20 Premier League clubs with crest URLs and TLA codes",
  },
  {
    table: "matches",
    description: "Every fixture — status, result, kickoff time, season",
  },
  {
    table: "odds",
    description: "Pre-match opening odds (Pinnacle preferred, Bet365 fallback)",
  },
  {
    table: "match_stats",
    description: "Shots on target, xG proxy, corners, cards per match",
  },
  {
    table: "elo_ratings",
    description: "Per-team ELO after each match with kickoff timestamp",
  },
  {
    table: "predictions",
    description: "XGBoost HOME/DRAW/AWAY probabilities + DQN action",
  },
  {
    table: "bets",
    description: "Virtual bets placed — stake, odds, outcome, P&L, balance",
  },
  {
    table: "wallet",
    description: "Running balance, total staked, total returned",
  },
  {
    table: "rl_episodes",
    description: "DQN replay buffer — state, action, reward transitions",
  },
  {
    table: "model_versions",
    description: "Registry of trained checkpoints with val log-loss",
  },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function DocsPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <div className="flex items-center gap-3">
          <span className="section-label">Technical documentation</span>
        </div>

        <div className="space-y-3">
          <h1 className="text-display-xl font-display text-primary leading-tight">
            How PL<span className="text-accent">Bot</span> works
          </h1>
          <p className="text-base text-muted max-w-2xl leading-relaxed">
            A self-running ML system that watches every Premier League match,
            predicts outcomes using XGBoost, decides whether to bet using a Deep
            Q-Network, sizes stakes with Kelly Criterion, and learns from every
            result. Built entirely on free-tier infrastructure.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary text-sm gap-2"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            View on GitHub
          </a>
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-border text-xs text-muted font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-profit inline-block" />
            Virtual bets only — no real money
          </div>
        </div>
      </section>

      {/* ─── Architecture diagram ─────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">
          Architecture
        </h2>

        <div className="card p-6 overflow-x-auto">
          <div className="flex items-start gap-0 min-w-max">
            {[
              {
                label: "Data Sources",
                items: [
                  "football-data.org",
                  "football-data.co.uk",
                  "FDCO CSVs",
                  "ELO engine",
                ],
                color: "border-accent/40 bg-accent-light",
              },
              {
                label: "Supabase DB",
                items: ["matches", "odds", "match_stats", "elo_ratings"],
                color: "border-border bg-subtle",
              },
              {
                label: "XGBoost",
                items: [
                  "16 features",
                  "3-class softprob",
                  "Isotonic cal.",
                  "before_date guard",
                ],
                color: "border-profit/40 bg-profit-bg",
              },
              {
                label: "DQN Agent",
                items: ["24-dim state", "4 actions", "Replay buffer", "Q-gate"],
                color: "border-draw/40 bg-draw-bg",
              },
              {
                label: "Kelly + DB",
                items: [
                  "De-vig odds",
                  "33% fraction",
                  "Bet logged",
                  "Wallet updated",
                ],
                color: "border-accent/40 bg-accent-light",
              },
            ].map((block, idx, arr) => (
              <div key={block.label} className="flex items-center">
                <div
                  className={`rounded-xl border px-4 py-3 w-36 ${block.color}`}
                >
                  <p className="text-xs font-semibold text-primary mb-2">
                    {block.label}
                  </p>
                  <ul className="space-y-0.5">
                    {block.items.map((item) => (
                      <li
                        key={item}
                        className="text-[10px] text-muted font-mono"
                      >
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                {idx < arr.length - 1 && (
                  <div className="flex items-center px-1">
                    <div className="w-6 h-px bg-border" />
                    <svg
                      className="w-3 h-3 text-muted flex-shrink-0"
                      viewBox="0 0 12 12"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="4,2 9,6 4,10" />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Pipeline steps ───────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">Pipeline</h2>

        <div className="grid gap-4 sm:grid-cols-2">
          {pipeline.map((block) => (
            <div key={block.step} className="card p-5 space-y-3">
              <div className="flex items-center gap-3">
                <span className={`font-mono text-xs font-bold ${block.color}`}>
                  {block.step}
                </span>
                <h3 className="text-sm font-semibold text-primary">
                  {block.title}
                </h3>
              </div>
              <ul className="space-y-2">
                {block.items.map((item) => (
                  <li key={item.name} className="flex gap-2">
                    <span
                      className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${block.bg} ${block.color} border ${block.border} flex-shrink-0 mt-px`}
                    >
                      {item.name}
                    </span>
                    <span className="text-xs text-muted leading-relaxed">
                      {item.detail}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* ─── Database schema ──────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">
          Database schema
        </h2>
        <p className="text-sm text-muted">
          Supabase PostgreSQL. All tables are read-only from the frontend via
          the anon key.
        </p>

        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {["Table", "Description"].map((h) => (
                  <th
                    key={h}
                    className="px-5 py-3 text-left text-xs font-semibold text-muted"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {schema.map((row, i) => (
                <tr
                  key={row.table}
                  className={`border-b border-border/50 table-row-hover last:border-0`}
                >
                  <td className="px-5 py-3">
                    <code className="text-xs font-mono font-semibold text-accent bg-accent-light px-2 py-0.5 rounded">
                      {row.table}
                    </code>
                  </td>
                  <td className="px-5 py-3 text-xs text-muted">
                    {row.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Key relationships */}
        <div className="card p-5 space-y-3">
          <p className="text-xs font-semibold text-muted uppercase tracking-widest">
            Key relationships
          </p>
          <div className="font-mono text-xs text-muted space-y-1.5 leading-relaxed">
            {[
              "teams ──< matches (home_team_id, away_team_id)",
              "matches ──< odds",
              "matches ──< match_stats       (one per match)",
              "matches ──< elo_ratings        (two per match, one per team)",
              "matches ──< predictions        (one per match per model version)",
              "matches ──< bets               (one per match if DQN confirms bet)",
              "predictions ──< bets",
              "wallet <── updated by settle_bets.py after each result",
              "rl_episodes <── written during DQN training loop",
            ].map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Infrastructure ───────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">
          Infrastructure
        </h2>
        <p className="text-sm text-muted">
          Entirely free-tier. No credit card. No paid APIs.
        </p>

        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {["Service", "Role", "Free limit"].map((h) => (
                  <th
                    key={h}
                    className="px-5 py-3 text-left text-xs font-semibold text-muted"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {infra.map((row) => (
                <tr
                  key={row.service}
                  className="border-b border-border/50 table-row-hover last:border-0"
                >
                  <td className="px-5 py-3">
                    <span className="text-xs font-semibold text-primary">
                      {row.service}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-xs text-muted">{row.role}</td>
                  <td className="px-5 py-3">
                    <span className="text-xs font-mono text-profit">
                      {row.free}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ─── Automation ───────────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">
          Automation
        </h2>
        <p className="text-sm text-muted">
          Three GitHub Actions workflows run the system end-to-end with zero
          manual intervention.
        </p>

        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              file: "matchday_orchestrator.yml",
              trigger: "3h before each PL matchday",
              does: [
                "Fetches latest odds from the-odds-api.com",
                "Runs XGBoost ➜ DQN ➜ Kelly pipeline",
                "Writes predictions and bets to DB",
                "Updates wallet balance",
              ],
            },
            {
              file: "ingest_results.yml",
              trigger: "Morning after matches finish",
              does: [
                "Fetches final results from football-data.org",
                "Settles all open bets (WIN / LOSS)",
                "Calculates P&L and updates wallet",
                "Pushes RL transitions to replay buffer",
              ],
            },
            {
              file: "monthly_retrain.yml",
              trigger: "Mid-season + end-of-season",
              does: [
                "Retrains XGBoost on completed seasons",
                "Incremental DQN update on new bets",
                "Registers new model version in DB",
                "Old version marked inactive",
              ],
            },
          ].map((wf) => (
            <div key={wf.file} className="card p-5 space-y-3">
              <div className="space-y-1">
                <code className="text-[10px] font-mono text-muted">
                  {wf.file}
                </code>
                <p className="text-xs font-semibold text-accent">
                  {wf.trigger}
                </p>
              </div>
              <ul className="space-y-1.5">
                {wf.does.map((d) => (
                  <li
                    key={d}
                    className="flex items-start gap-2 text-xs text-muted"
                  >
                    <span className="text-profit mt-0.5 flex-shrink-0">•</span>
                    {d}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* ─── Design decisions ─────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-display-sm font-display text-primary">
          Key design decisions
        </h2>

        <div className="space-y-3">
          {[
            {
              q: "Why XGBoost first, then DQN?",
              a: "XGBoost gives well-calibrated probabilities on small tabular datasets which is something neural nets struggle with at this data scale. The DQN then learns the meta-decision: when are those probabilities trustworthy enough to bet on, given bankroll state and market pricing.",
            },
            {
              q: "Why Pinnacle opening odds (not closing)?",
              a: "Closing odds reflect everything the market knew right before kickoff, including late team news and sharp money. Opening Pinnacle lines, collected Friday afternoons for weekend games, represent the realistic prices available when a pre-match decision is made.",
            },
            {
              q: "Why 33% fractional Kelly?",
              a: "Full Kelly maximises long-run growth but requires a perfectly calibrated model. Using 33% reduces variance substantially while still growing the bankroll when edge is genuine. The 20% hard cap prevents any single bet from being catastrophic.",
            },
            {
              q: "Why not include the current partial season in XGBoost training?",
              a: "Partial seasons have survivorship patterns  (teams still fighting relegation, title race compressing odds) that don't represent the full distribution. XGBoost trains on completed seasons only. The DQN handles current-season adaptation continuously via the replay buffer.",
            },
            {
              q: "How is lookahead bias prevented?",
              a: "The before_date parameter flows from run_simulation ➜ xgboost_model.train ➜ load_training_data ➜ build_feature_matrix ➜ bulk_fetch. Both training labels AND the in-memory feature cache are capped to data before the relevant date. Per-match temporal filters (kickoff_time < before_time) handle inference.",
            },
          ].map((item) => (
            <details
              key={item.q}
              className="card px-5 py-4 group cursor-pointer"
            >
              <summary className="flex items-center justify-between text-sm font-medium text-primary list-none">
                {item.q}
                <svg
                  className="w-4 h-4 text-muted flex-shrink-0 transition-transform duration-200 group-open:rotate-180"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </summary>
              <p className="mt-3 text-sm text-muted leading-relaxed border-t border-border pt-3">
                {item.a}
              </p>
            </details>
          ))}
        </div>
      </section>

      {/* ─── Footer ───────────────────────────────────────────────────────── */}
      <footer className="pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted">
        <span>
          PL Betting Bot — all bets are virtual. No real money is used.
        </span>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-primary transition-colors duration-150"
        >
          GitHub ➜
        </a>
      </footer>
    </div>
  );
}
