"use client";

import { useEffect, useState } from "react";
import { format } from "date-fns";
import { getBetHistory, type Bet } from "../../lib/supabase";

// ─── Helpers ─────────────────────────────────────────────────────────────────

type Filter = "all" | "win" | "loss" | "open";

function outcomeConfig(outcome: string | null) {
  if (outcome === "WIN")
    return {
      label: "Win",
      badge: "badge-profit",
      row: "bg-profit-bg/30 hover:bg-profit-bg/50",
    };
  if (outcome === "LOSS")
    return {
      label: "Loss",
      badge: "badge-loss",
      row: "bg-loss-bg/20  hover:bg-loss-bg/40",
    };
  return { label: "Open", badge: "badge-pending", row: "hover:bg-subtle/60" };
}

function actionLabel(action: string) {
  const map: Record<string, string> = {
    BET_HOME: "Home",
    BET_DRAW: "Draw",
    BET_AWAY: "Away",
  };
  return map[action] ?? action;
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-IE", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function HistoryPage() {
  const [bets, setBets] = useState<Bet[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBetHistory(200)
      .then((data) => {
        setBets(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("History fetch error:", err);
        setError(err?.message || String(err));
        setLoading(false);
      });
  }, []);

  const filtered = bets
    .filter((b) => {
      if (filter === "win") return b.outcome === "WIN";
      if (filter === "loss") return b.outcome === "LOSS";
      if (filter === "open") return b.outcome === null;
      return true;
    })
    .filter((b) => {
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        b.matches?.home?.short_name?.toLowerCase().includes(q) ||
        b.matches?.away?.short_name?.toLowerCase().includes(q) ||
        b.matches?.home?.tla?.toLowerCase().includes(q) ||
        b.matches?.away?.tla?.toLowerCase().includes(q)
      );
    });

  // Summary stats
  const wins = bets.filter((b) => b.outcome === "WIN").length;
  const losses = bets.filter((b) => b.outcome === "LOSS").length;
  const open = bets.filter((b) => b.outcome === null).length;
  const totalPnl = bets.reduce((s, b) => s + (b.pnl ?? 0), 0);

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-display-md font-display text-primary">
          Bet History
        </h1>
        <p className="text-muted text-sm mt-1">
          Complete log of every virtual bet placed by the model
        </p>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          {
            label: "Total bets",
            value: bets.length.toString(),
            color: "text-primary",
          },
          { label: "Wins", value: `${wins}`, color: "text-profit" },
          { label: "Losses", value: `${losses}`, color: "text-loss" },
          {
            label: "Net P&L",
            value: `${totalPnl >= 0 ? "+" : ""}€${fmt(totalPnl)}`,
            color: totalPnl >= 0 ? "text-profit" : "text-loss",
          },
        ].map(({ label, value, color }) => (
          <div key={label} className="card px-5 py-4">
            <p className="text-xs font-medium text-muted mb-1">{label}</p>
            <p
              className={`text-xl font-display font-semibold tabular ${color}`}
            >
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Filters + search */}
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
        {/* Outcome filter */}
        <div className="flex items-center gap-1 bg-subtle rounded-xl p-1">
          {(["all", "win", "loss", "open"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3.5 py-1.5 rounded-lg text-sm font-medium capitalize transition-all duration-150 ${
                filter === f
                  ? "bg-surface text-primary shadow-card"
                  : "text-muted hover:text-primary"
              }`}
            >
              {f === "all"
                ? `All (${bets.length})`
                : f === "win"
                  ? `Wins (${wins})`
                  : f === "loss"
                    ? `Losses (${losses})`
                    : `Open (${open})`}
            </button>
          ))}
        </div>

        {/* Club search */}
        <div className="relative flex-1 max-w-xs">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="Search by team..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm border border-border rounded-xl bg-surface
                       text-primary placeholder:text-muted
                       focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
          />
        </div>

        <span className="text-sm text-muted ml-auto">
          {filtered.length} result{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Error state */}
      {error && (
        <div className="card px-6 py-4 border-loss bg-loss-bg">
          <p className="text-sm font-medium text-loss mb-1">
            Failed to load bet history
          </p>
          <p className="text-xs font-mono text-loss/70">{error}</p>
          <p className="text-xs text-muted mt-2">
            Check that NEXT_PUBLIC_SUPABASE_URL and
            NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY are set in Vercel.
          </p>
        </div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-subtle/40">
                {[
                  "Date",
                  "Match",
                  "Season",
                  "Bet on",
                  "Odds",
                  "Stake",
                  "Outcome",
                  "P&L",
                  "Balance after",
                ].map((h) => (
                  <th
                    key={h}
                    className="px-5 py-3 text-left text-xs font-semibold text-muted whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading &&
                Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i} className="border-b border-border/50">
                    {Array.from({ length: 9 }).map((_, j) => (
                      <td key={j} className="px-5 py-4">
                        <div
                          className="skeleton h-3 rounded"
                          style={{ width: `${60 + Math.random() * 40}%` }}
                        />
                      </td>
                    ))}
                  </tr>
                ))}

              {!loading && filtered.length === 0 && (
                <tr>
                  <td
                    colSpan={9}
                    className="px-5 py-12 text-center text-muted text-sm"
                  >
                    {bets.length === 0
                      ? "No bets placed yet. The model will start betting when the next matchday begins."
                      : "No bets match your filter."}
                  </td>
                </tr>
              )}

              {!loading &&
                filtered.map((bet) => {
                  const m = bet.matches;
                  const config = outcomeConfig(bet.outcome);
                  const settled = bet.settled_at
                    ? format(new Date(bet.settled_at), "dd MMM yy")
                    : "—";
                  const kickoff = m?.kickoff_time
                    ? format(new Date(m.kickoff_time), "dd MMM yy")
                    : "—";

                  return (
                    <tr
                      key={bet.id}
                      className={`border-b border-border/40 last:border-0 transition-colors duration-100 ${config.row}`}
                    >
                      {/* Date */}
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className="text-xs tabular text-muted">
                          {kickoff}
                        </span>
                      </td>

                      {/* Match */}
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <div className="flex items-center gap-1.5 font-medium text-primary">
                          <span>{m?.home?.short_name ?? "?"}</span>
                          <span className="text-muted text-xs">vs</span>
                          <span>{m?.away?.short_name ?? "?"}</span>
                        </div>
                      </td>

                      {/* Season */}
                      <td className="px-5 py-3.5">
                        <span className="text-xs text-muted">
                          {m?.season ?? "—"}
                        </span>
                      </td>

                      {/* Bet on */}
                      <td className="px-5 py-3.5">
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-accent-light text-accent border border-accent/20">
                          {actionLabel(bet.action)}
                        </span>
                      </td>

                      {/* Odds */}
                      <td className="px-5 py-3.5">
                        <span className="tabular font-mono text-xs text-primary">
                          {bet.odds.toFixed(2)}
                        </span>
                      </td>

                      {/* Stake */}
                      <td className="px-5 py-3.5">
                        <span className="tabular font-mono text-xs text-primary">
                          €{fmt(bet.stake)}
                        </span>
                      </td>

                      {/* Outcome badge */}
                      <td className="px-5 py-3.5">
                        <span className={config.badge}>{config.label}</span>
                      </td>

                      {/* P&L */}
                      <td className="px-5 py-3.5">
                        {bet.pnl != null ? (
                          <span
                            className={`tabular font-mono text-xs font-bold ${
                              bet.pnl >= 0 ? "text-profit" : "text-loss"
                            }`}
                          >
                            {bet.pnl >= 0 ? "+" : ""}€{fmt(Math.abs(bet.pnl))}
                          </span>
                        ) : (
                          <span className="text-muted text-xs">—</span>
                        )}
                      </td>

                      {/* Balance after */}
                      <td className="px-5 py-3.5">
                        <span className="tabular font-mono text-xs text-muted">
                          {bet.balance_after != null
                            ? `€${fmt(bet.balance_after)}`
                            : "—"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-muted">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-profit-bg border border-profit-border" />
          Correct prediction (win)
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-loss-bg border border-loss-border" />
          Incorrect prediction (loss)
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-pending-bg border border-border" />
          Open / awaiting result
        </div>
      </div>
    </div>
  );
}
