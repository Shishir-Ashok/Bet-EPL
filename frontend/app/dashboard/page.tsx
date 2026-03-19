"use client";

import { useEffect, useState, useCallback } from "react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { format, subDays, subMonths, subYears, parseISO } from "date-fns";
import {
  getDailyPnl,
  getTeamRecords,
  getWallet,
  type DailyPnl,
  type TeamRecord,
} from "@/lib/supabase";

// ─── Types ────────────────────────────────────────────────────────────────────

type Period = "today" | "7d" | "30d" | "90d" | "1y" | "all";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-IE", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

function periodToDays(p: Period): number {
  const map: Record<Period, number> = {
    today: 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "1y": 365,
    all: 9999,
  };
  return map[p];
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────

function PnlTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value ?? 0;
  return (
    <div className="bg-surface border border-border rounded-xl px-4 py-3 shadow-card-md text-sm">
      <p className="text-muted text-xs mb-1">{label}</p>
      <p
        className={`font-semibold tabular font-mono ${val >= 0 ? "text-profit" : "text-loss"}`}
      >
        {val >= 0 ? "+" : ""}€{fmt(val)}
      </p>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [period, setPeriod] = useState<Period>("30d");
  const [clubFilter, setClubFilter] = useState<string>("all");
  const [daily, setDaily] = useState<DailyPnl[]>([]);
  const [teams, setTeams] = useState<TeamRecord[]>([]);
  const [wallet, setWallet] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const days = periodToDays(period);
      const [d, t, w] = await Promise.all([
        getDailyPnl(days),
        getTeamRecords(),
        getWallet(),
      ]);
      setDaily(d);
      setTeams(t);
      setWallet(w);
    } catch (err: any) {
      console.error("Dashboard fetch error:", err);
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Filter daily data by club if selected
  const filteredDaily = clubFilter === "all" ? daily : daily; // TODO: per-club daily when available

  // Cumulative P&L series
  const cumulative = filteredDaily.reduce<
    { date: string; cumPnl: number; dailyPnl: number }[]
  >((acc, row) => {
    const prev = acc.length > 0 ? acc[acc.length - 1].cumPnl : 0;
    acc.push({
      date: format(parseISO(row.bet_date), "dd MMM"),
      dailyPnl: Number(row.net_pnl),
      cumPnl: Number((prev + Number(row.net_pnl)).toFixed(2)),
    });
    return acc;
  }, []);

  // Aggregate stats
  const totalPnl = filteredDaily.reduce((s, r) => s + Number(r.net_pnl), 0);
  const totalBets = filteredDaily.reduce(
    (s, r) => s + Number(r.bets_placed),
    0,
  );
  const totalWins = filteredDaily.reduce((s, r) => s + Number(r.wins), 0);
  const totalLoss = filteredDaily.reduce((s, r) => s + Number(r.losses), 0);
  const winRate = totalBets > 0 ? (totalWins / totalBets) * 100 : 0;
  const staked = filteredDaily.reduce((s, r) => s + Number(r.total_staked), 0);

  // Pie data
  const pieData = [
    { name: "Wins", value: totalWins, color: "#10B981" },
    { name: "Losses", value: totalLoss, color: "#EF4444" },
    {
      name: "Draws",
      value: totalBets - totalWins - totalLoss,
      color: "#F59E0B",
    },
  ].filter((d) => d.value > 0);

  const periods: { key: Period; label: string }[] = [
    { key: "today", label: "Today" },
    { key: "7d", label: "7d" },
    { key: "30d", label: "30d" },
    { key: "90d", label: "90d" },
    { key: "1y", label: "1y" },
    { key: "all", label: "All" },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-display-md font-display text-primary">
            Dashboard
          </h1>
          <p className="text-muted text-sm mt-1">
            Model performance and betting trends
          </p>
        </div>

        {/* Period filter */}
        <div className="flex items-center gap-1 bg-subtle rounded-xl p-1">
          {periods.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setPeriod(key)}
              className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                period === key
                  ? "bg-surface text-primary shadow-card"
                  : "text-muted hover:text-primary"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          {
            label: "P&L",
            value: `${totalPnl >= 0 ? "+" : ""}€${fmt(totalPnl)}`,
            color: totalPnl >= 0 ? "text-profit" : "text-loss",
            sub: `${period === "all" ? "all time" : period} period`,
          },
          {
            label: "Win rate",
            value: `${fmt(winRate, 1)}%`,
            color: "text-primary",
            sub: `${totalWins}W / ${totalLoss}L`,
          },
          {
            label: "Total bets",
            value: totalBets.toString(),
            color: "text-primary",
            sub: `€${fmt(staked)} staked`,
          },
          {
            label: "Current balance",
            value: `€${fmt(wallet?.balance ?? 0)}`,
            color: "text-primary",
            sub: "virtual wallet",
          },
        ].map(({ label, value, color, sub }) => (
          <div key={label} className="card px-5 py-4">
            <p className="text-xs font-medium text-muted mb-1">{label}</p>
            <p
              className={`text-xl font-display font-semibold tabular ${color}`}
            >
              {value}
            </p>
            <p className="text-xs text-muted mt-1">{sub}</p>
          </div>
        ))}
      </div>

      {/* Error state */}
      {error && (
        <div className="card px-6 py-4 border-loss bg-loss-bg">
          <p className="text-sm font-medium text-loss mb-1">
            Failed to load dashboard data
          </p>
          <p className="text-xs font-mono text-loss/70">{error}</p>
          <p className="text-xs text-muted mt-2">
            Check that NEXT_PUBLIC_SUPABASE_URL and
            NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY are set in Vercel → Settings →
            Environment Variables, then redeploy.
          </p>
        </div>
      )}

      {/* Cumulative P&L chart */}
      <div className="card px-6 py-5">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="font-display font-semibold text-primary">
              Cumulative P&amp;L
            </h3>
            <p className="text-xs text-muted mt-0.5">
              Running total since period start
            </p>
          </div>
          <span
            className={`text-sm font-semibold tabular font-mono ${
              totalPnl >= 0 ? "text-profit" : "text-loss"
            }`}
          >
            {totalPnl >= 0 ? "+" : ""}€{fmt(totalPnl)}
          </span>
        </div>

        {loading ? (
          <div className="h-56 skeleton rounded-xl" />
        ) : cumulative.length === 0 ? (
          <div className="h-56 flex items-center justify-center text-muted text-sm">
            No data for this period yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart
              data={cumulative}
              margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="gradProfit" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10B981" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#EF4444" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#EF4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                stroke="#E2E8F0"
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "#94A3B8" }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#94A3B8" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `€${v}`}
              />
              <Tooltip content={<PnlTooltip />} />
              <Area
                type="monotone"
                dataKey="cumPnl"
                stroke={totalPnl >= 0 ? "#10B981" : "#EF4444"}
                strokeWidth={2}
                fill={totalPnl >= 0 ? "url(#gradProfit)" : "url(#gradLoss)"}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Daily P&L + Pie row */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Daily bar chart */}
        <div className="card px-6 py-5 md:col-span-2">
          <h3 className="font-display font-semibold text-primary mb-1">
            Daily P&amp;L
          </h3>
          <p className="text-xs text-muted mb-5">Per-day profit and loss</p>
          {loading ? (
            <div className="h-44 skeleton rounded-xl" />
          ) : (
            <ResponsiveContainer width="100%" height={176}>
              <BarChart
                data={cumulative}
                margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
                barCategoryGap="30%"
              >
                <CartesianGrid
                  stroke="#E2E8F0"
                  strokeDasharray="3 3"
                  vertical={false}
                />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11, fill: "#94A3B8" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#94A3B8" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `€${v}`}
                />
                <Tooltip content={<PnlTooltip />} />
                <Bar dataKey="dailyPnl" radius={[4, 4, 0, 0]} fill="#3B82F6">
                  {cumulative.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={entry.dailyPnl >= 0 ? "#10B981" : "#EF4444"}
                      opacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Outcome distribution */}
        <div className="card px-6 py-5">
          <h3 className="font-display font-semibold text-primary mb-1">
            Outcomes
          </h3>
          <p className="text-xs text-muted mb-5">Win / Loss / Draw split</p>
          {loading || pieData.length === 0 ? (
            <div className="h-44 flex items-center justify-center text-muted text-sm">
              {loading ? "Loading…" : "No data"}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <ResponsiveContainer width="100%" height={140}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={44}
                    outerRadius={64}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} strokeWidth={0} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v: any, n: any) => [`${v} bets`, n]}
                    contentStyle={{
                      borderRadius: "12px",
                      border: "1px solid #E2E8F0",
                      fontSize: "12px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex gap-4 text-xs">
                {pieData.map((d) => (
                  <div key={d.name} className="flex items-center gap-1.5">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ background: d.color }}
                    />
                    <span className="text-muted">{d.name}</span>
                    <span className="font-semibold tabular text-primary">
                      {d.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Club breakdown */}
      <div className="card overflow-hidden">
        <div className="px-6 py-5 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="font-display font-semibold text-primary">
              Performance by club
            </h3>
            <p className="text-xs text-muted mt-0.5">
              P&amp;L from bets involving each team
            </p>
          </div>
          <select
            value={clubFilter}
            onChange={(e) => setClubFilter(e.target.value)}
            className="text-sm border border-border rounded-lg px-3 py-1.5 bg-surface text-primary
                       focus:outline-none focus:ring-2 focus:ring-accent/20 cursor-pointer"
          >
            <option value="all">All clubs</option>
            {teams.map((t) => (
              <option key={t.team_id} value={t.tla}>
                {t.team_name}
              </option>
            ))}
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-subtle/30">
                {["Club", "Bets", "Wins", "Losses", "Win rate", "P&L"].map(
                  (h) => (
                    <th
                      key={h}
                      className="px-5 py-3 text-left text-xs font-semibold text-muted"
                    >
                      {h}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {(clubFilter === "all"
                ? teams
                : teams.filter((t) => t.tla === clubFilter)
              ).map((team) => (
                <tr
                  key={team.team_id}
                  className="border-b border-border/50 table-row-hover last:border-0"
                >
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2.5">
                      <div
                        className="w-6 h-6 rounded-full bg-subtle border border-border
                                      flex items-center justify-center text-[9px] font-bold text-muted"
                      >
                        {team.tla}
                      </div>
                      <span className="font-medium text-primary">
                        {team.team_name}
                      </span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5 tabular font-mono text-xs">
                    {team.total_bets ?? 0}
                  </td>
                  <td className="px-5 py-3.5 tabular font-mono text-xs text-profit">
                    {team.wins ?? 0}
                  </td>
                  <td className="px-5 py-3.5 tabular font-mono text-xs text-loss">
                    {team.losses ?? 0}
                  </td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 max-w-16 h-1 bg-subtle rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent rounded-full"
                          style={{ width: `${team.win_rate_pct ?? 0}%` }}
                        />
                      </div>
                      <span className="tabular font-mono text-xs text-muted">
                        {fmt(team.win_rate_pct ?? 0, 1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    <span
                      className={`tabular font-mono text-xs font-semibold ${
                        (team.total_pnl ?? 0) >= 0 ? "text-profit" : "text-loss"
                      }`}
                    >
                      {(team.total_pnl ?? 0) >= 0 ? "+" : ""}€
                      {fmt(team.total_pnl ?? 0)}
                    </span>
                  </td>
                </tr>
              ))}
              {teams.length === 0 && !loading && (
                <tr>
                  <td
                    colSpan={6}
                    className="px-5 py-10 text-center text-muted text-sm"
                  >
                    No betting data yet. Results will appear after the first
                    matchday.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
