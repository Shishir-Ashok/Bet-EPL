import { Suspense } from 'react'
import Link from 'next/link'
import { getWallet, getUpcomingMatches, getRecentBets } from '@/lib/supabase'
import { format, formatDistanceToNow } from 'date-fns'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 2) {
  return n.toLocaleString('en-IE', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function pct(n: number) {
  return (n * 100).toFixed(1) + '%'
}

function actionLabel(action: string | null) {
  if (!action) return null
  const map: Record<string, { label: string; color: string }> = {
    BET_HOME:  { label: 'Bet Home',  color: 'text-accent  bg-accent-light  border-accent/20' },
    BET_DRAW:  { label: 'Bet Draw',  color: 'text-draw    bg-draw-bg       border-draw-border' },
    BET_AWAY:  { label: 'Bet Away',  color: 'text-profit  bg-profit-bg     border-profit-border' },
    PASS:      { label: 'Pass',      color: 'text-muted   bg-subtle        border-border' },
  }
  return map[action] ?? null
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default async function HomePage() {
  const [wallet, upcoming, recentBets] = await Promise.all([
    getWallet(),
    getUpcomingMatches(),
    getRecentBets(5),
  ])

  const balance      = wallet?.balance ?? 10
  const staked       = wallet?.total_staked ?? 0
  const returned     = wallet?.total_returned ?? 0
  const pnlEur       = returned - staked
  const roi          = staked > 0 ? (pnlEur / staked) * 100 : 0
  const isProfit     = pnlEur >= 0
  const inception    = wallet?.inception_date
    ? format(new Date(wallet.inception_date), 'MMM d, yyyy')
    : '—'

  const wonBets  = recentBets.filter(b => b.outcome === 'WIN').length
  const lostBets = recentBets.filter(b => b.outcome === 'LOSS').length

  return (
    <div className="max-w-7xl mx-auto px-6 py-12 space-y-16">

      {/* ─── Hero ─────────────────────────────────────────────────────────── */}
      <section className="space-y-10">

        {/* Eyebrow */}
        <div className="flex items-center gap-3">
          <span className="section-label">Since {inception}</span>
          <span className="w-1 h-1 rounded-full bg-border inline-block" />
          <span className="section-label">Premier League</span>
        </div>

        {/* Big number */}
        <div className="space-y-4 animate-fade-up">
          <p className="text-sm font-medium text-muted">Total P&amp;L</p>
          <div className="flex items-baseline gap-4">
            <h1 className={`text-display-2xl font-display tabular leading-none ${
              isProfit ? 'text-profit' : 'text-loss'
            }`}>
              {isProfit ? '+' : ''}€{fmt(pnlEur)}
            </h1>
            <span className={`text-2xl font-semibold tabular ${
              isProfit ? 'text-profit' : 'text-loss'
            }`}>
              {isProfit ? '↑' : '↓'} {fmt(Math.abs(roi), 1)}% ROI
            </span>
          </div>
          <p className="text-muted text-sm">
            Starting balance <span className="tabular font-medium text-primary">€10.00</span>
            &nbsp;·&nbsp;
            Current balance <span className="tabular font-medium text-primary">€{fmt(balance)}</span>
          </p>
        </div>

        {/* Stat row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-up animate-delay-100 opacity-0-init">
          {[
            { label: 'Total staked',   value: `€${fmt(staked)}`,        sub: 'all time' },
            { label: 'Total returned', value: `€${fmt(returned)}`,      sub: 'from wins' },
            { label: 'Recent record',  value: `${wonBets}W / ${lostBets}L`, sub: 'last 5 settled' },
            { label: 'Next matches',   value: `${upcoming.length}`,     sub: 'scheduled' },
          ].map(({ label, value, sub }) => (
            <div key={label} className="card px-5 py-4 space-y-1">
              <p className="text-xs font-medium text-muted">{label}</p>
              <p className="text-xl font-display font-semibold tabular text-primary">{value}</p>
              <p className="text-xs text-muted">{sub}</p>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="flex gap-3 animate-fade-up animate-delay-200 opacity-0-init">
          <Link href="/dashboard" className="btn-primary">
            View Dashboard
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5-5 5M6 12h12" />
            </svg>
          </Link>
          <Link href="/history" className="btn-secondary">
            Bet History
          </Link>
        </div>
      </section>

      {/* ─── Upcoming Matches ─────────────────────────────────────────────── */}
      <section className="space-y-5 animate-fade-up animate-delay-300 opacity-0-init">
        <div className="flex items-center justify-between">
          <h2 className="text-display-sm font-display text-primary">Upcoming fixtures</h2>
          <span className="section-label">{upcoming.length} scheduled</span>
        </div>

        {upcoming.length === 0 ? (
          <div className="card px-6 py-12 text-center text-muted text-sm">
            No upcoming fixtures found. Check back closer to the next matchday.
          </div>
        ) : (
          <div className="grid gap-3">
            {upcoming.map(match => {
              const ph = match.prob_home ?? 0
              const pd = match.prob_draw ?? 0
              const pa = match.prob_away ?? 0
              const total = ph + pd + pa || 1
              const action = actionLabel(match.recommended_action)
              const kickoff = new Date(match.kickoff_time)

              return (
                <div key={match.match_id}
                  className="card px-5 py-4 flex flex-col sm:flex-row sm:items-center gap-4 hover:shadow-card-md transition-shadow duration-200">

                  {/* Teams + kickoff */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 font-display font-semibold text-primary">
                      <span>{match.home_team}</span>
                      <span className="text-muted font-normal text-sm">vs</span>
                      <span>{match.away_team}</span>
                    </div>
                    <p className="text-xs text-muted mt-0.5">
                      {format(kickoff, 'EEE d MMM')} · {format(kickoff, 'HH:mm')} UTC
                    </p>
                  </div>

                  {/* Probability bars */}
                  <div className="flex-1 space-y-1.5">
                    {[
                      { label: 'H', prob: ph / total, color: 'bg-accent'  },
                      { label: 'D', prob: pd / total, color: 'bg-draw'    },
                      { label: 'A', prob: pa / total, color: 'bg-profit'  },
                    ].map(({ label, prob, color }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="text-[10px] font-mono font-semibold text-muted w-3">{label}</span>
                        <div className="flex-1 h-1.5 bg-subtle rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${color} transition-all duration-700`}
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono tabular text-muted w-8 text-right">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Odds */}
                  <div className="flex gap-1.5 items-center flex-shrink-0">
                    {match.home_odds && (
                      <>
                        <span className="odds-pill">{match.home_odds?.toFixed(2)}</span>
                        <span className="odds-pill">{match.draw_odds?.toFixed(2)}</span>
                        <span className="odds-pill">{match.away_odds?.toFixed(2)}</span>
                      </>
                    )}
                  </div>

                  {/* Bot action */}
                  {action && (
                    <div className="flex-shrink-0">
                      <span className={`text-xs font-semibold px-2.5 py-1 rounded-lg border ${action.color}`}>
                        {action.label}
                      </span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* ─── Recent settled bets ──────────────────────────────────────────── */}
      {recentBets.length > 0 && (
        <section className="space-y-5 animate-fade-up animate-delay-400 opacity-0-init">
          <div className="flex items-center justify-between">
            <h2 className="text-display-sm font-display text-primary">Recent bets</h2>
            <Link href="/history" className="text-sm text-accent hover:underline font-medium">
              View all →
            </Link>
          </div>

          <div className="card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {['Match', 'Action', 'Odds', 'Stake', 'Outcome', 'P&L'].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentBets.map(bet => {
                  const m = bet.matches
                  const isWin = bet.outcome === 'WIN'
                  const isLoss = bet.outcome === 'LOSS'
                  return (
                    <tr key={bet.id} className="border-b border-border/50 table-row-hover last:border-0">
                      <td className="px-5 py-3.5">
                        <span className="font-medium text-primary">
                          {m?.home?.tla} vs {m?.away?.tla}
                        </span>
                        <p className="text-xs text-muted mt-0.5">
                          {m?.kickoff_time ? format(new Date(m.kickoff_time), 'dd MMM yy') : '—'}
                        </p>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="text-xs font-mono font-medium text-muted">
                          {bet.action.replace('BET_', '')}
                        </span>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="tabular font-mono text-xs">{bet.odds.toFixed(2)}</span>
                      </td>
                      <td className="px-5 py-3.5 tabular font-mono text-xs">
                        €{bet.stake.toFixed(2)}
                      </td>
                      <td className="px-5 py-3.5">
                        {isWin && <span className="badge-profit">Win</span>}
                        {isLoss && <span className="badge-loss">Loss</span>}
                        {!bet.outcome && <span className="badge-pending">Open</span>}
                      </td>
                      <td className="px-5 py-3.5">
                        <span className={`tabular font-mono text-xs font-semibold ${
                          isWin ? 'text-profit' : isLoss ? 'text-loss' : 'text-muted'
                        }`}>
                          {bet.pnl != null
                            ? `${bet.pnl >= 0 ? '+' : ''}€${Math.abs(bet.pnl).toFixed(2)}`
                            : '—'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ─── Footer ───────────────────────────────────────────────────────── */}
      <footer className="pt-8 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted">
        <span>PL Betting Bot — all bets are virtual. No real money is used.</span>
        <span>XGBoost + DQN · Updated automatically after each matchday</span>
      </footer>

    </div>
  )
}
