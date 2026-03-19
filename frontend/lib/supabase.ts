// lib/supabase.ts
// Browser-side Supabase client — uses the publishable key (read-only via RLS).
// Never expose the secret key here.

import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!;

export const supabase = createClient(supabaseUrl, supabaseKey);

// ─── Type definitions matching the DB schema ─────────────────────────────────

export interface Wallet {
  balance: number;
  total_staked: number;
  total_returned: number;
  inception_date: string;
}

export interface Bet {
  id: string;
  match_id: number;
  action: "BET_HOME" | "BET_DRAW" | "BET_AWAY";
  stake: number;
  odds: number;
  outcome: "WIN" | "LOSS" | null;
  pnl: number | null;
  balance_before: number;
  balance_after: number | null;
  placed_at: string;
  settled_at: string | null;
  matches?: {
    kickoff_time: string;
    season: string;
    home: { name: string; short_name: string; tla: string; crest_url?: string };
    away: { name: string; short_name: string; tla: string; crest_url?: string };
  };
}

export interface DailyPnl {
  bet_date: string;
  bets_placed: number;
  net_pnl: number;
  wins: number;
  losses: number;
  total_staked: number;
}

export interface TeamRecord {
  team_id: number;
  team_name: string;
  tla: string;
  crest_url?: string;
  total_bets: number;
  total_pnl: number;
  wins: number;
  losses: number;
  win_rate_pct: number;
}

export interface UpcomingMatch {
  match_id: number;
  kickoff_time: string;
  home_team: string;
  home_tla: string;
  home_crest?: string;
  away_team: string;
  away_tla: string;
  away_crest?: string;
  home_odds: number | null;
  draw_odds: number | null;
  away_odds: number | null;
  prob_home: number | null;
  prob_draw: number | null;
  prob_away: number | null;
  recommended_action: string | null;
}

// ─── Data fetchers ────────────────────────────────────────────────────────────

export async function getWallet(): Promise<Wallet | null> {
  const { data } = await supabase
    .from("wallet")
    .select("*")
    .eq("id", 1)
    .single();
  return data;
}

export async function getUpcomingMatches(): Promise<UpcomingMatch[]> {
  const { data } = await supabase
    .from("upcoming_matches_view")
    .select("*")
    .order("kickoff_time", { ascending: true })
    .limit(8);
  return data || [];
}

export async function getDailyPnl(days: number = 9999): Promise<DailyPnl[]> {
  // days=9999 means "all time" — skip the date filter entirely
  let query = supabase
    .from("daily_pnl")
    .select("*")
    .order("bet_date", { ascending: true });

  if (days < 9999) {
    const since = new Date();
    since.setDate(since.getDate() - days);
    query = query.gte("bet_date", since.toISOString().split("T")[0]);
  }

  const { data, error } = await query;
  if (error) throw new Error(`daily_pnl: ${error.message}`);
  return data || [];
}

export async function getTeamRecords(): Promise<TeamRecord[]> {
  const [recordsRes, teamsRes] = await Promise.all([
    supabase
      .from("team_betting_record")
      .select("*")
      .order("total_pnl", { ascending: false }),
    supabase.from("teams").select("tla, crest_url"),
  ]);
  if (recordsRes.error)
    throw new Error(`team_betting_record: ${recordsRes.error.message}`);
  const crestMap: Record<string, string> = {};
  for (const t of teamsRes.data || []) {
    if (t.tla && t.crest_url) crestMap[t.tla] = t.crest_url;
  }
  return (recordsRes.data || []).map((r: any) => ({
    ...r,
    crest_url: crestMap[r.tla],
  }));
}

export async function getBetHistory(limit: number = 100): Promise<Bet[]> {
  const { data, error } = await supabase
    .from("bets")
    .select(
      `
      *,
      matches (
        kickoff_time, season,
        home:teams!matches_home_team_id_fkey(name, short_name, tla, crest_url),
        away:teams!matches_away_team_id_fkey(name, short_name, tla, crest_url)
      )
    `,
    )
    .order("placed_at", { ascending: false })
    .limit(limit);
  if (error) throw new Error(`bets: ${error.message}`);
  return (data as Bet[]) || [];
}

export async function getRecentBets(limit: number = 5): Promise<Bet[]> {
  const { data } = await supabase
    .from("bets")
    .select(
      `
      *,
      matches (
        kickoff_time, season,
        home:teams!matches_home_team_id_fkey(name, short_name, tla, crest_url),
        away:teams!matches_away_team_id_fkey(name, short_name, tla, crest_url)
      )
    `,
    )
    .not("outcome", "is", null)
    .order("settled_at", { ascending: false })
    .limit(limit);
  return (data as Bet[]) || [];
}
