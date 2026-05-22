"""
backend/engine/kelly.py
------------------------
Stake sizing using the Kelly Criterion with bookmaker vig removal.

The core problem with naive Kelly:
  Bookmakers inflate implied probabilities so they sum to >100%.
  e.g. home 2.10, draw 3.40, away 3.60
  Implied: 1/2.10 + 1/3.40 + 1/3.60 = 0.476 + 0.294 + 0.278 = 1.048
  Overround = 4.8% — the bookmaker's guaranteed margin.

  If you use raw implied odds in Kelly, you're computing edge against
  inflated probabilities, so Kelly always understates your true edge.

The fix — de-vig before comparing to model probability:
  fair_prob = raw_implied / overround
  edge      = model_prob  - fair_prob
  Only bet if edge > MIN_EDGE (we require at least 3%).

Fractional Kelly (25%):
  Full Kelly maximises long-run growth but requires a perfectly
  calibrated model. Ours isn't perfect, so we use 25% of Kelly
  to reduce variance while still growing the bankroll when we have edge.
"""

from __future__ import annotations

# Minimum edge required to place a bet (after de-vig).
# Below this the variance isn't worth it even if Kelly suggests a positive stake.
MIN_EDGE       = 0.02   # 2% — lowered from 3% to find more bets.
                        # Our edge estimates have uncertainty; 3% was
                        # filtering out genuine value at 2-3% edge.

# Fraction of full Kelly to use. 33% balances growth and risk for a
# model with moderate calibration confidence.
KELLY_FRACTION = 0.33

# Hard cap — never risk more than 20% of bankroll on a single bet.
MAX_BET_FRACTION = 0.20


def remove_vig(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    """
    Converts raw bookmaker decimal odds to fair (de-vigged) probabilities.

    The overround (sum of implied probabilities) is always > 1.0 for a
    bookmaker to make profit. Dividing each implied probability by the
    overround distributes the margin equally across all outcomes.

    Returns fair probabilities for home, draw, away — summing to 1.0.
    """
    imp_home  = 1.0 / home_odds
    imp_draw  = 1.0 / draw_odds
    imp_away  = 1.0 / away_odds
    overround = imp_home + imp_draw + imp_away

    return {
        "HOME": round(imp_home / overround, 4),
        "DRAW": round(imp_draw / overround, 4),
        "AWAY": round(imp_away / overround, 4),
        "overround": round(overround, 4),
        "vig_pct":   round((overround - 1.0) * 100, 2),
    }


def compute_edge(
    model_prob:  float,
    fair_prob:   float,
) -> float:
    """
    Returns the edge: how much better our model thinks this outcome is
    compared to the fair market probability.

    Positive edge = we think this outcome is more likely than the market does.
    Negative edge = the market knows something we don't — skip this bet.
    """
    return round(model_prob - fair_prob, 4)


def kelly_stake(
    model_prob: float,
    odds:       float,
    fair_prob:  float,
    balance:    float,
) -> dict:
    """
    Calculates the optimal fractional Kelly stake for a single bet.

    Parameters
    ----------
    model_prob : our model's probability for this outcome
    odds       : decimal odds from the bookmaker (e.g. 2.10)
    fair_prob  : de-vigged market probability for this outcome
    balance    : current wallet balance in EUR

    Returns a dict with:
      stake     : amount to bet in EUR (0.0 if no edge)
      edge      : model_prob - fair_prob
      full_kelly: the unconstrained Kelly fraction
      reasoning : human-readable explanation of the decision
    """
    edge = compute_edge(model_prob, fair_prob)

    if edge <= MIN_EDGE:
        return {
            "stake":      0.0,
            "edge":       edge,
            "full_kelly": 0.0,
            "reasoning":  f"No edge ({edge:.1%} < {MIN_EDGE:.1%} minimum). Pass.",
        }

    b     = odds - 1.0          # net profit per unit staked on a win
    if b <= 0:
        return {
            "stake": 0.0, "edge": edge, "full_kelly": 0.0,
            "reasoning": "Odds ≤ 1.0 — invalid."
        }

    # Full Kelly: f* = (p*(b+1) - 1) / b
    full_kelly = (model_prob * (b + 1) - 1) / b
    if full_kelly <= 0:
        return {
            "stake": 0.0, "edge": edge, "full_kelly": full_kelly,
            "reasoning": "Kelly fraction is negative — expected value is negative. Pass.",
        }

    # Apply fraction and hard cap
    fractional = full_kelly * KELLY_FRACTION
    capped     = min(fractional, MAX_BET_FRACTION)
    stake      = round(capped * balance, 2)

    return {
        "stake":      stake,
        "edge":       edge,
        "full_kelly": round(full_kelly, 4),
        "frac_kelly": round(fractional, 4),
        "reasoning": (
            f"Edge {edge:.1%} | Full Kelly {full_kelly:.1%} | "
            f"Fractional ({KELLY_FRACTION:.0%}) {fractional:.1%} | "
            f"Stake €{stake:.2f}"
        ),
    }


def best_bet(
    model_probs: dict[str, float],
    home_odds:   float,
    draw_odds:   float,
    away_odds:   float,
    balance:     float,
) -> dict | None:
    """
    Given model probabilities and bookmaker odds for all three outcomes,
    finds the single best bet (highest edge after de-vig) and sizes it.

    Returns the best bet dict, or None if no outcome has sufficient edge.

    Parameters
    ----------
    model_probs : {"HOME": 0.54, "DRAW": 0.28, "AWAY": 0.18}
    home_odds   : e.g. 2.10
    draw_odds   : e.g. 3.40
    away_odds   : e.g. 3.60
    balance     : current wallet balance
    """
    fair   = remove_vig(home_odds, draw_odds, away_odds)
    odds_map = {"HOME": home_odds, "DRAW": draw_odds, "AWAY": away_odds}

    best      = None
    best_edge = MIN_EDGE   # must beat this to be worth betting

    for outcome in ["HOME", "DRAW", "AWAY"]:
        result = kelly_stake(
            model_prob = model_probs[outcome],
            odds       = odds_map[outcome],
            fair_prob  = fair[outcome],
            balance    = balance,
        )
        if result["stake"] > 0 and result["edge"] > best_edge:
            best_edge = result["edge"]
            best = {
                "action":     f"BET_{outcome}",
                "outcome":    outcome,
                "odds":       odds_map[outcome],
                "model_prob": model_probs[outcome],
                "fair_prob":  fair[outcome],
                "overround":  fair["overround"],
                "vig_pct":    fair["vig_pct"],
                **result,
            }

    return best