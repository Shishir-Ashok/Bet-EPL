"""
backend/data_pipeline/season_utils.py
---------------------------------------
Single source of truth for all season-related values.

The PL season runs August–May. The start year determines the season.
  April 2026 → month < 8 → start_year = 2025 → season "2025-26"
  October 2026 → month >= 8 → start_year = 2026 → season "2026-27"

Three formats needed across scripts:
  api_code   "2025"      football-data.org query param (start year as string)
  label      "2025-26"   stored in the DB matches.season column
  fdco_code  "2526"      football-data.co.uk CSV URL segment
"""

from datetime import datetime, timezone


def get_current_season() -> dict:
    """
    Returns season identifiers for the season currently in progress.

    >>> # Called in April 2026
    >>> get_current_season()
    {'api_code': '2025', 'label': '2025-26', 'fdco_code': '2526'}
    """
    now = datetime.now(timezone.utc)
    start_year = now.year if now.month >= 8 else now.year - 1
    return _season_from_start(start_year)


def get_season_for_label(label: str) -> dict:
    """
    Converts a DB season label back into the full dict.

    >>> get_season_for_label("2024-25")
    {'api_code': '2024', 'label': '2024-25', 'fdco_code': '2425'}
    """
    start_year = int(label[:4])
    return _season_from_start(start_year)


def get_historical_seasons(from_year: int = 2020) -> list[dict]:
    """
    Returns all COMPLETED seasons from from_year up to (not including)
    the current season. Safe to call at any point in the calendar year.

    In April 2026 (current season 2025-26), returns:
      2020-21, 2021-22, 2022-23, 2023-24, 2024-25
    """
    current_start = int(get_current_season()["api_code"])
    return [_season_from_start(y) for y in range(from_year, current_start)]


def get_all_seasons(from_year: int = 2020) -> list[dict]:
    """
    All seasons from from_year through and including the current season.
    Useful for full recalculations.
    """
    current_start = int(get_current_season()["api_code"])
    return [_season_from_start(y) for y in range(from_year, current_start + 1)]


def _season_from_start(start_year: int) -> dict:
    end_year = start_year + 1
    return {
        "api_code":  str(start_year),
        "label":     f"{start_year}-{str(end_year)[2:]}",
        "fdco_code": f"{str(start_year)[2:]}{str(end_year)[2:]}",
    }