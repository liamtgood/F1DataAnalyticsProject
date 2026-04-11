"""
data_loader.py
--------------
Functions to load raw data from FastF1 and the Ergast REST API.
All results are returned as pandas DataFrames.

FastF1 cache must be enabled before calling these functions.
Call `setup_cache(path)` once at startup.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------------

def setup_cache(cache_dir: str = "cache") -> None:
    """Enable FastF1 disk cache to avoid repeated downloads."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    logger.info("FastF1 cache enabled at: %s", cache_dir)


# ---------------------------------------------------------------------------
# FastF1 helpers
# ---------------------------------------------------------------------------

def _get_session(year: int, round_number: int, identifier: str) -> Optional[fastf1.core.Session]:
    """Load a FastF1 session, returning None on failure."""
    try:
        session = fastf1.get_session(year, round_number, identifier)
        session.load(telemetry=False, weather=False, messages=False)
        return session
    except Exception as exc:
        logger.warning("Could not load %s | year=%d round=%d: %s", identifier, year, round_number, exc)
        return None


def load_practice_laps(year: int, round_number: int) -> pd.DataFrame:
    """
    Return aggregated practice session lap data for all drivers.

    Columns returned per driver per session:
        driver, session (FP1/FP2/FP3), best_lap_time_s, s1_best_s, s2_best_s,
        s3_best_s, speed_trap_max, compound, num_laps
    """
    results = []
    for session_id in ("FP1", "FP2", "FP3"):
        session = _get_session(year, round_number, session_id)
        if session is None:
            continue

        laps = session.laps.copy()
        if laps.empty:
            continue

        # Keep only accurate laps (no pit in/out, no yellow, no deleted)
        laps = laps[laps["IsAccurate"] == True].copy()  # noqa: E712

        # Convert timedelta columns to seconds
        for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
            if col in laps.columns:
                laps[f"{col}_s"] = laps[col].dt.total_seconds()

        agg = (
            laps.groupby("Driver")
            .agg(
                best_lap_time_s=("LapTime_s", "min"),
                s1_best_s=("Sector1Time_s", "min"),
                s2_best_s=("Sector2Time_s", "min"),
                s3_best_s=("Sector3Time_s", "min"),
                speed_trap_max=("SpeedST", "max"),
                num_laps=("LapTime_s", "count"),
            )
            .reset_index()
            .rename(columns={"Driver": "driver"})
        )

        # Add most-used compound
        if "Compound" in laps.columns:
            compound_mode = (
                laps.groupby("Driver")["Compound"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "UNKNOWN")
                .reset_index()
                .rename(columns={"Driver": "driver", "Compound": "compound"})
            )
            agg = agg.merge(compound_mode, on="driver", how="left")
        else:
            agg["compound"] = "UNKNOWN"

        agg["session"] = session_id
        agg["year"] = year
        agg["round"] = round_number
        results.append(agg)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def load_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Return qualifying classification with grid position and best lap time.

    Columns: driver, grid_position, q_best_lap_s, team, year, round
    """
    session = _get_session(year, round_number, "Q")
    if session is None:
        return pd.DataFrame()

    results = session.results[["Abbreviation", "Position", "Q3", "Q2", "Q1", "TeamName"]].copy()
    results = results.rename(columns={"Abbreviation": "driver", "Position": "grid_position", "TeamName": "team"})

    # Best qualifying time across Q1/Q2/Q3
    for col in ("Q3", "Q2", "Q1"):
        results[f"{col}_s"] = results[col].dt.total_seconds()

    results["q_best_lap_s"] = results[["Q3_s", "Q2_s", "Q1_s"]].min(axis=1)
    results = results[["driver", "grid_position", "q_best_lap_s", "team"]].copy()
    results["year"] = year
    results["round"] = round_number
    results["grid_position"] = pd.to_numeric(results["grid_position"], errors="coerce")
    return results.dropna(subset=["grid_position"]).reset_index(drop=True)


def load_race_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Return race classification with finishing position, grid position,
    pit stop count, and status.

    Columns: driver, team, finish_position, grid_position, points,
             status, pit_stops, year, round
    """
    session = _get_session(year, round_number, "R")
    if session is None:
        return pd.DataFrame()

    res = session.results[
        ["Abbreviation", "TeamName", "Position", "GridPosition", "Points", "Status"]
    ].copy()
    res = res.rename(
        columns={
            "Abbreviation": "driver",
            "TeamName": "team",
            "Position": "finish_position",
            "GridPosition": "grid_position",
            "Points": "points",
            "Status": "status",
        }
    )

    # Count pit stops from race laps
    laps = session.laps.copy()
    if not laps.empty and "PitOutTime" in laps.columns:
        pit_counts = (
            laps[laps["PitOutTime"].notna()]
            .groupby("Driver")
            .size()
            .reset_index(name="pit_stops")
            .rename(columns={"Driver": "driver"})
        )
        res = res.merge(pit_counts, on="driver", how="left")
        res["pit_stops"] = res["pit_stops"].fillna(1).astype(int)
    else:
        res["pit_stops"] = 1

    res["year"] = year
    res["round"] = round_number
    res["finish_position"] = pd.to_numeric(res["finish_position"], errors="coerce")
    res["grid_position"] = pd.to_numeric(res["grid_position"], errors="coerce")
    return res.dropna(subset=["finish_position"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ergast API helpers
# ---------------------------------------------------------------------------

ERGAST_BASE = "https://api.jolpi.ca/ergast/f1"  # community mirror, unlimited
_ERGAST_CACHE: dict = {}


def _ergast_get(endpoint: str, retries: int = 3) -> dict:
    """GET from Ergast (or community mirror), with simple retry and in-memory cache."""
    if endpoint in _ERGAST_CACHE:
        return _ERGAST_CACHE[endpoint]

    url = f"{ERGAST_BASE}/{endpoint}.json?limit=100"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            _ERGAST_CACHE[endpoint] = data
            return data
        except requests.RequestException as exc:
            logger.warning("Ergast request failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return {}


def _load_final_driver_standings(year: int) -> pd.DataFrame:
    """Return final driver championship standings for a completed season."""
    data = _ergast_get(f"{year}/last/driverStandings")
    try:
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()
    rows = []
    for entry in standings_list:
        rows.append({
            "driver": entry["Driver"]["code"],
            "driver_points": float(entry.get("points", 0)),
            "driver_wins": int(entry.get("wins", 0)),
            "driver_championship_pos": int(entry.get("position", 99)),
        })
    return pd.DataFrame(rows)


def _load_final_constructor_standings(year: int) -> pd.DataFrame:
    """Return final constructor championship standings for a completed season."""
    data = _ergast_get(f"{year}/last/constructorStandings")
    try:
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()
    rows = []
    for entry in standings_list:
        rows.append({
            "team_ergast": entry["Constructor"]["name"],
            "constructor_points": float(entry.get("points", 0)),
            "constructor_championship_pos": int(entry.get("position", 99)),
        })
    return pd.DataFrame(rows)


def _load_final_season_recent_results(year: int, n_races: int = 3) -> pd.DataFrame:
    """Return avg finishing stats from the last n_races of a completed season."""
    data = _ergast_get(f"{year}/last/results")
    try:
        last_round = int(data["MRData"]["RaceTable"]["Races"][0]["round"])
    except (KeyError, IndexError, ValueError):
        return pd.DataFrame(columns=["driver", "avg_finish_last_n", "dnf_rate_last_n"])

    all_results = []
    for r in range(max(1, last_round - n_races + 1), last_round + 1):
        rdata = _ergast_get(f"{year}/{r}/results")
        try:
            race_results = rdata["MRData"]["RaceTable"]["Races"][0]["Results"]
        except (KeyError, IndexError):
            continue
        for entry in race_results:
            pos = entry.get("position")
            try:
                pos = int(pos)
            except (TypeError, ValueError):
                continue
            status = entry.get("status", "")
            finished = not any(kw in status for kw in
                               ("Retired", "Accident", "Mechanical", "Collision", "Disqualified"))
            all_results.append({
                "driver": entry["Driver"]["code"],
                "finish_position": pos,
                "dnf": 0 if finished else 1,
            })

    if not all_results:
        return pd.DataFrame(columns=["driver", "avg_finish_last_n", "dnf_rate_last_n"])
    df = pd.DataFrame(all_results)
    return (
        df.groupby("driver")
        .agg(avg_finish_last_n=("finish_position", "mean"), dnf_rate_last_n=("dnf", "mean"))
        .reset_index()
    )


def load_driver_standings_before_round(year: int, round_number: int) -> pd.DataFrame:
    """
    Return driver championship standings after round (round_number - 1).
    For round 1, falls back to the previous season's final standings.
    Columns: driver, driver_points, driver_wins, driver_championship_pos
    """
    prev_round = round_number - 1
    if prev_round < 1:
        # No current-season data yet — use previous season's final standings
        return _load_final_driver_standings(year - 1)

    data = _ergast_get(f"{year}/{prev_round}/driverStandings")
    try:
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = []
    for entry in standings_list:
        rows.append({
            "driver": entry["Driver"]["code"],
            "driver_points": float(entry.get("points", 0)),
            "driver_wins": int(entry.get("wins", 0)),
            "driver_championship_pos": int(entry.get("position", 99)),
        })
    return pd.DataFrame(rows)


def load_constructor_standings_before_round(year: int, round_number: int) -> pd.DataFrame:
    """
    Return constructor championship standings after round (round_number - 1).
    For round 1, falls back to the previous season's final standings.
    Columns: team_ergast, constructor_points, constructor_championship_pos
    """
    prev_round = round_number - 1
    if prev_round < 1:
        return _load_final_constructor_standings(year - 1)

    data = _ergast_get(f"{year}/{prev_round}/constructorStandings")
    try:
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    rows = []
    for entry in standings_list:
        rows.append({
            "team_ergast": entry["Constructor"]["name"],
            "constructor_points": float(entry.get("points", 0)),
            "constructor_championship_pos": int(entry.get("position", 99)),
        })
    return pd.DataFrame(rows)


def load_recent_race_results(year: int, round_number: int, n_races: int = 3) -> pd.DataFrame:
    """
    Return each driver's average finishing position across the last `n_races`
    races (before this round).
    For round 1, falls back to the last n_races of the previous season.
    Columns: driver, avg_finish_last_n, dnf_rate_last_n
    """
    if round_number <= 1:
        return _load_final_season_recent_results(year - 1, n_races)

    all_results = []
    for r in range(max(1, round_number - n_races), round_number):
        data = _ergast_get(f"{year}/{r}/results")
        try:
            race_results = data["MRData"]["RaceTable"]["Races"][0]["Results"]
        except (KeyError, IndexError):
            continue
        for entry in race_results:
            pos = entry.get("position")
            try:
                pos = int(pos)
            except (TypeError, ValueError):
                continue
            status = entry.get("status", "")
            finished = not any(kw in status for kw in ("Retired", "Accident", "Mechanical", "Collision", "Disqualified"))
            all_results.append({
                "driver": entry["Driver"]["code"],
                "finish_position": pos,
                "dnf": 0 if finished else 1,
            })

    if not all_results:
        return pd.DataFrame(columns=["driver", "avg_finish_last_n", "dnf_rate_last_n"])

    df = pd.DataFrame(all_results)
    agg = (
        df.groupby("driver")
        .agg(avg_finish_last_n=("finish_position", "mean"), dnf_rate_last_n=("dnf", "mean"))
        .reset_index()
    )
    return agg


def load_circuit_info(year: int, round_number: int) -> dict:
    """
    Return high-level circuit metadata.
    Returns dict with keys: circuit_name, locality, country, is_street_circuit
    """
    data = _ergast_get(f"{year}/{round_number}/races")
    try:
        race = data["MRData"]["RaceTable"]["Races"][0]
        circuit = race["Circuit"]
        locality = circuit.get("Location", {}).get("locality", "")
        country = circuit.get("Location", {}).get("country", "")
        name = circuit.get("circuitName", "")
        # Heuristic for street circuits
        street_keywords = ["Monaco", "Baku", "Singapore", "Melbourne", "Jeddah", "Las Vegas", "Miami"]
        is_street = any(kw.lower() in name.lower() or kw.lower() in locality.lower() for kw in street_keywords)
        return {"circuit_name": name, "locality": locality, "country": country, "is_street_circuit": int(is_street)}
    except (KeyError, IndexError):
        return {"circuit_name": "Unknown", "locality": "", "country": "", "is_street_circuit": 0}
