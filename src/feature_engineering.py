"""
feature_engineering.py
-----------------------
Builds two flat feature DataFrames from raw loader outputs:

  1. qualifying_features  — used to train/infer Stage 1 (qualifying predictor)
  2. race_features        — used to train/infer Stage 2 (race outcome predictor)

Entry points
------------
build_qualifying_features(year, round_number)  -> pd.DataFrame
build_race_features(year, round_number)         -> pd.DataFrame
build_and_save_dataset(years, rounds, out_dir)  -> saves CSVs for model training
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data_loader import (
    load_practice_laps,
    load_qualifying_results,
    load_race_results,
    load_driver_standings_before_round,
    load_constructor_standings_before_round,
    load_recent_race_results,
    load_circuit_info,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team name normalization (FastF1 names → Ergast names aren't always equal)
# ---------------------------------------------------------------------------

TEAM_ALIAS: dict[str, str] = {
    "Red Bull Racing": "Red Bull",
    "Mercedes": "Mercedes",
    "Ferrari": "Ferrari",
    "McLaren": "McLaren",
    "Aston Martin": "Aston Martin",
    "Alpine": "Alpine F1 Team",
    "Williams": "Williams",
    "AlphaTauri": "AlphaTauri",
    "RB": "RB F1 Team",
    "Alfa Romeo": "Alfa Romeo",
    "Sauber": "Kick Sauber",
    "Haas F1 Team": "Haas F1 Team",
}


def _normalize_team(name: str) -> str:
    return TEAM_ALIAS.get(name, name)


# ---------------------------------------------------------------------------
# Stage 1: Qualifying features
# ---------------------------------------------------------------------------

def build_qualifying_features(year: int, round_number: int) -> pd.DataFrame:
    """
    Combine FP1/FP2/FP3 practice laps with Ergast form data.
    Returns one row per driver with label = grid_position.
    """
    practice = load_practice_laps(year, round_number)
    quali = load_qualifying_results(year, round_number)

    if practice.empty or quali.empty:
        logger.warning("Missing data for year=%d round=%d, skipping.", year, round_number)
        return pd.DataFrame()

    # ---- Pivot practice sessions ----
    fp3 = practice[practice["session"] == "FP3"].copy()
    fp2 = practice[practice["session"] == "FP2"].copy()
    fp1 = practice[practice["session"] == "FP1"].copy()

    def _session_deltas(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """For each driver compute delta-to-session-best for lap time and sectors."""
        if df.empty:
            return pd.DataFrame(columns=["driver"])
        session_best_lap = df["best_lap_time_s"].min()
        session_best_s1 = df["s1_best_s"].min()
        session_best_s2 = df["s2_best_s"].min()
        session_best_s3 = df["s3_best_s"].min()

        out = df[["driver", "best_lap_time_s", "s1_best_s", "s2_best_s", "s3_best_s",
                  "speed_trap_max", "num_laps"]].copy()
        out[f"{prefix}_lap_delta_s"] = out["best_lap_time_s"] - session_best_lap
        out[f"{prefix}_s1_delta_s"] = out["s1_best_s"] - session_best_s1
        out[f"{prefix}_s2_delta_s"] = out["s2_best_s"] - session_best_s2
        out[f"{prefix}_s3_delta_s"] = out["s3_best_s"] - session_best_s3
        out[f"{prefix}_speed_max"] = out["speed_trap_max"]
        out[f"{prefix}_num_laps"] = out["num_laps"]
        return out[["driver", f"{prefix}_lap_delta_s", f"{prefix}_s1_delta_s",
                    f"{prefix}_s2_delta_s", f"{prefix}_s3_delta_s",
                    f"{prefix}_speed_max", f"{prefix}_num_laps"]]

    fp3_feat = _session_deltas(fp3, "fp3")
    fp2_feat = _session_deltas(fp2, "fp2")
    fp1_feat = _session_deltas(fp1, "fp1")

    # ---- Merge all sessions ----
    feat = quali[["driver", "team", "grid_position", "q_best_lap_s"]].copy()
    feat["team"] = feat["team"].apply(_normalize_team)

    for session_feat in (fp3_feat, fp2_feat, fp1_feat):
        if not session_feat.empty:
            feat = feat.merge(session_feat, on="driver", how="left")

    # ---- Teammate delta in FP3 ----
    if not fp3.empty:
        fp3_pace = fp3[["driver", "best_lap_time_s"]].copy()
        fp3_pace["team"] = fp3_pace["driver"].map(
            dict(zip(quali["driver"], quali["team"].apply(_normalize_team)))
        )
        teammate_best = (
            fp3_pace.groupby("team")["best_lap_time_s"]
            .min()
            .reset_index()
            .rename(columns={"best_lap_time_s": "teammate_best_s"})
        )
        fp3_pace = fp3_pace.merge(teammate_best, on="team", how="left")
        fp3_pace["fp3_teammate_delta_s"] = fp3_pace["best_lap_time_s"] - fp3_pace["teammate_best_s"]
        feat = feat.merge(fp3_pace[["driver", "fp3_teammate_delta_s"]], on="driver", how="left")

    # ---- Ergast form features ----
    driver_standings = load_driver_standings_before_round(year, round_number)
    constructor_standings = load_constructor_standings_before_round(year, round_number)
    recent_results = load_recent_race_results(year, round_number, n_races=3)

    if not driver_standings.empty:
        feat = feat.merge(driver_standings[["driver", "driver_points", "driver_championship_pos"]],
                          on="driver", how="left")
    else:
        feat["driver_points"] = np.nan
        feat["driver_championship_pos"] = np.nan

    if not constructor_standings.empty:
        constructor_standings["team"] = constructor_standings["team_ergast"].apply(_normalize_team)
        feat = feat.merge(
            constructor_standings[["team", "constructor_points", "constructor_championship_pos"]],
            on="team", how="left",
        )
    else:
        feat["constructor_points"] = np.nan
        feat["constructor_championship_pos"] = np.nan

    if not recent_results.empty:
        feat = feat.merge(recent_results, on="driver", how="left")
    else:
        feat["avg_finish_last_n"] = np.nan
        feat["dnf_rate_last_n"] = np.nan

    # ---- Circuit info ----
    circuit = load_circuit_info(year, round_number)
    feat["is_street_circuit"] = circuit["is_street_circuit"]
    feat["year"] = year
    feat["round"] = round_number
    feat["circuit_name"] = circuit["circuit_name"]

    # ---- Rookie / new-driver defaults ----
    # Drivers not in the previous season (rookies or those without history)
    # get 0 points, last-place championship positions, and a last-place avg finish.
    # This is applied BEFORE the generic median-fill so these columns get
    # sensible values rather than the median of the existing grid.
    n_drivers = len(feat)
    rookie_defaults = {
        "driver_points": 0.0,
        "driver_championship_pos": float(n_drivers),
        "constructor_points": 0.0,
        "constructor_championship_pos": float(n_drivers // 2),
        "avg_finish_last_n": 20.0,
        "dnf_rate_last_n": 0.0,
    }
    for col, default in rookie_defaults.items():
        if col in feat.columns:
            feat[col] = feat[col].fillna(default)

    # ---- Fill NaN with sensible defaults ----
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat[numeric_cols] = feat[numeric_cols].fillna(feat[numeric_cols].median())

    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: Race features
# ---------------------------------------------------------------------------

def build_race_features(year: int, round_number: int,
                        predicted_grid: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build race features. If `predicted_grid` is provided (DataFrame with
    columns driver, predicted_grid_pos) it is used instead of actual quali results.
    Label = finish_position.
    """
    race = load_race_results(year, round_number)
    quali = load_qualifying_results(year, round_number)
    practice = load_practice_laps(year, round_number)

    if race.empty:
        logger.warning("No race results for year=%d round=%d, skipping.", year, round_number)
        return pd.DataFrame()

    def _retire_label(status: str) -> str:
        s = str(status)
        if any(kw in s for kw in ("Finished", "Lapped", "+")):
            return ""
        sl = s.lower()
        if "not qualify" in sl or sl in ("dnq", "107%"):
            return "DNQ"
        if "not start" in sl or sl == "dns" or "withdrew" in sl:
            return "DNS"
        if "disqualified" in sl or sl == "dsq":
            return "DSQ"
        return "DNF"

    feat = race[["driver", "team", "finish_position", "grid_position", "pit_stops", "status"]].copy()
    feat["team"] = feat["team"].apply(_normalize_team)
    feat["retire_label"] = feat["status"].apply(_retire_label)
    feat["dnf"] = (feat["retire_label"] != "").astype(int)

    # Use predicted grid if provided (inference mode), else actual
    if predicted_grid is not None and not predicted_grid.empty:
        feat = feat.merge(
            predicted_grid[["driver", "predicted_grid_pos"]].rename(
                columns={"predicted_grid_pos": "grid_used"}
            ),
            on="driver", how="left",
        )
        feat["grid_used"] = feat["grid_used"].fillna(feat["grid_position"])
    else:
        feat["grid_used"] = feat["grid_position"]

    # ---- Qualifying pace ----
    if not quali.empty:
        feat = feat.merge(quali[["driver", "q_best_lap_s"]], on="driver", how="left")
        pole_time = feat["q_best_lap_s"].min()
        feat["q_gap_to_pole_s"] = (feat["q_best_lap_s"] - pole_time).round(4)
        # Teammate qualifying delta (how much slower than faster team-mate)
        team_best_q = feat.groupby("team")["q_best_lap_s"].min().reset_index()
        team_best_q = team_best_q.rename(columns={"q_best_lap_s": "_team_q_best"})
        feat = feat.merge(team_best_q, on="team", how="left")
        feat["q_teammate_gap_s"] = (feat["q_best_lap_s"] - feat["_team_q_best"]).round(4)
        feat = feat.drop(columns=["_team_q_best"])
    else:
        feat["q_best_lap_s"] = np.nan
        feat["q_gap_to_pole_s"] = np.nan
        feat["q_teammate_gap_s"] = np.nan

    # ---- Pace vs grid mismatch ----
    # grid_vs_pace_delta > 0: starting worse than pace implies (grid penalty / bad quali) -> will climb
    # grid_vs_pace_delta < 0: starting better than pace implies -> may drop back
    if feat["q_gap_to_pole_s"].notna().any():
        feat["_pace_rank"] = feat["q_gap_to_pole_s"].rank(method="first")
        feat["grid_vs_pace_delta"] = feat["grid_used"] - feat["_pace_rank"]
        feat = feat.drop(columns=["_pace_rank"])
    else:
        feat["grid_vs_pace_delta"] = 0.0

    # ---- Practice pace features (race trim proxy) ----
    # FP3 is closest to race setup; FP2 long-run pace is also informative
    for sess_id, prefix in (("FP3", "fp3"), ("FP2", "fp2")):
        sess_df = practice[practice["session"] == sess_id] if not practice.empty else pd.DataFrame()
        if sess_df.empty:
            for col in (f"{prefix}_race_lap_delta_s", f"{prefix}_race_s1_delta_s",
                        f"{prefix}_race_s2_delta_s", f"{prefix}_race_s3_delta_s",
                        f"{prefix}_race_speed_max"):
                feat[col] = np.nan
            continue

        session_best = sess_df["best_lap_time_s"].min()
        pace = sess_df[["driver", "best_lap_time_s", "s1_best_s", "s2_best_s",
                        "s3_best_s", "speed_trap_max"]].copy()
        pace[f"{prefix}_race_lap_delta_s"] = pace["best_lap_time_s"] - session_best
        pace[f"{prefix}_race_s1_delta_s"] = pace["s1_best_s"] - sess_df["s1_best_s"].min()
        pace[f"{prefix}_race_s2_delta_s"] = pace["s2_best_s"] - sess_df["s2_best_s"].min()
        pace[f"{prefix}_race_s3_delta_s"] = pace["s3_best_s"] - sess_df["s3_best_s"].min()
        pace[f"{prefix}_race_speed_max"] = pace["speed_trap_max"]

        # Teammate delta: how much slower than the faster team-mate in this session
        team_map = dict(zip(feat["driver"], feat["team"]))
        pace["team"] = pace["driver"].map(team_map)
        team_best = pace.groupby("team")["best_lap_time_s"].min().reset_index()
        team_best = team_best.rename(columns={"best_lap_time_s": "_team_best"})
        pace = pace.merge(team_best, on="team", how="left")
        pace[f"{prefix}_race_teammate_delta_s"] = pace["best_lap_time_s"] - pace["_team_best"]

        keep = ["driver", f"{prefix}_race_lap_delta_s", f"{prefix}_race_s1_delta_s",
                f"{prefix}_race_s2_delta_s", f"{prefix}_race_s3_delta_s",
                f"{prefix}_race_speed_max", f"{prefix}_race_teammate_delta_s"]
        feat = feat.merge(pace[keep], on="driver", how="left")

    # ---- Ergast form features ----
    driver_standings = load_driver_standings_before_round(year, round_number)
    constructor_standings = load_constructor_standings_before_round(year, round_number)
    recent_results = load_recent_race_results(year, round_number, n_races=3)

    if not driver_standings.empty:
        feat = feat.merge(driver_standings[["driver", "driver_points", "driver_championship_pos"]],
                          on="driver", how="left")
    else:
        feat["driver_points"] = np.nan
        feat["driver_championship_pos"] = np.nan

    if not constructor_standings.empty:
        constructor_standings["team"] = constructor_standings["team_ergast"].apply(_normalize_team)
        feat = feat.merge(
            constructor_standings[["team", "constructor_points", "constructor_championship_pos"]],
            on="team", how="left",
        )
    else:
        feat["constructor_points"] = np.nan
        feat["constructor_championship_pos"] = np.nan

    if not recent_results.empty:
        feat = feat.merge(recent_results, on="driver", how="left")
    else:
        feat["avg_finish_last_n"] = np.nan
        feat["dnf_rate_last_n"] = np.nan

    # ---- Circuit info ----
    circuit = load_circuit_info(year, round_number)
    feat["is_street_circuit"] = circuit["is_street_circuit"]
    feat["overtake_index"] = _get_overtake_index(circuit["circuit_name"])
    feat["year"] = year
    feat["round"] = round_number
    feat["circuit_name"] = circuit["circuit_name"]

    # ---- Rookie / new-driver defaults ----
    n_drivers = len(feat)
    rookie_defaults = {
        "driver_points": 0.0,
        "driver_championship_pos": float(n_drivers),
        "constructor_points": 0.0,
        "constructor_championship_pos": float(n_drivers // 2),
        "avg_finish_last_n": 20.0,
        "dnf_rate_last_n": 0.0,
    }
    for col, default in rookie_defaults.items():
        if col in feat.columns:
            feat[col] = feat[col].fillna(default)

    # ---- Fill NaN ----
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat[numeric_cols] = feat[numeric_cols].fillna(feat[numeric_cols].median())

    # ---- Delta target ----
    # position_delta = finish_position - grid_used  (negative = positions gained)
    feat["position_delta"] = feat["finish_position"] - feat["grid_used"]

    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Overtake difficulty index
# ---------------------------------------------------------------------------
# Scale: 0.0 (nearly impossible to pass) → 1.0 (lots of passing).
# Matched by checking if the key is a lowercase substring of the circuit name.
_OVERTAKE_INDEX_MAP: list[tuple[str, float]] = [
    ("monaco",          0.05),
    ("hungaroring",     0.15),
    ("marina bay",      0.25),
    ("zandvoort",       0.30),
    ("albert park",     0.35),
    ("imola",           0.35),
    ("enzo e dino",     0.35),  # Imola alternate Ergast name
    ("barcelona",       0.40),
    ("catalunya",       0.40),
    ("suzuka",          0.45),
    ("yas marina",      0.45),
    ("miami",           0.45),
    ("jeddah",          0.45),
    ("lusail",          0.50),
    ("silverstone",     0.55),
    ("baku",            0.55),
    ("hermanos",        0.55),  # Mexico
    ("circuit of the americas", 0.60),
    ("red bull ring",   0.60),
    ("shanghai",        0.60),
    ("gilles villeneuve", 0.60),  # Canada
    ("las vegas",       0.60),
    ("bahrain",         0.65),
    ("interlagos",      0.65),
    ("josé carlos pace", 0.65),  # Brazil alternate
    ("spa",             0.75),
    ("monza",           0.75),
    ("nazionale di monza", 0.75),
]
_OVERTAKE_INDEX_DEFAULT = 0.50


def _get_overtake_index(circuit_name: str) -> float:
    name_lower = circuit_name.lower()
    for keyword, value in _OVERTAKE_INDEX_MAP:
        if keyword in name_lower:
            return value
    return _OVERTAKE_INDEX_DEFAULT


# ---------------------------------------------------------------------------
# Batch dataset builder
# ---------------------------------------------------------------------------

# FastF1 round counts per year
ROUNDS_PER_YEAR = {2022: 22, 2023: 22, 2024: 24, 2025: 24, 2026: 22}


def build_and_save_dataset(
    years: List[int],
    out_dir: str = "data/processed",
    rounds_override: Optional[dict] = None,
) -> None:
    """
    Iterate over years × rounds, build features, concatenate, and save CSVs.

    Args:
        years: e.g. [2022, 2023, 2024]
        out_dir: directory to save qualifying_features.csv and race_features.csv
        rounds_override: dict mapping year → list of round numbers to process
                         (useful for partial runs / testing)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_quali: List[pd.DataFrame] = []
    all_race: List[pd.DataFrame] = []

    for year in years:
        round_count = rounds_override.get(year) if rounds_override else None
        if round_count is None:
            round_count = list(range(3, ROUNDS_PER_YEAR.get(year, 22) + 1))

        for rnd in round_count:
            # Skip rounds that haven't happened yet
            try:
                import fastf1 as _ff1
                from datetime import datetime, timezone
                _evt = _ff1.get_event(year, rnd)
                _race_date = _evt.get_session_date("Race", utc=True)
                if hasattr(_race_date, "tzinfo") and _race_date.tzinfo is None:
                    _race_date = _race_date.replace(tzinfo=timezone.utc)
                if _race_date > datetime.now(timezone.utc):
                    logger.info("Skipping future round year=%d round=%d (%s)", year, rnd, _evt["EventName"])
                    continue
            except Exception:
                pass  # if we can't determine date, attempt anyway

            logger.info("Processing year=%d round=%d ...", year, rnd)
            try:
                qf = build_qualifying_features(year, rnd)
                if not qf.empty:
                    all_quali.append(qf)
            except Exception as exc:
                logger.error("Qualifying features failed year=%d round=%d: %s", year, rnd, exc)

            try:
                rf = build_race_features(year, rnd)
                if not rf.empty:
                    all_race.append(rf)
            except Exception as exc:
                logger.error("Race features failed year=%d round=%d: %s", year, rnd, exc)

    if all_quali:
        pd.concat(all_quali, ignore_index=True).to_csv(
            Path(out_dir) / "qualifying_features.csv", index=False
        )
        logger.info("Saved qualifying_features.csv (%d rows)", sum(len(d) for d in all_quali))

    if all_race:
        pd.concat(all_race, ignore_index=True).to_csv(
            Path(out_dir) / "race_features.csv", index=False
        )
        logger.info("Saved race_features.csv (%d rows)", sum(len(d) for d in all_race))
