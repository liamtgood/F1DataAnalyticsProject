"""
streamlit_app.py
----------------
F1 Two-Stage Prediction Dashboard

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import setup_cache

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Sidebar nav list ─────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child {
    display: none;          /* hide widget label */
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    gap: 2px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.45rem 0.75rem;
    border-radius: 0.4rem;
    cursor: pointer;
    width: 100%;
    transition: background 0.15s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.07);
}
/* hide the radio circle dot */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-of-type {
    display: none !important;
}
/* selected item — highlighted rectangle */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(200, 16, 46, 0.18);
    border: 1px solid rgba(200, 16, 46, 0.55);
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Team colours (approximate 2024)
TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#6692FF",
    "RB F1 Team": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Kick Sauber": "#C92D4B",
    "Haas F1 Team": "#B6BABD",
}

ROUNDS_PER_YEAR = {2022: 22, 2023: 22, 2024: 24, 2025: 24, 2026: 22}

# Country flag emoji for each 2024 round
RACE_FLAGS_2024 = {
    1:  "🇧🇭",  # Bahrain
    2:  "🇸🇦",  # Saudi Arabia
    3:  "🇦🇺",  # Australia
    4:  "🇯🇵",  # Japan
    5:  "🇨🇳",  # China
    6:  "🇺🇸",  # Miami (USA)
    7:  "🇮🇹",  # Emilia Romagna (Italy)
    8:  "🇲🇨",  # Monaco
    9:  "🇨🇦",  # Canada
    10: "🇪🇸",  # Spain
    11: "🇦🇹",  # Austria
    12: "🇬🇧",  # Britain
    13: "🇭🇺",  # Hungary
    14: "🇧🇪",  # Belgium
    15: "🇳🇱",  # Netherlands
    16: "🇮🇹",  # Italy (Monza)
    17: "🇦🇿",  # Azerbaijan
    18: "🇸🇬",  # Singapore
    19: "🇺🇸",  # United States (COTA)
    20: "🇲🇽",  # Mexico
    21: "🇧🇷",  # Brazil
    22: "🇺🇸",  # Las Vegas (USA)
    23: "🇶🇦",  # Qatar
    24: "🇦🇪",  # Abu Dhabi
}
# Country flag emoji for each 2025 round
RACE_FLAGS_2025 = {
    1:  "🇦🇺",  # Australia
    2:  "🇨🇳",  # China
    3:  "🇯🇵",  # Japan
    4:  "🇧🇭",  # Bahrain
    5:  "🇸🇦",  # Saudi Arabia
    6:  "🇺🇸",  # Miami (USA)
    7:  "🇮🇹",  # Emilia Romagna (Italy)
    8:  "🇲🇨",  # Monaco
    9:  "🇪🇸",  # Spain
    10: "🇨🇦",  # Canada
    11: "🇦🇹",  # Austria
    12: "🇬🇧",  # Britain
    13: "🇧🇪",  # Belgium
    14: "🇭🇺",  # Hungary
    15: "🇳🇱",  # Netherlands
    16: "🇮🇹",  # Italy (Monza)
    17: "🇦🇿",  # Azerbaijan
    18: "🇸🇬",  # Singapore
    19: "🇺🇸",  # United States (COTA)
    20: "🇲🇽",  # Mexico
    21: "🇧🇷",  # Brazil
    22: "🇺🇸",  # Las Vegas (USA)
    23: "🇶🇦",  # Qatar
    24: "🇦🇪",  # Abu Dhabi
}
FLAGS_BY_YEAR = {2024: RACE_FLAGS_2024, 2025: RACE_FLAGS_2025}

# Country flag emoji for each 2026 round
RACE_FLAGS_2026 = {
    1:  "🇦🇺",  # Australia
    2:  "🇨🇳",  # China
    3:  "🇯🇵",  # Japan
    4:  "🇺🇸",  # Miami
    5:  "🇨🇦",  # Canada
    6:  "🇲🇨",  # Monaco
    7:  "🇪🇸",  # Barcelona (Spain)
    8:  "🇦🇹",  # Austria
    9:  "🇬🇧",  # Britain
    10: "🇧🇪",  # Belgium
    11: "🇭🇺",  # Hungary
    12: "🇳🇱",  # Netherlands
    13: "🇮🇹",  # Italy
    14: "🇪🇸",  # Madrid (Spain)
    15: "🇦🇿",  # Azerbaijan
    16: "🇸🇬",  # Singapore
    17: "🇺🇸",  # United States
    18: "🇲🇽",  # Mexico
    19: "🇧🇷",  # Brazil
    20: "🇺🇸",  # Las Vegas
    21: "🇶🇦",  # Qatar
    22: "🇦🇪",  # Abu Dhabi
}
FLAGS_BY_YEAR = {2024: RACE_FLAGS_2024, 2025: RACE_FLAGS_2025, 2026: RACE_FLAGS_2026}

# 2024 race calendar (round → name).
RACE_CALENDAR_2024 = {
    1:  "Bahrain GP",
    2:  "Saudi Arabian GP",
    3:  "Australian GP",
    4:  "Japanese GP",
    5:  "Chinese GP",
    6:  "Miami GP",
    7:  "Emilia Romagna GP",
    8:  "Monaco GP",
    9:  "Canadian GP",
    10: "Spanish GP",
    11: "Austrian GP",
    12: "British GP",
    13: "Hungarian GP",
    14: "Belgian GP",
    15: "Dutch GP",
    16: "Italian GP",
    17: "Azerbaijan GP",
    18: "Singapore GP",
    19: "United States GP",
    20: "Mexico City GP",
    21: "São Paulo GP",
    22: "Las Vegas GP",
    23: "Qatar GP",
    24: "Abu Dhabi GP",
}
# 2025 race calendar (round → name).
RACE_CALENDAR_2025 = {
    1:  "Australian GP",
    2:  "Chinese GP",
    3:  "Japanese GP",
    4:  "Bahrain GP",
    5:  "Saudi Arabian GP",
    6:  "Miami GP",
    7:  "Emilia Romagna GP",
    8:  "Monaco GP",
    9:  "Spanish GP",
    10: "Canadian GP",
    11: "Austrian GP",
    12: "British GP",
    13: "Belgian GP",
    14: "Hungarian GP",
    15: "Dutch GP",
    16: "Italian GP",
    17: "Azerbaijan GP",
    18: "Singapore GP",
    19: "United States GP",
    20: "Mexico City GP",
    21: "São Paulo GP",
    22: "Las Vegas GP",
    23: "Qatar GP",
    24: "Abu Dhabi GP",
}
CALENDAR_BY_YEAR = {2024: RACE_CALENDAR_2024, 2025: RACE_CALENDAR_2025}

# 2026 race calendar (round → name).
RACE_CALENDAR_2026 = {
    1:  "Australian GP",
    2:  "Chinese GP",
    3:  "Japanese GP",
    4:  "Miami GP",
    5:  "Canadian GP",
    6:  "Monaco GP",
    7:  "Barcelona GP",
    8:  "Austrian GP",
    9:  "British GP",
    10: "Belgian GP",
    11: "Hungarian GP",
    12: "Dutch GP",
    13: "Italian GP",
    14: "Madrid GP",
    15: "Azerbaijan GP",
    16: "Singapore GP",
    17: "United States GP",
    18: "Mexico City GP",
    19: "São Paulo GP",
    20: "Las Vegas GP",
    21: "Qatar GP",
    22: "Abu Dhabi GP",
}
CALENDAR_BY_YEAR = {2024: RACE_CALENDAR_2024, 2025: RACE_CALENDAR_2025, 2026: RACE_CALENDAR_2026}

# Official F1 CDN circuit map image filenames (from media.formula1.com).
_F1_CDN_BASE = (
    "https://media.formula1.com/image/upload/c_fit,h_704/q_auto/"
    "content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/"
)

F1_CDN_FILENAMES_2024 = {
    1:  "Bahrain_Circuit",
    2:  "Saudi_Arabia_Circuit",
    3:  "Australia_Circuit",
    4:  "Japan_Circuit",
    5:  "China_Circuit",
    6:  "Miami_Circuit",
    7:  "Emilia_Romagna_Circuit",
    8:  "Monaco_Circuit",
    9:  "Canada_Circuit",
    10: "Spain_Circuit",
    11: "Austria_Circuit",
    12: "Great_Britain_Circuit",
    13: "Hungary_Circuit",
    14: "Belgium_Circuit",
    15: "Netherlands_Circuit",
    16: "Italy_Circuit",
    17: "Baku_Circuit",
    18: "Singapore_Circuit",
    19: "USA_Circuit",
    20: "Mexico_Circuit",
    21: "Brazil_Circuit",
    22: "Las_Vegas_Circuit",
    23: "Qatar_Circuit",
    24: "Abu_Dhabi_Circuit",
}
F1_CDN_FILENAMES_2025 = {
    1:  "Australia_Circuit",
    2:  "China_Circuit",
    3:  "Japan_Circuit",
    4:  "Bahrain_Circuit",
    5:  "Saudi_Arabia_Circuit",
    6:  "Miami_Circuit",
    7:  "Emilia_Romagna_Circuit",
    8:  "Monaco_Circuit",
    9:  "Spain_Circuit",
    10: "Canada_Circuit",
    11: "Austria_Circuit",
    12: "Great_Britain_Circuit",
    13: "Belgium_Circuit",
    14: "Hungary_Circuit",
    15: "Netherlands_Circuit",
    16: "Italy_Circuit",
    17: "Baku_Circuit",
    18: "Singapore_Circuit",
    19: "USA_Circuit",
    20: "Mexico_Circuit",
    21: "Brazil_Circuit",
    22: "Las_Vegas_Circuit",
    23: "Qatar_Circuit",
    24: "Abu_Dhabi_Circuit",
}
F1_CDN_FILENAMES_BY_YEAR = {2024: F1_CDN_FILENAMES_2024, 2025: F1_CDN_FILENAMES_2025}

F1_CDN_FILENAMES_2026 = {
    1:  "Australia_Circuit",
    2:  "China_Circuit",
    3:  "Japan_Circuit",
    4:  "Miami_Circuit",
    5:  "Canada_Circuit",
    6:  "Monaco_Circuit",
    7:  "Spain_Circuit",       # Barcelona-Catalunya
    8:  "Austria_Circuit",
    9:  "Great_Britain_Circuit",
    10: "Belgium_Circuit",
    11: "Hungary_Circuit",
    12: "Netherlands_Circuit",
    13: "Italy_Circuit",
    14: "Madrid_Circuit",     # New Madring circuit — CDN may not exist yet
    15: "Baku_Circuit",
    16: "Singapore_Circuit",
    17: "USA_Circuit",
    18: "Mexico_Circuit",
    19: "Brazil_Circuit",
    20: "Las_Vegas_Circuit",
    21: "Qatar_Circuit",
    22: "Abu_Dhabi_Circuit",
}
F1_CDN_FILENAMES_BY_YEAR = {2024: F1_CDN_FILENAMES_2024, 2025: F1_CDN_FILENAMES_2025, 2026: F1_CDN_FILENAMES_2026}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=None)
def get_track_image_path(year: int, round_number: int) -> str | None:
    """
    Returns path to a locally cached track layout image.
    On first call fetches the circuit map from the official F1 CDN and saves
    it to assets/tracks/{year}_{round}.webp.
    Subsequent calls are instant (local file read).
    """
    import requests
    out_dir = Path("assets/tracks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}_{round_number}.webp"
    if out_path.exists():
        return str(out_path)
    filename = F1_CDN_FILENAMES_BY_YEAR.get(year, {}).get(round_number)
    if not filename:
        return None
    try:
        url = f"{_F1_CDN_BASE}{filename}.webp"
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return str(out_path)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading practice data...", ttl=3600)
def load_practice_data(year: int, round_number: int) -> pd.DataFrame:
    """Load raw practice laps for all three sessions."""
    setup_cache("cache")
    from src.data_loader import load_practice_laps
    return load_practice_laps(year, round_number)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load both models once and keep in memory."""
    import xgboost as xgb
    import joblib

    models = {}
    qm_path = "models/qualifying_model.json"
    rm_path = "models/race_model.pkl"
    qm_cols_path = "models/qualifying_feature_cols.pkl"
    rm_cols_path = "models/race_feature_cols.pkl"

    if Path(qm_path).exists():
        m = xgb.XGBRegressor()
        m.load_model(qm_path)
        models["qualifying"] = m
        models["qualifying_cols"] = joblib.load(qm_cols_path)
    if Path(rm_path).exists():
        models["race"] = joblib.load(rm_path)
        models["race_cols"] = joblib.load(rm_cols_path)
    return models


@st.cache_data(show_spinner="Computing season accuracy...", ttl=3600)
def load_season_accuracy(year: int, calendar: dict) -> pd.DataFrame:
    """
    For each completed round in the season, run predictions and compute
    qualifying + race MAE. Returns a DataFrame with one row per round.
    """
    setup_cache("cache")
    from src.data_loader import load_race_results
    rows = []
    for rnd, name in calendar.items():
        try:
            # Check if actual results exist (race has happened)
            rr = load_race_results(year, rnd)
            if rr.empty:
                continue
            result = run_prediction(year, rnd, use_actual_grid=False)
            q_df = result["qualifying"]
            r_df = result["race"]
            row = {"round": rnd, "race": name}
            if "actual_grid_pos" in q_df.columns:
                mask = q_df["actual_grid_pos"].notna()
                if mask.sum() >= 2:
                    row["quali_mae"] = float(
                        (q_df.loc[mask, "predicted_grid_pos"] - q_df.loc[mask, "actual_grid_pos"]).abs().mean()
                    )
            if "actual_finish_pos" in r_df.columns:
                mask = r_df["actual_finish_pos"].notna()
                if "dnf" in r_df.columns:
                    mask = mask & (r_df["dnf"].fillna(0).astype(int) == 0)
                if mask.sum() >= 2:
                    row["race_mae"] = float(
                        (r_df.loc[mask, "predicted_finish_pos"] - r_df.loc[mask, "actual_finish_pos"]).abs().mean()
                    )
            if len(row) > 2:
                rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running predictions...", ttl=3600)
def run_prediction(year: int, round_number: int, use_actual_grid: bool, mc_samples: int = 1):
    """Load from pre-computed JSON if available, otherwise run live pipeline."""
    import json

    pred_file = Path(f"data/predictions/{year}_{round_number:02d}.json")

    if pred_file.exists() and not use_actual_grid:
        with open(pred_file) as f:
            payload = json.load(f)

        def _to_df(records):
            return pd.DataFrame(records) if records else pd.DataFrame()

        def _probs_from_dict(d):
            if not d:
                return None
            return pd.DataFrame(
                {driver: {int(k): v for k, v in pos_probs.items()}
                 for driver, pos_probs in d.items()}
            ).T

        return {
            "qualifying": _to_df(payload["qualifying"]),
            "race": _to_df(payload["race"]),
            "race_probs": _probs_from_dict(payload.get("race_probs")),
            "quali_probs": _probs_from_dict(payload.get("quali_probs")),
            "quali_raw": _to_df(payload.get("quali_raw", [])),
            "race_raw": _to_df(payload.get("race_raw", [])),
        }

    # Fall back to live pipeline (no pre-computed file, or use_actual_grid requested)
    setup_cache("cache")
    from src.pipeline import predict_race
    return predict_race(year, round_number, use_actual_grid=use_actual_grid, mc_samples=mc_samples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_laptime(seconds: float) -> str:
    """Convert float seconds to M:SS.mmm string."""
    if pd.isna(seconds):
        return "—"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Timing-tower HTML rendering (F1 live-timing style)
# ---------------------------------------------------------------------------

_COMPOUND_STYLE = {
    "SOFT":         ("#E8002D", "#fff", "S"),
    "MEDIUM":       ("#FFF200", "#000", "M"),
    "HARD":         ("#F0F0F0", "#000", "H"),
    "INTERMEDIATE": ("#39B549", "#000", "I"),
    "WET":          ("#0067FF", "#fff", "W"),
}
_TC = "padding:6px 10px;font-size:12px;"
_TM = f"{_TC}font-family:monospace;"


def _compound_badge_html(compound: str) -> str:
    bg, fg, ltr = _COMPOUND_STYLE.get(str(compound).upper().strip(), ("#555", "#ccc", "?"))
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:22px;height:22px;border-radius:50%;background:{bg};color:{fg};'
        f'font-weight:800;font-size:11px;">{ltr}</span>'
    )


def _sector_cell_html(value_s, col_min: float, col_max: float) -> str:
    if pd.isna(value_s):
        return f'<td style="{_TC}color:#555;">—</td>'
    delta = float(value_s) - col_min
    spread = max(col_max - col_min, 0.001)
    bar_w = max(4, int((1.0 - delta / spread) * 50))
    bar_c = "#00D2BE" if delta < 0.05 else ("#FFF200" if delta < 0.25 else "#E8002D")
    txt_c = "#00D2BE" if delta < 0.05 else "#e0e0e0"
    return (
        f'<td style="{_TC}">'
        f'<span style="font-family:monospace;color:{txt_c};">{float(value_s):.3f}</span>'
        f'<div style="height:3px;width:{bar_w}px;background:{bar_c};'
        f'border-radius:2px;margin-top:2px;"></div></td>'
    )


def _tt_pos_driver(idx: int, pos: int, driver: str, team: str) -> str:
    tc = team_color(team)
    bg = "#1e1e1e" if idx % 2 == 0 else "#242424"
    p_bg = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(pos, "#2e2e2e")
    p_fg = "#000" if pos <= 3 else "#fff"
    return (
        f'<tr style="background:{bg};border-left:3px solid {tc};">'        
        f'<td style="{_TC}text-align:center;width:44px;">'
        f'<span style="background:{p_bg};color:{p_fg};font-weight:700;'
        f'font-size:12px;padding:2px 7px;border-radius:4px;">{pos}</span></td>'
        f'<td style="{_TC}white-space:nowrap;">'
        f'<span style="color:#fff;font-weight:700;font-size:14px;letter-spacing:1px;">{driver}</span><br>'
        f'<span style="color:#555;font-size:10px;">{team}</span></td>'
    )


def _tt_table(thead: str, tbody: str) -> str:
    return (
        '<div style="border-radius:8px;overflow:hidden;'
        'font-family:Segoe UI,Arial,sans-serif;margin-bottom:12px;">'
        '<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:#0d0d0d;color:#555;font-size:10px;'
        f'text-transform:uppercase;letter-spacing:1px;">{thead}</tr></thead>'
        f'<tbody>{tbody}</tbody>'
        '</table></div>'
    )


def _th(label: str, align: str = "left") -> str:
    return f'<th style="padding:7px 10px;text-align:{align};font-weight:600;">{label}</th>'


def practice_timing_html(sess_df: pd.DataFrame, team_map: dict) -> str:
    df = sess_df.sort_values("best_lap_time_s").reset_index(drop=True)
    pole_s = df["best_lap_time_s"].min()
    s1_mn, s1_mx = df["s1_best_s"].min(), df["s1_best_s"].max()
    s2_mn, s2_mx = df["s2_best_s"].min(), df["s2_best_s"].max()
    s3_mn, s3_mx = df["s3_best_s"].min(), df["s3_best_s"].max()
    thead = (
        _th("Pos", "center") + _th("Driver") + _th("Tyre", "center")
        + _th("Laps", "center") + _th("Best Lap") + _th("Gap")
        + _th("S1") + _th("S2") + _th("S3") + _th("Speed Trap")
    )
    tbody = ""
    for i, row in df.iterrows():
        team = team_map.get(row["driver"], row.get("team", ""))
        gap_s = row["best_lap_time_s"] - pole_s
        g_str = "BEST" if gap_s < 0.001 else f"+{gap_s:.3f}"
        g_c = "#00D2BE" if gap_s < 0.001 else "#e0e0e0"
        spd = f'{row["speed_trap_max"]:.0f} km/h' if pd.notna(row.get("speed_trap_max")) else "—"
        tbody += (
            _tt_pos_driver(i, i + 1, row["driver"], team)
            + f'<td style="{_TC}text-align:center;">{_compound_badge_html(row.get("compound", "?"))}</td>'
            + f'<td style="{_TC}text-align:center;color:#aaa;">{int(row.get("num_laps", 0))}</td>'
            + f'<td style="{_TM}color:#fff;white-space:nowrap;">{_fmt_laptime(row["best_lap_time_s"])}</td>'
            + f'<td style="{_TM}color:{g_c};font-weight:600;">{g_str}</td>'
            + _sector_cell_html(row.get("s1_best_s"), s1_mn, s1_mx)
            + _sector_cell_html(row.get("s2_best_s"), s2_mn, s2_mx)
            + _sector_cell_html(row.get("s3_best_s"), s3_mn, s3_mx)
            + f'<td style="{_TC}color:#aaa;">{spd}</td></tr>'
        )
    return _tt_table(thead, tbody)


def qualifying_timing_html(df: pd.DataFrame, lap_col, lap_col_label: str = "Lap Time") -> str:
    df = df.sort_values("predicted_grid_pos").reset_index(drop=True)
    has_lap = bool(lap_col and lap_col in df.columns)
    has_gap = "proj_pole_gap_s" in df.columns
    has_actual = "actual_grid_pos" in df.columns
    thead = _th("P", "center") + _th("Driver")
    if has_lap:
        thead += _th(lap_col_label)
    if has_gap:
        thead += _th("Gap to Pole")
    if has_actual:
        thead += _th("Actual P", "center")
    tbody = ""
    for i, row in df.iterrows():
        pos = int(row["predicted_grid_pos"])
        cells = _tt_pos_driver(i, pos, row["driver"], row.get("team", ""))
        if has_lap:
            v = row.get(lap_col)
            cells += (
                f'<td style="{_TM}color:#fff;white-space:nowrap;">{_fmt_laptime(float(v))}</td>'
                if pd.notna(v) else f'<td style="{_TC}color:#555;">—</td>'
            )
        if has_gap:
            g = row.get("proj_pole_gap_s")
            if pd.notna(g):
                g = float(g)
                g_str = "POLE" if g == 0 else f"+{g:.3f}"
                g_c = "#00D2BE" if g == 0 else "#e0e0e0"
            else:
                g_str, g_c = "—", "#555"
            cells += f'<td style="{_TM}color:{g_c};font-weight:600;">{g_str}</td>'
        if has_actual:
            act = row.get("actual_grid_pos")
            act_str = str(int(act)) if pd.notna(act) else "—"
            if pd.notna(act):
                diff = pos - int(act)
                if diff != 0:
                    arr = "▼" if diff > 0 else "▲"
                    dc = "#E8002D" if diff > 0 else "#00D2BE"
                    act_str += f' <span style="color:{dc};font-size:10px;">{arr}{abs(diff)}</span>'
            cells += f'<td style="{_TC}text-align:center;color:#aaa;">{act_str}</td>'
        tbody += cells + "</tr>"
    return _tt_table(thead, tbody)


def race_timing_html(df: pd.DataFrame, mae: float = 3.1) -> str:
    df = df.sort_values("predicted_finish_pos").reset_index(drop=True)
    has_grid = "grid_used" in df.columns
    has_chg = "position_change" in df.columns
    has_actual = "actual_finish_pos" in df.columns
    n_drivers = len(df)
    thead = _th("Pos", "center") + _th("Driver")
    if has_grid:
        thead += _th("Grid", "center")
    if has_chg:
        thead += _th("\u0394 Pos", "center")
    thead += _th("Range", "center")
    if has_actual:
        thead += _th("Actual", "center")
    tbody = ""
    for i, row in df.iterrows():
        pos = int(row["predicted_finish_pos"])
        cells = _tt_pos_driver(i, pos, row["driver"], row.get("team", ""))
        if has_grid:
            gv = row.get("grid_used")
            cells += f'<td style="{_TC}text-align:center;color:#aaa;">{int(gv) if pd.notna(gv) else "—"}</td>'
        if has_chg:
            chg = row.get("position_change")
            if pd.notna(chg):
                chg = float(chg)
                if chg > 0:
                    chg_html = f'<span style="color:#00D2BE;font-weight:700;">\u25b2{int(abs(chg))}</span>'
                elif chg < 0:
                    chg_html = f'<span style="color:#E8002D;font-weight:700;">\u25bc{int(abs(chg))}</span>'
                else:
                    chg_html = '<span style="color:#555;">—</span>'
            else:
                chg_html = '<span style="color:#555;">—</span>'
            cells += f'<td style="{_TC}text-align:center;">{chg_html}</td>'
        lo = max(1, round(pos - mae))
        hi = min(n_drivers, round(pos + mae))
        cells += f'<td style="{_TC}text-align:center;color:#666;font-size:11px;">{lo}–{hi}</td>'
        if has_actual:
            act = row.get("actual_finish_pos")
            is_dnf = int(row.get("dnf", 0)) == 1
            if is_dnf:
                label = str(row.get("retire_label", "")) or "DNF"
                act_html = f'<span style="color:#E8002D;font-size:10px;font-weight:600;">{label}</span>'
            elif pd.notna(act):
                act_html = str(int(act))
            else:
                act_html = "—"
            cells += f'<td style="{_TC}text-align:center;color:#aaa;">{act_html}</td>'
        tbody += cells + "</tr>"
    return _tt_table(thead, tbody)


def team_color(team_name: str) -> str:
    for k, v in TEAM_COLORS.items():
        if k.lower() in str(team_name).lower():
            return v
    return "#888888"


def highlight_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply background colours to a result DataFrame for display."""
    styled = df.style
    if "position_change" in df.columns:
        styled = styled.applymap(
            lambda v: "color: #00CC44; font-weight: bold" if float(v) > 0
            else ("color: #FF4444; font-weight: bold" if float(v) < 0 else ""),
            subset=["position_change"],
        )
    return styled


def accuracy_metrics(predicted: pd.Series, actual: pd.Series) -> dict:
    """Compute MAE, top-3 hit rate, and Spearman rank correlation."""
    from scipy.stats import spearmanr
    mask = actual.notna() & predicted.notna()
    if mask.sum() < 2:
        return {}
    p, a = predicted[mask].astype(float), actual[mask].astype(float)
    mae = float((p - a).abs().mean())
    corr, _ = spearmanr(a, p)
    top3_actual = set(a.nsmallest(3).index)
    top3_pred = set(p.nsmallest(3).index)
    top3_hit = len(top3_actual & top3_pred) / max(len(top3_actual), 1)
    return {"mae": mae, "spearman": corr, "top3_hit": top3_hit}


def accuracy_scatter(pred_col: str, actual_col: str, df: pd.DataFrame,
                     xlabel: str, ylabel: str, title: str):
    """Scatter plot of predicted vs actual positions, coloured by team."""
    plot_df = df[[pred_col, actual_col, "driver"]].dropna().copy()
    if "team" in df.columns:
        plot_df["team"] = df.loc[plot_df.index, "team"]
        color_col = "team"
        color_map = {t: team_color(t) for t in plot_df["team"].unique()}
    else:
        color_col = None
        color_map = None

    fig = px.scatter(
        plot_df, x=actual_col, y=pred_col, text="driver",
        color=color_col, color_discrete_map=color_map,
        title=title, labels={actual_col: xlabel, pred_col: ylabel},
    )
    # Perfect prediction diagonal
    max_val = max(plot_df[[pred_col, actual_col]].max()) + 1
    fig.add_shape(type="line", x0=1, y0=1, x1=max_val, y1=max_val,
                  line=dict(color="#666", dash="dash"))
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(
        height=420, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(dtick=1), yaxis=dict(dtick=1),
    )
    return fig


def show_metric_badges(metrics: dict) -> None:
    """Render MAE / Spearman / Top-3 hit rate as st.metric columns."""
    if not metrics:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Position MAE", f"{metrics['mae']:.2f} pos")
    c2.metric("Rank Correlation", f"{metrics['spearman']:.2f}")
    c3.metric("Top-3 Hit Rate", f"{metrics['top3_hit']:.0%}")


def shap_bar_chart(shap_values, feature_names: list, title: str):
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    feat_imp = feat_imp.sort_values("importance", ascending=True).tail(15)
    fig = px.bar(
        feat_imp, x="importance", y="feature", orientation="h",
        title=title, color_discrete_sequence=["#E8002D"],
        labels={"importance": "Mean |SHAP value|", "feature": ""},
    )
    fig.update_layout(
        height=400, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def probability_chart(probs_df: pd.DataFrame, drivers_ordered: list, title: str):
    """
    Stacked horizontal bar chart showing each driver's probability of finishing
    in positions P1, P2, P3, P4-6, P7-10, P11+.
    Drivers ordered by predicted finishing position (best at top).
    """
    buckets = {
        "P1":   ([1],          "#FFD700"),
        "P2":   ([2],          "#C0C0C0"),
        "P3":   ([3],          "#CD7F32"),
        "P4-6": ([4, 5, 6],    "#4CAF50"),
        "P7-10":([7, 8, 9, 10],"#2196F3"),
        "P11+": (list(range(11, 21)), "#555555"),
    }
    plot_data = []
    for driver in drivers_ordered:
        if driver not in probs_df.index:
            continue
        row = probs_df.loc[driver]
        for label, (positions, color) in buckets.items():
            prob = sum(row.get(p, 0) for p in positions)
            plot_data.append({"Driver": driver, "Bucket": label, "Probability": prob, "Color": color})

    plot_df = pd.DataFrame(plot_data)
    color_map = {label: color for label, (_, color) in buckets.items()}
    fig = px.bar(
        plot_df,
        x="Probability", y="Driver", color="Bucket", orientation="h",
        color_discrete_map=color_map,
        title=title,
        category_orders={"Driver": drivers_ordered,
                         "Bucket": list(buckets.keys())},
    )
    fig.update_layout(
        height=max(350, len(drivers_ordered) * 22),
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def grid_bar_chart(df: pd.DataFrame, pos_col: str, title: str):
    df = df.copy()
    df["color"] = df["team"].apply(team_color) if "team" in df.columns else "#888"
    fig = px.bar(
        df.sort_values(pos_col), x="driver", y=pos_col,
        color="driver", color_discrete_map={r["driver"]: team_color(r.get("team", ""))
                                             for _, r in df.iterrows()},
        title=title,
        labels={pos_col: "Predicted Position", "driver": "Driver"},
    )
    fig.update_layout(
        showlegend=False, height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_SECTIONS = ["📊 Qualifying", "🏁 Race Prediction", "🎯 Accuracy", "🔬 Practice Data"]

with st.sidebar:
    st.title("🏎️ F1 Race Predictor")
    st.markdown("Two-stage ML pipeline: **Practice → Quali → Race**")
    st.divider()

    year = st.selectbox("Season", [2024, 2025, 2026], index=2)
    calendar = CALENDAR_BY_YEAR.get(year, {})
    # Only show rounds that have pre-computed prediction data
    race_options = [
        (r, name) for r, name in calendar.items()
        if Path(f"data/predictions/{year}_{r:02d}.json").exists()
    ]
    if not race_options:
        st.warning("No prediction data available for this season yet.")
        st.stop()
    race_labels = [f"R{r} — {name}" for r, name in race_options]
    selected_idx = st.selectbox(
        "Race Weekend",
        options=range(len(race_options)),
        format_func=lambda i: race_labels[i],
        index=len(race_options) - 1,  # default to most recent
    )
    round_number, race_name = race_options[selected_idx]

    st.divider()
    st.caption("VIEW")
    _active_section = st.radio("View", _SECTIONS, label_visibility="collapsed")

    st.divider()
    use_actual_grid = st.checkbox(
        "Use actual qualifying result",
        value=False,
        help="Skip Stage 1 and feed actual grid positions into the race model",
    )

    mc_samples = 100

    st.divider()
    models_loaded = Path("models/race_model.pkl").exists() and Path("models/qualifying_model.json").exists()
    if models_loaded:
        st.success("Models loaded ✓")
    else:
        st.warning("Models not trained yet.\nRun:\n```\npython -m src.qualifying_model train\npython -m src.race_model train\n```")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

race_label = CALENDAR_BY_YEAR.get(year, {}).get(round_number, f"Round {round_number}")
race_flag  = FLAGS_BY_YEAR.get(year, {}).get(round_number, "")
_title_col, _map_col = st.columns([3, 1])
with _title_col:
    st.title(f"{race_flag}  F1 Prediction — {year} {race_label}")
with _map_col:
    _img_path = get_track_image_path(year, round_number)
    if _img_path:
        st.image(_img_path, use_container_width=True)
    else:
        st.caption("Track image unavailable")

st.divider()

if not models_loaded:
    st.info("Train the models first using the instructions in the sidebar, then come back here.")
    st.stop()

# ---- Run prediction ----
with st.spinner("Running pipeline..."):
    try:
        result = run_prediction(year, round_number, use_actual_grid, mc_samples)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

# ===========================================================================
# Section 1 – Qualifying
# ===========================================================================
if _active_section == "📊 Qualifying":
    st.subheader(f"Predicted Qualifying Order — {race_label}")
    qual_df = result["qualifying"]

    # Accuracy badges (only when actual grid is available)
    if "actual_grid_pos" in qual_df.columns:
        qm = accuracy_metrics(
            qual_df.set_index("driver")["predicted_grid_pos"],
            qual_df.set_index("driver")["actual_grid_pos"],
        )
        show_metric_badges(qm)

    # ---- Projected pole time and gap to pole ----
    # q_best_lap_s is the actual qualifying lap time (available for past races).
    # For future races fall back to scaling FP3 times (~1.5% faster in qualifying).
    qual_df = qual_df.copy()
    if "q_best_lap_s" in qual_df.columns and qual_df["q_best_lap_s"].notna().any():
        pole_time_s = qual_df["q_best_lap_s"].min()
        qual_df["proj_pole_gap_s"] = (qual_df["q_best_lap_s"] - pole_time_s).round(3)
        proj_pole_label = _fmt_laptime(pole_time_s)
        lap_col = "q_best_lap_s"
    elif "fp_theoretical_best_s" in qual_df.columns and qual_df["fp_theoretical_best_s"].notna().any():
        # Predicted mode: show theoretical best lap (personal-best sectors from practice)
        pole_time_s = qual_df["fp_theoretical_best_s"].min()
        qual_df["proj_pole_gap_s"] = qual_df["fp_theoretical_gap_s"].round(3)
        proj_pole_label = _fmt_laptime(pole_time_s)
        lap_col = "fp_theoretical_best_s"
    elif "fp3_lap_delta_s" in qual_df.columns:
        fp3_best_proxy = qual_df["fp3_lap_delta_s"].min()
        qual_df["proj_pole_gap_s"] = (qual_df["fp3_lap_delta_s"] - fp3_best_proxy).round(3)
        proj_pole_label = "(FP3-based estimate)"
        lap_col = None
    else:
        proj_pole_label = None
        lap_col = None

    if proj_pole_label:
        if lap_col == "q_best_lap_s":
            label_prefix = "Actual pole lap"
        elif lap_col == "fp_theoretical_best_s":
            label_prefix = "Projected pole lap (best practice sectors)"
        else:
            label_prefix = "Estimated pole lap (FP3-based)"
        st.markdown(f"**{label_prefix}:** `{proj_pole_label}`")

    _lap_col_label = "Proj. Lap Time" if lap_col == "fp_theoretical_best_s" else "Lap Time"
    st.markdown(qualifying_timing_html(qual_df, lap_col, _lap_col_label), unsafe_allow_html=True)

    # SHAP
    st.subheader("Feature Importance (Qualifying Model)")
    try:
        from src.pipeline import get_shap_values
        shap_vals, _, feat_names = get_shap_values("qualifying", result["quali_raw"])
        st.plotly_chart(shap_bar_chart(shap_vals, feat_names, "Qualifying Model — Mean |SHAP|"),
                        use_container_width=True)
    except Exception as e:
        st.info(f"SHAP unavailable: {e}")

# ===========================================================================
# Section 2 – Race Prediction
# ===========================================================================
elif _active_section == "🏁 Race Prediction":
    st.subheader(f"Predicted Race Finishing Order — {race_label}")
    race_df = result["race"]

    # Accuracy badges (only when actual finishing positions are available)
    rm = {}
    if "actual_finish_pos" in race_df.columns:
        _finished = race_df[race_df["dnf"].fillna(0).astype(int) == 0] if "dnf" in race_df.columns else race_df
        rm = accuracy_metrics(
            _finished.set_index("driver")["predicted_finish_pos"],
            _finished.set_index("driver")["actual_finish_pos"],
        )
        show_metric_badges(rm)

    _race_mae = rm.get("mae", 3.1)
    st.markdown(race_timing_html(race_df, mae=_race_mae), unsafe_allow_html=True)

    # Monte Carlo position probability chart
    if result.get("race_probs") is not None:
        st.subheader("Position Probability Distribution")
        st.caption(f"Based on {mc_samples} Monte Carlo simulations with Gaussian feature noise.")
        drivers_ordered = result["race"]["driver"].tolist()
        fig_prob = probability_chart(result["race_probs"], drivers_ordered,
                                     "Finishing Position Probabilities")
        st.plotly_chart(fig_prob, use_container_width=True)

    # SHAP
    st.subheader("Feature Importance (Race Model)")
    try:
        from src.pipeline import get_shap_values
        shap_vals_r, _, feat_names_r = get_shap_values("race", result["race_raw"])
        st.plotly_chart(shap_bar_chart(shap_vals_r, feat_names_r, "Race Model — Mean |SHAP|"),
                        use_container_width=True)
    except Exception as e:
        st.info(f"SHAP unavailable: {e}")

# ===========================================================================
# Section 3 – Accuracy
# ===========================================================================
elif _active_section == "🎯 Accuracy":
    st.subheader(f"Model Accuracy — {race_label}")
    st.markdown("Comparing model predictions against actual results for this race weekend.")

    has_quali_actual = "actual_grid_pos" in result["qualifying"].columns
    has_race_actual = "actual_finish_pos" in result["race"].columns

    if not has_quali_actual and not has_race_actual:
        st.info("Actual results not available for this round yet — check back after the race weekend.")
    else:
        acc_col1, acc_col2 = st.columns(2)

        with acc_col1:
            st.markdown("#### Qualifying Model")
            if has_quali_actual:
                qm = accuracy_metrics(
                    result["qualifying"].set_index("driver")["predicted_grid_pos"],
                    result["qualifying"].set_index("driver")["actual_grid_pos"],
                )
                show_metric_badges(qm)
                fig_qs = accuracy_scatter(
                    "predicted_grid_pos", "actual_grid_pos",
                    result["qualifying"].set_index("driver").reset_index(),
                    xlabel="Actual Grid Position",
                    ylabel="Predicted Grid Position",
                    title="Qualifying: Predicted vs Actual",
                )
                st.plotly_chart(fig_qs, use_container_width=True)
            else:
                st.info("No actual qualifying data available for this round.")

        with acc_col2:
            st.markdown("#### Race Model")
            if has_race_actual:
                _race_df = result["race"]
                _finished = _race_df[_race_df["dnf"].fillna(0).astype(int) == 0] if "dnf" in _race_df.columns else _race_df
                rm = accuracy_metrics(
                    _finished.set_index("driver")["predicted_finish_pos"],
                    _finished.set_index("driver")["actual_finish_pos"],
                )
                show_metric_badges(rm)
                fig_rs = accuracy_scatter(
                    "predicted_finish_pos", "actual_finish_pos",
                    _finished.set_index("driver").reset_index(),
                    xlabel="Actual Finishing Position",
                    ylabel="Predicted Finishing Position",
                    title="Race: Predicted vs Actual (finishers only)",
                )
                st.plotly_chart(fig_rs, use_container_width=True)
            else:
                st.info("No actual race data available for this round.")

        # Position error table
        st.markdown("#### Per-Driver Prediction Error")
        err_frames = []
        if has_quali_actual:
            qe = result["qualifying"][["driver", "predicted_grid_pos", "actual_grid_pos"]].copy()
            qe["quali_error"] = (qe["predicted_grid_pos"] - qe["actual_grid_pos"]).round(1)
            err_frames.append(qe.set_index("driver")[["predicted_grid_pos", "actual_grid_pos", "quali_error"]])
        if has_race_actual:
            re = result["race"][["driver", "predicted_finish_pos", "actual_finish_pos"] + 
                                 (["dnf"] if "dnf" in result["race"].columns else [])].copy()
            if "dnf" in re.columns:
                re = re[re["dnf"].fillna(0).astype(int) == 0]
            re["race_error"] = (re["predicted_finish_pos"] - re["actual_finish_pos"]).round(1)
            err_frames.append(re.set_index("driver")[["predicted_finish_pos", "actual_finish_pos", "race_error"]])

        if err_frames:
            err_df = err_frames[0].join(err_frames[1], how="outer") if len(err_frames) == 2 else err_frames[0]
            err_df = err_df.reset_index()
            err_df.columns = [c.replace("_", " ").title() for c in err_df.columns]
            st.dataframe(err_df, use_container_width=True, hide_index=True)

        # Rolling accuracy across the season
        st.markdown("#### Season Rolling Accuracy")
        st.caption("MAE per race across all completed rounds this season. Lower is better.")
        with st.spinner("Loading season accuracy..."):
            season_acc = load_season_accuracy(year, CALENDAR_BY_YEAR.get(year, {}))
        if season_acc.empty:
            st.info("Not enough completed races to show rolling accuracy yet.")
        else:
            fig_roll = go.Figure()
            if "quali_mae" in season_acc.columns:
                fig_roll.add_trace(go.Scatter(
                    x=season_acc["race"], y=season_acc["quali_mae"],
                    mode="lines+markers", name="Qualifying MAE",
                    line=dict(color="#00D2BE", width=2),
                    marker=dict(size=7),
                ))
            if "race_mae" in season_acc.columns:
                fig_roll.add_trace(go.Scatter(
                    x=season_acc["race"], y=season_acc["race_mae"],
                    mode="lines+markers", name="Race MAE",
                    line=dict(color="#E8002D", width=2),
                    marker=dict(size=7),
                ))
            fig_roll.update_layout(
                height=380,
                yaxis=dict(title="Mean Absolute Error (positions)", rangemode="tozero"),
                xaxis=dict(tickangle=-35),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                legend=dict(orientation="h", y=1.05),
                margin=dict(l=20, r=20, t=30, b=80),
            )
            st.plotly_chart(fig_roll, use_container_width=True)

# ===========================================================================
# Section 4 – Practice Data
# ===========================================================================
elif _active_section == "🔬 Practice Data":
    st.subheader(f"Practice Session Data — {race_label}")

    try:
        raw_practice = load_practice_data(year, round_number)
    except Exception as exc:
        st.error(f"Could not load practice data: {exc}")
        raw_practice = pd.DataFrame()

    if raw_practice.empty:
        st.info("No practice data available for this round.")
    else:
        # Team colour map for driver bars
        quali_team_map = (
            result["qualifying"].set_index("driver")["team"].to_dict()
            if "team" in result["qualifying"].columns else {}
        )

        for session_id in ("FP1", "FP2", "FP3"):
            sess_df = raw_practice[raw_practice["session"] == session_id].copy()
            if sess_df.empty:
                continue

            st.markdown(f"### {session_id}")

            sess_df = sess_df.sort_values("best_lap_time_s").reset_index(drop=True)
            pole_s = sess_df["best_lap_time_s"].min()
            sess_df["gap_to_best_s"] = (sess_df["best_lap_time_s"] - pole_s).round(3)
            sess_df["team"] = sess_df["driver"].map(quali_team_map)

            st.markdown(practice_timing_html(sess_df, quali_team_map), unsafe_allow_html=True)
            st.divider()

