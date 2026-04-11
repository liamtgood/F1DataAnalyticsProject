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

ROUNDS_PER_YEAR = {2022: 22, 2023: 22, 2024: 24}

# 2024 race calendar (round → name). Rounds 1-2 excluded (no prior-race form data).
RACE_CALENDAR_2024 = {
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
CALENDAR_BY_YEAR = {2024: RACE_CALENDAR_2024}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

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


@st.cache_data(show_spinner="Running predictions...", ttl=3600)
def run_prediction(year: int, round_number: int, use_actual_grid: bool):
    """Cache predictions per (year, round) so switching tabs doesn't re-run."""
    setup_cache("cache")
    from src.pipeline import predict_race
    return predict_race(year, round_number, use_actual_grid=use_actual_grid)


def run_whatif_prediction(year: int, round_number: int, overrides: dict, use_actual_grid: bool):
    """What-if predictions are NOT cached (need fresh result per override combo)."""
    setup_cache("cache")
    from src.pipeline import predict_race
    return predict_race(year, round_number, overrides=overrides, use_actual_grid=use_actual_grid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

with st.sidebar:
    st.title("🏎️ F1 Race Predictor")
    st.markdown("Two-stage ML pipeline: **Practice → Quali → Race**")
    st.divider()

    year = st.selectbox("Season", [2024], index=0)
    calendar = CALENDAR_BY_YEAR.get(year, {})
    race_options = list(calendar.items())  # [(round, name), ...]
    race_labels = [f"R{r} — {name}" for r, name in race_options]
    selected_idx = st.selectbox(
        "Race Weekend",
        options=range(len(race_options)),
        format_func=lambda i: race_labels[i],
        index=3,  # default to R6 Miami
    )
    round_number, race_name = race_options[selected_idx]
    use_actual_grid = st.checkbox(
        "Use actual qualifying result",
        value=False,
        help="Skip Stage 1 and feed actual grid positions into the race model",
    )

    st.divider()
    models_loaded = Path("models/race_model.pkl").exists() and Path("models/qualifying_model.json").exists()
    if models_loaded:
        st.success("Models loaded ✓")
    else:
        st.warning("Models not trained yet.\nRun:\n```\npython -m src.qualifying_model train\npython -m src.race_model train\n```")

    run_btn = st.button("▶ Run Prediction", type="primary", disabled=not models_loaded, use_container_width=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

race_label = CALENDAR_BY_YEAR.get(year, {}).get(round_number, f"Round {round_number}")
st.title(f"F1 Prediction — {year} {race_label}")

if not models_loaded:
    st.info("Train the models first using the instructions in the sidebar, then come back here.")
    st.stop()

# ---- Run prediction ----
with st.spinner("Running pipeline..."):
    try:
        result = run_prediction(year, round_number, use_actual_grid)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

tab_quali, tab_race, tab_accuracy, tab_whatif = st.tabs(["📊 Qualifying", "🏁 Race Prediction", "🎯 Accuracy", "🔧 What-If"])

# ===========================================================================
# Tab 1 – Qualifying
# ===========================================================================
with tab_quali:
    st.subheader(f"Predicted Qualifying Order — {race_label}")
    qual_df = result["qualifying"]

    # Accuracy badges (only when actual grid is available)
    if "actual_grid_pos" in qual_df.columns:
        qm = accuracy_metrics(
            qual_df.set_index("driver")["predicted_grid_pos"],
            qual_df.set_index("driver")["actual_grid_pos"],
        )
        show_metric_badges(qm)

    col1, col2 = st.columns([1, 1])
    with col1:
        display_cols = ["predicted_grid_pos", "driver"]
        if "team" in qual_df.columns:
            display_cols.append("team")
        if "actual_grid_pos" in qual_df.columns:
            display_cols.append("actual_grid_pos")
        if "predicted_grid_delta" in qual_df.columns:
            display_cols.append("predicted_grid_delta")

        show_df = qual_df[display_cols].copy()
        show_df.columns = (
            ["P", "Driver"]
            + (["Team"] if "team" in qual_df.columns else [])
            + (["Actual P"] if "actual_grid_pos" in qual_df.columns else [])
            + (["Δ to best (s)"] if "predicted_grid_delta" in qual_df.columns else [])
        )
        if "Δ to best (s)" in show_df.columns:
            show_df["Δ to best (s)"] = show_df["Δ to best (s)"].round(3)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    with col2:
        if "predicted_grid_delta" in qual_df.columns:
            fig = grid_bar_chart(qual_df, "predicted_grid_pos", "Predicted Qualifying Positions")
            st.plotly_chart(fig, use_container_width=True)

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
# Tab 2 – Race Prediction
# ===========================================================================
with tab_race:
    st.subheader(f"Predicted Race Finishing Order — {race_label}")
    race_df = result["race"]

    # Accuracy badges (only when actual finishing positions are available)
    if "actual_finish_pos" in race_df.columns:
        rm = accuracy_metrics(
            race_df.set_index("driver")["predicted_finish_pos"],
            race_df.set_index("driver")["actual_finish_pos"],
        )
        show_metric_badges(rm)

    col1, col2 = st.columns([1, 1])
    with col1:
        rcols = ["predicted_finish_pos", "driver"]
        if "team" in race_df.columns:
            rcols.append("team")
        if "grid_used" in race_df.columns:
            rcols.append("grid_used")
        if "position_change" in race_df.columns:
            rcols.append("position_change")
        if "actual_finish_pos" in race_df.columns:
            rcols.append("actual_finish_pos")

        show_race = race_df[rcols].copy()
        col_labels = (
            ["Pos", "Driver"]
            + (["Team"] if "team" in race_df.columns else [])
            + (["Grid"] if "grid_used" in race_df.columns else [])
            + (["Δ Pos"] if "position_change" in race_df.columns else [])
            + (["Actual"] if "actual_finish_pos" in race_df.columns else [])
        )
        show_race.columns = col_labels
        if "Δ Pos" in show_race.columns:
            show_race["Δ Pos"] = show_race["Δ Pos"].apply(
                lambda v: f"+{int(v)}" if float(v) > 0 else str(int(v))
            )
        st.dataframe(show_race, use_container_width=True, hide_index=True)

    with col2:
        fig2 = grid_bar_chart(race_df, "predicted_finish_pos", "Predicted Finishing Positions")
        st.plotly_chart(fig2, use_container_width=True)

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
# Tab 3 – Accuracy
# ===========================================================================
with tab_accuracy:
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
                rm = accuracy_metrics(
                    result["race"].set_index("driver")["predicted_finish_pos"],
                    result["race"].set_index("driver")["actual_finish_pos"],
                )
                show_metric_badges(rm)
                fig_rs = accuracy_scatter(
                    "predicted_finish_pos", "actual_finish_pos",
                    result["race"].set_index("driver").reset_index(),
                    xlabel="Actual Finishing Position",
                    ylabel="Predicted Finishing Position",
                    title="Race: Predicted vs Actual",
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
            re = result["race"][["driver", "predicted_finish_pos", "actual_finish_pos"]].copy()
            re["race_error"] = (re["predicted_finish_pos"] - re["actual_finish_pos"]).round(1)
            err_frames.append(re.set_index("driver")[["predicted_finish_pos", "actual_finish_pos", "race_error"]])

        if err_frames:
            err_df = err_frames[0].join(err_frames[1], how="outer") if len(err_frames) == 2 else err_frames[0]
            err_df = err_df.reset_index()
            err_df.columns = [c.replace("_", " ").title() for c in err_df.columns]
            st.dataframe(err_df, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 4 – What-If
# ===========================================================================
with tab_whatif:
    st.subheader("What-If Scenario Controls")
    st.markdown("Override inputs to the race model and see how predictions change.")

    drivers = result["qualifying"]["driver"].tolist()

    with st.form("whatif_form"):
        col_g, col_p = st.columns(2)

        with col_g:
            st.markdown("**Grid Position Overrides**")
            grid_overrides = {}
            driver_a = st.selectbox("Driver A", drivers, key="da")
            new_pos_a = st.number_input(f"{driver_a} new grid pos", min_value=1, max_value=20,
                                         value=int(result["qualifying"].set_index("driver").loc[driver_a, "predicted_grid_pos"])
                                         if driver_a in result["qualifying"]["driver"].values else 1,
                                         key="pa")
            driver_b = st.selectbox("Driver B", drivers, index=1, key="db")
            new_pos_b = st.number_input(f"{driver_b} new grid pos", min_value=1, max_value=20,
                                         value=int(result["qualifying"].set_index("driver").loc[driver_b, "predicted_grid_pos"])
                                         if driver_b in result["qualifying"]["driver"].values else 2,
                                         key="pb")

        with col_p:
            st.markdown("**Pit Stop Overrides**")
            pit_overrides = {}
            driver_pit = st.selectbox("Driver", drivers, key="dp")
            new_pits = st.slider(f"{driver_pit} pit stops", min_value=1, max_value=4, value=2)

        submitted = st.form_submit_button("Apply What-If", type="primary")

    if submitted:
        overrides = {
            "grid_position": {driver_a: new_pos_a, driver_b: new_pos_b},
            "pit_stops": {driver_pit: new_pits},
        }
        with st.spinner("Re-running race model with overrides..."):
            try:
                wi_result = run_whatif_prediction(year, round_number, overrides, use_actual_grid)
                wi_race = wi_result["race"]

                st.markdown("#### What-If Finishing Order")
                baseline_race = result["race"].set_index("driver")["predicted_finish_pos"]
                wi_compare = wi_race[["predicted_finish_pos", "driver"]].copy()
                wi_compare.columns = ["New Pos", "Driver"]
                wi_compare["Baseline Pos"] = wi_compare["Driver"].map(baseline_race)
                wi_compare["Change"] = (wi_compare["Baseline Pos"] - wi_compare["New Pos"]).apply(
                    lambda v: f"+{int(v)}" if v > 0 else str(int(v))
                )
                st.dataframe(wi_compare[["New Pos", "Driver", "Baseline Pos", "Change"]],
                             use_container_width=True, hide_index=True)

                # Side-by-side bar comparison
                combined = pd.merge(
                    result["race"][["driver", "predicted_finish_pos"]].rename(
                        columns={"predicted_finish_pos": "Baseline"}),
                    wi_race[["driver", "predicted_finish_pos"]].rename(
                        columns={"predicted_finish_pos": "What-If"}),
                    on="driver",
                )
                fig_wi = go.Figure()
                fig_wi.add_trace(go.Bar(name="Baseline", x=combined["driver"],
                                        y=combined["Baseline"], marker_color="#888"))
                fig_wi.add_trace(go.Bar(name="What-If", x=combined["driver"],
                                        y=combined["What-If"], marker_color="#E8002D"))
                fig_wi.update_layout(
                    barmode="group", title="Baseline vs What-If Finishing Positions",
                    yaxis=dict(autorange="reversed", title="Position"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white", height=380,
                )
                st.plotly_chart(fig_wi, use_container_width=True)

            except Exception as exc:
                st.error(f"What-if prediction failed: {exc}")
    else:
        st.info("Set overrides above and click **Apply What-If** to see results.")
