"""
pipeline.py
-----------
End-to-end inference pipeline.

    from src.pipeline import predict_race

    result = predict_race(2024, 6)          # Monaco 2024
    print(result["qualifying"])             # predicted grid order
    print(result["race"])                   # predicted finishing order

What-if overrides
-----------------
Pass an `overrides` dict to adjust Stage 2 inputs before re-running the
race model.  Recognised keys:

    overrides = {
        "grid_position": {"VER": 3, "HAM": 1},   # override grid pos per driver
        "pit_stops":     {"VER": 2},              # override pit stop count
        "tire_compound": {"HAM": "HARD"},         # stored as a label column
    }
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import setup_cache
from src.feature_engineering import build_qualifying_features, build_race_features
from src import qualifying_model as qm
from src import race_model as rm

logger = logging.getLogger(__name__)

# Default model paths
QM_MODEL = "models/qualifying_model.json"
QM_COLS = "models/qualifying_feature_cols.pkl"
RM_MODEL = "models/race_model.pkl"
RM_COLS = "models/race_feature_cols.pkl"

# ---------------------------------------------------------------------------
# Monte Carlo noise profiles
# ---------------------------------------------------------------------------
# Gaussian σ applied to measurement features only.
# Form / standings / flags are exact known values — no noise applied.
_QM_NOISE: dict = {
    "fp3_lap_delta_s": 0.05, "fp3_s1_delta_s": 0.02, "fp3_s2_delta_s": 0.02,
    "fp3_s3_delta_s": 0.02,  "fp3_speed_max": 2.0,   "fp3_teammate_delta_s": 0.03,
    "fp2_lap_delta_s": 0.05, "fp2_s1_delta_s": 0.02, "fp2_s2_delta_s": 0.02,
    "fp2_s3_delta_s": 0.02,  "fp2_speed_max": 2.0,
    "fp1_lap_delta_s": 0.08, "fp1_s1_delta_s": 0.03, "fp1_s2_delta_s": 0.03,
    "fp1_s3_delta_s": 0.03,
}

_RM_NOISE: dict = {
    "grid_used": 0.8,
    "q_best_lap_s": 0.05,
    "fp3_race_lap_delta_s": 0.05, "fp3_race_s1_delta_s": 0.02, "fp3_race_s2_delta_s": 0.02,
    "fp3_race_s3_delta_s": 0.02,  "fp3_race_speed_max": 2.0,   "fp3_race_teammate_delta_s": 0.03,
    "fp2_race_lap_delta_s": 0.05, "fp2_race_s2_delta_s": 0.02, "fp2_race_teammate_delta_s": 0.03,
    "pit_stops": 0.3,
}


def _mc_simulate(model, feature_cols: list, X_base: pd.DataFrame,
                 n_samples: int, noise: dict, seed: int = 42) -> np.ndarray:
    """
    Run n_samples noisy inferences using Gaussian feature perturbation.

    Returns score matrix of shape (n_samples, n_drivers).
    Lower score = higher predicted rank (consistent with both models).
    """
    rng = np.random.default_rng(seed)
    base = X_base.to_numpy(dtype=np.float64)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    noise_cols = [(col_idx[c], sigma) for c, sigma in noise.items()
                  if c in col_idx and sigma > 0]

    scores = np.zeros((n_samples, len(X_base)))
    for i in range(n_samples):
        X_noisy = base.copy()
        for ci, sigma in noise_cols:
            X_noisy[:, ci] += rng.normal(0, sigma, size=len(X_base))
        scores[i] = model.predict(pd.DataFrame(X_noisy, columns=feature_cols))
    return scores


def _scores_to_position_probs(scores_matrix: np.ndarray, drivers: list,
                               smooth_alpha: float = 0.08) -> pd.DataFrame:
    """
    Convert raw score matrix (n_samples × n_drivers) to a position probability DataFrame.
    Index = driver names, columns = integer positions 1..N.

    smooth_alpha blends the empirical probabilities with a uniform distribution, ensuring
    no driver ever has 0% or 100% chance at any position (reflecting real F1 chaos).
    0.08 means 8% of the weight goes to uniform, so max any cell can reach is ~98%.
    """
    n_samples, n_drivers = scores_matrix.shape
    counts = np.zeros((n_drivers, n_drivers), dtype=np.float64)
    for i in range(n_samples):
        order = np.argsort(scores_matrix[i])  # ascending: index 0 = lowest score = P1
        for pos_idx, driver_idx in enumerate(order):
            counts[driver_idx, pos_idx] += 1
    raw_probs = counts / n_samples
    uniform = np.full_like(raw_probs, 1.0 / n_drivers)
    smoothed = (1.0 - smooth_alpha) * raw_probs + smooth_alpha * uniform
    return pd.DataFrame(smoothed, index=drivers, columns=range(1, n_drivers + 1))


def predict_race(
    year: int,
    round_number: int,
    overrides: Optional[dict] = None,
    cache_dir: str = "cache",
    use_actual_grid: bool = False,
    mc_samples: int = 1,
) -> dict:
    """
    Run the full two-stage prediction for a given race.

    Parameters
    ----------
    year, round_number : race to predict
    overrides : optional what-if dict (see module docstring)
    cache_dir : FastF1 cache directory
    use_actual_grid : if True, skip Stage 1 and use actual qualifying results
    mc_samples : number of Monte Carlo samples (1 = deterministic). When >1,
                 Gaussian noise is added to measurement features each run and
                 predictions are averaged, also producing position probability
                 distributions.

    Returns
    -------
    dict with keys:
        "qualifying"    pd.DataFrame  predicted grid order
        "race"          pd.DataFrame  predicted finishing order
        "quali_raw"     pd.DataFrame  qualifying features used
        "race_raw"      pd.DataFrame  race features used (after overrides)
        "quali_probs"   pd.DataFrame | None  position probabilities (MC only)
        "race_probs"    pd.DataFrame | None  position probabilities (MC only)
    """
    setup_cache(cache_dir)
    overrides = overrides or {}

    # ------------------------------------------------------------------
    # Stage 1: Qualifying prediction
    # ------------------------------------------------------------------
    logger.info("Stage 1 – building qualifying features year=%d round=%d", year, round_number)
    quali_feat = build_qualifying_features(year, round_number)

    if quali_feat.empty:
        raise ValueError(f"No qualifying features available for year={year} round={round_number}")

    quali_probs = None

    if use_actual_grid:
        quali_feat["predicted_grid_pos"] = quali_feat["grid_position"]
        quali_feat["predicted_grid_delta"] = 0.0
        _actual_cols = ["driver", "team", "grid_position", "predicted_grid_pos", "predicted_grid_delta"]
        if "q_best_lap_s" in quali_feat.columns:
            _actual_cols.append("q_best_lap_s")
        quali_result = quali_feat[_actual_cols].copy()
    elif mc_samples > 1:
        import joblib, xgboost as xgb
        _qm_model = xgb.XGBRegressor()
        _qm_model.load_model(QM_MODEL)
        _qm_cols = joblib.load(QM_COLS)
        X_qm = pd.DataFrame(
            quali_feat.reindex(columns=_qm_cols, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64),
            columns=_qm_cols,
        )
        qm_scores = _mc_simulate(_qm_model, _qm_cols, X_qm, mc_samples, _QM_NOISE)
        mean_scores = qm_scores.mean(axis=0)
        quali_result = quali_feat.copy()
        quali_result["predicted_grid_delta"] = mean_scores
        quali_result["predicted_grid_pos"] = (
            pd.Series(mean_scores).rank(method="first").astype(int).values
        )
        quali_probs = _scores_to_position_probs(qm_scores, quali_feat["driver"].tolist())
    else:
        quali_result = qm.predict(quali_feat, model_path=QM_MODEL, feature_cols_path=QM_COLS)

    quali_display = (
        quali_result[["driver", "team", "predicted_grid_pos", "predicted_grid_delta"]]
        .sort_values("predicted_grid_pos")
        .reset_index(drop=True)
    )
    if use_actual_grid and "q_best_lap_s" in quali_result.columns:
        q_lap_map = quali_result.set_index("driver")["q_best_lap_s"]
        quali_display["q_best_lap_s"] = quali_display["driver"].map(q_lap_map)
    # Theoretical best lap (S1+S2+S3 personal bests across practice) for display when no actual quali
    if "fp_theoretical_best_s" in quali_feat.columns:
        theo_map = quali_feat.set_index("driver")["fp_theoretical_best_s"]
        quali_display["fp_theoretical_best_s"] = quali_display["driver"].map(theo_map)
    if "fp_theoretical_gap_s" in quali_feat.columns:
        theo_gap_map = quali_feat.set_index("driver")["fp_theoretical_gap_s"]
        quali_display["fp_theoretical_gap_s"] = quali_display["driver"].map(theo_gap_map)
    if "grid_position" in quali_result.columns:
        actual_grid_map = quali_result.set_index("driver")["grid_position"]
        quali_display["actual_grid_pos"] = quali_display["driver"].map(actual_grid_map)

    # ------------------------------------------------------------------
    # Stage 2: Race prediction
    # ------------------------------------------------------------------
    logger.info("Stage 2 – building race features year=%d round=%d", year, round_number)
    predicted_grid = quali_result[["driver", "predicted_grid_pos"]].copy()
    race_feat = build_race_features(year, round_number, predicted_grid=predicted_grid)

    if race_feat.empty:
        raise ValueError(f"No race features available for year={year} round={round_number}")

    # ---- Apply what-if overrides ----
    if overrides.get("grid_position"):
        for driver, new_pos in overrides["grid_position"].items():
            mask = race_feat["driver"] == driver
            if mask.any():
                race_feat.loc[mask, "grid_used"] = float(new_pos)

    if overrides.get("pit_stops"):
        for driver, n_stops in overrides["pit_stops"].items():
            mask = race_feat["driver"] == driver
            if mask.any():
                race_feat.loc[mask, "pit_stops"] = float(n_stops)

    # tire_compound stored as metadata but not a direct numeric feature (label encoded if needed)
    if overrides.get("tire_compound"):
        for driver, compound in overrides["tire_compound"].items():
            mask = race_feat["driver"] == driver
            if mask.any():
                race_feat.loc[mask, "tire_compound_override"] = compound

    race_probs = None

    if mc_samples > 1:
        import joblib
        _rm_model = joblib.load(RM_MODEL)
        _rm_cols = joblib.load(RM_COLS)
        X_rm = pd.DataFrame(
            race_feat.reindex(columns=_rm_cols, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64),
            columns=_rm_cols,
        )
        rm_scores = _mc_simulate(_rm_model, _rm_cols, X_rm, mc_samples, _RM_NOISE)
        mean_scores = rm_scores.mean(axis=0)
        race_result = race_feat.copy()
        race_result["predicted_finish_score"] = mean_scores
        race_result["predicted_finish_pos"] = (
            pd.Series(mean_scores).rank(method="first").astype(int).values
        )
        race_result = race_result.sort_values("predicted_finish_pos").reset_index(drop=True)
        race_probs = _scores_to_position_probs(rm_scores, race_feat["driver"].tolist())
    else:
        race_result = rm.predict(race_feat, model_path=RM_MODEL, feature_cols_path=RM_COLS)

    race_display = race_result[["driver", "team", "predicted_finish_pos", "predicted_finish_score"]].copy()
    if "finish_position" in race_result.columns:
        actual_finish_map = race_result.set_index("driver")["finish_position"]
        actual_grid_map2 = race_result.set_index("driver")["grid_used"]
        race_display["actual_finish_pos"] = race_display["driver"].map(actual_finish_map)
        race_display["grid_used"] = race_display["driver"].map(actual_grid_map2)
        race_display["position_change"] = (
            race_display["grid_used"].astype(float) - race_display["predicted_finish_pos"].astype(float)
        ).round(1)
    if "dnf" in race_result.columns:
        dnf_map = race_result.set_index("driver")["dnf"]
        race_display["dnf"] = race_display["driver"].map(dnf_map).fillna(0).astype(int)
    if "retire_label" in race_result.columns:
        label_map = race_result.set_index("driver")["retire_label"]
        race_display["retire_label"] = race_display["driver"].map(label_map).fillna("")

    return {
        "qualifying": quali_display,
        "race": race_display.sort_values("predicted_finish_pos").reset_index(drop=True),
        "quali_raw": quali_feat,
        "race_raw": race_feat,
        "quali_probs": quali_probs,
        "race_probs": race_probs,
    }


def get_shap_values(stage: str, features_df: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for a feature DataFrame.

    Parameters
    ----------
    stage : "qualifying" or "race"
    features_df : the *_raw DataFrame from predict_race output

    Returns
    -------
    (shap_values, expected_value, feature_names)
    """
    import shap
    import joblib
    import xgboost as xgb

    if stage == "qualifying":
        feature_cols = joblib.load(QM_COLS)
        model = xgb.XGBRegressor()
        model.load_model(QM_MODEL)
        X = features_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
        explainer = shap.TreeExplainer(model)
    else:
        feature_cols = joblib.load(RM_COLS)
        model = joblib.load(RM_MODEL)
        X = features_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value, list(X.columns)
