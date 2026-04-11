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


def predict_race(
    year: int,
    round_number: int,
    overrides: Optional[dict] = None,
    cache_dir: str = "cache",
    use_actual_grid: bool = False,
) -> dict:
    """
    Run the full two-stage prediction for a given race.

    Parameters
    ----------
    year, round_number : race to predict
    overrides : optional what-if dict (see module docstring)
    cache_dir : FastF1 cache directory
    use_actual_grid : if True, skip Stage 1 and use actual qualifying results
                      (useful for post-race evaluation)

    Returns
    -------
    dict with keys:
        "qualifying"    pd.DataFrame  predicted grid order
        "race"          pd.DataFrame  predicted finishing order
        "quali_raw"     pd.DataFrame  qualifying features used
        "race_raw"      pd.DataFrame  race features used (after overrides)
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

    if use_actual_grid:
        quali_feat["predicted_grid_pos"] = quali_feat["grid_position"]
        quali_feat["predicted_grid_delta"] = 0.0
        quali_result = quali_feat[["driver", "team", "grid_position", "predicted_grid_pos",
                                   "predicted_grid_delta"]].copy()
    else:
        quali_result = qm.predict(quali_feat, model_path=QM_MODEL, feature_cols_path=QM_COLS)

    quali_display = (
        quali_result[["driver", "team", "predicted_grid_pos", "predicted_grid_delta"]]
        .sort_values("predicted_grid_pos")
        .reset_index(drop=True)
    )
    if "grid_position" in quali_result.columns:
        quali_display["actual_grid_pos"] = quali_result["grid_position"].values

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

    race_result = rm.predict(race_feat, model_path=RM_MODEL, feature_cols_path=RM_COLS)

    race_display = race_result[["driver", "team", "predicted_finish_pos", "predicted_finish_score"]].copy()
    if "finish_position" in race_result.columns:
        race_display["actual_finish_pos"] = race_result["finish_position"].values
        race_display["grid_used"] = race_result["grid_used"].values
        race_display["position_change"] = (
            race_display["grid_used"].astype(float) - race_display["predicted_finish_pos"].astype(float)
        ).round(1)

    return {
        "qualifying": quali_display,
        "race": race_display.sort_values("predicted_finish_pos").reset_index(drop=True),
        "quali_raw": quali_feat,
        "race_raw": race_feat,
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
