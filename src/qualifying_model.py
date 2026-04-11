"""
qualifying_model.py
-------------------
Stage 1: Predicts qualifying grid positions from practice session features.

Strategy
--------
- Target:  grid_position (1–20)
- Model:   XGBoost regressor trained to predict best-lap delta from session best.
           Final grid prediction = rank(predicted_delta).
- Eval:    Spearman rank correlation, position MAE
- Artefact: models/qualifying_model.json

CLI
---
    python -m src.qualifying_model train
    python -m src.qualifying_model evaluate
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used for training
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Projected qualifying time (best sectors across all practice sessions)
    "fp_theoretical_gap_s",
    # FP3 (most important – closest to qualifying)
    "fp3_lap_delta_s",
    "fp3_s1_delta_s",
    "fp3_s2_delta_s",
    "fp3_s3_delta_s",
    "fp3_speed_max",
    "fp3_num_laps",
    "fp3_teammate_delta_s",
    # FP2
    "fp2_lap_delta_s",
    "fp2_s1_delta_s",
    "fp2_s2_delta_s",
    "fp2_s3_delta_s",
    "fp2_speed_max",
    # FP1
    "fp1_lap_delta_s",
    "fp1_s1_delta_s",
    "fp1_s2_delta_s",
    "fp1_s3_delta_s",
    # Form
    "driver_points",
    "driver_championship_pos",
    "constructor_points",
    "constructor_championship_pos",
    "avg_finish_last_n",
    "dnf_rate_last_n",
    # Circuit context
    "is_street_circuit",
]

TARGET_COL = "grid_position"


def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only rows that have at least fp3 data
    df = df.dropna(subset=["fp3_lap_delta_s", TARGET_COL])
    return df


def _prepare_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    # Impute remaining NaN with column median
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COL].values.astype(float)
    return X, y


def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def train(
    csv_path: str = "data/processed/qualifying_features.csv",
    model_out: str = "models/qualifying_model.json",
    feature_cols_out: str = "models/qualifying_feature_cols.pkl",
) -> xgb.XGBRegressor:
    """Train the qualifying model and save artefacts."""
    logger.info("Loading data from %s", csv_path)
    df = _load_data(csv_path)
    logger.info("Training samples: %d", len(df))

    X, y = _prepare_xy(df)
    feature_cols_used = list(X.columns)

    # Group split by (year, round) so whole races go into test
    groups = df["year"].astype(str) + "_" + df["round"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    _evaluate(model, X_val, y_val, label="Validation")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_out)
    joblib.dump(feature_cols_used, feature_cols_out)
    logger.info("Model saved to %s", model_out)
    logger.info("Feature columns saved to %s", feature_cols_out)
    return model


def _evaluate(model, X, y_true, label: str = "Test") -> dict:
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = spearmanr(y_true, y_pred)
    logger.info("[%s] Qualifying MAE=%.3f  Spearman=%.3f", label, mae, corr)
    print(f"[{label}] Qualifying position MAE: {mae:.3f}  |  Spearman rank corr: {corr:.3f}")
    return {"mae": mae, "spearman": corr}


def evaluate(
    csv_path: str = "data/processed/qualifying_features.csv",
    model_path: str = "models/qualifying_model.json",
    feature_cols_path: str = "models/qualifying_feature_cols.pkl",
) -> dict:
    """Load saved model and run evaluation on the full dataset."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    feature_cols = joblib.load(feature_cols_path)

    df = _load_data(csv_path)
    X, y = _prepare_xy(df)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return _evaluate(model, X, y, label="Full dataset")


def predict(
    features_df: pd.DataFrame,
    model_path: str = "models/qualifying_model.json",
    feature_cols_path: str = "models/qualifying_feature_cols.pkl",
) -> pd.DataFrame:
    """
    Run inference on a single race's feature DataFrame.

    Returns the input DataFrame with two new columns:
      - predicted_grid_delta  : raw model output
      - predicted_grid_pos    : rank of predicted_grid_delta (1 = pole)
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    feature_cols = joblib.load(feature_cols_path)

    # Rebuild as an explicit float64 DataFrame with exact column names.
    # This is required because XGBoost loaded via load_model() does not fully
    # restore sklearn's feature_names_in_ attribute, causing it to reject
    # inputs that aren't a cleanly-named DataFrame.
    X_raw = features_df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
    X = pd.DataFrame(X_raw.to_numpy(dtype=np.float64), columns=feature_cols)
    out = features_df.copy()
    out["predicted_grid_delta"] = model.predict(X)
    out["predicted_grid_pos"] = out["predicted_grid_delta"].rank(method="first").astype(int)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    else:
        print(f"Unknown command: {cmd}. Use 'train' or 'evaluate'.")
