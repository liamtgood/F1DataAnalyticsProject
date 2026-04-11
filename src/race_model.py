"""
race_model.py
-------------
Stage 2: Predicts race finishing positions.

Strategy
--------
- Target:  finish_position (1–20)
- Model:   LightGBM regressor
- Input:   qualifying stage output (grid_used) + form + circuit features
- Eval:    finishing position MAE, top-5 hit rate
- Artefact: models/race_model.pkl, models/race_feature_cols.pkl

CLI
---
    python -m src.race_model train
    python -m src.race_model evaluate
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Grid position — used as offset to reconstruct finish pos, still a signal
    "grid_used",
    # Qualifying pace proxy
    "q_best_lap_s",
    # FP3 pace (race-trim proxy — closest session to race setup)
    "fp3_race_lap_delta_s",
    "fp3_race_s1_delta_s",
    "fp3_race_s2_delta_s",
    "fp3_race_s3_delta_s",
    "fp3_race_speed_max",
    "fp3_race_teammate_delta_s",
    # FP2 long-run pace
    "fp2_race_lap_delta_s",
    "fp2_race_s2_delta_s",
    "fp2_race_teammate_delta_s",
    # Stint strategy
    "pit_stops",
    # Driver form
    "driver_points",
    "driver_championship_pos",
    "avg_finish_last_n",
    "dnf_rate_last_n",
    # Team / Constructor form
    "constructor_points",
    "constructor_championship_pos",
    # Circuit context
    "is_street_circuit",
    "overtake_index",
]

TARGET_COL = "finish_position"


def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COL, "grid_used"])
    return df


def _prepare_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COL].values.astype(float)
    return X, y


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _top5_hit_rate(y_true, y_pred) -> float:
    """Fraction of actual top-5 finishers correctly predicted in top 5."""
    true_top5 = set(np.where(y_true <= 5)[0])
    pred_top5 = set(np.argsort(y_pred)[:5])
    if not true_top5:
        return float("nan")
    return len(true_top5 & pred_top5) / len(true_top5)


def _evaluate(model, X, y_true, label: str = "Test") -> dict:
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = spearmanr(y_true, y_pred)
    top5 = _top5_hit_rate(y_true, y_pred)
    logger.info("[%s] Race MAE=%.3f  Spearman=%.3f  Top5-hit=%.2f", label, mae, corr, top5)
    print(f"[{label}] Race finishing pos MAE: {mae:.3f}  |  Spearman: {corr:.3f}  |  Top-5 hit rate: {top5:.2%}")
    return {"mae": mae, "spearman": corr, "top5_hit_rate": top5}


def train(
    csv_path: str = "data/processed/race_features.csv",
    model_out: str = "models/race_model.pkl",
    feature_cols_out: str = "models/race_feature_cols.pkl",
) -> lgb.LGBMRegressor:
    logger.info("Loading data from %s", csv_path)
    df = _load_data(csv_path)
    logger.info("Training samples: %d", len(df))

    X, y = _prepare_xy(df)
    feature_cols_used = list(X.columns)

    groups = df["year"].astype(str) + "_" + df["round"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    _evaluate(model, X_val, y_val, label="Validation")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(feature_cols_used, feature_cols_out)
    logger.info("Model saved to %s", model_out)
    return model


def evaluate(
    csv_path: str = "data/processed/race_features.csv",
    model_path: str = "models/race_model.pkl",
    feature_cols_path: str = "models/race_feature_cols.pkl",
) -> dict:
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)

    df = _load_data(csv_path)
    X, y = _prepare_xy(df)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return _evaluate(model, X, y, label="Full dataset")


def predict(
    features_df: pd.DataFrame,
    model_path: str = "models/race_model.pkl",
    feature_cols_path: str = "models/race_feature_cols.pkl",
) -> pd.DataFrame:
    """
    Run inference on a single race's feature DataFrame.

    Returns the input DataFrame with columns:
      - predicted_finish_score : raw model output (lower = better predicted position)
      - predicted_finish_pos   : rank of score (1 = predicted winner)
    """
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)

    X_raw = features_df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
    X = pd.DataFrame(X_raw.to_numpy(dtype=np.float64), columns=feature_cols)
    out = features_df.copy()
    out["predicted_finish_score"] = model.predict(X)
    out["predicted_finish_pos"] = out["predicted_finish_score"].rank(method="first").astype(int)
    return out.sort_values("predicted_finish_pos").reset_index(drop=True)


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
