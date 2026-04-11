"""
generate_predictions.py
-----------------------
Pre-compute pipeline predictions for one or more race rounds and save them
as JSON files under data/predictions/.

Files are named:  data/predictions/{year}_{round:02d}.json

Usage:
    # Single round
    python scripts/generate_predictions.py --year 2026 --round 4

    # All completed rounds in a season
    python scripts/generate_predictions.py --year 2026

    # Multiple specific rounds
    python scripts/generate_predictions.py --year 2025 --round 3 4 5

Each JSON file contains everything the Streamlit app needs to display:
    qualifying      list of dicts  — predicted + actual qualifying order
    race            list of dicts  — predicted + actual race order
    race_probs      dict           — {driver: {1: p, 2: p, ...}} position probabilities
    quali_raw       list of dicts  — raw qualifying features (for SHAP)
    race_raw        list of dicts  — raw race features (for SHAP)
    meta            dict           — year, round, generated_at, mc_samples
"""

import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import setup_cache
from src.pipeline import predict_race
from src.feature_engineering import ROUNDS_PER_YEAR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MC_SAMPLES = 100
OUT_DIR = Path("data/predictions")


def _is_completed(year: int, rnd: int) -> bool:
    """Return True if the race date is in the past."""
    try:
        import fastf1 as _ff1
        evt = _ff1.get_event(year, rnd)
        race_date = evt.get_session_date("Race", utc=True)
        if race_date.tzinfo is None:
            race_date = race_date.replace(tzinfo=timezone.utc)
        return race_date <= datetime.now(timezone.utc)
    except Exception:
        return False


def _df_to_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts, handling NaN/numpy types."""
    return json.loads(df.to_json(orient="records", default_handler=str))


def _probs_to_dict(probs_df) -> dict:
    """Convert position-probability DataFrame (index=driver, cols=1..N) to plain dict."""
    if probs_df is None:
        return {}
    # probs_df.columns are ints; JSON keys must be strings
    return {
        driver: {str(pos): float(prob) for pos, prob in row.items()}
        for driver, row in probs_df.iterrows()
    }


def generate(year: int, rnd: int, out_dir: Path, mc_samples: int) -> Path:
    logger.info("Generating predictions for %d R%02d ...", year, rnd)
    result = predict_race(year, rnd, use_actual_grid=False, mc_samples=mc_samples)

    payload = {
        "meta": {
            "year": year,
            "round": rnd,
            "mc_samples": mc_samples,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "qualifying": _df_to_json(result["qualifying"]),
        "race": _df_to_json(result["race"]),
        "race_probs": _probs_to_dict(result.get("race_probs")),
        "quali_probs": _probs_to_dict(result.get("quali_probs")),
        "quali_raw": _df_to_json(result["quali_raw"]),
        "race_raw": _df_to_json(result["race_raw"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}_{rnd:02d}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)

    logger.info("Saved %s", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="rounds", nargs="*", type=int, default=None,
                        help="Round number(s). Omit to process all completed rounds in the season.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--mc-samples", type=int, default=MC_SAMPLES)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if the file already exists.")
    args = parser.parse_args()

    setup_cache(args.cache_dir)
    out_dir = Path(args.out_dir)

    if args.rounds:
        rounds = args.rounds
    else:
        total = ROUNDS_PER_YEAR.get(args.year, 24)
        rounds = [r for r in range(3, total + 1) if _is_completed(args.year, r)]
        logger.info("Found %d completed rounds for %d", len(rounds), args.year)

    for rnd in rounds:
        out_path = out_dir / f"{args.year}_{rnd:02d}.json"
        if out_path.exists() and not args.force:
            logger.info("Skipping %d R%02d — file already exists (use --force to overwrite)", args.year, rnd)
            continue
        try:
            generate(args.year, rnd, out_dir, args.mc_samples)
        except Exception as exc:
            logger.error("Failed %d R%02d: %s", args.year, rnd, exc)
