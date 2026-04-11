"""
build_dataset.py
----------------
Runs feature engineering across 2022-2024 and saves the training CSVs.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --years 2024 --rounds 5 6 7
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import setup_cache
from src.feature_engineering import build_and_save_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--rounds", nargs="+", type=int, default=None,
                        help="Specific round numbers (applied to all years). Useful for testing.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    setup_cache(args.cache_dir)

    rounds_override = None
    if args.rounds:
        rounds_override = {y: args.rounds for y in args.years}

    build_and_save_dataset(args.years, out_dir=args.out_dir, rounds_override=rounds_override)
