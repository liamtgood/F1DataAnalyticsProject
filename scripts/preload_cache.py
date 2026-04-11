"""
preload_cache.py
----------------
Pre-downloads all FastF1 session data for 2022-2024 to the local cache.

Run this BEFORE the hackathon starts (~2 hours unattended).

Usage:
    python scripts/preload_cache.py
    python scripts/preload_cache.py --years 2024     # single year
    python scripts/preload_cache.py --years 2023 2024 --sessions FP3 Q R
"""

import sys
import argparse
import logging
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1
from src.data_loader import setup_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FastF1 API cap: ~500 calls/hour. Each session.load() makes ~5-8 internal
# requests, so we need at least 8 seconds between sessions to stay under it.
INTER_SESSION_SLEEP = 10   # seconds between sessions (normal)
RATE_LIMIT_SLEEP = 75      # seconds to pause when a rate-limit response is detected
RATE_LIMIT_RETRIES = 3     # max retries per session on rate-limit errors

ROUNDS_PER_YEAR = {2022: 22, 2023: 22, 2024: 24}
DEFAULT_SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]


def _load_with_retry(year: int, rnd: int, sess_id: str) -> bool:
    """Load one session, retrying on rate-limit errors. Returns True on success."""
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            session = fastf1.get_session(year, rnd, sess_id)
            session.load(telemetry=False, weather=False, messages=False)
            return True
        except Exception as exc:
            msg = str(exc)
            if "500 calls/h" in msg or "rate" in msg.lower():
                logger.warning(
                    "  Rate limited (attempt %d/%d). Pausing %ds before retry...",
                    attempt, RATE_LIMIT_RETRIES, RATE_LIMIT_SLEEP,
                )
                time.sleep(RATE_LIMIT_SLEEP)
            else:
                logger.warning("  FAILED: %s", exc)
                return False
    logger.warning("  FAILED after %d retries (persistent rate limit).", RATE_LIMIT_RETRIES)
    return False


def preload(years: list, sessions: list, cache_dir: str = "cache") -> None:
    setup_cache(cache_dir)

    total = sum(ROUNDS_PER_YEAR.get(y, 22) for y in years) * len(sessions)
    done = 0
    failed = []

    for year in years:
        max_round = ROUNDS_PER_YEAR.get(year, 22)
        for rnd in range(1, max_round + 1):
            for sess_id in sessions:
                done += 1
                logger.info("[%d/%d] Downloading year=%d round=%d session=%s ...",
                            done, total, year, rnd, sess_id)

                if _load_with_retry(year, rnd, sess_id):
                    logger.info("  OK")
                else:
                    failed.append((year, rnd, sess_id))

                # Polite inter-session pause to stay under the 500 calls/hr cap
                time.sleep(INTER_SESSION_SLEEP)

    logger.info("=" * 60)
    logger.info("Pre-load complete. %d/%d sessions downloaded.", done - len(failed), total)
    if failed:
        logger.warning("%d sessions failed:", len(failed))
        for item in failed:
            logger.warning("  year=%d round=%d session=%s", *item)
        logger.info("Re-run the script to retry -- already-cached sessions are skipped quickly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download FastF1 cache")
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS,
                        help="Sessions to download e.g. FP3 Q R")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    preload(args.years, args.sessions, args.cache_dir)
