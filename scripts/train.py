#!/usr/bin/env python3
"""
Offline training script — trains models for all (or selected) states
and saves the best model per state to disk.

Usage
-----
  python scripts/train.py                        # all states
  python scripts/train.py --states California Texas
  python scripts/train.py --states California --val-weeks 8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_pipeline import get_states, load_and_prepare
from app.model_selector import ModelSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train forecasting models offline")
    p.add_argument("--states", nargs="+", default=None, help="State(s) to train")
    p.add_argument("--val-weeks", type=int, default=12, help="Validation window size")
    p.add_argument("--output", default=None, help="Path to save JSON report")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading data …")
    df = load_and_prepare()
    all_states = get_states(df)

    target = args.states or all_states
    invalid = [s for s in target if s not in all_states]
    if invalid:
        logger.error("Unknown states: %s", invalid)
        sys.exit(1)

    logger.info("Training %d state(s) …", len(target))
    selector = ModelSelector()
    report = []
    t0 = time.time()

    for i, state in enumerate(target, 1):
        logger.info("[%d/%d] %s", i, len(target), state)
        try:
            result = selector.fit_state(df, state, val_weeks=args.val_weeks)
            report.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("  Failed: %s", exc)
            report.append({"state": state, "error": str(exc)})

    elapsed = time.time() - t0
    logger.info("Done. Total time: %.1fs", elapsed)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'State':<22} {'Best Model':<12} {'RMSE':>14}")
    print("-" * 70)
    for r in report:
        if "error" in r:
            print(f"{r['state']:<22} {'ERROR':<12}")
        else:
            print(f"{r['state']:<22} {r['best_model']:<12} {r['best_rmse']:>14,.0f}")
    print("=" * 70)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
