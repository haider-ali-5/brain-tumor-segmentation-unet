"""Print the saved results table and redraw the comparison plot - no training.

    python scripts/show_results.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.config import add_common_args, config_from_args  # noqa: E402
from src.plots import plot_comparison  # noqa: E402
from src.results import load_results  # noqa: E402

if __name__ == "__main__":
    cfg = config_from_args(add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args())
    results, histories = load_results(cfg.out_dir)
    if not results:
        sys.exit(f"No results in {cfg.out_dir}. Run scripts/train.py first.")
    df = pd.DataFrame(results).T.astype(float)
    print("=" * 60 + "\nFINAL RESULTS SUMMARY (test set)\n" + "=" * 60)
    print(df.round(4).to_string())
    best = df["test_dice"].idxmax()
    print(f"\nBest model: {best} (test Dice={df.loc[best, 'test_dice']:.4f})")
    if histories:
        plot_comparison(histories, os.path.join(cfg.out_dir, "training_comparison.png"))
