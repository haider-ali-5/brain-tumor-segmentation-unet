"""Phase 1 - EDA dashboard.

    python scripts/run_eda.py [--data_dir data/kaggle_3m]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import add_common_args, config_from_args  # noqa: E402
from src.eda import generate_clinical_dashboard  # noqa: E402

if __name__ == "__main__":
    cfg = config_from_args(add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args())
    generate_clinical_dashboard(cfg.data_dir, os.path.join(cfg.out_dir, "clinical_eda_results_final.png"))
