"""Results persistence (no torch needed). Merge-safe: re-running one model updates just its row."""
import json
import os

import pandas as pd


def results_paths(out_dir):
    return (os.path.join(out_dir, "final_results_summary.csv"),
            os.path.join(out_dir, "history_all_models.json"))


def load_results(out_dir):
    csv_p, json_p = results_paths(out_dir)
    results = pd.read_csv(csv_p, index_col=0).to_dict(orient="index") if os.path.exists(csv_p) else {}
    histories = {}
    if os.path.exists(json_p):
        with open(json_p) as f:
            histories = json.load(f)
    return results, histories


def save_results(out_dir, new_results, new_histories):
    os.makedirs(out_dir, exist_ok=True)
    results, histories = load_results(out_dir)
    results.update(new_results)
    histories.update(new_histories)
    csv_p, json_p = results_paths(out_dir)
    df = pd.DataFrame(results).T
    df.to_csv(csv_p)
    with open(json_p, "w") as f:
        json.dump(histories, f, indent=2)
    return df, histories
