"""Main experiments - same flow as the original notebook.

    python scripts/train.py                            # 5 main experiments
    python scripts/train.py --only unet_pretrained_aug # standalone pretrained + aug run (merged into results)
    python scripts/train.py --limit 200 --epochs 2     # quick smoke test

Outputs (in results/):
    final_results_summary.csv, history_all_models.json, training_comparison.png,
    predictions_<best>.png, checkpoints/<name>_best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import add_common_args, config_from_args  # noqa: E402
from src.data import build_data  # noqa: E402
from src.models import BaselineCNN, UNet, build_pretrained_unet  # noqa: E402
from src.plots import plot_comparison, visualize_predictions  # noqa: E402
from src.train import run_experiment, save_results  # noqa: E402
from src.utils import ensure_dirs, get_device, seed_everything, setup_torch  # noqa: E402


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--only", nargs="+",
                   help="run only these experiments, e.g. unet_pretrained_aug")
    args = p.parse_args()
    cfg = config_from_args(args)

    setup_torch()
    seed_everything(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    ensure_dirs(cfg.out_dir, cfg.ckpt_dir)

    ds, loaders = build_data(cfg, device)

    # Same order as the original notebook: all models are created up front.
    experiments = [
        ("baseline_plain", BaselineCNN(), loaders["train_plain"]),
        ("baseline_aug", BaselineCNN(), loaders["train_aug"]),
        ("unet_plain", UNet(), loaders["train_plain"]),
        ("unet_aug", UNet(), loaders["train_aug"]),
    ]
    if not cfg.skip_pretrained:
        experiments.append(("unet_pretrained_plain", build_pretrained_unet(), loaders["train_plain"]))

    if args.only:
        # Standalone runs (like the original "Run just unet_pretrained_aug" cell).
        loader_for = {"plain": loaders["train_plain"], "aug": loaders["train_aug"]}
        builders = {"baseline": BaselineCNN, "unet": UNet, "unet_pretrained": build_pretrained_unet}
        experiments = []
        for name in args.only:
            arch, kind = name.rsplit("_", 1)
            experiments.append((name, builders[arch](), loader_for[kind]))

    all_histories, final_results, trained_models, best_thresholds = {}, {}, {}, {}
    for name, model, loader in experiments:
        trained_model, history, row = run_experiment(name, model, loader, cfg, ds, loaders, device)
        all_histories[name] = history
        trained_models[name] = trained_model
        best_thresholds[name] = row["best_threshold"]
        final_results[name] = row

    # Merged with any earlier results, so a standalone run just adds/updates its row.
    results_df, histories = save_results(cfg.out_dir, final_results, all_histories)
    print("FINAL RESULTS SUMMARY (test set, tuned threshold + blob filter)")
    print(results_df.astype(float).round(4).to_string())

    plot_comparison(histories, os.path.join(cfg.out_dir, "training_comparison.png"))

    best_model_name = max(final_results, key=lambda k: final_results[k]["test_dice"])
    print(f"Best model on test set: {best_model_name} "
          f"(Dice={final_results[best_model_name]['test_dice']:.4f})")
    visualize_predictions(
        trained_models[best_model_name], ds["test"], device,
        os.path.join(cfg.out_dir, f"predictions_{best_model_name}.png"),
        thresh=best_thresholds[best_model_name], use_blob_filter=True,
        min_blob_size=cfg.min_blob_size,
    )


if __name__ == "__main__":
    main()
