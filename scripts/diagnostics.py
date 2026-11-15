"""Augmentation diagnostics - same as the original notebook (run AFTER scripts/train.py).

  1. Did augmented training actually converge? (unet_plain vs unet_aug curves)
  2. Re-run aug with more patience + softer augmentation -> unet_aug_soft_longtrain
  3. Is the plain vs aug gap statistically meaningful? (bootstrap 95% CI)

    python scripts/diagnostics.py
    python scripts/diagnostics.py --skip_soft      # only 1 and 3 (no extra training)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import add_common_args, config_from_args  # noqa: E402
from src.data import build_data  # noqa: E402
from src.metrics import bootstrap_ci, per_image_dice  # noqa: E402
from src.models import UNet  # noqa: E402
from src.plots import plot_convergence  # noqa: E402
from src.train import load_results, run_experiment, save_results  # noqa: E402
from src.utils import get_device, seed_everything, setup_torch  # noqa: E402


def load_unet(name, cfg, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.ckpt_dir, f"{name}_best.pt"), map_location=device))
    return model


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--skip_soft", action="store_true", help="skip Diagnostic 2 (extra training)")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_torch()
    seed_everything(cfg.seed)
    device = get_device()

    final_results, all_histories = load_results(cfg.out_dir)
    if "unet_plain" not in final_results or "unet_aug" not in final_results:
        sys.exit(f"unet_plain / unet_aug results not found in {cfg.out_dir}. Run scripts/train.py first.")

    # ---- Diagnostic 1: did augmented training actually converge? -------------
    print("\n### Diagnostic 1: convergence")
    plot_convergence(all_histories, "unet_plain", "unet_aug",
                     os.path.join(cfg.out_dir, "diag1_convergence_unet.png"))

    ds, loaders = build_data(cfg, device)

    # ---- Diagnostic 2: more patience + softer augmentation -------------------
    if not args.skip_soft:
        print("\n### Diagnostic 2: re-run aug with more patience + softer augmentation")
        name = "unet_aug_soft_longtrain"
        _, history, row = run_experiment(name, UNet(), loaders["train_aug_soft"], cfg, ds, loaders,
                                         device, epochs=40, patience=12)
        print(f"[{name}] best_val_dice={row['best_val_dice']:.4f} | best_threshold={row['best_threshold']:.2f}")
        print("\nCompare against:")
        print(f"  unet_plain   test_dice = {float(final_results['unet_plain']['test_dice']):.4f}")
        print(f"  unet_aug     test_dice = {float(final_results['unet_aug']['test_dice']):.4f}")
        print(f"  {name} test_dice = {row['test_dice']:.4f}")
        save_results(cfg.out_dir, {name: row}, {name: history})

    # ---- Diagnostic 3: is the gap statistically meaningful? ------------------
    print("\n### Diagnostic 3: bootstrap 95% CI")
    plain_dices = per_image_dice(load_unet("unet_plain", cfg, device), ds["test"], device)
    aug_dices = per_image_dice(load_unet("unet_aug", cfg, device), ds["test"], device)

    mean_p, lo_p, hi_p = bootstrap_ci(plain_dices)
    mean_a, lo_a, hi_a = bootstrap_ci(aug_dices)

    print(f"unet_plain: mean={mean_p:.4f}  95% CI=[{lo_p:.4f}, {hi_p:.4f}]")
    print(f"unet_aug:   mean={mean_a:.4f}  95% CI=[{lo_a:.4f}, {hi_a:.4f}]")
    if hi_a > lo_p:
        print("-> CIs overlap: the gap may not be statistically robust from one run alone.")
    else:
        print("-> CIs do NOT overlap: this looks like a real, reproducible effect.")


if __name__ == "__main__":
    main()
