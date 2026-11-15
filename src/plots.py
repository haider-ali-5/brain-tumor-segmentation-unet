"""Plots: training comparison, convergence diagnostic, prediction samples."""
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# torch is imported lazily inside visualize_predictions so that results-only
# scripts (show_results.py) don't need torch at all.


def plot_comparison(histories, out_path, show=False):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Model Comparison: Baseline vs U-Net vs Pretrained U-Net, With/Without Augmentation",
                 fontsize=16, fontweight="bold")
    panels = [("train_loss", "Training Loss"), ("val_dice", "Validation Dice"),
              ("val_iou", "Validation IoU"), ("val_pixel_acc", "Validation Pixel Accuracy")]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, h in histories.items():
            ax.plot(h[key], label=name)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    print(f"Saved: {out_path}")


def plot_convergence(histories, plain_name, aug_name, out_path, show=False):
    """Diagnostic 1: did the augmented run stop early while still improving?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for n in (plain_name, aug_name):
        axes[0].plot(histories[n]["train_loss"], label=n, marker="o")
        axes[1].plot(histories[n]["val_dice"], label=n, marker="o")
    axes[0].set_title("Training Loss"); axes[1].set_title("Validation Dice")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show() if show else plt.close(fig)

    n_p, n_a = len(histories[plain_name]["val_dice"]), len(histories[aug_name]["val_dice"])
    print(f"{plain_name} ran {n_p} epochs | {aug_name} ran {n_a} epochs | saved {out_path}")
    if n_a < n_p:
        print(f"-> {aug_name} stopped {n_p - n_a} epochs earlier; if its val_dice was still "
              f"rising it likely needed more patience.")


def visualize_predictions(model, dataset, device, out_path, thresh=0.5, use_blob_filter=False,
                          min_blob_size=20, n=6, show=False):
    """Input | Ground truth | Prediction for n random test slices."""
    import torch  # lazy: results-only scripts don't need torch
    from .metrics import remove_small_blobs

    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, idx in enumerate(idxs):
        img, mask = dataset[idx]
        with torch.no_grad():
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        pred_bin = (pred > thresh).astype(np.float32)
        if use_blob_filter:
            pred_bin = remove_small_blobs(pred_bin, min_size=min_blob_size)

        img_show = img.permute(1, 2, 0).numpy()
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-7)

        axes[row, 0].imshow(img_show); axes[row, 0].set_title("Input"); axes[row, 0].axis("off")
        axes[row, 1].imshow(mask[0].numpy(), cmap="gray"); axes[row, 1].set_title("Ground Truth"); axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_bin, cmap="gray"); axes[row, 2].set_title("Prediction"); axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    print(f"Saved: {out_path}")
