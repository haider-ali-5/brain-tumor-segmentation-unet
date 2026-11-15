"""EDA: 8-panel 'clinical dashboard' (scan balance, tumor size, intensity overlap, location heatmap)."""
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_clinical_dashboard(data_dir, output_img="results/clinical_eda_results_final.png", show=False):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    mask_paths = sorted(glob(os.path.join(data_dir, "**", "*_mask.tif"), recursive=True))
    image_paths = [p.replace("_mask.tif", ".tif") for p in mask_paths]
    if not mask_paths:
        raise FileNotFoundError(f"No *_mask.tif files found under {data_dir}")

    healthy, tumor = 0, 0
    severities = []
    healthy_hist, tumor_hist = np.zeros(256), np.zeros(256)
    heatmap = None
    sample_img = sample_mask = None

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if heatmap is None:
            heatmap = np.zeros(bin_mask.shape, dtype=np.float64)

        t_px = int(np.sum(bin_mask == 255))
        if sample_img is None and t_px > 500:
            sample_img, sample_mask = img.copy(), bin_mask.copy()

        if t_px > 0:
            tumor += 1
            severities.append(100.0 * t_px / bin_mask.size)
            if bin_mask.shape == heatmap.shape:
                heatmap += bin_mask / 255.0
            tumor_hist += np.histogram(img[bin_mask == 255], bins=256, range=[0, 256])[0]
            healthy_hist += np.histogram(img[bin_mask == 0], bins=256, range=[0, 256])[0]
        else:
            healthy += 1

    total = healthy + tumor
    print("=" * 40)
    print(f"Total MRI scans : {total}")
    print(f"Healthy (no tumor): {healthy}")
    print(f"Tumor           : {tumor}")
    if severities:
        print(f"Mean tumor size : {np.mean(severities):.2f}% of slice")
    print("=" * 40)

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    fig.suptitle("Comprehensive MRI Brain Tumor Analysis", fontsize=24, fontweight="bold", y=0.96)

    ax = axes[0, 0]
    sns.barplot(x=["Healthy Scans", "Tumor Scans"], y=[healthy, tumor], ax=ax,
                hue=["Healthy Scans", "Tumor Scans"], palette=["#2ECC71", "#E74C3C"], legend=False)
    for i, v in enumerate([healthy, tumor]):
        ax.text(i, v + max(healthy, tumor) * 0.01, str(v), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Scan Classification Breakdown", fontweight="bold"); ax.set_ylabel("Number of MRI Scans")

    ax = axes[0, 1]
    if severities:
        sns.histplot(severities, bins=30, kde=True, ax=ax, color="#5DADE2", edgecolor="white")
        m = np.mean(severities)
        ax.axvline(m, color="red", linestyle="--", linewidth=2, label=f"Mean: {m:.2f}%")
        ax.legend()
    ax.set_title("Tumor Severity (Size Analysis)", fontweight="bold")
    ax.set_xlabel("Percentage of Brain Affected (%)"); ax.set_ylabel("Frequency")

    ax = axes[0, 2]
    for hist, color, label in [(healthy_hist, "#2ECC71", "Healthy Tissue"), (tumor_hist, "#E74C3C", "Tumor Tissue")]:
        if hist.sum() > 0:
            h = hist / hist.sum()
            ax.plot(h, color=color, linewidth=1.5)
            ax.fill_between(range(256), h, color=color, alpha=0.3, label=label)
    ax.set_title("Pixel Intensity: Healthy vs Tumor", fontweight="bold")
    ax.set_xlabel("Pixel Value (0=Black, 255=White)"); ax.set_ylabel("Density"); ax.legend()

    ax = axes[0, 3]
    if heatmap is not None and heatmap.max() > 0:
        im = ax.imshow(heatmap / heatmap.max(), cmap="jet")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Global Tumor Location Heatmap", fontweight="bold"); ax.axis("off")

    ax = axes[1, 0]
    ax.axis("off")
    ax.text(0.5, 0.5, f"DATASET SUMMARY\n\nTotal Scans:\n{total}\n\nHealthy (No Tumor):\n{healthy}"
                      f"\n\nNon-Healthy (Tumor):\n{tumor}",
            fontsize=18, fontweight="bold", ha="center", va="center",
            bbox=dict(facecolor="#f8f9fa", edgecolor="#dee2e6", boxstyle="round,pad=1"))
    ax.set_title("Quick Stats", fontweight="bold")

    if sample_img is not None:
        axes[1, 1].imshow(sample_img, cmap="gray")
        axes[1, 2].imshow(sample_mask, cmap="gray")
        rgb = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB)
        red = np.zeros_like(rgb)
        red[sample_mask == 255] = [255, 50, 50]
        axes[1, 3].imshow(cv2.addWeighted(rgb, 1.0, red, 0.5, 0))
    for ax, t in zip(axes[1, 1:], ["Sample MRI Scan", "Ground Truth Mask", "Tumor Highlight Overlay"]):
        ax.set_title(t, fontweight="bold"); ax.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    os.makedirs(os.path.dirname(output_img) or ".", exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    print(f"Saved: {output_img}")
    return {"total": total, "healthy": healthy, "tumor": tumor,
            "mean_severity": float(np.mean(severities)) if severities else 0.0}
