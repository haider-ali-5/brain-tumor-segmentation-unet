"""Loss, metrics, threshold tuning, small-blob post-processing and bootstrap CI."""
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1e-7) / (
            probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-7
        )
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def remove_small_blobs(pred_mask_np, min_size=20):
    labeled, n = ndimage.label(pred_mask_np)
    if n == 0:
        return pred_mask_np
    sizes = ndimage.sum(pred_mask_np, labeled, range(1, n + 1))
    out = pred_mask_np.copy()
    for i, size in enumerate(sizes, start=1):
        if size < min_size:
            out[labeled == i] = 0
    return out


@torch.no_grad()
def evaluate(model, loader, device, thresh=0.5, use_blob_filter=False, min_blob_size=20):
    model.eval()
    dices, ious, precisions, recalls, accs = [], [], [], [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        probs = torch.sigmoid(model(imgs))
        preds = (probs > thresh).float()

        if use_blob_filter:
            preds_np = preds.cpu().numpy()
            for b in range(preds_np.shape[0]):
                preds_np[b, 0] = remove_small_blobs(preds_np[b, 0], min_size=min_blob_size)
            preds = torch.from_numpy(preds_np).to(device)

        tp = (preds * masks).sum(dim=(1, 2, 3))
        fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
        tn = ((1 - preds) * (1 - masks)).sum(dim=(1, 2, 3))

        dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        dices.extend(dice.cpu().tolist())
        ious.extend(iou.cpu().tolist())
        precisions.extend(precision.cpu().tolist())
        recalls.extend(recall.cpu().tolist())
        accs.extend(acc.cpu().tolist())

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "pixel_acc": float(np.mean(accs)),
    }


def find_best_threshold(model, val_loader, device, thresholds=np.arange(0.30, 0.71, 0.05)):
    best_t, best_dice = 0.5, -1.0
    for t in thresholds:
        m = evaluate(model, val_loader, device, thresh=float(t))
        if m["dice"] > best_dice:
            best_dice, best_t = m["dice"], float(t)
    return best_t, best_dice


# --------------------------------------------------------------------------- #
# Diagnostic 3: bootstrap confidence interval of per-image Dice
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_image_dice(model, dataset, device, thresh=0.5, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    dices = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = (torch.sigmoid(model(imgs)) > thresh).float()
        tp = (preds * masks).sum(dim=(1, 2, 3))
        fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
        dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        dices.extend(dice.cpu().tolist())
    return np.array(dices)


def bootstrap_ci(scores, n_boot=2000, ci=95):
    boots = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return np.mean(scores), lo, hi
