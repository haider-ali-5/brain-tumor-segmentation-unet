"""Training loop (AMP, grad clipping, ReduceLROnPlateau, early stopping) + experiment runner."""
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import DiceBCELoss, evaluate, find_best_threshold
from .results import load_results, save_results  # noqa: F401  (re-exported for scripts/notebook)


def train_model(model, name, train_loader, val_loader, device, epochs=25, lr=1e-3,
                patience=6, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    criterion = DiceBCELoss(bce_weight=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = {"train_loss": [], "val_dice": [], "val_iou": [], "val_precision": [],
               "val_recall": [], "val_pixel_acc": [], "lr": []}

    best_dice = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(ckpt_dir, f"{name}_best.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}", leave=False)

        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(imgs)
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["dice"])
        current_lr = opt.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_pixel_acc"].append(val_metrics["pixel_acc"])
        history["lr"].append(current_lr)

        print(f"[{name}] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
              f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | "
              f"val_acc={val_metrics['pixel_acc']:.4f} | lr={current_lr:.2e}")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{name}] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(best_path))
    return model, history, best_dice


def run_experiment(name, model, train_loader, cfg, ds, loaders, device, epochs=None, patience=None):
    """Train -> tune threshold on val -> evaluate on test (tuned threshold + blob filter)."""
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
    kwargs = {"patience": patience} if patience else {}
    trained_model, history, best_val_dice = train_model(
        model, name, train_loader, loaders["val"], device,
        epochs=epochs or cfg.epochs, ckpt_dir=cfg.ckpt_dir, **kwargs,
    )

    best_t, _ = find_best_threshold(trained_model, loaders["val"], device)

    test_loader = DataLoader(ds["test"], batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    test_metrics = evaluate(trained_model, test_loader, device, thresh=best_t,
                            use_blob_filter=True, min_blob_size=cfg.min_blob_size)

    row = {
        "best_val_dice": best_val_dice,
        "best_threshold": best_t,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    print(f"[{name}] best_threshold={best_t:.2f} | TEST metrics: {test_metrics}")
    return trained_model, history, row
