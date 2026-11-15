"""Data discovery, train/val/test split, augmentation pipelines, Dataset and loaders."""
import glob
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Discovery + split
# --------------------------------------------------------------------------- #
def get_image_and_mask_paths(data_dir):
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*_mask.tif"), recursive=True))
    image_paths = [p.replace("_mask.tif", ".tif") for p in mask_paths]
    return image_paths, mask_paths


def compute_tumor_flags(mask_paths):
    flags = []
    for mp in tqdm(mask_paths, desc="Scanning masks for stratification"):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        flags.append(1 if (m is not None and (m > 127).sum() > 0) else 0)
    return np.array(flags)


def patient_id(path):
    """kaggle_3m/<patient_folder>/<patient>_<slice>.tif -> patient folder name."""
    return os.path.basename(os.path.dirname(path))


def make_splits(cfg):
    """Returns dict {train,val,test} -> (image_paths, mask_paths). 70/15/15."""
    images, masks = get_image_and_mask_paths(cfg.data_dir)
    print(f"Total images: {len(images)} | Total masks: {len(masks)}")
    assert len(images) > 0, (
        f"No data found in {cfg.data_dir}. Download the Kaggle 'lgg-mri-segmentation' "
        f"dataset and put the 'kaggle_3m' folder in data/ (see data/README.md)."
    )

    if cfg.limit is not None:
        combined = list(zip(images, masks))
        random.Random(cfg.seed).shuffle(combined)
        combined = combined[: cfg.limit]
        images, masks = [c[0] for c in combined], [c[1] for c in combined]
        print(f"limit set: using a random subset of {len(images)} images")

    idx_all = np.arange(len(images))

    if cfg.split_mode == "patient":
        pids = np.array([patient_id(p) for p in images])
        uniq = np.unique(pids)
        p_train, p_temp = train_test_split(uniq, test_size=0.3, random_state=cfg.seed)
        p_val, p_test = train_test_split(p_temp, test_size=0.5, random_state=cfg.seed)
        idx_train = idx_all[np.isin(pids, p_train)]
        idx_val = idx_all[np.isin(pids, p_val)]
        idx_test = idx_all[np.isin(pids, p_test)]
    elif cfg.split_mode == "slice":
        flags = compute_tumor_flags(masks)
        idx_train, idx_temp = train_test_split(
            idx_all, test_size=0.3, random_state=cfg.seed, stratify=flags)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.5, random_state=cfg.seed, stratify=flags[idx_temp])
    else:
        raise ValueError(f"split_mode must be 'slice' or 'patient', got {cfg.split_mode!r}")

    def pick(idxs):
        return [images[i] for i in idxs], [masks[i] for i in idxs]

    splits = {"train": pick(idx_train), "val": pick(idx_val), "test": pick(idx_test)}
    print("Split ({}) -> train: {} | val: {} | test: {}".format(
        cfg.split_mode, len(splits["train"][0]), len(splits["val"][0]), len(splits["test"][0])))
    return splits


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(cfg):
    size, mean, std = cfg.img_size, list(cfg.mean), list(cfg.std)

    # Main augmentation. mask_interpolation=NEAREST keeps the 0/1 mask labels intact.
    aug = A.Compose([
        A.Resize(height=size, width=size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3,
                           interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5,
                           interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    plain = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # Diagnostic 2: softer + less frequent augmentation.
    aug_soft = A.Compose([
        A.Resize(height=size, width=size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.ElasticTransform(alpha=80, sigma=120 * 0.10, p=0.2),   # smoother, less frequent
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return {"plain": plain, "aug": aug, "aug_soft": aug_soft, "val": val}


# --------------------------------------------------------------------------- #
# Dataset + loaders
# --------------------------------------------------------------------------- #
class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = torch.unsqueeze(out["mask"], 0)
            mask = (mask > 0.5).float()  # re-binarize (safety net)
        return image, mask


def make_loader(dataset, cfg, shuffle, device):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=(device == "cuda"))


def build_data(cfg, device):
    """Everything the training scripts need, in one call."""
    splits = make_splits(cfg)
    tfm = build_transforms(cfg)

    ds = {
        "val": BrainMRIDataset(*splits["val"], transform=tfm["val"]),
        "test": BrainMRIDataset(*splits["test"], transform=tfm["val"]),
        "train_plain": BrainMRIDataset(*splits["train"], transform=tfm["plain"]),
        "train_aug": BrainMRIDataset(*splits["train"], transform=tfm["aug"]),
        "train_aug_soft": BrainMRIDataset(*splits["train"], transform=tfm["aug_soft"]),
    }
    loaders = {k: make_loader(v, cfg, k.startswith("train"), device) for k, v in ds.items()}
    return ds, loaders
