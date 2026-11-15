"""All settings in one place (same values as the original notebook). CLI flags can override them."""
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    data_dir: str = os.path.join(ROOT, "data", "kaggle_3m")   # path to your dataset
    out_dir: str = os.path.join(ROOT, "results")
    epochs: int = 25
    batch_size: int = 64
    num_workers: int = 0          # keep 0 unless you've confirmed /dev/shm headroom
    limit: int = None             # set e.g. 200 for a quick smoke test
    min_blob_size: int = 20       # min connected-component size kept after thresholding
    skip_pretrained: bool = False # True if no internet for ImageNet weights
    seed: int = 42
    img_size: int = 256
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    split_mode: str = "slice"     # "slice" = original split; "patient" = optional split by patient

    @property
    def ckpt_dir(self):
        return os.path.join(self.out_dir, "checkpoints")


def add_common_args(parser):
    """CLI flags shared by every script; anything not given keeps the Config default."""
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--limit", type=int, help="use only N random slices (smoke test)")
    parser.add_argument("--split_mode", choices=["slice", "patient"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip_pretrained", action="store_true", default=None,
                        help="skip the pretrained U-Net (no internet for ImageNet weights)")
    return parser


def config_from_args(args):
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
