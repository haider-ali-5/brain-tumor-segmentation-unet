import os
import random

import numpy as np
import torch


def setup_torch():
    # Many shared GPU environments cap /dev/shm at 64MB; the file_system sharing
    # strategy avoids "unable to allocate shared memory" crashes in DataLoader workers.
    torch.multiprocessing.set_sharing_strategy("file_system")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
