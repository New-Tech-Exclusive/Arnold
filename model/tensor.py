"""Shared tensor helpers for PyTorch-backed model code."""

from __future__ import annotations

import random

import numpy as np
import torch

# Keep one dtype across the custom Hebbian math code.
# Use float32 by default for better performance on modern hardware.
DTYPE = torch.float32


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_tensor(data, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    return torch.as_tensor(data, dtype=dtype or DTYPE, device=device or get_device())


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
