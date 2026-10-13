"""Shared paths, deterministic seeding and Unicode-safe image IO.

OpenCV's ``imread``/``imwrite`` go through the C runtime, which uses the
platform's narrow path API. On Windows that means any path containing
non-ASCII characters -- including a Cyrillic file name, which is exactly what
a real user drops into a folder -- silently returns ``None``. Every read in
this project therefore goes through ``imread``/``imwrite`` below, which use
``numpy.fromfile``/``tofile`` and keep the path as UTF-16 end to end.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

SEED = 0

ROOT = Path(__file__).resolve().parent.parent
FACES_DIR = ROOT / "data" / "faces"
DONOR_DIR = ROOT / "data" / "donor"
OUTPUTS_DIR = ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"
MODELS_DIR = ROOT / "models"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed: int = SEED) -> None:
    """Pin every RNG the project touches so a rerun reproduces the numbers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(directory: Path) -> list[Path]:
    """Every image under ``directory``, sorted, so run order is deterministic."""
    if not directory.is_dir():
        return []
    found = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(found)


def imread(path: Path | str) -> np.ndarray:
    """Read an image as BGR uint8. Raises instead of returning ``None``."""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def imwrite(path: Path | str, image: np.ndarray) -> Path:
    """Write a BGR uint8 image, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(path))
    return path


def read_rgb(path: Path | str) -> np.ndarray:
    """Read an image as RGB uint8, the layout torch models expect."""
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)