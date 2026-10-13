"""Visual-quality metrics: PSNR, SSIM and LPIPS.

These are the honesty check on the attack. A perturbation that wrecks the
photo is not protection, it is vandalism, so every result row in the report
carries them next to the ArcFace numbers.
"""

from __future__ import annotations

import functools
import math

import cv2
import numpy as np
import torch


def psnr(original: np.ndarray, protected: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between two uint8 images."""
    a = original.astype(np.float64)
    b = protected.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.outer(kernel, kernel)


def ssim(original: np.ndarray, protected: np.ndarray) -> float:
    """Mean SSIM over the luma channel, Gaussian window 11x11 sigma 1.5."""
    # float32 rather than float64: SSIM is a ratio of averaged products, the
    # precision difference is far below reporting noise, and on a multi-megapixel
    # photo the float64 path costs seconds for nothing.
    a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = _gaussian_window(11, 1.5).astype(np.float32)
    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    sigma_a = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a * mu_a
    sigma_b = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b * mu_b
    sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


@functools.lru_cache(maxsize=1)
def _lpips_model() -> torch.nn.Module:
    import lpips

    torch.manual_seed(0)
    return lpips.LPIPS(net="alex", verbose=False).eval()


def lpips_distance(original: np.ndarray, protected: np.ndarray) -> float | None:
    """LPIPS perceptual distance, or ``None`` if the weights are unavailable."""
    try:
        model = _lpips_model()
    except Exception:
        return None

    def to_tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        value = model(to_tensor(original), to_tensor(protected))
    return float(value.item())


def linf_255(original: np.ndarray, protected: np.ndarray) -> int:
    """Worst per-channel deviation, in 8-bit levels."""
    diff = protected.astype(np.int16) - original.astype(np.int16)
    return int(np.abs(diff).max())


def visual_quality(original: np.ndarray, protected: np.ndarray) -> dict:
    return {
        "psnr_db": round(psnr(original, protected), 4),
        "ssim": round(ssim(original, protected), 6),
        "lpips": (
            None
            if (value := lpips_distance(original, protected)) is None
            else round(value, 6)
        ),
        "linf_255": linf_255(original, protected),
    }