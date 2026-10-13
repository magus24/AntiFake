"""Difference and similarity metrics used to report (and test) a protection run.

Implemented directly on numpy/OpenCV so the MVP has no metrics dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.protection.base import RGB


@dataclass(frozen=True, slots=True)
class DifferenceMetrics:
    """How far a protected image sits from its source.

    Attributes:
        linf_255: Largest absolute per-channel difference, in 8-bit levels.
        linf_8bit: Same value, rounded to an int (what a viewer perceives).
        l2: Root-mean-square difference across all pixels/channels, 0-255 scale.
        mean_abs: Mean absolute difference, 0-255 scale.
        psnr_db: Peak signal-to-noise ratio in dB. Higher is closer to the
            original. ``inf`` for an identical pair.
        ssim: Structural similarity in ``[-1, 1]``, computed with an 11x11
            Gaussian window (the standard Wang et al. formulation).
    """

    linf_255: float
    linf_8bit: int
    l2: float
    mean_abs: float
    psnr_db: float
    ssim: float

    def to_dict(self) -> dict[str, float | int]:
        """Flat mapping, for JSON serialisation."""
        return asdict(self)


def _as_gray_255(array: RGB) -> np.ndarray:
    """Luma channel as float32 on a 0-255 scale.

    The scale matters: the C1/C2 stabilising constants below are defined for
    8-bit data, so normalising the image to ``[0, 1]`` here would make them
    ~65000x too large and collapse every SSIM score to approximately 1.
    """
    return cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)


def compute_psnr(original: RGB, protected: RGB) -> float:
    """PSNR in dB between two ``uint8`` RGB arrays.

    Identical inputs yield ``inf`` (reported as a large finite number by JSON
    encoders on some stacks, so callers should treat ``math.isinf`` specially).
    """
    if original.shape != protected.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {protected.shape}."
        )
    mse = float(
        np.mean(
            (original.astype(np.float64) - protected.astype(np.float64)) ** 2,
        )
    )
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def compute_ssim(
    original: RGB,
    protected: RGB,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean SSIM between two ``uint8`` RGB arrays on the luma channel."""
    if original.shape != protected.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {protected.shape}."
        )
    a = _as_gray_255(original)
    b = _as_gray_255(protected)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    kernel = cv2.getGaussianKernel(window_size, sigma)
    kernel_2d = kernel @ kernel.T

    def blur(img: np.ndarray) -> np.ndarray:
        return cv2.filter2D(img, -1, kernel_2d, borderType=cv2.BORDER_REFLECT)

    mu_a = blur(a)
    mu_b = blur(b)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = blur(a * a) - mu_a_sq
    sigma_b_sq = blur(b * b) - mu_b_sq
    sigma_ab = blur(a * b) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(numerator / denominator))


def compute_metrics(original: RGB, protected: RGB) -> DifferenceMetrics:
    """Bundle every metric the API and the tests care about."""
    if original.shape != protected.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {protected.shape}."
        )
    diff = original.astype(np.float64) - protected.astype(np.float64)
    abs_diff = np.abs(diff)
    linf = float(abs_diff.max())
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse))
    return DifferenceMetrics(
        linf_255=linf,
        linf_8bit=round(linf),
        l2=float(np.sqrt(mse)),
        mean_abs=float(abs_diff.mean()),
        psnr_db=psnr,
        ssim=compute_ssim(original, protected),
    )
