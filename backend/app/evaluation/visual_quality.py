"""Visual-quality metrics for a protected image.

L-inf, L2, mean absolute difference, PSNR and SSIM are re-used from
:mod:`app.protection.metrics` rather than reimplemented, so the benchmark and
the API can never disagree about how bad a perturbation looks. LPIPS is added
here because it is the one metric the product does not currently compute and
it answers a question the others cannot: two images with identical PSNR can
look nothing alike, and LPIPS is sensitive to perceptual structure rather
than to pixel arithmetic.

LPIPS is optional. If the dependency or its weights are unavailable the
metric is reported as ``null`` with a stated reason instead of failing the
run, because "we could not measure this" and "this number is bad" must not
look the same in the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.protection.base import RGB
from app.protection.metrics import compute_metrics

logger = logging.getLogger(__name__)

#: LPIPS reports in roughly [0, 1]; below this the pair is visually identical.
LPIPS_NET = "alex"
_LPIPS_CACHE: dict[str, Any] = {}


def _lpips_model(device: torch.device) -> Any:
    """Load and cache the LPIPS network, or return ``None`` if unavailable."""
    key = f"{device.type}:{LPIPS_NET}"
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips
    except ImportError:
        logger.warning("lpips.unavailable reason=package_not_installed")
        _LPIPS_CACHE[key] = None
        return None
    try:
        model = lpips.LPIPS(net=LPIPS_NET).to(device).eval()
    except Exception as exc:
        logger.warning("lpips.unavailable reason=%s", exc)
        _LPIPS_CACHE[key] = None
        return None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _LPIPS_CACHE[key] = model
    return model


def _lpips_tensor(image: RGB, device: torch.device) -> torch.Tensor:
    """``(H, W, 3) uint8`` -> ``(1, 3, H, W)`` in LPIPS' expected ``[-1, 1]``.

    Copies rather than wrapping: transform and decode pipelines hand out
    read-only arrays, and a tensor over read-only memory is undefined.
    """
    array = (
        torch.from_numpy(np.array(image, dtype=np.uint8, order="C", copy=True))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div(127.5)
        .sub(1.0)
    )
    # LPIPS downsamples internally; a 1x1 input would collapse its receptive
    # field, so tiny images are upsampled to something the network can read.
    if min(array.shape[-2:]) < 32:
        scale = 32.0 / float(min(array.shape[-2:]))
        array = torch.nn.functional.interpolate(
            array,
            size=(
                max(32, round(array.shape[-2] * scale)),
                max(32, round(array.shape[-1] * scale)),
            ),
            mode="bilinear",
            align_corners=False,
        )
    return array


def compute_lpips(original: RGB, protected: RGB, device: torch.device) -> float | None:
    """Perceptual distance in ``[0, ~1]``, or ``None`` if LPIPS is unavailable."""
    if original.shape != protected.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {protected.shape}.")
    model = _lpips_model(device)
    if model is None:
        return None
    with torch.no_grad():
        value = model(
            _lpips_tensor(original, device), _lpips_tensor(protected, device)
        )
    return float(value.flatten()[0])


def lpips_status(device: torch.device) -> str:
    """Short reason string describing LPIPS availability, for the report."""
    return "available" if _lpips_model(device) is not None else "unavailable"


@dataclass(frozen=True, slots=True)
class VisualQuality:
    """Every visual-quality number for one original/protected pair.

    ``linf_255`` is the honest bound: what the engine promised. The remaining
    fields describe the delivered file, which for JPEG is measurably worse
    than the in-memory array.
    """

    linf_255: float
    linf_8bit: int
    l2: float
    mean_abs: float
    psnr_db: float
    ssim: float
    lpips: float | None
    pixels_changed_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping with a stable key set."""
        psnr = self.psnr_db if np.isfinite(self.psnr_db) else None
        return {
            "linf_255": round(self.linf_255, 4),
            "linf_8bit": int(self.linf_8bit),
            "l2": round(self.l2, 5),
            "mean_abs": round(self.mean_abs, 5),
            "psnr_db": None if psnr is None else round(psnr, 4),
            "ssim": round(self.ssim, 6),
            "lpips": None if self.lpips is None else round(self.lpips, 6),
            "pixels_changed_fraction": round(self.pixels_changed_fraction, 6),
        }


def compute_visual_quality(
    original: RGB, protected: RGB, device: torch.device
) -> VisualQuality:
    """Full visual-quality bundle for one pair."""
    if original.shape != protected.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {protected.shape}.")
    metrics = compute_metrics(original, protected)
    diff = np.abs(original.astype(np.int16) - protected.astype(np.int16))
    return VisualQuality(
        linf_255=float(metrics.linf_255),
        linf_8bit=int(metrics.linf_8bit),
        l2=float(metrics.l2),
        mean_abs=float(metrics.mean_abs),
        psnr_db=float(metrics.psnr_db),
        ssim=float(metrics.ssim),
        lpips=compute_lpips(original, protected, device),
        pixels_changed_fraction=float((diff.max(axis=2) > 0).mean()),
    )
