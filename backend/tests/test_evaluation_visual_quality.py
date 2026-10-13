"""Tests for visual-quality measurement.

LPIPS is treated as optional on purpose, so both states are exercised: the
happy path asserts real behaviour, and the unavailable path asserts that the
metric degrades to ``null`` with a stated reason instead of raising. A
benchmark that hard-fails because one optional metric is missing reports less,
not more.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from app.evaluation.visual_quality import (
    LPIPS_NET,
    VisualQuality,
    _lpips_tensor,
    compute_lpips,
    compute_visual_quality,
    lpips_status,
)

from tests.helpers import make_detailed, make_photo

CPU = torch.device("cpu")


def test_identical_images_have_no_difference():
    image = make_photo(32, 32)
    quality = compute_visual_quality(image, image, CPU)
    assert quality.linf_255 == 0.0
    assert quality.mean_abs == 0.0
    assert quality.pixels_changed_fraction == 0.0


def test_psnr_of_identical_images_is_not_json_serialisable_inf():
    """`inf` becomes `null` in JSON, which is what the schema promises."""
    image = make_photo(16, 16)
    payload = compute_visual_quality(image, image, CPU).to_dict()
    assert payload["psnr_db"] is None


def test_uniform_shift_is_measured_exactly():
    original = np.full((16, 16, 3), 100, dtype=np.uint8)
    protected = np.full((16, 16, 3), 108, dtype=np.uint8)
    quality = compute_visual_quality(original, protected, CPU)
    assert quality.linf_255 == pytest.approx(8.0)
    assert quality.linf_8bit == 8
    assert quality.mean_abs == pytest.approx(8.0)
    assert quality.pixels_changed_fraction == 1.0


def test_shape_mismatch_is_rejected_not_silently_resized():
    """A silent resize here would corrupt every downstream comparison."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_visual_quality(make_photo(16, 16), make_photo(16, 24), CPU)


def test_ssim_and_psnr_are_bounded_sensibly():
    original = make_detailed(64, 64)
    protected = np.clip(original.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    quality = compute_visual_quality(original, protected, CPU)
    assert -1.0 <= quality.ssim <= 1.0
    assert 10.0 < quality.psnr_db < 60.0


def test_payload_has_a_stable_key_set():
    image = make_photo(16, 16)
    payload = compute_visual_quality(image, image, CPU).to_dict()
    assert set(payload) == {
        "linf_255",
        "linf_8bit",
        "l2",
        "mean_abs",
        "psnr_db",
        "ssim",
        "lpips",
        "pixels_changed_fraction",
    }


def test_lpips_tensor_maps_to_minus_one_to_one():
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert float(_lpips_tensor(black, CPU).min()) == pytest.approx(-1.0)
    assert float(_lpips_tensor(white, CPU).max()) == pytest.approx(1.0)


def test_lpips_tensor_upsamples_tiny_images():
    """A 1x1 input would collapse the network's receptive field."""
    tiny = _lpips_tensor(np.zeros((1, 1, 3), dtype=np.uint8), CPU)
    assert min(tiny.shape[-2:]) >= 32


def test_lpips_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_lpips(make_photo(16, 16), make_photo(24, 16), CPU)


def test_lpips_is_near_zero_for_identical_images():
    image = make_detailed(64, 64)
    assert compute_lpips(image, image, CPU) == pytest.approx(0.0, abs=1e-4)


def test_lpips_grows_with_perturbation():
    image = make_detailed(64, 64)
    small = np.clip(image.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    large = np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    assert compute_lpips(image, small, CPU) < compute_lpips(image, large, CPU)


def test_lpips_status_is_one_of_two_states():
    assert lpips_status(CPU) in {"available", "unavailable"}


def test_lpips_degrades_to_null_when_unavailable(monkeypatch):
    """Absent metric -> `null` plus a reason, never an exception."""
    monkeypatch.setattr(
        "app.evaluation.visual_quality._lpips_model", lambda device: None
    )
    assert compute_lpips(make_photo(16, 16), make_photo(16, 16), CPU) is None
    assert lpips_status(CPU) == "unavailable"

    quality = compute_visual_quality(make_photo(16, 16), make_photo(16, 16), CPU)
    assert quality.lpips is None
    assert quality.to_dict()["lpips"] is None
    # The other metrics must still be real numbers.
    assert quality.ssim == pytest.approx(1.0)


def test_lpips_net_is_declared():
    assert LPIPS_NET in {"alex", "vgg", "squeeze"}


def test_visual_quality_is_immutable():
    quality = VisualQuality(
        linf_255=8.0,
        linf_8bit=8,
        l2=1.0,
        mean_abs=1.0,
        psnr_db=40.0,
        ssim=0.95,
        lpips=0.01,
        pixels_changed_fraction=0.5,
    )
    with pytest.raises(AttributeError):
        quality.linf_255 = 99.0  # type: ignore[misc]
