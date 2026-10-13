"""Similarity and difference metric tests."""

from __future__ import annotations

import numpy as np
import pytest
from app.protection.metrics import compute_metrics, compute_psnr, compute_ssim

from tests.helpers import make_detailed, make_photo


class TestPsnr:
    def test_identical_images_are_infinite(self, photo: np.ndarray) -> None:
        assert compute_psnr(photo, photo) == float("inf")

    def test_known_small_difference(self) -> None:
        a = np.full((16, 16, 3), 100, dtype=np.uint8)
        b = np.full((16, 16, 3), 104, dtype=np.uint8)
        # MSE = 16, PSNR = 10*log10(255^2/16) = 36.09... dB
        assert compute_psnr(a, b) == pytest.approx(36.09, abs=0.01)

    def test_larger_difference_gives_lower_psnr(self, photo: np.ndarray) -> None:
        mild = np.clip(photo.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        harsh = np.clip(photo.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        assert compute_psnr(photo, harsh) < compute_psnr(photo, mild)

    def test_shape_mismatch_raises(self, photo: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(photo, make_photo(32, 32))


class TestSsim:
    def test_identical_images_score_one(self, photo: np.ndarray) -> None:
        assert compute_ssim(photo, photo) == pytest.approx(1.0, abs=1e-6)

    def test_inverted_images_score_far_below_zero(self, photo: np.ndarray) -> None:
        inverted = (255 - photo).astype(np.uint8)
        assert compute_ssim(photo, inverted) < 0.0

    def test_structurally_unrelated_images_score_far_below_one(self) -> None:
        """SSIM must collapse when the structure is destroyed, not just shifted."""
        a = make_detailed(96, 64, seed=0)
        b = make_detailed(96, 64, seed=7)
        assert compute_ssim(a, b) < 0.5

    def test_score_decreases_as_noise_grows(self) -> None:
        detailed = make_detailed(96, 64, seed=0)
        rng = np.random.default_rng(0)
        scores = []
        for sigma in (2.0, 8.0, 24.0, 48.0):
            noisy = np.clip(
                detailed.astype(np.float64) + rng.normal(0, sigma, detailed.shape),
                0,
                255,
            ).astype(np.uint8)
            scores.append(compute_ssim(detailed, noisy))
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > 0.98
        assert scores[-1] < 0.5

    def test_structural_change_is_penalised_less_than_noise(self) -> None:
        """A small blur keeps structure; heavy noise destroys it."""
        import cv2

        detailed = make_detailed(96, 64, seed=0)
        blurred = cv2.GaussianBlur(detailed, (0, 0), 1.2)
        rng = np.random.default_rng(1)
        noisy = np.clip(
            detailed.astype(np.float64) + rng.normal(0, 24, detailed.shape), 0, 255
        ).astype(np.uint8)
        assert compute_ssim(detailed, blurred) > compute_ssim(detailed, noisy)

    def test_detects_the_scale_mismatch_a_normalisation_bug_would_hide(self) -> None:
        """Regression guard for the 8-bit vs [0, 1] constant mismatch.

        With C1/C2 defined for 8-bit data but images normalised to [0, 1], the
        stabilising terms dominate and every score collapses to ~1.0, which
        would let a badly broken protection pass a naive similarity check.
        """
        a = make_detailed(96, 64, seed=0)
        b = make_detailed(96, 64, seed=7)
        assert compute_ssim(a, b) < 0.6

    def test_shape_mismatch_raises(self, photo: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(photo, make_photo(10, 10))


class TestMetricsBundle:
    def test_identical_images_report_zero_difference(self, photo: np.ndarray) -> None:
        metrics = compute_metrics(photo, photo)
        assert metrics.linf_255 == 0.0
        assert metrics.linf_8bit == 0
        assert metrics.l2 == 0.0
        assert metrics.mean_abs == 0.0
        assert metrics.psnr_db == float("inf")
        assert metrics.ssim == pytest.approx(1.0, abs=1e-6)

    def test_linf_is_the_peak_absolute_difference(self, photo: np.ndarray) -> None:
        shifted = photo.copy()
        shifted[5, 7, :] = np.clip(shifted[5, 7, :].astype(np.int16) + 5, 0, 255)
        metrics = compute_metrics(photo, shifted.astype(np.uint8))
        assert metrics.linf_8bit == 5
        assert metrics.linf_255 == 5.0
        # Only 3 of 96*64 pixels changed by 5, so the mean must be far lower.
        assert metrics.mean_abs < 0.02

    def test_to_dict_exposes_every_field(self, photo: np.ndarray) -> None:
        payload = compute_metrics(photo, photo).to_dict()
        assert set(payload) == {
            "linf_255",
            "linf_8bit",
            "l2",
            "mean_abs",
            "psnr_db",
            "ssim",
        }

    def test_metrics_are_finite_and_ordered(self) -> None:
        photo = make_photo(96, 64)
        shifted = np.clip(photo.astype(np.int16) + 4, 0, 255).astype(np.uint8)
        metrics = compute_metrics(photo, shifted)
        assert 0 < metrics.mean_abs <= metrics.l2 <= metrics.linf_255
        assert metrics.psnr_db > 0
        assert -1.0 <= metrics.ssim <= 1.0

    def test_shape_mismatch_raises(self, photo: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_metrics(photo, make_photo(40, 40))
