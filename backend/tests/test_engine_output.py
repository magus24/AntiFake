"""Engine output-generation tests, plus the one opt-in real-weights check."""

from __future__ import annotations

import io

import numpy as np
import pytest
from app.protection.config import ProtectionConfig, TargetModelConfig
from app.protection.engine import AdversarialProtectionEngine
from app.protection.image_io import encode_image
from app.protection.metrics import compute_metrics
from PIL import Image

from tests.helpers import TEST_TARGET, decode, make_photo


def build(**overrides: object) -> AdversarialProtectionEngine:
    """Engine with random weights and 4 PGD steps, tuned per test."""
    config_kwargs: dict[str, object] = {
        "eps": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "steps": 4,
        "seed": 0,
        "device": "cpu",
        "model_weights": "none",
    }
    config_kwargs.update(overrides)
    return AdversarialProtectionEngine(
        ProtectionConfig(**config_kwargs),  # type: ignore[arg-type]
        TEST_TARGET,
    )


class TestOutputGeneration:
    def test_output_is_uint8_rgb(self) -> None:
        result = build().protect_with_meta(make_photo(96, 64))
        assert isinstance(result.image, np.ndarray)
        assert result.image.dtype == np.uint8
        assert result.image.ndim == 3 and result.image.shape[2] == 3
        assert result.image.flags["C_CONTIGUOUS"] is True

    def test_output_is_writable(self) -> None:
        protected = build().protect(make_photo(64, 64))
        protected[0, 0] = 123
        assert protected[0, 0, 0] == 123

    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    def test_output_encodes_and_reopens(self, image_format: str) -> None:
        original = make_photo(96, 64)
        protected = build().protect(original)
        data = encode_image(protected, image_format)
        assert decode(data).shape == original.shape
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()

    def test_png_round_trip_is_bit_exact(self) -> None:
        """A PNG round trip must not add noise of its own.

        The only difference between input and PNG-decoded output is the
        attack, so the measured bound stays the engine's own eps.
        """
        protected = build().protect(make_photo(96, 64))
        assert np.array_equal(decode(encode_image(protected, "PNG")), protected)

    def test_meta_reports_processing_time(self) -> None:
        result = build().protect_with_meta(make_photo(96, 64))
        assert result.meta.attack_ms > 0.0
        assert np.isfinite(result.meta.attack_ms)

    def test_meta_reports_dimensions_and_target(self) -> None:
        result = build().protect_with_meta(make_photo(80, 50))
        assert (result.meta.width, result.meta.height) == (80, 50)
        assert "resnet18" in result.meta.target_model
        assert result.meta.engine == "pgd-linf-resnet18"
        assert result.meta.seed == 0
        assert result.meta.steps == 4

    def test_describe_reports_the_narrow_threat_model(self) -> None:
        described = build().describe()
        assert described["engine"] == "pgd-linf-resnet18"
        assert "resnet18" in described["target_model"]
        assert described["device"] == "cpu"
        assert described["eps_255"] == "8.000"

    @pytest.mark.parametrize(
        ("width", "height"), [(3, 3), (5, 300), (300, 5), (224, 224), (640, 480)]
    )
    def test_handle_varied_image_shapes(self, width: int, height: int) -> None:
        engine = build()
        original = make_photo(width, height)
        assert engine.protect(original).shape == original.shape

    @pytest.mark.parametrize("value", [0, 1, 127, 128, 254, 255])
    def test_uniform_images_do_not_crash(self, value: int) -> None:
        engine = build()
        flat = np.full((40, 40, 3), value, dtype=np.uint8)
        protected = engine.protect(flat)
        assert protected.shape == flat.shape
        assert protected.dtype == np.uint8
        assert protected.min() >= 0 and protected.max() <= 255

    @pytest.mark.parametrize("loss", ["cw_topk", "cw_top1"])
    def test_loss_variants_agree_on_dimensions(self, loss: str) -> None:
        engine = build(loss=loss)
        assert engine.protect(make_photo(64, 64)).shape == (64, 64, 3)


class TestRealPretrainedWeights:
    """Opt-in check that the real ImageNet pipeline behaves as advertised.

    Skipped by default so the suite runs offline. Enable with
    ``pytest -m network``.
    """

    @pytest.mark.network
    @pytest.mark.slow
    def test_protects_against_real_imagenet_resnet18(self) -> None:
        from app.protection.target_model import available_weights

        assert "imagenet1k_v1" in available_weights("resnet18")

        engine = AdversarialProtectionEngine(
            ProtectionConfig(
                eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=25, seed=0, device="cpu"
            ),
            TargetModelConfig(name="resnet18", weights="imagenet1k_v1"),
        )
        original = make_photo(320, 240)
        protected = engine.protect(original)

        assert protected.shape == original.shape
        assert protected.dtype == np.uint8

        metrics = compute_metrics(original, protected)
        # The hard guarantee: the L-inf budget is respected exactly.
        assert metrics.linf_255 <= 9.0
        # The perturbation is graded, so PSNR stays high.
        assert metrics.psnr_db > 30.0
        # SSIM is a *structural* metric and PGD produces a deliberately
        # high-frequency field, so it drops further than human perception would
        # suggest. Measured ~0.94 at eps=8/255 on a 320x240 image. This is not
        # a bug and it is not tuned away -- it is the honest cost of a
        # targeted perturbation on a smooth source image.
        assert 0.85 < metrics.ssim <= 1.0

        with Image.open(io.BytesIO(encode_image(protected, "PNG"))) as handle:
            handle.load()

    @pytest.mark.network
    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (320, 240),
            (334, 414),
            (400, 300),
            (512, 512),
        ],
    )
    def test_perturbation_actually_reduces_real_model_confidence(
        self, width: int, height: int
    ) -> None:
        """The point of the whole exercise, checked against real weights.

        The bar is deliberately a *margin*, not merely "confidence went down".
        A perturbation optimised in the wrong coordinate frame survives a weak
        ``attacked < clean`` check while doing nothing useful: measured on a
        334x414 source, a stretch-the-crop-to-the-whole-image bug left the top-1
        probability at 130% of its original value. Aspect ratio matters because
        the pipeline's centre crop covers a different share of the frame in a
        portrait than in a landscape, so the cases are checked separately.
        """
        import torch
        from app.protection.engine import _to_tensor

        target = TargetModelConfig(name="resnet18", weights="imagenet1k_v1")
        engine = AdversarialProtectionEngine(
            ProtectionConfig(
                eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=25, seed=0, device="cpu"
            ),
            target,
        )
        original = make_photo(width, height)
        protected = engine.protect(original)

        def top1(image: np.ndarray) -> float:
            with torch.no_grad():
                logits = engine.model(_to_tensor(image, engine.device))
            return float(torch.softmax(logits, dim=1)[0].max())

        clean = top1(original)
        attacked = top1(protected)
        retained = attacked / clean if clean else 1.0

        assert attacked < clean, (
            f"perturbation did not reduce top-1 confidence ({attacked} vs {clean})"
        )
        # Measured ~2% retained after the coordinate-frame fix; 25% leaves
        # generous headroom while still failing loudly on a diluted attack.
        assert retained < 0.25, (
            f"perturbation only reduced top-1 confidence to {retained:.1%} of its "
            f"original value on a {width}x{height} image "
            f"({clean:.4f} -> {attacked:.4f}); the pattern is not surviving "
            "the pipeline's own resize and crop"
        )

    @pytest.mark.network
    @pytest.mark.slow
    def test_is_reproducible_with_real_weights(self) -> None:
        original = make_photo(320, 240)
        first = AdversarialProtectionEngine(
            ProtectionConfig(eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=5, device="cpu"),
            TargetModelConfig(name="resnet18", weights="imagenet1k_v1"),
        ).protect(original)
        second = AdversarialProtectionEngine(
            ProtectionConfig(eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=5, device="cpu"),
            TargetModelConfig(name="resnet18", weights="imagenet1k_v1"),
        ).protect(original)
        np.testing.assert_array_equal(first, second)
