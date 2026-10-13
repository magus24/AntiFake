"""Perturbation-bound tests: the core safety guarantee of the engine."""

from __future__ import annotations

import numpy as np
import pytest
from app.protection.config import (
    MAX_EPS_255,
    MAX_PGD_STEPS,
    ProtectionConfig,
    ProtectionConfigError,
)
from app.protection.engine import AdversarialProtectionEngine
from app.protection.target_model import resolve_device

from tests.helpers import TEST_TARGET, make_photo


def build(**overrides: object) -> AdversarialProtectionEngine:
    """Engine with random weights and 4 PGD steps, tuned per test."""
    values: dict[str, object] = {
        "eps": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "steps": 4,
        "seed": 0,
        "device": "cpu",
        "model_weights": "none",
    }
    values.update(overrides)
    return AdversarialProtectionEngine(
        ProtectionConfig(**values),  # type: ignore[arg-type]
        TEST_TARGET,
    )


class TestConfigValidation:
    def test_defaults_are_within_bounds(self) -> None:
        config = ProtectionConfig()
        assert 0 < config.eps <= MAX_EPS_255 / 255.0
        assert config.alpha <= config.eps
        assert 1 <= config.steps <= MAX_PGD_STEPS
        assert config.eps_255 == pytest.approx(8.0)

    @pytest.mark.parametrize("eps", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_eps(self, eps: float) -> None:
        with pytest.raises(ProtectionConfigError, match="eps"):
            ProtectionConfig(eps=eps)

    def test_rejects_eps_above_hard_ceiling(self) -> None:
        with pytest.raises(ProtectionConfigError, match="must not exceed"):
            ProtectionConfig(eps=64.0 / 255.0)

    def test_rejects_alpha_greater_than_eps(self) -> None:
        with pytest.raises(ProtectionConfigError, match="alpha"):
            ProtectionConfig(eps=2.0 / 255.0, alpha=4.0 / 255.0)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_rejects_non_positive_alpha(self, alpha: float) -> None:
        with pytest.raises(ProtectionConfigError, match="alpha"):
            ProtectionConfig(alpha=alpha)

    @pytest.mark.parametrize("steps", [0, -5, MAX_PGD_STEPS + 1])
    def test_rejects_out_of_range_steps(self, steps: int) -> None:
        with pytest.raises(ProtectionConfigError, match="steps"):
            ProtectionConfig(steps=steps)

    def test_rejects_unknown_loss_and_init(self) -> None:
        with pytest.raises(ProtectionConfigError, match="loss"):
            ProtectionConfig(loss="nope")  # type: ignore[arg-type]
        with pytest.raises(ProtectionConfigError, match="init"):
            ProtectionConfig(init="nope")  # type: ignore[arg-type]

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ProtectionConfigError, match="device"):
            ProtectionConfig(device="tpu")  # type: ignore[arg-type]

    def test_rejects_topk_below_one(self) -> None:
        with pytest.raises(ProtectionConfigError, match="target_topk"):
            ProtectionConfig(target_topk=0)

    def test_create_folds_out_of_range_steps(self) -> None:
        assert ProtectionConfig.create(steps=10_000).steps == MAX_PGD_STEPS
        assert ProtectionConfig.create(steps=0).steps == 1
        assert ProtectionConfig.create(steps=20).steps == 20

    def test_create_keeps_other_fields(self) -> None:
        config = ProtectionConfig.create(steps=5, eps=4.0 / 255.0, seed=11)
        assert (config.steps, config.eps_255, config.seed) == (5, 4.0, 11)

    def test_config_is_immutable(self) -> None:
        config = ProtectionConfig()
        with pytest.raises(AttributeError):
            config.eps = 0.5  # type: ignore[misc]


class TestPerturbationBounds:
    @pytest.mark.parametrize(
        ("width", "height"),
        [(96, 64), (64, 96), (33, 47), (300, 200), (17, 200)],
    )
    @pytest.mark.parametrize("eps_255", [4.0, 8.0, 16.0])
    def test_linf_never_exceeds_eps(
        self, width: int, height: int, eps_255: float
    ) -> None:
        engine = build(eps=eps_255 / 255.0, alpha=eps_255 / 255.0 / 4.0)
        original = make_photo(width, height)
        protected = engine.protect(original)

        linf = int(np.abs(original.astype(np.int16) - protected.astype(np.int16)).max())
        # One level of slack for the round-to-nearest in the uint8 conversion.
        assert linf <= int(eps_255) + 1, f"linf {linf} exceeded eps {eps_255}/255"

    def test_output_stays_in_valid_uint8_range(self) -> None:
        engine = build(eps=32.0 / 255.0, alpha=8.0 / 255.0)
        for base in (0, 1, 127, 254, 255):
            image = np.full((32, 32, 3), base, dtype=np.uint8)
            protected = engine.protect(image)
            assert protected.dtype == np.uint8
            assert protected.min() >= 0 and protected.max() <= 255

    def test_extreme_eps_does_not_wrap_around(self) -> None:
        engine = build(eps=32.0 / 255.0, alpha=8.0 / 255.0)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        protected = engine.protect(image)
        # No wraparound: every output pixel must be within eps of black.
        assert protected.max() <= 33

    def test_perturbation_is_not_all_zero(self) -> None:
        engine = build()
        original = make_photo()
        protected = engine.protect(original)
        assert not np.array_equal(original, protected)

    def test_perturbation_is_graded_not_saturated(self) -> None:
        """A protective perturbation uses far less than the full budget.

        If every pixel sat at +-eps the result would be a high-frequency
        pattern, which is both visible and a much weaker attack than the
        budget allows.
        """
        engine = build(eps=16.0 / 255.0, alpha=4.0 / 255.0)
        original = make_photo(96, 64)
        protected = engine.protect(original)
        abs_diff = np.abs(original.astype(np.int16) - protected.astype(np.int16))
        assert abs_diff.max() > 0
        assert abs_diff.mean() < abs_diff.max() * 0.9

    def test_l2_is_bounded(self) -> None:
        engine = build(eps=8.0 / 255.0, alpha=2.0 / 255.0)
        original = make_photo(96, 64)
        protected = engine.protect(original)
        diff = original.astype(np.float64) - protected.astype(np.float64)
        l2 = float(np.sqrt(np.mean(diff**2)))
        # L2 of an L-inf-bounded field cannot exceed eps.
        assert l2 <= engine.config.eps * 255.0 + 1.0

    def test_reported_meta_matches_actual_difference(self) -> None:
        engine = build(eps=8.0 / 255.0, alpha=2.0 / 255.0)
        original = make_photo(96, 64)
        result = engine.protect_with_meta(original)
        measured = np.abs(
            original.astype(np.int16) - result.image.astype(np.int16)
        ).max()
        assert result.meta.linf_255 == pytest.approx(float(measured), abs=0.01)
        assert result.meta.max_abs_pixel_delta == round(float(measured))
        assert result.meta.width == 96
        assert result.meta.height == 64
        assert result.meta.device == "cpu"
        assert result.meta.eps_255 == pytest.approx(8.0)


class TestDimensionPreservation:
    @pytest.mark.parametrize(
        ("width", "height"),
        [(96, 64), (1, 1), (7, 3), (224, 224), (513, 97), (1200, 900)],
    )
    def test_dimensions_are_preserved_exactly(self, width: int, height: int) -> None:
        engine = build()
        original = make_photo(width, height)
        protected = engine.protect(original)
        assert protected.shape == original.shape
        assert protected.shape[1] == width
        assert protected.shape[0] == height

    def test_engine_declares_dimension_preservation(self) -> None:
        assert build().preserves_dimensions is True
        assert build().describe()["preserves_dimensions"] == "True"

    def test_channels_are_preserved(self) -> None:
        engine = build()
        protected = engine.protect(make_photo(64, 64))
        assert protected.shape[2] == 3


class TestDeterminism:
    def test_same_seed_gives_byte_identical_output(self) -> None:
        original = make_photo(96, 64)
        first = build(seed=7).protect(original)
        second = build(seed=7).protect(original)
        np.testing.assert_array_equal(first, second)

    def test_repeated_calls_on_one_engine_are_identical(self) -> None:
        engine = build(seed=3)
        original = make_photo(96, 64)
        np.testing.assert_array_equal(engine.protect(original), engine.protect(original))

    def test_different_seeds_change_the_result(self) -> None:
        original = make_photo(96, 64)
        zero_init_a = build(seed=0, init="zeros").protect(original)
        zero_init_b = build(seed=99, init="zeros").protect(original)
        # With a zero init PGD is seed-independent by construction.
        np.testing.assert_array_equal(zero_init_a, zero_init_b)

        noise_a = build(seed=0, init="uniform").protect(original)
        noise_b = build(seed=99, init="uniform").protect(original)
        assert not np.array_equal(noise_a, noise_b)

    def test_uniform_init_is_seed_reproducible(self) -> None:
        original = make_photo(96, 64)
        np.testing.assert_array_equal(
            build(seed=5, init="uniform").protect(original),
            build(seed=5, init="uniform").protect(original),
        )

    def test_engine_does_not_mutate_its_input(self) -> None:
        original = make_photo(96, 64)
        snapshot = original.copy()
        build().protect(original)
        np.testing.assert_array_equal(original, snapshot)

    def test_engine_does_not_mutate_reused_input_across_calls(self) -> None:
        engine = build()
        original = make_photo(96, 64)
        first = engine.protect(original)
        second = engine.protect(original)
        np.testing.assert_array_equal(first, second)


class TestInputValidation:
    def test_rejects_float_array(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            build().protect(np.zeros((8, 8, 3), dtype=np.float32))

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            build().protect(np.zeros((8, 8), dtype=np.uint8))

    def test_rejects_rgba_array(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            build().protect(np.zeros((8, 8, 4), dtype=np.uint8))

    def test_rejects_non_array(self) -> None:
        with pytest.raises(ValueError, match="numpy array"):
            build().protect([[1, 2, 3]])  # type: ignore[arg-type]

    def test_rejects_zero_sized_array(self) -> None:
        with pytest.raises(ValueError, match="zero width or height"):
            build().protect(np.zeros((0, 8, 3), dtype=np.uint8))


class TestDeviceResolution:
    def test_cpu_is_always_available(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_auto_resolves_to_something_real(self) -> None:
        assert resolve_device("auto").type in ("cpu", "cuda")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_device("gpu")

    def test_cuda_request_fails_loudly_when_absent(self) -> None:
        import torch

        if torch.cuda.is_available():
            assert resolve_device("cuda").type == "cuda"
        else:
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                resolve_device("cuda")


class TestStepCountMonotonicity:
    """The attack must actually optimise its objective, not just add noise."""

    @staticmethod
    def _top_logit(engine: AdversarialProtectionEngine, image: np.ndarray) -> float:
        import torch
        from app.protection.engine import _to_tensor

        with torch.no_grad():
            logits = engine.model(_to_tensor(image, engine.device))
        return float(logits.max())

    def test_perturbation_reduces_the_target_models_confidence(self) -> None:
        engine = build(steps=8)
        original = make_photo(96, 64)
        assert self._top_logit(engine, engine.protect(original)) < self._top_logit(
            engine, original
        )

    def test_more_steps_reduce_the_target_loss_further(self) -> None:
        original = make_photo(96, 64)
        short = build(steps=1)
        long = build(steps=12)
        assert self._top_logit(long, long.protect(original)) < self._top_logit(
            short, short.protect(original)
        )

    def test_zero_step_epsilon_run_is_impacted_by_more_steps(self) -> None:
        """A stronger budget should shrink the max logit more often than not."""
        original = make_photo(96, 64)
        deltas = [
            self._top_logit(engine, engine.protect(original))
            - self._top_logit(engine, original)
            for engine in (build(steps=1), build(steps=4), build(steps=12))
        ]
        assert deltas[0] < 0
        assert deltas[-1] <= deltas[0] + 1e-4

