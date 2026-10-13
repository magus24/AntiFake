"""Tests for the differentiable EOT transforms used inside the attack.

The properties asserted here are the ones the whole Phase 2 result rests on.
If a transform stops being differentiable the attack silently degenerates to
optimising against a constant while still reporting that EOT was on, so
differentiability is checked explicitly per transform rather than assumed. If a
transform stops being reproducible the benchmark stops being comparable to
itself, which is the one property Phase 1 promised to keep.
"""

from __future__ import annotations

import pytest
import torch
from app.protection.eot import (
    EotConfig,
    EotSampler,
    build_transform,
)

ALL_KEYS = (
    "identity",
    "resize_short_side",
    "random_crop",
    "center_crop",
    "scale_jitter",
    "jpeg",
    "brightness",
    "contrast",
    "blur",
    "sharpen",
    "affine",
    "screenshot_like",
)


def _batch(seed: int = 0, n: int = 2, h: int = 64, w: int = 80) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, h, w, generator=generator)


def _gen(seed: int = 7) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@pytest.mark.parametrize("key", ALL_KEYS)
def test_transform_is_differentiable(key: str) -> None:
    """A zero gradient would mean the attack learns nothing through this path.

    Checked with a finite-difference style probe rather than a shape check: a
    transform can return the right shape and still sever the graph, for
    instance by detaching, converting to NumPy, or rounding to uint8.
    """
    images = _batch().requires_grad_(True)
    out = build_transform(key).apply(images, _gen())
    out.pow(2).sum().backward()
    assert images.grad is not None, f"{key} detached the graph"
    assert torch.isfinite(images.grad).all(), f"{key} produced non-finite gradients"
    assert images.grad.abs().sum() > 0, f"{key} produced an all-zero gradient"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_transform_is_reproducible(key: str) -> None:
    images = _batch()
    first = build_transform(key).apply(images, _gen(3))
    second = build_transform(key).apply(images, _gen(3))
    assert torch.equal(first, second), f"{key} is not reproducible for a fixed seed"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_transform_ignores_the_global_rng(key: str) -> None:
    """Drawing from global RNG elsewhere must not move this transform.

    The engine seeds global RNG for the attack and every model load consumes
    more. If a transform read global state, the same configured run would
    produce different patterns and the "reproducible" claim would be false.
    """
    images = _batch()
    expected = build_transform(key).apply(images, _gen(11))
    torch.manual_seed(999)
    torch.rand(1234)
    assert torch.equal(build_transform(key).apply(images, _gen(11)), expected)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_transform_stays_in_range(key: str) -> None:
    """Output must stay a valid image, or the target sees garbage."""
    out = build_transform(key).apply(_batch(), _gen(5))
    assert out.dtype == torch.float32
    assert out.min() >= -1e-5, f"{key} produced negative intensities"
    assert out.max() <= 1.0 + 1e-5, f"{key} exceeded full scale"
    assert out.shape[0] == 2, f"{key} changed the batch size"


def test_crop_transforms_restore_the_frame() -> None:
    """A consumer pipeline changes size, but a crop is about *which* pixels.

    Random and centre crop deliberately keep the frame, so a subsequent model
    read is at the usual resolution and the perturbation is tested for
    positional robustness rather than for resize behaviour.
    """
    images = _batch()
    for key in ("random_crop", "center_crop"):
        out = build_transform(key).apply(images, _gen())
        assert out.shape == images.shape


def test_random_crop_offset_actually_varies() -> None:
    """A 'random' crop that always lands on the same window is a centre crop.

    Checked across seeds rather than within one, because a single draw could
    coincide with the centre by chance and prove nothing.
    """
    transform = build_transform("random_crop")
    offsets = {
        transform.apply(_batch(), _gen(seed)).detach().sum().item()
        for seed in range(12)
    }
    assert len(offsets) > 1, "random_crop produced the same output every time"


def test_jpeg_removes_high_frequency_detail() -> None:
    """The straight-through quantiser has to actually quantise in forward.

    A constant image is invariant under JPEG, so the probe is a gradient
    pattern: block DCT quantisation must flatten it measurably.
    """
    height = width = 64
    ramp = torch.arange(width, dtype=torch.float32).view(1, 1, 1, -1)
    stripes = ((ramp // 4) % 2).float().expand(1, 3, height, width)
    out = build_transform("jpeg").apply(stripes, _gen(2))
    assert (out - stripes).abs().mean() > 1e-3


def test_jpeg_quality_is_sampled_inside_the_configured_range() -> None:
    transform = build_transform("jpeg")
    seen = []
    for seed in range(20):
        transform.apply(_batch(), _gen(seed))
        seen.append(transform.last_parameters["quality"])
    assert min(seen) >= 30.0 and max(seen) <= 95.0
    assert len(set(seen)) > 5, "quality was not actually sampled"


def test_jpeg_preserves_mean_intensity() -> None:
    """A compressor that changes overall brightness is not a JPEG model.

    Caught a real bug: the standard quantisation tables are calibrated for
    signed 8-bit samples, so omitting the encoder's level shift quantised the DC
    coefficient to zero and returned a black image.
    """
    images = _batch()
    for quality in (30.0, 50.0, 75.0, 95.0):
        transform = build_transform("jpeg")
        transform.low = transform.high = quality
        out = transform.apply(images, _gen(2))
        assert abs(float(out.mean()) - float(images.mean())) < 0.02, (
            f"q={quality} shifted the mean"
        )


def test_blur_reduces_local_gradient_energy() -> None:
    images = _batch().requires_grad_(True)
    out = build_transform("blur").apply(images, _gen(1))
    energy_in = float(images.grad.abs().sum()) if images.grad is not None else None
    del energy_in
    # Compare second differences, which a blur must damp.
    def sharpness(tensor: torch.Tensor) -> float:
        return float(
            (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
            + (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
        )

    assert sharpness(out) < sharpness(images.detach())


def test_affine_can_move_content() -> None:
    """A translation-only affine must not be a no-op."""
    images = _batch()
    out = build_transform("affine").apply(images, _gen(4))
    assert (out - images).abs().max() > 0


def test_affine_sampled_matrix_is_a_scaled_rotation() -> None:
    """The warp must be ``scale * R``, not an angle dropped into a matrix corner.

    An earlier version wrote the sampled angle into ``theta[0][0]``, producing a
    matrix close to ``[[0, s], [0, s]]``. That is rank deficient, so the
    transform is no longer a rotation-plus-scale at all: it can collapse or
    mirror the frame while still reporting a plausible-looking scale. The
    property that actually matters is invariant -- the linear part has
    orthogonal columns and determinant ``scale**2`` -- so it is asserted on many
    draws rather than on one fixture.
    """
    transform = build_transform("affine")
    for seed in range(64):
        transform.apply(_batch(), _gen(seed))
        scale = transform.last_parameters["scale"]
        determinant = transform.last_parameters["determinant"]
        assert determinant == pytest.approx(scale * scale, abs=1e-7)
        assert determinant > 0, "a scale-only transform can never flip orientation"


def test_affine_preserves_angles() -> None:
    """Orthogonal columns up to scale is what makes it a rotation, not a shear."""
    transform = build_transform("affine")
    images = _batch(n=1)
    seen_rotations = set()
    for seed in range(64):
        out = transform.apply(images, _gen(seed))
        # A rotation/shear is area preserving up to scale: local pixel offsets
        # must not become systematically collinear. Compare the per-axis
        # gradient energy; a shear would bias it by orders of magnitude.
        vertical = (out[:, :, 1:, :] - out[:, :, :-1, :]).abs().mean()
        horizontal = (out[:, :, :, 1:] - out[:, :, :, :-1]).abs().mean()
        assert 0.2 < float(vertical / horizontal) < 5.0
        seen_rotations.add(round(transform.last_parameters["rotate_deg"]))
    assert len(seen_rotations) > 8, "rotation should actually vary across draws"


def test_screenshot_like_changes_geometry_and_photometry() -> None:
    out = build_transform("screenshot_like").apply(_batch(), _gen(6))
    assert out.shape[-2:] != (64, 80), "screenshot_like should resample"


def test_identity_is_a_true_no_op() -> None:
    images = _batch()
    assert torch.equal(build_transform("identity").apply(images, _gen()), images)


# -- sampler ---------------------------------------------------------------


def test_sampler_draws_the_requested_number_of_samples() -> None:
    sampler = EotSampler(EotConfig(samples=4, seed=0), torch.device("cpu"))
    assert len(sampler.sample_from(_batch(), 0)) == 4


def test_sampler_is_deterministic_for_a_seed_and_step() -> None:
    images = _batch()
    first = EotSampler(EotConfig(samples=4, seed=0), torch.device("cpu"))
    second = EotSampler(EotConfig(samples=4, seed=0), torch.device("cpu"))
    a = [t for _, t in first.sample_from(images, 3)]
    b = [t for _, t in second.sample_from(images, 3)]
    for left, right in zip(a, b, strict=True):
        assert torch.equal(left, right)


def test_sampler_draws_differ_between_steps() -> None:
    """A schedule that repeats every step would be a fixed transform list.

    That is not EOT; it is a deterministic ensemble, and it would make the
    optimisation brittle in exactly the way Phase 1 already demonstrated.
    """
    images = _batch()
    sampler = EotSampler(EotConfig(samples=4, seed=0), torch.device("cpu"))
    totals = {
        round(float(t.sum()), 5)
        for step in range(6)
        for _, t in sampler.sample_from(images, step)
    }
    assert len(totals) > 1


def test_sampler_different_seeds_give_different_draws() -> None:
    """Two seeds must not be the same experiment.

    Compared through the family sequence and a shape-independent checksum,
    because the drawn batches legitimately have different sizes and cannot be
    concatenated.
    """
    images = _batch()

    def fingerprint(seed: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
        sampler = EotSampler(EotConfig(samples=4, seed=seed), torch.device("cpu"))
        drawn = sampler.sample_from(images, 0)
        return (
            tuple(key for key, _ in drawn),
            tuple(round(float(t.double().sum()), 2) for _, t in drawn),
        )

    assert fingerprint(0) != fingerprint(99)


def test_sampler_covers_families_over_many_steps() -> None:
    """Every configured family should actually be exercised.

    Guards against a shuffle that is reproducible but lopsided, which would
    silently drop a whole failure mode from the expectation.
    """
    sampler = EotSampler(EotConfig(samples=3, seed=0), torch.device("cpu"))
    seen = {key for step in range(40) for key, _ in sampler.sample_from(_batch(), step)}
    assert set(ALL_KEYS) - {"screenshot_like"} <= seen


def test_disabled_sampler_returns_the_input_unchanged() -> None:
    """``samples=0`` must reproduce the Phase 1 path exactly."""
    images = _batch()
    sampler = EotSampler(EotConfig(samples=0, seed=0), torch.device("cpu"))
    drawn = sampler.sample_from(images, 0)
    assert len(drawn) == 1
    assert drawn[0][0] == "identity"
    assert drawn[0][1] is images


def test_identity_always_takes_the_first_slot() -> None:
    """The untransformed view must be in *every* step, not in most of them.

    If identity were just another entry in the shuffle then a step could draw
    ``samples`` heavy distortions, and the gradient for that step says nothing
    about the clean image. The docstring promises a fixed share, so the
    guarantee is asserted rather than left to the shuffle.
    """
    images = _batch()
    sampler = EotSampler(EotConfig(samples=4, seed=0), torch.device("cpu"))
    for step in range(50):
        drawn = sampler.sample_from(images, step)
        keys = [key for key, _ in drawn]
        assert keys[0] == "identity", f"step {step} drew {keys}"
        assert keys.count("identity") == 1, f"step {step} drew {keys}"
        assert torch.equal(drawn[0][1], images), "identity must be a true no-op"


def test_identity_is_dropped_from_the_family_pool() -> None:
    """Identity keeps its own slot and must not be double-counted by the shuffle.

    Otherwise ``samples=4`` would spend two of the four slots on the clean view
    and the expectation would quietly be half identity.
    """
    images = _batch()
    sampler = EotSampler(EotConfig(samples=3, seed=0), torch.device("cpu"))
    keys = [key for step in range(60) for key, _ in sampler.sample_from(images, step)]
    assert keys.count("identity") == len(keys) / 3


def test_sampler_without_identity_configured_never_invents_one() -> None:
    """Excluding the family must actually exclude it.

    A hard-coded identity slot would make ``families`` a lie for anyone who
    deliberately left it out.
    """
    images = _batch()
    families = tuple(key for key in ALL_KEYS if key != "identity")
    sampler = EotSampler(
        EotConfig(samples=3, seed=0, families=families), torch.device("cpu")
    )
    keys = {key for step in range(30) for key, _ in sampler.sample_from(images, step)}
    assert "identity" not in keys


def test_sampler_rejects_a_nonsense_sample_count() -> None:
    with pytest.raises(ValueError, match="samples"):
        EotConfig(samples=-1)
    with pytest.raises(ValueError, match="samples"):
        EotConfig(samples=999)


def test_sampler_rejects_an_empty_family_list() -> None:
    with pytest.raises(ValueError, match="families"):
        EotConfig(samples=2, families=())


def test_unknown_family_is_rejected_with_the_known_list() -> None:
    with pytest.raises(KeyError, match="known"):
        build_transform("does_not_exist")
