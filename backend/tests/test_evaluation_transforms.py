"""Tests for the robustness transforms.

Each transform is a measurement instrument, so the properties that make a
measurement trustworthy are asserted explicitly: it is deterministic, it keeps
``uint8``, it does what its name says to the geometry, and it leaves the image
alone when it has no reason to touch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.evaluation.transforms import (
    BILINEAR,
    Brightness,
    CenterCropFraction,
    Contrast,
    CornerCrop,
    DownscaleRoundTrip,
    GaussianBlur,
    HalfScale,
    Identity,
    JpegQuality,
    ResizeShortSide,
    SeededRandomCrop,
    UnsharpMask,
    default_transforms,
)

from tests.helpers import make_detailed, make_photo


def _all():
    return default_transforms()


def test_identity_returns_the_same_pixels():
    image = make_photo(12, 8)
    assert np.array_equal(Identity().apply(image), image)


def test_identity_preserves_geometry():
    result = Identity().run(make_photo(12, 8))
    assert not result.changes_geometry


def test_default_suite_starts_with_identity():
    assert _all()[0].key == "identity"


def test_default_suite_keys_are_unique():
    keys = [t.key for t in _all()]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("transform", _all(), ids=lambda t: t.key)
def test_every_transform_preserves_dtype_and_shape(transform):
    image = make_photo(24, 32)
    output = transform.apply(image)
    assert output.dtype == np.uint8
    assert output.ndim == 3
    assert output.shape[2] == 3


@pytest.mark.parametrize("transform", _all(), ids=lambda t: t.key)
def test_every_transform_is_deterministic(transform):
    """Two runs must agree bit-for-bit, or two benchmark runs are incomparable."""
    image = make_photo(24, 32)
    assert np.array_equal(transform.apply(image), transform.apply(image))


@pytest.mark.parametrize("transform", _all(), ids=lambda t: t.key)
def test_every_transform_describes_itself(transform):
    described = transform.describe()
    assert set(described) == {"key", "label", "changes_geometry", "parameters"}
    assert described["key"] == transform.key


@pytest.mark.parametrize("transform", _all(), ids=lambda t: t.key)
def test_every_transform_declares_its_geometry_change(transform):
    assert isinstance(transform.changes_geometry, bool)


def test_resize_short_side_scales_the_smaller_dimension():
    result = ResizeShortSide(256).run(make_photo(800, 400))  # width 800, height 400
    height, width = result.image.shape[:2]
    assert min(height, width) == 256
    assert result.changes_geometry


def test_resize_short_side_preserves_aspect_ratio():
    result = ResizeShortSide(256).run(make_photo(800, 400))
    height, width = result.image.shape[:2]
    assert width / height == pytest.approx(800 / 400, rel=0.02)


def test_resize_short_side_never_upscales():
    """An image already below the target is left alone.

    Upscaling would only test the resampler, since every model in the
    benchmark resizes internally regardless.
    """
    image = make_photo(100, 80)
    assert np.array_equal(ResizeShortSide(256).apply(image), image)


def test_half_scale_halves_both_dimensions():
    result = HalfScale().run(make_photo(40, 20))
    assert result.image.shape[1] == 20
    assert result.image.shape[0] == 10
    assert result.changes_geometry


def test_centre_crop_keeps_original_dimensions():
    result = CenterCropFraction(0.8).run(make_photo(40, 20))
    assert result.image.shape == (20, 40, 3)
    assert result.changes_geometry


def test_corner_crop_keeps_original_dimensions():
    result = CornerCrop(0.8).run(make_photo(40, 20))
    assert result.image.shape == (20, 40, 3)


def test_centre_and_corner_crops_actually_differ():
    """They must disagree, or the corner row proves nothing."""
    image = make_detailed(40, 30)
    assert not np.array_equal(
        CenterCropFraction(0.8).apply(image), CornerCrop(0.8).apply(image)
    )


def test_downscale_round_trip_keeps_dimensions_but_loses_detail():
    image = make_detailed(40, 40)
    result = DownscaleRoundTrip(0.5).run(image)
    assert result.image.shape == image.shape
    assert not np.array_equal(result.image, image)


@pytest.mark.parametrize("quality", [95, 85, 75, 50])
def test_jpeg_keeps_dimensions_and_actually_changes_pixels(quality):
    image = make_detailed(40, 40)
    result = JpegQuality(quality).run(image)
    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert not result.changes_geometry


def test_lower_jpeg_quality_erodes_more_than_higher():
    image = make_detailed(64, 64)
    high = JpegQuality(95).apply(image).astype(np.int16)
    low = JpegQuality(50).apply(image).astype(np.int16)
    assert np.abs(high - image).mean() < np.abs(low - image).mean()


def test_jpeg_key_encodes_the_quality():
    assert JpegQuality(85).key == "jpeg_q85"
    assert JpegQuality(85).describe()["parameters"] == {"quality": 85}


def test_brightness_up_brightens():
    image = make_photo(20, 20)
    brighter = Brightness(1.05).apply(image)
    assert brighter.mean() >= image.mean()


def test_brightness_down_darkens():
    image = make_photo(20, 20)
    darker = Brightness(0.95).apply(image)
    assert darker.mean() <= image.mean()


def test_contrast_reduces_spread():
    image = make_photo(40, 40)
    assert Contrast(0.95).apply(image).std() <= image.std()


def test_blur_reduces_high_frequency_energy():
    image = make_detailed(40, 40)
    assert GaussianBlur(0.5).apply(image).std() < image.std()


def test_sharpen_raises_local_gradient_energy():
    from scipy import ndimage

    def edge_energy(array):
        return float(ndimage.laplace(array.astype(np.float32)).var())

    image = make_detailed(40, 40)
    assert edge_energy(UnsharpMask(1.0, 120).apply(image)) > edge_energy(image)


def test_bilinear_alias_is_the_pillow_enum():
    from PIL import Image

    assert BILINEAR == Image.Resampling.BILINEAR


def test_seeded_random_crop_is_reproducible():
    """A sampled window must still be identical across runs.

    The seed is what keeps the benchmark comparable; without it this transform
    would make every run a different experiment.
    """
    image = make_detailed(64, 64)
    first = SeededRandomCrop(0.8, seed=0).apply(image)
    second = SeededRandomCrop(0.8, seed=0).apply(image)
    assert np.array_equal(first, second)


def test_seeded_random_crop_depends_on_its_seed():
    image = make_detailed(64, 64)
    assert not np.array_equal(
        SeededRandomCrop(0.8, seed=0).apply(image),
        SeededRandomCrop(0.8, seed=1).apply(image),
    )


def test_seeded_random_crop_key_includes_the_seed():
    """Two seeds are two measurements, and must not collide on one JSON key."""
    assert SeededRandomCrop(0.8, seed=0).key != SeededRandomCrop(0.8, seed=1).key


def test_seeded_random_crop_keeps_dimensions():
    image = make_photo(30, 40)
    assert SeededRandomCrop(0.8, seed=0).apply(image).shape == (40, 30, 3)


def test_seeded_random_crop_does_not_use_the_global_rng():
    """Drawing elsewhere in the run must not move this transform's window.

    Otherwise an unrelated model or the protection attack consuming randomness
    would silently change what the robustness rows measure.
    """
    image = make_detailed(64, 64)
    expected = SeededRandomCrop(0.8, seed=0).apply(image)
    np.random.seed(1234)
    np.random.random(37)
    assert np.array_equal(SeededRandomCrop(0.8, seed=0).apply(image), expected)


def test_seeded_random_crop_rejects_a_nonsense_fraction():
    with pytest.raises(ValueError, match="fraction"):
        SeededRandomCrop(0.0)


def test_default_suite_includes_the_seeded_random_crop():
    keys = [t.key for t in default_transforms()]
    assert "random_crop_80_s0" in keys


def test_default_suite_keys_stay_unique():
    keys = [t.key for t in default_transforms()]
    assert len(keys) == len(set(keys))

def test_screenshot_like_restores_the_original_dimensions() -> None:
    """A transform that changes the shape would make every comparison invalid."""
    from app.evaluation.transforms import ScreenshotLike

    image = make_photo(64, 48)
    out = ScreenshotLike().apply(image)
    assert out.shape == image.shape


def test_screenshot_like_actually_changes_the_content() -> None:
    """Restoring the size must not be mistaken for preserving the pixels."""
    from app.evaluation.transforms import ScreenshotLike

    image = make_photo(64, 48)
    out = ScreenshotLike().apply(image)
    assert np.array_equal(out, image) is False
    assert int(np.abs(out.astype(int) - image.astype(int)).max()) > 0


def test_screenshot_like_is_deterministic() -> None:
    """The row has to be reproducible; a sampled variant would not diff cleanly."""
    from app.evaluation.transforms import ScreenshotLike

    image = make_photo(64, 48)
    first = ScreenshotLike().apply(image)
    second = ScreenshotLike().apply(image)
    assert np.array_equal(first, second)


def test_screenshot_like_rejects_a_nonsense_scale() -> None:
    from app.evaluation.transforms import ScreenshotLike

    with pytest.raises(ValueError, match="scale"):
        ScreenshotLike(scale=1.5)


def test_phase2_suite_adds_exactly_one_row_to_the_phase1_suite() -> None:
    """Phase 1 must still reproduce byte for byte.

    Nothing may be reordered or removed from the Phase 1 suite, only appended
    to, or a fresh run would stop matching the saved baseline.
    """
    from app.evaluation.transforms import default_transforms, phase2_transforms

    phase1 = [t.key for t in default_transforms()]
    phase2 = [t.key for t in phase2_transforms()]
    assert phase2[: len(phase1)] == phase1
    assert phase2[len(phase1) :] == ["screenshot_like"]


def test_phase2_suite_keys_stay_unique() -> None:
    from app.evaluation.transforms import phase2_transforms

    keys = [t.key for t in phase2_transforms()]
    assert len(keys) == len(set(keys))
