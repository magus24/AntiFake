"""Reproducible post-processing transforms for robustness measurement.

A protection that only works on the exact bytes it produced is not a
protection. This module applies a fixed, declared set of mild transforms to
the protected image and re-measures, which answers the question a user
actually cares about: does this still work after the image has been through a
social network, a screenshot, or a messaging app.

Every transform is deterministic and re-runnable. Most are fixed functions;
:class:`SeededRandomCrop` samples a crop window, but derives it from a declared
seed rather than global RNG state, so a given run always crops the same way and
two runs stay comparable. That is a deliberate trade -- an EOT-style random
transform would show a more flattering average, but it would also make the
benchmark irreproducible, and Phase 1's job is measurement, not optimisation.

Transforms change geometry as well as pixels. The runner therefore applies each
transform to the original as well, and measures the protected result against
that matching baseline, rather than assuming the shapes still line up.
"""

from __future__ import annotations

import abc
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.protection.base import RGB

ArrayTransform = Callable[[np.ndarray], np.ndarray]

#: Pillow renamed the resampling filters; the module-level alias keeps every
#: call site short and makes the filter choice greppable.
BILINEAR = Image.Resampling.BILINEAR


def _resize_to(pil: Image.Image, size: tuple[int, int]) -> RGB:
    """Bilinear resize and read back as a ``uint8`` array."""
    return np.array(pil.resize(size, BILINEAR), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class TransformResult:
    """One transformed image plus how it should be described in the report.

    Attributes:
        key: Stable identifier used as the JSON key.
        image: Transformed ``uint8`` RGB array.
        changes_geometry: ``True`` when width or height changed, which means
            visual metrics must be computed against a matching resize of the
            original rather than against the original itself.
    """

    key: str
    label: str
    image: RGB
    changes_geometry: bool
    parameters: dict[str, Any]


class EvaluationTransform(abc.ABC):
    """One named, reproducible transform."""

    key: str = "abstract"
    label: str = "abstract"
    changes_geometry: bool = False

    @abc.abstractmethod
    def apply(self, image: RGB) -> RGB:
        """Return the transformed image."""

    def describe(self) -> dict[str, Any]:
        """JSON-ready description, for the reproducibility block."""
        return {
            "key": self.key,
            "label": self.label,
            "changes_geometry": self.changes_geometry,
            "parameters": self.parameters(),
        }

    def parameters(self) -> dict[str, Any]:
        """Parameters that define this transform, for the report."""
        return {}

    def run(self, image: RGB) -> TransformResult:
        """Apply and wrap in a :class:`TransformResult`."""
        return TransformResult(
            key=self.key,
            label=self.label,
            image=self.apply(image),
            changes_geometry=self.changes_geometry,
            parameters=self.parameters(),
        )


class Identity(EvaluationTransform):
    """No change. The reference row every other transform is read against."""

    key = "identity"
    label = "Identity"

    def apply(self, image: RGB) -> RGB:
        """Return the image untouched."""
        return image


class ResizeShortSide(EvaluationTransform):
    """Downscale the shortest side, which is what every vision model does.

    Never upscales: an image already smaller than ``side`` is returned
    untouched. Every model in the benchmark resizes internally anyway, so
    enlarging the file on disk would only test the resampler, not the
    protection.
    """

    key = "resize_short_side"
    label = "Resize shortest side to 256"
    changes_geometry = True

    def __init__(self, side: int = 256) -> None:
        """Set the target shortest side."""
        self.side = side

    def apply(self, image: RGB) -> RGB:
        """Scale the shortest side to ``side``, preserving aspect ratio.

        Returns the input unchanged when it is already at or below ``side``.
        """
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        scale = self.side / float(min(width, height))
        if scale >= 1.0:
            return image
        new = (max(1, round(width * scale)), max(1, round(height * scale)))
        return _resize_to(pil, new)

    def parameters(self) -> dict[str, Any]:
        """Target side."""
        return {"short_side": self.side}


class HalfScale(EvaluationTransform):
    """Halve both dimensions: a hard resize, well beyond any model default."""

    key = "scale_0.5"
    label = "Downscale to 50%"
    changes_geometry = True

    def apply(self, image: RGB) -> RGB:
        """Scale to 50% on each axis."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        new = (max(1, width // 2), max(1, height // 2))
        return _resize_to(pil, new)


class CenterCropFraction(EvaluationTransform):
    """Crop a central fraction: a different region of the same picture.

    This one is diagnostic rather than merely degradative. If a protection
    survives a resize but dies the moment a different region is shown, the
    pattern was never robust -- it was tuned to one crop window.
    """

    key = "center_crop_80"
    label = "Centre crop 80% then resize back"
    changes_geometry = True

    def __init__(self, fraction: float = 0.8) -> None:
        """Set the retained central fraction."""
        self.fraction = fraction

    def apply(self, image: RGB) -> RGB:
        """Crop centrally, then resize back to the original size."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        new_w = max(1, round(width * self.fraction))
        new_h = max(1, round(height * self.fraction))
        left = (width - new_w) // 2
        top = (height - new_h) // 2
        cropped = pil.crop((left, top, left + new_w, top + new_h))
        return _resize_to(cropped, (width, height))

    def parameters(self) -> dict[str, Any]:
        """Retained fraction."""
        return {"fraction": self.fraction}


class CornerCrop(EvaluationTransform):
    """Crop a corner instead of the centre: a deliberately shifted window."""

    key = "corner_crop"
    label = "Top-left crop 80%"
    changes_geometry = True

    def __init__(self, fraction: float = 0.8) -> None:
        """Set the retained fraction."""
        self.fraction = fraction

    def apply(self, image: RGB) -> RGB:
        """Crop from the top-left, then resize back to the original size."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        new_w = max(1, round(width * self.fraction))
        new_h = max(1, round(height * self.fraction))
        cropped = pil.crop((0, 0, new_w, new_h))
        return _resize_to(cropped, (width, height))

    def parameters(self) -> dict[str, Any]:
        """Retained fraction."""
        return {"fraction": self.fraction}


class SeededRandomCrop(EvaluationTransform):
    """A random crop window, reproducible because the seed is fixed.

    Centre and corner crops only probe two specific windows. A real pipeline
    re-crops somewhere else, and a protection that survives exactly the two
    windows someone thought to test has not been shown to be robust. This
    samples an offset instead, but derives it from ``seed`` rather than global
    RNG state: the same run always produces the same window, so two benchmark
    runs stay comparable.

    The seed is part of the key. Two different seeds are genuinely different
    measurements, and collapsing them onto one JSON key would silently discard
    one of them.
    """

    changes_geometry = True

    def __init__(self, fraction: float = 0.8, seed: int = 0) -> None:
        """Set the retained fraction and the seed fixing the crop window."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self.seed = seed
        percent = round(fraction * 100)
        self.key = f"random_crop_{percent}_s{seed}"
        self.label = f"Random crop {percent}% (seed {seed})"

    def apply(self, image: RGB) -> RGB:
        """Crop at a seeded offset, then resize back to the original size."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        new_w = max(1, round(width * self.fraction))
        new_h = max(1, round(height * self.fraction))
        # A local Generator, so this cannot perturb -- or be perturbed by --
        # whatever else in the run happens to draw from the global RNG.
        rng = np.random.default_rng(self.seed)
        left = int(rng.integers(0, width - new_w + 1))
        top = int(rng.integers(0, height - new_h + 1))
        cropped = pil.crop((left, top, left + new_w, top + new_h))
        return _resize_to(cropped, (width, height))

    def parameters(self) -> dict[str, Any]:
        """Retained fraction and the seed that fixes the window."""
        return {"fraction": self.fraction, "seed": self.seed}


class JpegQuality(EvaluationTransform):
    """Re-compress through JPEG at a given quality."""

    def __init__(self, quality: int) -> None:
        """Set the JPEG quality factor."""
        self.quality = quality
        self.key = f"jpeg_q{quality}"
        self.label = f"JPEG quality {quality}"

    def apply(self, image: RGB) -> RGB:
        """Round-trip through the JPEG encoder at ``quality``."""
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(
            buffer, format="JPEG", quality=self.quality, subsampling=0
        )
        buffer.seek(0)
        with Image.open(buffer) as handle:
            handle.load()
            return np.array(handle.convert("RGB"), dtype=np.uint8)

    def parameters(self) -> dict[str, Any]:
        """Quality factor."""
        return {"quality": self.quality}


def _factor_key(prefix: str, factor: float) -> str:
    """A unique, stable key for a factor-parameterised transform.

    ``brightness_1p05`` rather than ``brightness_small``: two brightness
    settings must never collide, because the JSON keys off this and a silent
    collision would drop a row from the report.
    """
    return f"{prefix}_{factor:.2f}".replace(".", "p").replace("-", "m")


class Brightness(EvaluationTransform):
    """Scale luminance by a factor."""

    def __init__(self, factor: float = 1.05) -> None:
        """Set the brightness factor."""
        self.factor = factor
        self.key = _factor_key("brightness", factor)
        self.label = f"Brightness x{factor}"

    def apply(self, image: RGB) -> RGB:
        """Apply the brightness factor."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        adjusted = ImageEnhance.Brightness(pil).enhance(self.factor)
        return np.array(adjusted, dtype=np.uint8)

    def parameters(self) -> dict[str, Any]:
        """Brightness factor."""
        return {"factor": self.factor}


class Contrast(EvaluationTransform):
    """Scale contrast around mid grey."""

    def __init__(self, factor: float = 0.95) -> None:
        """Set the contrast factor."""
        self.factor = factor
        self.key = _factor_key("contrast", factor)
        self.label = f"Contrast x{factor}"

    def apply(self, image: RGB) -> RGB:
        """Apply the contrast factor."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        adjusted = ImageEnhance.Contrast(pil).enhance(self.factor)
        return np.array(adjusted, dtype=np.uint8)

    def parameters(self) -> dict[str, Any]:
        """Contrast factor."""
        return {"factor": self.factor}


class GaussianBlur(EvaluationTransform):
    """Blur with a small radius, which erases high-frequency detail."""

    key = "blur_small"
    label = "Gaussian blur r=0.5"

    def __init__(self, radius: float = 0.5) -> None:
        """Set the blur radius."""
        self.radius = radius
        self.label = f"Gaussian blur r={radius}"

    def apply(self, image: RGB) -> RGB:
        """Apply the blur."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        blurred = pil.filter(ImageFilter.GaussianBlur(self.radius))
        return np.array(blurred, dtype=np.uint8)

    def parameters(self) -> dict[str, Any]:
        """Blur radius."""
        return {"radius": self.radius}


class UnsharpMask(EvaluationTransform):
    """Mild sharpening, which is what several re-scaling pipelines apply."""

    key = "sharpen_small"
    label = "Unsharp mask r=1.0 x1.2"

    def __init__(self, radius: float = 1.0, percent: int = 120) -> None:
        """Set the unsharp radius and strength."""
        self.radius = radius
        self.percent = percent
        self.label = f"Unsharp mask r={radius} x{percent / 100:.1f}"

    def apply(self, image: RGB) -> RGB:
        """Apply the unsharp mask."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        return np.array(
            pil.filter(ImageFilter.UnsharpMask(self.radius, self.percent, 3)),
            dtype=np.uint8,
        )

    def parameters(self) -> dict[str, Any]:
        """Radius and strength."""
        return {"radius": self.radius, "percent": self.percent}


class DownscaleRoundTrip(EvaluationTransform):
    """Halve, then restore. Geometry returns, high frequencies do not.

    This is the closest stand-in for a messaging app or CDN that re-encodes at
    a smaller size and then serves the original dimensions.
    """

    key = "downscale_roundtrip"
    label = "Downscale 50% and back"
    changes_geometry = True

    def __init__(self, factor: float = 0.5) -> None:
        """Set the intermediate scale factor."""
        self.factor = factor

    def apply(self, image: RGB) -> RGB:
        """Downscale then upscale back to the original size."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        small = pil.resize(
            (max(1, round(width * self.factor)), max(1, round(height * self.factor))),
            Image.Resampling.BILINEAR,
        )
        return _resize_to(small, (width, height))

    def parameters(self) -> dict[str, Any]:
        """Intermediate scale factor."""
        return {"factor": self.factor}


class ScreenshotLike(EvaluationTransform):
    """Display, resample and re-encode, the way a re-share does.

    Named for what it approximates and no more. It is *not* a reproduction of
    any named product's pipeline: the numbers are plausible display-side values
    chosen to span a range, and claiming this "simulates" a particular app would
    be unsupported. What it is, concretely, is the sequence a picture goes
    through when it is shown, captured and sent on again:

    1. displayed at a smaller size (LANCZOS downscale),
    2. tone adjusted slightly as a compositor and a display would,
    3. re-encoded through the **real** JPEG encoder with 4:2:0 chroma
       subsampling, which is what photo pipelines actually ship,
    4. scaled back to the original dimensions.

    The real encoder is the point. The attack's differentiable JPEG approximates
    this stage so it can be optimised against it; measuring against the same
    approximation would test the approximation, not the pipeline.

    Kept out of :func:`default_transforms` so the Phase 1 suite, and therefore
    the saved Phase 1 baseline, still reproduces byte for byte. Phase 2 asks for
    :func:`phase2_transforms`, which appends this row to the same 17.
    """

    key = "screenshot_like"
    label = "Screenshot-style resample and re-encode"
    changes_geometry = True

    def __init__(
        self,
        scale: float = 0.72,
        quality: int = 65,
        brightness: float = 1.02,
        contrast: float = 0.97,
    ) -> None:
        """Fix every parameter, so the row is reproducible rather than sampled."""
        if not 0.1 <= scale <= 1.0:
            raise ValueError(f"scale must be in [0.1, 1.0], got {scale}")
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be in [1, 100], got {quality}")
        self.scale = scale
        self.quality = quality
        self.brightness = brightness
        self.contrast = contrast

    def apply(self, image: RGB) -> RGB:
        """Run the composite in display order and restore the original size."""
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        width, height = pil.size
        small = pil.resize(
            (max(1, round(width * self.scale)), max(1, round(height * self.scale))),
            Image.Resampling.LANCZOS,
        )
        small = ImageEnhance.Brightness(small).enhance(self.brightness)
        small = ImageEnhance.Contrast(small).enhance(self.contrast)

        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=self.quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as handle:
            handle.load()
            decoded = handle.convert("RGB")
        return _resize_to(decoded, (width, height))

    def parameters(self) -> dict[str, Any]:
        """Every fixed parameter, so the row can be reconstructed."""
        return {
            "scale": self.scale,
            "quality": self.quality,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "chroma_subsampling": "4:2:0",
        }


def phase2_transforms() -> list[EvaluationTransform]:
    """The Phase 2 suite: the Phase 1 rows plus the screenshot-style row.

    The Phase 1 transforms come first, in their original order, so a Phase 2
    report diffed against the saved Phase 1 baseline shows exactly one added row
    and no reshuffled ones. Every variant in a Phase 2 comparison runs over this
    same list, which is what makes the four-way table comparable.
    """
    return [*default_transforms(), ScreenshotLike()]


def default_transforms() -> list[EvaluationTransform]:
    """The declared Phase 1 robustness suite.

    Identity first, because every later row is only interpretable relative to
    it. Order is fixed so that JSON output diffs cleanly between runs.
    """
    return [
        Identity(),
        ResizeShortSide(256),
        HalfScale(),
        CenterCropFraction(0.8),
        CornerCrop(0.8),
        SeededRandomCrop(0.8, seed=0),
        DownscaleRoundTrip(0.5),
        JpegQuality(95),
        JpegQuality(85),
        JpegQuality(75),
        JpegQuality(50),
        Brightness(1.05),
        Brightness(0.95),
        Contrast(0.95),
        Contrast(1.05),
        GaussianBlur(0.5),
        UnsharpMask(1.0, 120),
    ]
