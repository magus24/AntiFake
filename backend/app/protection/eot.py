"""Expectation over Transformation, inside the optimisation loop.

Phase 1 measured the consequence of optimising against exactly one rendering of
one image: the resulting pattern was tied to that geometry. A centre crop took
the target model's confidence drop from 96% to 13%, because the classifier that
was attacked re-read a different region of a different resampling of the frame,
and the pattern had never been optimised there. Compression and photometric
changes cost almost nothing; geometry cost almost everything. That asymmetry is
the signature of an overfitted perturbation, not of a fragile one.

This module closes the gap by making the *attack* see the same variety the
*evaluator* will. Every transform here is a differentiable ``torch`` operation
on a ``(N, 3, H, W)`` float tensor, so gradients flow from the model's loss back
through the geometry and the resampling into the perturbation. The Phase 1
transforms in :mod:`app.evaluation.transforms` cannot be used for this: they are
PIL and NumPy, they quantise to ``uint8``, and a gradient cannot pass through
either. Optimising against them would be optimising against a constant.

Sampling
--------
Each optimisation step draws a fresh set of samples from a seeded distribution,
so the perturbation is optimised against the *expectation* of the loss rather
than than a fixed list. Every draw comes from a local
:class:`torch.Generator` keyed by ``(seed, step, transform)``, never from global
RNG state. This matters for more than tidiness: the engine calls
``torch.manual_seed`` for the attack, the evaluation models load weights, and
LPIPS initialises nets. If any of those consumed global randomness between
steps, the EOT draws would shift and two runs declared identical would not be.
A test asserts exactly that.

What is and is not differentiable
---------------------------------
Real JPEG is a DCT, quantisation and Huffman stage, and quantisation has zero
derivative almost everywhere. :class:`DifferentiableJpeg` therefore approximates
it: block DCT, quantisation by a **straight-through estimator** (quantise in the
forward pass, pass the gradient through unchanged), inverse DCT. The gradient is
not the true gradient of JPEG, which is acceptable for the same reason plain PGD
is acceptable here -- the point is to find a perturbation that survives the
operator, not to differentiate it exactly. The *evaluation* transforms in
:mod:`app.evaluation.transforms` use the real encoder, so the benchmark still
measures genuine JPEG, not this approximation. Reporting a robustness number
measured against a differentiable stand-in for the thing it is supposed to
survive would be circular, and this separation is what prevents that.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

#: ``(N, 3, H, W)`` batch of images in ``[0, 1]``.
Batch = Tensor

#: Scale of the local generator used for every draw.
_U64 = 2**63 - 1


def _generator(
    seed: int, step: int, stream: int, device: torch.device
) -> torch.Generator:
    """A private generator for one ``(seed, step, stream)`` draw.

    Composed arithmetically rather than by hashing, so a given triple always
    yields the same sequence on any machine and any PyTorch version. Hashing
    would be shorter but ``hash()`` is salted per process for strings and is not
    stable across runs for the values used here.
    """
    mixed = (int(seed) * 0x9E3779B1 + int(step) * 0x85EBCA6B + int(stream) * 0xC2B2AE35)
    generator = torch.Generator(device=device)
    generator.manual_seed(mixed % (_U64 + 1))
    return generator


def _uniform(
    generator: torch.Generator, low: float, high: float, device: torch.device
) -> float:
    """One reproducible sample from ``[low, high)`` as a Python float."""
    value = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    return float(low + (high - low) * float(value))


def _resample(images: Batch, size: tuple[int, int], mode: str = "bilinear") -> Batch:
    return F.interpolate(
        images, size=size, mode=mode, align_corners=False, antialias=False
    )


def _shortest_side_resize(images: Batch, side: int) -> Batch:
    height, width = images.shape[-2], images.shape[-1]
    scale = side / float(min(height, width))
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    return _resample(images, (new_h, new_w))


class EotTransform(abc.ABC):
    """One differentiable transform sampled inside the attack loop.

    Attributes:
        key: Stable identifier for the report and for tests.
    """

    key: str = "abstract"

    #: Set by :meth:`apply` so the report can show what each step actually drew.
    last_parameters: dict[str, Any]

    @abc.abstractmethod
    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Return a transformed batch of the same device and dtype.

        Args:
            images: ``(N, 3, H, W)`` in ``[0, 1]``.
            generator: Private generator for this transform and step.
        """

    def describe(self) -> dict[str, Any]:
        """JSON-ready description of the transform family."""
        return {"key": self.key}

    @property
    def name(self) -> str:
        """Alias of :attr:`key`, for readable call sites."""
        return self.key


class Identity(EotTransform):
    """No change.

    Included because the expectation must include the case where a pipeline
    changes nothing, and because averaging over transforms that all degrade the
    image teaches the perturbation to be robust while forgetting the clean case.
    """

    key = "identity"

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Return the batch unchanged."""
        self.last_parameters = {}
        return images


class ResizeShortSide(EotTransform):
    """Resample the shortest side to a sampled length.

    The engine already works in a frame whose short side is the target's
    ``resize_short_side``, so a side equal to that value is a no-op. The sampler
    therefore draws *relative* factors and this class is documented in terms of
    the frame it is given.
    """

    key = "resize_short_side"

    def __init__(self, low: float = 0.5, high: float = 1.0) -> None:
        """Bound the drawn scale factor."""
        if not 0 < low <= high:
            raise ValueError(f"require 0 < low <= high, got {low}..{high}")
        self.low = low
        self.high = high

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Resize the shortest side by a sampled factor."""
        factor = _uniform(generator, self.low, self.high, images.device)
        side = max(8, round(min(images.shape[-2], images.shape[-1]) * factor))
        self.last_parameters = {"factor": round(factor, 6), "side": side}
        return _shortest_side_resize(images, side)


class RandomCrop(EotTransform):
    """Take a sampled crop window and resample it back to the original size.

    The crop-and-restore shape is what a consumer pipeline does: it throws away
    the border, keeps a region at a new scale, and a model reads it. Optimising
    only on the centre window is precisely the overfitting Phase 1 measured.
    """

    key = "random_crop"

    def __init__(self, low: float = 0.6, high: float = 1.0) -> None:
        """Bound the retained fraction."""
        if not 0 < low <= high <= 1.0:
            raise ValueError(f"require 0 < low <= high <= 1, got {low}..{high}")
        self.low = low
        self.high = high

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Crop at a sampled offset, then resize back to the input size."""
        height, width = images.shape[-2], images.shape[-1]
        fraction = _uniform(generator, self.low, self.high, images.device)
        new_h = max(1, round(height * fraction))
        new_w = max(1, round(width * fraction))
        top = int(
            torch.randint(
                0, height - new_h + 1, (), generator=generator, device=images.device
            )
        )
        left = int(
            torch.randint(
                0, width - new_w + 1, (), generator=generator, device=images.device
            )
        )
        cropped = images[:, :, top : top + new_h, left : left + new_w]
        self.last_parameters = {
            "fraction": round(fraction, 6),
            "top": top,
            "left": left,
        }
        return _resample(cropped, (height, width))


class CenterCrop(EotTransform):
    """Take a central window and resample it back.

    Kept alongside :class:`RandomCrop` because the centre window is the one the
    model is *actually* attacked on in the non-EOT path, and averaging over
    random windows only would leave the centre case underrepresented.
    """

    key = "center_crop"

    def __init__(self, low: float = 0.7, high: float = 1.0) -> None:
        """Bound the retained fraction."""
        if not 0 < low <= high <= 1.0:
            raise ValueError(f"require 0 < low <= high <= 1, got {low}..{high}")
        self.low = low
        self.high = high

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Crop centrally at a sampled fraction, then resize back."""
        height, width = images.shape[-2], images.shape[-1]
        fraction = _uniform(generator, self.low, self.high, images.device)
        new_h = max(1, round(height * fraction))
        new_w = max(1, round(width * fraction))
        top = (height - new_h) // 2
        left = (width - new_w) // 2
        cropped = images[:, :, top : top + new_h, left : left + new_w]
        self.last_parameters = {"fraction": round(fraction, 6)}
        return _resample(cropped, (height, width))


class ScaleJitter(EotTransform):
    """Resample the whole frame to a random aspect-preserving size.

    Distinct from :class:`ResizeShortSide` in intent rather than mechanics: this
    one models a pipeline that keeps the aspect ratio and re-encodes at a
    different resolution, which is what every thumbnailer and every re-upload
    does.
    """

    key = "scale_jitter"

    def __init__(self, low: float = 0.5, high: float = 1.0) -> None:
        """Bound the drawn scale factor."""
        if not 0 < low <= high:
            raise ValueError(f"require 0 < low <= high, got {low}..{high}")
        self.low = low
        self.high = high

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Resample both axes by a sampled factor."""
        factor = _uniform(generator, self.low, self.high, images.device)
        height, width = images.shape[-2], images.shape[-1]
        new_h = max(8, round(height * factor))
        new_w = max(8, round(width * factor))
        self.last_parameters = {"factor": round(factor, 6)}
        return _resample(images, (new_h, new_w))


#: Per-quality 8x8 quantisation matrices, the JPEG standard's luminance table
#: and the chrominance table derived from it. Only the *rounding grid* matters
#: for the straight-through approximation, so the luminance table is applied to
#: all three channels.
_JPEG_LUMA_QUANT: tuple[int, ...] = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)


def _dct_matrix(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Orthonormal 8x8 DCT-II basis, so the block transform is a real DCT."""
    n = 8
    index = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(math.pi / n * (index[None, :] + 0.5) * index[:, None])
    basis = basis * (2.0 / n) ** 0.5
    basis[0] = basis[0] / math.sqrt(2.0)
    return basis


class DifferentiableJpeg(EotTransform):
    """Block-DCT JPEG compression with a straight-through quantiser.

    Forward pass: 8x8 block DCT, divide by the standard quantisation table
    scaled by quality, round, multiply back, inverse DCT. The rounding is where
    the character of JPEG comes from -- it discards exactly the high-frequency
    detail an adversarial pattern relies on.

    Backward pass: the rounding's gradient is replaced by ``1`` (a
    straight-through estimator). The true derivative is zero almost everywhere,
    so a literal implementation would hand the attack no signal at all.

    Args:
        low: Lowest quality drawn, on the usual 1..100 scale.
        high: Highest quality drawn.
    """

    key = "jpeg"

    def __init__(self, low: float = 30.0, high: float = 95.0) -> None:
        """Bound the drawn quality factor."""
        if not 1 <= low <= high <= 100:
            raise ValueError(f"require 1 <= low <= high <= 100, got {low}..{high}")
        self.low = low
        self.high = high

    def _quant_table(
        self, quality: float, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        table = torch.tensor(_JPEG_LUMA_QUANT, device=device, dtype=dtype).view(8, 8)
        # The libjpeg quality scale: below 50 the table is scaled up, above 50
        # down. Reproduced because the mapping is not linear and getting it
        # wrong would make "q95" a different operator from the real encoder.
        quality = float(min(100.0, max(1.0, quality)))
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        return torch.clamp((table * scale + 50.0) / 100.0, min=1.0).floor()

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Apply one block-DCT compression at a sampled quality."""
        quality = _uniform(generator, self.low, self.high, images.device)
        batch, channels, height, width = images.shape
        # The level shift every JPEG encoder performs before the DCT. It is not
        # cosmetic: the standard quantisation tables are calibrated for signed
        # 8-bit samples, so feeding them unshifted [0, 1] values puts the DC
        # coefficient at 8*mean ~ 4 against a step of ~11, which rounds to zero
        # and turns the image black. The approximation has to reproduce the
        # encoder's own operating point, not just its maths.
        shifted = images * 255.0 - 128.0
        # Pad to whole 8x8 blocks; a real encoder does the same.
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        if pad_h or pad_w:
            shifted = F.pad(shifted, (0, pad_w, 0, pad_h), mode="replicate")
        block_h, block_w = shifted.shape[-2], shifted.shape[-1]

        basis = _dct_matrix(shifted.device, shifted.dtype)
        quant = self._quant_table(quality, shifted.device, shifted.dtype)

        blocks = (
            shifted.reshape(batch, channels, block_h // 8, 8, block_w // 8, 8)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 8, 8)
        )
        coefficients = basis @ blocks @ basis.t()
        # Straight-through: the forward value is quantised, the gradient passes
        # through the division untouched.
        quantised = torch.round(coefficients / quant)
        coefficients = coefficients + (quantised - coefficients).detach()
        restored = basis.t() @ (coefficients * quant) @ basis

        out = (
            restored.reshape(batch, channels, block_h // 8, block_w // 8, 8, 8)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, block_h, block_w)
        )
        out = out[:, :, :height, :width]
        self.last_parameters = {"quality": round(quality, 3)}
        return (out + 128.0).div_(255.0).clamp(0.0, 1.0)


class Brightness(EotTransform):
    """Additive brightness shift."""

    key = "brightness"

    def __init__(self, delta: float = 0.06) -> None:
        """Set the maximum absolute shift."""
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delta = delta

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Shift brightness by a sampled amount."""
        shift = _uniform(generator, -self.delta, self.delta, images.device)
        self.last_parameters = {"delta": round(shift, 6)}
        return (images + shift).clamp(0.0, 1.0)


class Contrast(EotTransform):
    """Contrast change about mid-grey."""

    key = "contrast"

    def __init__(self, strength: float = 0.12) -> None:
        """Set the maximum relative change."""
        if not 0 < strength < 1:
            raise ValueError(f"strength must be in (0, 1), got {strength}")
        self.strength = strength

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Scale deviations from mid-grey by a sampled factor."""
        factor = 1.0 + _uniform(generator, -self.strength, self.strength, images.device)
        self.last_parameters = {"factor": round(factor, 6)}
        return ((images - 0.5) * factor + 0.5).clamp(0.0, 1.0)


class GaussianBlur(EotTransform):
    """Separable Gaussian blur, sampled over a small radius range.

    Implemented as a depthwise convolution rather than a Gaussian pyramid
    because the convolution is one kernel application, keeps the gradient
    exactly, and needs no extra memory for a multi-scale decomposition.
    """

    key = "blur"

    def __init__(self, min_sigma: float = 0.3, max_sigma: float = 1.2) -> None:
        """Bound the drawn standard deviation."""
        if not 0 < min_sigma <= max_sigma:
            raise ValueError(f"require 0 < min <= max, got {min_sigma}..{max_sigma}")
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def _kernel(self, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        radius = max(1, round(3.0 * sigma))
        positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(positions**2) / (2.0 * sigma * sigma))
        return kernel / kernel.sum()

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Blur by a sampled sigma."""
        sigma = _uniform(generator, self.min_sigma, self.max_sigma, images.device)
        kernel_1d = self._kernel(sigma, images.device, images.dtype)
        channels = images.shape[1]
        weight = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        blurred = F.conv2d(
            images, weight, padding=(kernel_1d.numel() // 2, 0), groups=channels
        )
        weight = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        blurred = F.conv2d(
            blurred, weight, padding=(0, kernel_1d.numel() // 2), groups=channels
        )
        self.last_parameters = {"sigma": round(sigma, 6)}
        return blurred.clamp(0.0, 1.0)


class Sharpen(EotTransform):
    """Unsharp mask, the inverse failure mode of blur."""

    key = "sharpen"

    def __init__(self, amount: float = 0.8, radius: int = 1) -> None:
        """Set the mask strength and radius."""
        if amount <= 0:
            raise ValueError(f"amount must be > 0, got {amount}")
        self.amount = amount
        self.radius = radius

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Add back a scaled high-pass of the image."""
        size = 2 * self.radius + 1
        box = F.avg_pool2d(
            F.pad(images, (self.radius,) * 4, mode="replicate"),
            kernel_size=size,
            stride=1,
        )
        self.last_parameters = {"amount": self.amount, "radius": self.radius}
        return (images + self.amount * (images - box)).clamp(0.0, 1.0)


class SmallAffine(EotTransform):
    """Sampled translation, rotation and isotropic scale via ``grid_sample``.

    Covers the remaining geometric family Phase 1 never probed: a shifted,
    slightly rotated, slightly rescaled frame. ``grid_sample`` is chosen over
    composing separate crop and resize because it is one bilinear pull and its
    Jacobian is exactly the sampled transform, so the gradient reflects the
    geometry actually applied.
    """

    key = "affine"

    def __init__(
        self,
        max_translate: float = 0.06,
        max_rotate_deg: float = 4.0,
        min_scale: float = 0.94,
        max_scale: float = 1.0,
    ) -> None:
        """Bound each sampled parameter."""
        if not 0 <= max_translate < 1:
            raise ValueError(f"max_translate must be in [0, 1), got {max_translate}")
        if not 0 <= max_rotate_deg < 45:
            raise ValueError(f"max_rotate_deg must be in [0, 45), got {max_rotate_deg}")
        if not 0 < min_scale <= max_scale:
            raise ValueError(f"require 0 < min_scale <= max_scale, got {min_scale}")
        self.max_translate = max_translate
        self.max_rotate_deg = max_rotate_deg
        self.min_scale = min_scale
        self.max_scale = max_scale

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Apply a sampled rigid-plus-scale transform."""
        device = images.device
        batch = images.shape[0]
        dx = _uniform(generator, -self.max_translate, self.max_translate, device)
        dy = _uniform(generator, -self.max_translate, self.max_translate, device)
        angle = _uniform(generator, -self.max_rotate_deg, self.max_rotate_deg, device)
        scale = _uniform(generator, self.min_scale, self.max_scale, device)
        # A rotation is a rotation matrix, not a number dropped into a corner of
        # one. Writing ``angle`` in place of ``cos(angle)`` looks plausible in a
        # fixture and behaves nothing like the advertised transform: the matrix
        # is then close to [[0, s], [0, s]], which is singular, and the sampler
        # returns either a rank-deficient warp or a mirror image depending on the
        # sign of the draw. The Jacobian that the gradient depends on has to be
        # the one that is actually applied.
        radians = math.radians(angle)
        cos, sin = math.cos(radians), math.sin(radians)
        theta = torch.tensor(
            [
                [scale * cos, -scale * sin, dx],
                [scale * sin, scale * cos, dy],
            ],
            device=device,
            dtype=images.dtype,
        ).unsqueeze(0)
        # One transform shared by the whole batch, broadcast over N.
        theta = theta.expand(batch, 2, 3)
        grid = F.affine_grid(theta, list(images.shape), align_corners=False)
        self.last_parameters = {
            "rotate_deg": round(angle, 4),
            "scale": round(scale, 6),
            "translate": [round(dx, 6), round(dy, 6)],
            # det(scale * R) == scale**2 for any rotation, so this stays strictly
            # positive: the warp can never collapse or flip the frame. Derived
            # from the *rounded* scale that is reported next to it, so the two
            # fields agree for anyone who checks them by hand.
            "determinant": round(round(scale, 6) ** 2, 8),
        }
        return F.grid_sample(
            images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )


class ScreenshotLike(EotTransform):
    """A composite of the operations a screenshot or a chat re-share performs.

    Named for what it approximates and no more. It is *not* a reproduction of
    any named product's pipeline: the parameters are plausible display-side
    numbers chosen to span a range, and a claim that this "simulates Instagram"
    would be unsupported. The benchmark's own ``screenshot_like`` transform uses
    the real encoder for its JPEG stage; this one stays differentiable so the
    attack can be optimised against the same family.

    Order matters and follows the physical order: the image is displayed at
    some size (scale), resampled by the compositor (affine), softened by
    display scaling, tone-adjusted, then compressed for transport.
    """

    key = "screenshot_like"

    def __init__(self, seed: int = 0) -> None:
        """Bind the sub-transforms, each with a narrow, plausible range."""
        self._scale = ScaleJitter(0.55, 0.9)
        self._affine = SmallAffine(0.04, 2.0, 0.96, 1.0)
        self._blur = GaussianBlur(0.3, 0.8)
        self._brightness = Brightness(0.04)
        self._contrast = Contrast(0.08)
        self._jpeg = DifferentiableJpeg(55.0, 88.0)
        self.seed = seed

    def apply(self, images: Batch, generator: torch.Generator) -> Batch:
        """Run the full composite in display order."""
        out = self._scale.apply(images, generator)
        out = self._affine.apply(out, generator)
        out = self._blur.apply(out, generator)
        out = self._brightness.apply(out, generator)
        out = self._contrast.apply(out, generator)
        out = self._jpeg.apply(out, generator)
        self.last_parameters = {
            "scale": self._scale.last_parameters,
            "affine": self._affine.last_parameters,
            "blur": self._blur.last_parameters,
            "brightness": self._brightness.last_parameters,
            "contrast": self._contrast.last_parameters,
            "jpeg": self._jpeg.last_parameters,
        }
        return out


@dataclass(frozen=True, slots=True)
class EotConfig:
    """How many transforms to draw, and which families.

    Attributes:
        samples: Transforms drawn per optimisation step. ``1`` reduces to a
            single stochastic transform, which is a meaningful middle ground
            between the deterministic Phase 1 attack (no EOT at all) and the
            full expectation. ``0`` disables EOT entirely and reproduces the
            Phase 1 path exactly, which is what the comparison needs.
        families: Which transform families to sample from.
        seed: Base seed. Combined with the step index and the transform index to
            give every draw its own private stream.
    """

    samples: int = 4
    seed: int = 0
    families: tuple[str, ...] = (
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
    )

    def __post_init__(self) -> None:
        """Reject configurations that cannot be sampled."""
        if self.samples < 0:
            raise ValueError(f"samples must be >= 0, got {self.samples}")
        if self.samples > 32:
            raise ValueError(f"samples must be <= 32, got {self.samples}")
        if not self.families:
            raise ValueError("families must not be empty")


def build_transform(key: str, seed: int = 0) -> EotTransform:
    """Construct one transform by family key."""
    factories: dict[str, Any] = {
        "identity": Identity,
        "resize_short_side": ResizeShortSide,
        "random_crop": RandomCrop,
        "center_crop": CenterCrop,
        "scale_jitter": ScaleJitter,
        "jpeg": DifferentiableJpeg,
        "brightness": Brightness,
        "contrast": Contrast,
        "blur": GaussianBlur,
        "sharpen": Sharpen,
        "affine": SmallAffine,
        "screenshot_like": lambda s=seed: ScreenshotLike(s),
    }
    factory = factories.get(key)
    if factory is None:
        raise KeyError(f"unknown EOT family {key!r}; known: {sorted(factories)}")
    return factory()


class EotSampler:
    """Draws a reproducible set of transformed batches, one set per step.

    The family is chosen by round-robin over a shuffled, seeded permutation, so
    that over ``steps * samples`` draws every family is exercised roughly
    equally. Pure round-robin would be reproducible but would also always pair a
    given step index with a given family, which correlates the schedule with the
    optimisation; shuffling per step decorrelates them while staying
    deterministic.
    """

    def __init__(self, config: EotConfig, device: torch.device) -> None:
        """Bind a configuration and device."""
        self.config = config
        self.device = device
        self.transforms = tuple(
            build_transform(key, config.seed) for key in config.families
        )

    @property
    def enabled(self) -> bool:
        """``False`` when EOT is off, i.e. the Phase 1 path."""
        return self.config.samples > 0

    def sample_from(self, images: Batch, step: int) -> list[tuple[str, Batch]]:
        """Transform ``images`` for one step.

        Args:
            images: ``(N, 3, H, W)`` batch, already carrying the perturbation.
            step: Zero-based optimisation step.

        Returns:
            ``(family key, transformed batch)`` pairs, one per configured sample.

        When the ``identity`` family is configured it always takes the first
        slot, and the remaining slots are drawn from a per-step shuffle. Making it
        a guarantee rather than a probability is a deliberate choice: the
        untransformed view keeps a fixed ``1/samples`` share of every step's
        objective, so the attack cannot drift towards patterns that only work
        under heavy distortion. The cost is that with ``samples=4`` only three
        of the four slots explore transforms, and the report has to say so
        instead of implying a uniform expectation.
        """
        if not self.enabled:
            return [("identity", images)]
        # Identity holds its own slot and is removed from the pool the shuffle
        # draws from. Leaving it in the pool lets it be drawn twice, so
        # ``samples=4`` silently spends two slots on the clean view and the
        # expectation is not the one the sample count advertises.
        identity = next((t for t in self.transforms if t.key == "identity"), None)
        pool = [t for t in self.transforms if t.key != "identity"]
        order = torch.randperm(
            len(pool),
            generator=_generator(self.config.seed, step, 1, self.device),
            device=self.device,
        )
        drawn: list[tuple[str, Batch]] = []
        if identity is not None:
            generator = _generator(self.config.seed, step, 1000, self.device)
            drawn.append((identity.key, identity.apply(images, generator)))
        if not pool:  # identity-only configuration
            return drawn
        for slot in range(len(drawn), self.config.samples):
            transform = pool[int(order[(slot - len(drawn)) % len(order)])]
            generator = _generator(
                self.config.seed, step, 1000 + slot, self.device
            )
            drawn.append((transform.key, transform.apply(images, generator)))
        return drawn

    def history(self) -> list[dict[str, Any]]:
        """What the most recent step actually drew, for the report."""
        return [
            {"key": transform.key, **getattr(transform, "last_parameters", {})}
            for transform in self.transforms
            if hasattr(transform, "last_parameters")
        ]
