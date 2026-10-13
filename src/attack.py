"""Step 2: PGD attack that pushes the face embeddings apart.

The objective is deliberately narrow. We minimise the cosine similarity between
the embedding of the protected crop and the embedding of the *clean* crop, so
that a face-recognition model sees a different person. That is exactly the
quantity a face-swap pipeline depends on, and nothing else.

Three design points worth stating out loud:

* **The attack runs on a cropped, downscaled face**, and the perturbation is
  resized back to the original resolution. Bilinear interpolation is a convex
  combination, so the L-inf bound survives the round trip, and pixels outside
  the feathered mask are left bit-for-bit identical to the source.
* **The mask is applied inside the optimisation loop**, not pasted on
  afterwards. Otherwise the gradients would be computed for an image the model
  never actually sees.
* **EOT happens before the model**, so the objective asks for a perturbation
  that survives the ways a photo really gets moved around, not one tuned to a
  single rendering.

Imperceptibility is part of the objective rather than a happy accident of a small
``eps``. Four things work together:

* ``luminance_only`` optimises a **single** plane and writes it to R, G and B
  alike, so hue and saturation are untouched and the noise cannot be seen as a
  colour cast.
* ``lpips_weight`` adds a perceptual distance to the loss. It is implemented and
  switchable, but it **defaults to 0.0 on purpose**: measured on this project
  it is counterproductive. PGD moves by ``gradient.sign()``, which discards
  gradient magnitude, so a competing LPIPS term cannot trade off visibility
  against attack strength -- it only corrupts the sign pattern. Measured at
  eps 32 on the same photo, weight 0 -> 2 -> 10 -> 40 moved full-resolution
  LPIPS 0.0195 -> 0.0197 -> 0.0213 -> 0.0245 while ArcFace similarity got
  *worse* 0.334 -> 0.368 -> 0.412 -> 0.458. The perceptual gain this project
  actually got came from luminance-only output plus post-upsample smoothing,
  which cut LPIPS from 0.0039 to 0.0014 at eps 8.
* ``smooth_sigma`` blurs the perturbation after it is upsampled back to native
  resolution, removing the ringing that resizing a high-frequency pattern
  inevitably produces.
* ``mask_expand``/``mask_feather`` widen the soft mask, so the perturbation
  fades out over a wide band instead of stopping dead at the face border, which
  is what produced a visible bright rim.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.detect import DetectedFace, face_mask


@dataclass(frozen=True)
class AttackConfig:
    """Everything that defines one attack run."""

    eps: float = 8.0 / 255.0
    steps: int = 40
    seed: int = 0
    work_size: int = 160
    eot_copies: int = 2
    crop_margin: float = 0.6
    alpha_ratio: float = 0.1
    resize_low: float = 0.5
    blur_prob: float = 0.5
    noise_prob: float = 0.5
    noise_max: float = 0.02
    crop_prob: float = 0.5
    jpeg_prob: float = 0.5
    jpeg_qualities: tuple[int, ...] = (60, 85)
    surrogates: tuple[str, ...] = ("vggface2", "casia-webface")
    mask_expand: float = 0.30
    mask_feather: float = 0.55
    luminance_only: bool = True
    lpips_weight: float = 0.0
    smooth_sigma: float = 0.8

    @property
    def alpha(self) -> float:
        return self.eps * self.alpha_ratio


@functools.lru_cache(maxsize=1)
def _lpips_perceptual() -> nn.Module | None:
    """LPIPS used as a differentiable loss term, or ``None`` if unavailable."""
    try:
        import lpips

        torch.manual_seed(0)
        model = lpips.LPIPS(net="alex", verbose=False).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
    except Exception:
        return None


def _gaussian_kernel(sigma: float) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _smooth_2d(tensor: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur on an ``(N, C, H, W)`` tensor, edges replicated."""
    if sigma <= 0:
        return tensor
    kernel = _gaussian_kernel(sigma)
    radius = (kernel.numel() - 1) // 2
    channels = tensor.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
    padded = F.conv2d(padded, kx, groups=channels)
    padded = F.pad(padded, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded, ky, groups=channels)


class SurrogateEnsemble(nn.Module):
    """Differentiable face-recognition surrogates, averaged in cosine space."""

    def __init__(self, names: tuple[str, ...] = ("vggface2", "casia-webface")) -> None:
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.names = names
        self.models = nn.ModuleList(
            [InceptionResnetV1(pretrained=name).eval() for name in names]
        )
        for model in self.models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def forward(self, images: Tensor) -> list[Tensor]:
        """``images`` in [0, 1], NCHW. Returns one embedding per surrogate."""
        return [
            model((images - 0.5) / 0.5) for model in self.models
        ]

    @torch.no_grad()
    def reference(self, images: Tensor) -> list[Tensor]:
        return self.forward(images)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(torch.rand(1, generator=generator).item() * (high - low) + low)


def _jpeg_ste(images: Tensor, quality: int) -> Tensor:
    """JPEG round trip with a straight-through gradient.

    The forward value is a genuine JPEG of the image -- real DCT, real
    quantisation tables -- but the backward pass is the identity, which is the
    standard way to get a gradient through a non-differentiable operator.
    """
    array = (images.detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    for index, plane in enumerate(array):
        ok, encoded = cv2.imencode(
            ".jpg", plane.transpose(1, 2, 0)[:, :, ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        array[index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    quantised = torch.from_numpy(array).to(images.dtype).to(images.device) / 255.0
    return images + (quantised - images.detach())


def eot_views(images: Tensor, config: AttackConfig, generator: torch.Generator) -> Tensor:
    """``config.eot_copies`` independently augmented copies, stacked in the batch."""
    height, width = images.shape[-2:]
    views: list[Tensor] = []
    for _ in range(max(1, config.eot_copies)):
        view = images
        scale = _uniform(generator, config.resize_low, 1.0)
        if scale < 0.999:
            shrunk = (
                max(8, int(round(height * scale))),
                max(8, int(round(width * scale))),
            )
            view = F.interpolate(
                view, size=shrunk, mode="bilinear", align_corners=False
            )
            view = F.interpolate(
                view, size=(height, width), mode="bilinear", align_corners=False
            )
        if _uniform(generator, 0.0, 1.0) < config.noise_prob:
            view = view + torch.randn(
                view.shape, generator=generator, device=view.device
            ) * _uniform(generator, 0.0, config.noise_max)
        if _uniform(generator, 0.0, 1.0) < config.blur_prob:
            kernel = 3
            view = F.avg_pool2d(view, kernel, stride=1, padding=kernel // 2)
        if _uniform(generator, 0.0, 1.0) < config.crop_prob:
            margin_h = int(round(height * _uniform(generator, 0.02, 0.12)))
            margin_w = int(round(width * _uniform(generator, 0.02, 0.12)))
            view = view[
                :,
                :,
                margin_h : height - margin_h,
                margin_w : width - margin_w,
            ]
            view = F.interpolate(
                view, size=(height, width), mode="bilinear", align_corners=False
            )
        if config.jpeg_qualities and _uniform(generator, 0.0, 1.0) < config.jpeg_prob:
            view = _jpeg_ste(
                view, int(_uniform(generator, *config.jpeg_qualities))
            )
        views.append(view)
    return torch.cat(views, dim=0)


def pgd(
    ensemble: SurrogateEnsemble,
    clean: Tensor,
    mask: Tensor,
    config: AttackConfig,
    generator: torch.Generator,
) -> Tensor:
    """Return the perturbation ``delta``.

    With ``luminance_only`` the returned tensor has one channel; ``protect``
    writes it to all three colour planes, which keeps hue untouched.
    """
    with torch.no_grad():
        references = ensemble.reference(clean)

    perceptual = _lpips_perceptual() if config.lpips_weight > 0 else None
    planes = 1 if config.luminance_only else clean.shape[1]
    shape = (clean.shape[0], planes, clean.shape[2], clean.shape[3])

    delta = (
        torch.rand(shape, generator=generator, device=clean.device) * 2 - 1
    ) * config.eps
    delta = delta * mask

    for _ in range(config.steps):
        delta.requires_grad_(True)
        spread = delta.expand(-1, clean.shape[1], -1, -1) if planes == 1 else delta
        protected = (clean + spread).clamp(0.0, 1.0)
        views = eot_views(protected, config, generator)
        embeddings = ensemble(views)
        loss = torch.zeros((), device=clean.device)
        for embedded, reference in zip(embeddings, references):
            target = reference.expand(embedded.shape[0], -1)
            loss = loss + F.cosine_similarity(embedded, target, dim=1).mean()
        loss = loss / len(references)
        if perceptual is not None:
            # LPIPS is defined on [-1, 1]. The penalty is evaluated on a 64x64
            # copy: AlexNet's cost is dominated by the early full-resolution
            # convolutions, and the perceptual signal we want (is this a
            # visible, structured change?) is fully present at that size. The
            # *reported* LPIPS in the metrics is still full resolution.
            small_protected = F.interpolate(
                protected, size=(64, 64), mode="bilinear", align_corners=False
            )
            small_clean = F.interpolate(
                clean, size=(64, 64), mode="bilinear", align_corners=False
            )
            perceptual_loss = perceptual(
                small_protected * 2.0 - 1.0, small_clean * 2.0 - 1.0
            ).mean()
            loss = loss + config.lpips_weight * perceptual_loss
        gradient = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - config.alpha * gradient.sign()
            delta = delta.clamp(-config.eps, config.eps) * mask
    return delta.detach()


def _expand_box(
    bbox: tuple[int, int, int, int], margin: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = int(round((x2 - x1) * margin))
    pad_y = int(round((y2 - y1) * margin))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _to_tensor(image: np.ndarray) -> Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def _mask_to_tensor(mask: np.ndarray) -> Tensor:
    """A single-channel mask already in [0, 1] -> ``(1, 1, H, W)``.

    Deliberately not routed through :func:`_to_tensor`. That helper divides by
    255 because it takes uint8 image data; applying it to a float mask scales
    the mask down by 255, which silently collapses the perturbation to nothing
    after rounding to uint8.
    """
    return torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]


def _resize(tensor: Tensor, size: int) -> Tensor:
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def protect(
    image: np.ndarray,
    face: DetectedFace,
    ensemble: SurrogateEnsemble,
    config: AttackConfig,
) -> tuple[np.ndarray, dict]:
    """Protect one face in ``image`` and return the new image plus attack facts."""
    height, width = image.shape[:2]
    generator = torch.Generator().manual_seed(config.seed)

    full_mask = face_mask(
        (height, width),
        face.bbox,
        expand=config.mask_expand,
        feather=config.mask_feather,
    )
    x1, y1, x2, y2 = _expand_box(face.bbox, config.crop_margin, width, height)
    crop = image[y1:y2, x1:x2]
    crop_mask = full_mask[y1:y2, x1:x2]

    clean = _resize(_to_tensor(crop), config.work_size)
    mask = _resize(_mask_to_tensor(crop_mask), config.work_size)

    delta = pgd(ensemble, clean, mask, config, generator)

    with torch.no_grad():
        native = F.interpolate(
            delta, size=crop.shape[:2], mode="bicubic", align_corners=False
        )
        # Resizing a 160x160 pattern up to a full-resolution crop leaves visible
        # ringing, and a hard mask edge leaves a bright rim. Blurring the
        # upsampled perturbation removes both; clamping afterwards keeps the
        # L-inf budget honest.
        native = _smooth_2d(native, config.smooth_sigma)
        native = native.clamp(-config.eps, config.eps)
        if native.shape[1] == 1:
            native = native.expand(-1, 3, -1, -1)
        native_np = native[0].permute(1, 2, 0).cpu().numpy()

    # delta lives in [0, 1]; the image buffer is 0..255. Skipping this scale
    # leaves the perturbation below half a code value and rounding erases it.
    out = image.astype(np.float32).copy()
    out[y1:y2, x1:x2] += native_np * 255.0 * crop_mask[:, :, None]
    protected = np.clip(np.round(out), 0, 255).astype(np.uint8)

    return protected, {
        "crop": (x1, y1, x2, y2),
        "work_size": config.work_size,
        "mask_coverage": float((full_mask > 0.02).mean()),
        "delta_linf_255": float(np.abs(native_np).max() * 255.0),
        "luminance_only": bool(config.luminance_only),
        "lpips_weight": float(config.lpips_weight),
        "smooth_sigma": float(config.smooth_sigma),
    }


@torch.no_grad()
def surrogate_similarity(
    image: np.ndarray,
    other: np.ndarray,
    face: DetectedFace,
    ensemble: SurrogateEnsemble,
    config: AttackConfig,
) -> float:
    """Mean cosine between two images' surrogate embeddings, clean-to-clean.

    This is the *training* objective's own score. Comparing it with the
    held-out ArcFace number separates "the attack did not optimise" from "the
    attack optimised but did not transfer", which are very different problems.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _expand_box(face.bbox, config.crop_margin, width, height)
    first = _resize(_to_tensor(image[y1:y2, x1:x2]), config.work_size)
    second = _resize(_to_tensor(other[y1:y2, x1:x2]), config.work_size)
    left = ensemble(first)
    right = ensemble(second)
    cosines = [
        float(F.cosine_similarity(a, b, dim=1).mean().item())
        for a, b in zip(left, right)
    ]
    return float(np.mean(cosines))