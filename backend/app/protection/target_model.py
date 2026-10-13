"""The single AI pipeline this MVP protects against.

Threat model for experiment #1, stated explicitly and narrowly:

    ``torchvision.models.resnet18`` with ImageNet-1k weights, evaluated through
    the stock torchvision transform: shortest-side resize to 256 (bilinear),
    centre crop 224, normalise with the ImageNet mean/std.

An attacker who scrapes and re-encodes the image, resizes it, or feeds it to a
classifier that uses *this* preprocessing is the intended adversary. Nothing is
promised against other architectures, other datasets, other preprocessing, or
against generative re-creation of the subject. See README "Limitations".
"""

from __future__ import annotations

import logging
import threading
from enum import EnumMeta
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

from app.protection.config import TargetModelConfig

logger = logging.getLogger(__name__)

#: ``torch.nn.functional.interpolate`` takes mode strings, not the
#: ``torchvision`` enum values, so the config stores a string and we keep a
#: whitelist to reject typos early.
_INTERPOLATIONS: frozenset[str] = frozenset({"bilinear", "bicubic", "nearest"})

_MODEL_CACHE: dict[tuple[str, str, str], TargetModel] = {}
_CACHE_LOCK = threading.Lock()


def resolve_device(requested: str = "auto") -> torch.device:
    """Map a config device string onto a concrete :class:`torch.device`.

    ``"auto"`` selects CUDA when it is actually usable, which keeps the CPU
    default intact while leaving a single switch for GPU acceleration later.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but CUDA is not available on this host."
            )
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"Unknown device {requested!r}.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_weights_enum(candidate: Any) -> bool:
    """True only for the enum *class*, never for one of its members.

    torchvision attaches a single default ``Weights`` member to the builder as
    ``builder.Weights``. Because that member is itself an enum instance, it
    also exposes ``__members__`` -- so a naive duck-typed check happily returns
    a single weight where the whole enum was wanted.
    """
    return isinstance(candidate, EnumMeta) and bool(candidate.__members__)


def _weight_ids(enum: EnumMeta) -> list[str]:
    """Lowercase identifiers of an enum, e.g. ``["imagenet1k_v1", "default"]``.

    The identifier is the member *name*, lowercased. ``member.value`` is not
    usable: in current torchvision the members are themselves ``Weights``
    objects rather than strings, and that is not guaranteed either.
    """
    return [name.lower() for name in enum.__members__]


def _enum_matches_model(enum: EnumMeta, model_name: str) -> bool:
    """True if this weights enum plausibly belongs to ``model_name``.

    Compares the enum's own class name (``ResNet18_Weights``) against the
    model name, case-insensitively, because torchvision's capitalisation
    (``ResNet`` vs ``resnet``) does not follow a rule we can generate.
    """
    return model_name.lower() in enum.__name__.lower()


def _enum_is_exact_match(enum: EnumMeta, model_name: str) -> bool:
    """True for ``VGG16_Weights`` given ``vgg16``, but not ``VGG16_BN_Weights``.

    A substring test alone happily returns the batch-norm twin of a model,
    which torchvision then rejects at build time with a confusing error about
    the wrong weight class. Preferring the exact twin keeps the two apart.
    """
    return enum.__name__.lower() == f"{model_name.lower()}_weights"


def resolve_weights_enum(model_name: str) -> Any | None:
    """Find the ``Weights`` enum that belongs to a torchvision model.

    torchvision has moved this attribute around across releases -- sometimes
    it hangs off the builder, sometimes it is a module-level
    ``ResNet18_Weights`` class. Try the known locations first, then fall back
    to scanning the module, preferring an exact name match over a substring
    one so that ``vgg16`` resolves to ``VGG16_Weights`` and not to
    ``VGG16_BN_Weights``.
    """
    builder = getattr(models, model_name, None)
    if builder is None:
        return None

    for attribute in ("weights", "Weights"):
        candidate = getattr(builder, attribute, None)
        if _is_weights_enum(candidate):
            return candidate

    loose: list[EnumMeta] = []
    for attribute in dir(models):
        if not attribute.endswith("_Weights"):
            continue
        candidate = getattr(models, attribute)
        if _is_weights_enum(candidate) and _enum_matches_model(candidate, model_name):
            if _enum_is_exact_match(candidate, model_name):
                return candidate
            loose.append(candidate)
    return loose[0] if loose else None


def available_weights(name: str) -> list[str]:
    """Weight identifiers declared for a torchvision model, e.g. ``imagenet1k_v1``."""
    enum = resolve_weights_enum(name)
    if enum is None:
        return []
    return _weight_ids(enum)


def _resolve_weights(model_name: str, weights: str) -> Any:
    """Look up a single weights enum member, with an actionable error."""
    enum = resolve_weights_enum(model_name)
    if enum is None:
        raise ValueError(
            f"Could not locate a Weights enum for torchvision model {model_name!r}."
        )
    wanted = weights.lower()
    for member in enum:
        if member.name.lower() == wanted:
            return member
    raise ValueError(
        f"torchvision model {model_name!r} has no weights {weights!r}. "
        f"Available: {sorted(_weight_ids(enum))}"
    )


def _build_backbone(config: TargetModelConfig) -> nn.Module:
    builder = getattr(models, config.name, None)
    if builder is None:
        raise ValueError(f"torchvision has no model named {config.name!r}.")
    if config.weights in ("", "none"):
        # Offline / test path: random weights, deterministically seeded. The
        # attack mechanics are identical; only the semantics of the resulting
        # perturbation change, so these images are NOT actually protected
        # against a real classifier.
        torch.manual_seed(0)
        model = builder(weights=None)
        model.eval()
        return model
    model = builder(weights=_resolve_weights(config.name, config.weights))
    model.eval()
    return model


class TargetModel:
    """Wraps the target network and its preprocessing as one differentiable unit.

    The preprocessing is reimplemented with ``F.interpolate`` rather than the
    ``torchvision.transforms`` objects so the whole path is differentiable and
    can be traced back to the pixels we actually perturb.
    """

    def __init__(
        self,
        config: TargetModelConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build a target pipeline.

        Args:
            config: Which model, weights and preprocessing to load.
            device: Device to place the network on.
        """
        self.config = config or TargetModelConfig()
        self.device = torch.device(device)
        self.model = _build_backbone(self.config).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._mean = self._channel_tensor(self.config.mean)
        self._std = self._channel_tensor(self.config.std)
        if self.config.interpolation not in _INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {sorted(_INTERPOLATIONS)}, "
                f"got {self.config.interpolation!r}"
            )
        logger.info(
            "target_model.loaded model=%s weights=%s device=%s",
            self.config.name,
            self.config.weights,
            self.device,
        )

    def _channel_tensor(self, values: tuple[float, ...]) -> Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    def preprocess(self, images: Tensor) -> Tensor:
        """Differentiable equivalent of the stock eval transform.

        Args:
            images: ``(N, 3, H, W)`` float tensor in ``[0, 1]``, RGB.

        Returns:
            ``(N, 3, 224, 224)`` normalised tensor ready for the network.
        """
        height, width = images.shape[-2], images.shape[-1]
        side = self.config.resize_short_side
        scale = side / float(min(height, width))
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        resized = F.interpolate(
            images,
            size=(new_h, new_w),
            mode=self.config.interpolation,
            align_corners=False,
            antialias=False,
        )
        crop = self.config.crop_size
        top = max(0, (new_h - crop) // 2)
        left = max(0, (new_w - crop) // 2)
        cropped = resized[:, :, top : top + crop, left : left + crop]
        if cropped.shape[-2:] != (crop, crop):
            cropped = F.interpolate(
                cropped,
                size=(crop, crop),
                mode=self.config.interpolation,
                align_corners=False,
            )
        return (cropped - self._mean) / self._std

    def forward(self, images: Tensor) -> Tensor:
        """Differentiable forward pass through preprocessing and the network.

        Deliberately *not* wrapped in ``torch.no_grad()``: the protective
        perturbation is obtained by backpropagating from this exact call, so
        the graph has to survive. Parameter gradients are still impossible
        because every parameter has ``requires_grad=False``.
        """
        return self.model(self.preprocess(images))

    def predict(self, images: Tensor) -> Tensor:
        """Inference-only forward pass. Use this to read logits, not to attack."""
        with torch.no_grad():
            return self.forward(images)

    def __call__(self, images: Tensor) -> Tensor:
        """Alias for :meth:`forward`, so this object can stand in for ``nn.Module``."""
        return self.forward(images)

    def describe(self) -> str:
        """One-line description of the exact pipeline this object represents."""
        return self.config.description


def load_target_model(
    config: TargetModelConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    cache: bool = True,
) -> TargetModel:
    """Load (and optionally memoise) a :class:`TargetModel`.

    Loading ResNet18 costs ~50 MB of disk traffic, so repeated requests reuse
    one instance per ``(model, weights, device)`` triple.

    ``cache=False`` returns a model that is owned solely by the caller. The
    ensemble passes it so that ``close()`` really means "release the weights":
    a model that the global cache still references survives ``clear()``, so a
    chunked run -- whose entire purpose is to hold a few surrogates at a time --
    would otherwise pin every surrogate it ever loaded for the life of the
    process and defeat the memory bound it advertises.
    """
    resolved_config = config or TargetModelConfig()
    if not cache:
        return TargetModel(resolved_config, device=device)
    key = (
        resolved_config.name,
        resolved_config.weights,
        str(torch.device(device)),
    )
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            cached = TargetModel(resolved_config, device=device)
            _MODEL_CACHE[key] = cached
    return cached


def clear_model_cache() -> None:
    """Drop cached models. Used by tests that need a clean slate."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
