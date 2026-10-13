"""torchvision ImageNet classifiers as benchmark adapters.

Every model here shares the same preprocessing contract as the protection
target -- shortest-side resize, centre crop, ImageNet normalisation -- but each
keeps its own declared input size, because a ViT trained at 224 and one
trained at 384 are different models and silently sharing a size would make the
comparison meaningless.

The preprocessing is reimplemented with ``F.interpolate`` rather than
``torchvision.transforms`` so that the exact code path the protection engine
attacks is the same code path the benchmark reads through. That is the whole
point of measuring transfer: if the benchmark used a different resize than the
engine optimised against, a drop in confidence could not be attributed to the
perturbation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)

from app.evaluation.base import (
    ClassificationObservation,
    ModelAdapter,
    ModelUnavailableError,
)
from app.protection.base import RGB
from app.protection.target_model import resolve_weights_enum

logger = logging.getLogger(__name__)

#: ImageNet-1k normalisation, shared by every torchvision classifier here.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def to_unit_tensor(image: RGB, device: torch.device) -> torch.Tensor:
    """``(H, W, 3) uint8`` -> ``(1, 3, H, W) float32`` in ``[0, 1]`` on ``device``.

    Copies rather than wrapping: PIL-derived arrays are often read-only, and a
    torch tensor over read-only memory has undefined write behaviour.
    """
    return (
        torch.from_numpy(np.array(image, dtype=np.uint8, order="C", copy=True))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


class TorchvisionClassifierAdapter(ModelAdapter):
    """A torchvision classifier behind the benchmark interface.

    Args:
        device: Where to place the weights.
        model_name: torchvision builder name, e.g. ``"densenet121"``.
        weights_name: Weights enum member name, e.g. ``"IMAGENET1K_V1"``.
        crop_size: Model's declared input resolution.
        resize_short_side: Shortest-side resize target.
    """

    family = "classification"

    def __init__(
        self,
        device: torch.device,
        *,
        model_name: str,
        weights_name: str = "IMAGENET1K_V1",
        crop_size: int = 224,
        resize_short_side: int = 256,
    ) -> None:
        """Declare the model without loading it."""
        super().__init__(device)
        self.model_name = model_name
        self.weights_name = weights_name
        self.crop_size = crop_size
        self.resize_short_side = resize_short_side
        self.name = model_name
        self.description = (
            f"torchvision:{model_name} (weights={weights_name}) via "
            f"resize-{resize_short_side}/center-crop-{crop_size}"
        )
        self._model: torch.nn.Module | None = None
        self._categories: list[str] = []
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _net(self) -> torch.nn.Module:
        """The loaded network, or a loud error. Narrows the type for mypy.

        Preferred over ``assert`` because it survives ``python -O`` and because
        the failure names the adapter, which is what a log needs.
        """
        self._require_loaded()
        if self._model is None:
            raise RuntimeError(f"{self.name}: loaded but the network is missing")
        return self._model

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Resolve the weights enum, build the network, cache the label list."""
        if self._loaded:
            return
        from torchvision import models as tv

        enum = resolve_weights_enum(self.model_name)
        if enum is None:
            raise ModelUnavailableError(
                self.name, f"torchvision has no {self.model_name} weights enum"
            )
        member = None
        for candidate in enum:
            if candidate.name.lower() == self.weights_name.lower():
                member = candidate
                break
        if member is None:
            raise ModelUnavailableError(
                self.name,
                f"no weights {self.weights_name!r}; available: {[c.name for c in enum]}",
            )

        builder: Any = getattr(tv, self.model_name)
        model = builder(weights=member).to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model

        categories = (member.meta or {}).get("categories") or []
        self._categories = list(categories)
        self._loaded = True
        logger.info(
            "adapter.loaded name=%s weights=%s device=%s labels=%d",
            self.name,
            self.weights_name,
            self.device,
            len(self._categories),
        )

    def release(self) -> None:
        """Delete the network and free the device cache."""
        self._model = None
        self._loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------

    def preprocess(self, image: RGB) -> torch.Tensor:
        """Differentiable resize -> centre crop -> normalise, on device.

        Mirrors :class:`app.protection.target_model.TargetModel.preprocess` so
        that the benchmark reads the image through precisely the geometry the
        engine optimised against.
        """
        tensor = to_unit_tensor(image, self.device)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        scale = self.resize_short_side / float(min(height, width))
        new_h = max(self.crop_size, round(height * scale))
        new_w = max(self.crop_size, round(width * scale))
        resized = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        top = max(0, (new_h - self.crop_size) // 2)
        left = max(0, (new_w - self.crop_size) // 2)
        cropped = resized[:, :, top : top + self.crop_size, left : left + self.crop_size]
        return (cropped - self._mean) / self._std

    def evaluate(self, image: RGB) -> ClassificationObservation:
        """Read one image and describe its label distribution."""
        with torch.no_grad():
            logits = self._net()(self.preprocess(image))
            probabilities = torch.softmax(logits, dim=1)[0]
        return self._observe(logits, probabilities)

    def _observe(
        self, logits: torch.Tensor, probabilities: torch.Tensor
    ) -> ClassificationObservation:
        values, indices = torch.topk(probabilities, 5)
        top1_index = int(indices[0])
        label = (
            self._categories[top1_index]
            if 0 <= top1_index < len(self._categories)
            else f"class_{top1_index}"
        )
        entropy = float(
            -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
        )
        return ClassificationObservation(
            top1_index=top1_index,
            top1_label=label,
            top1_probability=float(values[0]),
            top1_logit=float(torch.max(logits)),
        top5=[
            (int(i), float(v))
            for v, i in zip(values.tolist(), indices.tolist(), strict=True)
        ],
            entropy=entropy,
        )


#: Declared input resolution per model, so the benchmark never guesses.
CROP_SIZES: dict[str, int] = {
    "resnet18": 224,
    "densenet121": 224,
    "efficientnet_b0": 224,
    "mobilenet_v3_small": 224,
    "vit_b_16": 224,
    "swin_t": 224,
}

#: Weight enums, pinned so the report can name exactly what was measured.
WEIGHT_ENUMS: dict[str, Any] = {
    "resnet18": ResNet18_Weights,
    "densenet121": DenseNet121_Weights,
    "efficientnet_b0": EfficientNet_B0_Weights,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights,
    "vit_b_16": ViT_B_16_Weights,
}


def build_classifier(
    device: torch.device, model_name: str, weights_name: str = "IMAGENET1K_V1"
) -> TorchvisionClassifierAdapter:
    """Construct a declared-but-unloaded classifier adapter."""
    crop = CROP_SIZES.get(model_name, 224)
    # The default ImageNet eval transform is resize 256 -> crop 224; models
    # evaluated at a different declared size scale the resize with it.
    resize = 256 if crop == 224 else round(256 * crop / 224)
    return TorchvisionClassifierAdapter(
        device,
        model_name=model_name,
        weights_name=weights_name,
        crop_size=crop,
        resize_short_side=resize,
    )
