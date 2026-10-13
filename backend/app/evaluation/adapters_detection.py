"""Object detection as a benchmark adapter.

Where the classification models read the picture and the face adapter reads a
person, an object detector does the third thing a viewer does: it names the
things in the scene and says where they are. A protection that leaves every
ImageNet label intact but causes a detector to drop its boxes is still doing
something, and no classification row would show it.

``ultralytics`` is imported here and nowhere else, so the rest of the benchmark
runs without it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from app.evaluation.base import (
    Detection,
    DetectionObservation,
    ModelAdapter,
    ModelUnavailableError,
)
from app.protection.base import RGB

logger = logging.getLogger(__name__)

#: Confidence floor for a box to be reported at all. It is deliberately low:
#: the question is whether the object is still recognised, not whether the
#: detector is feeling confident about it.
DEFAULT_CONFIDENCE = 0.25


class DetectionAdapter(ModelAdapter):
    """An Ultralytics detector behind the benchmark interface.

    Args:
        device: Where to place the weights.
        weights: Checkpoint path or a ``yolov8n.pt``-style short name.
        confidence: Minimum confidence for a reported box.
    """

    family = "detection"

    def __init__(
        self,
        device: torch.device,
        *,
        weights: str = "yolov8n.pt",
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        """Declare the model without loading it."""
        super().__init__(device)
        self.weights = weights
        self.confidence = float(confidence)
        self.name = "yolov8n"
        self.description = f"ultralytics:{weights} (conf>={self.confidence})"
        self._model: Any = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Fetch the checkpoint and move it to the device."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                self.name,
                "ultralytics is not installed (pip install ultralytics)",
            ) from exc

        try:
            model = YOLO(self.weights)
            model.to(self.device)
        except Exception as exc:
            raise ModelUnavailableError(
                self.name,
                f"could not load {self.weights}: {type(exc).__name__}: {exc}",
            ) from exc
        self._model = model
        self._loaded = True
        logger.info(
            "adapter.loaded name=%s weights=%s device=%s conf=%.2f",
            self.name,
            self.weights,
            self.device,
            self.confidence,
        )

    def release(self) -> None:
        """Drop the model and free the device."""
        self._model = None
        self._loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------

    def evaluate(self, image: RGB) -> DetectionObservation:
        """Detect objects in one ``uint8`` RGB image, best confidence first."""
        results = self._net().predict(
            source=np.ascontiguousarray(image),
            conf=self.confidence,
            verbose=False,
            device=str(self.device),
        )
        detections: list[Detection] = []
        for result in results:
            names: dict[int, str] = result.names or {}
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = names.get(int(box.cls[0]), str(int(box.cls[0])))
                # ``xyxy`` is a single row of four floats in source pixels.
                x1, y1, x2, y2 = (round(float(v)) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        box=(x1, y1, x2, y2),
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionObservation(detections=detections)

    def _net(self) -> Any:
        """The loaded model, or a loud error. Narrows the type for mypy."""
        self._require_loaded()
        if self._model is None:
            raise RuntimeError(f"{self.name}: loaded but the model is missing")
        return self._model


def build_detection(
    device: torch.device,
    *,
    weights: str = "yolov8n.pt",
    confidence: float = DEFAULT_CONFIDENCE,
) -> DetectionAdapter:
    """Construct a declared-but-unloaded detector."""
    return DetectionAdapter(device, weights=weights, confidence=confidence)
