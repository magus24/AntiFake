"""Face detection and face recognition as a benchmark adapter.

The classification models answer "what is this a picture of". A protection
aimed at a person is usually aimed at something else entirely: whether the
camera app still finds the face, and whether the login still recognises it.
Those are two different models with two different failure modes, and this
adapter deliberately runs both so the report can say which one moved.

Everything about ``facenet_pytorch`` is imported here and nowhere else, so the
rest of the benchmark -- and every test that does not need faces -- runs without
the dependency being installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.evaluation.base import FaceObservation, ModelAdapter, ModelUnavailableError
from app.protection.base import RGB

logger = logging.getLogger(__name__)

#: Resolution the embedding network was trained at. The crop is resized to this
#: before it reaches the network, not after, because InceptionResnetV1 has no
#: adaptive pooling that would forgive a different input size.
FACE_SIZE = 160

#: Default MTCNN thresholds for {stage: probability}. The middle value is the
#: one the detector actually gates on; raising it makes the probe easier to
#: defeat outright and would flatter the protection.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.6, 0.7, 0.7)


class FaceAdapter(ModelAdapter):
    """MTCNN for the box, InceptionResnetV1 for the descriptor.

    Args:
        device: Where to place the weights.
        thresholds: MTCNN's per-stage detection thresholds.
        min_confidence: Minimum detector confidence before a box is reported.
    """

    family = "face"

    def __init__(
        self,
        device: torch.device,
        *,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        min_confidence: float = 0.7,
    ) -> None:
        """Declare the model without loading it."""
        super().__init__(device)
        self.thresholds = tuple(thresholds)
        self.min_confidence = float(min_confidence)
        self.name = "mtcnn_facenet_v1"
        self.description = (
            "facenet-pytorch:MTCNN (thresholds="
            + "/".join(f"{t:.2f}" for t in self.thresholds)
            + ") + InceptionResnetV1(vggface2)"
        )
        self._detector: Any = None
        self._net: Any = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Build the detector and the embedding network."""
        if self._loaded:
            return
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                self.name,
                "facenet-pytorch is not installed (pip install facenet-pytorch)",
            ) from exc

        try:
            detector = MTCNN(
                image_size=FACE_SIZE,
                margin=0,
                keep_all=False,
                thresholds=list(self.thresholds),
                device=self.device,
            )
            network = InceptionResnetV1(pretrained="vggface2").to(self.device)
            network.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                self.name,
                f"could not load face weights: {type(exc).__name__}: {exc}",
            ) from exc

        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._detector = detector
        self._net = network
        self._loaded = True
        logger.info(
            "adapter.loaded name=%s device=%s thresholds=%s",
            self.name,
            self.device,
            self.thresholds,
        )

    def release(self) -> None:
        """Delete both networks and free the device."""
        self._detector = None
        self._net = None
        self._loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------

    def evaluate(self, image: RGB) -> FaceObservation:
        """Detect the largest face and describe it.

        The descriptor is L2-normalised, so the cosine similarity between two
        observations is their dot product and the comparison needs no
        normalisation of its own.
        """
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        boxes, probabilities = self._mtcnn().detect(pil, landmarks=False)

        if boxes is None or len(boxes) == 0:
            return FaceObservation(
                detected=False,
                confidence=None,
                box=None,
                embedding=None,
                reason="no face above the detector threshold",
            )

        boxes = np.atleast_2d(boxes)
        probabilities = (
            np.atleast_1d(probabilities)
            if probabilities is not None
            else np.zeros(len(boxes), dtype=np.float32)
        )
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence < self.min_confidence:
            return FaceObservation(
                detected=False,
                confidence=confidence,
                box=None,
                embedding=None,
                reason=(
                    f"best face confidence {confidence:.3f} is below "
                    f"{self.min_confidence:.3f}"
                ),
            )

        x1, y1, x2, y2 = (round(float(v)) for v in boxes[best])
        height, width = image.shape[0], image.shape[1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return FaceObservation(
                detected=False,
                confidence=confidence,
                box=None,
                embedding=None,
                reason="detector returned an empty box",
            )

        embedding = self._embed(image[y1:y2, x1:x2])
        return FaceObservation(
            detected=True,
            confidence=confidence,
            box=(x1, y1, x2, y2),
            embedding=embedding,
        )

    def _embed(self, crop: np.ndarray) -> np.ndarray:
        """Resize one face crop to the network's input and describe it."""
        resized = np.asarray(
            Image.fromarray(np.ascontiguousarray(crop), mode="RGB").resize(
                (FACE_SIZE, FACE_SIZE), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        # The network expects ``[-1, 1]`` rather than ImageNet's normalisation.
        tensor = torch.from_numpy(resized).permute(2, 0, 1) / 127.5 - 1.0
        with torch.no_grad():
            descriptor = self._resnet()(tensor.unsqueeze(0).to(self.device))
        descriptor = descriptor / descriptor.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        return descriptor.squeeze(0).cpu().numpy().astype(np.float32)

    def _mtcnn(self) -> Any:
        """The loaded detector, or a loud error. Narrows the type for mypy."""
        self._require_loaded()
        if self._detector is None:
            raise RuntimeError(f"{self.name}: loaded but the detector is missing")
        return self._detector

    def _resnet(self) -> Any:
        """The loaded embedding network, or a loud error."""
        self._require_loaded()
        if self._net is None:
            raise RuntimeError(f"{self.name}: loaded but the network is missing")
        return self._net


def build_face(
    device: torch.device,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_confidence: float = 0.7,
) -> FaceAdapter:
    """Construct a declared-but-unloaded face adapter."""
    return FaceAdapter(
        device, thresholds=thresholds, min_confidence=min_confidence
    )
