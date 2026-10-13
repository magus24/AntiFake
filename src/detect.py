"""Face detection and the soft mask that localises the perturbation.

Two detectors are used on purpose:

* ``insightface`` buffalo_l SCRFD is the primary one. It returns the box, the
  five landmarks and the ArcFace embedding used later as the held-out
  evaluator, so it is the detector that matches the model we will be judged
  against.
* ``facenet-pytorch`` MTCNN is the fallback. It is a genuinely different
  architecture, so when the two agree the detection is not an artefact of one
  model.

The multi-scale sweep is not paranoia. On a real 736x736 donor photograph
buffalo_l with the usual ``det_size=(640, 640)`` reports **zero** faces, while
``(320, 320)`` finds the face at score 0.88 and MTCNN confirms it at 0.99. A
single-scale detector would have silently dropped that photograph from the
dataset, which is the one thing a benchmark must never do quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.common import MODELS_DIR

DET_SIZES: tuple[tuple[int, int], ...] = ((640, 640), (320, 320), (960, 960))
MIN_DET_SCORE = 0.35


@dataclass(frozen=True)
class DetectedFace:
    """One face, in the coordinate system of the image it was found in."""

    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray  # (5, 2) float32: eyes, nose, mouth corners
    score: float
    detector: str

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def face_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    expand: float = 0.25,
    feather: float = 0.35,
) -> np.ndarray:
    """A feathered mask that covers ``bbox`` and fades to zero outside it.

    Perturbing only inside the mask keeps the background bit-for-bit clean,
    which is both the honest reading of "protect the face" and what makes the
    visual-quality numbers mean something.
    """
    height, width = shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int(round((x2 - x1) * expand))
    pad_y = int(round((y2 - y1) * expand))
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
        (max(1, (cx2 - cx1) // 2), max(1, (cy2 - cy1) // 2)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    # Feather width scales with the face so a small face and a close-up both
    # get a soft edge instead of a hard cut. The kernel is capped so that a very
    # large face in a very large photo cannot turn the blur into the most
    # expensive operation in the pipeline.
    blur = max(3, int(min(cx2 - cx1, cy2 - cy1) * feather) | 1)
    blur = min(blur, 401)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask, 0.0, 1.0)


class FaceDetector:
    """Lazily loaded, multi-scale face detector."""

    def __init__(self) -> None:
        self._insightface = None
        self._mtcnn = None

    def _app(self):
        if self._insightface is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            app.prepare(ctx_id=-1, det_size=DET_SIZES[0])
            self._insightface = app
        return self._insightface

    def _fallback(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(
                image_size=160, margin=10, keep_all=True, device="cpu"
            )
        return self._mtcnn

    def detect(self, image: np.ndarray) -> DetectedFace | None:
        """Best-scoring face in ``image``, or ``None`` if nothing was found."""
        app = self._app()
        best: DetectedFace | None = None
        for size in DET_SIZES:
            app.prepare(ctx_id=-1, det_size=size)
            try:
                faces = app.get(image)
            except Exception:  # a detector that crashes must not kill the run
                continue
            for face in faces:
                if face.det_score < MIN_DET_SCORE:
                    continue
                if face.kps is None or len(face.kps) != 5:
                    continue
                candidate = DetectedFace(
                    bbox=tuple(int(v) for v in face.bbox),
                    landmarks=np.asarray(face.kps, dtype=np.float32),
                    score=float(face.det_score),
                    detector=f"buffalo_l@{size[0]}",
                )
                if best is None or candidate.score > best.score:
                    best = candidate
            if best is not None:
                return best

        return self._detect_with_mtcnn(image, best)

    def _detect_with_mtcnn(
        self, image: np.ndarray, incumbent: DetectedFace | None
    ) -> DetectedFace | None:
        try:
            boxes, probs, landmarks = self._fallback().detect(
                image, landmarks=True
            )
        except Exception:
            return incumbent
        if boxes is None or probs is None:
            return incumbent
        for box, prob, lms in zip(boxes, probs, landmarks):
            if prob is None or float(prob) < MIN_DET_SCORE:
                continue
            candidate = DetectedFace(
                bbox=tuple(int(round(v)) for v in box[:4]),  # type: ignore[arg-type]
                landmarks=np.asarray(lms, dtype=np.float32),
                score=float(prob),
                detector="mtcnn",
            )
            if incumbent is None or candidate.score > incumbent.score:
                incumbent = candidate
        return incumbent


def inswapper_path() -> str:
    """Locate ``inswapper_128.onnx``; raise a clear error if it is absent."""
    path = MODELS_DIR / "inswapper_128.onnx"
    if not path.is_file():
        raise FileNotFoundError(
            f"face-swap weights missing at {path}. The face-swap step is "
            "optional; without them the report carries ArcFace metrics only."
        )
    return str(path)