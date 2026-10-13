"""Step 3: the held-out evaluator and the random-noise control.

Everything in this module is deliberately **not** used to build the
perturbation. The attack optimises against InceptionResnetV1 trained on
vggface2 and casia-webface; the numbers that decide whether the project works
come from insightface's ArcFace (buffalo_l), a different architecture and a
different training set, run as a black box.

The random-noise control exists for one reason: to show that any drop in
similarity is caused by the *adversarial optimisation* and not by the fact
that a pixel-perfect photo is slightly harder to embed than a clean one.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# insightface's ArcFace models are matched-identity at roughly 0.28-0.4 cosine.
# Both ends of that range are reported so the claim does not rest on one number.
VERIFY_THRESHOLDS: tuple[float, ...] = (0.28, 0.4)


def jpeg_roundtrip(image: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("jpeg encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def downscale_roundtrip(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Halve the resolution and bring it back -- what a re-post does."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def uniform_noise(
    image: np.ndarray, mask: np.ndarray, eps255: float, seed: int
) -> np.ndarray:
    """Random noise with exactly the same L-inf budget as the attack."""
    generator = np.random.default_rng(seed)
    shape = (image.shape[0], image.shape[1], image.shape[2])
    noise = generator.uniform(-eps255, eps255, size=shape)
    out = image.astype(np.float32) + noise * mask[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def gaussian_noise(
    image: np.ndarray, mask: np.ndarray, eps255: float, seed: int
) -> np.ndarray:
    """Gaussian noise with the same L-inf budget (sigma = eps/2, clipped)."""
    generator = np.random.default_rng(seed)
    shape = (image.shape[0], image.shape[1], image.shape[2])
    noise = np.clip(
        generator.normal(0.0, eps255 / 2.0, size=shape), -eps255, eps255
    )
    out = image.astype(np.float32) + noise * mask[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class ArcFaceEvaluator:
    """Black-box face recognition on insightface ArcFace (buffalo_l)."""

    def __init__(self, det_sizes: tuple[tuple[int, int], ...] = ((640, 640), (320, 320), (960, 960))) -> None:
        from insightface.app import FaceAnalysis

        self.det_sizes = det_sizes
        self.app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=det_sizes[0])

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        """512-d L2-normalised embedding, or ``None`` if no face is found."""
        for size in self.det_sizes:
            self.app.prepare(ctx_id=-1, det_size=size)
            try:
                faces = self.app.get(image)
            except Exception:
                continue
            usable = [
                f for f in faces
                if f.normed_embedding is not None and f.det_score >= 0.35
            ]
            if usable:
                return np.asarray(
                    max(usable, key=lambda f: f.det_score).normed_embedding,
                    dtype=np.float32,
                )
        return None

    @staticmethod
    def similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
        if left is None or right is None:
            return None
        return float(np.clip(np.dot(left, right), -1.0, 1.0))

    def similarity_after(
        self, reference: np.ndarray | None, image: np.ndarray, tag: str
    ) -> float | None:
        """Cosine similarity of ``image`` under ``tag``'s post-processing."""
        processed = {
            "clean": image,
            "jpeg75": jpeg_roundtrip(image, 75),
            "resize05": downscale_roundtrip(image, 0.5),
        }[tag]
        return self.similarity(reference, self.embed(processed))