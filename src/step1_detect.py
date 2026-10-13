"""Step 1: find every face in the corpus and save a localised mask.

Writes one preview per photo to ``outputs/step1/`` (original with the box and
the five landmarks drawn, plus the mask alone) and prints the skipped list, so
a photo that silently dropped out of the dataset is always visible.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.common import (
    ARTIFACTS_DIR,
    FACES_DIR,
    OUTPUTS_DIR,
    imwrite,
    list_images,
    set_seed,
)
from src.detect import DetectedFace, FaceDetector, face_mask

logging.basicConfig(
    level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s"
)
logger = logging.getLogger("step1")

STEP1_DIR = OUTPUTS_DIR / "step1"


def draw(image: np.ndarray, face: DetectedFace) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
    for index, (px, py) in enumerate(face.landmarks.astype(int)):
        cv2.circle(canvas, (px, py), 6, (0, 120, 255), -1)
        cv2.putText(
            canvas, str(index), (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2,
        )
    label = f"{face.detector} score={face.score:.3f}"
    cv2.putText(
        canvas, label, (x1, max(30, y1 - 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2,
    )
    return canvas


def main() -> int:
    set_seed()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STEP1_DIR.mkdir(parents=True, exist_ok=True)

    paths = list_images(FACES_DIR)
    if not paths:
        logger.error("no images under %s", FACES_DIR)
        return 1

    detector = FaceDetector()
    found: list[tuple[str, DetectedFace]] = []
    skipped: list[tuple[str, str]] = []

    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append((path.name, "unreadable"))
            continue
        face = detector.detect(image)
        if face is None:
            skipped.append((path.name, "no face above the score floor"))
            continue
        found.append((str(path), face))

    for path, face in found:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = face_mask(image.shape, face.bbox)
        stem = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        imwrite(STEP1_DIR / f"{stem}_box.jpg", draw(image, face))
        imwrite(
            STEP1_DIR / f"{stem}_mask.png",
            np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2),
        )
        np.save(ARTIFACTS_DIR / f"{stem}_mask.npy", mask)
        logger.info(
            "detected %s bbox=%s size=%dx%d score=%.3f via %s",
            stem, face.bbox, face.width, face.height, face.score, face.detector,
        )

    print(f"\nphotos scanned: {len(paths)}")
    print(f"faces found:    {len(found)}")
    print(f"skipped:        {len(skipped)}")
    for name, reason in skipped:
        print(f"  SKIPPED {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())