"""Diagnose why the perturbation does not reach the face-swap network.

Hypothesis: inswapper never looks at the full-resolution photo. It aligns the
face to a 128x128 crop, which for a 1400-pixel-wide face averages each output
pixel over roughly 11x11 input pixels. A random-ish L-inf=8/255 perturbation is
averaged away by that.

This measures the amplitude that actually survives into the crop, so the claim
is a number rather than a story.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.utils import face_align  # noqa: E402

from src.attack import AttackConfig, SurrogateEnsemble, protect  # noqa: E402
from src.common import ARTIFACTS_DIR, FACES_DIR, list_images  # noqa: E402
from src.detect import FaceDetector  # noqa: E402


def main() -> int:
    paths = list_images(FACES_DIR)
    detector = FaceDetector()
    ensemble = SurrogateEnsemble()

    print(f"{'photo':<30} {'linf_full':>9} {'linf_128crop':>13} {'survives':>9}")
    for path in paths[:3]:
        stem = path.name[:28]
        image = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        face = detector.detect(image)
        protected_path = ARTIFACTS_DIR / f"{path.name.rsplit('.', 1)[0]}_eps8.png"
        if not protected_path.is_file():
            config = AttackConfig(eps=8 / 255, steps=40, seed=0)
            protected, _ = protect(image, face, ensemble, config)
        else:
            protected = cv2.imdecode(
                np.fromfile(str(protected_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )

        diff_full = np.abs(
            protected.astype(np.int16) - image.astype(np.int16)
        ).max()

        # The exact crop the swapper feeds its network.
        aligned_full = face_align.norm_crop(
            protected, landmark=face.landmarks, image_size=128
        )
        aligned_orig = face_align.norm_crop(
            image, landmark=face.landmarks, image_size=128
        )
        diff_crop = np.abs(
            aligned_full.astype(np.int16) - aligned_orig.astype(np.int16)
        ).max()

        survives = diff_crop / diff_full if diff_full else 0.0
        print(f"{stem:<30} {diff_full:>9} {diff_crop:>13} {survives:>9.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())