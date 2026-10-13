"""Smoke test: does the attack move the held-out ArcFace embedding at all?

Ten steps, one photo, 112x112. This exists to fail fast -- if the direction of
the attack is wrong, nothing downstream matters.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
logging.getLogger().setLevel(logging.ERROR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack import AttackConfig, SurrogateEnsemble, protect  # noqa: E402
from src.common import FACES_DIR, imwrite, list_images  # noqa: E402
from src.detect import FaceDetector  # noqa: E402
from src.metrics import visual_quality  # noqa: E402


def main() -> int:
    from insightface.app import FaceAnalysis

    paths = list_images(FACES_DIR)
    print(f"photos: {len(paths)}")
    path = paths[0]
    print(f"using: {path.name}")

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    detector = FaceDetector()
    face = detector.detect(image)
    print(f"face bbox={face.bbox} score={face.score:.3f} via {face.detector}")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def arcface(img):
        found = app.get(img)
        if not found:
            return None
        return max(found, key=lambda f: f.det_score).normed_embedding

    before = arcface(image)

    ensemble = SurrogateEnsemble()
    config = AttackConfig(eps=8 / 255, steps=10, work_size=112, eot_copies=2, seed=0)
    started = time.perf_counter()
    protected, info = protect(image, face, ensemble, config)
    elapsed = time.perf_counter() - started

    after = arcface(protected)
    quality = visual_quality(image, protected)

    print(f"time: {elapsed:.1f}s for 10 steps at 112x112")
    print(f"info: {info}")
    print(f"visual: {quality}")
    if before is not None and after is not None:
        similarity = float(before @ after)
        print(f"ArcFace cos(orig, protected) = {similarity:.4f} (was 1.0000)")
        print(f"drop = {1.0 - similarity:.4f}")
    else:
        print("ArcFace: face not re-detected in the protected image")
    imwrite("outputs/_smoke_prot.jpg", protected)
    print("wrote outputs/_smoke_prot.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())