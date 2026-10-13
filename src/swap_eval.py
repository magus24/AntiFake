"""Step 4: does a real face-swap model fail when it is fed a protected face?

The direction of the swap is the whole point of this step, so it is spelled
out here.

``insightface inswapper.get(img, target_face, source_face)`` uses:

* ``target_face`` -- only its five landmarks. It decides *where* the face is
  pasted and what pose it has.
* ``source_face`` -- only its ArcFace embedding. **It decides whose identity
  appears in the output.**

The claim under test is "this photo resists being used as the source of a
face swap". So the protected photo must be the ``source_face``, and an
unrelated scene must be the ``target``. Swapping the arguments around measures
something else entirely: it measures whether a *stranger's* face can be pasted
into our photo, which never exercises our perturbation as an identity carrier.

For each photo we therefore build two swaps into the same donor scene -- one
whose identity comes from the original photo, one whose identity comes from the
protected photo -- and ask the held-out ArcFace model how strongly each result
still resembles the real person. If protection works, the second is lower.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.common import (
    ARTIFACTS_DIR,
    DONOR_DIR,
    FACES_DIR,
    OUTPUTS_DIR,
    SEED,
    imwrite,
    list_images,
    set_seed,
)
from src.detect import inswapper_path
from src.evaluate import ArcFaceEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
logger = logging.getLogger("swap")


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _best_face(app, image):
    for size in ((640, 640), (320, 320), (960, 960)):
        app.prepare(ctx_id=-1, det_size=size)
        try:
            found = app.get(image)
        except Exception:
            continue
        usable = [
            f for f in found if f.normed_embedding is not None and f.det_score >= 0.35
        ]
        if usable:
            return max(usable, key=lambda f: f.det_score)
    return None


def _fit(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), height),
        interpolation=cv2.INTER_AREA,
    )


def _bound(image: np.ndarray, max_side: int) -> np.ndarray:
    """Match the resolution the attack grid ran at, so original and protected
    artifacts are directly comparable here as well."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.swap_eval")
    parser.add_argument("--eps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all photos")
    parser.add_argument(
        "--max-side", type=int, default=1280,
        help="cap the long side, must match the attack grid",
    )
    parser.add_argument(
        "--prefix", default="",
        help="suffix for the output files, e.g. _quick writes swap_results_eps8_quick.json",
    )
    args = parser.parse_args()
    set_seed(SEED)

    from insightface import model_zoo
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = model_zoo.get_model(
        inswapper_path(), providers=["CPUExecutionProvider"]
    )
    evaluator = ArcFaceEvaluator()

    donors = list_images(DONOR_DIR)
    if not donors:
        logger.error("no donor scene under %s", DONOR_DIR)
        return 1
    scene = _read(donors[0])
    scene_face = _best_face(app, scene)
    if scene_face is None:
        logger.error("no usable face in the donor scene")
        return 1
    logger.info(
        "scene (target): %s face=%s score=%.3f",
        donors[0].name[:40], scene_face.bbox.astype(int).tolist(), scene_face.det_score,
    )

    examples = OUTPUTS_DIR / "examples"
    examples.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    paths = list_images(FACES_DIR)
    if args.limit:
        paths = paths[: args.limit]

    for path in paths:
        stem = path.name.rsplit(".", 1)[0]
        protected_path = ARTIFACTS_DIR / f"{stem}_eps{args.eps}.png"
        if not protected_path.is_file():
            logger.warning("no protected artifact for %s at eps=%s", stem, args.eps)
            continue

        original = _bound(_read(path), args.max_side)
        protected = _read(protected_path)

        # Identity carriers: one from the original photo, one from the protected.
        original_face = _best_face(app, original)
        protected_face = _best_face(app, protected)
        if original_face is None:
            logger.warning("no face in %s, skipping", stem)
            continue

        identity = evaluator.embed(original)
        row: dict = {"photo": stem, "eps_255": args.eps}
        from_original = from_protected = None

        if protected_face is None:
            logger.warning(
                "face no longer detectable in the protected image of %s -- "
                "a swap cannot be seeded from it", stem,
            )
            row["prot_source_available"] = False
        else:
            row["prot_source_available"] = True
            started = time.perf_counter()
            # target scene is the donor photo; the identity comes from our photos.
            from_original = swapper.get(scene, scene_face, original_face)
            from_protected = swapper.get(scene, scene_face, protected_face)
            row["swap_seconds"] = round(time.perf_counter() - started, 2)

            for tag, result in (
                ("orig", from_original),
                ("prot", from_protected),
            ):
                found = _best_face(app, result)
                embedding = (
                    None
                    if found is None
                    else np.asarray(found.normed_embedding, dtype=np.float32)
                )
                row[f"{tag}_face_detected"] = found is not None
                # The number that matters: does the output still carry the identity?
                row[f"{tag}_sim_to_identity"] = (
                    None
                    if embedding is None
                    else round(evaluator.similarity(embedding, identity) or 0.0, 4)
                )
                row[f"{tag}_sim_to_scene"] = (
                    None
                    if embedding is None
                    else round(
                        evaluator.similarity(embedding, scene_face.normed_embedding) or 0.0,
                        4,
                    )
                )

        rows.append(row)
        logger.info(
            "swap %s | identity in result: original=%.4f protected=%.4f",
            stem[:36],
            row.get("orig_sim_to_identity") or float("nan"),
            row.get("prot_sim_to_identity") or float("nan"),
        )

        if from_original is not None:
            imwrite(ARTIFACTS_DIR / f"{stem}_swap_from_original.jpg", from_original)
        if from_protected is not None:
            imwrite(ARTIFACTS_DIR / f"{stem}_swap_from_protected.jpg", from_protected)

        panels = [original, protected, from_original, from_protected]
        panels = [p for p in panels if p is not None]
        if len(panels) > 1:
            height = min(_fit(p, 900).shape[0] for p in panels)
            resized = [_fit(p, height) for p in panels]
            imwrite(
                examples / f"{stem}_swap_grid.jpg",
                np.concatenate(resized, axis=1),
            )

    payload = json.dumps(rows, indent=2)
    suffix = args.prefix or ""
    (OUTPUTS_DIR / f"swap_results_eps{args.eps}{suffix}.json").write_text(
        payload, encoding="utf-8"
    )
    (OUTPUTS_DIR / f"swap_results{suffix}.json").write_text(payload, encoding="utf-8")

    print("\nphoto                                  eps  from_original  from_protected  delta")
    for row in rows:
        a = row.get("orig_sim_to_identity")
        b = row.get("prot_sim_to_identity")
        delta = None if (a is None or b is None) else b - a
        print(
            "%-40s %3s  %13s  %14s  %6s"
            % (
                row["photo"][:40], row["eps_255"], a, b,
                "n/a" if delta is None else f"{delta:+.4f}",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())