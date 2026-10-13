"""Steps 2 and 3: run the attack grid and evaluate it on the held-out model.

One command reproduces every number in the report. Each row is one
(photo, eps) pair and carries four independent views of the same image:

``arcface_*``     the held-out judge (insightface ArcFace, never optimised on)
``surrogate_*``   the attack's own objective, to separate "did not optimise"
                  from "optimised but did not transfer"
``gauss_*`` /     random noise with an identical L-inf budget, which is the
``uniform_*``     only honest way to claim the gain came from optimisation
``psnr/ssim/lpips`` the honesty check on the image itself
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.attack import AttackConfig, SurrogateEnsemble, protect, surrogate_similarity
from src.common import (
    ARTIFACTS_DIR,
    FACES_DIR,
    OUTPUTS_DIR,
    SEED,
    imwrite,
    list_images,
    set_seed,
)
from src.detect import FaceDetector, face_mask
from src.evaluate import (
    VERIFY_THRESHOLDS,
    ArcFaceEvaluator,
    gaussian_noise,
    uniform_noise,
)
from src.metrics import visual_quality

#: Working resolution shared with the demo. See :func:`_bound`.
MAX_SIDE = 1280

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
logger = logging.getLogger("experiment")

TAGS = ("clean", "jpeg75", "resize05")


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _bound(image: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, bool]:
    """Cap the long side at ``max_side``.

    The sample photographs are 22-megapixel originals. The attack works on a
    160x160 crop either way, so full resolution buys nothing for the result and
    costs seconds per LPIPS/SSIM/detection call. Running the benchmark at the
    same bounded resolution as the demo (``app.MAX_SIDE``) also means the numbers
    in ``results.md`` are the numbers a judge sees in the browser.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, False
    scale = max_side / longest
    return (
        cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        ),
        True,
    )


def run(
    eps_list: list[float],
    steps: int,
    work_size: int,
    eot_copies: int,
    max_side: int = MAX_SIDE,
    prefix: str = "",
) -> pd.DataFrame:
    set_seed(SEED)
    torch.set_num_threads(8)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = list_images(FACES_DIR)
    detector = FaceDetector()
    ensemble = SurrogateEnsemble()
    evaluator = ArcFaceEvaluator()

    rows: list[dict] = []
    for path in paths:
        original, _ = _bound(_read(path), max_side)
        face = detector.detect(original)
        if face is None:
            logger.warning("skipping %s: no face", path.name)
            continue
        stem = path.name.rsplit(".", 1)[0]
        # Same mask parameters the shipped attack uses, so the control noise and
        # the perturbation cover exactly the same pixels.
        mask = face_mask(original.shape, face.bbox, expand=0.30, feather=0.55)
        reference = evaluator.embed(original)
        surrogate_reference = surrogate_similarity(
            original, original, face, ensemble,
            AttackConfig(work_size=work_size),
        )
        logger.info(
            "photo %s | face=%s score=%.3f | arcface_self=%.4f",
            stem[:40], face.bbox, face.score,
            evaluator.similarity(reference, reference) or float("nan"),
        )

        for eps255 in eps_list:
            config = AttackConfig(
                eps=eps255 / 255.0,
                steps=steps,
                work_size=work_size,
                eot_copies=eot_copies,
                seed=SEED,
            )
            started = time.perf_counter()
            protected, info = protect(original, face, ensemble, config)
            attack_seconds = time.perf_counter() - started

            row: dict = {
                "photo": stem,
                "eps_255": eps255,
                "steps": steps,
                "work_size": work_size,
                "eot_copies": eot_copies,
                "seed": SEED,
                "det_score": round(face.score, 4),
                "attack_seconds": round(attack_seconds, 2),
                "surrogate_self": round(surrogate_reference, 4),
            }
            row["surrogate_cos"] = round(
                surrogate_similarity(
                    original, protected, face, ensemble, config
                ),
                4,
            )

            quality = visual_quality(original, protected)
            row.update(
                psnr_db=quality["psnr_db"],
                ssim=quality["ssim"],
                lpips=quality["lpips"],
                linf_255=quality["linf_255"],
                mask_coverage=round(info["mask_coverage"], 4),
            )

            for tag in TAGS:
                row[f"arcface_{tag}"] = (
                    None
                    if (value := evaluator.similarity_after(reference, protected, tag))
                    is None
                    else round(value, 4)
                )

            for name, maker in (
                ("gauss", gaussian_noise),
                ("uniform", uniform_noise),
            ):
                control = maker(original, mask, eps255, SEED)
                row[f"{name}_clean"] = (
                    None
                    if (value := evaluator.similarity_after(reference, control, "clean"))
                    is None
                    else round(value, 4)
                )
                row[f"{name}_psnr_db"] = visual_quality(original, control)["psnr_db"]

            for threshold in VERIFY_THRESHOLDS:
                row[f"reject@{threshold}"] = (
                    row["arcface_clean"] is not None
                    and row["arcface_clean"] < threshold
                )

            rows.append(row)
            logger.info(
                "  eps=%-3g attack=%.1fs linf=%d psnr=%.2f | surrogate=%.4f "
                "arcface=%.4f gauss=%.4f",
                eps255, attack_seconds, quality["linf_255"], quality["psnr_db"],
                row["surrogate_cos"], row["arcface_clean"] or float("nan"),
                row["gauss_clean"] or float("nan"),
            )

            imwrite(ARTIFACTS_DIR / f"{stem}_eps{int(eps255)}.png", protected)

    frame = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = prefix or ""
    frame.to_csv(OUTPUTS_DIR / f"results{suffix}.csv", index=False)
    (OUTPUTS_DIR / f"results{suffix}_rows.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.experiment")
    parser.add_argument("--eps", type=float, nargs="*", default=[8, 12, 16])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--work-size", type=int, default=160)
    parser.add_argument("--eot-copies", type=int, default=2)
    parser.add_argument(
        "--max-side", type=int, default=MAX_SIDE,
        help="cap the long side (0 = use the original resolution)",
    )
    parser.add_argument(
        "--prefix", default="",
        help="suffix for the output files, e.g. _quick writes results_quick.csv",
    )
    args = parser.parse_args()
    frame = run(
        args.eps,
        args.steps,
        args.work_size,
        args.eot_copies,
        args.max_side if args.max_side > 0 else 10 ** 6,
        args.prefix,
    )
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())