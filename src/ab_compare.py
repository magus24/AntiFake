"""A/B comparison: the imperceptibility changes, before and after.

Four things were changed together because they all serve the same goal:

===========================  ==============================================
luminance-only perturbation   one plane, written to R, G and B alike
post-upsample smoothing       ``smooth_sigma`` Gaussian on the upsampled delta
wider soft mask               ``mask_expand`` 0.25 -> 0.30, feather 0.35 -> 0.55
LPIPS term in the loss       ``lpips_weight`` (measured, see below)
===========================  ==============================================

The fourth one is reported as a negative result rather than enabled: PGD moves by
``gradient.sign()``, so a competing perceptual term cannot trade visibility
against attack strength, it only corrupts the sign pattern. The table includes
the LPIPS-weight row so the claim is checkable rather than asserted.

    python -m src.ab_compare
    python -m src.ab_compare --eps 8 16 --steps 20

Runs at the same working resolution the demo uses (``app.MAX_SIDE``) so the
numbers describe what a judge actually sees when clicking through the UI.
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

from src.attack import AttackConfig, SurrogateEnsemble, protect
from src.common import FACES_DIR, OUTPUTS_DIR, imread, list_images, set_seed
from src.detect import FaceDetector
from src.evaluate import ArcFaceEvaluator
from src.metrics import lpips_distance, psnr, ssim

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("ab")

MAX_SIDE = 1280

#: "before" is the configuration that produced the first version of the report.
BEFORE = dict(
    luminance_only=False,
    smooth_sigma=0.0,
    mask_expand=0.25,
    mask_feather=0.35,
    lpips_weight=0.0,
)
#: "after" is the shipped default.
AFTER = dict(
    luminance_only=True,
    smooth_sigma=0.8,
    mask_expand=0.30,
    mask_feather=0.55,
    lpips_weight=0.0,
)


def _bound(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _channel_spread(original: np.ndarray, protected: np.ndarray) -> int:
    """Worst per-pixel disagreement between the three colour planes.

    This is the direct measure of the colour bug: a luminance-only
    perturbation leaves only rounding and clipping residue, a chromatic one
    leaves up to 2*eps.
    """
    diff = protected.astype(np.int16) - original.astype(np.int16)
    return int(
        max(
            int(np.abs(diff[:, :, 0] - diff[:, :, 1]).max()),
            int(np.abs(diff[:, :, 1] - diff[:, :, 2]).max()),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ab_compare")
    parser.add_argument("--eps", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    set_seed(0)
    detector = FaceDetector()
    ensemble = SurrogateEnsemble()
    evaluator = ArcFaceEvaluator()

    photos = list_images(FACES_DIR)
    rows: list[dict] = []
    variants = [("before", BEFORE), ("after", AFTER)]
    if args.eps == [8] and args.steps == 20:
        variants.append(("after+lpips2", {**AFTER, "lpips_weight": 2.0}))

    for path in photos:
        stem = path.name.rsplit(".", 1)[0]
        image = _bound(imread(path))
        face = detector.detect(image)
        if face is None:
            logger.warning("no face in %s", stem)
            continue
        reference = evaluator.embed(image)
        if reference is None:
            logger.warning("ArcFace found no face in %s", stem)
            continue

        for eps in args.eps:
            for name, overrides in variants:
                config = AttackConfig(eps=eps / 255.0, steps=args.steps, **overrides)
                started = time.perf_counter()
                protected, _ = protect(image, face, ensemble, config)
                elapsed = time.perf_counter() - started
                row = {
                    "photo": stem[:34],
                    "variant": name,
                    "eps_255": eps,
                    "steps": args.steps,
                    "psnr_db": round(psnr(image, protected), 2),
                    "ssim": round(ssim(image, protected), 5),
                    "lpips": round(lpips_distance(image, protected) or -1.0, 4),
                    "arcface": round(
                        evaluator.similarity(reference, evaluator.embed(protected)) or -1.0,
                        4,
                    ),
                    "channel_spread": _channel_spread(image, protected),
                    "seconds": round(elapsed, 1),
                }
                rows.append(row)
                logger.info(
                    "%-34s eps=%-3d %-12s psnr=%6.2f lpips=%.4f arcface=%.3f "
                    "chroma=%2d (%.0fs)",
                    stem[:34], eps, name, row["psnr_db"], row["lpips"],
                    row["arcface"], row["channel_spread"], elapsed,
                )

    frame = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS_DIR / "ab_compare.csv", index=False)

    means = frame.groupby(["variant", "eps_255"]).mean(numeric_only=True)
    # eps-major, before-then-after, so each row pair reads as a tradeoff.
    order = [(v, e) for e in sorted(args.eps) for v, _ in variants]
    means = means.loc[[k for k in order if k in means.index]]
    lines = [
        "# A/B: изменения незаметности",
        "",
        f"Рабочее разрешение {MAX_SIDE}px по длинной стороне, "
        f"{args.steps} шагов PGD, среднее по {frame['photo'].nunique()} фото.",
        "",
        "| eps | вариант | PSNR, дБ | SSIM | LPIPS | ArcFace | разброс каналов |",
        "|---|---|---|---|---|---|---|",
    ]
    for (variant, eps), row in means.iterrows():
        lines.append(
            f"| {int(eps)} | {variant} | {row['psnr_db']:.2f} | {row['ssim']:.5f} | "
            f"{row['lpips']:.4f} | {row['arcface']:.3f} | {int(row['channel_spread'])} |"
        )
    lines += [
        "",
        "`разброс каналов` = максимальное расхождение между каналами B, G, R.",
        "Ненулевое значение = хроматический шум, то есть цветовой оттенок.",
    ]
    (OUTPUTS_DIR / "ab_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUTS_DIR / "ab_compare_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())