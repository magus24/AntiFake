"""CLI entry point for the Phase 1 benchmark.

Usage::

    python -m app.evaluation.run --image ../../photo.jpg --out ../benchmark_out
    python -m app.evaluation.run --image a.png --image b.jpg --steps 60 --device cpu
    python -m app.evaluation.run --image a.jpg --models resnet18 clip_vit_b_32

Exit codes are meaningful: ``0`` means the run completed (individual models
may still be skipped, and the document says which), ``1`` means the run itself
failed and no usable document was produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.evaluation.adapters_embedding import DEFAULT_PROMPTS
from app.evaluation.base import ModelSpec
from app.evaluation.benchmark import (
    BenchmarkConfig,
    markdown_report,
    run_benchmark,
    write_report,
)
from app.evaluation.registry import DEFAULT_MODELS
from app.protection.config import MAX_PGD_STEPS, ProtectionConfig

logger = logging.getLogger("app.evaluation.run")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.run",
        description="Run the AntiFake Phase 1 transferability benchmark.",
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        type=Path,
        help="Source image. Repeat for several.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_out"),
        help="Output directory for the JSON, Markdown and artifacts.",
    )
    parser.add_argument("--steps", type=int, default=40, help="PGD steps.")
    parser.add_argument(
        "--eps", type=float, default=8.0, help="L-inf bound in 8-bit levels."
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0, help="Step size in 8-bit levels."
    )
    parser.add_argument("--seed", type=int, default=0, help="Protection seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Protection and evaluation device.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model keys to evaluate. Default: all.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Text probes for the CLIP adapter.",
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Skip the robustness transform sweep.",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Store raw embedding vectors in the JSON (large).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def _select_models(keys: list[str] | None) -> tuple[ModelSpec, ...]:
    if not keys:
        return DEFAULT_MODELS
    by_key = {spec.key: spec for spec in DEFAULT_MODELS}
    missing = [k for k in keys if k not in by_key]
    if missing:
        available = sorted(by_key)
        raise SystemExit(
            f"Unknown model key(s) {missing}. Available: {available}"
        )
    return tuple(by_key[k] for k in keys)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmark, write both report formats."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    missing = [str(p) for p in args.image if not p.is_file()]
    if missing:
        raise SystemExit(f"Image not found: {missing}")

    steps = max(1, min(args.steps, MAX_PGD_STEPS))
    config = BenchmarkConfig(
        image_paths=tuple(args.image),
        output_dir=args.out,
        protection=ProtectionConfig(
            eps=args.eps / 255.0,
            alpha=args.alpha / 255.0,
            steps=steps,
            seed=args.seed,
            device=args.device,
        ),
        models=_select_models(args.models),
        prompts=tuple(args.prompts),
        device=args.device,
        include_embeddings=args.include_embeddings,
        transforms=() if args.no_transforms else None,
    )

    try:
        document = run_benchmark(config)
    except Exception:
        logger.exception("benchmark.failed")
        return 1

    json_path = write_report(document, args.out / "benchmark.json")
    md_path = args.out / "benchmark.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_report(document), encoding="utf-8")

    summary = document["summary"]
    print(f"\nJSON:  {json_path}")
    print(f"Markdown: {md_path}")
    print(
        f"Models: {summary['models_ok']} ok, {summary['models_skipped']} skipped, "
        f"{summary['models_failed']} failed"
    )
    print(f"PNG label flips: {summary['png_label_flips']}")
    print(f"PNG confidence collapses: {summary['png_confidence_collapses']}")
    if summary["models_failed"]:
        logger.warning(
            "benchmark.models_failed keys=%s", sorted(
                k for k, m in document["models"].items() if m["status"] == "failed"
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
