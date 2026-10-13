"""CLI entry point for the four-way Phase 2 comparison.

Usage::

    python -m app.evaluation.run_phases --out ../benchmark_out/phase2
    python -m app.evaluation.run_phases --image ../../photo.jpg --eot-samples 8
    python -m app.evaluation.run_phases --variants phase1 eot --device cpu

With no ``--image`` the default corpus is resolved (``$ANTIFAKE_CORPUS_DIR``,
else captured photographs in the project root), which is the intended way to
run it: the corpus is part of the experiment, not a per-run choice.

Exit codes match the Phase 1 CLI: ``0`` means the comparison completed and a
document was written (individual models may still be skipped, and the document
says which), ``1`` means the run failed and produced nothing usable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.evaluation.phases import (
    DEFAULT_EOT_SAMPLES,
    VARIANT_KEYS,
    PhaseComparisonConfig,
    run_phase_comparison,
)
from app.evaluation.report_phase2 import write_phase2_report
from app.protection.config import MAX_PGD_STEPS, ProtectionConfig

logger = logging.getLogger("app.evaluation.run_phases")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.run_phases",
        description=(
            "Compare Phase 1, EOT, Ensemble and EOT+Ensemble with everything "
            "else held fixed."
        ),
    )
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=None,
        help="Source image. Repeat for several. Default: the project corpus.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_out/phase2"),
        help="Output directory for the report, document and artifacts.",
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
        "--eot-samples",
        type=int,
        default=DEFAULT_EOT_SAMPLES,
        help="Renderings per step for the arms that use EOT.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANT_KEYS),
        choices=list(VARIANT_KEYS),
        help="Which arms to run. Default: all four.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Protection and evaluation device.",
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Measure delivery only, skipping the robustness sweep.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the four arms, write the report and the evidence."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    steps = max(1, min(args.steps, MAX_PGD_STEPS))
    config = PhaseComparisonConfig(
        output_dir=args.out,
        image_paths=tuple(args.image) if args.image else (),
        variants=tuple(args.variants),
        base=ProtectionConfig(
            eps=args.eps / 255.0,
            alpha=args.alpha / 255.0,
            steps=steps,
            seed=args.seed,
            device=args.device,
        ),
        eot_samples=args.eot_samples,
        transforms=() if args.no_transforms else None,
        device=args.device,
    )

    try:
        document = run_phase_comparison(config)
    except Exception:
        logger.exception("phase_comparison.failed")
        return 1

    report_path = write_phase2_report(document, args.out / "report.md")
    print(f"\nReport:  {report_path}")
    print(f"Evidence: {report_path.with_suffix('.json')}")
    print(f"Arms:     {', '.join(document['run']['variants'])}")
    print(f"Wall time: {document['run']['seconds']} s on {document['run']['device']}")
    for key in document["run"]["variants"]:
        entry = document["variants"][key]["document"]
        headline = (entry.get("transfer") or {}).get("headline") or {}
        held_out = headline.get("held_out") or {}
        rate = headline.get("ensemble_transfer_success")
        rendered = "n/a" if rate is None else f"{100.0 * rate:.1f}%"
        print(
            f"  {key:<14} held-out collapse {rendered:>6} "
            f"({held_out.get('evaluated', 0)}/{held_out.get('expected', 0)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
