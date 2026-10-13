"""Reproduce every number in the README with one command.

    python run_experiment.py                # the full grid
    python run_experiment.py --quick        # eps 8 only, 12 steps
    python run_experiment.py --skip-swap    # Steps 2, 3, 5 only

Each stage writes its own output, so a stage that fails leaves the earlier
results intact rather than losing the lot. ``--quick`` writes to a ``_quick``
suffix, so a smoke test can never overwrite the published full-grid numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EPS_SETS = {"full": [8, 12, 16, 24, 32], "quick": [8]}


def stage(name: str, command: list[str]) -> bool:
    print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"-- {name}: {status} in {elapsed:.1f}s", flush=True)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-swap", action="store_true")
    args = parser.parse_args()

    eps = EPS_SETS["quick" if args.quick else "full"]
    steps = 12 if args.quick else 40
    swap_eps = [8] if args.quick else [8, 16, 32]

    ok = stage(
        "Step 1: face detection",
        [sys.executable, "-W", "ignore", "-m", "src.step1_detect"],
    )
    if not ok:
        print("no photos detected; stopping")
        return 1

    # A quick smoke run must never overwrite the published full-grid numbers, so
    # it writes to a _quick suffix and leaves results.* alone.
    suffix = "_quick" if args.quick else ""

    ok &= stage(
        "Steps 2+3: attack grid and held-out evaluation",
        [sys.executable, "-W", "ignore", "-m", "src.experiment",
         "--eps", *[str(v) for v in eps], "--steps", str(steps),
         "--prefix", suffix],
    )

    if not args.skip_swap:
        for value in swap_eps:
            stage(
                f"Step 4: face-swap test at eps={value}",
                [sys.executable, "-W", "ignore", "-m", "src.swap_eval",
                 "--eps", str(value), "--prefix", suffix],
            )

    ok &= stage(
        "Step 5: tables and chart",
        [sys.executable, "-W", "ignore", "-m", "src.report", "--prefix", suffix],
    )

    print(f"\n{'=' * 70}")
    print("Outputs:")
    for name in (
        f"results{suffix}.csv",
        f"results{suffix}.md",
        f"summary_chart{suffix}.png",
        f"swap_results{suffix}.csv",
    ):
        path = ROOT / "outputs" / name
        print(f"  {'ok ' if path.is_file() else 'MISSING'} outputs/{name}")
    if suffix:
        print("\nThis was a quick run: the full results in results.csv are untouched.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())