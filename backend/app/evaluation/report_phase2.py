"""Markdown rendering of a four-way Phase 2 comparison.

The document produced by :mod:`app.evaluation.phases` is complete on its own;
this module exists to make it *readable* in the specific way the comparison
demands. Four questions are answered, in order:

1. Did transfer to models the attack never saw get better? -- the held-out
   table, which is the headline and appears before anything else.
2. Did the probes agree? -- face, detection, CLIP and VLM reported separately,
   because they measure different things and a single "robustness" column
   would hide which one moved.
3. Did it survive being sent as a file? -- the robustness table, with the
   screenshot-like row pulled out and paired with its visual quality numbers,
   since that is the claim most likely to be made about it.
4. What did it cost? -- runtime and peak device memory per variant, next to
   the configuration that produced them.

Everything that could not be measured prints as ``n/a`` rather than ``0``. A
run in which the VLM was skipped must not look like a run in which the VLM saw
nothing wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Ordered families for the probe table. `text` is the VLM.
PROBE_FAMILIES: tuple[tuple[str, str], ...] = (
    ("face", "Face (MTCNN + FaceNet)"),
    ("detection", "Detection (YOLOv8n)"),
    ("embedding", "CLIP prompts"),
    ("text", "VLM"),
)

#: The transform most often claimed as "survives being sent", pulled out of the
#: robustness table so it cannot be lost among sixteen rows.
SCREENSHOT_TRANSFORM = "screenshot_like"


def _fmt_rate(value: Any) -> str:
    """A rate as a percentage, or ``n/a`` when nothing was measured.

    ``None`` and ``0`` are deliberately different here: the first means the
    branch did not run, the second means it ran and nothing happened.
    """
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    """A number to a fixed precision, or ``n/a`` without one."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A GitHub-flavoured markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _first_attack(document: dict[str, Any]) -> dict[str, Any]:
    """The attack entry for the run's first image, or an empty mapping."""
    attack = document.get("attack") or {}
    if not attack:
        return {}
    return attack[next(iter(attack))]


def _variant_label(document: dict[str, Any]) -> str:
    """The engine's own configuration label, when one was recorded."""
    return str(_first_attack(document).get("variant") or "-")


def _headline(document: dict[str, Any]) -> dict[str, Any]:
    """The held-out summary for the delivered PNG, or an empty mapping."""
    return ((document.get("transfer") or {}).get("headline")) or {}


def _family(headline: dict[str, Any], family: str) -> dict[str, Any]:
    """One probe family's bucket from a held-out summary."""
    value = headline.get(family)
    return value if isinstance(value, dict) else {}


def _status_counts(document: dict[str, Any]) -> dict[str, int]:
    """Ok / skipped / failed across every declared model."""
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for result in document.get("models", {}).values():
        status = str(result.get("status", "failed"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _skip_reasons(document: dict[str, Any]) -> list[tuple[str, str]]:
    """Models that did not run, with the stated reason, sorted by name."""
    rows = [
        (key, str(result.get("reason", "")))
        for key, result in document.get("models", {}).items()
        if result.get("status") != "ok"
    ]
    return sorted(rows)


def _visual(document: dict[str, Any], artifact_key: str) -> dict[str, Any]:
    """Visual quality for one artifact key, or an empty mapping."""
    return document.get("visual_quality", {}).get(artifact_key) or {}


def _artifact_key(document: dict[str, Any], suffix: str) -> str:
    """The artifact key ending in ``suffix`` for the first image, if any."""
    for key in document.get("comparison", {}):
        if key.endswith(suffix):
            return key
    return ""


def _rate(summary: dict[str, Any], field: str) -> Any:
    """Pull a headline rate, tolerating an absent summary."""
    held_out = summary.get("held_out") or {}
    return held_out.get(field)


def _fraction(bucket: dict[str, Any]) -> str:
    """``evaluated/expected`` for a bucket, with ``expected`` from the model list.

    A bucket reports its own ``expected`` when it has one; otherwise the
    declared models it carries are counted, so a group that has not been given
    an expectation still reads as a count rather than a division by zero.
    """
    evaluated = int(bucket.get("evaluated", 0) or 0)
    expected = bucket.get("expected")
    if expected is None:
        expected = len(bucket.get("models", []) or [])
    return f"{evaluated}/{int(expected)}"


def phase2_report(document: dict[str, Any]) -> str:
    """Render one comparison document as markdown.

    Args:
        document: Output of :func:`app.evaluation.phases.run_phase_comparison`.

    Returns:
        Markdown text, ready to write to a file.
    """
    variants: dict[str, Any] = document.get("variants", {})
    order = document.get("run", {}).get("variants", list(variants))
    titles = {
        key: (variants.get(key, {}).get("variant") or {}).get("title", key)
        for key in order
    }

    out: list[str] = []
    out.append("# AntiFake Phase 2: four-way protection comparison")
    out.append("")
    out.append(
        "Every arm shares the perturbation budget, the optimiser, the seed, the "
        "source images, the transform suite and the model set. The only "
        "differences are the two switches named below, so a difference in the "
        "outcome is attributable to them."
    )
    out.append("")

    # -- run header --------------------------------------------------------
    run = document.get("run", {})
    out.append("## Run")
    out.append("")
    hardware = run.get("hardware", {}) or {}
    out.extend(
        _table(
            ["field", "value"],
            [
                ["generated", str(document.get("generated_at", "-"))],
                ["device", str(run.get("device", "-"))],
                ["device reason", str(run.get("device_reason", "-"))],
                ["device name", str(hardware.get("name", "-"))],
                ["total wall time", f"{_fmt_number(run.get('seconds'), digits=1)} s"],
                ["variants", ", ".join(str(k) for k in order)],
            ],
        )
    )
    out.append("")

    config = document.get("config", {}) or {}
    out.append("Configuration, identical across arms:")
    out.append("")
    out.extend(
        _table(
            ["field", "value"],
            [
                ["EOT samples per step (EOT arms)", str(config.get("eot_samples"))],
                ["surrogates (ensemble arms)", ", ".join(config.get("surrogates", []))],
                ["transforms measured", str(len(config.get("transforms", [])))],
                ["CLIP prompts", str(config.get("prompts"))],
                ["models measured", str(len(config.get("models", [])))],
            ],
        )
    )
    out.append("")

    # -- corpus ------------------------------------------------------------
    corpus = document.get("corpus", []) or []
    out.append("## Corpus")
    out.append("")
    out.extend(
        _table(
            ["key", "provenance", "face evidence"],
            [
                [
                    str(image.get("key")),
                    str(image.get("kind")),
                    "allowed" if image.get("face_eligible") else "forbidden",
                ]
                for image in corpus
            ],
        )
    )
    out.append("")
    out.append(
        "Face and detection verdicts are only drawn from photographs. A "
        "synthetic image never contributes a face number, because a detector "
        "reading a generated gradient says nothing about hiding a person."
    )
    out.append("")

    # -- cost --------------------------------------------------------------
    out.append("## Cost: runtime and peak device memory")
    out.append("")
    cost_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        attack = _first_attack(document_for)
        cost_rows.append(
            [
                titles.get(key, key),
                _variant_label(document_for),
                f"{_fmt_number(attack.get('attack_seconds'), digits=2)} s",
                f"{_fmt_number(document_for.get('run', {}).get('seconds'), digits=1)} s",
                (
                    f"{_fmt_number(attack.get('peak_memory_mib'), digits=1)} MiB"
                    if attack.get("peak_memory_mib") is not None
                    else "n/a"
                ),
                "yes" if attack.get("oom_fallback") else "no",
            ]
        )
    out.extend(
        _table(
            [
                "variant",
                "engine label",
                "protection",
                "whole run",
                "peak VRAM",
                "OOM fallback",
            ],
            cost_rows,
        )
    )
    out.append("")
    out.append(
        "``protection`` is time inside the attack only; ``whole run`` also "
        "includes loading and evaluating every model. Peak VRAM is measured "
        "during the attack and is ``0.0`` on CPU, where there is nothing to "
        "measure."
    )
    out.append("")

    # -- headline ----------------------------------------------------------
    out.append("## Held-out transfer (the headline)")
    out.append("")
    out.append(
        "Computed over classification models that were deliberately kept out "
        "of every objective. Surrogate and target rows appear below as "
        "diagnostics only."
    )
    out.append("")
    held_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        headline = _headline(document_for)
        held_out = headline.get("held_out") or {}
        held_rows.append(
            [
                titles.get(key, key),
                _fmt_rate(headline.get("ensemble_transfer_success")),
                _fmt_rate(headline.get("ensemble_transfer_success_strict")),
                _fraction(held_out),
                ", ".join(held_out.get("models", []) or []) or "-",
            ]
        )
    out.extend(
        _table(
            [
                "variant",
                "collapse",
                "label flip",
                "measured",
                "held-out models",
            ],
            held_rows,
        )
    )
    out.append("")
    out.append(
        "``collapse`` is a confidence drop of at least 75%; ``label flip`` is "
        "the stricter verdict requiring a different top-1 class. Rates are "
        "``n/a`` when nothing in the group was measured."
    )
    out.append("")

    # -- diagnostics -------------------------------------------------------
    out.append("## Surrogate and target diagnostics")
    out.append("")
    out.append(
        "These models *were* optimised against. A drop here shows the "
        "optimiser worked; it is not evidence that the perturbation transferred."
    )
    out.append("")
    diag_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        headline = _headline(document_for)
        surrogates = headline.get("surrogates") or {}
        target = headline.get("target") or {}
        diag_rows.append(
            [
                titles.get(key, key),
                _fmt_rate(surrogates.get("collapse_rate")),
                _fraction(surrogates),
                _fmt_rate(target.get("collapse_rate")),
            ]
        )
    out.extend(
        _table(
            ["variant", "surrogate collapse", "measured", "target collapse"],
            diag_rows,
        )
    )
    out.append("")

    # -- probes ------------------------------------------------------------
    out.append("## Probes: face, detection, CLIP, VLM")
    out.append("")
    out.append(
        "Reported one family per column because none of them has a confidence "
        "to collapse. Each cell is that family's own metric, and an absent "
        "measurement prints ``n/a``."
    )
    out.append("")
    probe_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        headline = _headline(document_for)
        face = _family(headline, "face")
        detection = _family(headline, "detection")
        clip = _family(headline, "embedding")
        text = _family(headline, "text")
        probe_rows.append(
            [
                titles.get(key, key),
                (
                    f"{_fmt_rate(face.get('loss_rate'))} lost, "
                    f"cos {_fmt_number(face.get('mean_embedding_cosine'), digits=3)}"
                ),
                (
                    f"{_fmt_rate(detection.get('loss_rate'))} lost, "
                    f"dΔ {_fmt_number(detection.get('mean_count_delta'), digits=2)}"
                ),
                (
                    f"{_fmt_rate(clip.get('prompt_change_rate'))} changed, "
                    f"cos {_fmt_number(clip.get('mean_cosine_similarity'), digits=3)}"
                ),
                _fmt_rate(text.get("change_rate")),
            ]
        )
    out.extend(
        _table(
            ["variant", *[label for _, label in PROBE_FAMILIES]],
            probe_rows,
        )
    )
    out.append("")
    out.extend(
        _table(
            ["probe", "what a non-`n/a` cell means"],
            [
                ["Face", "faces the detector still finds, and mean descriptor cosine"],
                ["Detection", "objects lost entirely, and mean count change"],
                ["CLIP prompts", "rows whose best prompt changed, and mean cosine"],
                ["VLM", "rows whose answer changed"],
            ],
        )
    )
    out.append("")

    # -- status ------------------------------------------------------------
    out.append("## Model coverage")
    out.append("")
    cov_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        counts = _status_counts(document_for)
        cov_rows.append(
            [
                titles.get(key, key),
                str(counts.get("ok", 0)),
                str(counts.get("skipped", 0)),
                str(counts.get("failed", 0)),
            ]
        )
    out.extend(_table(["variant", "ok", "skipped", "failed"], cov_rows))
    out.append("")

    reasons: list[tuple[str, str]] = []
    if order:
        reasons = _skip_reasons(variants.get(order[0], {}).get("document", {}))
    if reasons:
        out.append("Reasons, from the first variant (they are identical across arms):")
        out.append("")
        out.extend(
            _table(
                ["model", "reason"],
                [[model, reason] for model, reason in reasons],
            )
        )
        out.append("")

    # -- robustness --------------------------------------------------------
    out.append("## Robustness by transform")
    out.append("")
    out.append(
        "Held-out collapse rate on the protected PNG after each transform, "
        "measured against that transform applied to the *original* so the "
        "transform's own effect is never credited to the protection."
    )
    out.append("")

    first_document = variants.get(order[0], {}).get("document", {}) if order else {}
    transforms = (
        ((first_document.get("transfer") or {}).get("by_transform")) or {}
    )
    robust_rows: list[list[str]] = []
    delivered = [
        titles.get(key, key) + " (PNG)"
        for key in order
    ]
    robust_rows.append(
        [
            "delivered PNG",
            *[
                _fmt_rate(
                    _rate(
                        _headline(variants.get(key, {}).get("document", {})),
                        "collapse_rate",
                    )
                )
                for key in order
            ],
        ]
    )
    for transform_key in sorted(transforms):
        row = [transform_key]
        for key in order:
            summary = (
                ((variants.get(key, {}).get("document", {}).get("transfer")) or {})
                .get("by_transform", {})
                .get(transform_key)
            )
            row.append(_fmt_rate(_rate(summary or {}, "collapse_rate")))
        robust_rows.append(row)
    out.extend(_table(["transform", *delivered], robust_rows))
    out.append("")

    # -- screenshot --------------------------------------------------------
    out.append(f"## Screenshot-like delivery ({SCREENSHOT_TRANSFORM})")
    out.append("")
    out.append(
        "Downscale, mild tone shift and a real JPEG re-encode -- the closest "
        "thing here to a screenshot being re-uploaded. Paired with the visual "
        "quality of the protected image itself, because a protection that only "
        "works by destroying the picture is not a protection."
    )
    out.append("")
    shot_rows: list[list[str]] = []
    for key in order:
        document_for = variants.get(key, {}).get("document", {})
        summary = (
            (document_for.get("transfer") or {}).get("by_transform", {}) or {}
        ).get(SCREENSHOT_TRANSFORM)
        artifact = _artifact_key(document_for, f"/robustness/{SCREENSHOT_TRANSFORM}")
        quality = _visual(document_for, artifact) if artifact else {}
        shot_rows.append(
            [
                titles.get(key, key),
                _fmt_rate(_rate(summary or {}, "collapse_rate")),
                _fmt_rate(_rate(summary or {}, "disruption_rate")),
                _fmt_number(quality.get("psnr_db"), digits=2),
                _fmt_number(quality.get("ssim"), digits=4),
                _fmt_number(quality.get("lpips"), digits=4),
            ]
        )
    out.extend(
        _table(
            [
                "variant",
                "collapse",
                "label flip",
                "PSNR dB",
                "SSIM",
                "LPIPS",
            ],
            shot_rows,
        )
    )
    out.append("")

    # -- caveats -----------------------------------------------------------
    out.append("## How to read this")
    out.append("")
    notes = (_headline(first_document).get("notes")) if first_document else None
    if notes:
        out.append(str(notes))
        out.append("")
    out.extend(
        [
            "* `n/a` means the measurement was not made. It never means zero.",
            "* Face and detection evidence is only drawn from photographs in "
            "the corpus.",
            "* The VLM arm reports `skipped` unless a backend was configured; "
            "it is never simulated.",
            "* Visual quality is measured on the delivered PNG against the "
            "original, so it describes the protection and not the transform.",
        ]
    )
    out.append("")
    return "\n".join(out)


def write_phase2_report(document: dict[str, Any], path: Path) -> Path:
    """Write the markdown report and the raw document beside it.

    Both are written because the report is the argument and the document is
    the evidence: a reader who disagrees with a rounding decision needs the
    numbers, and a reader who wants the summary should not have to parse JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(phase2_report(document), encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
