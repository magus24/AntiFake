"""The Phase 1 benchmark runner.

One entry point, :func:`run_benchmark`, that turns a set of source images into
a single JSON document containing everything needed to judge whether the
current protection transfers.

Three design decisions are worth stating, because they determine whether the
output means anything:

1. **The real engine is used, unmodified.** The benchmark does not
   re-implement or approximate the perturbation; it calls
   :class:`AdversarialProtectionEngine` exactly as the API does. A benchmark
   that measured a different attack would be measuring nothing.

2. **Artifacts are measured as delivered, not as computed.** The perturbed
   array is encoded to PNG *and* JPEG through the product's own
   :func:`encode_image`, then decoded again. The PNG row is the algorithm's
   upper bound; the JPEG row is what a user actually gets. Reporting only the
   in-memory array would overstate robustness by the amount JPEG erodes it.

3. **Every model sees every image in a single sequential pass.** Building all
   image variants first and passing them as one set means each model is loaded
   exactly once, for all variants, then released. That is what keeps peak
   memory to one model at a time.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.evaluation.adapters_embedding import DEFAULT_PROMPTS
from app.evaluation.base import ModelSpec
from app.evaluation.comparison import compare
from app.evaluation.registry import (
    DEFAULT_MODELS,
    ModelRegistry,
    hardware_summary,
    select_device,
)
from app.evaluation.transforms import EvaluationTransform, default_transforms
from app.evaluation.visual_quality import compute_visual_quality, lpips_status
from app.protection.base import RGB
from app.protection.config import ProtectionConfig, TargetModelConfig
from app.protection.engine import AdversarialProtectionEngine
from app.protection.image_io import encode_image, load_image_from_path

logger = logging.getLogger(__name__)

#: Bumped when the JSON shape changes incompatibly.
BENCHMARK_VERSION = "1.0.0"

#: The pipeline the benchmark measures by default, restated here so the report
#: is self-contained even if the product config changes.
DEFAULT_TARGET = TargetModelConfig()

#: How a run obtains the protection engine it measures. The concrete type is
#: required because the benchmark reports the engine's own metrics.
EngineFactory = Callable[
    [ProtectionConfig, TargetModelConfig], AdversarialProtectionEngine
]


@dataclass(slots=True)
class BenchmarkConfig:
    """Everything a benchmark run needs, as explicit, printable data.

    Attributes:
        image_paths: Source images to protect and measure.
        output_dir: Where artifacts and the JSON document are written.
        protection: The exact protection configuration to evaluate.
        target: The exact target pipeline the engine defends against.
        transforms: Robustness transforms applied to the protected PNG.
        models: Model specs to attempt.
        prompts: Text probes for the CLIP adapter.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        include_embeddings: Store raw embedding vectors in the JSON.
        record_attack_details: Also record the engine's Phase 2 metadata --
            variant label, peak device memory, EOT sample count, surrogate list
            and whether the OOM fallback fired -- on each image's attack entry.
            Off by default so a Phase 1 run keeps producing exactly the document
            shape it always has; the Phase 2 runner turns it on, because that is
            where runtime and VRAM are part of the question being asked.
        engine_factory: Builds the thing under test. Defaults to the product
            engine. Exists so the benchmark can be pointed at a *different*
            protection later, and so the schema can be tested without weights;
            the default is the real engine, so a normal run always measures the
            shipping code path.
    """

    image_paths: tuple[Path, ...]
    output_dir: Path
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    target: TargetModelConfig = field(default_factory=TargetModelConfig)
    transforms: tuple[EvaluationTransform, ...] | None = None
    models: tuple[ModelSpec, ...] = DEFAULT_MODELS
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    device: str = "auto"
    include_embeddings: bool = False
    record_attack_details: bool = False
    engine_factory: EngineFactory | None = None

    def build_engine(self) -> AdversarialProtectionEngine:
        """Construct the protection engine this run measures."""
        if self.engine_factory is not None:
            return self.engine_factory(self.protection, self.target)
        return AdversarialProtectionEngine(self.protection, self.target)

    def describe_transforms(self) -> list[EvaluationTransform]:
        """Return the configured transforms, or the declared default suite."""
        if self.transforms is None:
            return default_transforms()
        return list(self.transforms)


@dataclass(slots=True)
class BenchmarkArtifact:
    """One generated image variant and its provenance.

    Attributes:
        key: Stable identifier, e.g. ``"protected_png"``.
        image: Decoded ``uint8`` RGB array as a user would receive it.
        role: ``"original"``, ``"reference"``, ``"protected"`` or
            ``"transformed"``.
        reference_key: Key of the image this one is compared against, or
            ``None`` for originals. Storing it on the artifact rather than
            deriving it from the key string is what keeps the comparison
            correct when transforms change the image's geometry.
        encoded_bytes: Size on disk, for the report.
        sha256: Content hash, so a run can be tied to exact bytes.
        transform: Which transform produced it, if any.
    """

    key: str
    image: RGB
    role: str
    reference_key: str | None = None
    encoded_bytes: int | None = None
    sha256: str | None = None
    transform: dict[str, Any] | None = None


def _sha256(array: RGB) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _decode(encoded: bytes) -> RGB:
    """Round-trip bytes back through the product's own decoder.

    Copies the decoded buffer: PIL hands out read-only arrays, and torch
    tensors wrapping read-only memory have undefined write behaviour.
    """
    with Image.open(io.BytesIO(encoded)) as handle:
        handle.load()
        return np.array(handle.convert("RGB"), dtype=np.uint8)


def _encode_variant(array: RGB, image_format: str) -> bytes:
    """Encode through the product's encoder, with its own default settings."""
    if image_format == "PNG":
        return encode_image(array, "PNG", png_compress_level=6)
    return encode_image(array, "JPEG", jpeg_quality=95)


def _build_artifacts(
    config: BenchmarkConfig,
) -> tuple[list[BenchmarkArtifact], dict[str, Any]]:
    """Protect every source image and produce all measured variants."""
    artifacts: list[BenchmarkArtifact] = []
    attack_meta: dict[str, Any] = {}
    artifact_dir = config.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for path in config.image_paths:
        loaded = load_image_from_path(path)
        source = loaded.array
        name = path.stem

        engine = config.build_engine()
        output = engine.protect_with_meta(source)
        attack_seconds = float(output.meta.attack_ms) / 1000.0
        protected = output.image

        artifacts.append(
            BenchmarkArtifact(
                key=f"{name}/original",
                image=source,
                role="original",
                reference_key=None,
            )
        )
        # The in-memory array: the algorithm's own upper bound, before any
        # encoder has touched it.
        artifacts.append(
            BenchmarkArtifact(
                key=f"{name}/protected_raw",
                image=protected,
                role="protected",
                reference_key=f"{name}/original",
            )
        )

        # What the user actually receives, one row per delivery format.
        for image_format, extension in (("PNG", "png"), ("JPEG", "jpg")):
            encoded = _encode_variant(protected, image_format)
            decoded = _decode(encoded)
            (artifact_dir / f"{name}-protected.{extension}").write_bytes(encoded)
            artifacts.append(
                BenchmarkArtifact(
                    key=f"{name}/protected_{extension}",
                    image=decoded,
                    role="protected",
                    reference_key=f"{name}/original",
                    encoded_bytes=len(encoded),
                    sha256=_sha256(decoded),
                    transform={
                        "format": image_format,
                        "quality": 6 if extension == "png" else 95,
                    },
                )
            )

        # Robustness variants start from the lossless PNG. Measuring a transform
        # on top of JPEG would compound two degradations and blame the wrong
        # one, so the PNG is the only sensible baseline here.
        png = next(a for a in artifacts if a.key == f"{name}/protected_png")

        # Each transform is applied to the *original* as well, and the
        # protected-then-transformed image is compared against that. A corner
        # crop changes what the model sees no matter what the protection did;
        # without its own baseline the row would credit the protection with the
        # crop's effect and blame it for nothing. The same baseline makes the
        # visual metrics comparable, because both sides then share a shape.
        for transform in config.describe_transforms():
            if transform.key == "identity":
                continue
            baseline = transform.run(source)
            variant = transform.run(png.image)
            described = (
                transform.describe() | {"parameters": variant.parameters}
            )
            artifacts.append(
                BenchmarkArtifact(
                    key=f"{name}/baseline/{transform.key}",
                    image=baseline.image,
            role="reference",
            reference_key=None,
            transform=described,
                )
            )
            artifacts.append(
                BenchmarkArtifact(
                    key=f"{name}/robustness/{transform.key}",
                    image=variant.image,
                    role="transformed",
                    reference_key=f"{name}/baseline/{transform.key}",
                    transform=described,
                )
            )

        psnr = float(output.metrics.psnr_db)
        entry: dict[str, Any] = {
            "source_path": str(path),
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
            "source_format": loaded.source_format,
            "attack_seconds": round(attack_seconds, 3),
            "linf_255": round(float(output.metrics.linf_255), 4),
            "psnr_db": round(psnr, 4) if np.isfinite(psnr) else None,
        }
        if config.record_attack_details:
            # Phase 2 metadata, asked for explicitly: which configuration ran,
            # what it cost in device memory, and whether it had to fall back.
            # Always-on would change the Phase 1 document shape for no Phase 1
            # question, so it is the runner's choice rather than the default.
            meta = output.meta
            entry.update(
                {
                    "variant": meta.variant,
                    "peak_memory_mib": round(float(meta.peak_memory_mib), 2),
                    "eot_samples": int(meta.eot_samples),
                    "surrogate_models": list(meta.surrogate_models),
                    "oom_fallback": bool(meta.oom_fallback),
                }
            )
        attack_meta[name] = entry
        logger.info(
            "benchmark.protected image=%s size=%dx%d attack_s=%.2f",
            name,
            source.shape[1],
            source.shape[0],
            attack_seconds,
        )

    return artifacts, attack_meta


def _reference_for(artifact: BenchmarkArtifact) -> str | None:
    """The key an artifact should be compared against.

    A transformed variant is compared against the *same transform applied to the
    original*, not against the protected PNG and not against the untouched
    original. That isolates the protection's contribution: the row then answers
    "does the protection still work after this transform?" rather than "did
    this transform change the picture?" -- and, for a transform that changes
    geometry, it is also the only comparison where the shapes match.
    """
    return artifact.reference_key


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run the full benchmark and return the JSON-ready result document."""
    device, device_reason = select_device(config.device)
    transforms = config.describe_transforms()
    run_started = time.perf_counter()

    artifacts, attack_meta = _build_artifacts(config)
    artifact_by_key = {a.key: a for a in artifacts}

    # One sequential pass: each model is loaded once, sees every variant,
    # then is released.
    registry = ModelRegistry(device, config.models, prompts=config.prompts)
    model_results = registry.evaluate_all({a.key: a.image for a in artifacts})

    comparison: dict[str, dict[str, Any]] = {}
    for key, artifact in artifact_by_key.items():
        reference = _reference_for(artifact)
        if reference is None:
            continue
        per_model: dict[str, Any] = {}
        for model_key, result in model_results.items():
            if result.status != "ok":
                per_model[model_key] = {
                    "status": result.status,
                    "reason": result.reason,
                }
                continue
            original_obs = result.observations.get(reference)
            variant_obs = result.observations.get(key)
            if original_obs is None or variant_obs is None:
                continue
            per_model[model_key] = {
                "status": "ok",
                **compare(original_obs, variant_obs),
            }
        comparison[key] = per_model

    visual: dict[str, Any] = {}
    for key, artifact in artifact_by_key.items():
        reference = _reference_for(artifact)
        if reference is None:
            continue
        baseline = artifact_by_key[reference].image
        try:
            quality = compute_visual_quality(baseline, artifact.image, device)
            visual[key] = quality.to_dict()
        except ValueError as exc:  # pragma: no cover - references are shape-matched
            visual[key] = {"error": str(exc)}

    document: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "seed": config.protection.seed,
            "device": str(device),
            "device_reason": device_reason,
            "hardware": hardware_summary(device),
            "seconds": round(time.perf_counter() - run_started, 2),
            "images": len({a.key.split("/", 1)[0] for a in artifacts}),
            "variants": len(artifacts),
        },
        "environment": _environment(),
        "protection": {
            "engine": AdversarialProtectionEngine.name,
            "target_model": config.target.description,
            "eps_255": round(config.protection.eps_255, 4),
            "alpha_255": round(config.protection.alpha * 255.0, 4),
            "steps": config.protection.steps,
            "seed": config.protection.seed,
            "target_topk": config.protection.target_topk,
            "loss": config.protection.loss,
            "init": config.protection.init,
        },
        "lpips": lpips_status(device),
        "attack": attack_meta,
        "transforms": [t.describe() for t in transforms],
        "models": {
            key: result.to_dict(include_embeddings=config.include_embeddings)
            for key, result in model_results.items()
        },
        "comparison": comparison,
        "visual_quality": visual,
        # The manifest makes the pairing legible: a reader can see that
        # `source/robustness/scale_0.5` was measured against
        # `source/baseline/scale_0.5` and not against the original, without
        # having to infer it from the numbers.
        "artifacts": [_manifest_entry(a) for a in artifacts],
        "summary": _summarise(comparison, model_results),
    }
    return document


def _manifest_entry(artifact: BenchmarkArtifact) -> dict[str, Any]:
    """One row of the artifact manifest, without the pixel data."""
    return {
        "key": artifact.key,
        "role": artifact.role,
        "reference_key": artifact.reference_key,
        "transform": artifact.transform,
        "encoded_bytes": artifact.encoded_bytes,
        "sha256": artifact.sha256,
    }


def _environment() -> dict[str, Any]:
    """Versions that affect results, captured per run."""
    import torchvision

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
    }
    try:
        import open_clip

        info["open_clip"] = open_clip.__version__
    except ImportError:
        info["open_clip"] = None
    try:
        import lpips as _lpips

        info["lpips"] = getattr(_lpips, "__version__", "installed")
    except ImportError:
        info["lpips"] = None
    return info


def _summarise(
    comparison: dict[str, dict[str, Any]],
    model_results: dict[str, Any],
) -> dict[str, Any]:
    """Headline numbers, computed from the same rows the tables show.

    Deliberately not hand-picked: coverage counts every declared model, so a
    model that failed is visible in ``models_ok`` rather than quietly missing.

    Two transfer counts are reported because they measure different things.
    ``png_label_flips`` is the strict verdict (a different class came out on
    top). ``png_confidence_collapses`` counts a >=75% confidence drop whether
    or not the argmax moved -- which is what actually happens when a CW attack
    flattens the distribution, and is the difference between "the attack did
    not work" and "the attack worked and the model now has no opinion".
    """
    declared = len(model_results)
    ok = sum(1 for r in model_results.values() if r.status == "ok")
    skipped = sum(1 for r in model_results.values() if r.status == "skipped")
    failed = sum(1 for r in model_results.values() if r.status == "failed")

    classifiers = 0
    disrupted: list[str] = []
    collapsed: list[str] = []
    for key, per_model in comparison.items():
        if "/protected_png" not in key:
            continue
        for model_key, row in per_model.items():
            if row.get("status") != "ok" or "target_disrupted" not in row:
                continue
            classifiers += 1
            if row["target_disrupted"]:
                disrupted.append(model_key)
            if row["confidence_collapsed"]:
                collapsed.append(model_key)

    return {
        "models_declared": declared,
        "models_ok": ok,
        "models_skipped": skipped,
        "models_failed": failed,
        "classifiers_measured": classifiers,
        "png_label_flips": f"{len(disrupted)}/{classifiers}",
        "png_confidence_collapses": f"{len(collapsed)}/{classifiers}",
        "png_disrupted_models": sorted(set(disrupted)),
        "png_collapsed_models": sorted(set(collapsed)),
    }


def write_report(document: dict[str, Any], path: Path) -> Path:
    """Write the JSON document, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    return path


def markdown_report(document: dict[str, Any]) -> str:
    """Render the same data as a human-readable Markdown report."""
    lines: list[str] = ["# AntiFake Phase 1 Benchmark", ""]
    run = document["run"]
    protection = document["protection"]

    lines += [
        f"- Version: `{document['benchmark_version']}` "
        f"(schema {document['schema_version']})",
        f"- Generated: {document['generated_at']}",
        f"- Device: `{run['device']}` ({run['device_reason']})",
        f"- Images: {run['images']}, variants: {run['variants']}, "
        f"runtime: {run['seconds']}s",
        f"- Protection: eps {protection['eps_255']}/255, "
        f"alpha {protection['alpha_255']}/255, {protection['steps']} steps, "
        f"seed {protection['seed']}",
        f"- Target: {protection['target_model']}",
        f"- LPIPS: {document['lpips']}",
        "",
    ]

    summary = document["summary"]
    lines += [
        "## Summary",
        "",
        f"- Models: {summary['models_ok']} ok, {summary['models_skipped']} skipped, "
        f"{summary['models_failed']} failed (of {summary['models_declared']})",
        f"- Classifiers measured: {summary['classifiers_measured']}",
        f"- PNG: label flipped on {summary['png_label_flips']}",
        f"- PNG: confidence collapsed (>=75% drop) on "
        f"{summary['png_confidence_collapses']}",
        "",
    ]

    lines += [
        "## Models",
        "",
        "| Model | Family | Status | Description |",
        "| --- | --- | --- | --- |",
    ]
    for key, model in document["models"].items():
        status = model["status"]
        if status != "ok":
            status = f"{status}: {model.get('reason', '')[:80]}"
        lines.append(
            f"| `{key}` | {model['family']} | {status} | {model['description']} |"
        )
    lines.append("")

    lines += [
        "## PNG transfer (protected PNG vs original)",
        "",
        "| Image | Model | Top-1 | Label changed | Confidence | Relative drop |"
        " Entropy rise | Collapsed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, per_model in document["comparison"].items():
        if "/protected_png" not in key:
            continue
        for model_key, row in per_model.items():
            if "label_changed" not in row:
                continue
            before = row["original_top1_probability"]
            after = row["protected_top1_probability"]
            lines.append(
                f"| {key.split('/')[0]} | `{model_key}` | {before} -> {after} | "
                f"{row['label_changed']} | {row['confidence_drop']} | "
                f"{row['relative_drop']} | {row['entropy_rise']} | "
                f"{row['confidence_collapsed']} |"
            )
    lines.append("")

    lines += [
        "## Robustness",
        "",
        "Protection effect after each transform, measured against the *same*",
        "transform applied to the original. The numbers isolate the protection's",
        "contribution rather than the transform's own effect.",
        "",
        "| Image | Transform | Model | Relative drop | Entropy rise | Cosine |"
        " Best prompt |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, per_model in document["comparison"].items():
        if "/robustness/" not in key:
            continue
        transform = key.split("/robustness/", 1)[1]
        for model_key, row in per_model.items():
            if "relative_drop" in row:
                lines.append(
                    f"| {key.split('/')[0]} | {transform} | `{model_key}` | "
                    f"{row['relative_drop']} | {row['entropy_rise']} | - | - |"
                )
            elif "cosine_similarity" in row:
                lines.append(
                    f"| {key.split('/')[0]} | {transform} | `{model_key}` | - | - | "
                    f"{row['cosine_similarity']} | "
                    f"{row['protected_best_prompt']} |"
                )
    lines.append("")

    lines += [
        "## Visual quality (vs the matching baseline)",
        "",
        "Delivery formats are compared to the original; robustness rows are",
        "compared to the same transform applied to the original, so the number",
        "is the protection's own cost, not the transform's.",
        "",
        "| Variant | L-inf | PSNR dB | SSIM | LPIPS | Changed pixels |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, quality in document["visual_quality"].items():
        if "error" in quality:
            lines.append(f"| {key} | error | | | | {quality['error'][:60]} |")
            continue
        lines.append(
            f"| {key} | {quality['linf_8bit']} | {quality['psnr_db']} | "
            f"{quality['ssim']} | {quality['lpips']} | "
            f"{quality['pixels_changed_fraction']} |"
        )
    lines.append("")
    return "\n".join(lines)
