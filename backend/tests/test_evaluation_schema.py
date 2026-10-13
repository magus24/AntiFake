"""Tests for the benchmark's JSON contract.

The document is the deliverable, so its shape is a public interface: a future
reader scripts against these keys. Every test here asserts the *key set*, not
just that a value exists, so a key cannot be dropped or renamed by accident.

The full run is exercised with a stub engine and stub models, which keeps the
suite offline and fast. The real engine and real weights are covered by the
``network`` tests in ``test_evaluation_network.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from app.evaluation.base import (
    ClassificationObservation,
    EmbeddingObservation,
    ModelAdapter,
    ModelSpec,
)
from app.evaluation.benchmark import (
    BENCHMARK_VERSION,
    BenchmarkConfig,
    markdown_report,
    run_benchmark,
    write_report,
)
from app.protection.config import ProtectionConfig

from tests.helpers import encode_png, make_detailed

CPU = torch.device("cpu")

TOP_LEVEL_KEYS = {
    "benchmark_version",
    "schema_version",
    "generated_at",
    "run",
    "environment",
    "protection",
    "lpips",
    "attack",
    "transforms",
    "models",
    "comparison",
    "visual_quality",
    "summary",
    "artifacts",
}


# -- offline doubles --------------------------------------------------------


class StubEngine:
    """Adds a fixed, bounded perturbation, mimicking the real engine's contract."""

    name = "stub-engine"
    preserves_dimensions = True

    def __init__(self, delta: int = 8) -> None:
        """Set the perturbation magnitude in 8-bit levels."""
        self.delta = delta

    def protect(self, image):
        return self.protect_with_meta(image).image

    def protect_with_meta(self, image):
        from app.protection.config import ProtectionResultMeta
        from app.protection.engine import ProtectionOutput
        from app.protection.metrics import compute_metrics

        # A checkerboard offset, so JPEG and resize genuinely bite on it.
        pattern = np.indices(image.shape[:2]).sum(axis=0) % 2
        shifted = image.astype(np.int16) + pattern[..., None] * self.delta
        protected = np.clip(shifted, 0, 255).astype(np.uint8)
        metrics = compute_metrics(image, protected)
        meta = ProtectionResultMeta(
            engine=self.name,
            target_model="stub",
            device="cpu",
            seed=0,
            steps=1,
            eps_255=float(self.delta),
            linf_255=metrics.linf_255,
            l2=metrics.l2,
            psnr_db=metrics.psnr_db,
            ssim=metrics.ssim,
            max_abs_pixel_delta=int(metrics.linf_8bit),
            width=image.shape[1],
            height=image.shape[0],
            attack_ms=1.0,
        )
        return ProtectionOutput(image=protected, meta=meta, metrics=metrics)


class StubAdapter(ModelAdapter):
    """Returns a fixed observation, so schema tests need no weights."""

    def __init__(
        self, device, key: str = "stub", family: str = "classification", **kwargs: object
    ) -> None:
        """Bind a fixed observation family to the device."""
        super().__init__(device)
        self.name = key
        self.family = family
        self.description = f"stub:{key}"
        self._loaded = True

    def load(self):
        self._loaded = True

    def release(self):
        self._loaded = False

    def evaluate(self, image):
        if self.family == "embedding":
            return EmbeddingObservation(
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                text_similarities={"a dog": 0.3, "a street": 0.2},
                text_ranking=["a dog", "a street"],
            )
        return ClassificationObservation(
            top1_index=7,
            top1_label="alpha",
            top1_probability=0.5,
            top1_logit=1.0,
            top5=[(7, 0.5), (3, 0.2), (1, 0.1), (2, 0.1), (9, 0.1)],
            entropy=1.2,
        )


class _ShiftingAdapter(StubAdapter):
    """Confidence depends on the image, so comparisons are not all no-ops."""

    def evaluate(self, image):
        bright = float(image.astype(np.float32).mean())
        return ClassificationObservation(
            top1_index=7 if bright > 127 else 8,
            top1_label="alpha" if bright > 127 else "beta",
            top1_probability=min(0.95, max(0.02, bright / 255.0)),
            top1_logit=bright / 100.0,
            top5=[(7, 0.4), (3, 0.2), (1, 0.1), (2, 0.1), (9, 0.1)],
            entropy=1.2,
        )


@pytest.fixture
def source_png(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    path.write_bytes(encode_png(make_detailed(48, 64)))
    return path


@pytest.fixture
def stub_run(source_png, tmp_path, monkeypatch):
    """A complete offline benchmark run: stub engine, stub models, real files."""
    specs = (
        ModelSpec(
            key="stub_classifier",
            family="classification",
            module="torchvision",
            factory="stub",
        ),
        ModelSpec(
            key="stub_embedder",
            family="embedding",
            module="open_clip",
            factory="stub",
        ),
    )
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: (
            _ShiftingAdapter(device, key=spec.key)
            if spec.family == "classification"
            else StubAdapter(device, key=spec.key, family="embedding")
        ),
    )
    config = BenchmarkConfig(
        image_paths=(source_png,),
        output_dir=tmp_path / "out",
        protection=ProtectionConfig(steps=1),
        models=specs,
        device="cpu",
        transforms=(),
        engine_factory=lambda protection, target: StubEngine(),
    )
    return run_benchmark(config)


# -- schema -----------------------------------------------------------------


def test_top_level_schema_is_stable(stub_run):
    assert set(stub_run) == TOP_LEVEL_KEYS


def test_version_and_schema_are_declared(stub_run):
    assert stub_run["benchmark_version"] == BENCHMARK_VERSION
    assert stub_run["schema_version"] == 1


def test_generated_at_is_iso_utc(stub_run):
    from datetime import datetime

    parsed = datetime.fromisoformat(stub_run["generated_at"])
    assert parsed.tzinfo is not None


def test_run_block_records_reproducibility_inputs(stub_run):
    assert set(stub_run["run"]) == {
        "seed",
        "device",
        "device_reason",
        "hardware",
        "seconds",
        "images",
        "variants",
    }
    assert stub_run["run"]["seed"] == 0
    assert stub_run["run"]["device"] == "cpu"
    assert stub_run["run"]["device_reason"]


def test_environment_block_names_library_versions(stub_run):
    assert set(stub_run["environment"]) == {
        "python",
        "platform",
        "torch",
        "torchvision",
        "numpy",
        "open_clip",
        "lpips",
    }


def test_protection_block_restates_the_exact_configuration(stub_run):
    protection = stub_run["protection"]
    assert protection["eps_255"] == pytest.approx(8.0)
    assert protection["steps"] == 1
    assert protection["seed"] == 0
    assert "resnet18" in protection["target_model"]


def test_attack_block_records_per_image_facts(stub_run):
    entry = stub_run["attack"]["source"]
    assert set(entry) == {
        "source_path",
        "width",
        "height",
        "source_format",
        "attack_seconds",
        "linf_255",
        "psnr_db",
    }
    # The fixture image is make_detailed(width=48, height=64).
    assert entry["width"] == 48
    assert entry["height"] == 64
    assert entry["source_format"] == "PNG"


def test_models_block_reports_observations_per_variant(stub_run):
    model = stub_run["models"]["stub_classifier"]
    assert model["status"] == "ok"
    assert model["family"] == "classification"
    assert "source/protected_png" in model["observations"]


def test_skipped_model_still_appears_with_a_reason(source_png, tmp_path, monkeypatch):
    """Coverage must be reported honestly, not quietly narrowed."""
    from app.evaluation.base import ModelUnavailableError

    specs = (ModelSpec(key="gone", family="classification", module="x", factory="x"),)
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: (_ for _ in ()).throw(
            ModelUnavailableError(spec.key, "weights not cached")
        ),
    )
    document = run_benchmark(
        BenchmarkConfig(
            image_paths=(source_png,),
            output_dir=tmp_path / "out",
            protection=ProtectionConfig(steps=1),
            models=specs,
            device="cpu",
            transforms=(),
            engine_factory=lambda p, t: StubEngine(),
        )
    )
    assert document["models"]["gone"]["status"] == "skipped"
    assert document["summary"]["models_skipped"] == 1
    assert document["summary"]["models_ok"] == 0


# -- semantics --------------------------------------------------------------


def test_png_round_trip_is_lossless(stub_run):
    """The PNG row must equal the in-memory row, or the ceiling is wrong."""
    raw = stub_run["visual_quality"]["source/protected_raw"]
    png = stub_run["visual_quality"]["source/protected_png"]
    assert raw["linf_8bit"] == png["linf_8bit"]
    assert raw["psnr_db"] == pytest.approx(png["psnr_db"], abs=1e-3)
    assert raw["ssim"] == pytest.approx(png["ssim"], abs=1e-3)


def test_jpeg_erodes_more_than_png(stub_run):
    """If JPEG did not degrade the perturbation, the delivery test is useless."""
    png = stub_run["visual_quality"]["source/protected_png"]
    jpeg = stub_run["visual_quality"]["source/protected_jpg"]
    assert jpeg["linf_8bit"] >= png["linf_8bit"]
    assert (jpeg["psnr_db"] or 0) <= (png["psnr_db"] or 0) + 1e-6


def test_protection_actually_changed_the_image(stub_run):
    assert stub_run["visual_quality"]["source/protected_png"]["linf_8bit"] > 0
    assert stub_run["attack"]["source"]["linf_255"] > 0


def test_variants_cover_raw_png_jpeg(stub_run):
    keys = set(stub_run["visual_quality"])
    # The original is the reference, not a variant, so it has no row.
    assert "source/original" not in keys
    assert {
        "source/protected_raw",
        "source/protected_png",
        "source/protected_jpg",
    } <= keys


def test_baseline_rows_are_paired_with_robustness(source_png, tmp_path, monkeypatch):
    """Every transform produces a paired baseline and variant.

    The baseline is the same transform applied to the original, so a robustness
    row measures the protection's contribution rather than the transform's.
    """
    from app.evaluation.transforms import CornerCrop, HalfScale, JpegQuality

    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: StubAdapter(device, key=spec.key),
    )
    document = run_benchmark(
        BenchmarkConfig(
            image_paths=(source_png,),
            output_dir=tmp_path / "out",
            protection=ProtectionConfig(steps=1),
            models=(
                ModelSpec(
                    key="stub", family="classification", module="x", factory="x"
                ),
            ),
            device="cpu",
            transforms=(HalfScale(), CornerCrop(0.8), JpegQuality(75)),
            engine_factory=lambda p, t: StubEngine(),
        )
    )
    manifest = {entry["key"]: entry for entry in document["artifacts"]}
    for key in ("scale_0.5", "corner_crop", "jpeg_q75"):
        baseline = f"source/baseline/{key}"
        variant = f"source/robustness/{key}"
        assert manifest[baseline]["role"] == "reference"
        # A baseline is a reference, not a variant: it is not measured itself.
        assert manifest[baseline]["reference_key"] is None
        assert baseline not in document["visual_quality"]

        assert manifest[variant]["reference_key"] == baseline
        assert variant in document["visual_quality"]
        assert document["visual_quality"][variant]["linf_8bit"] > 0

    # The pairing keeps the reported cost attributable: the baseline absorbs the
    # transform, so what remains is the protection's own contribution.
    assert (
        document["visual_quality"]["source/robustness/scale_0.5"]["linf_8bit"] > 0
    )


def test_visual_quality_has_no_shape_mismatch_rows(source_png, tmp_path, monkeypatch):
    """A geometry-changing transform used to leave an unreadable error row.

    Applying the same transform to the original as well makes both sides share
    a shape, so every robustness row yields real numbers instead.
    """
    from app.evaluation.transforms import ResizeShortSide

    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: StubAdapter(device, key=spec.key),
    )
    document = run_benchmark(
        BenchmarkConfig(
            image_paths=(source_png,),
            output_dir=tmp_path / "out",
            protection=ProtectionConfig(steps=1),
            models=(),
            device="cpu",
            transforms=(ResizeShortSide(16),),
            engine_factory=lambda p, t: StubEngine(),
        )
    )
    assert "source/robustness/resize_short_side" in document["visual_quality"]
    for key, quality in document["visual_quality"].items():
        assert "error" not in quality, f"{key}: {quality}"


def test_identity_transform_is_not_duplicated_as_a_variant(stub_run):
    """The PNG row *is* the identity row; a separate one would be a duplicate."""
    assert not any("identity" in key for key in stub_run["comparison"])


def test_comparison_references_the_original_not_the_protected_png(stub_run):
    """Rows answer "is this still read correctly?".

    Not "did the transform eat the perturbation?".
    """
    row = stub_run["comparison"]["source/protected_png"]["stub_classifier"]
    assert row["original_top1_probability"] == pytest.approx(
        stub_run["comparison"]["source/protected_jpg"]["stub_classifier"][
            "original_top1_probability"
        ]
    )


def test_summary_counts_every_declared_model(stub_run):
    summary = stub_run["summary"]
    assert summary["models_declared"] == 2
    assert summary["models_ok"] == 2
    assert summary["models_failed"] == 0
    assert summary["classifiers_measured"] == 1
    assert "/" in summary["png_label_flips"]
    assert "/" in summary["png_confidence_collapses"]


def test_two_runs_with_the_same_seed_agree(stub_run, source_png, tmp_path, monkeypatch):
    """Determinism is what makes two benchmark runs comparable at all."""
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: (
            _ShiftingAdapter(device, key=spec.key)
            if spec.family == "classification"
            else StubAdapter(device, key=spec.key, family="embedding")
        ),
    )
    second = run_benchmark(
        BenchmarkConfig(
            image_paths=(source_png,),
            output_dir=tmp_path / "out2",
            protection=ProtectionConfig(steps=1),
            models=stub_run and BenchmarkConfig(
                image_paths=(source_png,),
                output_dir=tmp_path / "out3",
                models=(
                    ModelSpec(
                        key="stub_classifier",
                        family="classification",
                        module="torchvision",
                        factory="stub",
                    ),
                    ModelSpec(
                        key="stub_embedder",
                        family="embedding",
                        module="open_clip",
                        factory="stub",
                    ),
                ),
            ).models,
            device="cpu",
            transforms=(),
            engine_factory=lambda p, t: StubEngine(),
        )
    )
    for key in ("visual_quality", "comparison", "protection", "transforms"):
        assert json.dumps(second[key], sort_keys=True) == json.dumps(
            stub_run[key], sort_keys=True
        )


# -- serialisation ----------------------------------------------------------


def test_document_round_trips_through_json(stub_run):
    text = json.dumps(stub_run)
    assert json.loads(text) == stub_run


def test_no_infinity_leaks_into_json(stub_run):
    """`inf` is not valid JSON; it must have been mapped to `null`."""
    assert "Infinity" not in json.dumps(stub_run)


def test_write_report_creates_parents(stub_run, tmp_path):
    path = write_report(stub_run, tmp_path / "deep" / "nested" / "benchmark.json")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == stub_run


def test_markdown_report_mentions_the_headline_numbers(stub_run):
    report = markdown_report(stub_run)
    assert "AntiFake Phase 1 Benchmark" in report
    assert "## Summary" in report
    assert "stub_classifier" in report


def test_markdown_report_handles_infinite_psnr(source_png, tmp_path, monkeypatch):
    """A perfect reproduction must not render as `Infinity` in the table."""
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: StubAdapter(device, key=spec.key),
    )

    class _NoOpEngine(StubEngine):
        def protect_with_meta(self, image):
            from app.protection.config import ProtectionResultMeta
            from app.protection.engine import ProtectionOutput
            from app.protection.metrics import compute_metrics

            metrics = compute_metrics(image, image)
            meta = ProtectionResultMeta(
                engine="no-op",
                target_model="stub",
                device="cpu",
                seed=0,
                steps=0,
                eps_255=0.0,
                linf_255=0.0,
                l2=0.0,
                psnr_db=metrics.psnr_db,
                ssim=metrics.ssim,
                max_abs_pixel_delta=0,
                width=image.shape[1],
                height=image.shape[0],
                attack_ms=0.0,
            )
            return ProtectionOutput(image=image, meta=meta, metrics=metrics)

    document = run_benchmark(
        BenchmarkConfig(
            image_paths=(source_png,),
            output_dir=tmp_path / "out",
            protection=ProtectionConfig(steps=1),
            models=(
                ModelSpec(
                    key="stub", family="classification", module="x", factory="x"
                ),
            ),
            device="cpu",
            transforms=(),
            engine_factory=lambda p, t: _NoOpEngine(),
        )
    )
    assert "Infinity" not in markdown_report(document)


def test_artifacts_are_written_to_disk(stub_run, tmp_path):
    out = tmp_path / "out" / "artifacts"
    assert (out / "source-protected.png").is_file()
    assert (out / "source-protected.jpg").is_file()


def test_png_artifact_is_really_lossless_png(stub_run, tmp_path):
    from app.protection.image_io import load_image_from_path

    original = load_image_from_path(tmp_path / "source.png").array
    written = load_image_from_path(
        tmp_path / "out" / "artifacts" / "source-protected.png"
    ).array
    assert written.shape == original.shape
    assert not np.array_equal(original, written)
