"""Tests for the four-way Phase 2 comparison.

The runner's job is plumbing: four arms, everything else fixed, and one
transfer block per arm. What these tests check is that the plumbing is right --
that the arms differ *only* in the switches, that each arm records its own
attack metadata, and that the report renders every section without pretending
to have measured something it did not.

The numbers themselves come from a real run with real weights, which is what
the ``network`` marker is for. A stub that reproduced them here would only be
checking itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from app.evaluation.base import (
    ClassificationObservation,
    Detection,
    DetectionObservation,
    EmbeddingObservation,
    FaceObservation,
    ModelAdapter,
    ModelSpec,
    ModelUnavailableError,
    TextObservation,
)
from app.evaluation.phases import (
    SURROGATE_KEYS,
    VARIANT_KEYS,
    PhaseComparisonConfig,
    phase_variants,
    run_phase_comparison,
)
from app.evaluation.report_phase2 import phase2_report, write_phase2_report
from app.evaluation.transforms import ScreenshotLike
from app.protection.config import ProtectionConfig

from tests.helpers import encode_png, make_detailed

CPU = torch.device("cpu")


# -- offline doubles --------------------------------------------------------


class _StubEngine:
    """A stand-in whose cost and label track the configuration it was given.

    The label matters: it is how a test proves the four arms really are four
    different attacks rather than the same run written down four times.
    """

    name = "stub-engine"
    preserves_dimensions = True

    def __init__(self, delta: int, label: str) -> None:
        self.delta = delta
        self.label = label

    def protect(self, image):
        return self.protect_with_meta(image).image

    def protect_with_meta(self, image):
        from app.protection.config import ProtectionResultMeta
        from app.protection.engine import ProtectionOutput
        from app.protection.metrics import compute_metrics

        pattern = np.indices(image.shape[:2]).sum(axis=0) % 2
        shifted = image.astype(np.int16) - pattern[..., None] * self.delta
        protected = np.clip(shifted, 0, 255).astype(np.uint8)
        metrics = compute_metrics(image, protected)
        meta = ProtectionResultMeta(
            engine=self.name,
            target_model="stub",
            device="cpu",
            seed=0,
            steps=1,
            eps_255=8.0,
            linf_255=metrics.linf_255,
            l2=metrics.l2,
            psnr_db=metrics.psnr_db,
            ssim=metrics.ssim,
            max_abs_pixel_delta=int(metrics.linf_8bit),
            width=image.shape[1],
            height=image.shape[0],
            attack_ms=float(self.delta),
            variant=self.label,
            peak_memory_mib=float(self.delta * 16),
            eot_samples=0,
            surrogate_models=(),
            oom_fallback=False,
        )
        return ProtectionOutput(image=protected, meta=meta, metrics=metrics)


def _engine_factory(protection, target):
    """Give each arm a distinct, recognisable stand-in."""
    if protection.uses_eot and protection.uses_ensemble:
        delta, label = 2, "stub-both"
    elif protection.uses_ensemble:
        delta, label = 4, "stub-ensemble"
    elif protection.uses_eot:
        delta, label = 6, "stub-eot"
    else:
        delta, label = 8, "stub-phase1"
    return _StubEngine(delta, label)


class _FamilyAdapter(ModelAdapter):
    """One observation shape per family, all derived from image brightness.

    Brightness is what the stub engine moves, so every family reacts to the
    arm that ran and no row in the report is a constant.
    """

    def __init__(self, device, key: str, family: str) -> None:
        super().__init__(device)
        self.name = key
        self.family = family
        self.description = f"stub:{key}"
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    def release(self) -> None:
        self._loaded = False

    @staticmethod
    def _level(image) -> float:
        return float(np.asarray(image).astype(np.float32).mean())

    def evaluate(self, image):
        level = self._level(image)
        if self.family == "embedding":
            vector = np.array([level, 255.0 - level], dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            ranking = ["a dog", "a street"] if level > 127 else ["a street", "a dog"]
            return EmbeddingObservation(
                embedding=vector,
                text_similarities={"a dog": level / 255.0, "a street": 1 - level / 255.0},
                text_ranking=ranking,
            )
        if self.family == "face":
            vector = np.array([level, 255.0 - level], dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            return FaceObservation(
                detected=True,
                confidence=min(0.99, level / 200.0),
                box=(0, 0, 4, 4),
                embedding=vector,
            )
        if self.family == "detection":
            return DetectionObservation(
                detections=[
                    Detection(
                        label="person",
                        confidence=min(0.99, level / 255.0),
                        box=(0, 0, 4, 4),
                    )
                ]
            )
        if self.family == "text":
            return TextObservation(
                answers={"q": "bright" if level > 127 else "dark"},
                model="stub-vlm",
            )
        probability = min(0.95, max(0.01, level / 255.0))
        return ClassificationObservation(
            top1_index=7,
            top1_label="alpha" if level > 127 else "beta",
            top1_probability=probability,
            top1_logit=probability * 4.0,
            top5=[(7, probability)],
            entropy=1.0,
        )


def _stub_models() -> tuple[ModelSpec, ...]:
    from app.evaluation.phases import ALL_MODELS

    return ALL_MODELS


def _build_adapter(spec, device, prompts=None):
    """Every family gets a stub, except the VLM -- which stays unconfigured.

    Leaving the vision-language branch alone is deliberate. It is the one probe
    that has no default local model, so a run without a backend has to come back
    ``skipped``, and that is the only way to assert the report does not turn the
    gap into a zero.
    """
    if spec.family == "text":
        raise ModelUnavailableError(spec.key, "no VLM backend configured (stub)")
    return _FamilyAdapter(device, spec.key, spec.family)


@pytest.fixture
def phase_run(source_png: Path, tmp_path: Path, monkeypatch) -> dict:
    """A complete offline four-way run over one stub source image."""
    monkeypatch.setattr("app.evaluation.registry.build_adapter", _build_adapter)
    config = PhaseComparisonConfig(
        output_dir=tmp_path / "phases",
        image_paths=(source_png,),
        transforms=(ScreenshotLike(),),
        models=_stub_models(),
        device="cpu",
        engine_factory=_engine_factory,
        eot_samples=3,
    )
    return run_phase_comparison(config)


@pytest.fixture
def source_png(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    path.write_bytes(encode_png(make_detailed(48, 64)))
    return path


# -- the four arms ----------------------------------------------------------


def test_phase1_is_the_untouched_base():
    """The control must be the real Phase 1 configuration, not a copy of it."""
    variants = phase_variants(ProtectionConfig(steps=7))
    assert tuple(variants) == VARIANT_KEYS
    assert variants["phase1"].config == ProtectionConfig(steps=7)
    assert variants["phase1"].config.uses_eot is False
    assert variants["phase1"].config.uses_ensemble is False


def test_each_arm_switches_exactly_thing_it_claims():
    variants = phase_variants()

    assert variants["eot"].config.eot_samples > 0
    assert variants["eot"].config.ensemble is False
    assert variants["eot"].config.surrogates == ()

    assert variants["ensemble"].config.eot_samples == 0
    assert variants["ensemble"].config.ensemble is True
    assert variants["ensemble"].config.surrogates == SURROGATE_KEYS

    assert variants["eot_ensemble"].config.eot_samples > 0
    assert variants["eot_ensemble"].config.ensemble is True
    assert variants["eot_ensemble"].config.surrogates == SURROGATE_KEYS


def test_every_arm_keeps_the_budget_and_the_optimiser_fixed():
    """eps, alpha, steps and seed are the experiment's controls, not knobs."""
    variants = phase_variants(ProtectionConfig(steps=11, seed=5))
    for variant in variants.values():
        assert variant.config.eps == pytest.approx(8.0 / 255.0)
        assert variant.config.alpha == pytest.approx(2.0 / 255.0)
        assert variant.config.steps == 11
        assert variant.config.seed == 5
        assert variant.config.target_topk == 3
        assert variant.config.loss == "cw_topk"


def test_eot_samples_is_validated_when_building_the_arms():
    from app.protection.config import ProtectionConfigError

    with pytest.raises(ProtectionConfigError):
        phase_variants(eot_samples=999)


# -- running ----------------------------------------------------------------


def test_the_run_produces_one_document_per_arm(phase_run):
    assert tuple(phase_run["variants"]) == VARIANT_KEYS
    assert phase_run["run"]["variants"] == list(VARIANT_KEYS)
    for key in VARIANT_KEYS:
        entry = phase_run["variants"][key]
        assert entry["variant"]["key"] == key
        assert entry["document"]["benchmark_version"]


def test_each_arm_records_its_own_attack_metadata(phase_run):
    """Runtime and peak memory are the point of the cost table."""
    labels = set()
    for key in VARIANT_KEYS:
        document = phase_run["variants"][key]["document"]
        attack = document["attack"]["source"]
        assert "attack_seconds" in attack
        assert "peak_memory_mib" in attack
        assert attack["variant"]
        labels.add(attack["variant"])
    assert len(labels) == 4, "each arm must be distinguishable from its label"


def test_arms_are_not_interchangeable(phase_run):
    """Different switches must produce different delivered images."""
    digests = {
        key: phase_run["variants"][key]["document"]["attack"]["source"]["psnr_db"]
        for key in VARIANT_KEYS
    }
    assert len(set(digests.values())) > 1


def test_every_arm_carries_a_transfer_block(phase_run):
    for key in VARIANT_KEYS:
        transfer = phase_run["variants"][key]["document"]["transfer"]
        headline = transfer["headline"]
        assert headline["held_out"]["expected"] == 2, "vit_b_16 + swin_t"
        assert headline["held_out_status"]["vit_b_16"] == "ok"
        assert headline["held_out_complete"] is True
        # Probes are reported per family, never folded into the headline.
        assert headline["face"]["evaluated"] == 1
        assert headline["detection"]["evaluated"] == 1
        assert headline["embedding"]["evaluated"] == 1
        assert headline["text"]["evaluated"] == 0, "no VLM backend in this run"
        assert headline["text"]["change_rate"] is None
        assert "screenshot_like" in transfer["by_transform"]


def test_probes_never_reach_the_classification_headline(phase_run):
    """A face row in the held-out set must not change the classifier count."""
    headline = phase_run["variants"]["phase1"]["document"]["transfer"]["headline"]
    assert headline["held_out"]["expected"] == 2
    assert headline["held_out"]["models"] == ["swin_t", "vit_b_16"]
    assert set(headline["probe_status"]) == {"mtcnn_facenet_v1", "yolov8n", "vlm"}


def test_vlm_is_reported_rather_than_simulated(phase_run):
    """Without a backend the branch is skipped, and the document says so."""
    document = phase_run["variants"]["phase1"]["document"]
    vlm = document["models"]["vlm"]
    assert vlm["status"] == "skipped"
    assert "no VLM backend" in vlm["reason"]
    headline = document["transfer"]["headline"]
    assert headline["text"]["evaluated"] == 0
    assert headline["text"]["change_rate"] is None
    assert headline["evaluator_status"] or headline["probe_status"]


def test_an_unknown_arm_is_rejected_up_front(tmp_path: Path, source_png: Path):
    config = PhaseComparisonConfig(
        output_dir=tmp_path / "out",
        image_paths=(source_png,),
        variants=("phase1", "invented"),
        device="cpu",
        engine_factory=_engine_factory,
    )
    with pytest.raises(ValueError, match="invented"):
        run_phase_comparison(config)


def test_a_missing_source_image_fails_before_any_work(tmp_path: Path):
    config = PhaseComparisonConfig(
        output_dir=tmp_path / "out",
        image_paths=(tmp_path / "nowhere.png",),
        variants=("phase1",),
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match=r"nowhere\.png"):
        run_phase_comparison(config)


def test_a_subset_of_arms_is_allowed(source_png: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.evaluation.registry.build_adapter", _build_adapter)
    config = PhaseComparisonConfig(
        output_dir=tmp_path / "subset",
        image_paths=(source_png,),
        variants=("phase1", "ensemble"),
        transforms=(),
        device="cpu",
        engine_factory=_engine_factory,
    )
    document = run_phase_comparison(config)
    assert tuple(document["variants"]) == ("phase1", "ensemble")
    assert document["run"]["variants"] == ["phase1", "ensemble"]


# -- report -----------------------------------------------------------------


def test_report_renders_every_section(phase_run):
    markdown = phase2_report(phase_run)

    for heading in (
        "# AntiFake Phase 2: four-way protection comparison",
        "## Cost: runtime and peak device memory",
        "## Held-out transfer (the headline)",
        "## Surrogate and target diagnostics",
        "## Probes: face, detection, CLIP, VLM",
        "## Model coverage",
        "## Robustness by transform",
        "## Screenshot-like delivery (screenshot_like)",
        "## How to read this",
    ):
        assert heading in markdown, f"missing section: {heading}"

    for title in ("Phase 1", "EOT", "Ensemble", "EOT + Ensemble"):
        assert title in markdown


def test_report_shows_n_a_rather_than_zero_for_unmeasured(phase_run):
    """A skipped VLM must not read as a VLM that found nothing wrong."""
    markdown = phase2_report(phase_run)
    assert "n/a" in markdown
    assert "`n/a` means the measurement was not made. It never means zero." in markdown


def test_report_states_the_held_out_contract(phase_run):
    markdown = phase2_report(phase_run)
    assert "surrogate" in markdown.lower()
    assert "not evidence that the perturbation transferred" in markdown
    assert "vit_b_16" in markdown and "swin_t" in markdown


def test_writing_the_report_writes_the_evidence_alongside(phase_run, tmp_path: Path):
    path = tmp_path / "out" / "report.md"
    written = write_phase2_report(phase_run, path)
    assert written.read_text(encoding="utf-8").startswith("# AntiFake Phase 2")
    assert path.with_suffix(".json").is_file()


# -- corpus -----------------------------------------------------------------


def test_corpus_declares_provenance_and_face_eligibility():
    from app.evaluation.corpus import (
        PHOTOGRAPH,
        SYNTHETIC,
        face_eligible,
        photograph,
        synthetic,
    )

    real = photograph(Path("a.jpg"))
    made = synthetic(Path("b.png"))
    assert real.kind == PHOTOGRAPH and real.face_eligible is True
    assert made.kind == SYNTHETIC and made.face_eligible is False
    assert face_eligible([real, made]) == (real,)


def test_corpus_paths_are_checked_before_the_run():
    from app.evaluation.corpus import assert_paths_exist, photograph

    missing = photograph(Path("definitely-not-here.png"))
    with pytest.raises(FileNotFoundError, match="definitely-not-here"):
        assert_paths_exist([missing])


def test_default_corpus_is_overridable_by_environment(tmp_path, monkeypatch):
    from app.evaluation.corpus import CORPUS_ENV_VAR, default_corpus

    (tmp_path / "photo_one.jpg").write_bytes(b"not really an image")
    (tmp_path / "notes.txt").write_bytes(b"ignored")
    monkeypatch.setenv(CORPUS_ENV_VAR, str(tmp_path))

    corpus = default_corpus()
    assert [image.key for image in corpus] == ["photo_one"]
    assert all(image.face_eligible for image in corpus)
