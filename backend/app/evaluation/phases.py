"""The four-way Phase 2 comparison: Phase 1, EOT, Ensemble, EOT+Ensemble.

One run, four attacks, everything else held fixed. That is the whole point of
the module: the only differences between the four rows are the two switches
under test, so any difference in the outcome can be attributed to them rather
than to a different seed, a different image set or a different model list.

What is held fixed, explicitly:

* ``eps``, ``alpha``, ``steps``, ``seed``, ``target_topk`` and the target
  pipeline -- the perturbation budget and the optimiser never move;
* the source images (one corpus, resolved once);
* the transform suite (:func:`phase2_transforms`), so robustness rows are
  comparable across variants;
* the model set (classifiers, CLIP and the face/detection/VLM probes) and the
  CLIP prompt set, so a difference in a probe is not a difference in the probe.

Each variant produces a complete benchmark document. This module adds one
block to it -- ``transfer`` -- which reports the held-out rates for the
delivered PNG and for every robustness transform separately, because a
protection that survives its own delivery and one that survives a screenshot
are different claims and should not share a number.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.adapters_embedding import EXTENDED_PROMPTS
from app.evaluation.base import ModelSpec
from app.evaluation.benchmark import (
    BenchmarkConfig,
    EngineFactory,
    run_benchmark,
)
from app.evaluation.comparison import transfer_summary
from app.evaluation.corpus import (
    CorpusImage,
    assert_paths_exist,
    corpus_from_paths,
    default_corpus,
)
from app.evaluation.registry import (
    DEFAULT_MODELS,
    PROBE_MODELS,
    hardware_summary,
    select_device,
)
from app.evaluation.transforms import EvaluationTransform, phase2_transforms
from app.protection.config import ProtectionConfig, TargetModelConfig
from app.protection.ensemble import DEFAULT_SURROGATES

logger = logging.getLogger(__name__)

#: Bumped when this document's shape changes incompatibly.
PHASE_COMPARISON_VERSION = "1.0.0"

#: EOT renderings per optimisation step. Small enough to stay affordable,
#: large enough that the sampled distribution is not a coin flip.
DEFAULT_EOT_SAMPLES = 4

#: The four variants, in report order.
VARIANT_KEYS: tuple[str, ...] = ("phase1", "eot", "ensemble", "eot_ensemble")

#: Surrogate keys the ensemble variants optimise against, in declaration order.
SURROGATE_KEYS: tuple[str, ...] = tuple(spec.key for spec in DEFAULT_SURROGATES)

#: Every model a variant is measured on: the Phase 1 set plus the probes.
ALL_MODELS: tuple[ModelSpec, ...] = (*DEFAULT_MODELS, *PROBE_MODELS)


@dataclass(frozen=True, slots=True)
class PhaseVariant:
    """One arm of the experiment.

    Attributes:
        key: Stable identifier, matching :data:`VARIANT_KEYS`.
        title: Human label for the report table.
        summary: One line saying what is switched on.
        config: The exact protection configuration this arm runs.
    """

    key: str
    title: str
    summary: str
    config: ProtectionConfig

    def describe(self) -> dict[str, Any]:
        """JSON-ready description of the arm and its switches."""
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "eot_samples": self.config.eot_samples,
            "ensemble": self.config.ensemble,
            "surrogates": list(self.config.surrogates),
            "eps_255": round(self.config.eps * 255.0, 4),
            "alpha_255": round(self.config.alpha * 255.0, 4),
            "steps": self.config.steps,
            "seed": self.config.seed,
        }


def phase_variants(
    base: ProtectionConfig | None = None, *, eot_samples: int = DEFAULT_EOT_SAMPLES
) -> dict[str, PhaseVariant]:
    """Build the four arms from one base configuration.

    ``phase1`` is the untouched base, so the control is the real Phase 1 attack
    rather than a reconstruction of it. The other three each differ from it in
    exactly one way, except ``eot_ensemble`` which differs in both by design --
    it is the combination, not a third independent switch.

    Args:
        base: Configuration to derive from. Defaults to
            :class:`ProtectionConfig`, i.e. the shipping Phase 1 defaults.
        eot_samples: Renderings per step for the arms that use EOT.

    Returns:
        ``key -> PhaseVariant`` for every key in :data:`VARIANT_KEYS`.

    Raises:
        ProtectionConfigError: If ``eot_samples`` is out of range for the base.
    """
    base_config = base if base is not None else ProtectionConfig()
    # Constructed eagerly so a bad value fails here, with the whole experiment
    # described in the traceback, rather than on the third variant.
    with_eot = replace(base_config, eot_samples=eot_samples)
    with_ensemble = replace(
        base_config, ensemble=True, surrogates=SURROGATE_KEYS
    )
    with_both = replace(with_ensemble, eot_samples=eot_samples)

    return {
        "phase1": PhaseVariant(
            key="phase1",
            title="Phase 1",
            summary="single-model PGD against the target, no EOT, no ensemble",
            config=base_config,
        ),
        "eot": PhaseVariant(
            key="eot",
            title="EOT",
            summary=f"single-model PGD over {eot_samples} sampled renderings per step",
            config=with_eot,
        ),
        "ensemble": PhaseVariant(
            key="ensemble",
            title="Ensemble",
            summary=(
                "PGD against "
                f"{len(SURROGATE_KEYS)} architectures: {', '.join(SURROGATE_KEYS)}"
            ),
            config=with_ensemble,
        ),
        "eot_ensemble": PhaseVariant(
            key="eot_ensemble",
            title="EOT + Ensemble",
            summary=(
                f"both switches on: {eot_samples} renderings per step against "
                f"{len(SURROGATE_KEYS)} architectures"
            ),
            config=with_both,
        ),
    }


@dataclass(slots=True)
class PhaseComparisonConfig:
    """Everything one four-way run needs.

    Attributes:
        image_paths: Source images. Empty means "resolve the default corpus".
        output_dir: Where per-variant benchmark documents are written.
        variants: Which arms to run, in report order. A subset is allowed so a
            quick two-way check does not pay for all four.
        base: Configuration the arms derive from.
        target: Target pipeline every arm defends against.
        eot_samples: Renderings per step for the EOT arms.
        models: Model specs to measure. Defaults to every classifier, CLIP and
            the face/detection/VLM probes.
        transforms: Robustness transforms per arm. ``None`` means the Phase 2
            suite, which is the point of the comparison; an explicit empty
            tuple measures delivery only.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        engine_factory: Builds the protection engine under test. Defaults to
            the product engine. Exists so the four arms can be exercised
            offline with a stand-in, exactly as the Phase 1 benchmark does.
    """

    output_dir: Path
    image_paths: tuple[Path, ...] = ()
    variants: tuple[str, ...] = VARIANT_KEYS
    base: ProtectionConfig = field(default_factory=ProtectionConfig)
    target: TargetModelConfig = field(default_factory=TargetModelConfig)
    eot_samples: int = DEFAULT_EOT_SAMPLES
    models: tuple[ModelSpec, ...] = ALL_MODELS
    transforms: tuple[EvaluationTransform, ...] | None = None
    device: str = "auto"
    engine_factory: EngineFactory | None = None

    def resolve_corpus(self) -> tuple[CorpusImage, ...]:
        """The images to run on: explicit paths, else the default corpus."""
        if self.image_paths:
            return corpus_from_paths(self.image_paths)
        return default_corpus()

    def resolve_transforms(self) -> tuple[EvaluationTransform, ...]:
        """The robustness suite every arm is measured with."""
        if self.transforms is None:
            return tuple(phase2_transforms())
        return self.transforms


def _transfer_block(
    document: dict[str, Any], model_specs: tuple[ModelSpec, ...]
) -> dict[str, Any]:
    """Held-out rates for the delivered PNG and for every robustness row.

    Roles and families come from the declared specs rather than inferred from
    the results, so a model that failed still occupies its bucket and the
    "expected" count stays honest.
    """
    roles = {spec.key: spec.role for spec in model_specs}
    families = {spec.key: spec.family for spec in model_specs}
    statuses = {
        key: result.get("status", "unknown")
        for key, result in document.get("models", {}).items()
    }

    comparisons: dict[str, dict[str, Any]] = document.get("comparison", {})
    headline_key = next(
        (key for key in comparisons if key.endswith("/protected_png")), None
    )

    def _summarise(row: dict[str, Any]) -> dict[str, Any]:
        return transfer_summary(
            row, roles, families=families, evaluated=statuses
        )

    by_transform: dict[str, dict[str, Any]] = {}
    for key, row in comparisons.items():
        marker = "/robustness/"
        if marker in key:
            by_transform[key.split(marker, 1)[1]] = _summarise(row)

    return {
        "headline": _summarise(comparisons[headline_key])
        if headline_key
        else None,
        "headline_artifact": headline_key,
        "by_transform": by_transform,
    }


def run_phase_comparison(config: PhaseComparisonConfig) -> dict[str, Any]:
    """Run the selected arms and return one comparison document.

    Every arm is a complete benchmark run, so each one's document still stands
    on its own -- the aggregation below is additive, not a replacement.

    Raises:
        FileNotFoundError: If a declared source image does not exist.
        ValueError: If ``variants`` names something that is not a known arm.
    """
    corpus = config.resolve_corpus()
    assert_paths_exist(corpus)
    unknown = [key for key in config.variants if key not in VARIANT_KEYS]
    if unknown:
        raise ValueError(
            f"unknown variant(s) {unknown}; known variants are {list(VARIANT_KEYS)}"
        )

    variants = phase_variants(config.base, eot_samples=config.eot_samples)
    selected = [variants[key] for key in config.variants]
    device, device_reason = select_device(config.device)
    started = time.perf_counter()

    documents: dict[str, dict[str, Any]] = {}
    for variant in selected:
        logger.info(
            "phase.start variant=%s eot_samples=%d ensemble=%s surrogates=%d",
            variant.key,
            variant.config.eot_samples,
            variant.config.ensemble,
            len(variant.config.surrogates),
        )
        benchmark = BenchmarkConfig(
            image_paths=tuple(image.path for image in corpus),
            output_dir=config.output_dir / variant.key,
            protection=variant.config,
            target=config.target,
            transforms=config.resolve_transforms(),
            models=config.models,
            prompts=EXTENDED_PROMPTS,
            device=config.device,
            # Phase 2 metadata is the point of this run: runtime and peak
            # device memory are part of the question, not incidental.
            record_attack_details=True,
            engine_factory=config.engine_factory,
        )
        document = run_benchmark(benchmark)
        document["transfer"] = _transfer_block(document, config.models)
        document["variant"] = variant.describe()
        documents[variant.key] = document

    return {
        "phase_comparison_version": PHASE_COMPARISON_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "seconds": round(time.perf_counter() - started, 2),
            "device": str(device),
            "device_reason": device_reason,
            "hardware": hardware_summary(device),
            "variants": [variant.key for variant in selected],
        },
        "config": {
            "eot_samples": config.eot_samples,
            "surrogates": list(SURROGATE_KEYS),
            "transforms": [t.key for t in config.resolve_transforms()],
            "prompts": len(EXTENDED_PROMPTS),
            "models": [spec.key for spec in config.models],
        },
        "corpus": [image.to_dict() for image in corpus],
        "variants": {
            key: {
                "variant": variants[key].describe(),
                "document": documents[key],
            }
            for key in config.variants
        },
    }
