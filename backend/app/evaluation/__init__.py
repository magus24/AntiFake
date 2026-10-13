"""Phase 1 evaluation framework.

Measures how well the current protection transfers across vision models, image
delivery formats and post-processing transforms. It observes the product; it
does not modify it. Nothing in this package is imported by the API or the
protection engine, so the benchmark can be extended or broken without ever
affecting what a user uploads.

Entry point: :func:`app.evaluation.benchmark.run_benchmark`.
"""

from __future__ import annotations

from app.evaluation.base import (
    ClassificationObservation,
    EmbeddingObservation,
    ModelAdapter,
    ModelSpec,
    Observation,
)
from app.evaluation.benchmark import (
    BENCHMARK_VERSION,
    BenchmarkConfig,
    markdown_report,
    run_benchmark,
    write_report,
)
from app.evaluation.comparison import compare, compare_classification, compare_embedding
from app.evaluation.registry import (
    DEFAULT_MODELS,
    ModelRegistry,
    ModelResult,
    hardware_summary,
    select_device,
)
from app.evaluation.transforms import EvaluationTransform, default_transforms
from app.evaluation.visual_quality import VisualQuality, compute_visual_quality

__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_MODELS",
    "BenchmarkConfig",
    "ClassificationObservation",
    "EmbeddingObservation",
    "EvaluationTransform",
    "ModelAdapter",
    "ModelRegistry",
    "ModelResult",
    "ModelSpec",
    "Observation",
    "VisualQuality",
    "compare",
    "compare_classification",
    "compare_embedding",
    "compute_visual_quality",
    "default_transforms",
    "hardware_summary",
    "markdown_report",
    "run_benchmark",
    "select_device",
    "write_report",
]
