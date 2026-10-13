"""Model registry: declarative, lazy, and sequential.

The registry exists to enforce one discipline that any multi-model benchmark on
a laptop GPU eventually gets wrong: **at most one model is resident at a time.**
:func:`ModelRegistry.evaluate_all` walks its adapters, loads one, evaluates
every image, releases it, and moves on. A benchmark that builds six backbones
into a dict and evaluates them in a loop will either OOM at 8 GiB or start
thrashing, and the resulting timings are not comparable anyway.

It also separates two failure modes that are easy to conflate. A model that
*cannot run here* (no dependency, no weights, no memory) is recorded as
skipped with a reason and the run continues. A model that *ran and produced
nonsense* is a benchmark failure and is reported loudly. Conflating them is how
a benchmark ends up reporting coverage it does not have.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from app.evaluation.adapters_classification import build_classifier
from app.evaluation.adapters_detection import build_detection
from app.evaluation.adapters_embedding import build_clip
from app.evaluation.adapters_face import build_face
from app.evaluation.adapters_vlm import build_vlm
from app.evaluation.base import (
    HELD_OUT_ROLES,
    ClassificationObservation,
    EmbeddingObservation,
    FaceObservation,
    ModelAdapter,
    ModelFamily,
    ModelRole,
    ModelSpec,
    ModelUnavailableError,
    Observation,
)
from app.protection.base import RGB

logger = logging.getLogger(__name__)


def select_device(preference: str = "auto") -> tuple[torch.device, str]:
    """Choose a device and say why.

    Returns the device together with a human-readable reason, because a report
    that says ``cuda`` without saying *why* cannot be checked later.

    Args:
        preference: ``"auto"``, ``"cuda"`` or ``"cpu"``.
    """
    if preference == "cpu":
        return torch.device("cpu"), "requested:cpu"
    cuda_available = torch.cuda.is_available()
    if preference == "cuda":
        if not cuda_available:
            return torch.device("cpu"), "requested cuda but unavailable, fell back to cpu"
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return torch.device("cuda"), f"cuda:{index} {name}"
    if not cuda_available:
        return torch.device("cpu"), "auto: no cuda available"
    index = torch.cuda.current_device()
    return torch.device("cuda"), f"auto: cuda:{index} {torch.cuda.get_device_name(index)}"


def hardware_summary(device: torch.device) -> dict[str, Any]:
    """Everything needed to interpret timings and to reproduce a run."""
    summary: dict[str, Any] = {
        "device": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": round(properties.total_memory / (1024**2), 1),
                "gpu_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return summary


def _require_dependency(import_name: str, pip_name: str, model_key: str) -> None:
    """Import an optional dependency or fail as *unavailable*, not *broken*."""
    import importlib.util

    if importlib.util.find_spec(import_name) is None:
        raise ModelUnavailableError(
            model_key, f"{pip_name} is not installed (pip install {pip_name})"
        )


#: The declared Phase 1 model set, as data.
#:
#: ``role`` is the held-out contract of the run. The four convolutional models
#: form the attack's surrogate set (ResNet-18 is the Phase 1 target), the two
#: transformer models were deliberately kept out of every objective so a drop on
#: them is evidence rather than a purchase, and CLIP stays an evaluator.
#: :func:`test_surrogate_and_held_out_sets_are_disjoint` fails if the two lists
#: ever grow into each other.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="resnet18",
        family="classification",
        module="torchvision",
        factory="resnet18",
        options={"weights_name": "IMAGENET1K_V1"},
        notes="protection target",
        role="target",
    ),
    ModelSpec(
        key="densenet121",
        family="classification",
        module="torchvision",
        factory="densenet121",
        options={"weights_name": "IMAGENET1K_V1"},
        role="surrogate",
    ),
    ModelSpec(
        key="efficientnet_b0",
        family="classification",
        module="torchvision",
        factory="efficientnet_b0",
        options={"weights_name": "IMAGENET1K_V1"},
        role="surrogate",
    ),
    ModelSpec(
        key="mobilenet_v3_small",
        family="classification",
        module="torchvision",
        factory="mobilenet_v3_small",
        options={"weights_name": "IMAGENET1K_V1"},
        role="surrogate",
    ),
    ModelSpec(
        key="vit_b_16",
        family="classification",
        module="torchvision",
        factory="vit_b_16",
        options={"weights_name": "IMAGENET1K_V1"},
        notes="held out: transformer, never in the objective",
        role="held_out",
    ),
    ModelSpec(
        key="swin_t",
        family="classification",
        module="torchvision",
        factory="swin_t",
        options={"weights_name": "IMAGENET1K_V1"},
        notes="held out: hierarchical transformer, never in the objective",
        role="held_out",
    ),
    ModelSpec(
        key="clip_vit_b_32",
        family="embedding",
        module="open_clip",
        factory="ViT-B-32",
        options={"pretrained": "laion2b_s34b_b79k"},
        notes="vision-language probe",
        role="evaluator",
    ),
)


#: The face, object and vision-language probes.
#:
#: These are kept out of :data:`DEFAULT_MODELS` rather than appended to it,
#: because they change what a run *reports* and the Phase 1 suite is expected to
#: reproduce its saved baseline byte for byte. A probe is declared here, added
#: by the Phase 2 runner, and never enters any attack objective -- which is what
#: makes its evidence held-out rather than bought.
PROBE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="mtcnn_facenet_v1",
        family="face",
        module="facenet_pytorch",
        factory="mtcnn_inception_resnet_v1",
        options={"min_confidence": 0.7},
        notes="face detector + recognition probe",
        role="held_out",
    ),
    ModelSpec(
        key="yolov8n",
        family="detection",
        module="ultralytics",
        factory="yolov8n",
        options={"confidence": 0.25},
        notes="object detection probe",
        role="held_out",
    ),
    ModelSpec(
        key="vlm",
        family="text",
        module="vlm",
        factory="unconfigured",
        notes=(
            "vision-language probe; reports skipped unless a VlmBackend is "
            "configured, rather than standing in for one"
        ),
        role="held_out",
    ),
)


def build_adapter(
    spec: ModelSpec, device: torch.device, prompts: Sequence[str] | None = None
) -> ModelAdapter:
    """Construct an adapter from its declared spec, without loading weights."""
    if spec.module == "torchvision":
        if spec.family != "classification":
            raise ValueError(f"{spec.key}: torchvision only serves classification")
        return build_classifier(
            device,
            spec.factory,
            spec.options.get("weights_name", "IMAGENET1K_V1"),
        )
    if spec.module == "open_clip":
        if spec.family != "embedding":
            raise ValueError(f"{spec.key}: open_clip only serves embedding")
        _require_dependency("open_clip", "open_clip_torch", spec.key)
        return build_clip(
            device,
            model_name=spec.factory,
            pretrained=spec.options.get("pretrained", "laion2b_s34b_b79k"),
            prompts=tuple(prompts) if prompts else (),
        )
    if spec.module == "facenet_pytorch":
        if spec.family != "face":
            raise ValueError(f"{spec.key}: facenet-pytorch only serves face")
        _require_dependency("facenet_pytorch", "facenet-pytorch", spec.key)
        return build_face(
            device,
            min_confidence=float(spec.options.get("min_confidence", 0.7)),
        )
    if spec.module == "ultralytics":
        if spec.family != "detection":
            raise ValueError(f"{spec.key}: ultralytics only serves detection")
        _require_dependency("ultralytics", "ultralytics", spec.key)
        return build_detection(
            device,
            weights=spec.factory,
            confidence=float(spec.options.get("confidence", 0.25)),
        )
    if spec.module == "vlm":
        if spec.family != "text":
            raise ValueError(f"{spec.key}: vlm only serves text")
        # No backend is configured here on purpose. The adapter is constructed
        # so its spec and description are real, and it then reports ``skipped``
        # with a stated reason instead of answering questions it cannot ask.
        return build_vlm(device)
    raise ValueError(f"{spec.key}: unknown module {spec.module!r}")


@dataclass(slots=True)
class ModelResult:
    """One model's verdict on one set of images.

    Attributes:
        key: Registry key.
        family: Which observation shape was produced.
        status: ``"ok"``, ``"skipped"`` or ``"failed"``.
        reason: Why, when not ``"ok"``.
        observations: Per-image key -> observation.
        seconds: Wall time spent inside ``load() + evaluate()``.
        role: Whether the attack was allowed to see this model. Carried on the
            result rather than looked up later, so a report produced from the
            JSON alone still knows which rows are evidence.
    """

    key: str
    family: ModelFamily
    status: str
    description: str
    role: ModelRole = "held_out"
    observations: dict[str, Observation] = field(default_factory=dict)
    reason: str = ""
    seconds: float = 0.0

    @property
    def held_out(self) -> bool:
        """Whether a drop on this model counts as transfer evidence."""
        return self.role in HELD_OUT_ROLES

    def to_dict(self, *, include_embeddings: bool = False) -> dict[str, Any]:
        """JSON-ready mapping. Non-ok statuses carry their reason and no data."""
        payload: dict[str, Any] = {
            "key": self.key,
            "family": self.family,
            "role": self.role,
            "status": self.status,
            "description": self.description,
            "seconds": round(self.seconds, 3),
        }
        if self.status != "ok":
            payload["reason"] = self.reason
            return payload
        payload["observations"] = {
            name: (
                obs.to_dict(include_embedding=include_embeddings)
                if isinstance(obs, (EmbeddingObservation, FaceObservation))
                else obs.to_dict()
            )
            for name, obs in self.observations.items()
        }
        return payload


class ModelRegistry:
    """Builds, sequences and disposes of the benchmarked models."""

    def __init__(
        self,
        device: torch.device,
        specs: Iterable[ModelSpec] = DEFAULT_MODELS,
        *,
        prompts: Sequence[str] | None = None,
    ) -> None:
        """Declare which models to use and on which device."""
        self.device = device
        self.specs = tuple(s for s in specs if s.enabled)
        self.prompts = tuple(prompts) if prompts else None

    def evaluate_all(self, images: dict[str, RGB]) -> dict[str, ModelResult]:
        """Evaluate every image with every model, one model resident at a time.

        Args:
            images: Named ``uint8`` RGB arrays to read.

        Returns:
            ``model key -> ModelResult``, including skipped and failed entries
            so the caller can report coverage honestly.
        """
        results: dict[str, ModelResult] = {}
        for spec in self.specs:
            results[spec.key] = self._evaluate_one(spec, images)
        return results

    def _evaluate_one(
        self, spec: ModelSpec, images: dict[str, RGB]
    ) -> ModelResult:
        import time

        result = ModelResult(
            key=spec.key,
            family=spec.family,
            status="ok",
            description=spec.notes,
            role=spec.role,
        )
        try:
            adapter = build_adapter(spec, self.device, self.prompts)
        except ModelUnavailableError as exc:
            result.status = "skipped"
            result.reason = exc.reason
            logger.info("model.skipped key=%s reason=%s", spec.key, exc.reason)
            return result
        except Exception as exc:
            result.status = "failed"
            result.reason = f"{type(exc).__name__}: {exc}"
            logger.warning("model.failed key=%s reason=%s", spec.key, result.reason)
            return result

        result.description = adapter.description
        started = time.perf_counter()
        try:
            adapter.load()
            result.observations = {
                name: adapter.evaluate(image) for name, image in images.items()
            }
        except ModelUnavailableError as exc:
            result.status = "skipped"
            result.reason = exc.reason
            logger.info("model.skipped key=%s reason=%s", spec.key, exc.reason)
        except torch.cuda.OutOfMemoryError as exc:
            result.status = "skipped"
            result.reason = f"CUDA out of memory: {exc}"
            logger.warning("model.oom key=%s", spec.key)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            result.status = "failed"
            result.reason = f"{type(exc).__name__}: {exc}"
            logger.warning("model.failed key=%s reason=%s", spec.key, result.reason)
        finally:
            result.seconds = time.perf_counter() - started
            adapter.release()
        return result


def observation_confidence(observation: Observation) -> float | None:
    """Top-1 probability, or ``None`` for embedding models."""
    if isinstance(observation, ClassificationObservation):
        return observation.top1_probability
    return None
