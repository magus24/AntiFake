"""The contract every benchmarked model implements.

An adapter is the only thing in this package that knows how to turn a decoded
RGB array into numbers. The runner, the metrics and the report never import
torchvision or open_clip directly, so adding a model family means adding one
adapter and nothing else.

Two families exist, and they are deliberately not forced into a single
result shape:

* ``classification`` produces a ranked label distribution. What matters is
  whether the top-1 label and its confidence survive.
* ``embedding`` produces a vector, optionally comparable against text
  embeddings. What matters is cosine similarity and the text-similarity
  ranking, because a vision-language model does not report a class label at
  all.

Forcing both into "confidence" would be the single easiest way to make this
benchmark lie, so :class:`ModelObservation` is a tagged union and the report
prints whatever the family actually produced.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from app.protection.base import RGB

ModelFamily = Literal["classification", "embedding", "face", "detection", "text"]

#: What a model is *for* in a given run.
#:
#: This is the field that makes a transfer result mean something. A model the
#: attack optimised against cannot evidence transfer, however large its drop,
#: because it is the model the drop was purchased for. The roles are recorded as
#: data on the spec rather than inferred from the attack configuration, so a
#: report generated hours after the attack still knows which rows were allowed
#: to count.
#:
#: - ``target`` -- the single model Phase 1 optimised against.
#: - ``surrogate`` -- models included in the ensemble objective.
#: - ``held_out`` -- never seen by any variant of the attack.
#: - ``evaluator`` -- a non-classification probe (CLIP), always held out.
ModelRole = Literal["target", "surrogate", "held_out", "evaluator"]

#: Roles that are permitted to appear in the held-out verdict.
HELD_OUT_ROLES: tuple[ModelRole, ...] = ("held_out", "evaluator")


@dataclass(frozen=True, slots=True)
class ClassificationObservation:
    """One image, read by a classifier.

    Attributes:
        top1_index: Winning class index.
        top1_label: Human-readable label, when the weights ship a label list.
        top1_probability: Softmax probability of the winner.
        top1_logit: Raw winning logit. Probability saturates, so the logit is
            kept: a drop from 0.9 to 0.8 and a drop from 0.02 to 0.01 are very
            different events and only the logit tells them apart reliably.
        top5: ``(index, probability)`` for the five highest classes.
        entropy: Shannon entropy of the full distribution in nats. A
            perturbation that smears confidence across the whole label set
            raises it; one that merely swaps two labels leaves it alone.
    """

    top1_index: int
    top1_label: str
    top1_probability: float
    top1_logit: float
    top5: list[tuple[int, float]]
    entropy: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping with a stable key set."""
        return {
            "top1_index": self.top1_index,
            "top1_label": self.top1_label,
            "top1_probability": round(self.top1_probability, 6),
            "top1_logit": round(self.top1_logit, 6),
            "top5": [[i, round(p, 6)] for i, p in self.top5],
            "entropy": round(self.entropy, 6),
        }


@dataclass(frozen=True, slots=True)
class EmbeddingObservation:
    """One image, read by an embedding or vision-language model.

    Attributes:
        embedding: L2-normalised image embedding, so a dot product is cosine
            similarity.
        text_similarities: ``prompt -> cosine similarity`` against the
            configured prompt set.
        text_ranking: Prompts ordered best-first. Rank is what a
            vision-language assistant actually acts on; a similarity that
            moves 0.31 -> 0.28 without reordering anything is a much smaller
            event than the raw delta suggests.
    """

    embedding: np.ndarray
    text_similarities: dict[str, float]
    text_ranking: list[str]

    @property
    def dimension(self) -> int:
        """Embedding length."""
        return int(self.embedding.shape[0])

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """JSON-ready mapping. The raw vector is omitted unless asked for."""
        payload: dict[str, Any] = {
            "embedding_dim": self.dimension,
            "text_similarities": {
                k: round(v, 6) for k, v in sorted(self.text_similarities.items())
            },
            "text_ranking": list(self.text_ranking),
        }
        if include_embedding:
            payload["embedding"] = [round(float(v), 6) for v in self.embedding.tolist()]
        return payload


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One image, read by a face detector and a face-recognition network.

    Detector and embedder are recorded together because the protection is
    claiming to defeat a *pipeline*: a face that is still found but whose
    embedding moved is a different outcome from one that vanished, and merging
    them would lose the distinction the report is supposed to make.

    Attributes:
        detected: Whether the detector returned a box above its threshold.
        confidence: Detector confidence, or ``None`` when nothing was found.
        box: ``(x1, y1, x2, y2)`` in source pixels, or ``None``.
        embedding: L2-normalised ``512``-dimensional face descriptor, or
            ``None`` when the face was not found. Unit length so a dot product
            is cosine similarity against the original.
        reason: Why the face branch produced nothing, when ``detected`` is
            ``False`` and the model itself did run.
    """

    detected: bool
    confidence: float | None
    box: tuple[int, int, int, int] | None
    embedding: np.ndarray | None
    reason: str = ""

    @property
    def dimension(self) -> int | None:
        """Descriptor length, or ``None`` without a descriptor."""
        return None if self.embedding is None else int(self.embedding.shape[0])

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """JSON-ready mapping. The descriptor is omitted unless asked for."""
        payload: dict[str, Any] = {
            "detected": self.detected,
            "confidence": None
            if self.confidence is None
            else round(self.confidence, 6),
            "box": None if self.box is None else [int(v) for v in self.box],
            "embedding_dim": self.dimension,
            "reason": self.reason,
        }
        if include_embedding and self.embedding is not None:
            payload["embedding"] = [round(float(v), 6) for v in self.embedding.tolist()]
        return payload


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object: what, how sure, and where."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "box": [int(v) for v in self.box],
        }


@dataclass(frozen=True, slots=True)
class DetectionObservation:
    """One image, read by an object detector.

    Attributes:
        detections: Every box above the threshold, best first.
    """

    detections: list[Detection]

    @property
    def labels(self) -> list[str]:
        """Detected labels, in confidence order."""
        return [d.label for d in self.detections]

    def best(self, label: str) -> Detection | None:
        """The highest-confidence detection of ``label``, if any."""
        return next((d for d in self.detections if d.label == label), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass(frozen=True, slots=True)
class TextObservation:
    """One image, read by a vision-language model.

    Answers are stored as ``prompt -> answer`` rather than as a single string so
    that the comparison can report *which* question changed. A model that keeps
    naming the same subject but starts refusing the detail question is a
    different failure from one that changes its mind about everything, and a
    single concatenated string would hide that.

    Attributes:
        answers: Prompt -> the model's answer, verbatim.
        model: Human-readable model description for the report.
        reason: Why the branch produced nothing, when the model ran but could
            not answer.
    """

    answers: dict[str, str]
    model: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "model": self.model,
            "answers": dict(sorted(self.answers.items())),
            "answered": len(self.answers),
            "reason": self.reason,
        }


Observation = (
    ClassificationObservation
    | EmbeddingObservation
    | FaceObservation
    | DetectionObservation
    | TextObservation
)


class ModelUnavailableError(RuntimeError):
    """A model cannot be evaluated in this environment, with a stated reason.

    Carrying the reason as data rather than a bare exception is what lets the
    report distinguish "this model is broken" from "this model is not
    available on this machine, and here is why". The registry turns this one
    exception type into a *skipped* status; every other exception becomes a
    *failure*, so a genuine bug can never hide behind an absent dependency.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        """Record which model and why it could not run."""
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"{model_name} unavailable: {reason}")


class ModelAdapter(abc.ABC):
    """One benchmarked model, behind a uniform interface.

    The lifecycle is deliberate and is what keeps VRAM bounded:

    1. :meth:`load` pulls weights onto the device, once.
    2. :meth:`evaluate` is called for every image, any number of times.
    3. :meth:`release` drops the weights and empties the cache.

    The runner walks models one at a time in exactly that order rather than
    holding every model resident, which is the difference between a benchmark
    that fits in 8 GiB and one that does not.
    """

    #: Stable identifier used as the JSON key and in the report table.
    name: str = "abstract"

    #: Which result shape this adapter produces.
    family: ModelFamily = "classification"

    #: Human description of the weights and preprocessing, for the report.
    description: str = "unspecified"

    def __init__(self, device: torch.device) -> None:
        """Bind the adapter to a device. No weights are loaded yet."""
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """``True`` once :meth:`load` has completed."""
        return self._loaded

    @abc.abstractmethod
    def load(self) -> None:
        """Fetch and prepare weights. Called at most once per adapter."""

    @abc.abstractmethod
    def release(self) -> None:
        """Drop weights and free device memory. Must be safe to call twice."""

    @abc.abstractmethod
    def evaluate(self, image: RGB) -> Observation:
        """Read one image.

        Args:
            image: ``uint8`` RGB array of shape ``(H, W, 3)``.

        Returns:
            A :class:`ClassificationObservation` or :class:`EmbeddingObservation`
            matching :attr:`family`.

        Raises:
            ModelUnavailableError: If the model cannot be evaluated here for an
                environmental reason. Raise anything else for a real defect.
        """

    def context(self) -> Any:
        """Return the adapter itself, so the runner can manage its lifecycle."""
        return self

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"{self.name}: evaluate() called before load()")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declarative description of a model the benchmark should try.

    Kept as data so the model list can be written in configuration, printed in
    the report, and compared between runs without importing anything.
    """

    key: str
    family: ModelFamily
    module: str
    factory: str
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    notes: str = ""

    #: Whether the attack was allowed to see this model. See :data:`ModelRole`.
    role: ModelRole = "held_out"

    @property
    def held_out(self) -> bool:
        """Whether a drop on this model counts as transfer evidence."""
        return self.role in HELD_OUT_ROLES
