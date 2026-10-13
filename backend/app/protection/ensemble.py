"""Surrogate ensemble: optimise against several architectures at once.

Why an ensemble
---------------
Phase 1 optimised against ResNet-18 alone and measured the consequence: 96%
confidence drop on the target, 14% on MobileNetV3, and a cosine similarity of
0.944 for CLIP. A perturbation that only moves one network is not protection,
it is a trick against one network. Attacking a set of surrogates whose members
differ in depth, receptive field and tokenisation forces the perturbation into
the region of input space where *many* models are wrong, which is the only
place a perturbation can live and still survive contact with a model nobody
chose in advance.

Memory, and why gradient accumulation is exact here
----------------------------------------------------
Six ImageNet backbones do not all fit in 8 GiB together with their
activations, and the obvious workaround -- load them one at a time and add up
the gradients -- is usually implemented incorrectly. It is worth being precise
about why it is *not* an approximation for this particular loss:

    L_total = sum_i w_i * L_i(delta)
    d L_total / d delta = sum_i w_i * d L_i / d delta

Each ``L_i`` depends only on ``delta`` and on model ``i``'s weights. So
accumulating ``w_i * dL_i/d delta`` in chunks, loading and releasing a chunk
between additions, produces **bit-comparable** gradients to holding every model
resident. There is no approximation and no ordering effect, because the sum is
over independent terms. What chunking does cost is time: weights are read from
disk once per chunk per step, which is why :class:`SurrogateEnsemble` keeps
models resident whenever it can and only falls back to chunking when the memory
budget demands it.

A subtlety worth stating: with chunking, a *single* sample of the EOT draw must
be reused across chunks so that all models see the same transformed batch. If
each chunk resampled independently the accumulation would no longer be the
gradient of any single expectation, and the objective would quietly become
something else. :meth:`SurrogateEnsemble.backward_into` therefore takes the
already-transformed batch as an argument.

Held-out discipline
-------------------
The ensemble is a set of *surrogates*: the models the attack is allowed to see.
That makes them useless as evidence. :attr:`SurrogateEnsemble.keys` is exposed
so the benchmark can mark them, and every transfer number reported for a
surrogate is excluded from the held-out verdict. A protection that only damages
the models it was optimised against has not transferred, however good its
optimisation loss looks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from app.protection.config import TargetModelConfig
from app.protection.target_model import TargetModel, load_target_model

logger = logging.getLogger(__name__)


def cw_topk_loss(logits: Tensor, topk: int) -> Tensor:
    """Sum of the ``topk`` highest logits, per batch element.

    Identical in form to the Phase 1 single-model loss so that a Phase 1 run
    and an ensemble run are attacking the same quantity, just averaged over more
    models and more renderings. Returns one scalar per batch element, which the
    caller weights and sums.
    """
    count = min(topk, logits.shape[-1])
    return torch.topk(logits, k=count, dim=-1).values.sum(dim=-1)


@dataclass(frozen=True, slots=True)
class SurrogateSpec:
    """One model the attack is allowed to optimise against.

    Attributes:
        key: Stable identifier, matching the benchmark's model key so the two
            can be joined when reporting.
        weight: Relative weight in the ensemble loss. Normalised internally, so
            ``{resnet18: 1, densenet121: 1}`` and ``{resnet18: 50,
            densenet121: 50}`` are the same objective.
        notes: Free text for the report.
    """

    key: str
    config: TargetModelConfig
    weight: float = 1.0
    notes: str = ""


#: The declared surrogate set: every model the attack is allowed to optimise.
#:
#: Deliberately the four convolutional families. ``vit_b_16`` and ``swin_t`` are
#: excluded on purpose so that a transformer model is a member of no objective
#: in any variant -- a drop there is evidence that the perturbation left the
#: family it was tuned for, while a drop on a surrogate only shows the
#: optimiser did its job. CLIP is excluded too: it is an *evaluation* model, and
#: optimising against the evaluator destroys the held-out test.
#:
#: The Phase 1 configuration (``resnet18`` alone) is reachable by selecting just
#: the first entry.
DEFAULT_SURROGATES: tuple[SurrogateSpec, ...] = (
    SurrogateSpec(
        key="resnet18",
        config=TargetModelConfig(name="resnet18"),
        weight=1.0,
        notes="phase-1 target",
    ),
    SurrogateSpec(
        key="densenet121",
        config=TargetModelConfig(name="densenet121"),
        weight=1.0,
        notes="dense connections, different receptive-field reuse",
    ),
    SurrogateSpec(
        key="efficientnet_b0",
        config=TargetModelConfig(name="efficientnet_b0"),
        weight=1.0,
        notes="depthwise separable convolutions, scaled compound params",
    ),
    SurrogateSpec(
        key="mobilenet_v3_small",
        config=TargetModelConfig(name="mobilenet_v3_small"),
        weight=1.0,
        notes="mobile-width regime, squeezed excitation",
    ),
)


def normalize_weights(specs: Sequence[SurrogateSpec]) -> dict[str, float]:
    """Scale weights so they sum to one.

    Normalising makes the ensemble loss scale-comparable to the single-model
    loss, which keeps ``alpha`` meaningful when switching between Phase 1 and
    Phase 2: without this, adding five more models to a sum of top-k logits
    would silently multiply the effective step size by six.

    A negative member is rejected outright rather than allowed to cancel a
    positive one. The sum check alone passes ``{-1, 2}``, and a surrogate with
    negative weight steers the perturbation *towards* confident classification
    of a model the report lists as protected.
    """
    negatives = [s.key for s in specs if s.weight < 0]
    if negatives:
        raise ValueError(
            f"surrogate weights must not be negative, got {sorted(negatives)}"
        )
    total = sum(s.weight for s in specs)
    if total <= 0:
        raise ValueError(
            f"surrogate weights must sum above zero, got {total}; "
            "an all-zero request would divide by zero"
        )
    return {s.key: s.weight / total for s in specs}


@dataclass(slots=True)
class EnsembleStats:
    """Runtime and memory accounting for the report.

    Attributes:
        model_load_seconds: Total time spent materialising weights.
        peak_memory_mib: High-water mark of device memory, when measurable.
        chunked: ``True`` when models were loaded in chunks per step rather
            than held resident, which is the honest explanation for a long
            runtime.
        chunks_per_step: How many load cycles one step cost.
    """

    model_load_seconds: float = 0.0
    peak_memory_mib: float = 0.0
    chunked: bool = False
    chunks_per_step: int = 1
    per_model_seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "model_load_seconds": round(self.model_load_seconds, 3),
            "peak_memory_mib": round(self.peak_memory_mib, 1),
            "chunked": self.chunked,
            "chunks_per_step": self.chunks_per_step,
            "per_model_seconds": {
                k: round(v, 3) for k, v in sorted(self.per_model_seconds.items())
            },
        }


class SurrogateEnsemble:
    """A weighted, memory-bounded set of differentiable target pipelines.

    Args:
        specs: Which surrogates to optimise against.
        device: Device to run the attack on.
        topk: Top-k logits summed by the CW-style loss.
        max_resident: How many models may be held in device memory at once.
            ``None`` or a value >= ``len(specs)`` loads them all up front, which
            is faster. A smaller value chunks them per step, which is slower but
            bounds memory, and is *exact* rather than approximate.
        resident: Force the strategy, mostly so tests and reports can compare.
    """

    def __init__(
        self,
        specs: Sequence[SurrogateSpec],
        device: torch.device,
        *,
        topk: int = 3,
        max_resident: int | None = None,
        resident: bool = True,
    ) -> None:
        """Bind the surrogate set without loading anything yet."""
        if not specs:
            raise ValueError("an ensemble needs at least one surrogate")
        self.specs = tuple(specs)
        self.device = device
        self.topk = topk
        self.weights = normalize_weights(self.specs)
        self.stats = EnsembleStats()
        limit = len(self.specs) if max_resident is None else max_resident
        self._chunks: list[tuple[SurrogateSpec, ...]] = [
            self.specs[i : i + limit] for i in range(0, len(self.specs), limit)
        ]
        self.stats.chunks_per_step = len(self._chunks)
        self.stats.chunked = len(self._chunks) > 1
        self._resident = resident and len(self._chunks) == 1
        self._models: dict[str, TargetModel] = {}
        self._gradient: Tensor | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        """Surrogate keys, in declaration order."""
        return tuple(s.key for s in self.specs)

    @property
    def holds_resident(self) -> bool:
        """Whether weights are being kept in memory between steps."""
        return self._resident

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Load every surrogate up front, if the strategy is resident.

        A failure to load one surrogate is fatal for the ensemble rather than
        silently dropped: an operator who declared six surrogates and got four
        would be measuring a different attack than the one configured, and the
        weights would silently renormalise around the missing member.

        Models are loaded with ``cache=False`` because this object is their
        owner. Reusing the process-wide model cache would leave every surrogate
        reachable after :meth:`close`, which turns a documented memory bound
        into a slow leak that only shows up under VRAM pressure.
        """
        if not self._resident or self._models:
            return
        for spec in self.specs:
            started = time.perf_counter()
            self._models[spec.key] = load_target_model(
                spec.config, device=self.device, cache=False
            )
            self.stats.model_load_seconds += time.perf_counter() - started
            logger.info("ensemble.loaded key=%s", spec.key)
        self._record_memory()

    def close(self) -> None:
        """Drop references so the device allocator can reclaim the memory."""
        self._models.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._record_memory()

    def __enter__(self) -> SurrogateEnsemble:
        """Open on entry."""
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on exit."""
        self.close()

    # -- the loss ----------------------------------------------------------

    def backward_into(
        self,
        rendered: Tensor,
        target: Tensor,
        *,
        per_model: dict[str, float] | None = None,
    ) -> float:
        """Accumulate ``dL/d target`` for every surrogate, and return the loss.

        Differentiating with respect to ``target`` rather than with respect to
        ``rendered`` is what makes this compose with EOT. The rendered batch is
        the output of a chain of differentiable transforms whose internal
        buffers a single ``backward`` would free, so the gradient is taken all
        the way back to the leaf in one shot and ``retain_graph=True`` keeps that
        shared prefix alive for the remaining surrogates.

        An earlier version detached the input and differentiated the surrogate
        graphs separately. That silently severed the EOT chain: the returned
        gradient belonged to the rendered batch and the engine's own backward
        then found the leaf unused, which PyTorch reports as a confusing
        "differentiated Tensors appear to not have been used" error rather than
        as the modelling mistake it was.

        Args:
            rendered: The already-transformed batch handed to each surrogate.
            target: The leaf to differentiate with respect to, i.e. the batch
                carrying the perturbation.
            per_model: Optional mapping, updated in place with each surrogate's
                loss for the report.

        Returns:
            The summed, weighted loss value.
        """
        total_value = 0.0
        for index, chunk in enumerate(self._chunks):
            models = self._resident_models(chunk)
            try:
                for spec, model in models:
                    started = time.perf_counter()
                    logits = model(rendered)
                    per_model_loss = cw_topk_loss(logits, self.topk)
                    loss = (self.weights[spec.key] * per_model_loss).sum()
                    (gradient,) = torch.autograd.grad(
                        loss, target, retain_graph=True
                    )
                    self._gradient = (
                        gradient if self._gradient is None else self._gradient + gradient
                    )
                    value = float(loss.detach())
                    total_value += value
                    if per_model is not None:
                        per_model[spec.key] = per_model.get(spec.key, 0.0) + value
                    self.stats.per_model_seconds[spec.key] = (
                        self.stats.per_model_seconds.get(spec.key, 0.0)
                        + (time.perf_counter() - started)
                    )
            finally:
                if not self._resident:
                    del models
                    if self.device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if index == len(self._chunks) - 1:
                self._record_memory()
        return total_value

    def _resident_models(
        self, chunk: Sequence[SurrogateSpec]
    ) -> list[tuple[SurrogateSpec, TargetModel]]:
        """The loaded pipelines for one chunk, loading them if not resident.

        Chunked loads are deliberately uncached: they are released at the end of
        the step, and putting them in the process-wide cache would mean the next
        step finds them, and the step after that still has the previous chunk.

        For a resident set the model is loaded on demand if ``open()`` has not
        run, so correctness never depends on the caller remembering to call it.
        ``open()`` stays a warm-up that avoids a load landing mid-step.
        """
        if not self._resident:
            loaded: list[tuple[SurrogateSpec, TargetModel]] = []
            for spec in chunk:
                started = time.perf_counter()
                loaded.append(
                    (
                        spec,
                        load_target_model(spec.config, device=self.device, cache=False),
                    )
                )
                self.stats.model_load_seconds += time.perf_counter() - started
            return loaded
        resident_models: list[tuple[SurrogateSpec, TargetModel]] = []
        for spec in chunk:
            model = self._models.get(spec.key)
            if model is None:
                started = time.perf_counter()
                model = load_target_model(
                    spec.config, device=self.device, cache=False
                )
                self.stats.model_load_seconds += time.perf_counter() - started
                self._models[spec.key] = model
            resident_models.append((spec, model))
        return resident_models

    def start_step(self) -> None:
        """Clear the per-sample gradient accumulator."""
        self._gradient = None

    @property
    def gradient(self) -> Tensor:
        """The summed gradient of the ensemble loss w.r.t. the perturbation.

        This is the value the PGD loop steps against. It is the sum of the
        weighted per-surrogate gradients, which -- by the argument in the module
        docstring -- equals the gradient of the full ensemble loss whether the
        surrogates were held resident or loaded in chunks.
        """
        if self._gradient is None:
            raise RuntimeError("gradient requested before any backward_into() call")
        return self._gradient

    def _record_memory(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        self.stats.peak_memory_mib = max(self.stats.peak_memory_mib, allocated)
