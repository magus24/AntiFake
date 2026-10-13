"""Configuration for the protection engine.

This module is deliberately free of any FastAPI or I/O concerns: it only
describes *how* a perturbation should be produced. The API layer maps request
fields onto :class:`ProtectionConfig` and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Maximum number of PGD steps we allow. Guards against a hostile/erroneous
# request pinning a worker for minutes.
MAX_PGD_STEPS = 200

# Hard ceiling for the L-inf bound, expressed in 8-bit levels. 32/255 is well
# past the point of "barely perceptible" and exists only as a safety stop.
MAX_EPS_255 = 32.0

DeviceName = Literal["auto", "cpu", "cuda"]
LossName = Literal["cw_topk", "cw_top1"]
InitName = Literal["zeros", "uniform"]

#: Ceiling on EOT samples per step. Each sample is a full differentiable
#: forward and backward through every surrogate, so this is the single largest
#: cost multiplier in the attack and needs a stop.
MAX_EOT_SAMPLES = 32

#: Ceiling on ensemble members. Six ImageNet backbones is already the useful
#: set; beyond that the marginal transfer gain is small and the runtime and
#: memory cost are not.
MAX_SURROGATES = 12


class ProtectionConfigError(ValueError):
    """Raised when a :class:`ProtectionConfig` is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ProtectionConfig:
    """Immutable description of one protection run.

    Attributes:
        eps: L-inf bound of the perturbation in ``[0, 1]`` float space.
            ``8 / 255`` is the default: empirically at or below the visibility
            threshold of a 1:1 display while remaining large enough to survive
            JPEG re-encoding.
        alpha: Step size of the projected gradient descent loop, same units as
            ``eps``. Typically ``eps / 4``.
        steps: Number of PGD iterations. More steps == better attack, linear
            cost in runtime.
        seed: Makes a run reproducible. With ``init="zeros"`` the attack is
            deterministic regardless of the seed; with ``init="uniform"`` the
            seed selects the starting point.
        target_topk: How many top logits the Carlini-Wagner margin loss sums.
            ``1`` matches a standard CW margin attack, higher values push the
            decision boundary away from the whole top of the ranking.
        device: ``"cpu"``, ``"cuda"`` or ``"auto"`` (CUDA when available).
        init: Starting point of the perturbation.
        eot_samples: Transforms sampled per optimisation step. ``0`` (the
            default) disables EOT and reproduces the Phase 1 attack exactly,
            which is what makes the Phase 1 / Phase 2 comparison a controlled
            experiment rather than two different attacks.
        eot_families: Which differentiable transform families to sample from.
        eot_seed: Base seed for the EOT draws, kept separate from ``seed`` so
            the EOT schedule can be varied without changing the initialisation
            point, and vice versa.
        ensemble: Whether to optimise against several architectures. ``False``
            by default, for the same backward-compatibility reason.
        ensemble_weights: ``model key -> relative weight``. Empty means equal
            weights over the declared surrogates.
        surrogates: Surrogate model keys, in declaration order. Empty means the
            single Phase 1 target.
        max_resident_surrogates: How many surrogates may hold device memory at
            once. ``None`` loads all of them, which is faster; a smaller value
            chunks them per step, which is slower but bounds VRAM and is
            mathematically exact for a sum of per-model losses.
    """

    eps: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 40
    seed: int = 0
    target_topk: int = 3
    loss: LossName = "cw_topk"
    device: DeviceName = "auto"
    init: InitName = "zeros"
    model_weights: str = "imagenet1k_v1"
    preserve_dimensions: bool = True
    eot_samples: int = 0
    eot_families: tuple[str, ...] = (
        "identity",
        "resize_short_side",
        "random_crop",
        "center_crop",
        "scale_jitter",
        "jpeg",
        "brightness",
        "contrast",
        "blur",
        "sharpen",
        "affine",
    )
    eot_seed: int = 0
    ensemble: bool = False
    ensemble_weights: tuple[tuple[str, float], ...] = ()
    surrogates: tuple[str, ...] = ()
    max_resident_surrogates: int | None = None

    def __post_init__(self) -> None:
        """Reject internally inconsistent configurations at construction time.

        Failing here means a bad value is a startup error, not a subtly broken
        perturbation discovered later.
        """
        if not 0.0 < self.eps <= 1.0:
            raise ProtectionConfigError(f"eps must be in (0, 1], got {self.eps!r}")
        if self.eps * 255.0 > MAX_EPS_255:
            raise ProtectionConfigError(
                f"eps must not exceed {MAX_EPS_255}/255, got {self.eps * 255.0:.3f}/255"
            )
        if self.alpha <= 0.0:
            raise ProtectionConfigError(f"alpha must be > 0, got {self.alpha!r}")
        if self.alpha > self.eps:
            raise ProtectionConfigError(
                f"alpha ({self.alpha:.6f}) must not exceed eps ({self.eps:.6f}); "
                "otherwise the projection is meaningless"
            )
        if not 1 <= self.steps <= MAX_PGD_STEPS:
            raise ProtectionConfigError(
                f"steps must be in [1, {MAX_PGD_STEPS}], got {self.steps}"
            )
        if self.target_topk < 1:
            raise ProtectionConfigError(
                f"target_topk must be >= 1, got {self.target_topk}"
            )
        if self.loss not in ("cw_topk", "cw_top1"):
            raise ProtectionConfigError(f"unknown loss {self.loss!r}")
        if self.init not in ("zeros", "uniform"):
            raise ProtectionConfigError(f"unknown init {self.init!r}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ProtectionConfigError(f"unknown device {self.device!r}")
        if not 0 <= self.eot_samples <= MAX_EOT_SAMPLES:
            raise ProtectionConfigError(
                f"eot_samples must be in [0, {MAX_EOT_SAMPLES}], "
                f"got {self.eot_samples}"
            )
        if self.eot_samples > 0 and not self.eot_families:
            raise ProtectionConfigError(
                "eot_families must not be empty when eot_samples > 0"
            )
        if self.ensemble and not self.surrogates:
            raise ProtectionConfigError(
                "surrogates must name at least one model when ensemble=True"
            )
        if len(self.surrogates) > MAX_SURROGATES:
            raise ProtectionConfigError(
                f"at most {MAX_SURROGATES} surrogates are allowed, "
                f"got {len(self.surrogates)}"
            )
        if len(set(self.surrogates)) != len(self.surrogates):
            raise ProtectionConfigError(f"surrogates must be unique: {self.surrogates}")
        unknown = {key for key, _ in self.ensemble_weights} - set(self.surrogates)
        if unknown:
            raise ProtectionConfigError(
                f"ensemble_weights names non-surrogate models: {sorted(unknown)}"
            )
        if any(weight < 0 for _, weight in self.ensemble_weights):
            raise ProtectionConfigError("ensemble_weights must not be negative")
        if self.max_resident_surrogates is not None and self.max_resident_surrogates < 1:
            raise ProtectionConfigError(
                f"max_resident_surrogates must be >= 1, "
                f"got {self.max_resident_surrogates}"
            )

    @property
    def weight_map(self) -> dict[str, float]:
        """``model key -> weight`` for the ensemble.

        Members omitted from ``ensemble_weights`` default to 1.0 rather than
        disappearing from the map, so the result always covers every surrogate
        and a partial override reads as an override.
        """
        declared = dict(self.ensemble_weights)
        return {key: declared.get(key, 1.0) for key in self.surrogates}

    @property
    def uses_eot(self) -> bool:
        """Whether the attack samples transformed renderings."""
        return self.eot_samples > 0

    @property
    def uses_ensemble(self) -> bool:
        """Whether the attack optimises against a declared surrogate set.

        One surrogate is still an ensemble member, not a no-op: a run with
        ``ensemble=True, surrogates=('resnet18',)`` is the control that proves
        the ensemble machinery itself is not what changed the numbers.
        """
        return self.ensemble and bool(self.surrogates)

    @property
    def eps_255(self) -> float:
        """L-inf bound expressed in 8-bit levels, for logging and responses."""
        return self.eps * 255.0
    @classmethod
    def create(cls, *, steps: int, **kwargs: object) -> ProtectionConfig:
        """Build a config, folding an out-of-range ``steps`` into bounds.

        Used at the process boundary (environment variables, which are
        operator-supplied and easy to get wrong). The constructor itself stays
        strict, so a mistake deep in the code is still an error rather than a
        silent clamp.
        """
        return cls(steps=max(1, min(int(steps), MAX_PGD_STEPS)), **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TargetModelConfig:
    """Identifies the single AI pipeline this MVP defends against.

    AntiFake does **not** claim to defend against arbitrary models. Experiment
    #1 targets one specific, fully specified pipeline: a torchvision ResNet-18
    trained on ImageNet-1k and evaluated through the stock torchvision
    preprocessing transform (shortest-side resize 256, centre crop 224,
    ImageNet normalisation). Both the weights and the preprocessing are part of
    the threat model -- changing either invalidates the guarantee.
    """

    name: str = "resnet18"
    weights: str = "imagenet1k_v1"
    resize_short_side: int = 256
    crop_size: int = 224
    interpolation: str = "bilinear"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    source: str = "torchvision.models"

    @property
    def description(self) -> str:
        """One-line human description of the exact pipeline being defended."""
        return (
            f"{self.source}:{self.name} (weights={self.weights}) "
            f"via resize-{self.resize_short_side}/center-crop-{self.crop_size}"
        )


@dataclass(frozen=True, slots=True)
class ProtectionResultMeta:
    """Diagnostics produced alongside a protected image."""

    engine: str
    target_model: str
    device: str
    seed: int
    steps: int
    eps_255: float
    linf_255: float
    l2: float
    psnr_db: float
    ssim: float
    max_abs_pixel_delta: int
    width: int
    height: int
    attack_ms: float

    #: Phase 2 configuration label, e.g. ``pgd-linf-ensemble4-resnet18+eot4``.
    #: Recorded separately from ``engine`` so a Phase 2 row can never be
    #: mistaken for the Phase 1 baseline it is being compared against.
    variant: str = ""

    #: High-water mark of device memory during the attack, in MiB. ``0.0`` on
    #: CPU, where there is nothing to measure and a made-up number would be
    #: worse than none.
    peak_memory_mib: float = 0.0

    #: Number of distinct renderings averaged per optimisation step.
    eot_samples: int = 0

    #: Surrogate keys, empty when the run is the single-model Phase 1 attack.
    surrogate_models: tuple[str, ...] = ()

    #: ``True`` when a resident ensemble ran out of device memory and the attack
    #: restarted in chunked mode. Reported rather than hidden: the run still
    #: succeeded, but it took a different path to get there.
    oom_fallback: bool = False

    #: Ensemble accounting: load time, chunking, per-model time, peak memory.
    surrogate_stats: dict[str, Any] = field(default_factory=dict)

    extra: dict[str, float] = field(default_factory=dict)
