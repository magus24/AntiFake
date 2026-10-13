"""The protection engine: deterministic, bounded adversarial perturbation.

Algorithm
---------
Projected Gradient Descent (PGD) with an L-infinity budget, run in the *input
pixel space of the target pipeline* and then resampled to the source
resolution. Concretely:

1. The image is resized to the target's own working resolution (shortest side
   256, centre crop 224) using differentiable bilinear interpolation, so the
   perturbation is optimised against exactly the transform the classifier will
   apply. Optimising here instead of at full resolution is what makes the
   result survive the resize a scraper will perform, and keeps CPU cost
   independent of the input size.
2. A Carlini-Wagner style margin loss on the top-``k`` logits is minimised for
   ``steps`` iterations: each step takes a signed-gradient step of size ``alpha``
   and projects the perturbation back onto the L-infinity ball of radius
   ``eps``, then clamps the result back into the valid intensity range.
3. The final perturbation is resampled back to the original height/width and
   added to the source image, so the protected output keeps the source
   dimensions exactly.

This is a *reproducible* optimisation against one named model, not noise: the
same image, seed and configuration always yield a bit-identical result, and the
perturbation is graded (L-infinity bounded) rather than random.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from app.protection.base import RGB, ProtectionEngine
from app.protection.config import (
    ProtectionConfig,
    ProtectionResultMeta,
    TargetModelConfig,
)
from app.protection.ensemble import SurrogateEnsemble, SurrogateSpec
from app.protection.eot import EotConfig, EotSampler
from app.protection.metrics import DifferenceMetrics, compute_metrics
from app.protection.target_model import TargetModel, load_target_model, resolve_device

logger = logging.getLogger(__name__)


def _validate_rgb_array(image: RGB) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a numpy array, got {type(image).__name__}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 array, got dtype {image.dtype}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an (H, W, 3) RGB array, got shape {image.shape}."
        )
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("Image has zero width or height.")


def _to_tensor(image: RGB, device: torch.device) -> Tensor:
    """``(H, W, 3) uint8`` -> ``(1, 3, H, W) float32`` in ``[0, 1]``."""
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _to_array(tensor: Tensor) -> RGB:
    """``(1, 3, H, W) float32`` in ``[0, 1]`` -> ``(H, W, 3) uint8``."""
    array = (
        tensor.detach()
        .clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(array)


def _cw_margin_loss(logits: Tensor, topk: int) -> Tensor:
    """Sum of the ``topk`` highest logits.

    Minimising it drives the top of the ranking down without collapsing the
    whole distribution, which is the behaviour we want from a protective
    perturbation: we care about *denying* a confident prediction, not about
    forcing a specific wrong label.
    """
    count = min(topk, logits.shape[-1])
    return torch.topk(logits, k=count, dim=-1).values.sum()


def resolve_surrogates(config: ProtectionConfig) -> tuple[SurrogateSpec, ...]:
    """Turn the config's surrogate keys into concrete specs.

    Kept separate from the engine so that the same resolution logic is used by
    the API path, the benchmark and the tests. The Phase 1 target is always a
    legal member: a configuration that names surrogates but omits ResNet-18 is
    allowed, and a configuration that names nothing resolves to exactly the
    Phase 1 single-model attack.
    """
    if not config.surrogates:
        return (
            SurrogateSpec(
                key=config_target_key(config),
                config=TargetModelConfig(weights=config.model_weights),
                weight=1.0,
                notes="phase-1 target",
            ),
        )
    weights = config.weight_map
    return tuple(
        SurrogateSpec(
            key=key,
            config=TargetModelConfig(name=key, weights=config.model_weights),
            weight=weights.get(key, 1.0),
        )
        for key in config.surrogates
    )


def config_target_key(config: ProtectionConfig) -> str:
    """Registry key of the Phase 1 target model."""
    return TargetModelConfig(weights=config.model_weights).name


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDNN autotuning picks algorithms by wall-clock timing, which makes
    # results vary run to run. Pin it off when CUDA is in play.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AdversarialProtectionEngine(ProtectionEngine):
    """PGD protective perturbation against a single named target pipeline.

    Args:
        config: Attack hyper-parameters. See :class:`ProtectionConfig`.
        target_config: Identity of the pipeline being defended against.
        model: Pre-loaded target. When omitted, one is loaded (and cached)
            for the resolved config and device.
    """

    name = "pgd-linf-resnet18"
    preserves_dimensions = True

    def __init__(
        self,
        config: ProtectionConfig | None = None,
        target_config: TargetModelConfig | None = None,
        *,
        model: TargetModel | None = None,
    ) -> None:
        """Prepare the engine and load the target model.

        The model is resolved eagerly so a misconfigured or unreachable target
        fails here rather than on the first user upload.
        """
        self.config = config or ProtectionConfig()
        self.target_config = target_config or TargetModelConfig()
        self.target_description = self.target_config.description
        self.model = model or load_target_model(
            self.target_config, device=resolve_device(self.config.device)
        )
        # An injected model already knows where it lives; the config's device
        # preference must not override that, or the attack tensors and the
        # weights end up on different devices and fail deep inside the loop.
        self.device = self.model.device
        _set_determinism(self.config.seed)

        # The ensemble is built lazily-but-eagerly: it resolves the surrogate
        # specs now so a bad key fails at construction, but weights are loaded
        # on `open()` so that a caller who never attacks does not pay for them.
        self.ensemble: SurrogateEnsemble | None = None
        if self.config.uses_ensemble:
            self.ensemble = SurrogateEnsemble(
                resolve_surrogates(self.config),
                self.device,
                topk=1 if self.config.loss == "cw_top1" else self.config.target_topk,
                max_resident=self.config.max_resident_surrogates,
            )
        self.eot = EotSampler(
            EotConfig(
                samples=self.config.eot_samples,
                seed=self.config.eot_seed,
                families=self.config.eot_families,
            ),
            self.device,
        )
        #: Per-step diagnostics for the report: the mean loss and the fraction
        #: of it contributed by the largest single surrogate. A high concentration
        #: means the "ensemble" is really one model with five bystanders.
        self.trace: list[dict[str, float | int]] = []

        #: Set when the attack had to fall back to chunked surrogate loading.
        self.oom_fallback = False
        logger.info(
            "engine.constructed engine=%s variant=%s target=%s device=%s "
            "eps=%.4f/255 alpha=%.4f/255 steps=%d seed=%d eot=%d ensemble=%s",
            self.name,
            self.variant,
            self.target_description,
            self.device,
            self.config.eps_255,
            self.config.alpha * 255.0,
            self.config.steps,
            self.config.seed,
            self.config.eot_samples,
            self.ensemble.keys if self.ensemble else (),
        )

    @property
    def variant(self) -> str:
        """Which Phase 2 features are active, for the report.

        The class-level ``name`` stays ``"pgd-linf-resnet18"`` forever, because
        the API contract and the Phase 1 artefacts depend on it. Reporting a
        Phase 2 run under the Phase 1 name would be indistinguishable from a
        regression, so the configuration is exposed separately instead.
        """
        parts = ["pgd-linf"]
        if self.config.uses_ensemble:
            parts.append(f"ensemble{len(self.ensemble.keys) if self.ensemble else 0}")
        parts.append(self.target_config.name)
        if self.config.uses_eot:
            parts.append(f"+eot{self.config.eot_samples}")
        return "-".join(parts)

    # -- public API --------------------------------------------------------

    def protect(self, image: RGB) -> RGB:
        """Return a protected copy of ``image`` at identical dimensions."""
        return self.protect_with_meta(image).image

    def protect_with_meta(self, image: RGB) -> ProtectionOutput:
        """Like :meth:`protect`, but also returns timing and difference metrics."""
        _validate_rgb_array(image)
        started = time.perf_counter()
        self._begin_run()

        original = image.astype(np.float32) / 255.0
        base_small = self._to_working_resolution(image)

        try:
            perturbation = self._run_pgd(base_small)
        except torch.cuda.OutOfMemoryError:
            perturbation = self._restart_chunked(base_small)
        peak_memory_mib = self._peak_memory_mib()

        perturbation_full = self._resample_to_source(perturbation, image.shape)
        perturbation_full = np.clip(
            perturbation_full, -self.config.eps, self.config.eps
        )

        protected_float = np.clip(original + perturbation_full, 0.0, 1.0)
        protected = np.round(protected_float * 255.0).astype(np.uint8)
        protected = np.ascontiguousarray(protected)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = compute_metrics(image, protected)
        logger.info(
            "engine.protected width=%d height=%d device=%s steps=%d attack_ms=%.1f "
            "linf=%.1f/255 l2=%.3f psnr=%.2fdB ssim=%.5f peak_mem=%.1fMiB",
            image.shape[1],
            image.shape[0],
            self.device,
            self.config.steps,
            elapsed_ms,
            metrics.linf_255,
            metrics.l2,
            metrics.psnr_db if math.isfinite(metrics.psnr_db) else float("inf"),
            metrics.ssim,
            peak_memory_mib,
        )

        meta = ProtectionResultMeta(
            engine=self.name,
            target_model=self.target_description,
            device=str(self.device),
            seed=self.config.seed,
            steps=self.config.steps,
            eps_255=self.config.eps_255,
            linf_255=metrics.linf_255,
            l2=metrics.l2,
            psnr_db=metrics.psnr_db if math.isfinite(metrics.psnr_db) else 999.0,
            ssim=metrics.ssim,
            max_abs_pixel_delta=metrics.linf_8bit,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            attack_ms=round(elapsed_ms, 2),
            variant=self.variant,
            peak_memory_mib=peak_memory_mib,
            eot_samples=self.config.eot_samples,
            surrogate_models=self.ensemble.keys if self.ensemble else (),
            oom_fallback=self.oom_fallback,
            surrogate_stats=self.ensemble.stats.to_dict() if self.ensemble else {},
        )
        return ProtectionOutput(image=protected, meta=meta, metrics=metrics)

    # -- memory management -------------------------------------------------

    def _begin_run(self) -> None:
        """Reset per-run counters so the reported numbers belong to *this* run.

        Peak memory is a high-water mark for the whole device, so a peak left
        over from a previous request would be attributed to this one. Stats are
        reset immediately before the attack rather than at construction, so the
        figure measures the attack and not the weight loading.
        """
        self.oom_fallback = False
        self.trace = []
        if self.ensemble is not None:
            self.ensemble.stats.model_load_seconds = 0.0
            self.ensemble.stats.per_model_seconds = {}
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_mib(self) -> float:
        """High-water mark for this run, or ``0.0`` where there is none to read."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        return round(
            torch.cuda.max_memory_allocated(self.device) / (1024**2), 1
        )

    def _restart_chunked(self, base: Tensor) -> Tensor:
        """Retry the whole attack with surrogates loaded one chunk at a time.

        Restarting from step zero rather than resuming is deliberate: the
        optimiser's state is the perturbation itself, and resuming after a failed
        forward pass would continue from a partially projected iterate whose
        provenance cannot be reconstructed. Restarting costs time but keeps the
        run reproducible and its gradient correct.

        This handler only applies when the failure is a memory failure *and* an
        ensemble was holding every surrogate resident. Anything else is a real
        error and propagates unchanged, because quietly converting a coding bug
        into a retry would hide it.
        """
        if not self.config.uses_ensemble or self.ensemble is None:
            raise
        if not self.ensemble.holds_resident:
            raise
        logger.warning(
            "engine.oom.fallback surrogates=%s -> chunked max_resident=1",
            self.ensemble.keys,
        )
        self.oom_fallback = True
        self.ensemble.close()
        self.ensemble = SurrogateEnsemble(
            resolve_surrogates(self.config),
            self.device,
            topk=1 if self.config.loss == "cw_top1" else self.config.target_topk,
            max_resident=1,
            resident=False,
        )
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._run_pgd(base)

    def describe(self) -> dict[str, str]:
        """Static engine description for the API and logs."""
        return {
            "engine": self.name,
            "variant": self.variant,
            "target_model": self.target_description,
            "preserves_dimensions": str(self.preserves_dimensions),
            "device": str(self.device),
            "eps_255": f"{self.config.eps_255:.3f}",
            "steps": str(self.config.steps),
            "eot_samples": str(self.config.eot_samples),
            "ensemble": ",".join(self.ensemble.keys) if self.ensemble else "",
        }

    # -- internals ---------------------------------------------------------

    def _to_working_resolution(self, image: RGB) -> Tensor:
        """Resize to the target pipeline's *resize* frame on the attack device.

        The perturbation is optimised on the full frame whose short side equals
        the pipeline's ``resize_short_side`` — not on the centre crop. That is
        the coordinate frame the pattern will be applied and re-read in, so the
        only thing left to undo afterwards is one uniform scale.

        Optimising on the crop instead and stretching the result across the
        source image does not survive contact with the pipeline: the classifier
        re-reads a different region of the stretched pattern, at a different
        resolution, and a perturbation that is this fragile is no perturbation
        at all. The crop is therefore taken inside the graph, per step, by
        :meth:`TargetModel.preprocess` — which, given a frame that is already
        at the target's short side, performs a no-op resize and then the exact
        centre crop.
        """
        target = self.target_config
        tensor = _to_tensor(image, self.device)
        side = target.resize_short_side
        scale = side / float(min(tensor.shape[-2], tensor.shape[-1]))
        new_h = max(target.crop_size, round(tensor.shape[-2] * scale))
        new_w = max(target.crop_size, round(tensor.shape[-1] * scale))
        return F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode=target.interpolation,
            align_corners=False,
            antialias=False,
        )

    def _initial_perturbation(self, base: Tensor) -> Tensor:
        if self.config.init == "uniform":
            generator = torch.Generator(device=base.device)
            generator.manual_seed(self.config.seed)
            noise = torch.rand(
                base.shape,
                generator=generator,
                device=base.device,
                dtype=base.dtype,
            )
            scaled = (noise * 2.0 - 1.0) * self.config.eps
            return self._project(base, scaled)
        return torch.zeros_like(base)

    def _project(self, base: Tensor, perturbation: Tensor) -> Tensor:
        """Clamp to the L-inf ball and re-apply the valid pixel range."""
        perturbation = perturbation.clamp(-self.config.eps, self.config.eps)
        return (base + perturbation).clamp(0.0, 1.0) - base

    def _run_pgd(self, base: Tensor) -> Tensor:
        """Optimise the perturbation in the frame the pipeline will re-read.

        ``base`` is the resize frame, so ``self.model`` performs a no-op resize
        and the real centre crop on every step. Gradients therefore arrive at
        the pattern in the same geometry it is later stored in, and the part of
        the frame outside the crop receives no gradient and stays untouched.

        With ``eot_samples=0`` and no ensemble this is byte-for-byte the Phase 1
        loop: one forward, one backward, one signed step. The EOT and ensemble
        branches change *what* the gradient is of, never the projection or the
        step rule, so a Phase 1 result is reproducible from a Phase 2 checkout.
        """
        perturbation = self._initial_perturbation(base)
        topk = 1 if self.config.loss == "cw_top1" else self.config.target_topk
        if self.ensemble is not None:
            self.ensemble.open()
        try:
            for step in range(self.config.steps):
                loss, gradient = self._loss_and_gradient(base, perturbation, step, topk)
                # Minimising the top logits means stepping *down* the gradient.
                perturbation = perturbation.detach() - self.config.alpha * gradient.sign()
                perturbation = self._project(base, perturbation)
                if step == 0 or (step + 1) % 10 == 0 or step == self.config.steps - 1:
                    logger.debug(
                        "engine.pgd step=%d/%d loss=%.4f linf=%.6f",
                        step + 1,
                        self.config.steps,
                        loss,
                        float(perturbation.abs().max()),
                    )
        finally:
            if self.ensemble is not None:
                self.ensemble.close()
        return perturbation.detach()

    def _loss_and_gradient(
        self, base: Tensor, perturbation: Tensor, step: int, topk: int
    ) -> tuple[float, Tensor]:
        """Loss and gradient of the configured objective for one step.

        Three shapes of objective share this entry point, and the returned
        gradient is the gradient of the objective actually named by the config:

        * no EOT, no ensemble -- the Phase 1 loss against one model;
        * EOT only -- the mean loss over the sampled renderings;
        * ensemble (with or without EOT) -- the weighted sum of per-surrogate
          losses over the sampled renderings.

        EOT averages over *renderings* and the ensemble sums over *models*. They
        compose by simple distributivity, so the total is the gradient of the
        expectation over both, which is the quantity the objective is defined
        to be.
        """
        if self.ensemble is None and not self.eot.enabled:
            perturbation = perturbation.detach().requires_grad_(True)
            logits = self.model(base + perturbation)
            loss = _cw_margin_loss(logits, topk)
            (phase1_gradient,) = torch.autograd.grad(loss, perturbation)
            return float(loss.detach()), phase1_gradient

        batch = (base + perturbation).detach().requires_grad_(True)
        samples = self.eot.sample_from(batch, step)
        loss_value = 0.0
        gradient: Tensor | None = None
        per_model: dict[str, float] = {}
        for _, rendered in samples:
            if self.ensemble is None:
                logits = self.model(rendered)
                contribution = _cw_margin_loss(logits, topk)
                (piece,) = torch.autograd.grad(
                    contribution, batch, retain_graph=True
                )
                loss_value += float(contribution.detach())
            else:
                self.ensemble.start_step()
                loss_value += self.ensemble.backward_into(
                    rendered, batch, per_model=per_model
                )
                piece = self.ensemble.gradient
            gradient = piece if gradient is None else gradient + piece

        if gradient is None:  # pragma: no cover - the sampler always returns one
            raise RuntimeError("no gradient was produced for this step")
        self._record_step(loss_value, per_model, len(samples))
        return loss_value, gradient

    def _record_step(
        self, total_value: float, per_model: dict[str, float], samples: int
    ) -> None:
        """Log how concentrated the ensemble loss is.

        A single surrogate holding most of the objective is the standard way an
        "ensemble attack" quietly stays a single-model attack, and it is not
        visible in the final transfer table without this number.
        """
        if not per_model:
            return
        largest = max(per_model.values())
        self.trace.append(
            {
                "loss": round(total_value, 6),
                "samples": samples,
                "dominant_surrogate_share": round(
                    largest / abs(total_value) if total_value else 0.0, 4
                ),
            }
        )

    def _resample_to_source(
        self, perturbation: Tensor, shape: tuple[int, ...]
    ) -> np.ndarray:
        """Map the working-frame perturbation back onto the full source image.

        The working frame is a uniform ``resize_short_side`` scale of the source,
        so undoing it is a single bilinear resize. No content is invented and no
        region is duplicated, which is what lets the pattern still be
        adversarial when the pipeline scales the image back up.
        """
        if not self.config.preserve_dimensions:
            return _to_array(perturbation).astype(np.float32) / 255.0
        height, width = int(shape[0]), int(shape[1])
        if perturbation.shape[-2:] == (height, width):
            return _to_array(perturbation).astype(np.float32) / 255.0
        upsampled = F.interpolate(
            perturbation,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        moved = upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return moved.astype(np.float32)


class ProtectionOutput:
    """Result bundle: the protected array plus attack metadata and metrics."""

    __slots__ = ("image", "meta", "metrics")

    def __init__(
        self,
        image: RGB,
        meta: ProtectionResultMeta,
        metrics: DifferenceMetrics,
    ) -> None:
        """Bundle a protected image with the diagnostics that explain it."""
        self.image = image
        self.meta = meta
        self.metrics = metrics
