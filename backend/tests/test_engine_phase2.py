"""Engine-level tests for the Phase 2 features: EOT, ensemble and metadata.

These exercise the engine as a whole rather than its parts. The parts already
have their own suites, so what is left to check here is that the wiring is
honest: that the metadata a report is built from actually describes the run
that happened, and that the out-of-memory path retries exactly once and reports
that it did.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from app.protection.config import ProtectionConfig, TargetModelConfig
from app.protection.engine import (
    AdversarialProtectionEngine,
    resolve_surrogates,
)
from app.protection.ensemble import SurrogateEnsemble

CLASSES = 10


class _StubModel:
    """A differentiable stand-in for a classifier.

    Real weights are avoided so these tests never reach the network; only the
    shape of the map matters for everything the engine does here.
    """

    def __init__(self, key: str = "stub", seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.key = key
        self.device = torch.device("cpu")
        self.projection = torch.randn(CLASSES, 3, generator=generator) * 0.25

    @property
    def description(self) -> str:
        return "stub weights for testing"

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return pooled @ self.projection.T


def _image(height: int = 48, width: int = 64) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _patch_ensemble_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every surrogate resolve to a tiny local stub."""
    loaded: list[str] = []

    def loader(
        config: TargetModelConfig, device: torch.device, cache: bool = True
    ) -> _StubModel:
        del device, cache
        loaded.append(config.name)
        return _StubModel(config.name, seed=len(loaded))

    monkeypatch.setattr("app.protection.ensemble.load_target_model", loader)
    return loaded


# -- metadata ---------------------------------------------------------------


def test_meta_describes_the_phase_one_run() -> None:
    """The baseline row must remain identifiable as the baseline."""
    engine = AdversarialProtectionEngine(
        ProtectionConfig(steps=3, device="cpu"), model=_StubModel()
    )
    output = engine.protect_with_meta(_image())
    meta = output.meta
    assert meta.engine == "pgd-linf-resnet18"
    assert meta.variant == "pgd-linf-resnet18"
    assert meta.eot_samples == 0
    assert meta.surrogate_models == ()
    assert meta.oom_fallback is False
    assert meta.surrogate_stats == {}


def test_meta_describes_eot_and_ensemble_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without these fields a Phase 2 row is indistinguishable from Phase 1."""
    _patch_ensemble_loader(monkeypatch)
    config = ProtectionConfig(
        steps=3,
        device="cpu",
        eot_samples=3,
        eot_seed=0,
        ensemble=True,
        surrogates=("resnet18", "mobilenet_v3_small"),
    )
    engine = AdversarialProtectionEngine(config, model=_StubModel())
    output = engine.protect_with_meta(_image())
    meta = output.meta

    assert meta.eot_samples == 3
    assert meta.surrogate_models == ("resnet18", "mobilenet_v3_small")
    assert "ensemble2" in meta.variant
    assert "+eot3" in meta.variant
    assert meta.variant != meta.engine, "Phase 2 must not report as Phase 1"
    assert meta.surrogate_stats["chunked"] is False
    assert set(meta.surrogate_stats["per_model_seconds"]) == {
        "resnet18",
        "mobilenet_v3_small",
    }


def test_meta_records_a_chunked_ensemble() -> None:
    """Chunking is the honest explanation for a slow run, so it must be visible."""
    config = ProtectionConfig(
        steps=2, device="cpu", ensemble=True, surrogates=("resnet18", "swin_t")
    )
    specs = resolve_surrogates(config)
    ensemble = SurrogateEnsemble(specs, torch.device("cpu"), max_resident=1)
    assert ensemble.stats.chunked is True
    assert ensemble.stats.chunks_per_step == 2
    assert ensemble.stats.to_dict()["chunked"] is True


def test_peak_memory_is_zero_on_cpu() -> None:
    """A made-up VRAM figure would be worse than none on a device that has none."""
    engine = AdversarialProtectionEngine(
        ProtectionConfig(steps=2, device="cpu"), model=_StubModel()
    )
    assert engine._peak_memory_mib() == 0.0
    output = engine.protect_with_meta(_image())
    assert output.meta.peak_memory_mib == 0.0


def test_trace_is_reset_between_runs() -> None:
    """A stale trace would attribute one request's loss curve to the next."""
    engine = AdversarialProtectionEngine(
        ProtectionConfig(steps=2, device="cpu"), model=_StubModel()
    )
    engine.protect_with_meta(_image())
    assert engine.trace == [], "single-model runs do not produce ensemble traces"
    engine.protect_with_meta(_image())
    assert engine.trace == []


# -- out-of-memory fallback -------------------------------------------------


def test_oom_falls_back_to_chunked_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry must be reported, not performed silently.

    A run that quietly changed strategy still produced an image, so nothing in
    the output would otherwise reveal that the path taken was not the one
    configured.
    """
    _patch_ensemble_loader(monkeypatch)
    config = ProtectionConfig(
        steps=2,
        device="cpu",
        ensemble=True,
        surrogates=("resnet18", "mobilenet_v3_small", "efficientnet_b0"),
    )
    engine = AdversarialProtectionEngine(config, model=_StubModel())
    assert engine.ensemble is not None and engine.ensemble.holds_resident is True

    real_run = engine._run_pgd
    attempts: list[str] = []

    def flaky(base: torch.Tensor) -> torch.Tensor:
        attempts.append("attempt")
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("simulated: ran out of memory")
        return real_run(base)

    monkeypatch.setattr(engine, "_run_pgd", flaky)
    output = engine.protect_with_meta(_image())

    assert len(attempts) == 2, "the attack must be retried exactly once"
    assert output.meta.oom_fallback is True
    assert engine.ensemble is not None
    assert engine.ensemble.holds_resident is False, "fallback must be chunked"
    assert output.meta.surrogate_stats["chunked"] is True


def test_oom_is_not_swallowed_without_an_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converting an unrelated failure into a retry would hide a real error."""
    engine = AdversarialProtectionEngine(
        ProtectionConfig(steps=2, device="cpu"), model=_StubModel()
    )
    monkeypatch.setattr(
        engine,
        "_run_pgd",
        lambda base: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("simulated")
        ),
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.protect_with_meta(_image())


def test_oom_is_not_swallowed_when_already_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying a strategy that is already the fallback would loop forever."""
    _patch_ensemble_loader(monkeypatch)
    config = ProtectionConfig(
        steps=2,
        device="cpu",
        ensemble=True,
        surrogates=("resnet18", "mobilenet_v3_small"),
        max_resident_surrogates=1,
    )
    engine = AdversarialProtectionEngine(config, model=_StubModel())
    assert engine.ensemble is not None
    assert engine.ensemble.holds_resident is False

    monkeypatch.setattr(
        engine,
        "_run_pgd",
        lambda base: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("simulated")
        ),
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.protect_with_meta(_image())
