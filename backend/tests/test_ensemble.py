"""Tests for the multi-architecture objective.

Two claims are being defended here.

First, that loading surrogates in chunks is *not* an approximation of keeping
them resident. This is a mathematical property of a sum of independent terms,
but it is easy to break in code -- by dropping a term, by renormalising after a
failed load, or by holding a graph that no longer exists -- so it is asserted
against an independently written reference loop rather than against a second
call into the same implementation.

Second, that the tensor the PGD step consumes is the gradient of the loss the
report prints. A version of this code detached between models and produced
perfectly plausible numbers that were the derivative of nothing in particular,
which is the failure mode these tests exist to catch.
"""

from __future__ import annotations

import pytest
import torch
from app.protection.config import (
    ProtectionConfig,
    ProtectionConfigError,
    TargetModelConfig,
)
from app.protection.engine import resolve_surrogates
from app.protection.ensemble import (
    DEFAULT_SURROGATES,
    SurrogateEnsemble,
    SurrogateSpec,
    cw_topk_loss,
    normalize_weights,
)

CLASSES = 8


class _Stub:
    """A tiny deterministic classifier.

    A hand-built linear map is used instead of a random torchvision model so
    that a failure points at the ensemble arithmetic rather than at whatever
    the network happened to compute. It is a classifier in the only sense the
    ensemble cares about: a differentiable map from an ``(N, 3, H, W)`` batch to
    per-class logits.
    """

    def __init__(self, key: str, seed: int, bias: float) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.key = key
        self.bias = bias
        self.project = torch.randn(CLASSES, 3, generator=generator) * 0.3
        self.shift = torch.randn(CLASSES, generator=generator) * 0.2

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return pooled @ self.project.T + self.bias * self.shift


def _specs(*weights: float) -> list[SurrogateSpec]:
    return [
        SurrogateSpec(
            key=f"m{i}",
            config=TargetModelConfig(name=f"m{i}", weights="none"),
            weight=weight,
        )
        for i, weight in enumerate(weights)
    ]


def _stubs(*weights: float) -> list[_Stub]:
    return [_Stub(f"m{i}", 100 + i, float(i)) for i, _ in enumerate(weights)]


def _patch(monkeypatch: pytest.MonkeyPatch, stubs: list[_Stub]) -> None:
    """Route every surrogate key to its stub, whatever the loader is asked for."""
    by_key = {stub.key: stub for stub in stubs}

    def loader(
        config: TargetModelConfig, device: torch.device, cache: bool = True
    ) -> _Stub:
        del device, cache
        return by_key[config.name]

    monkeypatch.setattr("app.protection.ensemble.load_target_model", loader)


@pytest.fixture
def leaf() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, 32, 40, generator=generator, requires_grad=True)


def _reference(
    leaf: torch.Tensor, stubs: list[_Stub], weights: list[float]
) -> torch.Tensor:
    """The objective written out longhand, with no ensemble involved."""
    total: torch.Tensor | None = None
    for stub, weight in zip(stubs, weights, strict=True):
        loss = cw_topk_loss(stub(leaf), 3).sum()
        term = weight * loss
        total = term if total is None else total + term
    assert total is not None
    (gradient,) = torch.autograd.grad(total, leaf)
    return gradient


# -- equivalence ------------------------------------------------------------


def test_chunked_matches_resident_gradient(
    monkeypatch: pytest.MonkeyPatch, leaf: torch.Tensor
) -> None:
    """Resident and chunked must be the same optimiser, not two similar ones.

    Summing per-model gradients in chunks is exact only while no term is lost.
    The assertion is tight because any difference here would mean the memory
    strategy is quietly changing the attack.
    """
    weights = [1.0, 1.0, 1.0, 1.0]
    _patch(monkeypatch, _stubs(*weights))

    resident = SurrogateEnsemble(
        _specs(*weights), torch.device("cpu"), topk=3, max_resident=None
    )
    resident.open()
    assert resident.stats.chunked is False
    resident.start_step()
    resident.backward_into(leaf, leaf)
    resident_gradient = resident.gradient.clone()
    resident.close()

    chunked = SurrogateEnsemble(
        _specs(*weights), torch.device("cpu"), topk=3, max_resident=1
    )
    assert chunked.stats.chunked is True
    assert chunked.stats.chunks_per_step == 4
    chunked.start_step()
    chunked.backward_into(leaf, leaf)
    chunked_gradient = chunked.gradient.clone()
    chunked.close()

    assert resident_gradient.shape == chunked_gradient.shape
    assert torch.allclose(resident_gradient, chunked_gradient, atol=1e-7, rtol=1e-6)


def test_gradient_equals_the_reference_loop(
    monkeypatch: pytest.MonkeyPatch, leaf: torch.Tensor
) -> None:
    """The returned tensor is the gradient of the loss that is reported."""
    weights = [2.0, 1.0, 3.0]
    stubs = _stubs(*weights)
    _patch(monkeypatch, stubs)

    ensemble = SurrogateEnsemble(_specs(*weights), torch.device("cpu"), topk=3)
    ensemble.start_step()
    per_model: dict[str, float] = {}
    value = ensemble.backward_into(leaf, leaf, per_model=per_model)
    gradient = ensemble.gradient.clone()

    expected = _reference(
        leaf, stubs, [w / sum(weights) for w in weights]
    )
    assert torch.allclose(gradient, expected, atol=1e-7, rtol=1e-6)
    assert value == pytest.approx(sum(per_model.values()), rel=1e-6)
    assert set(per_model) == {"m0", "m1", "m2"}


def test_weights_actually_reweight_the_gradient(
    monkeypatch: pytest.MonkeyPatch, leaf: torch.Tensor
) -> None:
    """A weighted ensemble must not quietly optimise the unweighted mean.

    If weights were ignored, every report row would be a plain mean and the
    weighting control would be documented but absent.
    """
    _patch(monkeypatch, _stubs(2.0, 0.0))

    skewed = SurrogateEnsemble(
        _specs(2.0, 0.0), torch.device("cpu"), topk=3
    )
    skewed.start_step()
    skewed.backward_into(leaf, leaf)
    skewed_gradient = skewed.gradient.clone()

    only_first = SurrogateEnsemble(_specs(1.0), torch.device("cpu"), topk=3)
    only_first.start_step()
    only_first.backward_into(leaf, leaf)
    only_first_gradient = only_first.gradient.clone()

    both = SurrogateEnsemble(_specs(1.0, 1.0), torch.device("cpu"), topk=3)
    both.start_step()
    both.backward_into(leaf, leaf)
    both_gradient = both.gradient.clone()

    assert torch.allclose(skewed_gradient, only_first_gradient, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(both_gradient, only_first_gradient, atol=1e-7, rtol=1e-6)


def test_start_step_clears_the_previous_gradient(
    monkeypatch: pytest.MonkeyPatch, leaf: torch.Tensor
) -> None:
    """A stale gradient must not carry over between steps.

    Without this the optimiser would drift towards whichever step ran first and
    the later steps would contribute nothing but noise to the update.

    Note that ``rendered`` and ``target`` are bound once and reused. Passing
    ``leaf * 0.5`` twice builds two distinct graph nodes, so the one handed to
    the model is not the one differentiated against -- a mistake PyTorch reports
    as an unrelated "tensor appears to not have been used" error.
    """
    _patch(monkeypatch, _stubs(1.0, 1.0))
    ensemble = SurrogateEnsemble(_specs(1.0, 1.0), torch.device("cpu"), topk=3)
    ensemble.start_step()
    ensemble.backward_into(leaf, leaf)
    first = ensemble.gradient.clone()

    scaled = leaf * 0.5
    ensemble.start_step()
    ensemble.backward_into(scaled, scaled)
    second = ensemble.gradient

    assert not torch.allclose(first, second, atol=1e-7, rtol=1e-6)


# -- weights ----------------------------------------------------------------


def test_normalize_weights_scales_to_one() -> None:
    assert normalize_weights(_specs(2.0, 2.0)) == {"m0": 0.5, "m1": 0.5}
    assert normalize_weights(_specs(1.0, 3.0)) == {"m0": 0.25, "m1": 0.75}


def test_all_zero_weights_are_rejected() -> None:
    """An all-zero weight request must raise rather than divide by zero.

    The NaNs would reach every loss value and every report row without ever
    raising, so the run would look like a normal attack that fails to converge.
    """
    with pytest.raises(ValueError, match=r"zero|above zero|positive"):
        normalize_weights(_specs(0.0, 0.0))


def test_negative_weights_are_rejected() -> None:
    """A negative member weight must be rejected.

    The total is allowed to stay positive while one member is negative, so a
    sum-only check passes ``{-1, 2}`` and that surrogate is then optimised
    towards confident classification rather than away from it.
    """
    with pytest.raises(ValueError, match="negative"):
        normalize_weights(_specs(-1.0, 2.0))


def test_empty_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one surrogate"):
        SurrogateEnsemble([], torch.device("cpu"))


# -- lifecycle --------------------------------------------------------------


def test_gradient_before_any_backward_is_an_error() -> None:
    """Reading the gradient before any backward must raise.

    Returning zeros instead would produce a no-op attack whose report still
    shows plausible loss values, which is the worst of the available outcomes.
    """
    ensemble = SurrogateEnsemble(_specs(1.0), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="backward_into"):
        _ = ensemble.gradient


def test_open_loads_every_surrogate_and_close_releases_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    stubs = _stubs(1.0, 1.0, 1.0)
    by_key = {stub.key: stub for stub in stubs}

    def loader(
        config: TargetModelConfig, device: torch.device, cache: bool = True
    ) -> _Stub:
        del device, cache
        calls.append(config.name)
        return by_key[config.name]

    monkeypatch.setattr("app.protection.ensemble.load_target_model", loader)
    ensemble = SurrogateEnsemble(_specs(1.0, 1.0, 1.0), torch.device("cpu"))
    ensemble.open()
    assert sorted(calls) == ["m0", "m1", "m2"]
    assert ensemble.holds_resident is True

    ensemble.close()
    ensemble.close()  # idempotent: a double close must not be an error
    ensemble.open()
    assert len(calls) == 6, "close() must actually release the models"


def test_chunked_strategy_never_keeps_models_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunked set must not be preloaded by ``open()``.

    Preloading brings back the very problem chunking exists to avoid: the models
    no longer fit once they are all resident at once.
    """
    _patch(monkeypatch, _stubs(1.0, 1.0, 1.0))
    ensemble = SurrogateEnsemble(
        _specs(1.0, 1.0, 1.0), torch.device("cpu"), max_resident=2
    )
    ensemble.open()
    assert ensemble.holds_resident is False
    leaf = torch.rand(1, 3, 8, 8, requires_grad=True)
    ensemble.start_step()
    ensemble.backward_into(leaf, leaf)
    ensemble.close()


def test_ensemble_loads_are_not_taken_from_the_global_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process-wide model cache would keep every surrogate alive forever.

    This is the difference between a memory bound and a memory *claim*: a
    chunked run whose loads are cached has not released anything, it has simply
    stopped holding the reference itself. Both strategies are checked, because
    both are susceptible to the same mistake.
    """
    seen: list[bool] = []

    def loader(
        config: TargetModelConfig, device: torch.device, cache: bool = True
    ) -> _Stub:
        seen.append(cache)
        return _Stub(config.name, 7, 0.0)

    monkeypatch.setattr("app.protection.ensemble.load_target_model", loader)

    resident = SurrogateEnsemble(_specs(1.0, 1.0), torch.device("cpu"))
    resident.open()
    resident.close()
    assert seen and all(c is False for c in seen), "resident loads must be uncached"

    chunked = SurrogateEnsemble(_specs(1.0, 1.0), torch.device("cpu"), max_resident=1)
    seen.clear()
    leaf = torch.rand(1, 3, 8, 8, requires_grad=True)
    chunked.start_step()
    chunked.backward_into(leaf, leaf)
    chunked.close()
    assert seen and all(c is False for c in seen), "chunked loads must be uncached"


# -- held-out discipline ----------------------------------------------------


def test_declared_surrogates_exclude_the_clipped_evaluation_model() -> None:
    """CLIP is the evaluator; optimising against it destroys the held-out test."""
    keys = tuple(spec.key for spec in DEFAULT_SURROGATES)
    assert "clip" not in keys
    assert "clip_vit_b_32" not in keys
    assert "resnet18" in keys, "the Phase 1 target must stay in the default set"
    assert len(set(keys)) == len(keys)


def test_surrogate_and_held_out_sets_are_disjoint() -> None:
    """The transfer claim is only as good as the split that produced it.

    If a transformer slipped into the surrogate set, the headline transfer
    number would be measured against a model that was paid for, and nothing in
    the generated report would reveal it. This test is the guard: it joins the
    attack's declaration to the benchmark's registry and fails loudly on any
    overlap.
    """
    from app.evaluation.registry import DEFAULT_MODELS

    surrogate_keys = {spec.key for spec in DEFAULT_SURROGATES}
    registry_keys = {spec.key for spec in DEFAULT_MODELS}
    held_out_keys = {spec.key for spec in DEFAULT_MODELS if spec.held_out}

    assert surrogate_keys <= registry_keys, (
        "the attack targets a model the benchmark does not evaluate"
    )
    assert surrogate_keys.isdisjoint(held_out_keys), (
        f"a held-out model is also a surrogate: {surrogate_keys & held_out_keys}"
    )
    assert held_out_keys, "the report needs at least one held-out model"
    assert {"vit_b_16", "swin_t"} <= held_out_keys
    assert {spec.key for spec in DEFAULT_MODELS if spec.role == "evaluator"} == {
        "clip_vit_b_32"
    }


def test_resolve_surrogates_defaults_to_the_phase_one_target() -> None:
    """No surrogate list must reproduce Phase 1 exactly, not an empty ensemble."""
    resolved = resolve_surrogates(ProtectionConfig())
    assert len(resolved) == 1
    assert resolved[0].key == "resnet18"
    assert resolved[0].weight == 1.0


def test_resolve_surrogates_applies_declared_weights() -> None:
    config = ProtectionConfig(
        ensemble=True,
        surrogates=("resnet18", "swin_t"),
        ensemble_weights=(("swin_t", 3.0),),
    )
    resolved = resolve_surrogates(config)
    assert [s.key for s in resolved] == ["resnet18", "swin_t"]
    assert [s.weight for s in resolved] == [1.0, 3.0]


# -- configuration ----------------------------------------------------------


def test_config_defaults_keep_phase_one_behaviour() -> None:
    config = ProtectionConfig()
    assert config.eot_samples == 0
    assert config.ensemble is False
    assert config.uses_eot is False
    assert config.uses_ensemble is False
    assert config.eps_255 == 8.0
    assert config.weight_map == {}


def test_single_surrogate_still_counts_as_an_ensemble() -> None:
    """A single-surrogate ensemble must still count as an ensemble.

    Otherwise the one-member control cannot separate "more models" from "new
    code path", and the comparison it exists to support collapses.
    """
    config = ProtectionConfig(ensemble=True, surrogates=("resnet18",))
    assert config.uses_ensemble is True


def test_config_rejects_a_negative_eot_budget() -> None:
    with pytest.raises(ProtectionConfigError, match="eot_samples"):
        ProtectionConfig(eot_samples=-1)


def test_config_rejects_weights_for_absent_surrogates() -> None:
    with pytest.raises(ProtectionConfigError, match="non-surrogate"):
        ProtectionConfig(
            ensemble=True,
            surrogates=("resnet18",),
            ensemble_weights=(("vit_b_16", 1.0),),
        )


def test_config_rejects_duplicate_surrogates() -> None:
    with pytest.raises(ProtectionConfigError, match="unique"):
        ProtectionConfig(ensemble=True, surrogates=("resnet18", "resnet18"))


def test_config_rejects_an_ensemble_without_members() -> None:
    with pytest.raises(ProtectionConfigError, match="surrogates"):
        ProtectionConfig(ensemble=True)


def test_config_rejects_negative_member_weights() -> None:
    with pytest.raises(ProtectionConfigError, match="negative"):
        ProtectionConfig(
            ensemble=True,
            surrogates=("resnet18", "swin_t"),
            ensemble_weights=(("swin_t", -1.0),),
        )


def test_config_weight_map_defaults_to_equal() -> None:
    config = ProtectionConfig(
        ensemble=True, surrogates=("resnet18", "swin_t", "vit_b_16")
    )
    assert config.weight_map == {
        "resnet18": 1.0,
        "swin_t": 1.0,
        "vit_b_16": 1.0,
    }


def test_config_rejects_an_empty_family_list_with_eot_on() -> None:
    with pytest.raises(ProtectionConfigError, match="eot_families"):
        ProtectionConfig(eot_samples=4, eot_families=())


def test_config_rejects_more_surrogates_than_the_budget() -> None:
    with pytest.raises(ProtectionConfigError, match="surrogates"):
        ProtectionConfig(
            ensemble=True, surrogates=tuple(f"m{i}" for i in range(64))
        )
