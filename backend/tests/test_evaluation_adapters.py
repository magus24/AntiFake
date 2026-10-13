"""Tests for the evaluation adapter contract.

The adapter tests are deliberately offline: they use a stub adapter and a stub
network, so they assert the *contract* rather than any particular model's
accuracy. The one test that needs real weights checks the property that
actually matters for the benchmark's credibility -- that the benchmark reads an
image through the same geometry the protection engine attacked.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
from app.evaluation.adapters_classification import (
    build_classifier,
)
from app.evaluation.base import (
    ClassificationObservation,
    EmbeddingObservation,
    ModelAdapter,
    ModelSpec,
    ModelUnavailableError,
)
from app.evaluation.registry import ModelRegistry, select_device

from tests.helpers import make_detailed, make_photo

CPU = torch.device("cpu")


class StubClassifier(ModelAdapter):
    """A model that reports whatever the injected logits say.

    Lets the contract be tested without weights, and makes the expected
    observation computable by hand.
    """

    family = "classification"

    def __init__(self, device: torch.device, logits: torch.Tensor) -> None:
        """Bind a fixed logit vector to the device."""
        super().__init__(device)
        self.name = "stub_classifier"
        self.description = "stub"
        self._logits = logits
        self.load_calls = 0
        self.release_calls = 0
        self.evaluated: list[tuple[int, int]] = []

    def load(self) -> None:
        if self._loaded:
            return
        self.load_calls += 1
        self._loaded = True

    def release(self) -> None:
        self.release_calls += 1
        self._loaded = False

    def evaluate(self, image):
        self._require_loaded()
        self.evaluated.append((image.shape[0], image.shape[1]))
        return observe_logits(self._logits)


class StubEmbedder(ModelAdapter):
    """Returns a fixed unit embedding, so cosine maths is checkable by hand."""

    family = "embedding"

    def __init__(self, device: torch.device, vector: np.ndarray) -> None:
        """Bind a fixed unit vector to the device."""
        super().__init__(device)
        self.name = "stub_embedder"
        self.description = "stub"
        self._vector = vector
        self._categories = ["a", "b", "c"]

    def load(self) -> None:
        self._loaded = True

    def release(self) -> None:
        self._loaded = False

    def evaluate(self, image):
        return EmbeddingObservation(
            embedding=self._vector,
            text_similarities={"a": 0.9, "b": 0.5, "c": 0.2},
            text_ranking=["a", "b", "c"],
        )


def observe_logits(logits: torch.Tensor, categories=("alpha", "beta", "gamma")):
    """Build the observation the classification adapter would produce."""
    probabilities = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probabilities, min(3, probabilities.numel()))
    top1 = int(indices[0])
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())
    return ClassificationObservation(
        top1_index=top1,
        top1_label=categories[top1],
        top1_probability=float(values[0]),
        top1_logit=float(torch.max(logits)),
        top5=[
            (int(i), float(v))
            for v, i in zip(values.tolist(), indices.tolist(), strict=True)
        ],
        entropy=entropy,
    )


# -- contract ----------------------------------------------------------------


def test_evaluate_before_load_is_an_error():
    adapter = StubClassifier(CPU, torch.zeros(1, 3))
    with pytest.raises(RuntimeError, match="before load"):
        adapter.evaluate(make_photo(8, 8))


def test_load_is_idempotent():
    adapter = StubClassifier(CPU, torch.zeros(1, 3))
    adapter.load()
    adapter.load()
    assert adapter.load_calls == 1


def test_release_can_be_called_twice():
    adapter = StubClassifier(CPU, torch.zeros(1, 3))
    adapter.load()
    adapter.release()
    adapter.release()
    assert adapter.release_calls == 2
    assert not adapter.is_loaded


def test_classification_observation_is_json_ready():
    observation = observe_logits(torch.tensor([[2.0, 1.0, 0.5]]))
    payload = observation.to_dict()
    assert set(payload) == {
        "top1_index",
        "top1_label",
        "top1_probability",
        "top1_logit",
        "top5",
        "entropy",
    }
    assert payload["top1_label"] == "alpha"
    assert 0.0 <= payload["top1_probability"] <= 1.0


def test_embedding_observation_omits_vector_by_default():
    observation = EmbeddingObservation(
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        text_similarities={"a": 1.0},
        text_ranking=["a"],
    )
    assert "embedding" not in observation.to_dict()
    assert observation.to_dict(include_embedding=True)["embedding"] == [1.0, 0.0]
    assert observation.dimension == 2


def test_embedding_text_similarities_are_sorted_for_stable_json():
    observation = EmbeddingObservation(
        embedding=np.zeros(4, dtype=np.float32),
        text_similarities={"z": 0.1, "a": 0.9, "m": 0.5},
        text_ranking=["a", "m", "z"],
    )
    keys = list(observation.to_dict()["text_similarities"])
    assert keys == ["a", "m", "z"]


# -- registry ----------------------------------------------------------------


class _FakeSpecAdapter(ModelAdapter):
    """Built by the registry through monkeypatching; records load/release."""

    #: Every instance the registry has built, so tests can inspect the lifecycle.
    instances: ClassVar[list[_FakeSpecAdapter]] = []

    def __init__(self, device, key: str = "fake", **kwargs: object) -> None:
        """Record the instance so release/load counts can be asserted."""
        super().__init__(device)
        self.name = key
        self.description = f"fake:{key}"
        self.family = kwargs.get("family", "classification")
        self.load_calls = 0
        self.release_calls = 0
        self.evaluated: list[tuple[int, int]] = []
        _FakeSpecAdapter.instances.append(self)

    def load(self):
        if self._loaded:
            return
        self.load_calls += 1
        self._loaded = True

    def release(self):
        self.release_calls += 1
        self._loaded = False

    def evaluate(self, image):
        self._require_loaded()
        self.evaluated.append((image.shape[0], image.shape[1]))
        return observe_logits(torch.tensor([[1.0, 0.0]]))


def test_registry_releases_every_model(monkeypatch):
    _FakeSpecAdapter.instances.clear()
    spec = ModelSpec(
        key="fake", family="classification", module="torchvision", factory="fake"
    )
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: _FakeSpecAdapter(device, key=spec.key),
    )
    registry = ModelRegistry(CPU, (spec,))
    results = registry.evaluate_all({"img": make_photo(8, 8)})

    assert results["fake"].status == "ok"
    assert all(not a.is_loaded for a in _FakeSpecAdapter.instances)
    assert all(a.release_calls == 1 for a in _FakeSpecAdapter.instances)


def test_registry_reports_skip_with_reason(monkeypatch):
    def _raise(spec, device, prompts=None):
        raise ModelUnavailableError(spec.key, "weights not cached")

    spec = ModelSpec(
        key="fake", family="classification", module="torchvision", factory="fake"
    )
    monkeypatch.setattr("app.evaluation.registry.build_adapter", _raise)
    results = ModelRegistry(CPU, (spec,)).evaluate_all({"img": make_photo(8, 8)})

    assert results["fake"].status == "skipped"
    assert results["fake"].reason == "weights not cached"
    assert results["fake"].observations == {}


def test_registry_distinguishes_failure_from_skip(monkeypatch):
    """A real defect must not be reported as a missing dependency.

    This is the distinction that stops the benchmark from quietly claiming
    coverage it does not have.
    """

    class _Broken(_FakeSpecAdapter):
        def load(self):
            raise ZeroDivisionError("bug in adapter")

    spec = ModelSpec(
        key="fake", family="classification", module="torchvision", factory="fake"
    )
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: _Broken(device),
    )
    results = ModelRegistry(CPU, (spec,)).evaluate_all({"img": make_photo(8, 8)})

    assert results["fake"].status == "failed"
    assert "ZeroDivisionError" in results["fake"].reason


def test_registry_passes_every_image_to_one_resident_model(monkeypatch):
    _FakeSpecAdapter.instances.clear()
    spec = ModelSpec(
        key="fake", family="classification", module="torchvision", factory="fake"
    )
    monkeypatch.setattr(
        "app.evaluation.registry.build_adapter",
        lambda spec, device, prompts=None: _FakeSpecAdapter(device, key=spec.key),
    )
    images = {"a": make_photo(8, 8), "b": make_detailed(16, 24)}
    ModelRegistry(CPU, (spec,)).evaluate_all(images)

    # `make_detailed(width, height)`, so the second array is 24 rows x 16 cols.
    adapter = _FakeSpecAdapter.instances[0]
    assert adapter.evaluated == [(8, 8), (24, 16)]


def test_select_device_cpu_is_always_honoured():
    device, reason = select_device("cpu")
    assert device.type == "cpu"
    assert reason == "requested:cpu"


def test_select_device_returns_a_stated_reason():
    _, reason = select_device("auto")
    assert "auto" in reason or "requested" in reason


# -- preprocessing consistency ----------------------------------------------


@pytest.mark.parametrize(
    ("height", "width"),
    [(334, 414), (400, 300), (512, 512), (200, 200), (700, 260), (64, 900)],
)
def test_benchmark_preprocess_matches_the_engine_geometry(height, width):
    """The benchmark must read images through the engine's own transform.

    If these two ever diverge, a confidence drop could no longer be attributed
    to the perturbation, and the entire benchmark becomes uninterpretable.
    This is the single most important consistency check in the package.
    """
    image = make_detailed(height, width)
    adapter = build_classifier(CPU, "resnet18")
    # `load()` is skipped: preprocess only needs the normalisation buffers.
    adapter._mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    adapter._std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    expected = _engine_preprocess(tensor)
    actual = adapter.preprocess(image)
    assert actual.shape == expected.shape == (1, 3, 224, 224)
    assert torch.allclose(actual, expected, atol=1e-6)


def _engine_preprocess(tensor: torch.Tensor) -> torch.Tensor:
    """Independent reimplementation of TargetModel.preprocess, for comparison."""
    import torch.nn.functional as F

    height, width = tensor.shape[-2], tensor.shape[-1]
    scale = 256 / float(min(height, width))
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    resized = F.interpolate(
        tensor, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=False
    )
    top = max(0, (new_h - 224) // 2)
    left = max(0, (new_w - 224) // 2)
    cropped = resized[:, :, top : top + 224, left : left + 224]
    if cropped.shape[-2:] != (224, 224):
        cropped = F.interpolate(
            cropped, size=(224, 224), mode="bilinear", align_corners=False
        )
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (cropped - mean) / std


def test_classifier_preprocess_always_yields_224():
    for height, width in ((334, 414), (100, 700), (900, 64)):
        adapter = build_classifier(CPU, "resnet18")
        adapter._mean = torch.zeros(1, 3, 1, 1)
        adapter._std = torch.ones(1, 3, 1, 1)
        assert adapter.preprocess(make_detailed(height, width)).shape == (1, 3, 224, 224)


def test_build_classifier_uses_declared_crop_size():
    assert build_classifier(CPU, "resnet18").crop_size == 224
    assert build_classifier(CPU, "vit_b_16").crop_size == 224
