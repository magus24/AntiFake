"""Tests for the face, detection and vision-language probes.

The weights themselves are not exercised: MTCNN and YOLO fetch checkpoints,
which would make the suite depend on a download. What is under test is the part
that does not depend on the weights -- box clamping, the confidence gate, label
mapping, the status a missing backend produces -- driven through fake networks
so the logic is checked exactly once, deterministically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from app.evaluation.adapters_detection import DetectionAdapter
from app.evaluation.adapters_face import FaceAdapter
from app.evaluation.adapters_vlm import VlmAdapter, VlmBackend
from app.evaluation.base import (
    DetectionObservation,
    FaceObservation,
    ModelUnavailableError,
    TextObservation,
)

CPU = torch.device("cpu")


# -- fakes ----------------------------------------------------------------


class _FakeDetector:
    """Stands in for MTCNN's ``detect``."""

    def __init__(
        self,
        boxes: np.ndarray | None,
        probabilities: np.ndarray | None,
    ) -> None:
        self._boxes = boxes
        self._probabilities = probabilities

    def detect(self, image, *, landmarks: bool = False):
        return self._boxes, self._probabilities


class _FakeEmbedder:
    """Returns a constant descriptor so normalisation is easy to assert."""

    def __init__(self, values: tuple[float, ...] = (3.0, 4.0)) -> None:
        self._values = values

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tuple(tensor.shape) == (1, 3, 160, 160), "crop must be 160x160"
        return torch.tensor([list(self._values)], dtype=torch.float32)


def _face_adapter(
    boxes: np.ndarray | None,
    probabilities: np.ndarray | None,
    **kwargs: Any,
) -> FaceAdapter:
    adapter = FaceAdapter(CPU, **kwargs)
    adapter._loaded = True
    adapter._detector = _FakeDetector(boxes, probabilities)
    adapter._net = _FakeEmbedder()
    return adapter


# -- face -----------------------------------------------------------------


def test_face_adapter_reports_a_missing_face_with_a_reason():
    """An empty detector result is ``skipped``, not an exception."""
    adapter = _face_adapter(None, None)
    observation = adapter.evaluate(np.zeros((80, 100, 3), dtype=np.uint8))
    assert isinstance(observation, FaceObservation)
    assert observation.detected is False
    assert observation.box is None
    assert observation.embedding is None
    assert "no face" in observation.reason


def test_face_adapter_rejects_a_below_threshold_confidence():
    """The gate runs on detector confidence, before any descriptor is computed."""
    adapter = _face_adapter(
        np.array([[10.0, 10.0, 60.0, 70.0]]),
        np.array([0.5]),
        min_confidence=0.7,
    )
    observation = adapter.evaluate(np.zeros((80, 100, 3), dtype=np.uint8))
    assert observation.detected is False
    assert observation.confidence == pytest.approx(0.5)
    assert "0.700" in observation.reason


def test_face_adapter_clamps_a_box_that_runs_off_the_image():
    """A box from the detector may exceed the frame; the crop may not."""
    adapter = _face_adapter(
        np.array([[-10.0, -5.0, 150.0, 200.0]]),
        np.array([0.99]),
    )
    observation = adapter.evaluate(np.zeros((80, 100, 3), dtype=np.uint8))
    assert observation.detected is True
    assert observation.box == (0, 0, 100, 80)
    assert observation.confidence == pytest.approx(0.99)
    assert observation.embedding is not None
    assert observation.dimension == 2


def test_face_adapter_l2_normalises_the_descriptor():
    """A raw (3, 4) must arrive as a unit vector, so cosine is a dot product."""
    adapter = _face_adapter(
        np.array([[10.0, 10.0, 60.0, 70.0]]),
        np.array([0.99]),
    )
    observation = adapter.evaluate(np.zeros((80, 100, 3), dtype=np.uint8))
    assert observation.embedding is not None
    assert float(np.linalg.norm(observation.embedding)) == pytest.approx(1.0)
    assert observation.embedding[0] == pytest.approx(0.6, abs=1e-6)


def test_face_adapter_drops_a_box_that_collapses_to_nothing():
    adapter = _face_adapter(
        np.array([[50.0, 50.0, 40.0, 40.0]]),
        np.array([0.99]),
    )
    observation = adapter.evaluate(np.zeros((80, 100, 3), dtype=np.uint8))
    assert observation.detected is False
    assert "empty box" in observation.reason


def test_face_adapter_requires_load_before_evaluate():
    adapter = FaceAdapter(CPU)
    with pytest.raises(RuntimeError, match="before load"):
        adapter.evaluate(np.zeros((8, 8, 3), dtype=np.uint8))


def test_face_release_is_safe_to_call_twice():
    adapter = _face_adapter(None, None)
    adapter.release()
    adapter.release()
    assert adapter.is_loaded is False


# -- detection ------------------------------------------------------------


class _FakeBox:
    def __init__(self, cls: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = [cls]
        self.conf = [confidence]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox], names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class _FakeDetectorModel:
    def __init__(self, result: _FakeResult) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    def predict(self, **kwargs: Any):
        self.kwargs = kwargs
        return [self.result]


def _detection_adapter(result: _FakeResult, **kwargs: Any) -> DetectionAdapter:
    adapter = DetectionAdapter(CPU, **kwargs)
    adapter._loaded = True
    adapter._model = _FakeDetectorModel(result)
    return adapter


def test_detection_adapter_maps_labels_and_sorts_by_confidence():
    result = _FakeResult(
        boxes=[
            _FakeBox(0, 0.4, [0.0, 0.0, 10.0, 10.0]),
            _FakeBox(2, 0.9, [1.0, 1.0, 11.0, 11.0]),
        ],
        names={0: "person", 2: "tie"},
    )
    adapter = _detection_adapter(result)
    observation = adapter.evaluate(np.zeros((32, 32, 3), dtype=np.uint8))
    assert isinstance(observation, DetectionObservation)
    assert observation.labels == ["tie", "person"]
    assert [d.confidence for d in observation.detections] == [0.9, 0.4]
    assert observation.detections[0].box == (1, 1, 11, 11)


def test_detection_adapter_forwards_the_confidence_floor():
    result = _FakeResult(boxes=[], names={0: "person"})
    adapter = _detection_adapter(result, confidence=0.5)
    adapter.evaluate(np.zeros((32, 32, 3), dtype=np.uint8))
    assert adapter._model.kwargs["conf"] == 0.5
    assert adapter._model.kwargs["verbose"] is False


def test_detection_adapter_reports_nothing_when_the_model_finds_nothing():
    adapter = _detection_adapter(_FakeResult(boxes=[], names={}))
    observation = adapter.evaluate(np.zeros((32, 32, 3), dtype=np.uint8))
    assert observation.detections == []
    assert observation.labels == []


# -- vision-language ------------------------------------------------------


class _FakeVlm(VlmBackend):
    name = "fake-vlm"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.loaded = False
        self.released = False

    def load(self) -> None:
        self.loaded = True

    def generate(self, image, prompt: str) -> str:
        self.calls.append(prompt)
        return f"  {prompt.upper()}  "

    def release(self) -> None:
        self.released = True


def test_vlm_without_a_backend_is_unavailable_not_a_failure():
    """No backend means the branch cannot run -- it must not be reported as broken."""
    adapter = VlmAdapter(CPU, backend=None)
    with pytest.raises(ModelUnavailableError, match="no VLM backend configured"):
        adapter.load()


def test_vlm_row_is_recorded_as_skipped_with_a_reason():
    from app.evaluation.registry import PROBE_MODELS, ModelRegistry

    spec = next(s for s in PROBE_MODELS if s.family == "text")
    registry = ModelRegistry(CPU, specs=[spec])
    results = registry.evaluate_all({"a": np.zeros((8, 8, 3), dtype=np.uint8)})

    result = results["vlm"]
    assert result.status == "skipped"
    assert "no VLM backend" in result.reason
    assert result.observations == {}
    assert result.role == "held_out"


def test_vlm_evaluate_asks_every_prompt_and_strips_the_answer():
    backend = _FakeVlm()
    adapter = VlmAdapter(CPU, backend=backend)
    adapter.load()

    observation = adapter.evaluate(np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(observation, TextObservation)
    assert set(observation.answers) == set(adapter.prompts)
    assert all(answer == answer.strip() for answer in observation.answers.values())
    assert observation.model == "fake-vlm"
    assert len(backend.calls) == len(adapter.prompts)
    assert adapter.is_loaded is True

    adapter.release()
    assert backend.released is True
    assert adapter.is_loaded is False


def test_vlm_survives_a_backend_that_raises_on_one_prompt():
    """One broken question must not cost the other answers on the same row."""

    class _HalfBroken(_FakeVlm):
        def __init__(self) -> None:
            super().__init__()
            self.failing = ""

        def generate(self, image, prompt: str) -> str:
            if prompt == self.failing:
                raise ValueError("boom")
            return "ok"

    backend = _HalfBroken()
    adapter = VlmAdapter(CPU, backend=backend)
    backend.failing = adapter.prompts[0]
    adapter.load()

    observation = adapter.evaluate(np.zeros((8, 8, 3), dtype=np.uint8))
    first = observation.answers[adapter.prompts[0]]
    assert first.startswith("<error: ValueError")
    others = {
        k: v for k, v in observation.answers.items() if k != adapter.prompts[0]
    }
    assert others and all(not v.startswith("<error") for v in others.values())


# -- declaration ----------------------------------------------------------


def test_extended_prompts_keep_the_phase_1_set_untouched():
    from app.evaluation.adapters_embedding import DEFAULT_PROMPTS, EXTENDED_PROMPTS

    assert EXTENDED_PROMPTS[: len(DEFAULT_PROMPTS)] == DEFAULT_PROMPTS
    assert len(EXTENDED_PROMPTS) > len(DEFAULT_PROMPTS)
    assert len(set(EXTENDED_PROMPTS)) == len(EXTENDED_PROMPTS)


def test_every_probe_declares_its_family():
    from app.evaluation.registry import PROBE_MODELS, build_adapter

    families = {spec.family for spec in PROBE_MODELS}
    assert families == {"face", "detection", "text"}
    for spec in PROBE_MODELS:
        try:
            adapter = build_adapter(spec, CPU)
        except ModelUnavailableError:
            # The dependency is optional; its absence must not fail the suite,
            # and the registry already records it as skipped for the same reason.
            pytest.skip(f"{spec.key} dependency is not installed")
        assert adapter.family == spec.family


def test_probes_are_held_out_and_absent_from_the_attack_objective():
    """A probe the optimiser can see is not evidence, and must not be one."""
    from app.evaluation.registry import DEFAULT_MODELS, PROBE_MODELS

    assert all(spec.role == "held_out" for spec in PROBE_MODELS)
    assert {spec.key for spec in PROBE_MODELS}.isdisjoint(
        {spec.key for spec in DEFAULT_MODELS}
    )
    assert {spec.key for spec in DEFAULT_MODELS if spec.role == "held_out"} == {
        "vit_b_16",
        "swin_t",
    }


def test_probe_results_serialise_with_a_role_and_their_own_shape():
    from app.evaluation.base import Detection
    from app.evaluation.registry import PROBE_MODELS, ModelResult

    face_spec = next(s for s in PROBE_MODELS if s.family == "face")
    result = ModelResult(
        key=face_spec.key,
        family=face_spec.family,
        status="ok",
        description=face_spec.notes,
        role=face_spec.role,
        observations={
            "a": FaceObservation(
                detected=True,
                confidence=0.9,
                box=(1, 2, 3, 4),
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        },
    )
    payload = result.to_dict()
    assert payload["role"] == "held_out"
    assert payload["observations"]["a"]["box"] == [1, 2, 3, 4]
    assert payload["observations"]["a"]["embedding_dim"] == 2
    assert "embedding" not in payload["observations"]["a"]

    detection_result = ModelResult(
        key="yolov8n",
        family="detection",
        status="ok",
        description="",
        role="held_out",
        observations={
            "a": DetectionObservation(
                detections=[Detection(label="person", confidence=0.9, box=(0, 0, 5, 5))]
            )
        },
    )
    detection_payload = detection_result.to_dict()
    assert detection_payload["observations"]["a"]["count"] == 1
    assert detection_payload["observations"]["a"]["detections"][0]["label"] == "person"
