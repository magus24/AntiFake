"""Tests that need real pretrained weights.

Marked ``network`` because they download checkpoints on first run. Run with::

    pytest -m network

These are the tests that would catch a silently broken adapter: a model that
loads but mis-preprocesses, or a CLIP whose embeddings are not comparable,
produces a benchmark that is confidently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from app.evaluation.adapters_classification import build_classifier
from app.evaluation.adapters_embedding import DEFAULT_PROMPTS, build_clip
from app.evaluation.base import ClassificationObservation, EmbeddingObservation
from app.evaluation.comparison import compare

from tests.helpers import make_photo

pytestmark = pytest.mark.network

CPU = torch.device("cpu")

CLASSIFIERS = [
    "resnet18",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "vit_b_16",
]


@pytest.mark.parametrize("model_name", CLASSIFIERS)
def test_classifier_loads_and_produces_a_valid_distribution(model_name):
    adapter = build_classifier(CPU, model_name)
    adapter.load()
    try:
        observation = adapter.evaluate(make_photo(96, 128))
        assert isinstance(observation, ClassificationObservation)
        assert 0.0 <= observation.top1_probability <= 1.0
        assert 0 <= observation.top1_index < 1000
        assert len(observation.top5) == 5
        assert observation.entropy > 0.0
    finally:
        adapter.release()


@pytest.mark.parametrize("model_name", CLASSIFIERS)
def test_classifier_preprocess_is_224_and_normalised(model_name):
    adapter = build_classifier(CPU, model_name)
    adapter.load()
    try:
        tensor = adapter.preprocess(make_photo(96, 128))
        assert tensor.shape == (1, 3, 224, 224)
        # Normalised tensors are roughly zero-mean; raw [0,1] pixels are not.
        assert abs(float(tensor.mean())) < 1.0
    finally:
        adapter.release()


def test_classifier_labels_are_human_readable():
    adapter = build_classifier(CPU, "resnet18")
    adapter.load()
    try:
        observation = adapter.evaluate(make_photo(64, 64))
        assert observation.top1_label
        assert not observation.top1_label.startswith("class_")
    finally:
        adapter.release()


def test_classifier_rejects_an_unknown_weights_name():
    from app.evaluation.base import ModelUnavailableError

    adapter = build_classifier(CPU, "resnet18", "IMAGENET1K_V9")
    with pytest.raises(ModelUnavailableError, match="no weights"):
        adapter.load()


def test_two_adapters_agree_on_the_same_image():
    """Determinism across separate instances: no hidden state leaking."""
    image = make_photo(96, 96)
    results = []
    for _ in range(2):
        adapter = build_classifier(CPU, "resnet18")
        adapter.load()
        try:
            results.append(adapter.evaluate(image).top1_probability)
        finally:
            adapter.release()
    assert results[0] == pytest.approx(results[1], abs=1e-5)


def test_benchmark_resnet_matches_the_engine_target():
    """The benchmark must read exactly the pipeline the engine defends.

    Any drift between these two numbers would invalidate every transfer
    measurement in the report.
    """
    from app.protection.target_model import TargetModel, TargetModelConfig

    image = make_photo(334, 414)
    adapter = build_classifier(CPU, "resnet18")
    adapter.load()
    try:
        from_benchmark = adapter.evaluate(image)
    finally:
        adapter.release()

    engine_target = TargetModel(TargetModelConfig(), device=CPU)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float() / 255.0
    logits = engine_target.predict(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    from_engine = float(probabilities.max())

    assert from_benchmark.top1_probability == pytest.approx(from_engine, abs=1e-5)


def test_clip_loads_and_embeds():
    adapter = build_clip(CPU)
    adapter.load()
    try:
        observation = adapter.evaluate(make_photo(96, 96))
        assert isinstance(observation, EmbeddingObservation)
        assert observation.dimension == 512
        # The adapter normalises, so this must hold to float precision.
        norm = float(np.linalg.norm(observation.embedding))
        assert norm == pytest.approx(1.0, abs=1e-5)
        assert set(observation.text_similarities) == set(DEFAULT_PROMPTS)
        assert sorted(
            observation.text_ranking,
            key=lambda p: -observation.text_similarities[p],
        ) == observation.text_ranking
    finally:
        adapter.release()


def test_clip_embedding_is_deterministic():
    image = make_photo(96, 96)
    results = []
    for _ in range(2):
        adapter = build_clip(CPU)
        adapter.load()
        try:
            results.append(adapter.evaluate(image).embedding)
        finally:
            adapter.release()
    assert np.allclose(results[0], results[1], atol=1e-5)


def test_clip_cosine_of_identical_images_is_one():
    adapter = build_clip(CPU)
    adapter.load()
    try:
        image = make_photo(96, 96)
        row = compare(adapter.evaluate(image), adapter.evaluate(image))
        assert row["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
        assert row["best_prompt_changed"] is False
    finally:
        adapter.release()


def test_clip_detects_a_semantic_change():
    """A large semantic edit must move the embedding, or CLIP is measuring noise."""
    from tests.helpers import make_detailed

    adapter = build_clip(CPU)
    adapter.load()
    try:
        structured = make_detailed(224, 224, seed=1)
        inverted = 255 - structured
        row = compare(adapter.evaluate(structured), adapter.evaluate(inverted))
        assert row["cosine_similarity"] < 0.99
    finally:
        adapter.release()


def test_classifier_adapter_releases_weights():
    adapter = build_classifier(CPU, "resnet18")
    adapter.load()
    assert adapter.is_loaded
    adapter.release()
    assert not adapter.is_loaded
    with pytest.raises(RuntimeError, match="before load"):
        adapter.evaluate(make_photo(32, 32))
