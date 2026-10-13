"""Shared fixtures.

The heavyweight default is a target model with *randomly initialised* weights.
That keeps the suite offline and fast while exercising the exact same
differentiable path as production. Two opt-in tests (``network`` marker) load
the real ImageNet weights to prove the pipeline works end to end.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from app.core.config import Settings
from app.main import create_app
from app.protection.config import ProtectionConfig
from app.protection.engine import AdversarialProtectionEngine
from app.protection.target_model import clear_model_cache
from app.services.protection_service import ProtectionService
from fastapi.testclient import TestClient

from tests.helpers import TEST_TARGET, encode_jpeg, encode_png, make_photo

#: Cheap attack settings. Three PGD steps is plenty to assert bounds,
#: determinism and dimension preservation.
FAST_EPS_255 = 8.0
FAST_STEPS = 3
FAST_SEED = 0


@pytest.fixture(scope="session", autouse=True)
def _deterministic_torch() -> Iterator[None]:
    """Pin torch to single-threaded CPU math so results are bit-reproducible."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture(autouse=True)
def _reset_model_cache() -> Iterator[None]:
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def settings() -> Settings:
    """Test settings: real engine shape, tiny step count, CPU only."""
    return Settings(
        default_eps_255=FAST_EPS_255,
        default_steps=FAST_STEPS,
        default_seed=FAST_SEED,
        default_device="cpu",
        model_weights="none",
        jpeg_quality=95,
    )


@pytest.fixture
def fast_config(settings: Settings) -> ProtectionConfig:
    """Attack config derived from the test settings."""
    return ProtectionConfig(
        eps=settings.default_eps_255 / 255.0,
        alpha=(settings.default_eps_255 / 255.0) / 4.0,
        steps=settings.default_steps,
        seed=settings.default_seed,
        device="cpu",
        model_weights="none",
    )


@pytest.fixture
def engine(fast_config: ProtectionConfig) -> AdversarialProtectionEngine:
    """PGD engine over a randomly initialised ResNet-18."""
    return AdversarialProtectionEngine(fast_config, TEST_TARGET)


@pytest.fixture
def service(
    engine: AdversarialProtectionEngine, settings: Settings
) -> ProtectionService:
    return ProtectionService(engine=engine, settings=settings)


@pytest.fixture
def client(service: ProtectionService) -> Iterator[TestClient]:
    """TestClient wired to the offline engine."""
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


# -- image fixtures --------------------------------------------------------


@pytest.fixture
def photo() -> np.ndarray:
    """A structured 96x64 RGB test image."""
    return make_photo()


@pytest.fixture
def photo_png_bytes() -> bytes:
    return encode_png(make_photo())


@pytest.fixture
def photo_jpeg_bytes() -> bytes:
    return encode_jpeg(make_photo())
