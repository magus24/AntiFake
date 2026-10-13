"""Test helpers: synthetic image factories and codecs.

Kept out of ``conftest.py`` so test modules can import them explicitly without
relying on conftest injection order.
"""

from __future__ import annotations

import io

import numpy as np
from app.protection.config import TargetModelConfig
from PIL import Image

#: Random-weight target used by every default test, so the suite never needs
#: the network. Attack mechanics are identical; only the semantics change.
TEST_TARGET_NAME = "resnet18"
TEST_TARGET_WEIGHTS = "none"


def make_photo(width: int = 96, height: int = 64, seed: int = 0) -> np.ndarray:
    """A structured, photo-like RGB image: gradients, a disc, and mild noise.

    Structured content matters. A flat grey image has no gradients for the
    attack to exploit, which would make the bound and similarity assertions
    pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    channels = (
        (xs * 1.4 + ys * 0.6) % 256,
        (ys * 1.8 - xs * 0.3) % 256,
        (xs * 0.5 + ys * 1.2 + 40) % 256,
    )
    image = np.stack(
        [np.broadcast_to(channel, (height, width)) for channel in channels],
        axis=-1,
    ).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = min(height, width) / 3
    disc = (yy - height / 2) ** 2 + (xx - width / 2) ** 2 < radius**2
    image[disc] = np.array([240.0, 30.0, 200.0], dtype=np.float32)
    image += rng.normal(0.0, 6.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_detailed(width: int = 96, height: int = 64, seed: int = 0) -> np.ndarray:
    """A busy, high-detail RGB image built from upsampled random blobs.

    :func:`make_photo` is deliberately close to a smooth gradient, which is
    what you want when asserting that an attack *did* something. Structural
    similarity is the opposite: on a near-linear gradient, SSIM is dominated by
    the luminance term and barely moves. This image has real edges and texture
    so SSIM responds the way the standard says it should.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8).astype(np.float32)
    mid = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8).astype(np.float32)
    field = np.asarray(
        Image.fromarray(coarse.astype(np.uint8), mode="RGB").resize(
            (width, height), Image.BILINEAR
        ),
        dtype=np.float32,
    )
    fine = np.asarray(
        Image.fromarray(mid.astype(np.uint8), mode="RGB").resize(
            (width, height), Image.BILINEAR
        ),
        dtype=np.float32,
    )
    combined = 0.55 * field + 0.30 * fine + rng.normal(0.0, 10.0, (height, width, 3))
    return np.clip(combined, 0, 255).astype(np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    """Encode a ``uint8`` RGB array to PNG bytes (lossless)."""
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def encode_jpeg(array: np.ndarray, quality: int = 95) -> bytes:
    """Encode a ``uint8`` RGB array to JPEG bytes (lossy)."""
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=0
    )
    return buffer.getvalue()


def decode(data: bytes) -> np.ndarray:
    """Decode image bytes back to a ``uint8`` RGB array."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def test_target() -> TargetModelConfig:
    """A target pipeline description with random, offline-safe weights."""
    return TargetModelConfig(name=TEST_TARGET_NAME, weights=TEST_TARGET_WEIGHTS)


#: Module-level instance so tests can import it directly.
TEST_TARGET = test_target()

