"""Image loading, validation and encoding.

Isolated from the protection engine so that engine tests can feed raw arrays
and API tests can exercise the real decode path without duplicating rules.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.protection.base import RGB

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})
#: ``image/jpg`` is non-standard but common enough in the wild to accept.
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/jpg", "image/pjpeg", "image/png"}
)
MAX_IMAGE_PIXELS = 50_000_000
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


class ImageValidationError(ValueError):
    """Raised when an upload cannot be accepted as a single JPG/PNG image."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """A validated RGB array plus the provenance we need for encoding."""

    array: RGB
    source_format: str
    original_mode: str
    was_converted: bool

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.array.shape[1])


def _normalise_mime(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def validate_upload_size(num_bytes: int, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Reject obviously empty or oversized payloads before decoding."""
    if num_bytes <= 0:
        raise ImageValidationError("Uploaded file is empty.")
    if num_bytes > max_bytes:
        raise ImageValidationError(
            f"Uploaded file is {num_bytes} bytes, limit is {max_bytes} bytes."
        )


def load_image_from_bytes(
    raw: bytes,
    *,
    content_type: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> LoadedImage:
    """Decode and validate an uploaded image.

    The declared content type is only used as an early rejection; the bytes
    themselves are the source of truth, because browsers and curl disagree
    often enough that trusting the header alone breaks real uploads.
    """
    validate_upload_size(len(raw), max_bytes)

    mime = _normalise_mime(content_type)
    if mime is not None and mime not in SUPPORTED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported content type {mime!r}; expected one of "
            f"{sorted(SUPPORTED_MIME_TYPES)}."
        )

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            source_format = (img.format or "UNKNOWN").upper()
            original_mode = img.mode
            has_alpha = original_mode in ("RGBA", "LA", "PA") or (
                original_mode == "P" and "transparency" in img.info
            )
            if has_alpha:
                # Composite over white so a transparent PNG does not turn into
                # invisible black once the alpha channel is dropped.
                rgba = img.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgba = Image.alpha_composite(background, rgba)
                converted = rgba.convert("RGB")
            else:
                converted = img.convert("RGB")
            array = np.asarray(converted, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageValidationError("File is not a decodable image.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image dimensions are too large.") from exc
    except (OSError, ValueError) as exc:
        raise ImageValidationError(f"Could not decode image: {exc}") from exc

    if source_format not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format {source_format}; expected JPEG or PNG."
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    if height < 1 or width < 1:
        raise ImageValidationError("Image has zero width or height.")
    if height * width > MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            f"Image is {width}x{height} ({height * width} pixels), limit is "
            f"{MAX_IMAGE_PIXELS}."
        )

    logger.info(
        "image.loaded width=%d height=%d format=%s mode=%s bytes=%d converted=%s",
        width,
        height,
        source_format,
        original_mode,
        len(raw),
        has_alpha,
    )
    # Copy rather than wrap: numpy views onto a PIL buffer are read-only, and
    # torch.from_numpy refuses those without a warning.
    return LoadedImage(
        array=np.array(array, dtype=np.uint8, order="C", copy=True),
        source_format=source_format,
        original_mode=original_mode,
        was_converted=has_alpha or original_mode != "RGB",
    )


def load_image_from_path(path: str | Path) -> LoadedImage:
    """Decode an image from disk. Convenience wrapper for tests and the CLI."""
    file_path = Path(path)
    return load_image_from_bytes(file_path.read_bytes())


def encode_image(
    array: RGB,
    image_format: str,
    *,
    jpeg_quality: int = 95,
    png_compress_level: int = 6,
) -> bytes:
    """Encode a ``uint8`` RGB array back to JPEG or PNG bytes.

    Args:
        array: ``uint8`` RGB array of shape ``(H, W, 3)``.
        image_format: ``"JPEG"`` or ``"PNG"`` (case-insensitive).
        jpeg_quality: Quality factor for JPEG output. 95 keeps the protective
            perturbation largely intact; lower values erode it.
        png_compress_level: zlib level for PNG output. PNG is lossless, so the
            perturbation is preserved bit-exactly.

    Raises:
        ValueError: If the array dtype/shape or the format is unusable.
    """
    normalised = image_format.upper()
    if normalised not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format {image_format!r}; expected JPEG or PNG."
        )
    if array.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 array, got {array.dtype}.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) array, got shape {array.shape}.")

    image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    buffer = io.BytesIO()
    if normalised == "JPEG":
        image.save(
            buffer, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True
        )
    else:
        buffer.write(
            _encode_png(image, compress_level=png_compress_level)
        )
    return buffer.getvalue()


def _encode_png(image: Image.Image, *, compress_level: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def output_extension(image_format: str) -> str:
    """``"JPEG"`` -> ``"jpg"``, ``"PNG"`` -> ``"png"``."""
    return "jpg" if image_format.upper() == "JPEG" else "png"


def output_media_type(image_format: str) -> str:
    """``"JPEG"`` -> ``"image/jpeg"``, ``"PNG"`` -> ``"image/png"``."""
    return "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
