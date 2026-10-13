"""Image loading and validation tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
from app.protection.image_io import (
    ImageValidationError,
    encode_image,
    load_image_from_bytes,
    output_extension,
    output_media_type,
    validate_upload_size,
)
from PIL import Image

from tests.helpers import decode, encode_jpeg, encode_png, make_photo


class TestLoadFromBytes:
    def test_png_round_trip_preserves_pixels(self, photo_png_bytes: bytes) -> None:
        loaded = load_image_from_bytes(photo_png_bytes, content_type="image/png")
        assert loaded.array.dtype == np.uint8
        assert loaded.array.shape == make_photo().shape
        assert loaded.source_format == "PNG"
        np.testing.assert_array_equal(loaded.array, make_photo())

    def test_jpeg_is_decoded_to_rgb_array(self, photo_jpeg_bytes: bytes) -> None:
        loaded = load_image_from_bytes(photo_jpeg_bytes, content_type="image/jpeg")
        assert loaded.source_format == "JPEG"
        assert loaded.array.dtype == np.uint8
        assert loaded.array.shape == make_photo().shape

    def test_reports_dimensions(self, photo_png_bytes: bytes) -> None:
        loaded = load_image_from_bytes(photo_png_bytes, content_type="image/png")
        assert (loaded.width, loaded.height) == (96, 64)

    def test_grayscale_png_is_widened_to_three_channels(self) -> None:
        buffer = io.BytesIO()
        Image.fromarray(make_photo()[:, :, 0], mode="L").save(buffer, format="PNG")
        loaded = load_image_from_bytes(buffer.getvalue(), content_type="image/png")
        assert loaded.array.shape[2] == 3
        assert loaded.was_converted is True
        # Widening a grey channel must not change the value.
        np.testing.assert_array_equal(
            loaded.array[:, :, 0], loaded.array[:, :, 1]
        )

    def test_rgba_png_is_composited_over_white(self) -> None:
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., :3] = 0
        rgba[..., 3] = 0  # fully transparent
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        loaded = load_image_from_bytes(buffer.getvalue(), content_type="image/png")
        # Transparent black over white must become white, not black.
        assert loaded.array.min() == 255
        assert loaded.was_converted is True

    def test_non_contiguous_input_is_normalised(self) -> None:
        array = make_photo()
        view = array[::2, ::2]  # strided view
        loaded = load_image_from_bytes(encode_png(np.ascontiguousarray(view)))
        assert loaded.array.flags["C_CONTIGUOUS"] is True

    @pytest.mark.parametrize(
        "content_type",
        ["image/jpeg", "image/jpg", "image/png", "image/PNG; charset=binary"],
    )
    def test_accepts_known_content_types(self, content_type: str) -> None:
        raw = (
            encode_png(make_photo())
            if "png" in content_type.lower()
            else encode_jpeg(make_photo())
        )
        assert load_image_from_bytes(raw, content_type=content_type).array is not None

    def test_missing_content_type_falls_back_to_byte_sniffing(self) -> None:
        assert load_image_from_bytes(encode_png(make_photo())).source_format == "PNG"


class TestLoadRejections:
    def test_empty_payload(self) -> None:
        with pytest.raises(ImageValidationError, match="empty"):
            load_image_from_bytes(b"")

    def test_oversized_payload(self) -> None:
        with pytest.raises(ImageValidationError, match="limit"):
            load_image_from_bytes(b"x" * 100, max_bytes=10)

    def test_rejects_unsupported_content_type(self, photo_png_bytes: bytes) -> None:
        with pytest.raises(ImageValidationError, match="Unsupported content type"):
            load_image_from_bytes(photo_png_bytes, content_type="image/gif")

    def test_rejects_non_image_bytes(self) -> None:
        with pytest.raises(ImageValidationError, match="not a decodable image"):
            load_image_from_bytes(
                b"this is definitely not an image", content_type="image/png"
            )

    def test_rejects_truncated_image(self) -> None:
        raw = encode_png(make_photo())
        with pytest.raises(ImageValidationError):
            load_image_from_bytes(raw[: len(raw) // 3], content_type="image/png")

    def test_rejects_webp(self) -> None:
        buffer = io.BytesIO()
        Image.fromarray(make_photo(), mode="RGB").save(buffer, format="WEBP")
        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            load_image_from_bytes(buffer.getvalue())

    def test_rejects_bmp(self) -> None:
        buffer = io.BytesIO()
        Image.fromarray(make_photo(), mode="RGB").save(buffer, format="BMP")
        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            load_image_from_bytes(buffer.getvalue())

    def test_rejects_too_many_pixels(self) -> None:
        with pytest.raises(ImageValidationError, match="limit is"):
            load_image_from_bytes(encode_png(make_photo(6000, 6000)))

    def test_validate_upload_size_boundaries(self) -> None:
        validate_upload_size(10, max_bytes=10)
        with pytest.raises(ImageValidationError):
            validate_upload_size(11, max_bytes=10)


class TestEncode:
    @pytest.mark.parametrize("image_format", ["PNG", "png", "JPEG", "jpeg"])
    def test_output_is_reopenable(self, photo: np.ndarray, image_format: str) -> None:
        data = encode_image(photo, image_format)
        assert data
        decoded = decode(data)
        assert decoded.shape == photo.shape

    def test_png_is_lossless(self, photo: np.ndarray) -> None:
        np.testing.assert_array_equal(decode(encode_image(photo, "PNG")), photo)

    def test_jpeg_keeps_dimensions(self, photo: np.ndarray) -> None:
        assert decode(encode_image(photo, "JPEG")).shape == photo.shape

    def test_rejects_unknown_format(self, photo: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unsupported output format"):
            encode_image(photo, "TIFF")

    def test_rejects_float_array(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            encode_image(np.zeros((4, 4, 3), dtype=np.float32), "PNG")

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            encode_image(np.zeros((4, 4, 4), dtype=np.uint8), "PNG")

    @pytest.mark.parametrize(
        ("image_format", "extension", "media_type"),
        [("JPEG", "jpg", "image/jpeg"), ("PNG", "png", "image/png")],
    )
    def test_extension_and_media_type_mapping(
        self, image_format: str, extension: str, media_type: str
    ) -> None:
        assert output_extension(image_format) == extension
        assert output_media_type(image_format) == media_type
