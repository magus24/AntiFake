"""Service-layer tests: temp-file lifecycle and artifact assembly."""

from __future__ import annotations

import contextlib
import io
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from app.core.config import Settings
from app.protection.base import ProtectionEngine
from app.protection.engine import ProtectionOutput
from app.protection.image_io import ImageValidationError
from app.services.protection_service import ProtectionService
from PIL import Image

from tests.helpers import decode, encode_jpeg, encode_png, make_photo


@contextlib.contextmanager
def scoped_temp_root() -> Iterator[str]:
    """Yield a private temp root, removed on exit."""
    with tempfile.TemporaryDirectory(prefix="antifake-test-") as root:
        yield root


class TestProtectUpload:
    def test_returns_a_reopenable_protected_image(
        self, service: ProtectionService, photo_png_bytes: bytes
    ) -> None:
        artifact = service.protect_upload(photo_png_bytes, "image/png")
        assert artifact.protected_bytes
        assert decode(artifact.protected_bytes).shape == make_photo().shape

    def test_png_in_png_out(self, service: ProtectionService) -> None:
        artifact = service.protect_upload(encode_png(make_photo()), "image/png")
        assert artifact.output_format == "PNG"
        assert artifact.media_type == "image/png"
        assert artifact.filename == "antifake-protected.png"

    def test_jpeg_in_jpeg_out(self, service: ProtectionService) -> None:
        artifact = service.protect_upload(encode_jpeg(make_photo()), "image/jpeg")
        assert artifact.output_format == "JPEG"
        assert artifact.media_type == "image/jpeg"
        assert artifact.filename == "antifake-protected.jpg"

    @pytest.mark.parametrize(("width", "height"), [(96, 64), (640, 480), (33, 47)])
    def test_dimensions_match_the_input(
        self, service: ProtectionService, width: int, height: int
    ) -> None:
        raw = encode_png(make_photo(width, height))
        artifact = service.protect_upload(raw, "image/png")
        assert (artifact.width, artifact.height) == (width, height)
        assert decode(artifact.protected_bytes).shape[:2] == (height, width)

    def test_reports_both_metric_sets(
        self, service: ProtectionService, photo_png_bytes: bytes
    ) -> None:
        artifact = service.protect_upload(photo_png_bytes, "image/png")
        # PNG is lossless, so pre- and post-encode metrics must agree exactly.
        assert artifact.perturbation_metrics.linf_255 == pytest.approx(
            artifact.delivered_metrics.linf_255
        )
        assert artifact.delivered_metrics.psnr_db > 30.0
        assert artifact.process_ms > 0.0

    def test_protected_bytes_differ_from_the_original(
        self, service: ProtectionService, photo_png_bytes: bytes
    ) -> None:
        artifact = service.protect_upload(photo_png_bytes, "image/png")
        assert not np.array_equal(decode(artifact.protected_bytes), artifact.original)

    def test_jpeg_delivery_can_erode_the_perturbation(
        self, engine: ProtectionEngine
    ) -> None:
        settings = Settings(jpeg_quality=70, max_upload_bytes=20 * 1024 * 1024)
        service = ProtectionService(engine=engine, settings=settings)
        artifact = service.protect_upload(
            encode_jpeg(make_photo(320, 240)), "image/jpeg"
        )
        assert (
            artifact.delivered_metrics.linf_255
            >= artifact.perturbation_metrics.linf_255
        )

    def test_invalid_upload_raises_validation_error(
        self, service: ProtectionService
    ) -> None:
        with pytest.raises(ImageValidationError):
            service.protect_upload(b"not an image at all", "image/png")

    def test_oversized_upload_is_rejected(self, engine: ProtectionEngine) -> None:
        service = ProtectionService(
            engine=engine, settings=Settings(max_upload_bytes=64)
        )
        with pytest.raises(ImageValidationError, match="limit"):
            service.protect_upload(encode_png(make_photo(64, 64)), "image/png")

    def test_transparent_png_is_flattened(self, service: ProtectionService) -> None:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 3] = 0  # fully transparent
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        artifact = service.protect_upload(buffer.getvalue(), "image/png")
        assert artifact.output_format == "PNG"
        # Must come back as a plain three-channel image, not invisible black.
        assert decode(artifact.protected_bytes).shape == (32, 32, 3)


class TestTempFileLifecycle:
    def test_nothing_is_left_behind_on_success(
        self, service: ProtectionService
    ) -> None:
        with scoped_temp_root() as root:
            service.protect_upload(encode_png(make_photo()), "image/png")
            assert list(Path(root).iterdir()) == []

    def test_nothing_is_left_behind_on_failure(
        self, service: ProtectionService
    ) -> None:
        with scoped_temp_root() as root:
            with pytest.raises(ImageValidationError):
                service.protect_upload(b"garbage", "image/png")
            assert list(Path(root).iterdir()) == []

    def test_files_exist_during_processing_and_are_removed_after(
        self, engine: ProtectionEngine, settings: Settings
    ) -> None:
        """The staging file must be present mid-request and gone afterwards."""
        observed: dict[str, int] = {}
        original = engine.protect_with_meta

        with scoped_temp_root() as root:
            temp_settings = replace(settings, temp_dir=root)
            service = ProtectionService(engine=engine, settings=temp_settings)

            def spy(image: np.ndarray) -> ProtectionOutput:
                observed["during"] = len(list(Path(root).iterdir()))
                return original(image)

            engine.protect_with_meta = spy  # type: ignore[method-assign]
            try:
                service.protect_upload(encode_png(make_photo()), "image/png")
            finally:
                engine.protect_with_meta = original  # type: ignore[method-assign]

            assert observed["during"] >= 1
            assert list(Path(root).iterdir()) == []

    def test_configured_temp_dir_is_created_if_missing(
        self, engine: ProtectionEngine, tmp_path: Path
    ) -> None:
        nested = tmp_path / "a" / "b"
        service = ProtectionService(
            engine=engine, settings=Settings(temp_dir=str(nested))
        )
        artifact = service.protect_upload(encode_png(make_photo()), "image/png")
        assert artifact.protected_bytes
        assert nested.exists()
        assert list(nested.iterdir()) == []


class TestMinimalEngineFallback:
    """A bare ProtectionEngine implementation must still be usable."""

    def test_engine_without_protect_with_meta_still_works(self) -> None:
        class TintedEngine(ProtectionEngine):
            name = "tint"
            target_description = "none"
            preserves_dimensions = True

            def protect(self, image: np.ndarray) -> np.ndarray:
                return np.clip(image.astype(np.int16) + 3, 0, 255).astype(np.uint8)

        service = ProtectionService(
            engine=TintedEngine(),
            settings=Settings(max_upload_bytes=10 * 1024 * 1024),
        )
        artifact = service.protect_upload(encode_png(make_photo()), "image/png")
        assert artifact.protected_bytes
        assert artifact.perturbation_metrics.linf_8bit == 3
        assert artifact.engine_meta.engine == "tint"  # type: ignore[attr-defined]
        assert artifact.width == 96 and artifact.height == 64
