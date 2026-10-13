"""Integration tests for POST /api/protect."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from app.core.config import Settings
from app.main import create_app
from app.protection.engine import AdversarialProtectionEngine
from app.protection.image_io import ImageValidationError
from app.protection.metrics import compute_metrics
from app.services.protection_service import ProtectionService
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import encode_jpeg, encode_png, make_photo


def png_upload(
    data: bytes | None = None, name: str = "photo.png"
) -> dict[str, bytes]:
    """Build a multipart payload for the single-file ``image`` field."""
    payload = data if data is not None else encode_png(make_photo())
    return {"image": (name, payload, "image/png")}


def jpeg_upload(name: str = "photo.jpg") -> dict[str, bytes]:
    return {"image": (name, encode_jpeg(make_photo()), "image/jpeg")}


class TestHealth:
    def test_reports_engine_configuration(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["engine"]["engine"] == "pgd-linf-resnet18"
        assert "resnet18" in body["engine"]["target_model"]
        assert body["engine"]["preserves_dimensions"] is True
        assert body["engine"]["device"] == "cpu"
        assert set(body["supported_formats"]) == {"JPEG", "PNG"}

    def test_index_points_at_the_protect_endpoint(self, client: TestClient) -> None:
        assert client.get("/").json()["protect"] == "POST /api/protect"


class TestProtectBinaryResponse:
    def test_returns_an_openable_image(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload())
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as handle:
            handle.load()
            assert handle.format == "PNG"
            assert handle.mode == "RGB"

    def test_output_dimensions_match_input(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload())
        with Image.open(io.BytesIO(response.content)) as handle:
            assert handle.size == (96, 64)
        assert response.headers["X-AntiFake-Width"] == "96"
        assert response.headers["X-AntiFake-Height"] == "64"

    @pytest.mark.parametrize(("width", "height"), [(200, 120), (64, 300), (301, 301)])
    def test_dimensions_preserved_for_various_sizes(
        self, client: TestClient, width: int, height: int
    ) -> None:
        raw = encode_png(make_photo(width, height))
        response = client.post("/api/protect", files=png_upload(raw))
        with Image.open(io.BytesIO(response.content)) as handle:
            assert handle.size == (width, height)

    def test_content_type_is_preserved(self, client: TestClient) -> None:
        png = client.post("/api/protect", files=png_upload())
        assert png.headers["content-type"] == "image/png"
        jpeg = client.post("/api/protect", files=jpeg_upload())
        assert jpeg.headers["content-type"] == "image/jpeg"

    def test_triggers_a_download_with_a_filename(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload())
        assert "attachment" in response.headers["content-disposition"]
        assert "antifake-protected.png" in response.headers["content-disposition"]

    def test_metrics_headers_are_present_and_sane(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload())
        headers = response.headers
        assert headers["X-AntiFake-Engine"] == "pgd-linf-resnet18"
        assert "resnet18" in headers["X-AntiFake-Target-Model"]
        assert headers["X-AntiFake-Device"] == "cpu"
        assert headers["X-AntiFake-Source-Format"] == "PNG"
        assert 0.0 < float(headers["X-AntiFake-Linf"]) <= 9.0
        assert float(headers["X-AntiFake-PSNR"]) > 30.0
        assert 0.9 < float(headers["X-AntiFake-SSIM"]) <= 1.0
        assert float(headers["X-AntiFake-Process-Ms"]) > 0.0

    def test_protected_bytes_differ_from_the_original(
        self, client: TestClient
    ) -> None:
        original = make_photo()
        response = client.post("/api/protect", files=png_upload())
        with Image.open(io.BytesIO(response.content)) as handle:
            delivered = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        assert not np.array_equal(original, delivered)

    def test_similarity_metrics_match_a_local_computation(
        self, client: TestClient
    ) -> None:
        original = make_photo()
        response = client.post("/api/protect", files=png_upload())
        with Image.open(io.BytesIO(response.content)) as handle:
            delivered = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        metrics = compute_metrics(original, delivered)
        assert float(response.headers["X-AntiFake-Linf"]) == pytest.approx(
            metrics.linf_255, abs=1e-3
        )
        assert float(response.headers["X-AntiFake-PSNR"]) == pytest.approx(
            metrics.psnr_db, abs=1e-2
        )
        assert float(response.headers["X-AntiFake-SSIM"]) == pytest.approx(
            metrics.ssim, abs=1e-5
        )

    def test_requests_are_reproducible(self, client: TestClient) -> None:
        first = client.post("/api/protect", files=png_upload())
        second = client.post("/api/protect", files=png_upload())
        assert first.content == second.content

    def test_jpeg_upload_returns_jpeg(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=jpeg_upload())
        assert response.headers["content-type"] == "image/jpeg"
        with Image.open(io.BytesIO(response.content)) as handle:
            handle.load()
            assert handle.format == "JPEG"
            assert handle.size == (96, 64)


class TestProtectJsonResponse:
    def test_returns_base64_image_and_metrics(self, client: TestClient) -> None:
        response = client.post(
            "/api/protect", files=png_upload(), params={"return_image": "false"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["media_type"] == "image/png"
        assert body["width"] == 96 and body["height"] == 64
        assert body["output_format"] == "PNG"
        assert body["engine"]["engine"] == "pgd-linf-resnet18"
        assert body["delivered_metrics"]["psnr_db"] > 30.0
        assert body["perturbation_metrics"]["linf_8bit"] <= 9
        assert "not a general defence" in body["disclaimer"]

    def test_base64_payload_decodes_to_the_same_image(
        self, client: TestClient
    ) -> None:
        binary = client.post("/api/protect", files=png_upload())
        json_response = client.post(
            "/api/protect", files=png_upload(), params={"return_image": "false"}
        )
        decoded = base64.b64decode(json_response.json()["image_base64"])
        assert decoded == binary.content

    def test_envelope_is_json_serialisable_without_nan(
        self, client: TestClient
    ) -> None:
        text = json.dumps(
            client.post(
                "/api/protect", files=png_upload(), params={"return_image": "false"}
            ).json()
        )
        assert "NaN" not in text and "Infinity" not in text


class TestProtectRejections:
    def test_missing_file_field(self, client: TestClient) -> None:
        response = client.post("/api/protect", data={"notimage": "x"})
        assert response.status_code == 422

    def test_empty_file(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload(b""))
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_non_image_bytes(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload(b"nope, not an image"))
        assert response.status_code == 400

    def test_truncated_image(self, client: TestClient) -> None:
        raw = encode_png(make_photo())[:80]
        response = client.post("/api/protect", files=png_upload(raw))
        assert response.status_code == 400

    def test_unsupported_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/protect",
            files={"image": ("photo.gif", encode_png(make_photo()), "image/gif")},
        )
        assert response.status_code == 400
        assert "content type" in response.json()["detail"].lower()

    def test_unsupported_inner_format(self, client: TestClient) -> None:
        buffer = io.BytesIO()
        Image.fromarray(make_photo(), mode="RGB").save(buffer, format="WEBP")
        response = client.post(
            "/api/protect",
            files={"image": ("photo.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 400
        assert "WEBP" in response.json()["detail"]

    def test_oversized_upload(self, client: TestClient) -> None:
        # The test client runs with a 20 MB cap; 21 MB of PNG bytes trips it.
        payload = encode_png(make_photo(64, 64)) + b"\x00" * (21 * 1024 * 1024)
        response = client.post("/api/protect", files=png_upload(payload))
        assert response.status_code in (400, 413)

    def test_error_detail_never_leaks_a_traceback(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload(b"garbage"))
        assert response.status_code == 400
        assert "Traceback" not in response.text


class TestEdgeCases:
    def test_tiny_image(self, client: TestClient) -> None:
        raw = encode_png(make_photo(1, 1))
        response = client.post("/api/protect", files=png_upload(raw))
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as handle:
            assert handle.size == (1, 1)

    def test_non_square_image(self, client: TestClient) -> None:
        raw = encode_png(make_photo(400, 120))
        response = client.post("/api/protect", files=png_upload(raw))
        with Image.open(io.BytesIO(response.content)) as handle:
            assert handle.size == (400, 120)

    def test_filename_with_spaces(self, client: TestClient) -> None:
        response = client.post(
            "/api/protect", files=png_upload(name="my photo 2024.png")
        )
        assert response.status_code == 200

    def test_unknown_declared_content_type_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/protect",
            files={
                "image": (
                    "photo.png",
                    encode_png(make_photo()),
                    "application/octet-stream",
                )
            },
        )
        # The declared type is not a supported image type, so it is refused
        # explicitly rather than silently guessed from the bytes.
        assert response.status_code == 400


class TestTempFileCleanup:
    def test_service_leaves_no_temporary_state(
        self,
        tmp_path: Path,
        settings: Settings,
        engine: AdversarialProtectionEngine,
    ) -> None:
        """A full request through the API must not leave files behind."""
        temp_root = tmp_path / "api-temp"
        service = ProtectionService(
            engine=engine,
            settings=replace(settings, temp_dir=str(temp_root)),
        )
        with TestClient(create_app(service=service)) as client:
            assert client.post("/api/protect", files=png_upload()).status_code == 200
        assert temp_root.exists()
        assert list(temp_root.iterdir()) == []


class TestValidationErrorMapping:
    def test_validation_error_carries_a_human_message(self) -> None:
        with pytest.raises(ImageValidationError, match="boom"):
            raise ImageValidationError("boom")

    def test_service_surfaces_it_to_the_api_as_400(self, client: TestClient) -> None:
        response = client.post("/api/protect", files=png_upload(b""))
        assert response.status_code == 400
        assert "detail" in response.json()
