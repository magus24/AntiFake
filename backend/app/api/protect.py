"""HTTP layer for the protect endpoint.

Responsibilities stop at: validate the request, hand bytes to the service, and
serialise the answer. No image logic and no attack logic lives here.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import Settings, get_settings
from app.protection.image_io import SUPPORTED_FORMATS, ImageValidationError
from app.protection.metrics import DifferenceMetrics
from app.schemas.protect import (
    DifferenceMetricsOut,
    EngineInfoOut,
    HealthResponseOut,
    ProtectResponseOut,
)
from app.services.protection_service import ProtectionArtifact, ProtectionService

logger = logging.getLogger(__name__)

router = APIRouter()

DISCLAIMER = (
    "Perturbation is optimised against one named pipeline only "
    "(torchvision ResNet-18 / ImageNet-1k with resize-256 centre-crop-224 "
    "preprocessing). It is not a general defence against AI image models, "
    "and no such claim is made."
)

#: When true the protected image bytes are returned directly; when false the
#: same bytes are base64-wrapped in JSON so a browser can preview them.
_RETURN_IMAGE = Query(
    default=True,
    description="Return image/jpeg or image/png bytes instead of a JSON envelope.",
)


def get_service(request: Request) -> ProtectionService:
    """Resolve the service from app state, set up once at startup."""
    service: ProtectionService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only if startup was skipped
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Protection service is not initialised.",
        )
    return service


def _metrics_out(metrics: DifferenceMetrics) -> DifferenceMetricsOut:
    return DifferenceMetricsOut(
        linf_255=round(metrics.linf_255, 4),
        linf_8bit=metrics.linf_8bit,
        l2=round(metrics.l2, 4),
        mean_abs=round(metrics.mean_abs, 4),
        psnr_db=(
            round(metrics.psnr_db, 4) if metrics.psnr_db != float("inf") else 999.0
        ),
        ssim=round(metrics.ssim, 6),
    )


def _engine_out(service: ProtectionService, settings: Settings) -> EngineInfoOut:
    described = service.engine.describe()
    return EngineInfoOut(
        engine=described["engine"],
        target_model=described["target_model"],
        preserves_dimensions=described["preserves_dimensions"] == "True",
        device=described.get("device", "cpu"),
        eps_255=float(described.get("eps_255", "0")),
        steps=int(described.get("steps", "0")),
    )


@router.get("/health", response_model=HealthResponseOut, tags=["meta"])
def health(
    service: ProtectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponseOut:
    """Liveness probe and a dump of the currently loaded engine configuration."""
    return HealthResponseOut(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        engine=_engine_out(service, settings),
        supported_formats=sorted(SUPPORTED_FORMATS),
        max_upload_bytes=settings.max_upload_bytes,
    )


@router.post(
    "/api/protect",
    responses={
        200: {
            "content": {"image/png": {}, "image/jpeg": {}, "application/json": {}},
            "description": "Protected image bytes, or a JSON metadata envelope.",
        },
        400: {"description": "The upload is not an acceptable JPG/PNG image."},
        413: {"description": "The upload exceeds the size limit."},
    },
    tags=["protection"],
    response_model=None,
)
def protect(
    request: Request,
    image: UploadFile = File(..., description="A single JPG or PNG image."),
    return_image: bool = _RETURN_IMAGE,
    service: ProtectionService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse | JSONResponse:
    """Protect one uploaded image.

    The response body is the protected image itself (identical dimensions to
    the input) with a ``X-AntiFake-*`` header set carrying the metrics, or,
    with ``return_image=false``, a JSON envelope with base64 image data.
    """
    raw = _read_upload(image, settings)
    try:
        artifact = service.protect_upload(raw, image.content_type)
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    engine_info = _engine_out(service, settings)
    envelope = ProtectResponseOut(
        filename=artifact.filename,
        media_type=artifact.media_type,
        width=artifact.width,
        height=artifact.height,
        source_format=artifact.source_format,
        output_format=artifact.output_format,
        process_ms=artifact.process_ms,
        perturbation_metrics=_metrics_out(artifact.perturbation_metrics),
        delivered_metrics=_metrics_out(artifact.delivered_metrics),
        engine=engine_info,
        disclaimer=DISCLAIMER,
    )

    if return_image:
        headers = _metric_headers(artifact, engine_info)
        return StreamingResponse(
            BytesIO(artifact.protected_bytes),
            media_type=artifact.media_type,
            headers=headers,
        )

    payload = envelope.model_dump()
    payload["image_base64"] = base64.b64encode(artifact.protected_bytes).decode("ascii")
    return JSONResponse(content=payload)


# -- helpers ---------------------------------------------------------------


def _read_upload(upload: UploadFile, settings: Settings) -> bytes:
    if upload.filename is not None and not upload.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file has no name."
        )
    try:
        raw = upload.file.read()
    finally:
        upload.file.close()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Upload is {len(raw)} bytes, limit is {settings.max_upload_bytes} bytes."
            ),
        )
    logger.info(
        "request.received filename=%s content_type=%s bytes=%d",
        upload.filename,
        upload.content_type,
        len(raw),
    )
    return raw


def _metric_headers(
    artifact: ProtectionArtifact, engine: EngineInfoOut
) -> dict[str, str]:
    delivered = artifact.delivered_metrics
    return {
        "Content-Disposition": f'attachment; filename="{artifact.filename}"',
        "X-AntiFake-Engine": engine.engine,
        "X-AntiFake-Target-Model": engine.target_model,
        "X-AntiFake-Device": engine.device,
        "X-AntiFake-Source-Format": artifact.source_format,
        "X-AntiFake-Output-Format": artifact.output_format,
        "X-AntiFake-Width": str(artifact.width),
        "X-AntiFake-Height": str(artifact.height),
        "X-AntiFake-Process-Ms": f"{artifact.process_ms:.2f}",
        "X-AntiFake-Linf": f"{delivered.linf_255:.4f}",
        "X-AntiFake-PSNR": f"{delivered.psnr_db:.4f}",
        "X-AntiFake-SSIM": f"{delivered.ssim:.6f}",
    }
