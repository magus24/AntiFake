"""Application services: orchestration between the API layer and the engine.

The protection engine never sees an HTTP request or a file path. This module is
the single place that knows both, and the single place that touches the
filesystem, so the engine stays testable with plain arrays.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.protection.base import RGB, ProtectionEngine
from app.protection.engine import ProtectionOutput
from app.protection.image_io import (
    encode_image,
    load_image_from_bytes,
    output_extension,
    output_media_type,
)
from app.protection.metrics import DifferenceMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProtectionArtifact:
    """Everything the API needs to answer one request.

    Two metric sets are reported on purpose:

    * ``perturbation_metrics`` is measured on the in-memory array straight out
      of the attack, where the L-infinity bound is exact. This is what the
      perturbation bounds tests assert against.
    * ``delivered_metrics`` is measured after the result has been encoded and
      decoded again, i.e. on the bytes the client actually downloads. JPEG is
      lossy, so these can be worse; reporting both avoids overstating how much
      survived the encoder.
    """

    original: RGB
    protected: RGB
    protected_bytes: bytes
    media_type: str
    filename: str
    width: int
    height: int
    source_format: str
    output_format: str
    process_ms: float
    perturbation_metrics: DifferenceMetrics
    delivered_metrics: DifferenceMetrics
    engine_meta: object


class ProtectionService:
    """Runs a protection request end to end.

    Args:
        engine: The protection engine. Constructed by the caller so the
            expensive target model is loaded once at startup rather than per
            request.
        settings: Process settings, used for upload limits and output quality.
    """

    def __init__(self, engine: ProtectionEngine, settings: Settings) -> None:
        """Bind an engine to the process settings.

        Args:
            engine: Built once by the caller so model loading is not per-request.
            settings: Upload limits and output encoding quality.
        """
        self._engine = engine
        self._settings = settings

    @property
    def engine(self) -> ProtectionEngine:
        """The protection engine backing this service."""
        return self._engine

    def protect_upload(
        self, raw: bytes, content_type: str | None
    ) -> ProtectionArtifact:
        """Protect an uploaded file, staging it through a temporary directory.

        The temp directory is created, the upload is written and read back
        through it, and it is removed before returning -- including on failure.
        Nothing is retained between requests and there is no database.
        """
        started = time.perf_counter()
        with self._temp_dir() as workdir:
            source_path = Path(workdir) / f"original{_ext_for(content_type)}"
            source_path.write_bytes(raw)
            logger.debug("temp.wrote path=%s bytes=%d", source_path, len(raw))

            loaded = load_image_from_bytes(
                source_path.read_bytes(),
                content_type=content_type,
                max_bytes=self._settings.max_upload_bytes,
            )
            logger.info(
                "protect.start width=%d height=%d format=%s bytes=%d",
                loaded.width,
                loaded.height,
                loaded.source_format,
                len(raw),
            )

            result = self._engine_protect(loaded.array)

            output_format = "JPEG" if loaded.source_format == "JPEG" else "PNG"
            protected_bytes = encode_image(
                result.image,
                output_format,
                jpeg_quality=self._settings.jpeg_quality,
            )
            # Round-trip through the encoder so the reported numbers describe
            # the bytes the client receives, not an in-memory ideal.
            delivered = load_image_from_bytes(protected_bytes)

        process_ms = (time.perf_counter() - started) * 1000.0
        delivered_metrics = compute_metrics(loaded.array, delivered.array)
        logger.info(
            "protect.done width=%d height=%d engine=%s device=%s attack_ms=%.1f "
            "process_ms=%.1f output_bytes=%d delivered_linf=%.1f/255 psnr=%.2fdB",
            delivered.width,
            delivered.height,
            result.meta.engine,
            result.meta.device,
            result.meta.attack_ms,
            process_ms,
            len(protected_bytes),
            delivered_metrics.linf_255,
            delivered_metrics.psnr_db,
        )
        return ProtectionArtifact(
            original=loaded.array,
            protected=delivered.array,
            protected_bytes=protected_bytes,
            media_type=output_media_type(output_format),
            filename=f"antifake-protected.{output_extension(output_format)}",
            width=delivered.width,
            height=delivered.height,
            source_format=loaded.source_format,
            output_format=output_format,
            process_ms=round(process_ms, 2),
            perturbation_metrics=result.metrics,
            delivered_metrics=delivered_metrics,
            engine_meta=result.meta,
        )

    def protect_array(self, image: RGB) -> ProtectionOutput:
        """Protect an in-memory array. No filesystem involved; used by tests."""
        return self._engine_protect(image)

    # -- internals ---------------------------------------------------------

    def _engine_protect(self, image: RGB) -> ProtectionOutput:
        protect_with_meta = getattr(self._engine, "protect_with_meta", None)
        if protect_with_meta is None:
            # A minimal ProtectionEngine implementation. Wrap it so callers get
            # the same richer result shape regardless of engine.
            protected = self._engine.protect(image)
            from app.protection.config import ProtectionResultMeta
            from app.protection.engine import ProtectionOutput as Output

            metrics = compute_metrics(image, protected)
            return Output(
                image=protected,
                meta=ProtectionResultMeta(
                    engine=self._engine.name,
                    target_model=self._engine.target_description,
                    device="cpu",
                    seed=0,
                    steps=0,
                    eps_255=metrics.linf_255,
                    linf_255=metrics.linf_255,
                    l2=metrics.l2,
                    psnr_db=metrics.psnr_db if metrics.psnr_db != float("inf") else 999.0,
                    ssim=metrics.ssim,
                    max_abs_pixel_delta=metrics.linf_8bit,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                    attack_ms=0.0,
                ),
                metrics=metrics,
            )
        result: ProtectionOutput = protect_with_meta(image)
        return result

    def _temp_dir(self) -> tempfile.TemporaryDirectory[str]:
        parent = self._settings.temp_dir
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(
            prefix="antifake-", dir=parent, ignore_cleanup_errors=False
        )


def _ext_for(content_type: str | None) -> str:
    return ".png" if (content_type or "").lower().endswith("png") else ".jpg"


def get_protection_service(
    engine: ProtectionEngine, settings: Settings
) -> ProtectionService:
    """Factory kept separate so wiring is obvious at the call site."""
    return ProtectionService(engine=engine, settings=settings)
