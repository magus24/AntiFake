"""Protection engine package.

The rest of the application depends only on :class:`ProtectionEngine`, so
swapping the algorithm means implementing one method.
"""

from app.protection.base import RGB, ProtectionEngine
from app.protection.config import (
    ProtectionConfig,
    ProtectionConfigError,
    ProtectionResultMeta,
    TargetModelConfig,
)
from app.protection.engine import AdversarialProtectionEngine, ProtectionOutput
from app.protection.image_io import (
    ImageValidationError,
    LoadedImage,
    encode_image,
    load_image_from_bytes,
    load_image_from_path,
    output_extension,
    output_media_type,
)
from app.protection.metrics import (
    DifferenceMetrics,
    compute_metrics,
    compute_psnr,
    compute_ssim,
)

__all__ = [
    "RGB",
    "AdversarialProtectionEngine",
    "DifferenceMetrics",
    "ImageValidationError",
    "LoadedImage",
    "ProtectionConfig",
    "ProtectionConfigError",
    "ProtectionEngine",
    "ProtectionOutput",
    "ProtectionResultMeta",
    "TargetModelConfig",
    "compute_metrics",
    "compute_psnr",
    "compute_ssim",
    "encode_image",
    "load_image_from_bytes",
    "load_image_from_path",
    "output_extension",
    "output_media_type",
]
