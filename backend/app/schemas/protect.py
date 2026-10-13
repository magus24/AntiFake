"""Pydantic response models for the protect endpoint.

Separate from the protection engine on purpose: the engine should not grow an
opinion about wire formats.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DifferenceMetricsOut(BaseModel):
    """Difference/similarity numbers between the original and protected image."""

    model_config = ConfigDict(extra="forbid")

    linf_255: float = Field(description="Max per-channel difference in 8-bit levels")
    linf_8bit: int = Field(description="Same value rounded to an integer")
    l2: float = Field(description="RMS difference on a 0-255 scale")
    mean_abs: float = Field(description="Mean absolute difference on a 0-255 scale")
    psnr_db: float = Field(description="PSNR in dB; 999 when the images are identical")
    ssim: float = Field(description="Structural similarity in [-1, 1]")


class EngineInfoOut(BaseModel):
    """Static description of the protection engine and its target."""

    model_config = ConfigDict(extra="forbid")

    engine: str
    target_model: str
    preserves_dimensions: bool
    device: str
    eps_255: float
    steps: int


class ProtectResponseOut(BaseModel):
    """JSON metadata returned alongside the protected image bytes."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    media_type: str
    width: int
    height: int
    source_format: str
    output_format: str
    process_ms: float
    perturbation_metrics: DifferenceMetricsOut
    delivered_metrics: DifferenceMetricsOut
    engine: EngineInfoOut
    disclaimer: str


class HealthResponseOut(BaseModel):
    """Liveness plus enough engine detail to confirm which pipeline is loaded."""

    model_config = ConfigDict(extra="forbid")

    status: str
    app: str
    version: str
    engine: EngineInfoOut
    supported_formats: list[str]
    max_upload_bytes: int
