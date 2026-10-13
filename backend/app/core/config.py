"""Application settings.

Environment variables use the ``ANTIFAKE_`` prefix. Values are read once at
import time; nothing here imports the protection engine, so configuration
stays independent of the algorithm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from app.protection.config import TargetModelConfig


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class Settings:
    """Process-wide configuration."""

    app_name: str = field(
        default_factory=lambda: _env_str("ANTIFAKE_APP_NAME", "AntiFake API")
    )
    version: str = field(
        default_factory=lambda: _env_str("ANTIFAKE_VERSION", "0.1.0")
    )
    log_level: str = field(default_factory=lambda: _env_str("ANTIFAKE_LOG_LEVEL", "INFO"))

    # Upload limits
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("ANTIFAKE_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    )

    # Protection defaults
    default_eps_255: float = field(
        default_factory=lambda: _env_float("ANTIFAKE_EPS_255", 8.0)
    )
    default_steps: int = field(
        default_factory=lambda: _env_int("ANTIFAKE_STEPS", 40)
    )
    default_seed: int = field(
        default_factory=lambda: _env_int("ANTIFAKE_SEED", 0)
    )
    default_device: str = field(
        default_factory=lambda: _env_str("ANTIFAKE_DEVICE", "auto")
    )
    jpeg_quality: int = field(
        default_factory=lambda: _env_int("ANTIFAKE_JPEG_QUALITY", 95)
    )

    # Target model identity
    model_name: str = field(
        default_factory=lambda: _env_str("ANTIFAKE_MODEL_NAME", "resnet18")
    )
    model_weights: str = field(
        default_factory=lambda: _env_str("ANTIFAKE_MODEL_WEIGHTS", "imagenet1k_v1")
    )

    # Temporary files
    temp_dir: str | None = field(
        default_factory=lambda: _env_str("ANTIFAKE_TEMP_DIR", "") or None
    )
    keep_temp_files: bool = field(
        default_factory=lambda: _env_bool("ANTIFAKE_KEEP_TEMP_FILES", False)
    )

    # CORS origins for the dev frontend
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in _env_str(
                "ANTIFAKE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        )
    )

    def target_model_config(self) -> TargetModelConfig:
        """Project the settings onto the protection layer's model description."""
        return TargetModelConfig(name=self.model_name, weights=self.model_weights)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
