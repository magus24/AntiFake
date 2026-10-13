"""AntiFake backend application factory.

Wiring only. The target model is loaded once during startup so the first
request does not pay a 45 MB download, and so a broken model configuration
fails loudly at boot rather than on a user's upload.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.protect import router as protect_router
from app.core.config import Settings, get_settings
from app.core.logging_config import configure_logging
from app.protection.config import ProtectionConfig
from app.protection.engine import AdversarialProtectionEngine
from app.services.protection_service import ProtectionService

logger = logging.getLogger(__name__)

DESCRIPTION = """
AntiFake MVP: adds a bounded, reproducible adversarial perturbation to a single
JPG/PNG so that **one specific** AI pipeline (torchvision ResNet-18 / ImageNet-1k
with resize-256 centre-crop-224 preprocessing) can no longer read the image
confidently.

This is deliberately narrow. It is not a general defence against AI-generated
or AI-detected media, and results against other models are unknown.
"""


def build_protection_config(settings: Settings) -> ProtectionConfig:
    """Map process settings onto the engine's own configuration object.

    ``eps`` and ``alpha`` are derived rather than configured separately, so an
    operator cannot accidentally set a step size larger than the perturbation
    budget and turn the projection into a no-op.
    """
    eps = settings.default_eps_255 / 255.0
    return ProtectionConfig.create(
        steps=settings.default_steps,
        eps=eps,
        alpha=eps / 4.0,
        seed=settings.default_seed,
        device=settings.default_device,
        model_weights=settings.model_weights,
    )


def create_app(
    *,
    service: ProtectionService | None = None,
    load_model: bool = True,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        service: Pre-built service. Tests inject one backed by randomly
            initialised weights so they need neither the network nor 45 MB.
        load_model: When ``False``, no engine is built at startup. Only used by
            tests that assert on a missing service.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None and load_model:
            config = build_protection_config(settings)
            target = settings.target_model_config()
            logger.info(
                "startup.loading_model model=%s weights=%s device=%s "
                "steps=%d eps=%.2f/255",
                target.name,
                target.weights,
                config.device,
                config.steps,
                config.eps_255,
            )
            engine = AdversarialProtectionEngine(config, target)
            app.state.service = ProtectionService(engine=engine, settings=settings)
        elif service is not None:
            app.state.service = service
        yield
        app.state.service = None

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.include_router(protect_router)

    @app.get("/", tags=["meta"], include_in_schema=False)
    def index() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
            "protect": "POST /api/protect",
        }

    return app


app = create_app()
