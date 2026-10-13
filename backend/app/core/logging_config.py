"""Logging setup.

The API contract is: one line per request with method, path, status, duration,
and the processing diagnostics that matter (dimensions, perturbation size,
device). Anything noisier belongs at DEBUG and is off by default.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler on the root logger.

    Safe to call more than once: existing handlers are replaced, not stacked.
    """
    root = logging.getLogger()
    resolved = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(resolved)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    # uvicorn installs its own handlers; let records propagate to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    # Third-party noise we do not need.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
