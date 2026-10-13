"""The contract every protection engine implements.

Kept tiny on purpose. An engine receives a decoded RGB image and returns a
decoded RGB image. It knows nothing about HTTP, files, or numpy-vs-PIL
conventions beyond the two documented below.
"""

from __future__ import annotations

import abc

import numpy as np

RGB = np.ndarray
"""A ``uint8`` array of shape ``(H, W, 3)`` in RGB channel order."""


class ProtectionEngine(abc.ABC):
    """Applies an image protection transform.

    Implementations must honour these invariants:

    * The returned array is ``uint8`` with shape ``(H, W, 3)``.
    * Output height and width match the input unless the engine documents
      otherwise via :attr:`preserves_dimensions` being ``False``.
    * The output lives in the same intensity range as the input, so encoding
      the result never clips or wraps.
    """

    #: Human-readable engine identifier, surfaced in API responses and logs.
    name: str = "abstract"

    #: Pipeline the perturbation is computed against, for honest reporting.
    target_description: str = "unspecified"

    #: ``True`` when the engine guarantees ``(H, W)`` is unchanged.
    preserves_dimensions: bool = True

    @abc.abstractmethod
    def protect(self, image: RGB) -> RGB:
        """Return a protected copy of ``image``.

        Args:
            image: ``uint8`` RGB array of shape ``(H, W, 3)``.

        Returns:
            A ``uint8`` RGB array of shape ``(H, W, 3)``.

        Raises:
            ValueError: If ``image`` is not a valid ``uint8`` RGB array.
        """

    def describe(self) -> dict[str, str]:
        """Static description of the engine, used for API metadata."""
        return {
            "engine": self.name,
            "target_model": self.target_description,
            "preserves_dimensions": str(self.preserves_dimensions),
        }
