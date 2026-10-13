"""Vision-language model as a benchmark adapter.

A VLM answers questions about a picture in words, which is the form of reading
this benchmark has not had until now. It is also the one branch that is
expected to come back ``skipped`` on most machines: there is no single correct
local model to ship, and inventing a stand-in that confidently reports an answer
it never produced would be worse than reporting nothing.

So the adapter is an abstraction first. It defines what is measured -- a fixed
question set asked of the same image before and after protection -- and refuses
to measure anything unless a real :class:`VlmBackend` is handed to it. Without
one, :meth:`VlmAdapter.load` raises :class:`ModelUnavailableError`, which the
registry records as ``status="skipped"`` with the reason attached. That is the
honest outcome, and the report shows it as such.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np
from PIL import Image

from app.evaluation.base import ModelAdapter, ModelUnavailableError, TextObservation
from app.protection.base import RGB

logger = logging.getLogger(__name__)

#: The question set. Deliberately mixed: identity, content, detail and the
#: presence question. A protection that only defeats "what is this" while
#: "what is the person wearing" still answers correctly has not hidden
#: anything that matters.
DEFAULT_VLM_PROMPTS: tuple[str, ...] = (
    "What is shown in this image?",
    "Describe the main subject of this image.",
    "What is the person wearing?",
    "Is there a person in this image?",
    "What colors are most prominent?",
    "Where was this photo likely taken?",
)


class VlmBackend(abc.ABC):
    """How a VLM is actually reached.

    Implementing this is the only way to give the VLM branch a model. The
    benchmark never picks one on your behalf.

    Attributes:
        name: Human-readable model description for the report.
    """

    name: str = "unnamed backend"

    @abc.abstractmethod
    def load(self) -> None:
        """Bring the model up. Called at most once per adapter run."""

    @abc.abstractmethod
    def generate(self, image: Image.Image, prompt: str) -> str:
        """Answer one question about one image."""

    def release(self) -> None:  # noqa: B027 - deliberately optional
        """Tear the model down. Default is a no-op; override if it holds VRAM."""


class VlmAdapter(ModelAdapter):
    """Ask a :class:`VlmBackend` the same questions before and after protection.

    Args:
        device: Declared for interface symmetry; the backend owns its own device.
        backend: The model to interrogate. ``None`` means the branch reports
            ``skipped`` rather than guessing.
        prompts: Question set.
    """

    family = "text"

    def __init__(
        self,
        device: Any,
        *,
        backend: VlmBackend | None = None,
        prompts: tuple[str, ...] = DEFAULT_VLM_PROMPTS,
    ) -> None:
        """Declare the adapter without loading anything."""
        super().__init__(device)
        self.backend = backend
        self.prompts = tuple(prompts) or DEFAULT_VLM_PROMPTS
        self.name = "vlm"
        self.description = (
            f"VLM ({backend.name})" if backend is not None else "VLM (not configured)"
        )

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the backend, or report the branch as unavailable.

        The reason is specific on purpose: "not configured" and "not installed"
        are different problems for a reader trying to reproduce the run.
        """
        if self._loaded:
            return
        if self.backend is None:
            raise ModelUnavailableError(
                self.name,
                "no VLM backend configured; pass a VlmBackend to use this branch",
            )
        try:
            self.backend.load()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                self.name,
                f"backend {self.backend.name} failed to load: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        self._loaded = True
        logger.info("adapter.loaded name=%s backend=%s", self.name, self.backend.name)

    def release(self) -> None:
        """Release the backend and forget it ran."""
        if self.backend is not None:
            self.backend.release()
        self._loaded = False

    # -- inference ---------------------------------------------------------

    def evaluate(self, image: RGB) -> TextObservation:
        """Ask every prompt and keep the answers verbatim."""
        if self.backend is None:
            # Unreachable in practice -- ``load`` fails first -- but returning a
            # reason beats raising mid-run if a caller ever forces it.
            return TextObservation(
                answers={}, model="", reason="no VLM backend configured"
            )
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        answers: dict[str, str] = {}
        for prompt in self.prompts:
            try:
                answer = self.backend.generate(pil, prompt)
            except Exception as exc:
                answers[prompt] = f"<error: {type(exc).__name__}: {exc}>"
            else:
                answers[prompt] = str(answer).strip()
        return TextObservation(answers=answers, model=self.backend.name)


def build_vlm(
    device: Any,
    *,
    backend: VlmBackend | None = None,
    prompts: tuple[str, ...] = DEFAULT_VLM_PROMPTS,
) -> VlmAdapter:
    """Construct a declared-but-unloaded VLM adapter.

    Without a ``backend`` this produces an adapter that reports ``skipped`` with
    a stated reason, which is the intended default for a run with no VLM.
    """
    return VlmAdapter(device, backend=backend, prompts=prompts)
