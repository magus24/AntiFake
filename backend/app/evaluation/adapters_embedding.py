"""CLIP as a benchmark adapter: embeddings and text-image similarity.

This is the adapter that answers the question the classification models cannot.
A vision-language assistant does not emit a class label; it ranks candidate
meanings of a picture. Measuring cosine similarity between image embeddings,
and the ranking of a prompt set against those embeddings, is the closest local
proxy for "can a multimodal model still tell what this is".

It is deliberately the *only* place open_clip is imported, so the rest of the
benchmark -- and every test that does not need CLIP -- runs without the
dependency being present.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.evaluation.base import EmbeddingObservation, ModelAdapter, ModelUnavailableError
from app.protection.base import RGB

logger = logging.getLogger(__name__)

#: Default probe set. Deliberately broad: a protection that only survives
#: coarse captions but loses fine ones is not protecting anything, and a set
#: of near-synonyms would hide that.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "a person",
    "a car",
    "a dog",
    "a cat",
    "a building",
    "a street",
    "a landscape",
    "a tree",
    "food",
    "a room",
    "a phone",
    "a computer",
    "a book",
    "a flower",
    "a bicycle",
    "a boat",
)

#: The Phase 2 probe set: everything in :data:`DEFAULT_PROMPTS`, plus the
#: finer-grained questions a protection aimed at a *person* has to survive.
#:
#: It is a separate constant rather than an extension of the default one on
#: purpose. Phase 1 is expected to keep reproducing its saved baseline byte for
#: byte, and adding a prompt to the default set would silently rewrite every
#: CLIP row that suite has ever produced -- including the ones already recorded
#: as the reference. The Phase 2 runner passes this tuple explicitly.
EXTENDED_PROMPTS: tuple[str, ...] = (
    *DEFAULT_PROMPTS,
    "a person wearing a suit",
    "a necktie",
    "a close-up portrait of a person",
    "an indoor room",
    "a group of people",
    "a hand holding an object",
    "a vehicle interior",
    "text on a sign",
    "a computer screen",
    "a document or paper",
    "an outdoor street",
    "a person's face",
)

#: CLIP's own normalisation, which is not the ImageNet one.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class ClipAdapter(ModelAdapter):
    """An open_clip model behind the benchmark interface.

    Args:
        device: Where to place the weights.
        model_name: open_clip architecture name, e.g. ``"ViT-B-32"``.
        pretrained: open_clip pretrained tag, e.g. ``"laion2b_s34b_b79k"``.
        prompts: Probe set for text-image similarity.
    """

    family = "embedding"

    def __init__(
        self,
        device: torch.device,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    ) -> None:
        """Declare the model without loading it."""
        super().__init__(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.prompts = tuple(prompts)
        self.name = f"clip_{model_name.lower().replace('-', '_')}"
        self.description = f"open_clip:{model_name} (pretrained={pretrained})"
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_embeddings: torch.Tensor | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Build the model, tokenizer and preprocessing, and pre-encode prompts."""
        if self._loaded:
            return
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                self.name,
                "open_clip_torch is not installed (pip install open_clip_torch)",
            ) from exc

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
        except Exception as exc:
            raise ModelUnavailableError(
                self.name,
                f"could not load {self.model_name}/{self.pretrained}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

        with torch.no_grad():
            tokens = self._tokenizer(list(self.prompts)).to(self.device)
            self._text_embeddings = _normalise(model.encode_text(tokens))
        self._loaded = True
        logger.info(
            "adapter.loaded name=%s pretrained=%s device=%s prompts=%d",
            self.name,
            self.pretrained,
            self.device,
            len(self.prompts),
        )

    def release(self) -> None:
        """Delete the model and cached text embeddings, and free the device."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_embeddings = None
        self._loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------

    def preprocess(self, image: RGB) -> torch.Tensor:
        """Run the model's own declared image transform.

        Uses open_clip's transform rather than a hand-rolled resize so the
        benchmark measures the model as it is actually deployed, including its
        bicubic resize and its own normalisation.
        """
        pil = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        return self._preprocess(pil).unsqueeze(0).to(self.device)

    def embed(self, image: RGB) -> torch.Tensor:
        """L2-normalised image embedding, shape ``(1, D)``."""
        return self._embed_normalised(image)

    def evaluate(self, image: RGB) -> EmbeddingObservation:
        """Embed one image and score it against the prompt set."""
        embedding = self._embed_normalised(image)
        # Both sides are unit vectors, so a dot product is cosine similarity.
        similarities = (self._text() @ embedding.T).squeeze(1)
        pairs = list(zip(self.prompts, similarities.tolist(), strict=True))
        ranking = [p for p, _ in sorted(pairs, key=lambda kv: kv[1], reverse=True)]
        return EmbeddingObservation(
            embedding=embedding.squeeze(0).cpu().numpy().astype(np.float32),
            text_similarities={p: float(s) for p, s in pairs},
            text_ranking=ranking,
        )

    def _embed_normalised(self, image: RGB) -> torch.Tensor:
        with torch.no_grad():
            return _normalise(self._net().encode_image(self.preprocess(image)))

    def _net(self) -> Any:
        """The loaded network, or a loud error. Narrows the type for mypy."""
        self._require_loaded()
        if self._model is None:
            raise RuntimeError(f"{self.name}: loaded but the model is missing")
        return self._model

    def _text(self) -> torch.Tensor:
        """Cached prompt embeddings, or a loud error."""
        self._require_loaded()
        if self._text_embeddings is None:
            raise RuntimeError(f"{self.name}: loaded but prompt embeddings are missing")
        return self._text_embeddings


def _normalise(tensor: torch.Tensor) -> torch.Tensor:
    """Project onto the unit sphere along the feature axis."""
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_clip(
    device: torch.device,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> ClipAdapter:
    """Construct a declared-but-unloaded CLIP adapter.

    An empty prompt list falls back to :data:`DEFAULT_PROMPTS` rather than
    producing a model that can be evaluated but reports no meaning at all.
    """
    return ClipAdapter(
        device,
        model_name=model_name,
        pretrained=pretrained,
        prompts=tuple(prompts) if prompts else DEFAULT_PROMPTS,
    )
