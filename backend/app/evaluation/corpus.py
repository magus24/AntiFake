"""The set of source images the Phase 2 comparison is run on.

Two rules govern this module, and both are about not fooling the report:

1. **Every image carries its provenance.** A synthetic image is labelled
   synthetic at construction time, not inferred later from its filename, so a
   document produced months from now still knows which rows came from a real
   photograph. The distinction matters because face and detection evidence is
   only meaningful on real photographs -- a detector reading a generated
   gradient says nothing about whether a protection hides a person.

2. **Eligibility is a property of the image, not of the caller.** Whether a
   face verdict is allowed comes from :attr:`CorpusImage.face_eligible`, so the
   benchmark cannot accidentally score a face on a synthetic tile because
   someone forgot a filter somewhere.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Provenance labels. Exactly two values; a third would mean the face rule
#: needs rethinking rather than another case.
PHOTOGRAPH = "photograph"
SYNTHETIC = "synthetic"

#: Environment variable that overrides where the corpus is looked up.
CORPUS_ENV_VAR = "ANTIFAKE_CORPUS_DIR"

#: Suffixes accepted as source images.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True, slots=True)
class CorpusImage:
    """One source image, with what is known about where it came from.

    Attributes:
        key: Stable identifier used in artifact keys and report rows.
        path: Where the bytes live.
        kind: :data:`PHOTOGRAPH` or :data:`SYNTHETIC`.
        notes: Free text for the report -- camera, crop, what it shows.
        face_eligible: Whether face evidence may be drawn from this image.
            Only real photographs qualify; see the module docstring.
    """

    key: str
    path: Path
    kind: str
    notes: str = ""
    face_eligible: bool = False

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping, without the pixels."""
        return {
            "key": self.key,
            "path": str(self.path),
            "kind": self.kind,
            "notes": self.notes,
            "face_eligible": self.face_eligible,
        }


def photograph(
    path: Path, *, key: str | None = None, notes: str = ""
) -> CorpusImage:
    """Declare a real photograph. Face evidence is allowed on it."""
    return CorpusImage(
        key=key or path.stem,
        path=path,
        kind=PHOTOGRAPH,
        notes=notes,
        face_eligible=True,
    )


def synthetic(path: Path, *, key: str | None = None, notes: str = "") -> CorpusImage:
    """Declare a generated image. Face evidence is forbidden on it."""
    return CorpusImage(
        key=key or path.stem,
        path=path,
        kind=SYNTHETIC,
        notes=notes,
        face_eligible=False,
    )


def face_eligible(corpus: Sequence[CorpusImage]) -> tuple[CorpusImage, ...]:
    """The subset a face verdict may be computed from."""
    return tuple(image for image in corpus if image.face_eligible)


def assert_paths_exist(corpus: Sequence[CorpusImage]) -> None:
    """Fail loudly on a missing file rather than halfway through a run."""
    missing = [str(image.path) for image in corpus if not image.path.is_file()]
    if missing:
        raise FileNotFoundError(
            "corpus image(s) not found: " + ", ".join(sorted(missing))
        )


def project_root() -> Path:
    """The repository root, derived from this file's own location.

    ``app/evaluation/corpus.py`` sits three levels below the project root, so
    the answer does not depend on the working directory the runner happened to
    be started in.
    """
    return Path(__file__).resolve().parents[3]


def _scan(directory: Path) -> list[Path]:
    """Image files directly inside ``directory``, sorted for reproducibility."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def default_corpus(root: Path | None = None) -> tuple[CorpusImage, ...]:
    """The corpus a normal run uses.

    ``$ANTIFAKE_CORPUS_DIR`` wins when set, and every file in it is treated as a
    photograph -- it is a directory you pointed the tool at, and declaring it
    synthetic would be inventing information. Otherwise the project root is
    scanned for captured photographs.

    An empty result is a valid answer; :func:`assert_paths_exist` is what
    turns it into a clear error at the point of use.
    """
    override = os.environ.get(CORPUS_ENV_VAR)
    if override:
        return tuple(photograph(path) for path in _scan(Path(override)))

    base = root if root is not None else project_root()
    captured = [
        path
        for path in _scan(base)
        if path.name.lower().startswith("photo_")
    ]
    return tuple(photograph(path) for path in captured)


def corpus_from_paths(paths: Iterable[Path]) -> tuple[CorpusImage, ...]:
    """Wrap explicit paths as photographs, de-duplicated by resolved path.

    For callers that already know exactly what they want to measure. Provenance
    is declared by the caller here, which is why the default corpus exists: a
    command line flag is a worse place to encode "this one is synthetic".
    """
    seen: dict[Path, CorpusImage] = {}
    for path in paths:
        resolved = Path(path)
        if resolved not in seen:
            seen[resolved] = photograph(resolved)
    return tuple(seen.values())
