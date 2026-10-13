"""Turning a pair of observations into a transfer verdict.

Raw probabilities invite the wrong conclusion. A confidence falling from
0.34 to 0.008 and one falling from 0.34 to 0.30 are the same number of
characters apart and are not remotely the same event, so every comparison here
is expressed in units that say which of the two happened:

* absolute confidence drop, which shows the size of the change;
* *relative* drop, which is the only honest way to score a model whose
  confidence was already low -- a 90% drop from 0.02 is a fluke being fixed,
  not an attack succeeding;
* label change, and how much of the top-5 survives, which separate "the
  picture now looks like something else" from "the model's whole ranking
  collapsed";
* cosine similarity of embeddings, which is the only scale-free way to ask
  whether a vision-language model still sees the same thing.

Held-out discipline
-------------------
A per-model number is only half the answer; the other half is *which models are
allowed to add up*. :func:`transfer_summary` keeps the optimised-against models
in their own bucket and computes the headline rate over the held-out set alone,
because averaging the two together is exactly how an ensemble attack ends up
reporting success without having transferred.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.evaluation.base import (
    ClassificationObservation,
    Detection,
    DetectionObservation,
    EmbeddingObservation,
    FaceObservation,
    ModelFamily,
    ModelRole,
    Observation,
    TextObservation,
)


def _top5_overlap(
    original: ClassificationObservation, protected: ClassificationObservation
) -> float:
    """Fraction of the original top-5 still in the protected top-5."""
    original_ids = {index for index, _ in original.top5}
    protected_ids = {index for index, _ in protected.top5}
    return len(original_ids & protected_ids) / max(1, len(original_ids))


def compare_classification(
    original: ClassificationObservation, protected: ClassificationObservation
) -> dict[str, Any]:
    """Transfer verdict for one classifier.

    Attributes:
        confidence_drop: ``original - protected``. Positive means the attack
            reduced confidence.
        relative_drop: Drop as a fraction of the original confidence, or
            ``None`` when the original was effectively zero and a ratio would
            be meaningless.
        confidence_collapsed: Confidence fell by at least 75% of its original
            value, *whether or not* the label changed.
        target_disrupted: The strict verdict: the label changed **and**
            confidence fell by at least 75%.
        entropy_rise: How much more the answer spread out. A large rise with
            no label change is the signature of a successful attack that left
            the argmax standing: the model no longer believes anything, but the
            winner of a near-uniform distribution is arbitrary.

    Both verdicts are reported because they answer different questions, and
    reporting only the strict one hides a real effect. A model going from 34%
    "suit" to 1.2% "suit" has been successfully neutralised while still
    technically naming the same class, and a report that only counted label
    flips would call that a failure.
    """
    drop = original.top1_probability - protected.top1_probability
    if original.top1_probability > 1e-9:
        relative = drop / original.top1_probability
    else:
        relative = None
    label_changed = original.top1_index != protected.top1_index
    collapsed = bool(relative is not None and relative >= 0.75)
    return {
        "original_top1_index": original.top1_index,
        "original_top1_label": original.top1_label,
        "original_top1_probability": round(original.top1_probability, 6),
        "original_entropy": round(original.entropy, 6),
        "protected_top1_index": protected.top1_index,
        "protected_top1_label": protected.top1_label,
        "protected_top1_probability": round(protected.top1_probability, 6),
        "protected_entropy": round(protected.entropy, 6),
        "label_changed": label_changed,
        "confidence_drop": round(drop, 6),
        "relative_drop": None if relative is None else round(relative, 6),
        "entropy_rise": round(protected.entropy - original.entropy, 6),
        "top5_overlap": round(_top5_overlap(original, protected), 6),
        "confidence_collapsed": collapsed,
        "target_disrupted": bool(label_changed and collapsed),
    }


def compare_embedding(
    original: EmbeddingObservation, protected: EmbeddingObservation
) -> dict[str, Any]:
    """Transfer verdict for one embedding model.

    Both embeddings are unit vectors, so ``dot`` is cosine similarity and
    ``norm(a - b)`` equals ``sqrt(2 - 2 * cos)``; the explicit difference is
    used because it stays correct if a future adapter returns unnormalised
    vectors.

    ``text_regression_count`` counts prompts that lost ground while competing
    for the top answer. A model can keep every absolute similarity high and
    still change its verdict, and this is the number that catches that. Only
    the original top two are considered: a prompt far down the ranking losing
    a fraction of a percent did not change the model's reading of the image.
    """
    cosine = float(np.dot(original.embedding, protected.embedding))
    l2 = float(np.linalg.norm(original.embedding - protected.embedding))

    deltas = {
        prompt: round(
            protected.text_similarities.get(prompt, 0.0)
            - original.text_similarities.get(prompt, 0.0),
            6,
        )
        for prompt in sorted(original.text_similarities)
    }
    original_best = original.text_ranking[0] if original.text_ranking else None
    protected_best = protected.text_ranking[0] if protected.text_ranking else None
    original_order = {p: i for i, p in enumerate(original.text_ranking)}

    # Only the contest for the *best* answer counts as a regression. A prompt
    # sitting fifteenth that loses 0.001 has not degraded the model's reading
    # of the image, and counting it would drown the rows that did.
    regressions = sum(
        1
        for prompt, delta in deltas.items()
        if delta < 0 and original_order.get(prompt, len(deltas)) < 2
    )
    return {
        "embedding_dim": original.dimension,
        "cosine_similarity": round(cosine, 6),
        "embedding_l2": round(l2, 6),
        "original_best_prompt": original_best,
        "protected_best_prompt": protected_best,
        "best_prompt_changed": original_best != protected_best,
        "best_prompt_similarity_original": round(
            original.text_similarities.get(original_best or "", 0.0), 6
        ),
        "best_prompt_similarity_protected": round(
            protected.text_similarities.get(protected_best or "", 0.0), 6
        ),
        "text_similarity_deltas": deltas,
        # Positive means the prompt moved *down* the ranking, which is worse.
        "rank_deltas": {
            prompt: protected.text_ranking.index(prompt) - original_order[prompt]
            if prompt in protected.text_ranking
            else None
            for prompt in sorted(original.text_similarities)
        },
        "text_regression_count": regressions,
    }


#: Cosine similarity at or above which a face descriptor is called unchanged.
#:
#: This is a reporting threshold, not a claim about any vendor's verification
#: policy. It is chosen as the midpoint of the meaningful range so that a
#: descriptor that has been rotated past 60 degrees -- far beyond what any
#: matcher would tolerate -- is the one that trips it, and the raw cosine is
#: always reported alongside so a reader can apply their own cut.
FACE_IDENTITY_THRESHOLD = 0.5

#: Minimum IoU for two boxes to count as the same object across the pair.
DETECTION_IOU_THRESHOLD = 0.5


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection) / union if union > 0 else 0.0


def compare_face(original: FaceObservation, protected: FaceObservation) -> dict[str, Any]:
    """Transfer verdict for the face pipeline: detection *and* identity.

    The two are reported separately because they fail differently, and only one
    of them is what most people mean by "the face disappeared". A box that is
    still there with a descriptor that has rotated halfway around the identity
    space is a recognition failure hiding behind a passing detector; a box that
    vanished with the descriptor still close to the original would be the
    opposite. Collapsing either into a single number would make a protection
    look half as good as it is, or twice.

    The box shift is reported in pixels rather than IoU alone because IoU is
    insensitive to movement on large boxes and wildly sensitive on small ones --
    a 51x67 face and a 500x500 face can lose the same IoU from very different
    amounts of motion.
    """
    cosine: float | None = None
    l2: float | None = None
    if original.embedding is not None and protected.embedding is not None:
        cosine = round(float(np.dot(original.embedding, protected.embedding)), 6)
        l2 = round(float(np.linalg.norm(original.embedding - protected.embedding)), 6)

    iou: float | None = None
    shift: float | None = None
    if original.box is not None and protected.box is not None:
        iou = round(_box_iou(original.box, protected.box), 6)
        shift = round(
            float(
                np.hypot(
                    (original.box[0] + original.box[2])
                    / 2
                    - (protected.box[0] + protected.box[2]) / 2,
                    (original.box[1] + original.box[3])
                    / 2
                    - (protected.box[1] + protected.box[3]) / 2,
                )
            ),
            6,
        )

    return {
        "face_detected_original": original.detected,
        "face_detected": protected.detected,
        "face_lost": original.detected and not protected.detected,
        "confidence_original": original.confidence,
        "confidence_protected": protected.confidence,
        "confidence_delta": None
        if original.confidence is None or protected.confidence is None
        else round(protected.confidence - original.confidence, 6),
        "box_original": None if original.box is None else list(original.box),
        "box_protected": None if protected.box is None else list(protected.box),
        "box_iou": iou,
        "box_center_shift": shift,
        "face_embedding_cosine": cosine,
        "face_embedding_l2": l2,
        "face_identity_preserved": None
        if cosine is None
        else cosine >= FACE_IDENTITY_THRESHOLD,
        "reason": protected.reason,
    }


def compare_detection(
    original: DetectionObservation, protected: DetectionObservation
) -> dict[str, Any]:
    """Transfer verdict for the object detector.

    Counts, labels and confidence answer three different questions and none of
    them substitutes for another. A detector that drops the count to zero has
    been defeated outright; one that keeps every label at lower confidence has
    only been made less sure, which is a real but much smaller effect; and one
    that keeps the count while swapping a label for something else has been
    fooled in the way a reader of the report would actually care about.
    """
    original_labels = [d.label for d in original.detections]
    protected_labels = [d.label for d in protected.detections]

    # Matched pairs are what a confidence comparison has to be built on.
    # Comparing the two means over their own sets would credit a confidence
    # change that is really the count change: dropping the weaker of two boxes
    # raises the surviving mean by 0.15 while not one detection got surer.
    pairs: list[tuple[Detection, Detection]] = []
    for left in original.detections:
        for right in protected.detections:
            if left.label != right.label:
                continue
            if _box_iou(left.box, right.box) >= DETECTION_IOU_THRESHOLD:
                pairs.append((left, right))
                break

    def _mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    confidence_deltas = [right.confidence - left.confidence for left, right in pairs]

    return {
        "detection_count_original": len(original.detections),
        "detection_count_protected": len(protected.detections),
        "detection_count_delta": len(protected.detections) - len(original.detections),
        "detection_labels_original": original_labels,
        "detection_labels_protected": protected_labels,
        "detection_labels_changed": sorted(original_labels) != sorted(protected_labels),
        "detection_lost_labels": sorted(set(original_labels) - set(protected_labels)),
        "detection_new_labels": sorted(set(protected_labels) - set(original_labels)),
        "detection_matched": len(pairs),
        "detection_match_rate": round(len(pairs) / len(original.detections), 6)
        if original.detections
        else None,
        # Each mean is over its own detection set, so only the paired delta
        # below isolates a confidence change from a count change.
        "detection_mean_confidence_original": _mean(
            [d.confidence for d in original.detections]
        ),
        "detection_mean_confidence_protected": _mean(
            [d.confidence for d in protected.detections]
        ),
        "detection_confidence_delta": _mean(confidence_deltas),
    }


def _normalise_answer(text: str) -> str:
    """Case-fold, strip and collapse whitespace.

    A generated answer is free text, so an exact comparison would report a
    change for "a man in a suit" -> "A man in a suit." and miss nothing
    important, while a comparison that ignored punctuation entirely would stop
    noticing refusals and hedges. Only the whitespace and case noise is
    removed; everything the model actually said is kept.
    """
    return " ".join(text.split()).casefold()


def compare_text(
    original: TextObservation, protected: TextObservation
) -> dict[str, Any]:
    """Transfer verdict for a vision-language model.

    There is no probability to threshold here, so the event of interest is
    simply whether the answer changed, reported per prompt. Reporting the
    per-prompt booleans as well as the row-level flag matters because "the
    model changed one of six answers" and "the model changed all six" both read
    as ``changed: True`` at the row level.
    """
    prompts = sorted(set(original.answers) | set(protected.answers))
    per_prompt = {
        prompt: _normalise_answer(original.answers.get(prompt, ""))
        != _normalise_answer(protected.answers.get(prompt, ""))
        for prompt in prompts
    }
    changed = [prompt for prompt, changed_flag in per_prompt.items() if changed_flag]
    return {
        "text_answers_total": len(prompts),
        "text_answers_changed": bool(changed),
        "text_answers_changed_count": len(changed),
        "text_changed_prompts": changed,
        "text_model": protected.model or original.model,
        "text_reason": protected.reason,
    }


def compare(
    original: Observation, protected: Observation
) -> dict[str, Any]:
    """Dispatch to the right comparison for the observation family.

    A mismatch is a ``TypeError`` rather than a silently skipped row: the two
    sides always come from the same registry entry, so a family mismatch means
    the evaluation is comparing two different models and every number derived
    from it would be meaningless.
    """
    if isinstance(original, ClassificationObservation) and isinstance(
        protected, ClassificationObservation
    ):
        return compare_classification(original, protected)
    if isinstance(original, EmbeddingObservation) and isinstance(
        protected, EmbeddingObservation
    ):
        return compare_embedding(original, protected)
    if isinstance(original, FaceObservation) and isinstance(
        protected, FaceObservation
    ):
        return compare_face(original, protected)
    if isinstance(original, DetectionObservation) and isinstance(
        protected, DetectionObservation
    ):
        return compare_detection(original, protected)
    if isinstance(original, TextObservation) and isinstance(
        protected, TextObservation
    ):
        return compare_text(original, protected)
    raise TypeError(
        f"Mismatched observation families: {type(original).__name__} vs "
        f"{type(protected).__name__}"
    )


def _bucket(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts and rates for one already-filtered group of models."""
    total = len(comparisons)
    collapsed = sum(
        1 for c in comparisons.values() if c.get("confidence_collapsed")
    )
    disrupted = sum(1 for c in comparisons.values() if c.get("target_disrupted"))
    changed = sum(1 for c in comparisons.values() if c.get("label_changed"))
    relatives = [
        c["relative_drop"]
        for c in comparisons.values()
        if c.get("relative_drop") is not None
    ]
    return {
        "evaluated": total,
        "confidence_collapsed": collapsed,
        "target_disrupted": disrupted,
        "label_changed": changed,
        "collapse_rate": round(collapsed / total, 6) if total else None,
        "disruption_rate": round(disrupted / total, 6) if total else None,
        "mean_relative_drop": round(sum(relatives) / len(relatives), 6)
        if relatives
        else None,
        "max_relative_drop": round(max(relatives), 6) if relatives else None,
        "min_relative_drop": round(min(relatives), 6) if relatives else None,
    }


def _evaluator_bucket(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A separate summary for the vision-language probe.

    CLIP produces no confidence and no label, so the classification counts would
    read as an all-zero verdict for a model that may well have changed its mind.
    Its evidence is whether the best text answer moved and how many prompts lost
    ground while competing for the top slot, which is reported here rather than
    squeezed into the classification shape.
    """
    total = len(comparisons)
    changed = sum(1 for c in comparisons.values() if c.get("best_prompt_changed"))
    regressions = [
        c.get("text_regression_count", 0) for c in comparisons.values()
    ]
    cosines = [
        c["cosine_similarity"]
        for c in comparisons.values()
        if isinstance(c.get("cosine_similarity"), (int, float))
    ]
    return {
        "evaluated": total,
        "best_prompt_changed": changed,
        "prompt_change_rate": round(changed / total, 6) if total else None,
        "mean_text_regression": round(sum(regressions) / len(regressions), 6)
        if regressions
        else None,
        "max_text_regression": max(regressions) if regressions else None,
        "mean_cosine_similarity": round(sum(cosines) / len(cosines), 6)
        if cosines
        else None,
        "min_cosine_similarity": round(min(cosines), 6) if cosines else None,
    }


def _face_bucket(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Face evidence: did the detector still find it, and did the identity move.

    Two independent questions. A face that disappears is a detection failure; one
    that stays but whose descriptor rotates is a recognition failure. Reporting
    only the first would grade a protection as weak for the very property it is
    strongest at.
    """
    total = len(comparisons)
    lost = sum(1 for c in comparisons.values() if c.get("face_lost"))
    still_detected = [c for c in comparisons.values() if c.get("face_detected")]
    cosines = [
        c["face_embedding_cosine"]
        for c in comparisons.values()
        if isinstance(c.get("face_embedding_cosine"), (int, float))
    ]
    return {
        "evaluated": total,
        "faces_lost": lost,
        "loss_rate": round(lost / total, 6) if total else None,
        "face_detected": len(still_detected),
        "detection_retention": round(len(still_detected) / total, 6)
        if total
        else None,
        "mean_embedding_cosine": round(sum(cosines) / len(cosines), 6)
        if cosines
        else None,
        "min_embedding_cosine": round(min(cosines), 6) if cosines else None,
    }


def _detection_bucket(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Object-detection evidence: counts, label survival and confidence.

    Confidence alone is not enough here because a detector that keeps labelling
    the same objects at lower confidence has not been fooled, only made less
    certain; the count and the label set are what say whether the object was
    recognised at all.
    """
    total = len(comparisons)
    original_counts = [c["detection_count_original"] for c in comparisons.values()]
    deltas = [
        c["detection_count_protected"] - c["detection_count_original"]
        for c in comparisons.values()
    ]
    lost_all = sum(
        1
        for c in comparisons.values()
        if c["detection_count_original"] > 0 and c["detection_count_protected"] == 0
    )
    label_changes = sum(
        1 for c in comparisons.values() if c.get("detection_labels_changed")
    )
    return {
        "evaluated": total,
        "detections_lost_entirely": lost_all,
        "loss_rate": round(lost_all / total, 6) if total else None,
        "labels_changed": label_changes,
        "label_change_rate": round(label_changes / total, 6) if total else None,
        "mean_count_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "mean_original_count": round(sum(original_counts) / len(original_counts), 6)
        if original_counts
        else None,
    }


def _text_bucket(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Vision-language evidence: which of the answers actually changed.

    An answer changing is the whole event with a generative model -- there is no
    probability to threshold -- so the rate of change is the signal, and it is
    reported per row rather than folded into a score.
    """
    total = len(comparisons)
    changed = sum(1 for c in comparisons.values() if c.get("text_answers_changed"))
    per_prompt = [
        c.get("text_answers_changed_count", 0) for c in comparisons.values()
    ]
    return {
        "evaluated": total,
        "rows_changed": changed,
        "change_rate": round(changed / total, 6) if total else None,
        "mean_prompts_changed": round(sum(per_prompt) / len(per_prompt), 6)
        if per_prompt
        else None,
    }


def _infer_family(row: dict[str, Any]) -> str:
    """Which observation shape a comparison row came from.

    Used when the caller does not supply the registry's family map. The keys are
    produced by exactly one :func:`compare_*` function each, so this is a
    dispatch on structure rather than a heuristic -- and it falls back to
    ``"classification"`` only for the row shape that actually has the fields.
    """
    if "face_detected" in row:
        return "face"
    if "detection_count_original" in row:
        return "detection"
    if "text_answers_changed" in row:
        return "text"
    if "cosine_similarity" in row:
        return "embedding"
    return "classification"


def transfer_summary(
    comparisons: dict[str, dict[str, Any]],
    roles: dict[str, ModelRole],
    *,
    families: dict[str, ModelFamily] | None = None,
    evaluated: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-model verdicts into one honest headline number.

    The whole point of this function is that the average is computed **over
    held-out classifiers only**. Every model the attack optimised against is
    grouped separately and reported for diagnosis, because a number averaged
    across both groups answers neither question: it dilutes the evidence with
    the models that were paid for, and it is the specific way an ensemble attack
    is made to look successful without having transferred.

    Non-classification evidence is kept in its own buckets for the same reason
    and then some: face, detection, embedding and VLM rows have no
    ``relative_drop``, so folding them into the classification bucket would count
    every one of them as a non-collapse and drag the headline rate downwards by
    an amount that reflects nothing about the attack.

    Args:
        comparisons: ``model key -> comparison dict`` from :func:`compare`.
            Rows carrying a ``status`` other than ``"ok"`` -- the benchmark's
            record of a model that did not run -- are dropped before anything
            is counted, because a skipped model measured nothing and must not
            appear as a measured non-event.
        roles: ``model key -> role`` for **every** declared model, whether it ran
            or not. Missing keys are treated as unassigned and never counted as
            held-out evidence.
        families: Optional ``model key -> family`` from the registry. Omitted,
            families are inferred from the row shape, which is unambiguous
            because each ``compare_*`` writes a disjoint key set.
        evaluated: Optional ``model key -> status`` for the full set, used to
            report how much of each group was actually measured.

    Returns:
        A mapping with one bucket per group, plus the transfer rates. Rates are
        ``None`` rather than ``0.0`` when nothing was measured -- reporting zero
        success for a run that evaluated nothing would be indistinguishable from
        a run where nothing worked.
    """
    by_family: dict[str, dict[str, dict[str, Any]]] = {
        "classification": {},
        "embedding": {},
        "face": {},
        "detection": {},
        "text": {},
    }
    # A row that was not measured must not be counted as measured. The
    # benchmark writes ``{"status": ..., "reason": ...}`` for a model that did
    # not run, and folding that into ``evaluated`` would turn an absent
    # measurement into a confident zero -- the exact failure this function
    # exists to avoid.
    measured = {
        key: row
        for key, row in comparisons.items()
        if row.get("status", "ok") == "ok"
    }

    for key, comparison in measured.items():
        family = (families or {}).get(key) or _infer_family(comparison)
        by_family.get(family, by_family["classification"])[key] = comparison

    classification = by_family["classification"]
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        "held_out": {},
        "surrogate": {},
        "target": {},
        "evaluator": {},
        "unassigned": {},
    }
    for key, row in classification.items():
        role = roles.get(key, "unassigned")
        grouped[role if role in grouped else "unassigned"][key] = row

    # Which declared models belong to which bucket. A model with no comparison
    # row cannot be classified by shape, so it defaults to "classification" --
    # the conservative choice, since the held-out count must be satisfied by
    # actually running it rather than assumed away.
    declared_family: dict[str, str] = {}
    for key, role in roles.items():
        if role == "held_out":
            known: str | None = (families or {}).get(key)
            if known is None:
                known = (
                    _infer_family(measured[key])
                    if key in measured
                    else "classification"
                )
            declared_family[key] = known

    held_out_declared = {
        key for key, family in declared_family.items() if family == "classification"
    }
    evaluator_declared = {key for key, role in roles.items() if role == "evaluator"}
    probe_keys = set(declared_family) - held_out_declared
    probe_keys |= set(by_family["face"]) | set(by_family["detection"]) | set(
        by_family["text"]
    )

    bucket = _bucket(grouped["held_out"])
    bucket["expected"] = len(held_out_declared)
    bucket["models"] = sorted(grouped["held_out"])
    bucket["not_evaluated"] = sorted(held_out_declared - set(grouped["held_out"]))

    surrogate_bucket = _bucket(grouped["surrogate"])
    surrogate_bucket["models"] = sorted(grouped["surrogate"])

    embedding_bucket = _evaluator_bucket(by_family["embedding"])
    embedding_bucket["expected"] = len(evaluator_declared)
    embedding_bucket["models"] = sorted(by_family["embedding"])
    embedding_bucket["not_evaluated"] = sorted(
        evaluator_declared - set(by_family["embedding"])
    )
    face_bucket = _face_bucket(by_family["face"])
    detection_bucket = _detection_bucket(by_family["detection"])
    text_bucket = _text_bucket(by_family["text"])

    statuses = evaluated or {}
    return {
        # The headline: transfer to classification models the attack never saw.
        "held_out": bucket,
        "surrogates": surrogate_bucket,
        "target": _bucket(grouped["target"]),
        # An evaluator-role row with a classification shape has nowhere else to
        # go; it is reported rather than dropped, but it never touches the rates.
        "evaluator": _bucket(grouped["evaluator"]),
        "unassigned": _bucket(grouped["unassigned"]),
        # Non-classification evidence, one bucket per family.
        "embedding": embedding_bucket,
        "face": face_bucket,
        "detection": detection_bucket,
        "text": text_bucket,
        "ensemble_transfer_success": bucket["collapse_rate"],
        "ensemble_transfer_success_strict": bucket["disruption_rate"],
        "evaluator_transfer_success": embedding_bucket["prompt_change_rate"],
        "held_out_status": {
            key: statuses.get(key, "unknown") for key in sorted(held_out_declared)
        },
        "evaluator_status": {
            key: statuses.get(key, "unknown") for key in sorted(evaluator_declared)
        },
        "probe_status": {
            key: statuses.get(key, "unknown") for key in sorted(probe_keys)
        },
        "held_out_complete": bucket["evaluated"] == bucket["expected"]
        and bucket["expected"] > 0,
        "evaluator_complete": embedding_bucket["evaluated"]
        == len(evaluator_declared)
        and bool(evaluator_declared),
        "notes": (
            "Classification rates are computed over held-out classifiers only. "
            "Surrogate and target rows are diagnostics: a drop there shows the "
            "optimiser worked, not that the perturbation transferred. Face, "
            "detection, embedding and VLM evidence is reported per family "
            "because none of them has a confidence to collapse."
        ),
    }
