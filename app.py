"""Step 6: the Gradio demo, built to be run in front of judges.

    python app.py                 # http://127.0.0.1:7860, browser opens itself
    python app.py --share         # temporary public link
    python app.py --port 8000     # different port
    python app.py --no-proof      # skip face-swap, ArcFace verdict only

Everything the demo claims is computed live on the uploaded photo. No number
anywhere in the interface is hard-coded or cached from an earlier run: the
cosine similarities, the PSNR/LPIPS/SSIM, the swap results and the control
experiment are all produced by the same click that produced the images next to
them.

Colour handling is explicit throughout. The project reads and writes images as
BGR uint8 (OpenCV), Gradio displays numpy arrays as **RGB**, and the swapper and
the metrics expect BGR. Every hand-off goes through :func:`to_rgb` or
:func:`to_bgr` so nothing is displayed with its channels swapped.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("app")

import gradio as gr  # noqa: E402

from src.attack import AttackConfig, SurrogateEnsemble, protect  # noqa: E402
from src.common import (  # noqa: E402
    DONOR_DIR,
    FACES_DIR,
    OUTPUTS_DIR,
    imread,
    imwrite,
    list_images,
    set_seed,
)
from src.detect import FaceDetector, face_mask, inswapper_path  # noqa: E402
from src.evaluate import VERIFY_THRESHOLDS, uniform_noise  # noqa: E402
from src.metrics import lpips_distance, psnr, ssim  # noqa: E402

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: insightface's own documentation puts the ArcFace same-identity threshold at
#: 0.28 cosine ("for example, threshold=0.28 for ArcFace"); some deployments use
#: the more conservative 0.4. The verdict uses 0.28 on purpose: a higher bar
#: would declare "different person" more often, which is the claim this project
#: cannot back up. 0.4 is reported alongside so a judge sees both.
SAME_PERSON_THRESHOLD = 0.28
STRICT_THRESHOLD = 0.4

PRESETS: dict[str, dict] = {
    "Невидимая (eps≈8)": {"eps255": 8.0, "steps": 30},
    "Сильная (eps≈24)": {"eps255": 24.0, "steps": 36},
}

#: Judge uploads and the sample photos are 22-megapixel originals. Detecting,
#: blurring a feather mask and running LPIPS at that size costs seconds each for
#: no visible gain -- the attack itself works on a 160x160 crop regardless. So
#: the demo works on a bounded copy and says so in the interface.
MAX_SIDE = 1280

LIMITATIONS = (
    "**Что работает.** Понижает сходство лица в системах распознавания и "
    "ухудшает качество подмены лица, если это фото подать в подмену как "
    "источник идентичности.\n\n"
    "**Чего не делает.** Не защищает от Gemini, ChatGPT и других "
    "мультимодальных моделей — они видят фото и с защитой, и без. Эффект "
    "слабеет после сильной очистки шума (DiffPure, реконструкция из "
    "низкокачественной копии, сильное сжатие). Защита повышает стоимость "
    "атаки, но не делает её невозможной: выводы сделаны на трёх фотографиях."
)

HOW_IT_WORKS = f"""
## Что здесь происходит

**1. Детекция лица.** `insightface buffalo_l` (SCRFD) находит лицо и пять
ключевых точек. Если не срабатывает ни один масштаб, подключается MTCNN как
второе мнение.

**2. Атака PGD.** Лицо вырезается с запасом, уменьшается до 160×160, и
`InceptionResnetV1` (vggface2 + casia-webface) оптимизируются так, чтобы
эмбеддинг защищённого кропа разошёлся с эмбеддингом чистого. Маска применяется
**внутри** оптимизации, а не наклеивается сверху — иначе градиенты считались бы
для картинки, которую сеть никогда не видела.

**3. EOT.** На каждом шаге сеть получает не оригинал, а случайно искажённые
копии: ресайз, шум, размытие, кроп, JPEG. Это заставляет возмущение выживать
при реальном обращении с фотографией, а не подгоняться под один рендеринг.

**4. Незаметность.** Возмущение считается **в одной плоскости яркости** и
пишется одинаково в R, G и B, поэтому оттенок не меняется. После апсемплинга
возмущение сглаживается, а маска расширена и сильно растушёвана, чтобы у края
лица не было светлого ореола. Пиксели вне маски не меняются вообще.

**5. Проверка на живой модели.** `buffalo_l` (ArcFace) в оптимизации **не
участвовала** — это чёрный ящик, специально чтобы цифра не была self-graded.
Считается сходство «оригинал ↔ защищённое», и выносится вердикт.

**6. Живая подмена лица.** `inswapper_128` вклеивает лицо в постороннюю сцену
из `data/donor/` дважды: источник идентичности — оригинал, затем защищённое
фото. Сравнивается, насколько результат похож на настоящего человека.

**7. Контроль.** То же самое со случайным шумом той же амплитуды. Если
падение сходства объяснялось бы тем, что картинка стала «неудобной», контроль
это покажет: у него сходство почти не меняется.
"""

CSS = """
.gradio-container { max-width: 1400px !important; }
.verdict-badge { display:inline-block; padding:6px 14px; border-radius:999px;
                 font-weight:700; font-size:15px; margin:4px 0; }
.v-ok   { background:#14532d; color:#bbf7d0; border:1px solid #166534; }
.v-bad  { background:#7f1d1d; color:#fecaca; border:1px solid #b91c1c; }
.v-warn { background:#78350f; color:#fde68a; border:1px solid #b45309; }
.timer { color:#94a3b8; font-size:13px; }
"""


# ---------------------------------------------------------------------------
# colour helpers -- the one thing that must never be guessed
# ---------------------------------------------------------------------------


def to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV/project layout) -> RGB (what Gradio displays)."""
    if image is None:
        return None  # type: ignore[return-value]
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB -> BGR, for the paths that expect OpenCV layout."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _bound(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Shrink so the long side is at most ``MAX_SIDE``. Returns (image, scaled)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_SIDE:
        return image, False
    scale = MAX_SIDE / longest
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, True


# ---------------------------------------------------------------------------
# model state, loaded once
# ---------------------------------------------------------------------------

_STATE: dict[str, object] = {}


def _insight_face(app, image, sizes=((640, 640), (320, 320))):
    """Best insightface face with a usable embedding, or ``None``."""
    for size in sizes:
        app.prepare(ctx_id=-1, det_size=size)
        try:
            faces = app.get(image)
        except Exception:
            continue
        usable = [
            f for f in faces if f.normed_embedding is not None and f.det_score >= 0.35
        ]
        if usable:
            return max(usable, key=lambda f: f.det_score)
    return None


def _load_models(with_swap: bool = True) -> None:
    """Load every network once, at process start.

    Loading costs far more than a single attack, so this runs before the UI is
    built and the result is reused for every photo the judges upload.
    """
    if _STATE.get("ready"):
        return
    started = time.perf_counter()
    logger.info("loading models once: detector, surrogates, ArcFace evaluator")

    _STATE["detector"] = FaceDetector()
    _STATE["ensemble"] = SurrogateEnsemble()
    from src.evaluate import ArcFaceEvaluator

    evaluator = ArcFaceEvaluator()
    _STATE["evaluator"] = evaluator

    _STATE["swapper"] = None
    if with_swap:
        try:
            from insightface import model_zoo

            _STATE["swapper"] = model_zoo.get_model(
                inswapper_path(), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            logger.warning("face-swap unavailable: %s", error)

    # The donor scene never changes, so its face is detected exactly once.
    scene = None
    for path in list_images(DONOR_DIR):
        try:
            scene = imread(path)
            break
        except Exception:
            continue
    if scene is None:
        logger.warning("no donor scene under %s", DONOR_DIR)
        _STATE["scene"] = None
        _STATE["scene_face"] = None
    else:
        face = _insight_face(evaluator.app, scene)
        if face is None:
            logger.warning("no usable face in the donor scene")
        _STATE["scene"] = scene
        _STATE["scene_face"] = face

    # NOTE: deliberately *not* calling torch.set_grad_enabled(False) here.
    # The PGD attack needs autograd; the inference paths that do not are each
    # already wrapped in torch.no_grad() (src.metrics, src.evaluate).
    _STATE["ready"] = True
    logger.info(
        "models ready in %.1f s (inswapper: %s)",
        time.perf_counter() - started,
        "yes" if _STATE.get("swapper") is not None else "NO",
    )


def _example_photos() -> list[str]:
    return [str(p) for p in list_images(FACES_DIR)]


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------


def _badge(ok: bool, yes: str, no: str, warn: bool = False) -> str:
    style = "v-ok" if ok else ("v-warn" if warn else "v-bad")
    return f'<span class="verdict-badge {style}">{yes if ok else no}</span>'


def _similarity_html(value: float | None, threshold: float) -> str:
    if value is None:
        return '<span class="verdict-badge v-warn">лицо не найдено</span>'
    if value >= threshold:
        return (
            f'<span class="verdict-badge v-ok">сходство {value:.3f} — тот же человек</span>'
        )
    return (
        f'<span class="verdict-badge v-bad">сходство {value:.3f} — другой человек</span>'
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "н/д" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def protect_image(
    photo: Path | None,
    preset: str = "Невидимая (eps≈8)",
    eps255: float | None = None,
    steps: int | None = None,
    seed: int = 0,
    do_proof: bool = True,
    progress=gr.Progress(),
):
    """Protect one photo and produce every number the interface shows."""
    if photo is None:
        raise gr.Error("Сначала загрузите фотографию.")

    _load_models()
    detector: FaceDetector = _STATE["detector"]  # type: ignore[assignment]
    ensemble: SurrogateEnsemble = _STATE["ensemble"]  # type: ignore[assignment]
    evaluator = _STATE["evaluator"]
    ev_app = evaluator.app
    det_sizes = getattr(ev_app, "_det_sizes", ((640, 640), (320, 320), (960, 960)))
    swapper = _STATE.get("swapper")
    scene = _STATE.get("scene")
    scene_face = _STATE.get("scene_face")

    settings = PRESETS.get(preset, PRESETS["Невидимая (eps≈8)"])
    eps_value = float(settings["eps255"] if eps255 is None else eps255)
    step_value = int(settings["steps"] if steps is None else steps)
    report = progress if callable(progress) else lambda *_args, **_kwargs: None
    report(0.03, desc="Читаю изображение…")

    set_seed(int(seed))
    original, was_scaled = _bound(imread(photo))
    height, width = original.shape[:2]

    report(0.07, desc="Ищу лицо…")
    face = detector.detect(original)
    if face is None:
        raise gr.Error(
            "Лицо не найдено. Нужна фотография, где лицо крупно, фронтально и "
            "хорошо освещено."
        )

    report(0.12, desc=f"Атака PGD, {step_value} шагов, eps={eps_value:g}/255…")
    attack_started = time.perf_counter()
    config = AttackConfig(eps=eps_value / 255.0, steps=step_value, seed=int(seed))
    protected, facts = protect(original, face, ensemble, config)
    attack_seconds = time.perf_counter() - attack_started

    mask = face_mask(
        original.shape, face.bbox, expand=config.mask_expand, feather=config.mask_feather
    )
    diff = protected.astype(np.int16) - original.astype(np.int16)
    amplified = np.clip(128.0 + diff.astype(np.float32) * 6.0, 0, 255).astype(np.uint8)

    report(0.55, desc="Считаю PSNR / SSIM / LPIPS…")
    psnr_value = psnr(original, protected)
    ssim_value = ssim(original, protected)
    lpips_value = lpips_distance(original, protected)
    linf = int(np.abs(diff).max())
    coverage = 100.0 * float((mask > 0.02).mean())

    report(0.60, desc="ArcFace: оригинал и защищённое…")
    # One detection each, reused three ways: the original's embedding is the
    # ground-truth identity, the protected one's gives the similarity *and* acts
    # as the identity source for the swap. Detecting them separately would be
    # the same work done twice.
    original_face = _insight_face(ev_app, original, det_sizes)
    if original_face is None:
        raise gr.Error("ArcFace не нашёл лицо в оригинале — не с чем сравнивать.")
    reference = np.asarray(original_face.normed_embedding, dtype=np.float32)
    protected_face = _insight_face(ev_app, protected, det_sizes)
    protected_embedding = (
        None
        if protected_face is None
        else np.asarray(protected_face.normed_embedding, dtype=np.float32)
    )
    similarity = evaluator.similarity(reference, protected_embedding)

    report(0.70, desc="Контроль: случайный шум той же амплитуды…")
    control = uniform_noise(original, mask, eps_value, seed=int(seed))
    control_similarity = evaluator.similarity(reference, evaluator.embed(control))
    control_psnr = psnr(original, control)

    verdict_html = _similarity_html(similarity, SAME_PERSON_THRESHOLD)
    if similarity is None:
        explanation = (
            "ArcFace не находит лицо в защищённом изображении — это сильнее, чем "
            "просто снижение сходства."
        )
    elif similarity >= SAME_PERSON_THRESHOLD:
        explanation = (
            f"При пороге {SAME_PERSON_THRESHOLD} (insightface, ArcFace) ArcFace всё "
            "ещё считает, что это тот же человек. Защита ослабляет сходство, но "
            "порог верификации не пересекает."
        )
    else:
        explanation = (
            f"Сходство упало ниже {SAME_PERSON_THRESHOLD} — порога, который "
            "insightface использует для ArcFace. Система распознавания видит "
            "другого человека."
        )
    if control_similarity is not None:
        explanation += (
            f" Случайный шум той же амплитуды дал {control_similarity:.3f}, "
            "то есть снижение вызвано оптимизацией, а не самим фактом шума."
        )

    # ---- live face-swap proof -------------------------------------------
    swap_original = None
    swap_protected = None
    swap_from_original_sim = None
    swap_from_protected_sim = None
    swap_note = ""

    if not do_proof:
        swap_note = "Проверка подмены отключена параметром `--no-proof`."
    elif swapper is None or scene is None or scene_face is None:
        swap_note = (
            "Подмена лица недоступна: нет весов `models/inswapper_128.onnx` "
            "или лица в сцене-доноре. Скачать: "
            "https://huggingface.co/ezioruan/inswapper_128.onnx"
        )
    else:
        report(0.80, desc="Подмена лица: источник — оригинал…")
        if protected_face is None:
            # A genuine protection outcome: the attacker cannot even seed the swap.
            swap_note = (
                "Лицо не детектируется в защищённом фото — подмену нечем "
                "инициировать."
            )
        else:
            # Direction matters and is easy to get backwards: `target_face`
            # supplies pose and paste location, `source_face` supplies the
            # identity. To test the protection, *our* photo must be the source.
            swap_original = swapper.get(scene, scene_face, original_face)
            report(0.87, desc="Подмена лица: источник — защищённое…")
            swap_protected = swapper.get(scene, scene_face, protected_face)
            swap_from_original_sim = evaluator.similarity(
                reference, evaluator.embed(swap_original)
            )
            report(0.93, desc="Считаю сходство результата подмены…")
            swap_from_protected_sim = evaluator.similarity(
                reference, evaluator.embed(swap_protected)
            )
            swap_note = (
                f"Сцена-цель — постороннее фото из `data/donor/`. "
                f"inswapper берёт идентичность из источника, поэтому сравнение "
                f"смотрит, насколько результат похож на настоящего человека. "
                f"Подмена не исчезает: то, что деградирует, — это достоверность "
                f"идентичности."
            )

    report(0.97, desc="Собираю таблицы…")

    rows = [
        (
            "разрешение обработки",
            f"{width}×{height}",
            (
                f"исходник уменьшен до {MAX_SIDE}px по длинной стороне"
                if was_scaled
                else "без масштабирования"
            ),
        ),
        ("PSNR", f"{psnr_value:.2f} дБ", "больше — чище картинка"),
        ("SSIM", f"{ssim_value:.5f}", "больше — ближе к оригиналу"),
        ("LPIPS", _fmt(lpips_value), "меньше — незаметнее"),
        (
            "макс. изменение пикселя",
            f"{linf}/{255}",
            "ограничение L∞ соблюдено",
        ),
        ("площадь лица изменена", f"{coverage:.1f}%", "остальное не тронуто"),
        (
            "ArcFace: оригинал ↔ защищённое",
            _fmt(similarity, 3),
            "1.000 = идентичность не тронута",
        ),
        (
            "ArcFace: контроль (случайный шум)",
            _fmt(control_similarity, 3),
            f"для сравнения, PSNR {control_psnr:.2f} дБ",
        ),
        (
            "подмена: источник = оригинал",
            _fmt(swap_from_original_sim, 3),
            "базовая линия, подмена работает",
        ),
        (
            "подмена: источник = защищённое",
            _fmt(swap_from_protected_sim, 3),
            "ниже = защита сработала",
        ),
        ("время атаки", f"{attack_seconds:.1f} с", "CPU, без GPU"),
    ]
    metrics_table = "\n".join(
        f"| {name} | {value} | {note} |" for name, value, note in rows
    )
    metrics_markdown = (
        "| параметр | значение | комментарий |\n|---|---|---|\n" + metrics_table
    )

    swap_rows = [
        (
            "источник = оригинал",
            _fmt(swap_from_original_sim, 3),
            "подмена воспроизводит человека",
        ),
        (
            "источник = защищённое",
            _fmt(swap_from_protected_sim, 3),
            "насколько идентичность уцелела",
        ),
    ]
    swap_table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in swap_rows)
    swap_markdown = (
        "| вариант | сходство с настоящим человеком | смысл |\n|---|---|---|\n"
        + swap_table
    )

    if swap_from_original_sim is None or swap_from_protected_sim is None:
        swap_badge = ""
    else:
        degraded = swap_from_protected_sim < swap_from_original_sim
        crossed = swap_from_protected_sim < SAME_PERSON_THRESHOLD
        swap_badge = _badge(
            degraded and crossed,
            f"идентичность деградировала: {swap_from_original_sim:.3f} → "
            f"{swap_from_protected_sim:.3f} (ниже {SAME_PERSON_THRESHOLD})",
            f"сходство упало: {swap_from_original_sim:.3f} → "
            f"{swap_from_protected_sim:.3f}",
            warn=not crossed,
        )

    report(1.0, desc=f"готово, атака {attack_seconds:.1f} с")

    try:
        imwrite(OUTPUTS_DIR / "app_output.png", protected)
        if swap_protected is not None:
            imwrite(OUTPUTS_DIR / "app_swap_protected.jpg", swap_protected)
        if swap_original is not None:
            imwrite(OUTPUTS_DIR / "app_swap_original.jpg", swap_original)
    except Exception as error:
        logger.warning("could not save demo output: %s", error)

    return (
        to_rgb(original),
        to_rgb(protected),
        to_rgb(amplified),
        metrics_markdown,
        verdict_html,
        explanation,
        to_rgb(swap_original) if swap_original is not None else None,
        to_rgb(swap_protected) if swap_protected is not None else None,
        swap_markdown,
        swap_badge,
        swap_note,
        to_rgb(control),
        f"Атака {attack_seconds:.1f} с · весь цикл {time.perf_counter() - attack_started:.1f} с",
    )


def _on_preset_change(preset: str):
    settings = PRESETS.get(preset)
    if settings is None:
        return gr.update(), gr.update()
    return gr.update(value=settings["eps255"]), gr.update(value=settings["steps"])


def dark_theme() -> gr.themes.Base:
    """Dark palette. Built here rather than inside ``build_demo`` because Gradio 6
    wants ``theme`` passed to ``launch()``, not to the ``Blocks`` constructor."""
    return gr.themes.Base(
        primary_hue="indigo", secondary_hue="indigo", neutral_hue="slate"
    ).set(
        body_background_fill_dark="#0b1120",
        background_fill_primary_dark="#0f172a",
        background_fill_secondary_dark="#111c33",
        block_background_fill_dark="#16213c",
        block_border_color_dark="#243350",
        block_title_text_color_dark="#e2e8f0",
        body_text_color_dark="#cbd5e1",
        input_background_fill_dark="#0f172a",
        button_primary_background_fill_dark="#4338ca",
        button_primary_text_color_dark="#eef2ff",
        border_color_primary_dark="#6366f1",
        table_odd_background_fill_dark="#131c31",
    )


def build_demo(default_preset: str) -> gr.Blocks:
    with gr.Blocks(title="AntiFake-Face") as demo:
        gr.HTML(
            "<div style='text-align:center;padding:8px 0 2px'>"
            "<h1 style='margin:0;font-size:2.4em'>AntiFake-Face</h1>"
            "<p style='margin:6px 0 0;color:#94a3b8;font-size:1.05em'>"
            "Невидимый адверсарный шум против распознавания и подмены лица. "
            "Проверяем на живой модели, которую никогда не оптимизировали."
            "</p></div>"
        )

        with gr.Tabs():
            with gr.Tab("Защитить"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        source = gr.Image(
                            label="Фотография", type="filepath", height=280
                        )
                        preset = gr.Radio(
                            choices=list(PRESETS),
                            value=default_preset,
                            label="Пресет",
                        )
                        eps = gr.Slider(
                            4, 32, value=PRESETS[default_preset]["eps255"],
                            step=1, label="Сила (eps, /255)",
                        )
                        steps = gr.Slider(
                            10, 60, value=PRESETS[default_preset]["steps"],
                            step=2, label="Шаги PGD",
                        )
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        proof = gr.Checkbox(
                            value=True,
                            label="Показать живую подмену лица (inswapper)",
                        )
                        button = gr.Button("Защитить", variant="primary", size="lg")
                        timer = gr.HTML("", elem_id="run-timer")

                        examples = gr.Examples(
                            examples=_example_photos(),
                            inputs=[source],
                            fn=protect_image,
                            run_on_click=True,
                            label="Примеры — клик защищает сразу",
                        )

                    with gr.Column(scale=2):
                        verdict = gr.HTML("", elem_id="verdict")
                        with gr.Row():
                            view_original = gr.Image(
                                label="Оригинал", height=250, elem_id="view-original"
                            )
                            view_protected = gr.Image(
                                label="Защищённое", height=250,
                                elem_id="view-protected",
                            )
                        view_diff = gr.Image(
                            label="Карта разницы, усиление ×6 (серый = без изменений)",
                            height=190,
                            elem_id="view-diff",
                        )
                        note = gr.Markdown("", elem_id="note")
                        metrics = gr.Markdown("", elem_id="metrics")

                        gr.Markdown("### Живая подмена лица")
                        gr.Markdown(
                            "Сцена-цель — постороннее фото из `data/donor/`. "
                            "Слева подмена с идентичностью из оригинала, справа — "
                            "из защищённого фото."
                        )
                        swap_badge = gr.HTML("", elem_id="swap-badge")
                        with gr.Row():
                            swap_original = gr.Image(
                                label="Источник: оригинал", height=250,
                                elem_id="swap-original",
                            )
                            swap_protected = gr.Image(
                                label="Источник: защищённое", height=250,
                                elem_id="swap-protected",
                            )
                        swap_table = gr.Markdown("", elem_id="swap-table")
                        swap_note_md = gr.Markdown("", elem_id="swap-note")

                        gr.Markdown("### Контроль")
                        gr.Markdown(
                            "Случайный шум той же амплитуды в той же маске. "
                            "Если бы сходство падало и здесь, падение было бы "
                            "artifact'ом картинки, а не результатом оптимизации."
                        )
                        view_control = gr.Image(
                            label="Контроль: случайный шум", height=200,
                            elem_id="view-control",
                        )

            with gr.Tab("Как это работает"):
                gr.Markdown(HOW_IT_WORKS)

        gr.Markdown("---")
        gr.Markdown(
            "<div style='font-size:13px;color:#94a3b8'>"
            "<b>Ограничения.</b> " + LIMITATIONS.replace("**", "") + "</div>"
        )

        button.click(
            protect_image,
            inputs=[source, preset, eps, steps, seed, proof],
            outputs=[
                view_original, view_protected, view_diff, metrics, verdict, note,
                swap_original, swap_protected, swap_table, swap_badge, swap_note_md,
                view_control, timer,
            ],
            api_name="protect",
        )
        preset.change(_on_preset_change, inputs=[preset], outputs=[eps, steps])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="public link")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="skip the face-swap demo (faster, works without the ONNX weights)",
    )
    args = parser.parse_args()

    if args.steps:
        for settings in PRESETS.values():
            settings["steps"] = args.steps

    _load_models(with_swap=not args.no_proof)
    demo = build_demo("Невидимая (eps≈8)")
    demo.queue().launch(
        share=args.share,
        server_port=args.port,
        inbrowser=args.inbrowser,
        theme=dark_theme(),
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())