"""Ad-hoc end-to-end check against a live server.

Not part of the test suite. Run with a server already listening on :8000:

    python scripts/smoke_e2e.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "tests"))

from helpers import decode, encode_jpeg, encode_png, make_photo  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"


def post_multipart(url: str, filename: str, content_type: str, raw: bytes) -> dict:
    """POST a single-file multipart form without pulling in a dependency."""
    boundary = f"----antifake{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=head + raw + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def check(label: str, condition: bool) -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def main() -> int:
    width, height = 640, 400
    ok = True

    print("health")
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        health = json.loads(response.read())
    print(f"  engine      {health['engine']['engine']}")
    print(f"  target      {health['engine']['target_model']}")
    print(f"  device      {health['engine']['device']}")
    ok &= check("status ok", health["status"] == "ok")

    for label, filename, content_type, raw, expect_lossy in [
        ("PNG in / PNG out", "photo.png", "image/png", encode_png(make_photo(width, height)), False),
        ("JPEG in / JPEG out", "photo.jpg", "image/jpeg", encode_jpeg(make_photo(width, height)), True),
    ]:
        print(f"\n{label}")
        started = time.perf_counter()
        payload = post_multipart(
            f"{BASE_URL}/api/protect?return_image=false", filename, content_type, raw
        )
        wall = time.perf_counter() - started
        delivered = base64.b64decode(payload["image_base64"])
        decoded = decode(delivered)
        asked = payload["perturbation_metrics"]
        got = payload["delivered_metrics"]

        print(
            f"  out         {payload['output_format']} "
            f"{decoded.shape[1]}x{decoded.shape[0]}"
        )
        print(
            f"  perturbation linf {asked['linf_8bit']}/255  "
            f"psnr {asked['psnr_db']:.2f} dB  ssim {asked['ssim']:.4f}"
        )
        print(
            f"  delivered    linf {got['linf_8bit']}/255  "
            f"psnr {got['psnr_db']:.2f} dB  ssim {got['ssim']:.4f}"
        )
        print(f"  server ms   {payload['process_ms']:.0f}")
        print(f"  wall s      {wall:.2f}")

        ok &= check("dimensions preserved", (decoded.shape[1], decoded.shape[0]) == (width, height))
        ok &= check("opens with Pillow", Image.open(io.BytesIO(delivered)).size == (width, height))
        ok &= check(
            "differs from original",
            not np.array_equal(decoded, make_photo(width, height)),
        )
        # The engine's promise is about the perturbation it adds, which is what
        # is bounded and reproducible. The delivered bytes can differ by more
        # when a lossy encoder adds its own error on top.
        ok &= check(
            "perturbation within engine budget",
            asked["linf_8bit"] <= payload["engine"]["eps_255"] + 1,
        )
        ok &= check("delivered psnr above 30 dB", got["psnr_db"] > 30)
        ok &= check("delivered ssim above 0.9", got["ssim"] > 0.9)
        ok &= check("format preserved", payload["output_format"] in ("PNG", "JPEG"))
        ok &= check(
            "lossless output keeps the exact perturbation" if not expect_lossy
            else "lossy output stays within the encoder's own error",
            got["linf_8bit"] == asked["linf_8bit"] if not expect_lossy
            else got["linf_8bit"] <= asked["linf_8bit"] + 12,
        )

    print("\nrejections")
    for label, filename, content_type, raw in [
        ("garbage bytes", "photo.png", "image/png", b"not an image"),
        ("unsupported type", "photo.gif", "image/gif", encode_png(make_photo(32, 32))),
    ]:
        try:
            post_multipart(f"{BASE_URL}/api/protect?return_image=false", filename, content_type, raw)
            ok &= check(f"{label} rejected", False)
        except urllib.error.HTTPError as error:
            detail = json.loads(error.read()).get("detail", "")
            print(f"  PASS  {label} -> {error.code}: {detail}")
            ok &= True

    print("\nRESULT:", "all checks passed" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
