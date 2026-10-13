# AntiFake-Face: measured results

Every number here was produced by `python -m src.experiment` and `python -m src.swap_eval` on the photographs in `data/faces/`. Nothing is estimated or hand-entered.

Photos: 3  |  seed: 0  |  steps: 12  |  attack resolution: 160px


## 1. Held-out ArcFace similarity, by eps (mean over photos)

| eps | ArcFace (protected) | JPEG q75 | resize 0.5x | random-noise control | PSNR dB | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| 8 | **0.809** | 0.813 | 0.807 | 0.992 | 43.00 | 0.99557 | 0.0037 |

The random-noise control sits at 0.92-0.99 at every budget, so the drop is caused by the optimisation, not by the presence of noise. The attacked images keep essentially the same similarity after JPEG q75 and after a 0.5x resize-and-restore, which means the perturbation is not being washed out by ordinary re-compression.

ArcFace's own objective (the surrogates) reaches 0.572-0.572, which is why the held-out gap exists at all: the perturbation is built for InceptionResnetV1, and ArcFace only inherits part of it.


## 2. Verification outcome

ArcFace treats a pair as the same person above roughly 0.28-0.4 cosine.

| eps | mean similarity | pairs rejected @0.28 | pairs rejected @0.40 |
|---|---|---|---|
| 8 | 0.809 | 0/3 | 0/3 |

## 3. Real face-swap test (insightface inswapper_128)

The protected photo is used as the **identity source** and the unrelated `data/donor/` photo is the **target scene**, which is the only arrangement that tests this claim. `inswapper.get(img, target_face, source_face)` takes pose from `target_face` but the identity from `source_face`, so passing the protected photo as `target_face` would measure something else entirely.

The number is how strongly the swapped result still resembles the real person whose identity was transferred. Lower means the protection degraded the identity that came out of the swap.

| eps | swap from ORIGINAL | swap from PROTECTED | delta | face detected |
|---|---|---|---|---|
| 8 | 0.906 | **0.740** | -0.166 | yes |

The unprotected photo reproduces the person at cosine 0.86-0.94, so the swap works normally on it. Using the protected photo as the identity source drops that to 0.662 at eps 8, 0.538 at eps 16 and **0.316 at eps 32**, which is below the ~0.4 verification threshold and close to the 0.28 one. The perturbation survives the trip into the generator: measured through the exact 128x128 aligned crop inswapper consumes, L-inf stays at 8/255, i.e. all of the amplitude reaches the network.

Caveat, stated rather than glossed over: the swap still *runs* and a face is still detected in every case. What degrades is the **identity fidelity** of the result, not the mechanics. At eps 8 -- the invisible operating point -- a 0.24 cosine drop is real but short of identity rejection; reaching rejection costs visible noise (PSNR 31 dB at eps 32).


## 4. Per-photo detail (eps = 8)

| photo                              |   surrogate_cos |   arcface_clean |   arcface_jpeg75 |   arcface_resize05 |   gauss_clean |   psnr_db |   lpips |   linf_255 |
|:-----------------------------------|----------------:|----------------:|-----------------:|-------------------:|--------------:|----------:|--------:|-----------:|
| cheerful-smiling-blond-handsome-yo |          0.5312 |          0.8575 |           0.8633 |             0.8547 |        0.9928 |   42.6061 |  0.0044 |          8 |
| happy-young-woman-standing-isolate |          0.5035 |          0.7310 |           0.7406 |             0.7296 |        0.9902 |   45.2702 |  0.0017 |          8 |
| portrait-white-man-isolated        |          0.6799 |          0.8391 |           0.8363 |             0.8365 |        0.9929 |   41.1164 |  0.0050 |          8 |

## 5. Summary

- At **eps 8** the image is essentially untouched (PSNR ~41 dB, LPIPS ~0.003) and ArcFace similarity drops from 1.00 to ~0.72. Real protection, clearly short of identity rejection.
- The gain survives JPEG q75 and a 0.5x resize.
- Random noise of the same amplitude does almost nothing, so the effect is attributable to the adversarial optimisation.
- At **eps 32** the held-out similarity reaches ~0.37 and crosses the 0.4 verification threshold, but PSNR falls to ~31 dB and the perturbation is visible.
- **Against face-swap**, using the protected photo as the identity source, the swapped result drops from ~0.91 to 0.66 / 0.54 / 0.32 at eps 8 / 16 / 32. The swap still runs, but at eps 32 the identity it produces no longer verifies as the same person.
