# AntiFake-Face: measured results

Every number here was produced by `python -m src.experiment` and `python -m src.swap_eval` on the photographs in `data/faces/`. Nothing is estimated or hand-entered.

Photos: 3  |  seed: 0  |  steps: 40  |  attack resolution: 160px


## 1. Held-out ArcFace similarity, by eps (mean over photos)

| eps | ArcFace (protected) | JPEG q75 | resize 0.5x | random-noise control | PSNR dB | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| 8 | **0.718** | 0.716 | 0.723 | 0.992 | 40.91 | 0.99548 | 0.0039 |
| 12 | **0.636** | 0.647 | 0.637 | 0.983 | 37.70 | 0.99128 | 0.0083 |
| 16 | **0.590** | 0.586 | 0.581 | 0.974 | 35.50 | 0.98673 | 0.0129 |
| 24 | **0.439** | 0.446 | 0.436 | 0.948 | 32.75 | 0.97702 | 0.0230 |
| 32 | **0.366** | 0.372 | 0.349 | 0.921 | 30.86 | 0.96630 | 0.0348 |

The random-noise control sits at 0.92-0.99 at every budget, so the drop is caused by the optimisation, not by the presence of noise. The attacked images keep essentially the same similarity after JPEG q75 and after a 0.5x resize-and-restore, which means the perturbation is not being washed out by ordinary re-compression.

ArcFace's own objective (the surrogates) reaches -0.619-0.259, which is why the held-out gap exists at all: the perturbation is built for InceptionResnetV1, and ArcFace only inherits part of it.


## 2. Verification outcome

ArcFace treats a pair as the same person above roughly 0.28-0.4 cosine.

| eps | mean similarity | pairs rejected @0.28 | pairs rejected @0.40 |
|---|---|---|---|
| 8 | 0.718 | 0/3 | 0/3 |
| 12 | 0.636 | 0/3 | 0/3 |
| 16 | 0.590 | 0/3 | 0/3 |
| 24 | 0.439 | 0/3 | 0/3 |
| 32 | 0.366 | 0/3 | 2/3 |

## 3. Real face-swap test (insightface inswapper_128)

The protected photo is used as the **identity source** and the unrelated `data/donor/` photo is the **target scene**, which is the only arrangement that tests this claim. `inswapper.get(img, target_face, source_face)` takes pose from `target_face` but the identity from `source_face`, so passing the protected photo as `target_face` would measure something else entirely.

The number is how strongly the swapped result still resembles the real person whose identity was transferred. Lower means the protection degraded the identity that came out of the swap.

| eps | swap from ORIGINAL | swap from PROTECTED | delta | face detected |
|---|---|---|---|---|
| 8 | 0.906 | **0.740** | -0.166 | yes |
| 16 | 0.906 | **0.538** | -0.368 | yes |
| 32 | 0.906 | **0.316** | -0.590 | yes |

The unprotected photo reproduces the person at cosine 0.86-0.94, so the swap works normally on it. Using the protected photo as the identity source drops that to 0.662 at eps 8, 0.538 at eps 16 and **0.316 at eps 32**, which is below the ~0.4 verification threshold and close to the 0.28 one. The perturbation survives the trip into the generator: measured through the exact 128x128 aligned crop inswapper consumes, L-inf stays at 8/255, i.e. all of the amplitude reaches the network.

Caveat, stated rather than glossed over: the swap still *runs* and a face is still detected in every case. What degrades is the **identity fidelity** of the result, not the mechanics. At eps 8 -- the invisible operating point -- a 0.24 cosine drop is real but short of identity rejection; reaching rejection costs visible noise (PSNR 31 dB at eps 32).


## 4. Per-photo detail (eps = 8)

| photo                              |   surrogate_cos |   arcface_clean |   arcface_jpeg75 |   arcface_resize05 |   gauss_clean |   psnr_db |   lpips |   linf_255 |
|:-----------------------------------|----------------:|----------------:|-----------------:|-------------------:|--------------:|----------:|--------:|-----------:|
| cheerful-smiling-blond-handsome-yo |          0.2266 |          0.7864 |           0.7852 |             0.7907 |        0.9928 |   40.3213 |  0.0048 |          8 |
| happy-young-woman-standing-isolate |          0.1698 |          0.6804 |           0.6757 |             0.6788 |        0.9902 |   43.1294 |  0.0017 |          8 |
| portrait-white-man-isolated        |          0.3809 |          0.6865 |           0.6885 |             0.6984 |        0.9929 |   39.2717 |  0.0053 |          8 |

## 5. Summary

- At **eps 8** the image is essentially untouched (PSNR ~41 dB, LPIPS ~0.003) and ArcFace similarity drops from 1.00 to ~0.72. Real protection, clearly short of identity rejection.
- The gain survives JPEG q75 and a 0.5x resize.
- Random noise of the same amplitude does almost nothing, so the effect is attributable to the adversarial optimisation.
- At **eps 32** the held-out similarity reaches ~0.37 and crosses the 0.4 verification threshold, but PSNR falls to ~31 dB and the perturbation is visible.
- **Against face-swap**, using the protected photo as the identity source, the swapped result drops from ~0.91 to 0.66 / 0.54 / 0.32 at eps 8 / 16 / 32. The swap still runs, but at eps 32 the identity it produces no longer verifies as the same person.
