# AntiFake Phase 1 Benchmark

- Version: `1.0.0` (schema 1)
- Generated: 2026-09-29T06:32:36.894021+00:00
- Device: `cuda` (cuda:0 NVIDIA GeForce RTX 3070 Ti Laptop GPU)
- Images: 1, variants: 36, runtime: 13.67s
- Protection: eps 8.0/255, alpha 2.0/255, 40 steps, seed 0
- Target: torchvision.models:resnet18 (weights=imagenet1k_v1) via resize-256/center-crop-224
- LPIPS: available

## Summary

- Models: 7 ok, 0 skipped, 0 failed (of 7)
- Classifiers measured: 6
- PNG: label flipped on 0/6
- PNG: confidence collapsed (>=75% drop) on 1/6

## Models

| Model | Family | Status | Description |
| --- | --- | --- | --- |
| `resnet18` | classification | ok | torchvision:resnet18 (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `densenet121` | classification | ok | torchvision:densenet121 (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `efficientnet_b0` | classification | ok | torchvision:efficientnet_b0 (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `mobilenet_v3_small` | classification | ok | torchvision:mobilenet_v3_small (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `vit_b_16` | classification | ok | torchvision:vit_b_16 (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `swin_t` | classification | ok | torchvision:swin_t (weights=IMAGENET1K_V1) via resize-256/center-crop-224 |
| `clip_vit_b_32` | embedding | ok | open_clip:ViT-B-32 (pretrained=laion2b_s34b_b79k) |

## PNG transfer (protected PNG vs original)

| Image | Model | Top-1 | Label changed | Confidence | Relative drop | Entropy rise | Collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| photo_2026-09-28_22-46-08 | `resnet18` | 0.341333 -> 0.012258 | False | 0.329075 | 0.964088 | 3.359773 | True |
| photo_2026-09-28_22-46-08 | `densenet121` | 0.637707 -> 0.456379 | False | 0.181328 | 0.284343 | 1.19336 | False |
| photo_2026-09-28_22-46-08 | `efficientnet_b0` | 0.140376 -> 0.074728 | False | 0.065648 | 0.467659 | 0.529552 | False |
| photo_2026-09-28_22-46-08 | `mobilenet_v3_small` | 0.614891 -> 0.527779 | False | 0.087112 | 0.141671 | 0.781377 | False |
| photo_2026-09-28_22-46-08 | `vit_b_16` | 0.098003 -> 0.088013 | True | 0.00999 | 0.101931 | 0.078025 | False |
| photo_2026-09-28_22-46-08 | `swin_t` | 0.358541 -> 0.166655 | False | 0.191886 | 0.535186 | 0.763158 | False |

## Robustness

Protection effect after each transform, measured against the *same*
transform applied to the original. The numbers isolate the protection's
contribution rather than the transform's own effect.

| Image | Transform | Model | Relative drop | Entropy rise | Cosine | Best prompt |
| --- | --- | --- | --- | --- | --- | --- |
| photo_2026-09-28_22-46-08 | resize_short_side | `resnet18` | 0.946605 | 3.554879 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `densenet121` | 0.399024 | 1.546737 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `efficientnet_b0` | 0.415285 | 0.475515 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `mobilenet_v3_small` | 0.176428 | 0.909685 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `vit_b_16` | 0.151767 | 0.092299 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `swin_t` | 0.476922 | 0.722518 | - | - |
| photo_2026-09-28_22-46-08 | resize_short_side | `clip_vit_b_32` | - | - | 0.946672 | a book |
| photo_2026-09-28_22-46-08 | scale_0.5 | `resnet18` | 0.466824 | 2.62785 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `densenet121` | 0.128512 | 1.577116 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `efficientnet_b0` | 0.088201 | -0.058095 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `mobilenet_v3_small` | -0.017916 | 0.768687 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `vit_b_16` | 0.467851 | 0.206539 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `swin_t` | 0.542263 | 1.025336 | - | - |
| photo_2026-09-28_22-46-08 | scale_0.5 | `clip_vit_b_32` | - | - | 0.9444 | food |
| photo_2026-09-28_22-46-08 | center_crop_80 | `resnet18` | 0.135014 | 1.355728 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `densenet121` | 0.25512 | 0.737372 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `efficientnet_b0` | 0.222647 | 0.140665 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `mobilenet_v3_small` | 0.302879 | 1.42099 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `vit_b_16` | 0.075736 | 0.024672 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `swin_t` | -0.025648 | 0.192066 | - | - |
| photo_2026-09-28_22-46-08 | center_crop_80 | `clip_vit_b_32` | - | - | 0.933913 | food |
| photo_2026-09-28_22-46-08 | corner_crop | `resnet18` | 0.696705 | 0.936908 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `densenet121` | 0.49353 | 2.254176 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `efficientnet_b0` | -0.098173 | 0.334647 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `mobilenet_v3_small` | 0.683862 | 0.625608 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `vit_b_16` | 0.669658 | 0.443708 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `swin_t` | 0.300219 | 0.537712 | - | - |
| photo_2026-09-28_22-46-08 | corner_crop | `clip_vit_b_32` | - | - | 0.939116 | food |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `resnet18` | 0.082243 | 1.387222 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `densenet121` | 0.121523 | 1.286384 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `efficientnet_b0` | 0.405246 | 0.307092 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `mobilenet_v3_small` | 0.068332 | 1.062449 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `vit_b_16` | 0.190842 | 0.081353 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `swin_t` | 0.256366 | 0.95983 | - | - |
| photo_2026-09-28_22-46-08 | random_crop_80_s0 | `clip_vit_b_32` | - | - | 0.946543 | food |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `resnet18` | 0.395211 | 2.284683 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `densenet121` | 0.41414 | 1.735036 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `efficientnet_b0` | 0.104579 | -0.087958 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `mobilenet_v3_small` | 0.017862 | 0.777984 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `vit_b_16` | 0.414906 | 0.215222 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `swin_t` | 0.502009 | 1.085931 | - | - |
| photo_2026-09-28_22-46-08 | downscale_roundtrip | `clip_vit_b_32` | - | - | 0.97095 | food |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `resnet18` | 0.952147 | 3.350561 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `densenet121` | 0.327066 | 1.356772 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `efficientnet_b0` | 0.42297 | 0.461534 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `mobilenet_v3_small` | 0.164539 | 0.925017 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `vit_b_16` | 0.09965 | 0.06867 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `swin_t` | 0.399255 | 0.587759 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q95 | `clip_vit_b_32` | - | - | 0.949635 | food |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `resnet18` | 0.912053 | 3.112537 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `densenet121` | 0.215723 | 1.21209 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `efficientnet_b0` | 0.357577 | 0.312578 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `mobilenet_v3_small` | 0.114247 | 0.752361 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `vit_b_16` | 0.155636 | 0.132987 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `swin_t` | 0.372534 | 0.320458 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q85 | `clip_vit_b_32` | - | - | 0.945388 | food |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `resnet18` | 0.880446 | 3.08719 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `densenet121` | 0.157492 | 1.054254 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `efficientnet_b0` | 0.140173 | 0.074823 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `mobilenet_v3_small` | 0.230479 | 1.099112 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `vit_b_16` | 0.150252 | 0.131915 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `swin_t` | 0.468509 | 0.493182 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q75 | `clip_vit_b_32` | - | - | 0.964288 | food |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `resnet18` | 0.824907 | 2.509944 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `densenet121` | 0.367148 | 1.676831 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `efficientnet_b0` | 0.18673 | 0.065467 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `mobilenet_v3_small` | -0.019727 | 0.189843 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `vit_b_16` | -0.155386 | -0.069422 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `swin_t` | -0.01745 | 0.02346 | - | - |
| photo_2026-09-28_22-46-08 | jpeg_q50 | `clip_vit_b_32` | - | - | 0.976863 | food |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `resnet18` | 0.962181 | 3.445587 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `densenet121` | 0.318635 | 1.256387 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `efficientnet_b0` | 0.444619 | 0.499132 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `mobilenet_v3_small` | 0.112541 | 0.801568 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `vit_b_16` | 0.098117 | 0.080157 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `swin_t` | 0.576984 | 0.960596 | - | - |
| photo_2026-09-28_22-46-08 | brightness_1p05 | `clip_vit_b_32` | - | - | 0.947175 | food |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `resnet18` | 0.96301 | 3.284381 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `densenet121` | 0.271222 | 1.192838 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `efficientnet_b0` | 0.483155 | 0.556097 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `mobilenet_v3_small` | 0.166453 | 0.813391 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `vit_b_16` | 0.098704 | 0.084862 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `swin_t` | 0.557519 | 0.699203 | - | - |
| photo_2026-09-28_22-46-08 | brightness_0p95 | `clip_vit_b_32` | - | - | 0.946021 | food |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `resnet18` | 0.963346 | 3.291808 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `densenet121` | 0.272376 | 1.209814 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `efficientnet_b0` | 0.449274 | 0.503925 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `mobilenet_v3_small` | 0.170109 | 0.890203 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `vit_b_16` | 0.108354 | 0.087592 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `swin_t` | 0.524938 | 0.695078 | - | - |
| photo_2026-09-28_22-46-08 | contrast_0p95 | `clip_vit_b_32` | - | - | 0.946637 | food |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `resnet18` | 0.959863 | 3.483158 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `densenet121` | 0.313181 | 1.210156 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `efficientnet_b0` | 0.4748 | 0.569049 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `mobilenet_v3_small` | 0.165227 | 0.747137 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `vit_b_16` | 0.041053 | 0.058013 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `swin_t` | 0.589519 | 0.879087 | - | - |
| photo_2026-09-28_22-46-08 | contrast_1p05 | `clip_vit_b_32` | - | - | 0.947462 | food |
| photo_2026-09-28_22-46-08 | blur_small | `resnet18` | 0.960039 | 3.558517 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `densenet121` | 0.401775 | 1.629607 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `efficientnet_b0` | 0.434964 | 0.520469 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `mobilenet_v3_small` | 0.127075 | 0.831299 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `vit_b_16` | 0.192242 | 0.124387 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `swin_t` | 0.465479 | 0.626829 | - | - |
| photo_2026-09-28_22-46-08 | blur_small | `clip_vit_b_32` | - | - | 0.941897 | a book |
| photo_2026-09-28_22-46-08 | sharpen_small | `resnet18` | 0.950407 | 3.215017 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `densenet121` | 0.323059 | 1.330431 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `efficientnet_b0` | 0.340136 | 0.656523 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `mobilenet_v3_small` | 0.352975 | 1.225159 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `vit_b_16` | 0.073837 | 0.104894 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `swin_t` | 0.55956 | 0.593816 | - | - |
| photo_2026-09-28_22-46-08 | sharpen_small | `clip_vit_b_32` | - | - | 0.949575 | food |

## Visual quality (vs the matching baseline)

Delivery formats are compared to the original; robustness rows are
compared to the same transform applied to the original, so the number
is the protection's own cost, not the transform's.

| Variant | L-inf | PSNR dB | SSIM | LPIPS | Changed pixels |
| --- | --- | --- | --- | --- | --- |
| photo_2026-09-28_22-46-08/protected_raw | 8 | 38.3392 | 0.952783 | 0.077393 | 0.621836 |
| photo_2026-09-28_22-46-08/protected_png | 8 | 38.3392 | 0.952783 | 0.077393 | 0.621836 |
| photo_2026-09-28_22-46-08/protected_jpg | 20 | 38.2019 | 0.95061 | 0.084754 | 0.732282 |
| photo_2026-09-28_22-46-08/robustness/resize_short_side | 8 | 39.9979 | 0.969203 | 0.042785 | 0.626109 |
| photo_2026-09-28_22-46-08/robustness/scale_0.5 | 8 | 41.9141 | 0.983444 | 0.011511 | 0.624722 |
| photo_2026-09-28_22-46-08/robustness/center_crop_80 | 8 | 37.8981 | 0.95277 | 0.094194 | 0.88716 |
| photo_2026-09-28_22-46-08/robustness/corner_crop | 8 | 38.5151 | 0.960727 | 0.096354 | 0.756227 |
| photo_2026-09-28_22-46-08/robustness/random_crop_80_s0 | 8 | 38.1223 | 0.954312 | 0.087554 | 0.8538 |
| photo_2026-09-28_22-46-08/robustness/downscale_roundtrip | 8 | 43.2096 | 0.991378 | 0.028374 | 0.627354 |
| photo_2026-09-28_22-46-08/robustness/jpeg_q95 | 17 | 38.1971 | 0.950499 | 0.084659 | 0.647133 |
| photo_2026-09-28_22-46-08/robustness/jpeg_q85 | 21 | 38.3026 | 0.954409 | 0.074698 | 0.647806 |
| photo_2026-09-28_22-46-08/robustness/jpeg_q75 | 36 | 37.2 | 0.952389 | 0.069832 | 0.647466 |
| photo_2026-09-28_22-46-08/robustness/jpeg_q50 | 62 | 34.0733 | 0.942852 | 0.055454 | 0.631201 |
| photo_2026-09-28_22-46-08/robustness/brightness_1p05 | 9 | 38.1992 | 0.951301 | 0.07272 | 0.621402 |
| photo_2026-09-28_22-46-08/robustness/brightness_0p95 | 8 | 38.7621 | 0.95532 | 0.076778 | 0.621417 |
| photo_2026-09-28_22-46-08/robustness/contrast_0p95 | 8 | 38.7919 | 0.956182 | 0.074872 | 0.621655 |
| photo_2026-09-28_22-46-08/robustness/contrast_1p05 | 9 | 37.874 | 0.948267 | 0.080978 | 0.621033 |
| photo_2026-09-28_22-46-08/robustness/blur_small | 8 | 38.8913 | 0.956279 | 0.080346 | 0.62214 |
| photo_2026-09-28_22-46-08/robustness/sharpen_small | 20 | 35.6534 | 0.924952 | 0.089922 | 0.606403 |
