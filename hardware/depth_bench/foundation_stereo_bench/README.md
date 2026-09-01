# FoundationStereo / RealSense D455 benchmark

This directory contains an isolated feasibility benchmark for using the two
rectified D455 infrared streams as FoundationStereo input. No existing deploy
path is modified.

## Measured host

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB, 160 W power limit
- Driver: 580.173.02
- PyTorch: 2.8.0+cu128, FP16 autocast, batch size 1
- Input: 848 x 480, resized official FoundationStereo demo stereo pair
- FoundationStereo commit: `6e8806816b533e4d13ddbb95ffa907b797060a62`
- xFormers disabled; these are conservative PyTorch-path measurements

Median latency in milliseconds (20 samples for ViT-S, 15 for ViT-L):

| Model | Iterations | GPU forward | CPU-frame pipeline | FPS | Peak allocated |
|---|---:|---:|---:|---:|---:|
| ViT-S | 8 | 266.9 | 269.9 | 3.75 | 2.04 GiB |
| ViT-S | 16 | 376.9 | 378.1 | 2.65 | 2.04 GiB |
| ViT-S | 32 | 567.6 | 573.3 | 1.76 | 2.04 GiB |
| ViT-L | 8 | 390.9 | 407.9 | 2.56 | 3.26 GiB |
| ViT-L | 16 | 512.7 | 509.6 | 1.95 | 3.26 GiB |
| ViT-L | 32 | 702.0 | 704.3 | 1.42 | 3.26 GiB |

`GPU forward` uses CUDA events with resident inputs. `CPU-frame pipeline` uses
wall time and includes NumPy-to-tensor conversion, H2D, padding, inference,
unpadding, and D2H, but not disk decode or camera acquisition.

Machine-readable results are under `results/`.

## Capture a synchronized D455 sample

The current project logs contain native depth and one grayscale frame only.
FoundationStereo requires both rectified infrared images, so a genuine
side-by-side comparison needs a new capture:

```bash
python capture_d455.py --out results/d455_sample
```

Then pass the printed `left_intrinsics.fx`, `baseline_m`, and `depth_scale_m`
to the benchmark:

```bash
python benchmark.py \
  --repo /path/to/FoundationStereo \
  --checkpoint /path/to/model_best_bp2.pth \
  --left results/d455_sample/left_ir.png \
  --right results/d455_sample/right_ir.png \
  --native-depth results/d455_sample/native_depth_z16.png \
  --native-depth-scale DEPTH_SCALE \
  --fx FX --baseline-m BASELINE_M \
  --out results/d455_comparison
```

This writes `native_vs_foundation_stereo.png` with a shared metric-depth color
scale, plus disparity/depth arrays and overlap/error metrics.

## Deployment caveats

- The official FoundationStereo research weights are non-commercial research
  only. Product deployment needs a separately licensed path such as NVIDIA TAO.
- The original PyTorch model is not real-time at D455 resolution on this host.
  Evaluate Fast-FoundationStereo/TensorRT before integrating it into the sensor
  loop, and keep native D455 depth as the low-latency fallback.
- Depth is `fx * baseline / disparity`; use the live stream profile rather than
  hard-coding D455 calibration.
