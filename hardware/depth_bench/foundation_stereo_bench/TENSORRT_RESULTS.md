# Fast-FoundationStereo TensorRT benchmark

Measured on 2026-09-01 with an NVIDIA GeForce RTX 5090 Laptop GPU (24 GiB,
160 W), TensorRT 11.2.1.2, PyTorch 2.8.0+cu128, and batch size 1. The source
images are resized to 848 x 480, padded with replicated edge pixels to the
engine's fixed 864 x 480 input, and ImageNet-normalized outside the engine.

The engine uses the fastest `20-30-48` checkpoint, max disparity 192, and an
explicit ModelOpt-selected FP16/FP32 mixed-precision ONNX. TensorRT 11 strongly
typed engines do not use the removed legacy `BuilderFlag.FP16` switch.

| Backend | GRU iterations | GPU median | Pipeline median | GPU p95 | FPS |
|---|---:|---:|---:|---:|---:|
| PyTorch/Triton FP16 | 4 | 34.69 ms | 37.72 ms | 35.93 ms | 28.82 |
| TensorRT mixed FP16 | 4 | **14.42 ms** | **16.32 ms** | 15.85 ms | **69.32** |
| PyTorch/Triton FP16 | 8 | 41.62 ms | 44.64 ms | 43.34 ms | 24.03 |
| TensorRT mixed FP16 | 8 | **17.32 ms** | **18.86 ms** | 21.00 ms | **57.74** |

GPU timing uses CUDA events with preprocessed input and output buffers resident
on the GPU. Pipeline wall time includes CPU tensor conversion, host-to-device
copy, normalization, padding, inference, cropping, and device-to-host copy, but
not image decoding. Each reported TensorRT row is 10 warmups followed by 50
measured runs. A second 50-run test on a dedicated non-default CUDA stream gave
14.53 ms median for the 4-iteration engine, confirming that the result is not a
default-stream timing artifact.

The 4-iteration engine is 50.8 MiB and took about 56.5 seconds to build. The
8-iteration engine is 51.4 MiB and took 58.8 seconds. Incremental GPU residency
after engine load, buffers, and one execution was approximately 424 MiB and
512 MiB respectively. This includes allocations outside PyTorch's allocator.

## Numerical parity

TensorRT output was compared pixel-for-pixel with the corresponding
PyTorch/Triton FP16 output on the same stereo pair.

| Iterations | MAE | Median abs. | p95 abs. | Pixels > 1 px | Max abs. |
|---|---:|---:|---:|---:|---:|
| 4 | 0.0571 px | 0.0337 px | 0.1441 px | 0.431% | 18.48 px |
| 8 | 0.0553 px | 0.0338 px | 0.1332 px | 0.410% | 13.13 px |

The largest differences are sparse boundary/occlusion outliers; 99% of pixels
are within 0.47 px for the 4-iteration engine and within 0.44 px for the
8-iteration engine.

## Artifacts

- `build_trt_engine.py`: TensorRT 11 strongly typed engine builder.
- `benchmark_trt.py`: reproducible GPU and pipeline benchmark.
- `results/trt_fast_20-30-48_i4_480x864/`: 4-iteration ONNX and engine.
- `results/trt_fast_20-30-48_i8_480x864/`: 8-iteration ONNX, engine, and build metadata.
- `results/trt_fast_20-30-48_i{4,8}_848x480/`: raw disparity and benchmark JSON.
- `results/trt_vs_pytorch_disparity.png`: shared-scale output and error comparison.

TensorRT and ModelOpt are isolated under `_deps_trt` rather than installed into
the active Conda environment. That directory occupies approximately 6.8 GiB,
mostly because NVIDIA's Python TensorRT runtime wheel bundles CUDA libraries.
Use `_deps_trt` before `_deps` on `PYTHONPATH` when running the scripts.
