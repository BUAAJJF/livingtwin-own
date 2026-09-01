# Fast-FoundationStereo benchmark

Host: RTX 5090 Laptop GPU (24 GiB, 160 W), PyTorch 2.8.0+cu128, FP16,
batch size 1. Input is 848 x 480 and is padded to 864 x 480. Results use the
Triton cost-volume backend, 10 warmups, and 50 measured runs.

| Checkpoint | Iterations | GPU median | Pipeline median | GPU p95 | FPS | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| 20-30-48 (fastest) | 4 | 34.69 ms | 37.72 ms | 35.93 ms | 28.82 | 864 MiB |
| 20-30-48 (fastest) | 8 | 41.62 ms | 44.64 ms | 43.34 ms | 24.03 | 864 MiB |
| 20-26-39 (balanced) | 4 | 46.37 ms | 48.26 ms | 47.82 ms | 21.57 | 873 MiB |
| 20-26-39 (balanced) | 8 | 53.42 ms | 55.14 ms | 55.27 ms | 18.72 | 872 MiB |
| 23-36-37 (accurate) | 4 | 49.94 ms | 51.88 ms | 52.26 ms | 20.02 | 873 MiB |
| 23-36-37 (accurate) | 8 | 62.20 ms | 64.21 ms | 63.49 ms | 16.08 | 873 MiB |

The one-time Triton JIT compilation took about 10.7 seconds and is excluded.
TensorRT was not measured because TensorRT and `trtexec` are not installed on
this host. Each result directory contains the full JSON statistics, disparity
array, and visualization.
