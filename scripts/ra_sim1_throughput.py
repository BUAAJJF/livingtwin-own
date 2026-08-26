"""G6: what the actuator model costs, at 64, 256 and 512 environments.

    python scripts/ra_sim1_throughput.py --actuator <ckpt> --sizes 64 256 512

The same scripted command stream through the same scene, with and without the
model, timed after a warm-up so the measurement is the steady state and not
Warp's kernel compilation.  An out-of-memory at a size is recorded as such
rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def bench(num_envs, device, seed, actuator, steps=200, warmup=40) -> dict:
  import sys
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / "src"))
  import importlib.util
  spec = importlib.util.spec_from_file_location(
    "ra1_stress", ROOT / "scripts" / "ra_sim1_stress.py")
  st = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(st)
  env = st.build(num_envs, device, seed, "holdout", actuator=actuator)
  n_act = env.action_manager.total_action_dim
  acts = st.scripted("S2", num_envs, n_act, steps, device, seed)
  env.reset()
  with torch.no_grad():
    for t in range(warmup):
      env.step(acts[t])
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(warmup, steps):
      env.step(acts[t])
    torch.cuda.synchronize()
  dt = time.time() - t0
  mib = torch.cuda.max_memory_allocated() / 2**20
  env.close()
  torch.cuda.reset_peak_memory_stats()
  return {"env_steps_per_s": (steps - warmup) * num_envs / max(dt, 1e-9),
          "gpu_mib": mib}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--actuator", required=True)
  ap.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 512])
  ap.add_argument("--seed", type=int, default=7801)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", default="results/ra_sim1/throughput.json")
  a = ap.parse_args()
  out: dict = {}
  for n in a.sizes:
    row = {}
    for name, ck in (("nominal", None), ("actuator", a.actuator)):
      try:
        r = bench(n, a.device, a.seed, ck)
        row[name] = r["env_steps_per_s"]
        row[f"{name}_gpu_mib"] = r["gpu_mib"]
      except Exception as exc:                       # OOM is a result
        row[name] = None
        row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]
      print(f"  {n} envs {name}: {row.get(name)}", flush=True)
    if row.get("nominal") and row.get("actuator"):
      row["throughput_loss"] = 1.0 - row["actuator"] / row["nominal"]
    out[str(n)] = row
  p = Path(a.out)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
