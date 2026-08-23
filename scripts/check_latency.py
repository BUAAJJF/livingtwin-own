"""In-simulator plumbing check: is the camera actually late, and by how much?

    python scripts/check_latency.py --device cuda:0 --probs 0.4,0,0,0.3,0.3

The unit tests check the delay buffer's arithmetic and the term's bookkeeping.
Neither of them can see whether the observation the *policy* receives from a
running environment is the frame from `lag` control steps ago, which is the
only thing any of this is about.

The check installs a second, undelayed copy of the camera term alongside the
delayed one in the same environment, so the comparison is between two
observations of one simulation rather than between two runs of a simulator
that is not bitwise reproducible. For every environment and every step it then
asks: does the delayed group equal the reference group from `lag_e` steps ago,
and does it equal it at *any other* shift? A term that delayed everything by a
constant regardless of its per-environment assignment would pass the first
question and fail the second.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import load_env_cfg

from piper_push import latency

TASK = "Mjlab-Pick-Place-PiperX-Vision"


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=TASK)
  p.add_argument("--probs", default="0.4,0.0,0.0,0.3,0.3")
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--steps", type=int, default=40)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=7)
  p.add_argument("--json", default=None)
  a = p.parse_args()

  raw = [float(x) for x in a.probs.split(",")]
  prior = latency.LatencyPrior(tuple(x / sum(raw) for x in raw))

  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  # The undelayed reference, added before the prior is installed so that it is
  # a copy of the untouched term.
  cfg.observations["camera_ref"] = ObservationGroupCfg(
    terms=copy.deepcopy(dict(cfg.observations["camera"].terms)),
    enable_corruption=False, concatenate_terms=True)
  applied = latency.apply_latency_prior(cfg, prior, seed=a.seed)
  print(f"  prior {prior.probs}  ->  {applied.get('latency_prior', {})}")

  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  u = env.unwrapped
  term = u.observation_manager._group_obs_term_cfgs["camera"][0].func
  buf = u.observation_manager._group_obs_term_delay_buffer["camera"]["scene"]

  obs, _ = env.reset()
  hist: list[torch.Tensor] = []
  matches = torch.zeros(a.num_envs, len(latency.LAGS), dtype=torch.long)
  seen = 0
  g = torch.Generator(device=a.device).manual_seed(a.seed)
  for t in range(a.steps):
    lags = term.lags.clone()
    if not torch.equal(lags, buf.current_lags):
      raise SystemExit(
        f"step {t}: the term's lags {lags[:8].tolist()} are not the buffer's "
        f"{buf.current_lags[:8].tolist()}; the assignment is not reaching the "
        "delay stage")
    ref = obs["camera_ref"].clone()
    got = obs["camera"].clone()
    hist.append(ref)
    if t >= max(latency.LAGS):
      seen += 1
      for ci, c in enumerate(latency.LAGS):
        same = (got - hist[-1 - c]).abs().amax(dim=-1) < 1e-5
        matches[:, ci] += same.long().cpu()
    act = torch.randn(a.num_envs, u.action_manager.total_action_dim,
                      device=a.device, generator=g) * 0.1
    obs, *_ = env.step(act)

  lags = term.lags.cpu()
  ok, bad = 0, []
  for b in range(a.num_envs):
    want = latency.LAGS.index(int(lags[b]))
    row = matches[b]
    if int(row[want]) == seen and all(int(row[j]) < seen
                                      for j in range(len(latency.LAGS))
                                      if j != want):
      ok += 1
    else:
      bad.append({"env": b, "lag": int(lags[b]),
                  "matches_by_shift": row.tolist(), "of": seen})

  print(f"  {seen} comparable steps x {a.num_envs} environments")
  print(f"  lag assignment: "
        f"{torch.bincount(lags, minlength=len(latency.LAGS)).tolist()}")
  print(f"  environments whose camera matches its own lag and no other: "
        f"{ok}/{a.num_envs}")
  for r in bad[:5]:
    print(f"    env {r['env']} lag {r['lag']}: {r['matches_by_shift']} "
          f"of {r['of']}")
  env.close()

  out = {"prior": prior.to_json(), "num_envs": a.num_envs,
         "comparable_steps": seen, "exact": ok,
         "assignment": torch.bincount(lags, minlength=len(latency.LAGS)).tolist(),
         "failures": bad}
  if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"  wrote {a.json}")
  return 0 if ok == a.num_envs else 1


if __name__ == "__main__":
  raise SystemExit(main())
