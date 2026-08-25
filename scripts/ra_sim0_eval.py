"""Stage 5: how wrong is each candidate simulator, on data it has never seen.

    python scripts/ra_sim0_eval.py --rec results/ra_sim0/data/test_target_7401.pt \
        --candidate residual --residual results/ra_sim0/model/residual_seed0.pt

One candidate per process, so six candidates fit on four GPUs.  Every number
below comes from a real MJWarp rollout with the recorded action stream played
through it; nothing is predicted by the surrogate here.

Two passes.  ``P = 1`` resynchronises every step and gives the one-step error
at every step; ``P = 25`` resynchronises every twenty-five and gives horizons
1 to 25 from each anchor, which is where the 10-step and 25-step numbers come
from.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Pick-Place-PiperX-Vision"
SHAPE_PRESETS = {"all": None, "train": (0.34, 0.22, 0.16, 0.0, 0.0),
                 "holdout": (0.0, 0.0, 0.0, 0.16, 0.12)}


def build(meta, *, damping=1.0, hidden=False, residual=None, num_envs=64,
          device="cuda:0"):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import perturb
  from piper_push.hidden_plant import HiddenPlantCfg, apply_hidden_plant
  from piper_push.residual import ResidualHookCfg, apply_residual

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = meta["seed"]
  w = SHAPE_PRESETS[meta["shapes"]]
  if w is not None:
    for name, ev in cfg.events.items():
      if name.startswith("object_shape"):
        ev.params = dict(ev.params)
        ev.params["shape_weights"] = tuple(w)
  if damping != 1.0:
    perturb.apply_session_mismatch(
      cfg, perturb.SessionMismatchCfg(servo_damping_scale=damping))
  if hidden:
    apply_hidden_plant(cfg, HiddenPlantCfg())
  if residual:
    apply_residual(cfg, ResidualHookCfg(checkpoint=str(residual)))
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rec", required=True)
  ap.add_argument("--candidate", required=True)
  ap.add_argument("--damping", type=float, default=1.0)
  ap.add_argument("--latency", type=int, default=0)
  ap.add_argument("--response", type=float, default=1.0)
  ap.add_argument("--deadband", type=float, default=0.0)
  ap.add_argument("--lowpass", type=float, default=-1.0)
  ap.add_argument("--hidden-target", action="store_true")
  ap.add_argument("--residual", default="")
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--tag", default="")
  ap.add_argument("--out", default="results/ra_sim0/accuracy")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import replay as rp

  rec, meta = rp.Recording.load(a.rec, steps=a.steps)
  env = build(meta, damping=a.damping, hidden=a.hidden_target,
              residual=a.residual or None, num_envs=rec.num_envs,
              device=a.device)
  arm = env.action_manager.get_term("arm")
  arm.set_plant(latency_steps=a.latency, response_scale=a.response,
                deadband=a.deadband,
                lowpass_hz=None if a.lowpass < 0 else a.lowpass)
  harness = rp.ReplayHarness(env)
  t0 = time.time()

  res_hook = None
  for h in arm._hooks:
    if hasattr(h, "last_spread"):
      res_hook = h

  results: dict = {
    "candidate": a.candidate, "tag": a.tag or a.candidate,
    "rec": a.rec, "steps": rec.steps, "num_envs": rec.num_envs,
    "rec_meta": {k: meta[k] for k in
                 ("split", "seed", "shapes", "domain", "budget_s")},
    "plant": {"damping": a.damping, "latency_steps": a.latency,
              "response_scale": a.response, "deadband": a.deadband,
              "lowpass_hz": None if a.lowpass < 0 else a.lowpass,
              "hidden_target": a.hidden_target,
              "residual": a.residual or None},
  }

  for period, horizons in ((1, (1,)), (25, (1, 5, 10, 25))):
    spread_log, delta_log = [], []
    if res_hook is not None:
      orig = harness.run

      def run_logging(r, p, log_every=0, _orig=orig):
        return _orig(r, p, log_every)

    tp = time.time()
    if res_hook is None:
      out = harness.run(rec, period=period, log_every=1000)
    else:
      # the same loop, with the ensemble's disagreement recorded alongside
      dev = env.device
      pq, pqd, cdone = [], [], []
      env.reset()
      with torch.no_grad():
        for t in range(rec.steps):
          if t % period == 0:
            harness.write_state(rec.q[t].to(dev), rec.qd[t].to(dev),
                                rec.gq[t].to(dev), rec.obj[t].to(dev))
          o = env.step(rec.a[t].to(dev))
          cdone.append(o[2].bool().cpu())
          pq.append(harness.robot.data.joint_pos[:, harness.arm_ids].clone().cpu())
          pqd.append(harness.robot.data.joint_vel[:, harness.arm_ids].clone().cpu())
          spread_log.append(res_hook.last_spread.clone().cpu())
          delta_log.append(res_hook.last_delta.clone().cpu())
      out = {"q": torch.stack(pq), "qd": torch.stack(pqd),
             "done": torch.stack(cdone)}
    wall = time.time() - tp
    ok, hor = rp.segment_mask(rec, out["done"], period)
    block = {"period": period,
             "env_steps_per_s": rec.steps * rec.num_envs / max(wall, 1e-9),
             "wall_clock_s": wall}
    for h in horizons:
      block[f"h{h}"] = {"q": rp.nrms(out["q"], rec, ok, hor, period, h, "q"),
                        "qd": rp.nrms(out["qd"], rec, ok, hor, period, h, "qd")}
    # divergence: the first horizon whose median absolute position error
    # crosses 10 mrad, which is the scale of the target's own backlash band.
    if period > 1:
      div = None
      for h in range(1, period + 1):
        sel = ok[:rec.steps - 1] & (hor[:rec.steps - 1] == h)
        if not bool(sel.any()):
          continue
        idx = sel.nonzero(as_tuple=False)
        e = (out["q"][idx[:, 0], idx[:, 1]]
             - rec.q[idx[:, 0] + 1, idx[:, 1]]).abs().median()
        if float(e) > 0.010 and div is None:
          div = h
      block["divergence_steps_at_10mrad"] = div

    # physical sanity, on the candidate's own states
    q, qd = out["q"], out["qd"]
    block["sanity"] = {
      "finite": bool(torch.isfinite(q).all() and torch.isfinite(qd).all()),
      "max_abs_qd": float(qd.abs().max()),
      "p999_abs_qd": float(qd.abs().flatten().quantile(0.999)),
      "max_step_jump_rad": float((q[1:] - q[:-1]).abs().max()),
    }
    # sub-regions the hidden target actually lives in
    flip = rp.reversal_mask(rec)
    dead = rp.deadband_mask(rec)
    for name, mask in (("reversal", flip), ("deadband", dead)):
      T = rec.steps - 1
      sel = ok[:T].unsqueeze(-1) & mask[:T] & (hor[:T] == max(horizons)).unsqueeze(-1)
      if bool(sel.any()):
        err = (out["q"][:T] - rec.q[1:T + 1])[sel]
        block[f"{name}_rms"] = float(err.pow(2).mean().sqrt())
        block[f"{name}_n"] = int(sel.sum())
      else:
        block[f"{name}_rms"] = float("nan")
        block[f"{name}_n"] = 0
    if spread_log:
      sp = torch.stack(spread_log)[:rec.steps - 1]
      dl = torch.stack(delta_log)[:rec.steps - 1]
      T = rec.steps - 1
      sel = ok[:T] & (hor[:T] == max(horizons))
      if bool(sel.any()):
        i = sel.nonzero(as_tuple=False)
        e = (out["q"][i[:, 0], i[:, 1]] - rec.q[i[:, 0] + 1, i[:, 1]]).abs()
        s = sp[i[:, 0], i[:, 1]]
        ef, sf = e.flatten(), s.flatten()
        c = float(((ef - ef.mean()) * (sf - sf.mean())).mean()
                  / (ef.std() * sf.std() + 1e-12))
        block["ensemble"] = {
          "spread_error_correlation": c,
          "mean_spread_rad": float(s.mean()),
          "mean_abs_delta_rad": float(dl[i[:, 0], i[:, 1]].abs().mean()),
          "p99_abs_delta_rad": float(dl.abs().flatten().quantile(0.99)),
          "coverage_at_2_sigma": float((ef <= 2.0 * sf).float().mean()),
        }
    results[f"period{period}"] = block
    print(f"  P={period} done in {wall:.0f}s", flush=True)

  if torch.cuda.is_available():
    results["gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
  results["wall_clock_s"] = time.time() - t0
  env.close()
  out_dir = Path(a.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  name = f"{results['tag']}_{meta['split']}.json"
  (out_dir / name).write_text(json.dumps(results, indent=2))
  print(json.dumps({"tag": results["tag"],
                    "one_step_nrms_q": results["period1"]["h1"]["q"]["nrms"],
                    "h10": results["period25"]["h10"]["q"]["nrms"],
                    "h25": results["period25"]["h25"]["q"]["nrms"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
