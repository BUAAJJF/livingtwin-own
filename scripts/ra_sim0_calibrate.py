"""Stage 3: fit the existing parameter axes as hard as they can be fitted.

    python scripts/ra_sim0_calibrate.py --damping 1.0 --device cuda:0 \
        --rec results/ra_sim0/data/val_target_7201.pt --steps 600

The baseline the whole phase is measured against has to be the best the
current simulator can do, not a straw man.  So this is a real search: a
coordinate descent over the four command-path axes that can be retuned on a
built environment, run once per damping value because damping cannot, plus a
broad-randomisation draw and the two anchors.

Scored on **one-step NRMS of the arm's position**, on the ``val`` split and on
nothing else.  The hidden target's formula is not in the search space; that is
the point of the stage.

One process handles one damping value so the four can run on four GPUs at
once.  ``scripts/ra_sim0_gate.py`` merges their JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Pick-Place-PiperX-Vision"

LATENCY = (0, 1, 2, 3)
RESPONSE = (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)
DEADBAND = (0.000, 0.002, 0.004, 0.008, 0.015, 0.030)
LOWPASS = (None, 40.0, 20.0, 10.0, 5.0)
DAMPING = (0.75, 1.0, 1.25, 1.5)


def build_env(rec_meta: dict, damping: float, device: str, hidden: bool,
              num_envs: int):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import perturb
  from piper_push.hidden_plant import HiddenPlantCfg, apply_hidden_plant

  SHAPE_PRESETS = {"all": None, "train": (0.34, 0.22, 0.16, 0.0, 0.0),
                   "holdout": (0.0, 0.0, 0.0, 0.16, 0.12)}

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = rec_meta["seed"]
  w = SHAPE_PRESETS[rec_meta["shapes"]]
  if w is not None:
    for name, ev in cfg.events.items():
      if name.startswith("object_shape"):
        ev.params = dict(ev.params)
        ev.params["shape_weights"] = tuple(w)
  if damping != 1.0:
    # Through perturb's own axis rather than by hand: it scales the derivative
    # gain on every non-gripper actuator group, which is what
    # servo_damping_scale has meant since Phase WM0, and it is tested.
    perturb.apply_session_mismatch(
      cfg, perturb.SessionMismatchCfg(servo_damping_scale=damping))
  if hidden:
    apply_hidden_plant(cfg, HiddenPlantCfg())
  # A replay must not be allowed to terminate.  A candidate whose command
  # path is badly wrong trips the safety shell within a few steps, resets, and
  # draws a fresh object -- so the arms that are worst at the task would be
  # scored on the handful of environments that happened to survive, and the
  # oracle on all of them.  Measured before this line existed: the nominal
  # simulator kept 0.9% of its steps and the oracle 88.6%, which is not a
  # comparison of accuracy at all.  With no termination terms nothing resets,
  # every candidate is scored on exactly the window the RECORDING allows, and
  # what a candidate would have tripped is reported through the velocity
  # statistics instead.
  cfg.terminations = {}
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def score(harness, rec, plant: dict, gripper_rate: float | None = None
          ) -> dict:
  """One-step NRMS of the arm's position under one plant setting."""
  import sys
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import replay as rp

  arm = harness.env.action_manager.get_term("arm")
  arm.set_plant(latency_steps=plant["latency_steps"],
                response_scale=plant["response_scale"],
                deadband=plant["deadband"],
                lowpass_hz=plant["lowpass_hz"])
  if gripper_rate is not None:
    grip = harness.env.action_manager.get_term("gripper")
    if not hasattr(grip, "_ra_base"):
      grip._ra_base = grip._max_step.clone()
    grip._max_step = grip._ra_base * float(gripper_rate)
  out = harness.run(rec, period=1)
  ok, hor = rp.segment_mask(rec, out["done"], 1, out.get("pristine"))
  q = rp.nrms(out["q"], rec, ok, hor, 1, 1, "q")
  qd = rp.nrms(out["qd"], rec, ok, hor, 1, 1, "qd")
  return {"q": q, "qd": qd, "objective": q["nrms"]}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rec", required=True)
  ap.add_argument("--damping", type=float, required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--steps", type=int, default=600)
  ap.add_argument("--rounds", type=int, default=3)
  ap.add_argument("--dr-draws", type=int, default=0)
  ap.add_argument("--oracle", action="store_true")
  ap.add_argument("--out", default="results/ra_sim0/calibration")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / "src"))
  from piper_push import replay as rp

  rec, meta = rp.Recording.load(a.rec, steps=a.steps)
  env = build_env(meta, a.damping, a.device, a.oracle, rec.num_envs)
  harness = rp.ReplayHarness(env)
  t0 = time.time()
  trace = []

  def ev(plant, tag, gripper_rate=None):
    s = score(harness, rec, plant, gripper_rate)
    row = {"tag": tag, "damping": a.damping,
           "gripper_rate_scale": gripper_rate, **plant,
           "nrms_q": s["q"]["nrms"], "nrms_qd": s["qd"]["nrms"],
           "rms_q": s["q"]["rms"], "n": s["q"]["n"],
           "per_joint_rms": s["q"]["per_joint_rms"]}
    trace.append(row)
    print(f"  [{time.time() - t0:6.0f}s] {tag:28s} nrms_q={row['nrms_q']:.4f}"
          f"  nrms_qd={row['nrms_qd']:.4f}", flush=True)
    return row

  nominal = {"latency_steps": 0, "response_scale": 1.0, "deadband": 0.0,
             "lowpass_hz": None}
  if a.oracle:
    ev(dict(nominal), "oracle")
  else:
    best = ev(dict(nominal), "nominal")
    cur = dict(nominal)
    axes = (("latency_steps", LATENCY), ("response_scale", RESPONSE),
            ("deadband", DEADBAND), ("lowpass_hz", LOWPASS))
    single = []
    for r in range(a.rounds):
      improved = False
      for name, values in axes:
        for v in values:
          if v == cur[name]:
            continue
          cand = dict(cur)
          cand[name] = v
          row = ev(cand, f"r{r}:{name}={v}")
          if r == 0 and all(cand[k] == nominal[k] for k in cand if k != name):
            single.append(row)
          if row["nrms_q"] < best["nrms_q"] - 1e-6:
            best, cur, improved = row, cand, True
      if not improved:
        break
    if single:
      s1 = min(single, key=lambda r: r["nrms_q"])
      s1 = dict(s1)
      s1["tag"] = "param_1d_best"
      trace.append(s1)
    # The gripper's rate is in the pre-registered search space, so it is
    # searched -- at the best arm plant, once, rather than inside the descent.
    # The hidden target does not touch the gripper's command path, so this
    # sweep is expected to be flat and is reported either way.
    for gr in (0.5, 0.75, 1.0, 1.25, 1.5):
      ev(dict(cur), f"gripper_rate={gr}", gripper_rate=gr)
    ev(dict(cur), "gripper_rate=1.0_restored", gripper_rate=1.0)
    if a.dr_draws:
      # Broad randomisation, scored as the phase asks: the best single
      # predictor a broad prior contains, not the average of one.
      g = torch.Generator().manual_seed(4242 + int(a.damping * 100))
      for i in range(a.dr_draws):
        cand = {
          "latency_steps": int(torch.randint(0, 4, (1,), generator=g)),
          "response_scale": float(0.3 + 0.7 * torch.rand(1, generator=g)),
          "deadband": float(0.03 * torch.rand(1, generator=g)),
          "lowpass_hz": [None, 40.0, 20.0, 10.0, 5.0][
            int(torch.randint(0, 5, (1,), generator=g))],
        }
        ev(cand, f"dr{i}")

  env.close()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  name = f"damping{a.damping}{'_oracle' if a.oracle else ''}"
  payload = {"damping": a.damping, "oracle": a.oracle, "rec": a.rec,
             "rec_meta": {k: meta[k] for k in
                          ("split", "seed", "shapes", "domain", "num_envs")},
             "steps_scored": a.steps, "wall_clock_s": time.time() - t0,
             "trace": trace}
  (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
  print(f"wrote {out / (name + '.json')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
