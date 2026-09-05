"""Declare the v10c criteria before training; judge and freeze after.

    --snapshot-only   write config_snapshot.json (curriculum, criteria, versions, command)
    --out-dir OUT     read the run's JSONs, judge against OUT/config_snapshot.json,
                      write verdict.json and, on PASS, baseline_manifest.json

Nothing here may be changed after a run has started: the verdict reads the
criteria from the snapshot the run wrote at launch, not from this file.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

CRITERIA = {
  # Training (from curriculum.jsonl, u_telemetry.jsonl, teacher.log)
  "training": {
    "must_reach_final_stage": True,
    "min_iterations_in_final_stage": 1000,
    "max_total_iterations": 9000,
    "max_frac_u_gt10_last_windows": 1e-3,   # over the last 50 telemetry windows
    "max_nan_termination_rate_last_100_iters": 0.05,
    "no_traceback": True,
  },
  # Evaluation on Mjlab-Pick-Place-PiperX-Robust (full heavy DR), 3 seeds, medians
  "eval": {
    "task": "Mjlab-Pick-Place-PiperX-Robust",
    "seeds": [101, 202, 303],
    "accept_num_envs": 256, "accept_steps": 2400,
    "min_median_success": 0.85,             # accept_s1's own gate
    "min_median_throughput_per_min": 15.0,  # the best prior state teacher under sight rewards (strong_teacher)
    "max_median_trips_per_arm_hour": 30.0,  # safety shell: at most one trip every two arm-minutes
    "min_median_late_over_early": 0.80,     # eval_endurance, 128 envs x 1200 steps
    "all_seeds_finite": True,
    "action_api_ok": True,
  },
  "checkpoint_rule": "the final checkpoint of the run; never picked by result",
}


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def versions():
  import importlib.metadata as md
  out = {"python": sys.version.split()[0]}
  for d in ("torch", "mjlab", "rsl_rl_lib", "mujoco", "mujoco_warp", "warp_lang"):
    try:
      out[d] = md.version(d)
    except Exception:
      out[d] = "unknown"
  return out


def git(*args):
  try:
    return subprocess.run(("git", *args), capture_output=True, text=True, timeout=15).stdout.strip()
  except Exception:
    return ""


def snapshot(a):
  import mjlab.tasks  # noqa: F401
  import piper_push.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from piper_push import action_api
  from piper_push import robot as piper
  from piper_push.tasks.pick_place import cold_curriculum
  cfg = load_env_cfg(a.task)
  rl = load_rl_cfg(a.task)
  home = {j: piper.PICK_HOME_KEYFRAME.joint_pos.get(j, 0.0) for j in piper.ARM_JOINT_ORDER}
  snap = {
    "tag_written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "task": a.task, "sight": bool(int(a.sight)), "seed": int(a.seed),
    "num_envs": int(a.envs), "iterations": int(a.iters), "physical_gpu": int(a.gpu),
    "command": a.command,
    "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "versions": versions(),
    "warm_start": False, "resume": False, "loaded_checkpoint": None,
    "action_api": action_api.for_convention("bounded"),
    "action_spec": piper.action_spec("bounded"),
    "policy_head": rl.actor.distribution_cfg,
    "curriculum": cold_curriculum.schedule(bool(int(a.sight))),
    "initial_reward_weights": {n: cfg.rewards[n].weight for n in cfg.rewards},
    "initial_dr": {"actuator": 0.0, "scene": 0.0, "vision": 0.0,
                   "note": "nominal ranges; heavy profile reached only through the curriculum"},
    "heavy_dr_profile": cold_curriculum.robust_cfg.HEAVY_DR_PROFILE,
    "criteria": CRITERIA,
  }
  txt = json.dumps(snap, indent=1, default=str)
  if a.out == "/dev/stdout":
    print(txt[:6000])
  else:
    with open(a.out, "w") as fh:
      fh.write(txt + "\n")
  return snap


def load(path):
  try:
    return json.load(open(path))
  except Exception:
    return None


def judge(a):
  out = a.out_dir
  snap = load(os.path.join(out, "config_snapshot.json"))
  if snap is None:
    raise SystemExit("no config_snapshot.json: nothing was declared, nothing can pass")
  crit = snap["criteria"]
  checks = {}
  # -- training
  cur = [json.loads(l) for l in open(os.path.join(out, "curriculum.jsonl")) if l.strip()] if os.path.exists(os.path.join(out, "curriculum.jsonl")) else []
  n_stages = len(snap["curriculum"]["stages"])
  final_entries = [r for r in cur if r.get("event") == "advance" and r.get("stage") == n_stages - 1]
  last_it = max((r.get("iteration", 0) for r in cur), default=0)
  entered = final_entries[0]["iteration"] if final_entries else None
  checks["reached_final_stage"] = entered is not None
  checks["iterations_in_final_stage"] = (last_it - entered) if entered is not None else 0
  checks["iterations_in_final_stage_ok"] = checks["iterations_in_final_stage"] >= crit["training"]["min_iterations_in_final_stage"]
  checks["total_iterations"] = snap["iterations"]
  checks["total_iterations_ok"] = snap["iterations"] <= crit["training"]["max_total_iterations"]
  tel = [json.loads(l) for l in open(os.path.join(out, "u_telemetry.jsonl")) if l.strip()] if os.path.exists(os.path.join(out, "u_telemetry.jsonl")) else []
  tail = [r for r in tel if r.get("tag") == "teacher"][-50:]
  frac10 = max((max(r["frac_u_gt10"]) for r in tail), default=0.0)
  checks["max_frac_u_gt10_last_windows"] = frac10
  checks["frac_u_gt10_ok"] = frac10 <= crit["training"]["max_frac_u_gt10_last_windows"]
  log = os.path.join(out, "teacher.log")
  nan_rates, tb = [], 0
  if os.path.exists(log):
    for line in open(log, errors="replace"):
      if "Termination/nan" in line:
        try:
          nan_rates.append(float(line.strip().split()[-1]))
        except ValueError:
          pass
      if "Traceback" in line:
        tb += 1
  checks["nan_termination_rate_last_100"] = float(np.mean(nan_rates[-100:])) if nan_rates else None
  checks["nan_ok"] = (checks["nan_termination_rate_last_100"] is not None
                      and checks["nan_termination_rate_last_100"] <= crit["training"]["max_nan_termination_rate_last_100_iters"])
  checks["tracebacks"] = tb
  checks["traceback_ok"] = tb == 0
  # -- eval
  acc = [load(f) for f in sorted(glob.glob(os.path.join(out, "accept_teacher_robust_s*.json")))]
  acc = [d for d in acc if d]
  end = [load(f) for f in sorted(glob.glob(os.path.join(out, "endurance_teacher_s*.json")))]
  end = [d for d in end if d]
  def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None
  succ = [d["metrics"].get("success") for d in acc]
  thr = [d["metrics"].get("throughput_per_min") for d in acc]
  trips = [d["metrics"].get("trips_per_arm_hour") for d in acc]
  loe = [d.get("late_over_early") for d in end]
  checks["eval_seeds_accept"] = len(acc); checks["eval_seeds_endurance"] = len(end)
  checks["success_per_seed"] = succ; checks["throughput_per_seed"] = thr; checks["trips_per_arm_hour_per_seed"] = trips
  checks["late_over_early_per_seed"] = loe
  checks["median_success"] = med(succ); checks["median_throughput"] = med(thr)
  checks["median_trips_per_arm_hour"] = med(trips); checks["median_late_over_early"] = med(loe)
  e = crit["eval"]
  checks["success_ok"] = checks["median_success"] is not None and checks["median_success"] >= e["min_median_success"]
  checks["throughput_ok"] = checks["median_throughput"] is not None and checks["median_throughput"] >= e["min_median_throughput_per_min"]
  checks["trips_ok"] = checks["median_trips_per_arm_hour"] is not None and checks["median_trips_per_arm_hour"] <= e["max_median_trips_per_arm_hour"]
  checks["endurance_ok"] = checks["median_late_over_early"] is not None and checks["median_late_over_early"] >= e["min_median_late_over_early"]
  checks["all_seeds_finite"] = (len(acc) == len(e["seeds"]) and len(end) == len(e["seeds"])
                                and all(x is not None and np.isfinite(x) for x in succ + thr + loe))
  api = [((d.get("domain") or {}).get("weights") or {}).get("action_api", {}).get("status") for d in acc]
  checks["action_api_status_per_seed"] = api
  checks["action_api_ok"] = len(api) > 0 and all(s == "ok" for s in api)
  verdict = all(checks[k] for k in ("reached_final_stage", "iterations_in_final_stage_ok", "total_iterations_ok", "frac_u_gt10_ok",
                                      "nan_ok", "traceback_ok", "success_ok", "throughput_ok", "trips_ok", "endurance_ok",
                                      "all_seeds_finite", "action_api_ok"))
  res = {"verdict": "PASS" if verdict else "FAIL", "judged": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "checkpoint": a.checkpoint, "checks": checks, "criteria": crit}
  with open(os.path.join(out, "verdict.json"), "w") as fh:
    fh.write(json.dumps(res, indent=1) + "\n")
  print(json.dumps({k: v for k, v in checks.items()}, indent=1, default=str))
  print("VERDICT:", res["verdict"])
  if verdict:
    man = {
      "baseline": os.path.basename(out.rstrip("/")), "frozen": res["judged"],
      "git_commit": snap["git_commit"], "checkpoint": a.checkpoint,
      "checkpoint_sha256": sha256(a.checkpoint) if a.checkpoint and os.path.exists(a.checkpoint) else None,
      "config_snapshot": "config_snapshot.json", "action_api": snap["action_api"], "action_spec": snap["action_spec"],
      "seed": snap["seed"], "command": snap["command"], "versions": snap["versions"],
      "training": {k: checks[k] for k in ("total_iterations", "iterations_in_final_stage", "max_frac_u_gt10_last_windows",
                                          "nan_termination_rate_last_100", "tracebacks")},
      "eval": {k: checks[k] for k in ("success_per_seed", "throughput_per_seed", "trips_per_arm_hour_per_seed",
                                      "late_over_early_per_seed", "median_success", "median_throughput",
                                      "median_trips_per_arm_hour", "median_late_over_early")},
      "suggested_git_tag": f"baseline/{os.path.basename(out.rstrip('/'))}",
    }
    with open(os.path.join(out, "baseline_manifest.json"), "w") as fh:
      fh.write(json.dumps(man, indent=1) + "\n")
    print("baseline_manifest.json written")
  else:
    print("no baseline manifest: this run is not a baseline and must not feed distillation")
  return res


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--snapshot-only", action="store_true")
  p.add_argument("--task"); p.add_argument("--sight", default="1"); p.add_argument("--seed", default="42")
  p.add_argument("--envs", default="8192"); p.add_argument("--iters", default="9000"); p.add_argument("--gpu", default="0")
  p.add_argument("--command", default=""); p.add_argument("--out", default="/dev/stdout")
  p.add_argument("--out-dir"); p.add_argument("--checkpoint", default=None)
  a = p.parse_args()
  if a.snapshot_only:
    snapshot(a)
  else:
    judge(a)


if __name__ == "__main__":
  main()
