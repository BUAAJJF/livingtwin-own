"""Why does the policy stop starting?  Attempts, waits, stalls and what the cloud showed of the target.

Every throughput number so far is a mean over a window, and the first
generation's within-episode decay was read off ``late/early`` alone.  This
rollout keeps the per-step record the mean throws away and reports, for one
checkpoint on one task:

    placed/min, success per attempt, attempts/min, late/early (placements and attempts)
    wait from a placement (and from a drop) to the next attempt
    stretches during which an object waits on the table and the hand does not engage it
    the jaw command against the measured opening, by phase
    target switches (the command's), and the target's visible pixels / sampled points, by phase
    drops, safety-shell trips, terminations by cause

Definitions, so that "no object to grab" is never scored as a stall:

* ``engaged``   not grasped and the grasp site within ``--reach-m`` of the target
                (``PickCommandCfg.grasp_reach_m``, 90 mm).  An *attempt* is the step
                ``engaged`` turns on.
* ``carry``     ``cmd.grasped`` -- a secure grasp by the task's own test.
* ``place``     not grasped, and the object is off the table (released, in flight)
                or inside the bin footprint before the placement registers.
* ``approach``  everything else while the object sits on the table: the hand is
                free, an object is waiting, the policy is (or is not) going for it.
                ``first`` before the episode's first placement, ``re`` after one.
* a *stall* is a run of consecutive approach steps of at least ``--idle-s``
                seconds; runs are ended by an attempt, by a reset (termination or
                time-out) or by the end of the rollout (censored, counted apart).

With one object on the table (every task here) the table is never empty: the
object is put back the step a placement registers.  A reset is either a
termination (``object_lost``, ``over_speed``, ``nan``) or the play config's
time-out; ``--episode-length-s`` above the run length turns the time-out off,
which is what the long-horizon evaluation wants.

    python scripts/pc/eval_initiation.py --checkpoint X --task Mjlab-Pick-Place-PiperX-PC-P1B-Vision \\
        --num-envs 256 --steps 1800 --seed 101 --out initiation_s101.json
    python scripts/pc/eval_initiation.py ... --steps 9000 --episode-length-s 1000000 --out long_s101.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASES = ("approach_first", "approach_re", "engaged", "carry", "place")


def _runs(mask_1d: np.ndarray):
  """Start, length and the index one past the end for each run of True in a 1-D mask."""
  m = np.concatenate([[False], mask_1d.astype(bool), [False]])
  d = np.diff(m.astype(np.int8))
  starts = np.flatnonzero(d == 1)
  ends = np.flatnonzero(d == -1)
  return starts, ends - starts, ends


def _pct(xs, q):
  xs = np.asarray(xs, dtype=np.float64)
  return float(np.percentile(xs, q)) if xs.size else float("nan")


def summarise(tr: dict, dt: float, window: int, idle_s: float, drop_grace_s: float = 3.0) -> dict:
  """The verdict from the per-step record.  Pure numpy so an off-by-one is testable.

  ``tr`` holds ``(T, B)`` arrays: ``grasped``, ``engaged``, ``on_table``,
  ``in_bin``, ``placed`` (the just_placed edge), ``reset`` (the env was reset
  AFTER this step's action), ``term`` (int code, 0 none), ``fresh``,
  ``target_full``, ``target_sampled``, ``jaw_cmd``, ``jaw_meas``, ``dist``,
  ``target_idx``.
  """
  g = tr["grasped"].astype(bool)
  eng = tr["engaged"].astype(bool) & ~g
  placed = tr["placed"].astype(bool)
  reset = tr["reset"].astype(bool)
  on_table = tr["on_table"].astype(bool)
  in_bin = tr["in_bin"].astype(bool)
  T, B = g.shape
  place = ~g & (~on_table | in_bin)
  approach = ~g & ~eng & ~place
  # placements since the last reset, per step (counted BEFORE this step's placement)
  since = np.zeros((T, B), dtype=np.int32)
  acc = np.zeros(B, dtype=np.int32)
  for t in range(T):
    since[t] = acc
    acc = acc + placed[t].astype(np.int32)
    acc[reset[t]] = 0
  approach_first = approach & (since == 0)
  approach_re = approach & (since > 0)
  phase = np.full((T, B), -1, dtype=np.int8)
  for i, m in enumerate((approach_first, approach_re, eng, g, place)):
    phase[m] = i

  # -- events
  prev_eng = np.concatenate([np.zeros((1, B), bool), eng[:-1]])
  prev_reset = np.concatenate([np.zeros((1, B), bool), reset[:-1]])
  attempt = eng & ~prev_eng & ~prev_reset
  prev_g = np.concatenate([np.zeros((1, B), bool), g[:-1]])
  grasp = g & ~prev_g
  release = ~g & prev_g & ~prev_reset
  # drops: a release not followed by a placement or a re-grasp within the grace window
  grace = int(round(drop_grace_s / dt))
  drop = np.zeros((T, B), bool)
  for b in range(B):
    for t in np.flatnonzero(release[:, b]):
      end = min(T, t + grace + 1)
      seg_p = placed[t:end, b]
      seg_g = grasp[t + 1:end, b]
      seg_r = reset[t:end, b]
      # first event in the window decides
      tp = np.flatnonzero(seg_p); tg = np.flatnonzero(seg_g) + 1; trr = np.flatnonzero(seg_r)
      first = min([x for x in (tp[:1], tg[:1], trr[:1]) if x.size], key=lambda x: x[0], default=None)
      if first is None:
        if end - t > grace:       # the whole window elapsed with nothing: a drop
          drop[t, b] = True
        # else censored by the end of the rollout
      elif trr.size and first[0] == trr[0] and not (tp.size and tp[0] <= trr[0]):
        drop[t, b] = True         # the episode ended (object lost) before a placement or re-grasp

  # -- waits: from a placement / a drop to the next attempt in the same episode
  def waits(after: np.ndarray):
    out, censored = [], 0
    for b in range(B):
      a_idx = np.flatnonzero(attempt[:, b])
      r_idx = np.flatnonzero(reset[:, b])
      for t in np.flatnonzero(after[:, b]):
        nxt = a_idx[a_idx > t]
        rs = r_idx[r_idx >= t]
        if nxt.size and (not rs.size or nxt[0] <= rs[0]):
          out.append((nxt[0] - t) * dt)
        else:
          censored += 1
    return out, censored

  w_succ, c_succ = waits(placed)
  w_drop, c_drop = waits(drop)
  # time from a reset (episode start) to the first attempt
  first_attempt, first_censored = [], 0
  for b in range(B):
    starts = np.concatenate([[0], np.flatnonzero(reset[:, b]) + 1])
    starts = starts[starts < T]
    a_idx = np.flatnonzero(attempt[:, b])
    r_idx = np.flatnonzero(reset[:, b])
    for s0 in starts:
      nxt = a_idx[a_idx >= s0]
      rs = r_idx[r_idx >= s0]
      if nxt.size and (not rs.size or nxt[0] <= rs[0]):
        first_attempt.append((nxt[0] - s0) * dt)
      else:
        first_censored += 1

  # -- stalls: runs of approach steps
  idle_steps = int(round(idle_s / dt))
  stall_lengths, stall_end = [], {"attempt": 0, "reset": 0, "censored": 0}
  stalled = np.zeros((T, B), bool)
  run_lengths_all = []
  for b in range(B):
    # a reset splits runs
    m = approach[:, b].copy()
    starts, lengths, ends = _runs(m)
    # split at resets inside a run
    r_idx = np.flatnonzero(reset[:, b])
    pieces = []
    for s0, ln, e0 in zip(starts, lengths, ends):
      cuts = r_idx[(r_idx >= s0) & (r_idx < e0 - 1)]
      last = s0
      for c in cuts:
        pieces.append((last, c + 1 - last, "reset")); last = c + 1
      how = "censored" if e0 >= T else ("attempt" if attempt[e0, b] else ("reset" if reset[e0 - 1, b] else "other"))
      pieces.append((last, e0 - last, how))
    for s0, ln, how in pieces:
      if ln <= 0:
        continue
      run_lengths_all.append(ln * dt)
      if ln >= idle_steps:
        stall_lengths.append(ln * dt)
        stalled[s0:s0 + ln, b] = True
        stall_end[how if how in stall_end else "reset"] += 1

  # -- windows
  nw = T // window
  def per_window(mask):
    x = mask[:nw * window].reshape(nw, window, B).sum(axis=(1, 2)) / B / (window * dt) * 60.0
    return x.tolist()
  def frac_window(mask):
    x = mask[:nw * window].reshape(nw, window, B).mean(axis=(1, 2))
    return x.tolist()
  half = nw // 2
  pw, aw = per_window(placed), per_window(attempt)
  def ratio(xs):
    e = float(np.mean(xs[:half])) if half else float("nan")
    l = float(np.mean(xs[half:])) if nw - half else float("nan")
    return e, l, (l / e if e > 1e-9 else float("nan"))
  pe, pl, pr = ratio(pw)
  ae, al, ar = ratio(aw)

  # -- target visibility by phase, on fresh captures only
  fresh = tr["fresh"].astype(bool)
  tf, ts = tr["target_full"].astype(np.float64), tr["target_sampled"].astype(np.float64)
  by_phase = {}
  for i, name in enumerate(PHASES):
    m = (phase == i)
    mf = m & fresh
    n = int(mf.sum())
    by_phase[name] = {
      "steps_fraction": float(m.mean()),
      "fresh_frames": n,
      "target_pixels_mean": float(tf[mf].mean()) if n else None,
      "target_pixels_p50": _pct(tf[mf], 50) if n else None,
      "target_points_mean": float(ts[mf].mean()) if n else None,
      "target_points_p50": _pct(ts[mf], 50) if n else None,
      "target_points_p10": _pct(ts[mf], 10) if n else None,
      "frac_frames_zero_target_points": float((ts[mf] == 0).mean()) if n else None,
      "frac_frames_lt4_target_points": float((ts[mf] < 4).mean()) if n else None,
      "jaw_cmd_mm": float(tr["jaw_cmd"][m].mean() * 1000) if m.any() else None,
      "jaw_meas_mm": float(tr["jaw_meas"][m].mean() * 1000) if m.any() else None,
      "dist_mm_p50": _pct(tr["dist"][m] * 1000, 50) if m.any() else None,
    }
  stall_m = stalled
  # An object that is neither on the table nor grasped for a long time is not
  # waiting to be picked: it is resting on the bin rim or wedged somewhere.
  # Reported apart so that it is never read as a stall, and so that a policy
  # that leaves objects there is seen to.
  stuck_lengths, stuck = [], np.zeros((T, B), bool)
  for b in range(B):
    starts, lengths, ends = _runs(place[:, b])
    for s0, ln in zip(starts, lengths):
      if ln >= idle_steps:
        stuck_lengths.append(ln * dt)
        stuck[s0:s0 + ln, b] = True
  end_state = {
    "stalled": float(stall_m[-1].mean()), "stuck_object": float(stuck[-1].mean()),
    "carrying": float(g[-1].mean()), "engaged": float(eng[-1].mean()),
  }
  # The same rates over the arm-time in which an object was actually available:
  # a placement can only happen in a live environment, so the live rate is the
  # policy's own throughput and its late/early is the policy's own decay.
  live = ~stuck
  live_min = live.sum() * dt / 60.0
  def per_window_live(mask):
    out = []
    for k in range(nw):
      sl = slice(k * window, (k + 1) * window)
      mins = live[sl].sum() * dt / 60.0
      out.append(float(mask[sl].sum() / mins) if mins > 1e-9 else None)
    return out
  pw_live, aw_live = per_window_live(placed), per_window_live(attempt)
  def ratio_live(xs):
    a_ = [x for x in xs[:half] if x is not None]; b_ = [x for x in xs[half:] if x is not None]
    e = float(np.mean(a_)) if a_ else float("nan"); l = float(np.mean(b_)) if b_ else float("nan")
    return e, l, (l / e if e > 1e-9 else float("nan"))
  term = tr["term"].astype(np.int64)
  term_names = tr.get("term_names", [])
  term_counts = {name: int((term == i + 1).sum()) for i, name in enumerate(term_names)}
  tswitch = int((np.diff(tr["target_idx"].astype(np.int64), axis=0) != 0).sum())
  arm_min = T * dt * B / 60.0
  n_att, n_pl, n_gr, n_dr = int(attempt.sum()), int(placed.sum()), int(grasp.sum()), int(drop.sum())
  return {
    "steps": T, "envs": B, "dt": dt, "seconds": T * dt, "arm_minutes": arm_min, "window": window,
    "placed_per_min": n_pl / arm_min, "attempts_per_min": n_att / arm_min, "grasps_per_min": n_gr / arm_min,
    "success_per_attempt": (n_pl / n_att) if n_att else None,
    "success_per_grasp": (n_pl / n_gr) if n_gr else None,
    "drops": n_dr, "drop_rate_per_grasp": (n_dr / n_gr) if n_gr else None,
    "placed_per_window": pw, "attempts_per_window": aw,
    "early_placed_per_min": pe, "late_placed_per_min": pl, "late_over_early_placed": pr,
    "early_attempts_per_min": ae, "late_attempts_per_min": al, "late_over_early_attempts": ar,
    "phase_fraction_per_window": {name: frac_window(phase == i) for i, name in enumerate(PHASES)},
    "stalled_fraction_per_window": frac_window(stall_m),
    "target_points_fresh_mean_per_window": [
      float(ts[k * window:(k + 1) * window][fresh[k * window:(k + 1) * window]].mean())
      if fresh[k * window:(k + 1) * window].any() else None for k in range(nw)],
    "wait_after_success_s": {"n": len(w_succ), "censored": c_succ, "mean": float(np.mean(w_succ)) if w_succ else None,
                             "p50": _pct(w_succ, 50), "p90": _pct(w_succ, 90), "max": float(max(w_succ)) if w_succ else None,
                             "frac_gt_idle": float(np.mean(np.asarray(w_succ) >= idle_s)) if w_succ else None},
    "wait_after_drop_s": {"n": len(w_drop), "censored": c_drop, "mean": float(np.mean(w_drop)) if w_drop else None,
                          "p50": _pct(w_drop, 50), "p90": _pct(w_drop, 90)},
    "time_to_first_attempt_s": {"n": len(first_attempt), "censored": first_censored,
                                "p50": _pct(first_attempt, 50), "p90": _pct(first_attempt, 90),
                                "mean": float(np.mean(first_attempt)) if first_attempt else None},
    "approach_run_s": {"n": len(run_lengths_all), "p50": _pct(run_lengths_all, 50), "p90": _pct(run_lengths_all, 90),
                       "max": float(max(run_lengths_all)) if run_lengths_all else None},
    "stalls": {"idle_s": idle_s, "n": len(stall_lengths), "per_arm_minute": len(stall_lengths) / arm_min,
               "stalled_step_fraction": float(stall_m.mean()),
               "length_s_p50": _pct(stall_lengths, 50), "length_s_p90": _pct(stall_lengths, 90),
               "length_s_max": float(max(stall_lengths)) if stall_lengths else None, "ended_by": stall_end,
               "jaw_meas_mm_while_stalled": float(tr["jaw_meas"][stall_m].mean() * 1000) if stall_m.any() else None,
               "jaw_cmd_mm_while_stalled": float(tr["jaw_cmd"][stall_m].mean() * 1000) if stall_m.any() else None,
               "dist_mm_p50_while_stalled": _pct(tr["dist"][stall_m] * 1000, 50) if stall_m.any() else None,
               "target_points_fresh_mean_while_stalled": float(ts[stall_m & fresh].mean()) if (stall_m & fresh).any() else None},
    "placed_per_live_min": float(n_pl / live_min) if live_min > 1e-9 else None,
    "attempts_per_live_min": float(n_att / live_min) if live_min > 1e-9 else None,
    "placed_per_window_live": pw_live, "attempts_per_window_live": aw_live,
    "late_over_early_placed_live": ratio_live(pw_live)[2], "late_over_early_attempts_live": ratio_live(aw_live)[2],
    "stuck_object": {"n": len(stuck_lengths), "step_fraction": float(stuck.mean()),
                     "length_s_p50": _pct(stuck_lengths, 50), "length_s_max": float(max(stuck_lengths)) if stuck_lengths else None},
    "stuck_fraction_per_window": frac_window(stuck),
    "end_state_env_fraction": end_state,
    "by_phase": by_phase,
    "target_switches": tswitch,
    "terminations": term_counts, "resets": int(reset.sum()),
    "terminations_per_arm_minute": {k: v / arm_min for k, v in term_counts.items()},
  }


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-PC-P1B-Vision")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=1800)
  p.add_argument("--window", type=int, default=100)
  p.add_argument("--seed", type=int, default=101)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--reach-m", type=float, default=None, help="engagement radius; default the command's grasp_reach_m")
  p.add_argument("--idle-s", type=float, default=3.0, help="an approach run at least this long is a stall")
  p.add_argument("--episode-length-s", type=float, default=None,
                 help="override the play config's time-out (40 s); above the run length = no time-out")
  p.add_argument("--out", default=None)
  p.add_argument("--trace-npz", default=None, help="also save the per-step record of the first --trace-envs envs")
  p.add_argument("--trace-envs", type=int, default=8)
  from piper_push import evalcfg
  evalcfg.add_sensor_arg(p, default="measured")
  evalcfg.add_action_api_arg(p)
  a = p.parse_args()
  evalcfg.apply_action_api_arg(a)

  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from eval_occlusion import reset_recurrent
  from piper_push import robot as piper

  torch.manual_seed(a.seed)
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  if a.episode_length_s is not None:
    cfg.episode_length_s = float(a.episode_length_s)
  sensor_prov = evalcfg.apply_sensor(cfg, a.task, a.sensor)
  bounded = bool(cfg.actions["arm"].bounded)
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
  agent = load_rl_cfg(a.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(wrapped, asdict(agent), None, a.device)
  loaded = evalcfg.load_weights(runner, a.checkpoint, a.device)
  policy = runner.get_inference_policy(device=a.device)
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("pick")
  tm = env.termination_manager
  causes = [n for n in ("object_lost", "over_speed", "nan", "time_out") if n in tm.active_terms]
  reach = float(a.reach_m if a.reach_m is not None else cmd.cfg.grasp_reach_m)
  lift = float(cmd.cfg.grasp_lift_m)
  bin_c = torch.tensor(cmd.cfg.bin_center, device=a.device)
  bin_in = torch.tensor(cmd.cfg.bin_inner, device=a.device)
  owner = getattr(env, "_pc_cloud_owner", None)

  n, dev, T = a.num_envs, a.device, a.steps
  keys = ("grasped", "engaged", "on_table", "in_bin", "placed", "reset", "term", "fresh",
          "target_full", "target_sampled", "jaw_cmd", "jaw_meas", "dist", "target_idx")
  rec = {k: torch.zeros(T, n, device=dev, dtype=(torch.int16 if k in ("term", "target_idx") else torch.float32)) for k in keys}

  env.reset()
  obs = wrapped.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]
  origins = env.scene.env_origins
  nonfinite = 0
  with torch.inference_mode():
    for t in range(T):
      u = policy(obs)
      if not torch.isfinite(u).all():
        nonfinite += 1
        u = torch.nan_to_num(u)
      act = torch.tanh(u) if bounded else u
      jaw_c = piper.GRIPPER_OFFSET + piper.GRIPPER_SCALE * act[:, -1]
      # the state the action was chosen in
      obj = cmd._object_pos_local()
      site = cmd._site_pos_w() - origins
      half = cmd.object_half_size
      dist = torch.linalg.norm(obj - site, dim=-1)
      clear = obj[:, 2] - half[:, 2]
      rec["grasped"][t] = cmd.grasped.float()
      rec["engaged"][t] = (dist < reach).float()
      rec["on_table"][t] = (clear < lift).float()
      rec["in_bin"][t] = ((obj[:, :2] - bin_c).abs() < bin_in).all(dim=-1).float()
      rec["jaw_cmd"][t] = jaw_c
      rec["jaw_meas"][t] = robot.data.joint_pos[:, 6]
      rec["dist"][t] = dist
      rec["target_idx"][t] = cmd.target.to(torch.int16)
      if owner is not None and owner.target_full_count is not None:
        rec["fresh"][t] = owner.fresh.float() if owner.fresh is not None else 0.0
        rec["target_full"][t] = owner.target_full_count.float()
        rec["target_sampled"][t] = owner.target_sampled_count.float()
      obs, _, dones, _ = wrapped.step(u)
      reset_recurrent(policy, dones)
      rec["placed"][t] = cmd.just_placed.float()          # registered by this step
      rec["reset"][t] = dones.float()
      code = torch.zeros(n, device=dev, dtype=torch.int16)
      for i, c in enumerate(causes):
        code = torch.where(tm.get_term(c), torch.full_like(code, i + 1), code)
      rec["term"][t] = code
  env.close()

  tr = {k: v.cpu().numpy() for k, v in rec.items()}
  tr["term_names"] = causes
  out = summarise(tr, float(env.step_dt), a.window, a.idle_s)
  out.update({"checkpoint": a.checkpoint, "task": a.task, "seed": a.seed, "num_envs": n, "reach_m": reach,
              "episode_length_s": float(cfg.episode_length_s), "nonfinite_action_steps": nonfinite,
              "num_objects": int(cmd.num_objects),
              "provenance": evalcfg.provenance(argv=sys.argv, sensor=sensor_prov, weights=loaded)})
  print(f"{a.checkpoint} on {a.task}: {n} envs x {T} steps ({T * env.step_dt:.0f} s), episode {cfg.episode_length_s} s")
  print(f"placed/min {out['placed_per_min']:.2f}  attempts/min {out['attempts_per_min']:.2f}  success/attempt "
        f"{out['success_per_attempt']}  l/e placed {out['late_over_early_placed']:.2f}  l/e attempts {out['late_over_early_attempts']:.2f}")
  print(f"wait after success p50/p90 {out['wait_after_success_s']['p50']:.2f}/{out['wait_after_success_s']['p90']:.2f} s "
        f"(n {out['wait_after_success_s']['n']}, censored {out['wait_after_success_s']['censored']})  "
        f"first attempt p50 {out['time_to_first_attempt_s']['p50']:.2f} s")
  st = out["stalls"]
  print(f"stalls >= {a.idle_s} s: {st['n']} ({st['per_arm_minute']:.2f}/arm-min), stalled step fraction {st['stalled_step_fraction']:.3f}, "
        f"length p50/p90/max {st['length_s_p50']:.1f}/{st['length_s_p90']:.1f}/{st['length_s_max']} s, ended by {st['ended_by']}, "
        f"jaw while stalled cmd/meas {st['jaw_cmd_mm_while_stalled']}/{st['jaw_meas_mm_while_stalled']} mm")
  for name, d in out["by_phase"].items():
    print(f"  {name:15s} steps {d['steps_fraction']:.3f}  target px mean {d['target_pixels_mean']}  "
          f"points mean/p50/p10 {d['target_points_mean']}/{d['target_points_p50']}/{d['target_points_p10']}  "
          f"zero {d['frac_frames_zero_target_points']}  jaw cmd/meas {d['jaw_cmd_mm']}/{d['jaw_meas_mm']}")
  print(f"drops {out['drops']}  terminations {out['terminations']}  resets {out['resets']}  target switches {out['target_switches']}")
  print(f"stuck object (place phase >= {a.idle_s} s): {out['stuck_object']}  end-state env fractions {out['end_state_env_fraction']}")
  print(f"live-time rates: placed/min {out['placed_per_live_min']}  attempts/min {out['attempts_per_live_min']}  "
        f"l/e placed {out['late_over_early_placed_live']}  l/e attempts {out['late_over_early_attempts_live']}")
  if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
  if a.trace_npz:
    k = min(a.trace_envs, n)
    np.savez_compressed(a.trace_npz, **{key: tr[key][:, :k] for key in keys}, term_names=np.array(causes))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
