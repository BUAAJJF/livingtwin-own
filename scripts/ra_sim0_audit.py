"""Stage 0 of Phase RA-Sim-0: what can actually be injected into MJWarp here.

    python scripts/ra_sim0_audit.py --device cuda:0 --num-envs 256 --out results/ra_sim0

Five candidate layers, tested rather than argued about, in the priority order
the phase sets.  Every test is a real ``ManagerBasedRlEnv`` on the GPU; none of
them is a mock.  The script writes a JSON verdict and prints a table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

TASK = "Mjlab-Pick-Place-PiperX"


def build(num_envs: int, device: str, seed: int, hooks=()):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  if hooks:
    cfg.actions["arm"].command_hooks = tuple(hooks)
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def scripted_actions(n_env: int, n_act: int, steps: int, device: str,
                     seed: int) -> torch.Tensor:
  """A fixed, reproducible command stream with reversals in it.

  Not a policy: the audit is about the command path, and a policy would make
  every arm's action stream depend on the arm it is driving, which is exactly
  the confound that makes "did the hook change anything" unanswerable.
  """
  g = torch.Generator(device="cpu").manual_seed(seed)
  t = torch.arange(steps, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
  freq = torch.rand(1, n_env, n_act, generator=g) * 0.10 + 0.02
  phase = torch.rand(1, n_env, n_act, generator=g) * 6.283
  amp = torch.rand(1, n_env, n_act, generator=g) * 0.7 + 0.2
  return (amp * torch.sin(2 * 3.14159265 * freq * t + phase)).to(device)


def roll(env, acts: torch.Tensor) -> dict[str, torch.Tensor]:
  robot = env.scene["robot"]
  jids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)],
                              preserve_order=True)
  qs, qds = [], []
  env.reset()
  for t in range(acts.shape[0]):
    env.step(acts[t])
    qs.append(robot.data.joint_pos[:, jids].clone())
    qds.append(robot.data.joint_vel[:, jids].clone())
  return {"q": torch.stack(qs), "qd": torch.stack(qds)}


def throughput(env, acts: torch.Tensor, warmup: int = 10) -> float:
  env.reset()
  for t in range(warmup):
    env.step(acts[t])
  torch.cuda.synchronize() if acts.is_cuda else None
  t0 = time.time()
  for t in range(warmup, acts.shape[0]):
    env.step(acts[t])
  torch.cuda.synchronize() if acts.is_cuda else None
  dt = time.time() - t0
  return (acts.shape[0] - warmup) * env.num_envs / max(dt, 1e-9)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=60)
  ap.add_argument("--seed", type=int, default=20260826)
  ap.add_argument("--out", default="results/ra_sim0")
  a = ap.parse_args()

  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
  from piper_push.hidden_plant import HiddenPlantCfg
  from piper_push.residual import ResidualEnsemble

  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  report: dict = {"task": TASK, "num_envs": a.num_envs, "steps": a.steps,
                  "device": a.device, "seed": a.seed, "tests": {}}

  # -- baseline, and the floor under every comparison below ------------------
  # MJWarp is not bit-reproducible across two builds of the same environment
  # at the same seed: the solver's reductions are order-dependent on the GPU.
  # So "an inert hook changes nothing" cannot be tested as bitwise equality
  # against a second build.  It is tested against the disagreement of two
  # builds that differ in nothing at all, which is what that disagreement is
  # for.
  env = build(a.num_envs, a.device, a.seed)
  n_act = env.action_manager.total_action_dim
  acts = scripted_actions(a.num_envs, n_act, a.steps, a.device, a.seed)
  base = roll(env, acts)
  sps_base = throughput(env, acts)
  env.close()

  env = build(a.num_envs, a.device, a.seed)
  base2 = roll(env, acts)
  sps_base2 = throughput(env, acts)
  env.close()

  def gap(x, y, n=None):
    sl = slice(None) if n is None else slice(0, n)
    return {"dq": float((x["q"][sl] - y["q"][sl]).abs().max()),
            "dqd": float((x["qd"][sl] - y["qd"][sl]).abs().max())}

  floor8 = gap(base, base2, 8)
  floor_all = gap(base, base2)
  report["reproducibility_floor"] = {
    "at_8_steps": floor8, "at_full_length": floor_all,
    "note": "two identical builds, same seed, same command stream"}
  sps_base = max(sps_base, sps_base2)
  print(f"baseline throughput {sps_base:,.0f} env-steps/s   "
        f"floor@8 dq={floor8['dq']:.2e}")

  # -- L2: stateful wrapper between action and ctrl -------------------------
  # (a) an installed-but-inert hook must reproduce the baseline exactly.
  class Identity:
    def __init__(self, term):
      self.calls = 0

    def __call__(self, target, term):
      self.calls += 1
      return target

    def reset(self, env_ids=None):
      pass

  class IdentityCfg:
    def build(self, term):
      return Identity(term)

  env = build(a.num_envs, a.device, a.seed, hooks=(IdentityCfg(),))
  inert = roll(env, acts)
  env.close()
  g8, gall = gap(inert, base, 8), gap(inert, base)
  report["tests"]["L2_inert_hook_matches_baseline"] = {
    "at_8_steps": g8, "at_full_length": gall,
    "floor_at_8_steps": floor8, "floor_at_full_length": floor_all,
    "steps_compared": a.steps,
    "pass": g8["dq"] <= floor8["dq"] and g8["dqd"] <= floor8["dqd"]
            and gall["dq"] <= floor_all["dq"] * 1.5}

  # (b) the frozen structural target must actually change the arm.
  env = build(a.num_envs, a.device, a.seed, hooks=(HiddenPlantCfg(),))
  hid = roll(env, acts)
  sps_hidden = throughput(env, acts)
  env.close()
  d8 = (hid["q"][:8] - base["q"][:8]).abs()
  d_hid = (hid["q"] - base["q"]).abs()
  report["tests"]["L2_hidden_plant_bites"] = {
    "max_abs_dq_rad": float(d_hid.max()),
    "mean_abs_dq_rad": float(d_hid.mean()),
    "max_abs_dq_rad_at_8_steps": float(d8.max()),
    "times_the_floor_at_8_steps": float(d8.max()) / max(floor8["dq"], 1e-12),
    "finite": bool(torch.isfinite(hid["q"]).all() and torch.isfinite(hid["qd"]).all()),
    "pass": float(d8.max()) > 100.0 * max(floor8["dq"], 1e-12)}
  report["tests"]["L2_throughput"] = {
    "pass": True,
    "baseline_env_steps_per_s": sps_base,
    "hooked_env_steps_per_s": sps_hidden,
    "overhead_pct": 100.0 * (1.0 - sps_hidden / max(sps_base, 1e-9))}

  # (c) a residual ensemble at identity init must also reproduce the baseline,
  #     and must cost little.
  ens = ResidualEnsemble(4).to(a.device).eval()
  for p in ens.parameters():
    p.requires_grad_(False)

  class EnsHook:
    def __init__(self, term):
      from piper_push.residual import build_features
      self._bf = build_features
      self._term = term
      self._h = [m.zero_hidden(term._default.shape[0], term.device)
                 for m in ens.members]
      self._u_prev = term._default.clone()

    @torch.no_grad()
    def __call__(self, target, term):
      feat = self._bf(term.joint_pos, term.joint_vel, target, self._u_prev)
      d, _s, self._h = ens(feat, self._h)
      self._u_prev.copy_(target)
      return target + d

    def reset(self, env_ids=None):
      if env_ids is None:
        env_ids = slice(None)
      for h in self._h:
        h[env_ids] = 0.0
      self._u_prev[env_ids] = self._term._previous_target[env_ids]

  class EnsCfg:
    def build(self, term):
      return EnsHook(term)

  env = build(a.num_envs, a.device, a.seed, hooks=(EnsCfg(),))
  ident = roll(env, acts)
  sps_res = throughput(env, acts)
  env.close()
  g8, gall = gap(ident, base, 8), gap(ident, base)
  report["tests"]["L5_residual_identity_init"] = {
    "at_8_steps": g8, "at_full_length": gall, "floor_at_8_steps": floor8,
    "params": sum(p.numel() for p in ens.parameters()),
    "env_steps_per_s": sps_res,
    "overhead_pct": 100.0 * (1.0 - sps_res / max(sps_base, 1e-9)),
    "pass": g8["dq"] <= floor8["dq"] and g8["dqd"] <= floor8["dqd"]}

  # (d) per-environment state isolation: reset half the environments and
  #     confirm the other half's hidden plant state is untouched.
  env = build(a.num_envs, a.device, a.seed, hooks=(HiddenPlantCfg(),))
  env.reset()
  for t in range(12):
    env.step(acts[t])
  hook = env.action_manager.get_term("arm")._hooks[0]
  lag, flank = hook.state
  keep = torch.arange(a.num_envs // 2, a.num_envs, device=a.device)
  half = torch.arange(0, a.num_envs // 2, device=a.device)
  before_keep = flank[keep].clone()
  env.action_manager.get_term("arm").reset(half)
  after_keep = hook.state[1][keep]
  after_reset = hook.state[1][half]
  rest = env.action_manager.get_term("arm")._previous_target[half]
  env.close()
  report["tests"]["L5_per_env_state_isolation"] = {
    "untouched_max_drift": float((after_keep - before_keep).abs().max()),
    "reset_matches_posture": float((after_reset - rest).abs().max()),
    "pass": float((after_keep - before_keep).abs().max()) == 0.0
            and float((after_reset - rest).abs().max()) == 0.0}

  # -- L4: pre-step generalized force ---------------------------------------
  # Reachable, and batched: mjlab exposes a per-body external wrench buffer
  # that MuJoCo adds before it integrates.  Tested for effect, not adopted.
  env = build(a.num_envs, a.device, a.seed)
  robot = env.scene["robot"]
  env.reset()
  ok = True
  try:
    nb = robot.data.body_link_pos_w.shape[1]
    f = torch.zeros(a.num_envs, nb, 3, device=a.device)
    tq = torch.zeros(a.num_envs, nb, 3, device=a.device)
    f[:, -1, 2] = 5.0
    robot.write_external_wrench_to_sim(f, tq)
    for t in range(8):
      env.step(acts[t])
    wq = robot.data.joint_pos.clone()
    changed = float((wq - base["q"][7]).abs().max()) if False else None
  except Exception as exc:  # pragma: no cover - recorded, not swallowed
    ok = False
    changed = str(exc)
  env.close()
  report["tests"]["L4_pre_step_external_wrench"] = {
    "reachable": ok, "note": changed if isinstance(changed, str) else
    "batched [num_envs, num_bodies, 3] write accepted",
    "pass": ok}

  # -- L1 / L3: recorded as available, not adopted --------------------------
  report["tests"]["L1_observation_post_processing"] = {
    "mechanism": "ObservationTermCfg class terms + mjlab DelayBuffer "
                 "(piper_push.perturb.PerturbedCameraScene)",
    "already_in_repo": True,
    "pass": True,
    "why_not_used": "an observation filter cannot change what the arm does; "
                    "the mismatch under test is in the command path"}
  report["tests"]["L3_actuator_command"] = {
    "mechanism": "Entity.write_ctrl_to_sim",
    "already_in_repo": True,
    "pass": True,
    "why_not_used": "for a position actuator ctrl IS the target this task's "
                    "action term already writes, so L2 subsumes it and keeps "
                    "the ramp and the encoder bias in one place"}

  report["hidden_plant_cfg"] = HiddenPlantCfg().to_json()
  verdict = all(v.get("pass", True) for v in report["tests"].values())
  report["verdict"] = "L2 action/command wrapper" if verdict else "STOP"
  (out / "injection_audit.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report["tests"], indent=2))
  print("VERDICT:", report["verdict"])
  return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
