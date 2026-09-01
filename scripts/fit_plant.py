"""Fit the simulated arm to the measured one, with mujoco.sysid.

Every hand-rolled estimate of this plant disagreed with the others -- five
methods gave kp_eff between 138 and 510 -- because the two things that make a
per-experiment fit fail are both present here.  Stiction puts the settled
position anywhere inside a band, so a static hold cannot be inverted for
stiffness; and during a step the position error and the velocity decay
together, so a two-parameter regression cannot separate them.

A trajectory fit has neither problem.  Friction is a fitted parameter rather
than something to be avoided, the delay is fitted rather than separated by
hand, and every sample constrains every parameter at once.

    python scripts/fit_plant.py --train logs/ident/<chirp> --check logs/ident/<chirp>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

import mujoco
import mujoco.sysid as sysid

ARM = tuple(f"joint{i}" for i in range(1, 7))


def load_run(d: pathlib.Path) -> dict:
  """A recorded ident run as arrays, resampled onto a uniform clock.

  The 200 Hz loop is not exactly uniform -- it sleeps to a deadline and misses
  it occasionally -- and every downstream resampler assumes a rate.  So the
  clock is rebuilt at the nominal period and the signals are interpolated onto
  it rather than pretending the recorded stamps were even.
  """
  z = np.load(d / "trace.npz")
  meta = json.loads((d / "run.json").read_text())["meta"]
  t = z["t"].astype(np.float64)
  dt = float(np.median(np.diff(t)))
  grid = np.arange(t[0], t[-1], dt)
  get = lambda k: np.interp(grid, t, z[k].astype(np.float64))
  return {
    "dir": d, "meta": meta, "dt": dt, "t": grid,
    "q": np.stack([get(f"q{i+1}") for i in range(6)], axis=1),
    "dq": np.stack([get(f"dq{i+1}") for i in range(6)], axis=1),
    "qref": np.stack([get(f"qref{i+1}") for i in range(6)], axis=1),
    "tau": np.stack([get(f"tau{i+1}") for i in range(6)], axis=1),
  }



def measured_friction(d: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
  """(Coulomb, viscous) per joint, from the constant-velocity strokes.

  friction(v) = C + B v, fitted through the three speeds.  These are DIRECT
  measurements -- torque minus the model's gravity, at constant velocity where
  the acceleration term vanishes -- so the trajectory fit has no business
  moving them.  Left free, it drove joint2's Coulomb to 1e-4 against a
  measured 0.92 N.m and used it as a compensator.
  """
  rows = json.loads((d / "run.json").read_text())["summary"]["rows"]
  C, B = np.zeros(6), np.zeros(6)
  for j in range(1, 7):
    r = [x for x in rows if x["joint"] == j]
    if len(r) < 2:
      continue
    v = np.array([x["speed"] for x in r])
    f = np.array([x["coulomb"] for x in r])
    A = np.vstack([np.ones_like(v), v]).T
    c, b = np.linalg.lstsq(A, f, rcond=None)[0]
    C[j - 1], B[j - 1] = max(c, 0.0), max(b, 0.0)
  return C, B


def arm_spec() -> mujoco.MjSpec:
  """The identified arm alone: no table, no bin, no objects, no visuals.

  Anything the chirp did not touch is a body whose parameters the fit cannot
  constrain, and leaving it in only widens the search.
  """
  from piper_push import robot as R
  spec = R.get_pick_spec(R.BARE_GRIPPER)
  # gravcomp is what the MIT feedforward WOULD provide.  The chirp was streamed
  # with t_ff = 0, so the real drives fought gravity through their own loop and
  # the model must too, or the fit will absorb gravity into the gains.
  for b in spec.bodies:
    b.gravcomp = 0.0
  # Not sysid.remove_visuals(): it strips the mesh assets, and this arm's
  # COLLISION geoms are meshes, so the spec no longer compiles.  The chirp is
  # free space, so contacts cost nothing here anyway.
  #
  # mjlab applies the collision filtering (contype/conaffinity) through
  # EntityCfg too, so a bare compile has every geom colliding with every other
  # and the folded identification posture is self-intersecting.  Measured: a
  # +11.5 deg step on joint1 moved it 0.013 deg while the actuator held 25 N.m
  # against a contact that does not exist on the real arm.  The chirp is free
  # space, so the honest model of it has no contacts at all.
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

  # mjlab attaches the actuators through EntityArticulationInfoCfg rather than
  # into the spec, so a bare compile has none and every ctrl would be ignored.
  # Add them here in the canonical position-actuator form the sysid modifiers
  # expect: force = kp (ctrl - q) - kd qvel, i.e. gainprm[0] = kp,
  # biasprm[1] = -kp, biasprm[2] = -kd.
  import re
  for name in ARM:
    kp = kd = None
    for expr, (p_, d_) in R.SYSID_GAINS.items():
      if re.fullmatch(expr, name):
        kp, kd = p_, d_
    act = spec.add_actuator()
    act.name = name
    act.target = name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0] = kp
    act.biasprm[1] = -kp
    act.biasprm[2] = -kd
    act.ctrlrange = [-6.3, 6.3]
    act.forcerange = [-200.0, 200.0]
  return spec


def joint_index(spec_model: mujoco.MjModel, name: str) -> int:
  return mujoco.mj_name2id(spec_model, mujoco.mjtObj.mjOBJ_JOINT, name)


def make_params(model, free: set[str], coulomb=None, viscous=None,
                kp_fix=None, kd_fix=None):
  """Per-joint motor parameters, plus one delay shared by the whole arm.

  The mass model is deliberately NOT free by default: with the wrist bare, the
  measured holding torque already matches ``qfrc_bias`` to a median 0.2 N.m
  over 14 postures spanning +-10 N.m, so mass is the one part of this model
  that was already right and freeing it invites the fit to launder motor error
  into link inertia.
  """
  from piper_push import robot as R
  import re

  def per_joint(nominal_of):
    return np.array([nominal_of(j) for j in ARM], dtype=np.float64)

  def gain_of(j, which):
    for expr, (kp, kd) in R.SYSID_GAINS.items():
      if re.fullmatch(expr, j):
        return kp if which == "kp" else kd
    raise KeyError(j)

  # apply_modifier calls modifier(spec, self), so the second argument is the
  # Parameter, not its value.
  val = lambda x: np.atleast_1d(getattr(x, "value", x))

  def joint_setter(field):
    def modifier(spec, value):
      v = val(value)
      for k, name in enumerate(ARM):
        for jt in spec.joints:
          if jt.name == name:
            # MjsJoint.damping is a 3-vector (ball and free joints use all
            # three); armature and frictionloss are scalars.  Assigning the
            # wrong shape is a pybind TypeError, not a silent no-op.
            cur = getattr(jt, field)
            if isinstance(cur, np.ndarray):
              cur[:] = float(v[k])
            else:
              setattr(jt, field, float(v[k]))
      return spec
    return modifier

  def gain_setter(which):
    def modifier(spec, value):
      v = val(value)
      for k, name in enumerate(ARM):
        spec = (sysid.model_modifier.apply_pgain if which == "kp"
                else sysid.model_modifier.apply_dgain)(spec, name, float(v[k]))
      return spec
    return modifier

  p = sysid.ParameterDict()
  add = lambda param: p.update({param.name: param})
  # Bounds must be per-ELEMENT for a vector parameter.  Scalars are accepted at
  # construction and only fail inside optimize(), where the flattened x0 (30)
  # meets a bounds array of one entry per parameter (5).
  six = lambda v: np.full(6, float(v))
  add(sysid.Parameter("armature", per_joint(lambda j: 0.005),
                      six(1e-4), six(0.6), frozen="armature" not in free,
                      modifier=joint_setter("armature")))
  add(sysid.Parameter("damping",
                      np.asarray(viscous) if viscous is not None
                      else per_joint(lambda j: 0.05),
                      six(0.0), six(8.0), frozen="damping" not in free,
                      modifier=joint_setter("damping")))
  add(sysid.Parameter("frictionloss",
                      np.asarray(coulomb) if coulomb is not None
                      else per_joint(lambda j: R.SYSID_COULOMB_NM[j]),
                      (np.asarray(coulomb) * 0.5 if coulomb is not None
                       else six(0.0)),
                      (np.asarray(coulomb) * 2.0 + 1e-3 if coulomb is not None
                       else six(3.0)),
                      frozen="frictionloss" not in free,
                      modifier=joint_setter("frictionloss")))
  add(sysid.Parameter("kp",
                      np.asarray(kp_fix) if kp_fix is not None
                      else per_joint(lambda j: gain_of(j, "kp")),
                      six(5.0), six(4000.0), frozen="kp" not in free,
                      modifier=gain_setter("kp")))
  add(sysid.Parameter("kd",
                      np.asarray(kd_fix) if kd_fix is not None
                      else per_joint(lambda j: gain_of(j, "kd")),
                      six(0.05), six(400.0), frozen="kd" not in free,
                      modifier=gain_setter("kd")))
  # Signal-side, not a model parameter: the command the drives act on at time t
  # was issued 16 ms earlier (measured from the step's torque onset).  Without
  # it the fit can only match phase by moving the gains, which is how three of
  # them ended up pinned at their upper bound.
  add(sysid.Parameter("delay", 0.016, 0.004, 0.045,
                      frozen="delay" not in free, modifier=None))
  return p




def initial_state(model, r) -> np.ndarray:
  """MuJoCo's rollout state is mjSTATE_FULLPHYSICS: time, then qpos, then qvel
  (then act).  Packing it as plain [qpos; qvel] is one element short and the
  error says so only at rollout time."""
  n = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  x0 = np.zeros(n)
  x0[1:1 + 6] = r["q"][0]
  x0[1 + model.nq:1 + model.nq + 6] = r["dq"][0]
  return x0


def sequences(model, runs, name):
  """Measured trajectories in the form the residual pipeline iterates over."""
  from mujoco.sysid._src.timeseries import SignalType
  ctrl_map = {"ctrl": (SignalType.MjCtrl, np.arange(6))}
  sens_map = {"qpos": (SignalType.MjStateQPos, np.arange(6)),
              "qvel": (SignalType.MjStateQVel, np.arange(6))}
  controls, sensors, states, names = [], [], [], []
  for k, r in enumerate(runs):
    controls.append(sysid.TimeSeries(r["t"] - r["t"][0], r["qref"], ctrl_map))
    sensors.append(sysid.TimeSeries(r["t"] - r["t"][0],
                                    np.hstack([r["q"], r["dq"]]), sens_map))
    states.append(initial_state(model, r))
    names.append(f"{name}{k}")
  return controls, sensors, states, names


def simulate(model, r, delay: float = 0.0):
  """Roll the model along the recorded command stream; return sim q and dq.

  Steps by hand rather than through ``sysid.sysid_rollout``.  That helper
  advances the model ONE timestep per control sample, so a 5.05 ms control
  clock driving a 2 ms model runs simulated time at 40% of real: a 30 s
  recording came out as 11.87 s, compressed 2.53x.  Everything fitted through
  it was compensating for that -- the optimiser reached kp = 2055 because
  stiffness scales with the square of the frequency it was being asked to
  reproduce.

  The gains are read back out of the model's own position actuators, so this
  is the same law they encode: ``tau = kp (q* - q) - kd qdot``.
  """
  kp = -model.actuator_biasprm[:6, 1].copy()
  kd = -model.actuator_biasprm[:6, 2].copy()
  return simulate_ff(model, r, kp, kd, np.zeros(6), delay)


def simulate_ff(model, r, kp, kd, kff, delay: float):
  """Step MuJoCo by hand under a control law with velocity feedforward.

  ``tau = kp (q* - q) - kd qdot + kff qdot*``

  The third term is the hypothesis.  A drive that feeds the commanded velocity
  forward tracks like a stiff servo while producing only modest error torque,
  which is the one structure that reconciles the two measurements: position
  says the arm tracks as if kp were ~2000, torque says kp is ~350.  A pure PD
  cannot be both.

  Stepping by hand rather than through a position actuator because MuJoCo's
  position actuator has no feedforward input; the gains here are applied as a
  joint torque directly.
  """
  # The spec carries position actuators for the other code path.  Left alone
  # their ctrl is 0, so each one hauls its joint toward zero at kp=125 while
  # the applied torque fights it -- that read as 13.9 deg of RMSE before it was
  # spotted.  Silence them; here the whole law is qfrc_applied.
  model.actuator_gainprm[:6, 0] = 0.0
  model.actuator_biasprm[:6, 1] = 0.0
  model.actuator_biasprm[:6, 2] = 0.0
  dt = model.opt.timestep
  t0 = r["t"] - r["t"][0]
  n_ctrl = len(t0)
  qref = r["qref"]
  vref = np.gradient(qref, t0, axis=0)
  data = mujoco.MjData(model)
  data.qpos[:6] = r["q"][0]
  data.qvel[:6] = r["dq"][0]
  mujoco.mj_forward(model, data)
  lo, hi = -200.0, 200.0
  out_q = np.zeros((n_ctrl, 6))
  out_dq = np.zeros((n_ctrl, 6))
  # Resample the (delayed) command onto the PHYSICS clock once.  Interpolating
  # inside the loop is twelve np.interp calls per 2 ms step, which dominates
  # the rollout and makes a finite-difference fit impractical.
  n_steps = int(np.ceil(t0[-1] / dt)) + 2
  tp = np.arange(n_steps) * dt
  qr_grid = np.stack([np.interp(np.maximum(tp - delay, 0.0), t0, qref[:, j])
                      for j in range(6)], axis=1)
  vr_grid = np.stack([np.interp(np.maximum(tp - delay, 0.0), t0, vref[:, j])
                      for j in range(6)], axis=1)
  k = 0
  t = 0.0
  i = 0
  while k < n_ctrl and i < n_steps:
    qr, vr = qr_grid[i], vr_grid[i]
    i += 1
    tau = kp * (qr - data.qpos[:6]) - kd * data.qvel[:6] + kff * vr
    data.qfrc_applied[:6] = np.clip(tau, lo, hi)
    mujoco.mj_step(model, data)
    t += dt
    while k < n_ctrl and t0[k] <= t:
      out_q[k] = data.qpos[:6]
      out_dq[k] = data.qvel[:6]
      k += 1
  # The bug this replaced was invisible in the output and obvious in the clock.
  if abs(data.time - t0[-1]) > 0.05 * max(t0[-1], 1e-9):
    raise RuntimeError(
      f"rollout covered {data.time:.2f} s of a {t0[-1]:.2f} s recording")
  return out_q, out_dq, n_ctrl


def similarity_ff(model, r, kp, kd, kff, delay) -> dict:
  q_sim, dq_sim, n = simulate_ff(model, r, kp, kd, kff, delay)
  return _similarity_from(q_sim, dq_sim, r, n)


def similarity(model, r, delay: float = 0.0) -> dict:
  """How close the simulated arm is to the real one on the same commands.

  Three numbers per joint, because RMSE alone hides which way a model is
  wrong: a sim that leads the real arm and one that lags it by the same amount
  score identically, and so does one whose motion is simply too large.

    rmse   position error, degrees
    lag    cross-correlation shift; positive means the SIM lags the real arm
    gain   least-squares slope of sim motion against real motion; 1.0 is right
  """
  q_sim, dq_sim, n = simulate(model, r, delay)
  return _similarity_from(q_sim, dq_sim, r, n)


def _similarity_from(q_sim, dq_sim, r, n) -> dict:
  q_real, dq_real = r["q"][:n], r["dq"][:n]
  dt = r["dt"]
  out = {"rmse_deg": [], "lag_ms": [], "gain": [], "corr": [],
         "dq_rmse": []}
  for j in range(6):
    a_ = q_sim[:, j] - q_sim[:, j].mean()
    b_ = q_real[:, j] - q_real[:, j].mean()
    out["rmse_deg"].append(float(np.degrees(np.sqrt(
      ((q_sim[:, j] - q_real[:, j]) ** 2).mean()))))
    out["dq_rmse"].append(float(np.sqrt(
      ((dq_sim[:, j] - dq_real[:, j]) ** 2).mean())))
    # Cross-correlate over +-150 ms; the peak is the shift between them.
    span = int(0.150 / dt)
    cc = np.correlate(a_, b_, mode="full")
    mid = len(cc) // 2
    k = int(np.argmax(cc[mid - span:mid + span + 1])) - span
    out["lag_ms"].append(float(-k * dt * 1000))
    denom = float((b_ ** 2).sum())
    out["gain"].append(float((a_ * b_).sum() / denom) if denom > 0 else float("nan"))
    sd = a_.std() * b_.std()
    out["corr"].append(float((a_ * b_).mean() / sd) if sd > 0 else float("nan"))
  return out


def report(tag, sim):
  print("\n%s" % tag)
  print("  joint    RMSE(deg)   lag(ms)    gain    corr   dq RMSE(rad/s)")
  for j in range(6):
    print("    %d      %8.3f   %+7.1f   %6.3f  %6.3f   %8.3f"
          % (j + 1, sim["rmse_deg"][j], sim["lag_ms"][j], sim["gain"][j],
             sim["corr"][j], sim["dq_rmse"][j]))
  print("   mean     %8.3f   %+7.1f   %6.3f  %6.3f   %8.3f"
        % (np.mean(sim["rmse_deg"]), np.mean(sim["lag_ms"]),
           np.mean(sim["gain"]), np.mean(sim["corr"]),
           np.mean(sim["dq_rmse"])))
  print("  lag > 0 means the SIMULATED arm lags the real one.")




def _set_vector(params, x):
  """Write a flat decision vector back into the ParameterDict."""
  i = 0
  for v in params.values():
    if v.frozen:
      continue
    n = np.atleast_1d(v.nominal).size
    v.value = x[i:i + n] if n > 1 else float(x[i])
    i += n


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--train", required=True, type=pathlib.Path)
  ap.add_argument("--check", type=pathlib.Path)
  ap.add_argument("--free", default="armature,damping,kp,kd,delay",
                  help="frictionloss is deliberately NOT free by default: it "
                       "is measured directly and a free one becomes a "
                       "compensator")
  ap.add_argument("--kp", type=float, nargs=6,
                  help="hold the position gains here instead of fitting them; "
                       "this is what iit-DLSLab's pipeline does, and freeing "
                       "them is what let kp walk to 2055")
  ap.add_argument("--kd", type=float, nargs=6)
  ap.add_argument("--torque-weight", type=float, default=1.0,
                  help="weight on the measured-torque residual, in units of "
                       "'degrees of position error per N.m'.  0 disables it "
                       "and the gains become degenerate with the delay")
  ap.add_argument("--friction", type=pathlib.Path,
                  help="an ident friction run; its Coulomb and viscous terms "
                       "seed (and freeze) the friction parameters")
  ap.add_argument("--nominal-only", action="store_true",
                  help="roll out with the current model and stop; this is the "
                       "baseline every fitted number has to beat")
  ap.add_argument("--optimizer", default="scipy",
                  choices=("scipy", "mujoco", "scipy_parallel_fd"))
  ap.add_argument("--out", type=pathlib.Path,
                  default=pathlib.Path("results/plant_fit.json"))
  a = ap.parse_args()

  spec = arm_spec()
  model = spec.compile()
  free = set(x for x in a.free.split(",") if x)
  coulomb = viscous = None
  if a.friction:
    coulomb, viscous = measured_friction(a.friction)
    print("measured friction  Coulomb", np.round(coulomb, 3))
    print("                   viscous", np.round(viscous, 3))
  params = make_params(model, free, coulomb, viscous,
                       np.array(a.kp) if a.kp else None,
                       np.array(a.kd) if a.kd else None)
  print("model: %d dof, %d actuators, timestep %.4f s"
        % (model.nv, model.nu, model.opt.timestep))
  print("free parameters:", sorted(free) or "none")

  run = load_run(a.train)
  print("train: %s  %d samples at %.0f Hz"
        % (a.train.name, len(run["t"]), 1.0 / run["dt"]))
  print("\nnominal per-joint |q| RMS of the measured motion (deg):")
  print("  ", np.round(np.degrees(run["q"].std(axis=0)), 2))
  print("tracking error the REAL arm showed (deg, rms):")
  print("  ", np.round(np.degrees((run["qref"] - run["q"]).std(axis=0)), 2))

  base = similarity(model, run)
  report("BASELINE -- today's model against the real arm, same commands", base)
  if a.nominal_only:
    return 0

  # A residual we own end to end.  build_residual_fn wants the predicted state
  # TimeSeries to carry the same signal mapping as the measured one, and
  # reverse-engineering that convention is more work than computing the thing
  # it would produce: the difference between simulated and measured joint
  # angles along the recorded command stream.
  free_names = [k for k, v in params.items() if not v.frozen]
  print("\noptimising %d free values across %s"
        % (sum(np.atleast_1d(params[k].nominal).size for k in free_names),
           ", ".join(free_names)))

  scale = np.radians(1.0)          # residual in "degrees", so all joints weigh alike
  # Position alone cannot separate stiffness from delay: a stiffer, prompter
  # model and a softer, later one draw the same trajectory.  Left to itself the
  # fit took kp to 1500-2055 and the delay to 6.5 ms against a measured 16.5.
  # The torque register breaks the tie, because a position actuator's force IS
  # kp (ctrl - q) - kd qvel -- one equation per joint per sample that sees the
  # gains directly and does not care about the delay's effect on position.
  tau_scale = float(a.torque_weight)
  calls = {"n": 0, "best": np.inf}

  def predicted_torque(mdl, r, q_sim, dq_sim, n, delay):
    kp = -mdl.actuator_biasprm[:6, 1]
    kd = -mdl.actuator_biasprm[:6, 2]
    t0 = r["t"] - r["t"][0]
    ref = np.stack([np.interp(t0 - delay, t0, r["qref"][:, j])
                    for j in range(6)], axis=1)[:n]
    f = kp * (ref - q_sim) - kd * dq_sim
    lo, hi = mdl.actuator_forcerange[:6, 0], mdl.actuator_forcerange[:6, 1]
    return np.clip(f, lo, hi)

  def residual_fn(x, opt_params=None):
    # optimize() calls residual_fn(x, params) and unpacks a 3-tuple.  It hands
    # back the SAME dict it was given, so use it directly -- ParameterDict.update
    # refuses to overwrite an existing key.
    p_use = opt_params if opt_params is not None else params
    x = np.atleast_2d(x)
    out = []
    for row in x:
      _set_vector(p_use, row)
      d_ = float(np.atleast_1d(p_use["delay"].value)[0]) if "delay" in p_use else 0.0
      sp = sysid.apply_param_modifiers_spec(p_use, arm_spec())
      try:
        mdl = sp.compile()
        q_sim, dq_sim, n = simulate(mdl, run, d_)
        r_pos = ((q_sim - run["q"][:n]) / scale).ravel()
        if tau_scale > 0.0:
          tau_p = predicted_torque(mdl, run, q_sim, dq_sim, n, d_)
          r_tau = ((tau_p - run["tau"][:n]) * tau_scale).ravel()
          r = np.concatenate([r_pos, r_tau])
        else:
          r = r_pos
      except Exception:
        r = np.full(len(run["q"]) * 6 * (2 if tau_scale > 0 else 1), 1e3)
      out.append(r)
      calls["n"] += 1
      c = float((r ** 2).mean())
      if c < calls["best"]:
        calls["best"] = c
        print("    eval %4d  mean sq residual %.4f deg^2  (rms %.3f deg)"
              % (calls["n"], c, np.sqrt(c)), flush=True)
    # optimize() does np.concatenate(residuals), so this is a LIST of
    # per-sequence residual arrays, not one flat array.
    return out, None, None

  best, result = sysid.optimize(params, residual_fn, optimizer=a.optimizer,
                                verbose=True)
  spec2 = arm_spec()
  spec2 = sysid.apply_param_modifiers_spec(best, spec2)
  fitted = spec2.compile()
  d_fit = float(np.atleast_1d(best["delay"].value)[0]) if "delay" in best else 0.0
  after = similarity(fitted, run, d_fit)
  report("FITTED -- on the training chirp", after)
  print("  fitted delay: %.1f ms  (measured torque onset was 16.5 ms)"
        % (d_fit * 1000))
  rows = {}
  for k, v in best.items():
    val = np.atleast_1d(v.value)
    rows[k] = val.tolist()
    lo, hi = np.atleast_1d(v.min_value), np.atleast_1d(v.max_value)
    at = ["*" if (x <= l * 1.001 + 1e-9 or x >= h * 0.999 - 1e-9) else " "
          for x, l, h in zip(val, np.broadcast_to(lo, val.shape),
                             np.broadcast_to(hi, val.shape))]
    print("  %-14s %s   %s" % (k, np.round(val, 4), "".join(at)))
  print("  (* = pinned at a bound, i.e. not identified by the data)")
  if a.check:
    chk = load_run(a.check)
    b0, b1 = similarity(model, chk), similarity(fitted, chk, d_fit)
    report("HELD OUT (%s) -- before" % a.check.name, b0)
    report("HELD OUT (%s) -- after" % a.check.name, b1)
    rows["heldout_before"] = b0
    rows["heldout_after"] = b1
  rows["train_before"] = base
  rows["train_after"] = after
  a.out.parent.mkdir(parents=True, exist_ok=True)
  a.out.write_text(json.dumps(rows, indent=1) + "\n")
  print("\nwrote %s" % a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
