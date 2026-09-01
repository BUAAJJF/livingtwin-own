"""Run the deployed pipeline against a scene the simulator rendered.

Nothing else in this package can be checked without hardware, and hardware is
the worst place to discover that a depth image is upside down or that the
proprioception vector is in the wrong order.  So the simulator plays the part
of the world: it renders the same scene twice, once through a camera with the
D405's resolution and field of view and once through the policy's own camera,
and the pipeline has to turn the first into the second.

That is a real test and not a tautology.  The two renders differ in resolution
and in field of view and in nothing the pipeline is allowed to know -- the
resampling has to recover the second from the first through the extrinsic, and
a sign error anywhere in the chain of conventions (MuJoCo looks down -z with
+y up, OpenCV down +z with +y down, row zero is the top of the image) produces
an image that is visibly wrong rather than slightly wrong.

Five stages, each answering one question:

  geometry     does the resampled depth match what the policy's camera saw?
  observation  do the three channels match, including what a hole reads as?
  mask         does the segmenter find the object the simulator labelled, and
               does the tracker stay on it?
  proprio      is the 36-vector the same one the observation manager builds?
  command      does an action become the same joint target?

``tests/test_deploy.py`` covers the same arithmetic against closed-form answers
in a second and without a GPU; this is the one that involves a renderer.

Usage:
    micromamba run -n mjlab python hardware/deploy/selftest.py
    micromamba run -n mjlab python hardware/deploy/selftest.py --policy <dir>
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mjlab.tasks  # noqa: F401,E402
import mujoco  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.sensor import CameraSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

from deploy import config, mask as mask_mod, obs as obs_mod, rectify  # noqa: E402
from deploy.proprio import JointFeedback, Kinematics, ProprioBuilder  # noqa: E402
from deploy.robot import ARM_JOINTS, GRIPPER_JOINT, ActionMapper  # noqa: E402
from piper_push import camera as sim_camera  # noqa: E402

D405_CAM = "d405_stand_in"
SENSOR_NAME = D405_CAM          #: the old name, kept so nothing breaks
TASK = "Mjlab-Pick-Place-PiperX-Vision"
FAILURES: list[str] = []


def d405_fovy() -> float:
  """The stand-in camera's vertical field of view, from the real intrinsics."""
  return 2 * math.degrees(math.atan(
    0.5 * config.D405_HEIGHT / rectify._default_d405_K()[1, 1]))


def build_env(device: str = "cuda:0", num_envs: int = 1):
  """The vision task with a D405-shaped camera bolted alongside the policy's.

  Shared with ``simrecord.py`` and ``scripts/plot_depth_model.py``: all three
  need a scene rendered twice through two different lenses, and two of them
  would otherwise reconstruct it from this file's docstring.
  """
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (d405_camera_cfg(),)
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def settle(env, steps: int = 12):
  """Step with no action until the arm has stopped reacting to its reset."""
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  obs = env.reset()[0]
  for _ in range(steps):
    obs, *_ = env.step(zero)
  return obs


def _object_pixels(env, sensor_name: str) -> torch.Tensor:
  """``(1, 1, H, W)`` marking the pixels that are objects rather than scene.

  The sensor model's surface-quality draw applies to these and not to the
  table.  The bench measured a blank white *patch* on a sheet, not a blank
  white room, and the deployment can put a textured mat on the table but cannot
  choose what the objects look like.
  """
  seg = env.scene[sensor_name].data.segmentation
  assert seg is not None, (
    f"{sensor_name} renders no segmentation; add it to data_types in "
    "selftest.d405_camera_cfg()."
  )
  ids = seg[..., 0]
  is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
  cmd = env.command_manager.get_term("pick")
  objects = cmd.all_geom_ids if hasattr(cmd, "all_geom_ids") \
    else cmd.target_geom_ids
  objects = objects.to(ids.device)
  m = (ids.unsqueeze(-1) == objects[:, None, None, :]).any(-1) & is_geom
  return m.unsqueeze(1).float()


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
  if not ok:
    FAILURES.append(f"{name}: {detail}")


def d405_camera_cfg() -> CameraSensorCfg:
  """A camera with the D405's geometry, at the policy camera's pose.

  Same place, same orientation: the extrinsic between them is the identity, so
  any error the resampling makes is the resampling's, not a calibration's.  The
  hand-eye path is checked separately in ``tests/test_deploy.py``, against a
  pose it knows the answer to.

  ``CAMERA_QUAT``, not ``look_at_quat(CAMERA_POS)``.  The two agreed until the
  policy camera took its orientation from the measured D455 calibration, which
  carries 1.91 degrees of roll; ``look_at_quat`` rebuilds the frame from the
  world's up vector and so cannot express roll at all.  The identity this
  docstring claims was then off by that angle, and every check downstream read
  it as the pipeline's error: resampled depth 7.5 mm at p50 against 0.40, mask
  IoU 0.664 against 0.918, and "the policy cannot tell the two paths apart"
  failing at 19% of what the image is worth.  None of it was the pipeline.
  """
  return CameraSensorCfg(
    name=D405_CAM,
    parent_body=sim_camera.PARENT_BODY,
    pos=sim_camera.CAMERA_POS,
    quat=sim_camera.CAMERA_QUAT,
    fovy=d405_fovy(),
    width=config.D405_WIDTH,
    height=config.D405_HEIGHT,
    # Colour as well, because ``simrecord.py`` writes sessions that
    # ``autolabel.py`` then labels, and a session with no image is a session
    # nothing can be trained on.  It is flat-shaded -- ``use_textures`` is off,
    # as it is for the policy's own camera -- so it is good enough to prove the
    # labelling pipeline runs end to end and not good enough to train a model
    # that will see a real table.  That set comes from a recorded session.
    data_types=("depth", "segmentation", "rgb"),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 2),
  )


def mask_sweep(env, rig, reproj, scenes: int, device: str) -> None:
  """Where the depth segmenter's size limit actually is, under the real sensor.

  One scene tells you whether the segmenter works.  It does not tell you on
  what, and the answer is not flattering: the objects are 25-45 mm and land
  anywhere on the table, so they arrive at the policy's grid as anything from
  50 to 400 pixels, and the small end is where a segmenter that has to reject
  noise blobs starts rejecting objects too.

  Reported as found-and-missed sizes rather than a single rate, because the
  useful thing to know is the threshold, not the average.
  """
  from piper_push import depth_noise

  print(f"\nmask, {scenes} fresh scenes under the measured sensor")
  segmenter = mask_mod.DepthSegmenter(rig, reproj)
  kin = Kinematics()
  corr = depth_noise.DepthCorruption(
    1, config.D405_HEIGHT, config.D405_WIDTH, float(rig.K[1, 1]), device,
    sim_camera.DEPTH_NOISE)

  found, missed, ious = [], [], []
  for _ in range(scenes):
    settle(env, 12)
    truth = _object_pixels(env, sim_camera.CAMERA_NAME)[0, 0].cpu().numpy() > 0
    size = int(truth.sum())
    if size == 0:
      continue
    clean = env.scene[D405_CAM].data.depth.permute(0, 3, 1, 2) \
      .clamp(config.MIN_DEPTH_M, config.CUTOFF_M)
    featureless = _object_pixels(env, D405_CAM)
    kin.update(env.scene["robot"].data.joint_pos[0].cpu().numpy()
               .astype(np.float64))
    arm = kin.link_spheres()

    tracker = mask_mod.TargetTracker()
    label, seg, frame = 0, None, None
    for _ in range(tracker.confirm + 1):
      corrupted, ok = corr(clean, featureless=featureless)
      frame = torch.where(ok, corrupted, torch.zeros_like(corrupted))[0, 0] \
        .cpu().numpy().astype(np.float32)
      seg = segmenter(frame, arm=arm)
      label = tracker.update(seg, kin.site_pos)

    if not label:
      missed.append(size)
      continue
    _, valid, got = reproj(
      frame, payload=mask_mod.full_mask(seg, label, segmenter.decimate))
    hit = (got > 0) & valid
    iou = float((hit & truth).sum()) / max(float((hit | truth).sum()), 1.0)
    (found if iou > 0.2 else missed).append(size)
    if iou > 0.2:
      ious.append(iou)

  n = len(found) + len(missed)
  print(f"  found {len(found)} of {n}")
  if found:
    print(f"    found:  {min(found)}-{max(found)} px, median IoU "
          f"{float(np.median(ious)):.2f}")
  if missed:
    print(f"    missed: {min(missed)}-{max(missed)} px")
  if found and missed:
    print(f"    the limit sits between {max(missed)} and {min(found)} pixels "
          "of the policy's image")


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--steps", type=int, default=12)
  p.add_argument("--mask-sweep", type=int, default=0,
                 help="after the checks, run the segmenter over N fresh scenes "
                      "under the measured sensor and report where its size "
                      "limit is.  One scene says whether it works; this says "
                      "on what.")
  p.add_argument("--checkpoint", default=None,
                 help="the actor checkpoint the ONNX was exported from.  "
                      "Given, the exported graph is compared against a policy "
                      "built on the *deployment* task and loaded from it -- "
                      "which is the only thing that catches an export made "
                      "against the wrong task.")
  p.add_argument("--policy", default=None,
                 help="directory holding policy.onnx, to check it runs")
  a = p.parse_args()

  env = build_env(a.device)
  agent = load_rl_cfg(TASK)
  obs = settle(env, a.steps)
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)

  rig = config.Rig.nominal()
  # The stand-in is a MuJoCo camera, so it has MuJoCo's intrinsics: square
  # pixels and the principal point exactly centred.  Handing it the real
  # D405's off-centre principal point throws every ray by up to 8 mrad, which
  # is millimetres of depth error on a table seen at an angle.
  rig.K = rectify.mujoco_K(config.D405_WIDTH, config.D405_HEIGHT,
                           d405_fovy())
  reproj = rectify.Reprojector(rig, device=a.device)

  truth = env.scene[sim_camera.CAMERA_NAME].data.depth[0, ..., 0] \
    .cpu().numpy().astype(np.float64)
  stand_in = env.scene[SENSOR_NAME].data
  src = stand_in.depth[0, ..., 0].cpu().numpy().astype(np.float32)

  # ---------------------------------------------------------------- geometry
  print("geometry")
  got, valid, _ = reproj(src)
  both = valid & (truth > 0) & (truth < config.CUTOFF_M) & (got > 0)
  err = (got - truth)[both]
  check("resampled depth agrees with the policy camera",
        float(np.percentile(np.abs(err), 95)) < 0.006,
        f"p50 {np.median(np.abs(err)) * 1000:.2f} mm  "
        f"p95 {np.percentile(np.abs(err), 95) * 1000:.2f} mm  n {int(both.sum())}")
  check("no systematic offset", abs(float(np.median(err))) < 0.002,
        f"median {np.median(err) * 1000:+.2f} mm")
  in_range = (truth > 0) & (truth < config.CUTOFF_M)
  check("resampling fills the frame", float(valid[in_range].mean()) > 0.995,
        f"fill {valid[in_range].mean():.4f}")

  # An upside-down or mirrored image passes an average test and fails this one.
  flip = np.flipud(both)
  check("not vertically flipped",
        float(np.abs(got - truth)[both].mean())
        < 0.2 * float(np.abs(np.flipud(got) - truth)[both & flip].mean()),
        "compared against a flipped copy")

  # ------------------------------------------------------------- observation
  print("observation")
  seg = stand_in.segmentation[0].cpu().numpy()
  cmd = env.command_manager.get_term("pick")
  target_ids = cmd.target_geom_ids[0].cpu().numpy()
  src_target = (np.isin(seg[..., 0], target_ids)
                & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))

  _, _, carried = reproj(src, payload=src_target.astype(np.int32))
  built = obs_mod.camera_obs(got, valid, carried > 0)
  ref = obs["camera"][0].cpu().numpy()
  d_err = np.abs(built[0] - ref[0])
  check("depth channel matches", float(np.percentile(d_err, 99)) < 0.01,
        f"p50 {np.median(d_err):.5f}  p99 {np.percentile(d_err, 99):.5f}")

  mine_m, ref_m = built[1] > 0.5, ref[1] > 0.5
  union = float((mine_m | ref_m).sum())
  check("target mask carried through the resampling",
        union > 0 and float((mine_m & ref_m).sum()) / union > 0.75,
        f"IoU {float((mine_m & ref_m).sum()) / max(union, 1):.3f}  "
        f"px {int(ref_m.sum())}")
  check("third channel is the product",
        bool(np.allclose(built[2], built[0] * built[1], atol=1e-6)))

  hv = valid.copy()
  hv[:, :20] = False
  check("holes read as the far plane",
        float(obs_mod.camera_obs(got, hv, carried > 0)[0][:, :20].min()) == 1.0)

  # --------------------------------------------------------------------- mask
  print("mask")
  robot_ent = env.scene["robot"]
  q = robot_ent.data.joint_pos[0].cpu().numpy().astype(np.float64)
  kin = Kinematics()
  kin.update(q)
  segmenter = mask_mod.DepthSegmenter(rig, reproj)
  seg_out = segmenter(src, arm=kin.link_spheres())
  check("segmenter finds something", len(seg_out.instances) >= 1,
        f"{len(seg_out.instances)} instance(s): "
        + ", ".join(f"{i.n_px}px z={i.top_z * 1000:.0f}mm"
                    for i in seg_out.instances))

  if seg_out.instances:
    tracker = mask_mod.TargetTracker()
    hand = kin.ee_pose_b()[:3]
    # One frame is not enough to choose a target, on purpose.  An instance has
    # to be seen ``confirm`` times before the tracker will aim at it, because
    # the measured noise on this camera throws up blobs that look like short
    # objects and the nearest-to-the-hand rule cannot tell them apart.  What
    # separates them is that they do not come back.
    check("one sighting is not enough to aim at",
          tracker.update(seg_out, hand) == 0,
          f"needs {tracker.confirm} in {tracker.window} frames")

    label = 0
    for _ in range(tracker.confirm):
      obs, *_ = env.step(zero)
      frame = env.scene[SENSOR_NAME].data.depth[0, ..., 0].cpu().numpy() \
        .astype(np.float32)
      kin.update(robot_ent.data.joint_pos[0].cpu().numpy().astype(np.float64))
      label = tracker.update(segmenter(frame, arm=kin.link_spheres()),
                             kin.ee_pose_b()[:3])
    check("a persistent instance is confirmed and chosen", label != 0)

    seg_out = segmenter(frame, arm=kin.link_spheres())
    ref_m = obs["camera"][0].cpu().numpy()[1] > 0.5
    found = mask_mod.full_mask(seg_out, label, segmenter.decimate)
    _, _, found_pol = reproj(frame, payload=found)
    recall = float((found_pol > 0)[ref_m].sum()) / max(float(ref_m.sum()), 1)
    check("the tracked instance is the simulator's target", recall > 0.5,
          f"recall {recall:.3f} over {int(ref_m.sum())} target pixels")
    check("the target does not change while it is still there",
          tracker.update(seg_out, kin.ee_pose_b()[:3]) == label)

  # ----------------------------------------------------------------- proprio
  print("proprio")
  builder = ProprioBuilder()
  names = builder.joint_names
  # Re-read: the mask stage stepped the environment to give the tracker frames
  # to confirm against, so the arm is no longer where it was at the top.
  q = robot_ent.data.joint_pos[0].cpu().numpy().astype(np.float64)
  qd = robot_ent.data.joint_vel[0].cpu().numpy().astype(np.float64)
  tgt = robot_ent.data.joint_pos_target[0].cpu().numpy().astype(np.float64)
  last = env.action_manager.action[0].cpu().numpy().astype(np.float64)
  ai = [names.index(j) for j in ARM_JOINTS]
  gi = names.index(GRIPPER_JOINT)
  fb = JointFeedback.from_arm(q[ai], qd[ai], q[gi], qd[gi], tgt[ai], tgt[gi],
                              0.0, joint_names=names)
  mine = builder(fb, last)
  theirs = obs["proprio"][0].cpu().numpy()
  # The second finger is compared apart from the rest.  No drive reports it --
  # it is a mechanical mimic of the first -- so ``from_arm`` fills it in as the
  # negative of the first, while the simulator solves it as an equality
  # constraint with a finite stiffness.  The two agree to about a millimetre of
  # travel and 2 mm/s, and that gap is the modelling choice, not an error in
  # the assembly.  Folding it into the same tolerance as everything else would
  # either hide a real regression in the driven joints or fail for a reason
  # that has nothing to do with them.
  mimic = [i for i, n in enumerate(names) if n == "gripper_joint2"]

  def _worst(term, keep) -> float:
    sl = slice(term["offset"], term["offset"] + term["width"])
    d = np.abs(mine[sl] - theirs[sl])
    return float(d[keep].max()) if np.any(keep) else 0.0

  by_term, mimic_err = {}, 0.0
  for t in builder.terms:
    keep = np.ones(t["width"], dtype=bool)
    if t["name"] in ("joint_pos", "joint_vel"):
      keep[mimic] = False
      mimic_err = max(mimic_err, _worst(t, ~keep))
    by_term[t["name"]] = _worst(t, keep)

  # pad_contact is a drive-current bit here and two contact sensors there; it
  # cannot match, and proprio.py says so.  Everything else must.
  rebuilt = {k: v for k, v in by_term.items() if k != "pad_contact"}
  worst = max(rebuilt, key=rebuilt.get)
  check("every driven term matches", rebuilt[worst] < 2e-4,
        f"worst {worst} {rebuilt[worst]:.2e}  "
        + "  ".join(f"{k}={v:.1e}" for k, v in by_term.items()))
  check("the mimicked finger is close enough", mimic_err < 5e-3,
        f"{mimic_err:.2e} against the simulator's equality constraint")

  # ----------------------------------------------------------------- command
  print("command")
  import json
  spec = json.loads(pathlib.Path(config.HERE / "obs_spec.json").read_text())
  mapper = ActionMapper(spec, clip_actions=agent.clip_actions)
  term = env.action_manager.get_term("arm")
  # Seeded from the simulator's own previous target, not from the measured
  # joint position.  Deployment seeds from the measurement -- that is what
  # ``ActionMapper.reset`` is for and why -- but the two differ by the servo's
  # tracking error, and once both are slewing at the ceiling that difference
  # never closes.  Comparing them from different starting points would
  # measure the servo, not the mapping.
  mapper.previous = np.concatenate([
    term._previous_target[0].cpu().numpy(),
    env.action_manager.get_term("gripper")._previous_target[0].cpu().numpy(),
  ])
  rng = np.random.default_rng(0)
  worst_rad = 0.0
  for _ in range(20):
    act = rng.uniform(-1.5, 1.5, size=env.action_manager.total_action_dim)
    mine_t = mapper(act)
    env.step(torch.as_tensor(act, dtype=torch.float32,
                             device=env.device).reshape(1, -1))
    theirs_t = term._previous_target[0].cpu().numpy()
    worst_rad = max(worst_rad, float(np.abs(mine_t[:6] - theirs_t).max()))
  check("joint targets match the action term", worst_rad < 1e-5,
        f"worst {worst_rad:.2e} rad over 20 random actions")

  # ----------------------------------------------------- the two paths agree
  # The only stage that compares the simulator to the robot rather than the
  # robot to the simulator.  The sensor model is stated in angle, so applying
  # it at the sensor's 848x480 and resampling to 224x168 has to land in the
  # same place as applying it directly at 224x168 -- which is what the policy
  # trains against.  If it does not, one of the two is scaling a per-pixel
  # constant across grids of different resolution, and the policy is being
  # trained for a camera the robot does not have.
  print("the two paths")
  from piper_push import depth_noise

  torch.manual_seed(7)
  clean_pol = env.scene[sim_camera.CAMERA_NAME].data.depth.permute(0, 3, 1, 2) \
    .clamp(config.MIN_DEPTH_M, config.CUTOFF_M)
  sim_corr = depth_noise.DepthCorruption(
    1, config.HEIGHT, config.WIDTH, sim_camera.f_px_per_rad(), a.device,
    sim_camera.DEPTH_NOISE)
  _, sim_valid = sim_corr(clean_pol, featureless=_object_pixels(
    env, sim_camera.CAMERA_NAME))

  src_corr = depth_noise.DepthCorruption(
    1, config.D405_HEIGHT, config.D405_WIDTH, float(rig.K[1, 1]), a.device,
    sim_camera.DEPTH_NOISE)
  clean_src = env.scene[D405_CAM].data.depth.permute(0, 3, 1, 2) \
    .clamp(config.MIN_DEPTH_M, config.CUTOFF_M)
  noisy_src, ok_src = src_corr(clean_src,
                               featureless=_object_pixels(env, D405_CAM))
  raw = torch.where(ok_src, noisy_src, torch.zeros_like(noisy_src))[0, 0] \
    .cpu().numpy().astype(np.float32)
  rec_depth, rec_valid, _ = reproj(raw)

  near = (truth > 0) & (truth < config.CUTOFF_M)
  sim_holes = float((~sim_valid[0, 0].cpu().numpy())[near].mean())
  dep_holes = float((~rec_valid)[near].mean())
  ratio = max(sim_holes, dep_holes) / max(min(sim_holes, dep_holes), 1e-6)
  check("the hole rates agree between the two paths", ratio < 2.5,
        f"simulator {sim_holes * 100:.1f}%  robot {dep_holes * 100:.1f}%  "
        f"ratio {ratio:.2f}x")

  # Gross outliers excluded at 50 mm, the same threshold the bench used and for
  # the same reason: a resampled pixel that landed on the far side of a
  # silhouette is not a noisy measurement of this surface, it is a measurement
  # of a different one.  Including them reported 146 mm of "noise" on a camera
  # whose inliers are within 10.
  seen = rec_valid & near
  e = (rec_depth - truth)[seen]
  e = e[np.abs(e) < 0.05]
  # Robust sigma, because what is left is still heavier-tailed than a Gaussian.
  resid = 1.4826 * float(np.median(np.abs(e - np.median(e))))
  expected = math.hypot(depth_noise.SIGMA_STATIC_PER_M,
                        depth_noise.SIGMA_TEMPORAL_PER_M)
  z = float(np.median(truth[near]))
  check("the noise survives the resampling",
        0.4 < resid / (expected * z * z) < 1.8,
        f"{resid * 1000:.1f} mm at {z:.2f} m, against "
        f"{expected * z * z * 1000:.1f} mm applied directly; "
        f"{100 * (1 - e.size / max(seen.sum(), 1)):.1f}% discarded as gross")

  # ------------------------------------------------------------------ policy
  if a.policy:
    print("policy")
    from deploy.policy import Policy
    pol = Policy(a.policy, providers=["CPUExecutionProvider"])
    out = pol(mine.astype(np.float32), built.astype(np.float32))
    check("the exported graph runs and returns an action",
          out.shape == (env.action_manager.total_action_dim,),
          f"{out.shape} {np.round(out, 3)}")

    if a.checkpoint:
      # ``scripts/check_export.py`` cannot catch this and it is worth saying
      # why: it compares the exported graph against the policy it built from
      # the same task config, so exporting the wrong network and comparing it
      # to itself passes.  Run against the distillation task rather than the
      # vision one, the export came out disagreeing with the trained policy by
      # 4.35 on actions of magnitude 1-5 -- a completely different network --
      # and check_export still printed OK.
      #
      # This is the comparison that finds it: the deployment task's policy,
      # loaded from the checkpoint, against the graph as ``deploy.policy``
      # actually feeds it.
      from dataclasses import asdict as _asdict

      from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
      from mjlab.tasks.registry import load_runner_cls

      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
        wrapped, _asdict(agent), None, a.device)
      runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True,
                  map_location=a.device)
      ref_pol = runner.alg.get_policy()
      ref_pol.reset()
      pol.reset()
      o = wrapped.get_observations()
      if isinstance(o, tuple):
        o = o[0]
      worst_a = 0.0
      with torch.inference_mode():
        for _ in range(10):
          ref_a = ref_pol(o)
          got_a = pol(o["proprio"][0].cpu().numpy(),
                      o["camera"][0].cpu().numpy())
          worst_a = max(worst_a,
                        float(np.abs(ref_a[0].cpu().numpy() - got_a).max()))
          o = wrapped.step(ref_a)[0]
          if isinstance(o, tuple):
            o = o[0]
      check("the exported graph is the policy that was trained",
            worst_a < 1e-4,
            f"max |torch - onnx| over 10 steps {worst_a:.3e}")

    # And the question the whole directory exists to answer: does the policy
    # do the same thing with the observation this pipeline reconstructs as
    # with the one the simulator hands it?
    #
    # Not "are the two identical" -- they cannot be, because the sensor model
    # draws different noise on each path.  The scale to compare against is how
    # much the policy's action already moves between two draws of the *same*
    # path, which is the variation it was trained through and tolerates by
    # construction.  If the deploy path sits inside that, it is delivering a
    # picture the policy cannot tell apart from one it expects.
    from piper_push import depth_noise

    pol_corr = depth_noise.DepthCorruption(
      1, config.HEIGHT, config.WIDTH, sim_camera.f_px_per_rad(), a.device,
      sim_camera.DEPTH_NOISE)

    # From a *realistic* hidden state, and this is not a detail.  Measured
    # from a zeroed one, feeding the policy eight copies of any frame produced
    # the same action to five decimal places, and the obvious reading -- "this
    # policy ignores its camera" -- was wrong.  A fresh GRU has accumulated
    # nothing, so its output is the bias path; the actions it produces there
    # are 0.09 in magnitude against 1.36 in a rollout, which is the tell.  From
    # a state the policy actually reaches, swapping the image moves the action
    # by half its magnitude.
    warm = obs
    for _ in range(30):
      warm = env.step(torch.as_tensor(
        pol(warm["proprio"][0].cpu().numpy().astype(np.float32),
            warm["camera"][0].cpu().numpy().astype(np.float32)),
        dtype=torch.float32, device=env.device).reshape(1, -1))[0]
    hidden = pol.hidden.copy()
    warm_p = warm["proprio"][0].cpu().numpy().astype(np.float32)

    # Everything the two paths are built from is read here, after the warm-up,
    # so that both describe the same scene.
    clean_pol2 = env.scene[sim_camera.CAMERA_NAME].data.depth \
      .permute(0, 3, 1, 2).clamp(config.MIN_DEPTH_M, config.CUTOFF_M)
    feat = _object_pixels(env, sim_camera.CAMERA_NAME)
    truth_mask = feat[0, 0].cpu().numpy() > 0

    def sim_obs_draw():
      d, ok = pol_corr(clean_pol2, featureless=feat)
      return obs_mod.camera_obs(d[0, 0].cpu().numpy(),
                                ok[0, 0].cpu().numpy(), truth_mask)

    def act(camera):
      pol.hidden = hidden.copy()
      return pol(warm_p, camera.astype(np.float32))

    # What the image is worth to this policy, from this state: the action with
    # the camera blanked.  Without that anchor the next number is unreadable --
    # a policy ignoring its camera scores perfectly on it.
    blank = np.zeros_like(built, dtype=np.float32)
    a_ref = act(sim_obs_draw())
    worth = float(np.abs(a_ref - act(blank)).max())
    check("the image matters to this policy at all", worth > 0.05,
          f"blanking the camera moves the action by {worth:.4f}")

    # How much the action already moves between two draws of the *simulator's*
    # own sensor: the variation the policy was trained through and tolerates by
    # construction.
    a_sim = [act(sim_obs_draw()) for _ in range(6)]
    within = float(np.mean([np.abs(x - y).max()
                            for i, x in enumerate(a_sim)
                            for y in a_sim[i + 1:]]))

    # And the deploy path: the same sensor model, applied at the sensor's own
    # resolution and resampled, on the same scene.
    src_corr2 = depth_noise.DepthCorruption(
      1, config.D405_HEIGHT, config.D405_WIDTH, float(rig.K[1, 1]), a.device,
      sim_camera.DEPTH_NOISE)
    clean_src2 = env.scene[D405_CAM].data.depth.permute(0, 3, 1, 2) \
      .clamp(config.MIN_DEPTH_M, config.CUTOFF_M)
    feat_src = _object_pixels(env, D405_CAM)
    # Read *after* the warm-up.  The scene has moved thirty control steps since
    # the top of this file, and comparing a current image against a stale mask
    # reported the two paths as 182% apart when they are 11% -- a test failing
    # for a reason that has nothing to do with what it is testing.
    seg_src = env.scene[D405_CAM].data.segmentation[0].cpu().numpy()
    tgt_src = (np.isin(seg_src[..., 0],
                       env.command_manager.get_term("pick")
                       .target_geom_ids[0].cpu().numpy())
               & (seg_src[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
    across = []
    for _ in range(6):
      c, ok = src_corr2(clean_src2, featureless=feat_src)
      raw2 = torch.where(ok, c, torch.zeros_like(c))[0, 0].cpu().numpy() \
        .astype(np.float32)
      d2, v2, m2 = reproj(raw2, payload=tgt_src.astype(np.int32))
      across.append(act(obs_mod.camera_obs(d2, v2, m2 > 0)))
    between = float(np.mean([np.abs(x - y).max()
                             for x in across for y in a_sim]))

    check("the policy cannot tell the two paths apart",
          between < 2.5 * max(within, 1e-9) and between < 0.5 * worth,
          f"between paths {between:.4f}, within the simulator's own noise "
          f"{within:.4f}, blanking the camera {worth:.4f} -- the two paths "
          f"differ by {100 * between / max(worth, 1e-9):.0f}% of what the "
          "image is worth")

  if a.mask_sweep:
    mask_sweep(env, rig, reproj, a.mask_sweep, a.device)

  env.close()
  print()
  if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for f in FAILURES:
      print("  -", f)
    return 1
  print("all checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
