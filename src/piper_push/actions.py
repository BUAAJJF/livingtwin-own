"""A joint-position action the hardware could actually execute.

The deployed command path cannot move a position target faster than the joints'
velocity limits, and the safety shell ends the run above them.  A policy
trained without that ceiling learns dynamics the robot will refuse: it plans a
0.5 rad step at 50 Hz, which is 25 rad/s, seven times the fastest joint's trip
point.  Putting the limit in the command path -- rather than only punishing
overspeed afterwards -- means commands the arm could never execute are simply
not in the action space.

The trip itself stays as a termination elsewhere, because on hardware it also
fires for *dynamic* overspeed that no command limiter can prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets with a per-joint slew ceiling.

    ``velocity_limit`` maps joint-name regexes to a ceiling in the joint's own
    units per second -- rad/s for the arm's hinges, m/s for the gripper's
    slide.  Joints not matched are unlimited.

    The four fields below it are evaluation-time command shaping, all inert at
    their defaults.  They exist to answer one question: how much of the
    residual safety-shell rate is reachable with a deterministic filter that
    needs no retraining?  Nothing in the training path sets them.
    """

    velocity_limit: dict[str, float] | None = None

    slew_scale: float = 1.0
    """Multiplies ``velocity_limit``.  1.0 is the trained command path; below
    that is a tighter slew ceiling, which is the trivial control every other
    filter here has to beat."""

    accel_limit: float | None = None
    """Ceiling on the commanded acceleration, rad/s^2, applied to every arm
    joint.  The commanded velocity is ``(target - previous) / dt``; this bounds
    how much it may change from one control step to the next.

    This is the filter with a mechanism behind it.  The residual trips are an
    underdamped plant (zeta ~ 0.35 on joints 1-3) answering a step change in
    commanded velocity; a slew limiter bounds the velocity but says nothing
    about how abruptly it may arrive."""

    lowpass_hz: float | None = None
    """First-order low-pass on the target, cutoff in Hz.  A softer version of
    the same idea: it attenuates the high-frequency content of the command
    without a hard bound, at the cost of lag on every move."""

    interp: str = "linear"
    """Shape of the within-control-step command ramp: ``linear`` or ``cubic``.

    Linear is what the trained policy ran under.  Cubic (smoothstep) makes the
    commanded velocity zero at both ends of the step, which removes the
    acceleration discontinuity at the boundary -- but its peak rate is 1.5x the
    average for the same displacement, so it buys smoothness with headroom.
    Which of those dominates is the measurement, not an assumption."""

    def build(self, env) -> "RateLimitedJointPositionAction":
        return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env) -> None:
        super().__init__(cfg=cfg, env=env)
        self._substeps = max(int(env.cfg.decimation), 1)
        self._substep = 0
        limits = torch.full((self._num_targets,), float("inf"), device=self.device)
        if cfg.velocity_limit:
            index_list, name_list, value_list = resolve_matching_names_values(
                cfg.velocity_limit, self._target_names
            )
            if not index_list:
                raise ValueError(
                    f"velocity_limit {list(cfg.velocity_limit)} matched none of "
                    f"the controlled joints {self._target_names}."
                )
            limits[index_list] = torch.tensor(value_list, device=self.device)
        # One control step of travel, not one physics substep: process_actions
        # runs once per control step and the target is held across the rest.
        self._dt = float(env.step_dt)
        self._max_step = limits * self._dt * float(cfg.slew_scale)
        self._default = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._previous_target = self._default.clone()
        self._ramp_from = self._default.clone()

        # -- evaluation-time shaping, all inert unless configured -------------
        if cfg.interp not in ("linear", "cubic"):
            raise ValueError(f"interp must be 'linear' or 'cubic', got {cfg.interp!r}")
        self._cubic = cfg.interp == "cubic"
        # Commanded velocity carried across steps, which both the acceleration
        # limit and the jerk statistic are defined against.
        self._prev_cmd_vel = torch.zeros_like(self._default)
        self._prev_cmd_acc = torch.zeros_like(self._default)
        self._max_accel_step = (
            None if cfg.accel_limit is None
            else float(cfg.accel_limit) * self._dt
        )
        if cfg.lowpass_hz is None:
            self._lp_alpha = None
        else:
            tau = 1.0 / (2.0 * torch.pi * float(cfg.lowpass_hz))
            self._lp_alpha = self._dt / (self._dt + tau)
        self._lp_state = self._default.clone()

        # What each stage actually did, for the report.  Counted rather than
        # asserted: a filter whose limit never binds is a filter that is not
        # the reason anything changed.
        self.stats = {
            k: torch.zeros((), device=self.device)
            for k in ("steps", "elements", "clipped_slew", "clipped_accel",
                      "moved_lowpass")
        }
        self.cmd_vel = torch.zeros_like(self._default)
        self.cmd_acc = torch.zeros_like(self._default)
        self.cmd_jerk = torch.zeros_like(self._default)

    @property
    def max_step(self) -> torch.Tensor:
        """Largest change in target permitted per control step, per joint."""
        return self._max_step

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        target = self._processed_actions

        # 1. Low-pass, on the raw target.  First, because it is a model of a
        #    softer command source rather than a ceiling on one.
        if self._lp_alpha is not None:
            filtered = self._lp_state + self._lp_alpha * (target - self._lp_state)
            self.stats["moved_lowpass"] += (
                (filtered - target).abs() > 1e-9).sum()
            target = filtered
            # copy_, not rebind -- see the note in the slew block below.
            self._lp_state.copy_(filtered)

        delta = target - self._previous_target

        # 2. Acceleration ceiling, on the commanded velocity implied by delta.
        if self._max_accel_step is not None:
            vel = delta / self._dt
            lo = self._prev_cmd_vel - self._max_accel_step
            hi = self._prev_cmd_vel + self._max_accel_step
            bound = vel.clamp(lo, hi)
            self.stats["clipped_accel"] += ((bound - vel).abs() > 1e-9).sum()
            delta = bound * self._dt

        # 3. Slew ceiling last, because it is the one the hardware imposes:
        #    anything a filter above hands down still has to fit under it.
        clipped = delta.clamp(-self._max_step, self._max_step)
        self.stats["clipped_slew"] += ((clipped - delta).abs() > 1e-9).sum()
        delta = clipped

        self._processed_actions = self._previous_target + delta

        # Command derivatives, after every stage: this is what the servo is
        # actually asked for, which is the quantity the plant answers.
        vel = delta / self._dt
        acc = (vel - self._prev_cmd_vel) / self._dt
        self.cmd_jerk = (acc - self._prev_cmd_acc) / self._dt
        self.cmd_vel, self.cmd_acc = vel, acc
        self._prev_cmd_vel.copy_(vel)
        self._prev_cmd_acc.copy_(acc)
        self.stats["steps"] += 1
        self.stats["elements"] += delta.numel()
        # Persistent buffers, copied into rather than rebound.  Rebinding makes
        # the tensor whatever mode it was created under -- a rollout runs in
        # inference mode, so the next reset outside one cannot write to it --
        # and the two names would alias for a step before the rebind split them.
        self._ramp_from.copy_(self._previous_target)
        self._previous_target.copy_(self._processed_actions)
        self._substep = 0

    def apply_actions(self) -> None:
        # Hand the servo a ramp, not a stair.  Holding one target for the whole
        # control step means the joint sees a 50 Hz staircase whose steps are a
        # full control period of travel each, and it answers each step with a
        # velocity spike well above the rate that was asked for: measured, a
        # zero action returning the arm to its home pose from a reset posture
        # peaked at 1.000 of joint2's safety-shell trip while the command path
        # was limited to 0.75 of it.  Interpolating across the substeps sends
        # the same average rate as a constant-velocity command, which is what
        # the limit was chosen against.
        self._substep = min(self._substep + 1, self._substeps)
        alpha = self._substep / self._substeps
        if self._cubic:
            # Smoothstep: zero slope at both ends of the control step, so the
            # commanded velocity no longer jumps at the boundary.  Same total
            # displacement, so the average rate is unchanged and the slew limit
            # still means what it meant -- but the peak rate is 1.5x it.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        target = self._ramp_from + (self._processed_actions - self._ramp_from) * alpha
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            target - encoder_bias, joint_ids=self._target_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # Slew from where the robot ACTUALLY is, not from its default pose.
        # Reset events run before this, so joint_pos is the fresh posture; a
        # limiter that starts at the default instead hands the servo the whole
        # reset offset as a step, and a zero action then trips the safety
        # shell's velocity limit -- measured at 30% of episodes before this.
        if env_ids is None:
            env_ids = slice(None)
        self._previous_target[env_ids] = self._entity.data.joint_pos[env_ids][
            :, self._target_ids
        ]
        self._ramp_from[env_ids] = self._previous_target[env_ids]
        # The shaping state goes back to rest with it.  A filter that carried
        # the last episode's commanded velocity across a reset would spend the
        # first steps of the new one unwinding a move that is no longer being
        # made, and the acceleration limit would fight the reset posture.
        self._lp_state[env_ids] = self._previous_target[env_ids]
        self._prev_cmd_vel[env_ids] = 0.0
        self._prev_cmd_acc[env_ids] = 0.0
