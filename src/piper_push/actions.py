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


def apply_plant(
    commanded: torch.Tensor,
    prev_effective: torch.Tensor,
    delay_buf: list[torch.Tensor],
    response_scale: float,
    deadband: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """What the servo does with a command, as opposed to what we asked for.

    Transport delay, then stiction, then incomplete travel -- the order the
    signal meets them on the way down. ``delay_buf`` is mutated in place and
    holds one whole batch of targets per step of delay.

    Free-standing rather than a method so it can be tested without building a
    simulator: the arithmetic here is the entire content of three of the Phase
    WM0 axes, and an error in it would look exactly like "actuation mismatch
    does not matter".

    Returns the effective target and a count of dead-banded elements.
    """
    if delay_buf:
        delay_buf.append(commanded.clone())
        commanded = delay_buf.pop(0)
    step = commanded - prev_effective
    held = torch.zeros((), device=step.device)
    if deadband > 0.0:
        mask = step.abs() < deadband
        held = mask.sum()
        step = torch.where(mask, torch.zeros_like(step), step)
    if response_scale != 1.0:
        step = step * response_scale
    return prev_effective + step, held


@dataclass(kw_only=True)
class RandomizedPlantHookCfg:
    """Per-episode timing and servo mismatch for robust training.

    The real deployment loop updates at roughly 24 Hz while the simulation
    policy runs at 50 Hz.  ``hold_weights`` therefore puts most probability on
    holding a command for two simulator steps, without pretending that timing
    is perfectly periodic.  All quantities are sampled independently per
    environment at reset and stay fixed for that episode.
    """

    latency_weights: tuple[float, ...] = (0.15, 0.55, 0.30)
    hold_weights: tuple[float, ...] = (0.15, 0.70, 0.15)
    response_range: tuple[float, float] = (0.70, 1.0)
    deadband_range: tuple[float, float] = (0.0, 0.004)

    def build(self, action_term) -> "RandomizedPlantHook":
        return RandomizedPlantHook(self, action_term)


class RandomizedPlantHook:
    """Batched implementation of :class:`RandomizedPlantHookCfg`."""

    def __init__(self, cfg: RandomizedPlantHookCfg, action_term) -> None:
        self.cfg = cfg
        self._action_term = action_term
        self.device = action_term.device
        self._shape = action_term._default.shape
        self._latency_probs = self._validate_weights(
            cfg.latency_weights, "latency_weights")
        self._hold_probs = self._validate_weights(cfg.hold_weights, "hold_weights")
        self._max_latency = len(cfg.latency_weights) - 1
        self._history = torch.empty(
            self._max_latency + 1, *self._shape, device=self.device)
        self._cursor = 0
        n = self._shape[0]
        self.latency = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hold_steps = torch.ones(n, dtype=torch.long, device=self.device)
        self._countdown = torch.zeros(n, dtype=torch.long, device=self.device)
        self.response = torch.ones(n, 1, device=self.device)
        self.deadband = torch.zeros(n, 1, device=self.device)
        self._held = action_term._default.clone()
        self._previous = action_term._default.clone()
        self.reset(None)

    @staticmethod
    def _validate_weights(values: tuple[float, ...], name: str) -> torch.Tensor:
        if not values or any(float(x) < 0.0 for x in values) or sum(values) <= 0.0:
            raise ValueError(f"{name} must be non-negative and have positive mass")
        return torch.tensor(values, dtype=torch.float32) / float(sum(values))

    def _sample_category(self, probs: torch.Tensor, n: int) -> torch.Tensor:
        # Sampling on CPU avoids a device-specific generator and happens only
        # on reset, not in the 50 Hz path.
        return torch.multinomial(probs, n, replacement=True).to(self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            ids = torch.arange(self._shape[0], device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self._shape[0], device=self.device)[env_ids]
        else:
            ids = env_ids.to(self.device, dtype=torch.long)
        n = int(ids.numel())
        if n == 0:
            return
        self.latency[ids] = self._sample_category(self._latency_probs, n)
        self.hold_steps[ids] = 1 + self._sample_category(self._hold_probs, n)
        self._countdown[ids] = 0
        lo, hi = self.cfg.response_range
        self.response[ids] = lo + (hi - lo) * torch.rand(n, 1, device=self.device)
        lo, hi = self.cfg.deadband_range
        self.deadband[ids] = lo + (hi - lo) * torch.rand(n, 1, device=self.device)
        # The action term resets this before calling us.  Flush every lag slot
        # so a command from the previous episode can never cross the boundary.
        posture = self._action_term._previous_target[ids]
        self._previous[ids] = posture
        self._held[ids] = posture
        for slot in self._history:
            slot[ids] = posture

    def __call__(self, target: torch.Tensor, action_term) -> torch.Tensor:
        fresh = self._countdown <= 0
        self._held.copy_(torch.where(fresh[:, None], target, self._held))
        self._countdown.copy_(torch.where(
            fresh, self.hold_steps - 1, self._countdown - 1))

        self._history[self._cursor].copy_(self._held)
        rows = (self._cursor - self.latency) % self._history.shape[0]
        envs = torch.arange(self._shape[0], device=self.device)
        delayed = self._history[rows, envs]
        self._cursor = (self._cursor + 1) % self._history.shape[0]

        delta = delayed - self._previous
        delta = torch.where(delta.abs() < self.deadband, 0.0, delta)
        effective = self._previous + self.response * delta
        self._previous.copy_(effective)
        return effective


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

    bounded: bool = False
    """The task's convention is that ``a`` lies in [-1, 1] (a tanh head).  A
    policy trained under the old unbounded convention produces |a| of 3-28 on
    its first step; rather than drive a different robot silently, the term
    raises.  Checked on every call for the first 200 control steps and every
    100th after, because the check is a device sync."""

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

    # -- session-persistent plant mismatch (Phase WM0) --------------------
    # Distinct in purpose from the four fields above: those are *filters we
    # choose to apply*, these are *ways the real arm differs from the model*.
    # Both are inert at their defaults and neither is set by any training
    # config; see piper_push.perturb.

    latency_steps: int = 0
    """Whole control steps of delay between the policy's command and the
    servo receiving it.  The deployed stack has a USB-CAN hop, a driver
    queue and a 200 Hz inner loop, none of which is modelled at all -- the
    simulator currently hands the command over in the same tick it was
    computed."""

    response_scale: float = 1.0
    """Multiplies the commanded *displacement* from the previous target.  A
    real position servo under load does not travel the full commanded step
    within one control period; below 1.0 the arm undershoots every command
    by a fixed fraction, which is what a stiffness or gear-ratio error looks
    like from outside."""

    deadband: float = 0.0
    """Radians of commanded displacement below which the servo does not move
    at all.  Stiction and encoder quantisation both present this way."""

    # -- structural command-path stage (Phase RA-Sim-0) --------------------
    # Everything above is a *parameter*: a scalar the calibration axes in
    # piper_push.perturb can search over.  This is the slot for a mismatch
    # that is not a scalar -- a per-joint hysteresis, a rate-dependent lag --
    # and for the learned residual that tries to cancel one.  Empty by
    # default, and an empty tuple leaves process_actions byte-for-byte the
    # function it was.
    command_hooks: tuple = ()
    """Config objects with ``.build(action_term) -> hook``.  A hook is called
    as ``hook(target, action_term) -> target`` once per control step, after
    the delay/deadband/response plant, and must own a ``reset(env_ids)``.

    Batched Torch only.  A hook that loops over environments in Python is a
    hook that cannot run at 512 environments; see
    docs/residual_injection_audit.md for the throughput this was held to."""

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
        self._bounded = bool(cfg.bounded)
        self._bounded_calls = 0
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

        # -- session-persistent plant mismatch, all inert at defaults --------
        self._latency = max(int(cfg.latency_steps), 0)
        # One slot per step of delay, each holding a whole batch of targets.
        # Seeded with the reset posture rather than zeros: an empty pipeline
        # should behave as "the arm was already where it is", not as "the arm
        # was commanded to the origin".
        self._delay_buf = (
            [self._default.clone() for _ in range(self._latency)]
            if self._latency else []
        )
        self._response_scale = float(cfg.response_scale)
        self._deadband = float(cfg.deadband)
        self._prev_effective = self._default.clone()
        self._plant_active = bool(
            self._latency or self._response_scale != 1.0 or self._deadband > 0.0
        )
        self.stats["held_deadband"] = torch.zeros((), device=self.device)

        # -- the structural stage, downstream of the parametric one ----------
        self._hooks = tuple(h.build(self) for h in (cfg.command_hooks or ()))
        # The ramp is drawn from whatever was handed to the servo last step,
        # so once a hook can change that, the ramp origin has to track the
        # hook's output rather than the plant's.
        self._prev_hooked = self._default.clone()

    def set_plant(self, *, latency_steps: int | None = None,
                  response_scale: float | None = None,
                  deadband: float | None = None,
                  lowpass_hz: float | None = -1.0) -> None:
        """Retune the parametric plant on an already-built term.

        A calibration search over four scalars would otherwise rebuild the
        whole environment once per candidate, which at 512 environments is
        thirty seconds of MJWarp compilation to answer a question about four
        numbers.  These four fields have no effect on the model, the scene or
        the compiled kernels, so they can move in place.

        ``lowpass_hz`` takes ``None`` to mean "no filter" and so cannot use
        ``None`` as "leave alone"; ``-1.0`` is the sentinel.

        Damping is deliberately absent: it lives on the actuator config and
        genuinely does need a rebuild.
        """
        if latency_steps is not None:
            n = max(int(latency_steps), 0)
            if n != self._latency:
                self._latency = n
                self._delay_buf = [self._prev_effective.clone()
                                   for _ in range(n)]
        if response_scale is not None:
            self._response_scale = float(response_scale)
        if deadband is not None:
            self._deadband = float(deadband)
        if lowpass_hz != -1.0:
            if lowpass_hz is None:
                self._lp_alpha = None
            else:
                tau = 1.0 / (2.0 * torch.pi * float(lowpass_hz))
                self._lp_alpha = self._dt / (self._dt + tau)
        self._plant_active = bool(
            self._latency or self._response_scale != 1.0 or self._deadband > 0.0
        )

    @property
    def plant(self) -> dict:
        """What the parametric plant is currently set to."""
        return {"latency_steps": self._latency,
                "response_scale": self._response_scale,
                "deadband": self._deadband,
                "lowpass_alpha": self._lp_alpha}

    @property
    def joint_pos(self) -> torch.Tensor:
        """Measured position of the joints this term commands."""
        return self._entity.data.joint_pos[:, self._target_ids]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Measured velocity of the joints this term commands."""
        return self._entity.data.joint_vel[:, self._target_ids]

    @property
    def servo_error(self) -> torch.Tensor:
        """Measured position minus the target the servo is currently holding.

        The quantity a real controller reports, and the only window a
        deployable model gets onto the plant's own state."""
        return (self._entity.data.joint_pos[:, self._target_ids]
                - self._entity.data.joint_pos_target[:, self._target_ids])

    @property
    def max_step(self) -> torch.Tensor:
        """Largest change in target permitted per control step, per joint."""
        return self._max_step

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._bounded:
            self._bounded_calls += 1
            if self._bounded_calls <= 200 or self._bounded_calls % 100 == 0:
                worst = actions.detach().abs().max()
                if bool(worst > 1.0 + 1e-3):
                    raise ValueError(
                        f"action {float(worst):.2f} outside [-1, 1] on a bounded task: "
                        "this policy was trained under the pre-2026-09-05 unbounded "
                        "convention; evaluate it on the matching '-V1' task id.")
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

        # -- the plant, downstream of everything we chose to do ---------------
        # `_previous_target` tracks the COMMAND stream, which is what the slew
        # and acceleration ceilings are constraints on.  What the servo
        # receives is that stream delayed, dead-banded and scaled -- a
        # different sequence -- so the substep ramp has to start from the
        # previous EFFECTIVE target rather than the previous commanded one, or
        # the ramp would jump to close a gap the servo never saw.
        if self._plant_active:
            effective, held = apply_plant(
                self._processed_actions, self._prev_effective,
                self._delay_buf, self._response_scale, self._deadband)
            self.stats["held_deadband"] += held
            self._ramp_from.copy_(self._prev_effective)
            self._prev_effective.copy_(effective)
            self._processed_actions = effective

        # -- structural stage: hysteresis, rate-dependent lag, residual ------
        # After the plant, because the parametric axes model transport and the
        # servo's linear behaviour and this models what is left: the gearbox
        # and the current limit, which the command meets last.
        if self._hooks:
            hooked = self._processed_actions
            for hook in self._hooks:
                hooked = hook(hooked, self)
            self._ramp_from.copy_(self._prev_hooked)
            self._prev_hooked.copy_(hooked)
            self._processed_actions = hooked
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
        # ``apply_actions`` may legitimately run before the first fresh
        # command (for example while timestamp-accurate replay holds the
        # command that preceded a recorded interval).  The base reset clears
        # ``_processed_actions``; leaving it cleared here would make that
        # first hold drive towards zero/default even though every piece of
        # limiter state says to hold the measured reset posture.
        self._processed_actions[env_ids] = self._previous_target[env_ids]
        # The shaping state goes back to rest with it.  A filter that carried
        # the last episode's commanded velocity across a reset would spend the
        # first steps of the new one unwinding a move that is no longer being
        # made, and the acceleration limit would fight the reset posture.
        self._lp_state[env_ids] = self._previous_target[env_ids]
        # The plant's pipeline is flushed with the reset posture, not left
        # holding commands aimed at where the arm used to be: a delayed
        # command surviving a teleport would drive the fresh episode towards
        # the previous one's pose for as many steps as the delay is long.
        self._prev_effective[env_ids] = self._previous_target[env_ids]
        for slot in self._delay_buf:
            slot[env_ids] = self._previous_target[env_ids]
        self._prev_cmd_vel[env_ids] = 0.0
        self._prev_cmd_acc[env_ids] = 0.0
        # A hook carries per-environment state -- a backlash position, a
        # recurrent hidden vector -- and one environment's must never survive
        # into another's episode.  Reset before the hooks so they can read the
        # fresh posture off `_previous_target`.
        self._prev_hooked[env_ids] = self._previous_target[env_ids]
        for hook in self._hooks:
            hook.reset(env_ids)
