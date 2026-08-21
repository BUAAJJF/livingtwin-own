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
    """

    velocity_limit: dict[str, float] | None = None

    def build(self, env) -> "RateLimitedJointPositionAction":
        return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env) -> None:
        super().__init__(cfg=cfg, env=env)
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
        self._max_step = limits * float(env.step_dt)
        self._default = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._previous_target = self._default.clone()

    @property
    def max_step(self) -> torch.Tensor:
        """Largest change in target permitted per control step, per joint."""
        return self._max_step

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        delta = self._processed_actions - self._previous_target
        self._processed_actions = self._previous_target + delta.clamp(
            -self._max_step, self._max_step
        )
        self._previous_target = self._processed_actions.clone()

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
