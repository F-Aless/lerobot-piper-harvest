# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anchor + joint-limit resolution shared by the QP-smoothed inference engines.

Both ``qp_rtc`` and ``qp_sync`` need the same three pieces of glue around
:class:`QPSmoother`:

* resolve per-joint range limits (config override → robot introspection);
* convert joint values from the policy action unit (pct/deg/rad) to degrees;
* extract a joint-space anchor from either the last commanded action tensor
  or a raw observation dict.

The engines keep their own locking and anchor-priority policy; this class is
pure conversion/lookup logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

logger = logging.getLogger(__name__)


EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


def ee_state_names(state_names: list[str], joint_keys: list[str]) -> list[str]:
    """Rewrite a joint-space ``observation.state`` name list into EE space.

    The contiguous block of ``joint_keys`` is replaced in place by the six
    ``EE_POSE_KEYS``; every other entry (``gripper.pos``, ``gripper.tau``, ...)
    keeps its position.  This matches how the ``*_ee`` datasets lay out their
    state: ``[ee.x..ee.yaw, gripper.pos, gripper.tau]``.
    """
    jset = set(joint_keys)
    out: list[str] = []
    inserted = False
    for name in state_names:
        if name in jset:
            if not inserted:
                out.extend(EE_POSE_KEYS)
                inserted = True
        else:
            out.append(name)
    if not inserted:
        out = list(EE_POSE_KEYS) + out
    return out


def resolve_ee_state_obs(
    *,
    ee_state_obs: bool | None,
    ee_mode: bool,
    fk_available: bool,
    engine_name: str,
) -> bool:
    """Decide whether the engine feeds the policy an FK-computed EE state.

    Policies trained on ``*_ee`` datasets expect ``observation.state`` to be
    ``[ee.x..ee.yaw, gripper...]``.  A joint-space robot natively reports joint
    positions — same dimension, entirely different meaning — which the policy
    normalizer silently mangles, leaving the policy blind to its own pose
    (symptom: the same approach chunk replayed over and over).

    ``None`` auto-enables the conversion when EE mode is active and the robot
    exposes ``ee_pose_from_observation``; explicit ``True`` fails loudly when
    it cannot be honored.
    """
    if ee_state_obs is None:
        if ee_mode and not fk_available:
            logger.warning(
                "%s: ee_mode is active but the robot does not expose ee_pose_from_observation — "
                "the policy will receive JOINT state. If it was trained on an ee.* dataset it is "
                "blind to its own pose.",
                engine_name,
            )
        return ee_mode and fk_available
    if ee_state_obs and not ee_mode:
        raise ValueError(f"{engine_name}: ee_state_obs=true requires EE mode (see ee_mode).")
    if ee_state_obs and not fk_available:
        raise ValueError(
            f"{engine_name}: ee_state_obs=true but the robot does not expose ee_pose_from_observation."
        )
    return bool(ee_state_obs)


def resolve_ee_mode(
    *,
    ee_mode: bool | None,
    joint_keys: list[str],
    ordered_action_keys: list[str],
    policy_action_names: list[str] | None,
    ee_factory_available: bool,
    engine_name: str,
) -> bool:
    """Decide whether a QP engine should run in EE (Cartesian) mode.

    EE mode means: the policy emits ``ee.*`` pose actions and the engine
    converts each chunk to the execution action space (projector → batch IK →
    joint QP) through the robot's ``make_ee_chunk_smoother`` factory.

    ``ee_mode=None`` auto-detects from the *policy* action names (the policy
    emits the six ``ee.*`` pose keys) or, when those are unavailable, from a
    Cartesian execution space.  Explicit True/False is validated against what
    the policy, the action space and the robot actually support, failing
    loudly on impossible combinations (e.g. an EE policy commanding a
    joint-space robot through a non-converting path).
    """
    policy_is_ee = policy_action_names is not None and all(k in policy_action_names for k in EE_POSE_KEYS)
    exec_is_ee = all(k in ordered_action_keys for k in EE_POSE_KEYS)
    joint_ok = all(k in ordered_action_keys for k in joint_keys)

    if ee_mode is None:
        # Fail-safe: when the checkpoint does not expose action names and the
        # robot can run BOTH interpretations (joint execution space + EE
        # conversion available), guessing is a hardware hazard — an EE chunk
        # mislabelled as joints moves the arm wrongly.  Demand the flag.
        if policy_action_names is None and not exec_is_ee and ee_factory_available:
            raise ValueError(
                f"{engine_name}: cannot infer the policy action space — the checkpoint does not "
                "expose action_feature_names and the robot supports both joint and EE execution. "
                "State it explicitly: --inference.ee_mode=true for an ee.* pose policy, "
                "--inference.ee_mode=false for a joint-space policy. (Checkpoints trained with "
                "this repo persist the names and won't need the flag.)"
            )
        ee_mode = (policy_is_ee or exec_is_ee) and ee_factory_available

    if ee_mode and not ee_factory_available:
        raise ValueError(
            f"{engine_name}: ee_mode requested but the robot does not expose "
            "make_ee_chunk_smoother (Cartesian chunk conversion is robot-specific; "
            "piper_full and piper_ee support it)."
        )
    if ee_mode and policy_action_names is not None and not policy_is_ee:
        raise ValueError(
            f"{engine_name}: ee_mode requested but the policy does not emit ee.* pose "
            f"actions (policy action names: {list(policy_action_names)!r})."
        )
    if not ee_mode and policy_is_ee and not exec_is_ee:
        raise ValueError(
            f"{engine_name}: the policy emits ee.* pose actions but the execution "
            f"action space is joint-space {list(ordered_action_keys)!r} and ee_mode "
            "is disabled — the chunk would be mislabelled as joints. Enable "
            "--inference.ee_mode=true (or leave it unset for auto-detection)."
        )
    if not ee_mode and not joint_ok:
        missing = [k for k in joint_keys if k not in ordered_action_keys]
        raise ValueError(
            f"{engine_name}: joint keys {missing!r} are not in the action space "
            f"{list(ordered_action_keys)!r} and EE mode is unavailable or disabled. "
            "Set --inference.joint_keys to the actual joint action names, or use an "
            "EE-capable robot (piper_full / piper_ee) with an ee.* policy."
        )
    return bool(ee_mode)


class QPAnchorResolver:
    """Joint-limit and anchor helpers for QP-smoothed engines.

    Parameters
    ----------
    qp_config
        Config carrying ``joint_keys``, ``action_unit`` and (optionally)
        ``joint_limits_deg`` — e.g. ``QPRTCInferenceConfig``.
    ordered_action_keys
        Canonical action key order of the policy output tensor.
    robot_wrapper
        Used as fallback source for joint limits via ``joint_limits_deg``.
    """

    def __init__(
        self,
        qp_config,
        ordered_action_keys: list[str],
        robot_wrapper: ThreadSafeRobot,
    ) -> None:
        self._qp_config = qp_config
        self._robot = robot_wrapper
        self._joint_indices = [ordered_action_keys.index(k) for k in qp_config.joint_keys]
        self._joint_limits_warned = False

    @property
    def joint_indices(self) -> list[int]:
        return self._joint_indices

    def resolve_joint_limits_deg(self) -> np.ndarray | None:
        """Pull joint limits, in degrees, from config or robot, as a (N, 2) array."""
        cfg_limits = self._qp_config.joint_limits_deg
        if cfg_limits is not None:
            return np.asarray(cfg_limits, dtype=np.float64)

        robot_limits = self._robot.joint_limits_deg
        if robot_limits is not None:
            lo, hi = robot_limits
            return np.column_stack([np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)])

        if not self._joint_limits_warned:
            logger.warning(
                "QPAnchorResolver: no joint_limits_deg available "
                "(config and robot both returned None). Range bounds will not be enforced."
            )
            self._joint_limits_warned = True
        return None

    def unit_to_deg(self, joint_values: np.ndarray) -> np.ndarray:
        """Convert a 1D array of joint values (in cfg.action_unit) to degrees."""
        unit = self._qp_config.action_unit
        if unit == "deg":
            return joint_values
        if unit == "rad":
            return np.degrees(joint_values)
        # pct — needs joint_limits_deg.
        limits = self.resolve_joint_limits_deg()
        if limits is None:
            raise RuntimeError("QPAnchorResolver: cannot convert pct anchor without joint_limits_deg.")
        lo = limits[:, 0]
        hi = limits[:, 1]
        return lo + (joint_values + 100.0) / 200.0 * (hi - lo)

    def anchor_from_action(self, action: torch.Tensor | None) -> np.ndarray | None:
        """Anchor (degrees) from an action tensor in ordered_action_keys space."""
        if action is None or action.numel() == 0:
            return None
        joint_unit = action[self._joint_indices].cpu().numpy().astype(np.float64)
        return self.unit_to_deg(joint_unit)

    def anchor_from_obs(self, obs: dict | None) -> np.ndarray | None:
        """Anchor (degrees) from a raw observation dict (joint keys looked up by name)."""
        if obs is None:
            return None
        joint_keys = self._qp_config.joint_keys
        if all(k in obs and isinstance(obs[k], (int, float)) for k in joint_keys):
            joint_unit = np.array([float(obs[k]) for k in joint_keys], dtype=np.float64)
            return self.unit_to_deg(joint_unit)
        return None
