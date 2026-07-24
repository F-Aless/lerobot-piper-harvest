# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""AgileX Piper 6-DOF arm + parallel gripper, modernised for LeRobot v0.5.2.

Uses the ``piper_sdk`` SDK with the classic position-control framing.
This is the same SDK family the legacy lerobot fork used.

Exposes the same observation / action schema as before so that policies
and datasets trained against ``piper_full`` remain compatible:

  * Observation: ``{joint_{1..6}.pos, gripper.pos, gripper.tau, + cameras}``
  * Action:      ``{joint_{1..6}.pos, gripper.pos}``

Joint angle unit is selected with ``unit`` ("pct" | "deg" | "rad"):
``[-100, 100]`` normalized by default; degrees or radians expose raw signed
angles instead (NB: "deg"/"rad" are implemented but not hardware-tested).
The gripper is in ``[0, 100]`` for "pct" (0 = closed, 100 = fully open at
``gripper_max_mm``) and in raw mm for "deg"/"rad".
"""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceNotConnectedError

from .config_piper_full import PiperFullConfig
from .piper_full_sdk import PiperFullSDK

logger = logging.getLogger(__name__)


class PiperFull(Robot):
    """Piper 6-DOF arm with gripper and optional cameras."""

    config_class = PiperFullConfig
    name = "piper_full"

    def __init__(self, config: PiperFullConfig):
        super().__init__(config)
        self.config = config
        self._sdk: PiperFullSDK | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._tau_filtered: float = 0.0
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_joint_deg: list[float] | None = None
        self._last_gripper_mm: float | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return (
            self._sdk is not None
            and self._sdk.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def joint_limits_deg(self) -> tuple[list[float], list[float]] | None:
        if self._sdk is None or not self._sdk.is_connected:
            return None
        oriented_min, oriented_max = self._oriented_joint_limits_deg()
        return list(oriented_min), list(oriented_max)

    @property
    def action_angle_unit(self) -> str:
        """Joint angle unit of this robot's observations/actions: "pct" | "deg" | "rad".

        Duck-typed capability: the QP inference engines read it to resolve
        their ``action_unit`` automatically when the flag is left unset.
        """
        return self.config.unit

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 — SDK self-calibrates
        if self._sdk is None:
            self._sdk = PiperFullSDK(
                can_channel=self.config.can_channel,
                can_interface=self.config.can_interface,
                can_bitrate=self.config.can_bitrate,
                firmware=self.config.firmware,
                speed_percent=self.config.speed_percent,
                enable_timeout_s=self.config.enable_timeout_s,
            )
        self._sdk.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("[PiperFull] Connected (%s, fw=%s)", self.config.can_channel, self.config.firmware)

    def disconnect(self) -> None:
        if self._sdk is not None:
            self._sdk.disconnect()
            self._sdk = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("[PiperFull] Disconnected")

    # ------------------------------------------------------------------
    # Calibration (SDK enforces preset limits; calibration file is optional)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Piper joint limits are firmware-defined; gripper stroke is configured via `gripper_max_mm`."""
        return None

    def configure(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    @property
    def _motors_obs_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {f"{name}.pos": float for name in self.config.joint_names}
        ft["gripper.pos"] = float
        ft["gripper.tau"] = float
        return ft

    @property
    def _motors_act_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {f"{name}.pos": float for name in self.config.joint_names}
        ft["gripper.pos"] = float
        return ft

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_obs_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_act_ft

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        sdk = self._require_sdk()

        joint_deg = sdk.get_joint_angles_deg()
        if joint_deg is None:
            if self._last_joint_deg is None:
                raise RuntimeError("Piper has not reported joint angles yet — CAN bus up?")
            joint_deg = self._last_joint_deg
        else:
            self._last_joint_deg = joint_deg

        gripper_mm = sdk.get_gripper_width_mm()
        if gripper_mm is None:
            gripper_mm = self._last_gripper_mm if self._last_gripper_mm is not None else 0.0
        else:
            self._last_gripper_mm = gripper_mm

        self._filter_gripper_tau(sdk.get_gripper_force_n())

        obs: dict[str, Any] = {}
        unit = self.config.unit
        if unit == "pct":
            oriented_min, oriented_max = self._oriented_joint_limits_deg()
            for i, name in enumerate(self.config.joint_names):
                signed_deg = joint_deg[i] * self.config.joint_signs[i]
                obs[f"{name}.pos"] = self._deg_to_pct(signed_deg, oriented_min[i], oriented_max[i])
            obs["gripper.pos"] = self._mm_to_pct(gripper_mm)
        else:  # "deg" | "rad" — raw signed angles, gripper in mm
            for i, name in enumerate(self.config.joint_names):
                signed_deg = joint_deg[i] * self.config.joint_signs[i]
                obs[f"{name}.pos"] = signed_deg if unit == "deg" else math.radians(signed_deg)
            obs["gripper.pos"] = gripper_mm

        obs["gripper.tau"] = self._tau_filtered

        for cam_key, cam in self.cameras.items():
            try:
                frame = cam.async_read()
                self._last_frames[cam_key] = frame
            except Exception as exc:
                logger.warning("[PiperFull] Camera '%s' read failed: %s — using last frame", cam_key, exc)
                frame = self._last_frames.get(cam_key)
                if frame is None:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            obs[cam_key] = frame

        return obs

    def _filter_gripper_tau(self, force_n: float | None) -> None:
        # The SDK reports gripper force with the opposite sign of the physical
        # squeeze direction. Keep the observation positive-only and suppress
        # small no-contact noise before the EMA.
        tau = max(0.0, -(force_n or 0.0))
        deadband = max(0.0, float(self.config.gripper_tau_deadband_n))
        if tau < deadband:
            tau = 0.0
        alpha = max(0.0, min(1.0, float(self.config.tau_lowpass_alpha)))
        self._tau_filtered = max(0.0, alpha * tau + (1.0 - alpha) * self._tau_filtered)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        sdk = self._require_sdk()

        # Convert the action (in ``config.unit``) to hardware degrees / mm.
        unit = self.config.unit
        if unit == "pct":
            oriented_min, oriented_max = self._oriented_joint_limits_deg()
            hw_joints_deg = []
            for i, name in enumerate(self.config.joint_names):
                pct = max(-100.0, min(100.0, float(action[f"{name}.pos"])))
                signed_deg = self._pct_to_deg(pct, oriented_min[i], oriented_max[i])
                hw_joints_deg.append(signed_deg * self.config.joint_signs[i])
            gripper_mm = self._pct_to_mm(float(action["gripper.pos"])) if "gripper.pos" in action else None
        else:  # "deg" | "rad" — raw signed angles, gripper in mm
            # Clamp to the SDK joint limits: the pct path clamps implicitly
            # (±100 maps onto the exact limits), the raw-angle path must not
            # rely on the firmware alone.
            lim_min, lim_max = sdk.joint_limits_deg
            hw_joints_deg = []
            for i, name in enumerate(self.config.joint_names):
                value = float(action[f"{name}.pos"])
                signed_deg = value if unit == "deg" else math.degrees(value)
                hw_deg = signed_deg * self.config.joint_signs[i]
                hw_joints_deg.append(min(max(hw_deg, lim_min[i]), lim_max[i]))
            # Clamp to the physical stroke: the pct path clamps implicitly via
            # _pct_to_mm, the raw-mm path must not forward out-of-range targets.
            gripper_mm = (
                min(max(float(action["gripper.pos"]), self.config.gripper_min_mm), self.config.gripper_max_mm)
                if "gripper.pos" in action
                else None
            )

        # Optional safety cap on relative delta.
        if self.config.max_relative_target is not None and self._last_joint_deg is not None:
            hw_joints_deg = self._apply_relative_cap(hw_joints_deg)

        sdk.set_joint_positions_deg(
            hw_joints_deg,
            gripper_mm=gripper_mm,
            gripper_force=self.config.gripper_force_n,
        )
        return action

    # ------------------------------------------------------------------
    # EE-policy execution support (engine-side EE→joint conversion)
    # ------------------------------------------------------------------

    def make_ee_chunk_smoother(
        self,
        *,
        policy_action_keys: list[str],
        output_action_keys: list[str],
        joint_keys: list[str] | None = None,  # noqa: ARG002 — joint names come from this config
        gripper_key: str | None = "gripper.pos",
        v_max_deg_s: float,
        lambda_a: float,
        lambda_j: float,
        rate_hz: float,
        strict_anchor: bool = False,
    ):
        """Build an EE→joint chunk smoother emitting SDK-ready joint actions.

        Called (duck-typed) by the ``qp_sync``/``qp_rtc`` engines when the
        policy emits ``ee.*`` poses: the chunk is projected, batch-IK'd and
        joint-QP-smoothed, and the output columns are this robot's joint
        actions (pct, signed degrees or radians, matching ``config.unit``) —
        the arm then receives plain joint commands, no Cartesian round-trip.
        """
        from ..piper_ee.ee_action_projector import EEActionProjector
        from ..piper_ee.ee_chunk_smoother import EEChunkSmoother

        cfg = self.config
        kin = self._get_ee_kinematics()
        null_space_target = np.full(kin.nq, np.nan, dtype=np.float64)
        has_null_space_target = False
        for idx, value in (
            (3, cfg.ee_null_space_neutral_j4_deg),
            (5, cfg.ee_null_space_neutral_j6_deg),
        ):
            if value is not None:
                null_space_target[idx] = math.radians(float(value))
                has_null_space_target = True
        projector = EEActionProjector(
            kin,
            max_dpos_per_step=cfg.ee_projector_max_lin_vel_m_s / rate_hz,
            max_drot_per_step=cfg.ee_projector_max_ang_vel_rad_s / rate_hz,
            solve_kwargs={
                "yaw_mode": cfg.ee_yaw_mode,
                "yaw_weight": cfg.ee_yaw_weight,
                "fallback_joint_jump_rad": cfg.ee_fallback_joint_jump_rad,
                "max_joint_step": math.radians(v_max_deg_s) / rate_hz,
                "null_space_target": null_space_target if has_null_space_target else None,
                "null_space_target_weight": cfg.ee_null_space_neutral_weight,
                "null_space_target_tolerance": math.radians(cfg.ee_null_space_neutral_tolerance_deg),
            },
        )
        return EEChunkSmoother(
            kin,
            projector,
            policy_action_keys=policy_action_keys,
            output_action_keys=output_action_keys,
            output_space="joint",
            joint_output_keys=[f"{name}.pos" for name in cfg.joint_names],
            gripper_key=gripper_key,
            q_rad_to_action=self._q_signed_rad_to_action,
            v_max_deg_s=v_max_deg_s,
            lambda_a=lambda_a,
            lambda_j=lambda_j,
            rate_hz=rate_hz,
            strict_anchor=strict_anchor,
        )

    def _get_ee_kinematics(self):
        """Cached pinocchio FK/IK backend on the bundled Piper URDF."""
        if getattr(self, "_ee_kinematics", None) is None:
            from ..piper_ee.piper_ee_ik import PiperEEKinematics

            self._ee_kinematics = PiperEEKinematics(urdf_path=self.config.ee_urdf_path)
        return self._ee_kinematics

    def _q_signed_rad_to_action(self, q_rad: np.ndarray) -> np.ndarray:
        """EE-frame (signed) radians → this robot's joint action unit.

        Counterpart of ``ee_anchor_q_from_observation``: no sign flip — the
        signed angles ARE the action convention, mapped to ``config.unit``
        (pct through the oriented limits, degrees, or radians).
        ``send_action`` then undoes the signs toward the SDK, which lands the
        physical arm on the mirror of the EE-frame pose — i.e. exactly the
        pose the ``*_ee``-trained policy intended.
        """
        q_rad = np.asarray(q_rad, dtype=np.float64)
        if self.config.unit == "rad":
            return q_rad
        signed_deg = np.degrees(q_rad)
        if self.config.unit == "deg":
            return signed_deg
        omin, omax = self._oriented_joint_limits_deg()
        omin = np.asarray(omin, dtype=np.float64)
        omax = np.asarray(omax, dtype=np.float64)
        span = np.where((omax - omin) <= 0, 1.0, omax - omin)
        pct = (signed_deg - omin) / span * 200.0 - 100.0
        return np.clip(pct, -100.0, 100.0)

    def ee_anchor_q_from_observation(self, obs: dict[str, Any]) -> np.ndarray | None:
        """Observation joint values → radians in the EE-convention (signed) frame.

        The ``*_ee`` datasets were generated by running FK directly on the
        signed observation degrees — ``joint_signs`` NOT undone (verified to
        zero error against the ``_real``/``_real_ee`` pair).  Negating joints
        1/4/6 mirrors the pose across the xz-plane, so this frame is a
        mirrored twin of the physical arm; those joints have symmetric limits,
        hence the same URDF model and limits apply.  The entire engine-side EE
        path (state FK, IK anchor, chunk smoothing) works in this frame and
        ``_q_signed_rad_to_action`` maps its joints back to robot actions.
        """
        vals = []
        for name in self.config.joint_names:
            v = obs.get(f"{name}.pos")
            if not isinstance(v, (int, float)):
                return None
            vals.append(float(v))
        values = np.asarray(vals, dtype=np.float64)
        unit = self.config.unit
        if unit == "rad":
            return values
        if unit == "pct":
            omin, omax = self._oriented_joint_limits_deg()
            omin = np.asarray(omin, dtype=np.float64)
            omax = np.asarray(omax, dtype=np.float64)
            values = omin + (values + 100.0) / 200.0 * (omax - omin)
        return np.deg2rad(values)

    def urdf_q_from_observation(self, obs: dict[str, Any]) -> np.ndarray | None:
        """Map an observation dict to URDF joint radians, or None if incomplete.

        Inverse of the observation convention: pct → signed degrees (through
        the oriented limits) for unit="pct", then the sign is
        undone and degrees become radians.
        """
        q_signed = self.ee_anchor_q_from_observation(obs)
        if q_signed is None:
            return None
        signs = np.asarray(self.config.joint_signs, dtype=np.float64)
        return q_signed * signs

    def ee_pose_from_observation(self, obs: dict[str, Any]) -> dict[str, float] | None:
        """FK: observation joint values → the six ``ee.*`` pose scalars.

        Computed in the EE-convention (signed) frame the ``*_ee`` datasets
        were generated in (see ``ee_anchor_q_from_observation``), so policies
        trained on them see a consistent state.  Used (duck-typed) by the qp
        engines to feed EE-state policies running on this joint-space robot.
        Returns None when the observation lacks joint values.
        """
        q_rad = self.ee_anchor_q_from_observation(obs)
        if q_rad is None:
            return None
        pos, _rot, rpy = self._get_ee_kinematics().fk(q_rad)
        return {
            "ee.x": float(pos[0]),
            "ee.y": float(pos[1]),
            "ee.z": float(pos[2]),
            "ee.roll": float(rpy[0]),
            "ee.pitch": float(rpy[1]),
            "ee.yaw": float(rpy[2]),
        }

    # ------------------------------------------------------------------
    # Optional helpers exposed to scripts (parallel to legacy API)
    # ------------------------------------------------------------------

    def go_home_slow(self, *, speed_pct: int = 20, timeout_s: float = 8.0) -> None:
        self._require_sdk().go_home_slow(speed_pct=speed_pct, timeout_s=timeout_s)

    def electronic_emergency_stop(self) -> None:
        self._require_sdk().electronic_emergency_stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_sdk(self) -> PiperFullSDK:
        if not self.is_connected or self._sdk is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._sdk

    def _oriented_joint_limits_deg(self) -> tuple[list[float], list[float]]:
        sdk = self._require_sdk()
        min_deg, max_deg = sdk.joint_limits_deg
        oriented_min: list[float] = []
        oriented_max: list[float] = []
        for i, sign in enumerate(self.config.joint_signs):
            if sign >= 0:
                oriented_min.append(min_deg[i])
                oriented_max.append(max_deg[i])
            else:
                oriented_min.append(-max_deg[i])
                oriented_max.append(-min_deg[i])
        return oriented_min, oriented_max

    @staticmethod
    def _deg_to_pct(deg: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        pct = (deg - lo) / (hi - lo) * 200.0 - 100.0
        return max(-100.0, min(100.0, pct))

    @staticmethod
    def _pct_to_deg(pct: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return lo
        return lo + (pct + 100.0) / 200.0 * (hi - lo)

    def _mm_to_pct(self, mm: float) -> float:
        lo, hi = self.config.gripper_min_mm, self.config.gripper_max_mm
        if hi <= lo:
            return 0.0
        pct = (mm - lo) / (hi - lo) * 100.0
        return max(0.0, min(100.0, pct))

    def _pct_to_mm(self, pct: float) -> float:
        lo, hi = self.config.gripper_min_mm, self.config.gripper_max_mm
        pct = max(0.0, min(100.0, pct))
        return lo + (hi - lo) * pct / 100.0

    def _apply_relative_cap(self, goal_deg: list[float]) -> list[float]:
        assert self._last_joint_deg is not None
        # Both goal_deg and _last_joint_deg are in the raw HW frame: the
        # magnitude cap is frame-invariant, so no sign flip is needed.
        goal_present = {
            name: (goal_deg[i], self._last_joint_deg[i]) for i, name in enumerate(self.config.joint_names)
        }
        safe = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        return [safe[name] for name in self.config.joint_names]
