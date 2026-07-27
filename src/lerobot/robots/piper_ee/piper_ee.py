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

"""End-effector controlled Piper.

Action space:

    ee.x, ee.y, ee.z       — target EE position in metres.
    ee.roll, ee.pitch, ee.yaw — target orientation in radians (R = Rz·Ry·Rx).
    gripper.pos            — gripper opening in [0, 100] percent.

Each ``send_action`` call solves an IK against the bundled URDF, applies a
per-command safety cap on the joint delta, and forwards the joint targets
to the underlying ``PiperFullSDK``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from ..piper_full.piper_full_sdk import PiperFullSDK
from .config_piper_ee import PiperEEConfig

logger = logging.getLogger(__name__)


_EE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


class PiperEE(Robot):
    """Piper 6-DOF with Cartesian (EE) action interface."""

    config_class = PiperEEConfig
    name = "piper_ee"

    def __init__(self, config: PiperEEConfig):
        super().__init__(config)
        self.config = config
        self._sdk: PiperFullSDK | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._tau_filtered: float = 0.0
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_joint_deg: list[float] | None = None
        self._last_gripper_mm: float | None = None

        # Lazy IK construction — pinocchio is imported only here, on first init.
        from .piper_ee_ik import PiperEEKinematics

        self._ik = PiperEEKinematics(
            urdf_path=config.urdf_path,
            ee_frame_parent=config.ee_frame_parent,
            ee_translation=tuple(config.ee_translation),
            ee_rpy=tuple(config.ee_rpy),
        )

        self._projector = self.make_action_projector(rate_hz=config.rate_hz)
        # Last commanded joint target (URDF rad) — IK seed / projector reference.
        self._q_cmd_last: np.ndarray | None = None
        self._dump_path: Path | None = None
        self._dump_count: int = 0

    # ------------------------------------------------------------------
    # IK solve kwargs / projector + chunk-smoother factories
    # ------------------------------------------------------------------

    def _ik_solve_kwargs(self) -> dict[str, Any]:
        """Solver kwargs derived from the config (shared by all projectors)."""
        cfg = self.config
        return {
            "max_iter": cfg.ik_max_iter,
            "damping": cfg.ik_damping,
            "pos_tol": cfg.ik_pos_tol_m,
            "rot_tol": cfg.ik_rot_tol_rad,
            "step_cap": cfg.ik_step_cap_rad_per_iter,
            "yaw_mode": cfg.yaw_mode,
            "yaw_weight": cfg.yaw_weight,
            "condition_fallback_threshold": cfg.condition_fallback_threshold,
            "fallback_joint_jump_rad": cfg.fallback_joint_jump_rad,
            "max_joint_step": cfg.max_joint_step_rad,
            "null_space_weight": cfg.null_space_weight,
            "null_space_target": self._null_space_target_rad(),
            "null_space_target_weight": cfg.null_space_neutral_weight,
            "null_space_target_tolerance": math.radians(cfg.null_space_neutral_tolerance_deg),
            "adaptive_damping": cfg.ik_adaptive_damping,
            "damping_max": cfg.ik_damping_max,
            "sigma_threshold": cfg.ik_sigma_threshold,
        }

    def _null_space_target_rad(self) -> np.ndarray | None:
        """Optional yaw-free neutral posture target, in URDF radians."""
        cfg = self.config
        target = np.full(self._ik.nq, np.nan, dtype=np.float64)
        has_target = False
        for idx, value in (
            (3, cfg.null_space_neutral_j4_deg),
            (5, cfg.null_space_neutral_j6_deg),
        ):
            if value is not None:
                target[idx] = math.radians(float(value))
                has_target = True
        return target if has_target else None

    def make_action_projector(self, rate_hz: float | None = None):
        """Build an :class:`EEActionProjector` sized for ``rate_hz`` ticks."""
        from .ee_action_projector import EEActionProjector

        rate = float(rate_hz or self.config.rate_hz)
        cfg = self.config
        return EEActionProjector(
            self._ik,
            max_dpos_per_step=cfg.projector_max_lin_vel_m_s / rate,
            max_drot_per_step=cfg.projector_max_ang_vel_rad_s / rate,
            scale_retry_factor=cfg.projector_scale_retry_factor,
            solve_kwargs=self._ik_solve_kwargs(),
        )

    def make_ee_chunk_smoother(
        self,
        *,
        policy_action_keys: list[str],
        output_action_keys: list[str],
        joint_keys: list[str] | None = None,  # noqa: ARG002 — exec space is Cartesian here
        gripper_key: str | None = "gripper.pos",
        v_max_deg_s: float,
        lambda_a: float,
        lambda_j: float,
        rate_hz: float,
        strict_anchor: bool = False,
    ):
        """Build an :class:`EEChunkSmoother` emitting Cartesian (EE) actions.

        Called (duck-typed) by the ``qp_sync``/``qp_rtc`` inference engines —
        see ee_chunk_smoother.py.  ``piper_ee``'s execution action space is EE, so the
        smoothed joint trajectory is FK'd back to poses; for direct SDK joint
        commands use ``piper_full`` instead.
        """
        from .ee_chunk_smoother import EEChunkSmoother

        return EEChunkSmoother(
            self._ik,
            self.make_action_projector(rate_hz=rate_hz),
            policy_action_keys=policy_action_keys,
            output_action_keys=output_action_keys,
            output_space="ee",
            gripper_key=gripper_key,
            v_max_deg_s=v_max_deg_s,
            lambda_a=lambda_a,
            lambda_j=lambda_j,
            rate_hz=rate_hz,
            strict_anchor=strict_anchor,
        )

    @property
    def ee_kinematics(self):
        """FK/IK backend (shared with engine-side EE chunk smoothing)."""
        return self._ik

    def urdf_q_from_observation(self, obs: dict[str, Any]) -> np.ndarray | None:
        """Map an observation dict to URDF joint radians, or None if incomplete.

        Observation joints are exported as ``hw_deg * dataset_joint_signs``;
        the URDF frame is plain ``deg2rad(hw_deg)`` (see
        :meth:`_hw_deg_to_urdf_rad`), so the sign is undone here.
        """
        vals = []
        for i, name in enumerate(self.config.joint_names):
            key = f"{name}.pos"
            v = obs.get(key)
            if not isinstance(v, (int, float)):
                return None
            vals.append(float(v) * self.config.dataset_joint_signs[i])
        return np.deg2rad(np.asarray(vals, dtype=np.float64))

    # ------------------------------------------------------------------
    # Connection / lifecycle (mirrors piper_full)
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return (
            self._sdk is not None
            and self._sdk.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
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
        # Fresh session: drop stale seeding/fallback state from a previous run.
        self._q_cmd_last = None
        self._projector.reset()
        self._dump_path = None
        self._dump_count = 0
        logger.info("[PiperEE] Connected (%s, fw=%s)", self.config.can_channel, self.config.firmware)

    def disconnect(self) -> None:
        if self._sdk is not None:
            self._sdk.disconnect()
            self._sdk = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("[PiperEE] Disconnected")

    def go_home_slow(self, *, speed_pct: int = 20, timeout_s: float = 8.0) -> None:
        """Slowly home all joints to true (unmirrored) 0° — the generic safe
        end-of-session pose, NOT "0" in any normalized/EE convention.  Mirrors
        ``PiperFull.go_home_slow``; duck-typed by
        ``rollout/strategies/core.py``'s teardown, which prefers this over
        interpolating back to the connect-time pose when the robot exposes it.
        """
        self._require_sdk().go_home_slow(speed_pct=speed_pct, timeout_s=timeout_s)
        # The commanded/seed joint state now jumped outside the IK-tracked
        # path — drop it so the next send_action re-anchors from the
        # measured (home) position instead of interpolating from stale state.
        self._q_cmd_last = None
        self._projector.reset()
        # Also drop the last-known-good pose cache: if the very next
        # get_observation/send_action hits a transient CAN read failure, fail
        # loud instead of silently seeding from the pre-home pose.
        self._last_joint_deg = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Feature schemas
    # ------------------------------------------------------------------

    @property
    def _joint_obs_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def _ee_ft(self) -> dict[str, type]:
        return dict.fromkeys(_EE_KEYS, float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        # joints + ee pose (FK) + gripper opening/force + cameras
        ft: dict[str, Any] = {
            **self._joint_obs_ft,
            **self._ee_ft,
            "gripper.pos": float,
            "gripper.tau": float,
        }
        ft.update(self._cameras_ft)
        return ft

    @property
    def action_features(self) -> dict:
        # Cartesian + gripper. Names are NOT .pos-suffixed for the EE components —
        # the rollout context filter will need a relax (task #6) before these flow
        # into the dataset. They DO flow into ``send_action`` regardless.
        return {**self._ee_ft, "gripper.pos": float}

    @property
    def joint_limits_deg(self) -> tuple[list[float], list[float]] | None:
        if self._sdk is None or not self._sdk.is_connected:
            return None
        oriented_min, oriented_max = self._oriented_joint_limits_deg()
        return list(oriented_min), list(oriented_max)

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

        # Joint values to URDF-frame radians, then FK.
        q_urdf_rad = self._hw_deg_to_urdf_rad(joint_deg)
        pos, _R, rpy = self._ik.fk(q_urdf_rad)

        obs: dict[str, Any] = {
            f"{name}.pos": joint_deg[i] * self.config.dataset_joint_signs[i]
            for i, name in enumerate(self.config.joint_names)
        }
        obs["ee.x"] = float(pos[0])
        obs["ee.y"] = float(pos[1])
        obs["ee.z"] = float(pos[2])
        obs["ee.roll"] = float(rpy[0])
        obs["ee.pitch"] = float(rpy[1])
        obs["ee.yaw"] = float(rpy[2])
        obs["gripper.pos"] = self._mm_to_pct(gripper_mm)
        obs["gripper.tau"] = self._tau_filtered

        for cam_key, cam in self.cameras.items():
            try:
                frame = cam.async_read()
                self._last_frames[cam_key] = frame
            except Exception as exc:
                logger.warning("[PiperEE] Camera '%s' read failed: %s — using last frame", cam_key, exc)
                frame = self._last_frames.get(cam_key)
                if frame is None:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            obs[cam_key] = frame

        return obs

    def _filter_gripper_tau(self, force_n: float | None) -> None:
        tau = max(0.0, -(force_n or 0.0))
        deadband = max(0.0, float(self.config.gripper_tau_deadband_n))
        if tau < deadband:
            tau = 0.0
        alpha = max(0.0, min(1.0, float(self.config.tau_lowpass_alpha)))
        self._tau_filtered = max(0.0, alpha * tau + (1.0 - alpha) * self._tau_filtered)

    # ------------------------------------------------------------------
    # Action — EE → IK → joints → SDK
    # ------------------------------------------------------------------

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        sdk = self._require_sdk()

        joint_deg = sdk.get_joint_angles_deg()
        if joint_deg is None:
            if self._last_joint_deg is None:
                raise RuntimeError("Piper has not reported joint angles yet — CAN bus up?")
            joint_deg = self._last_joint_deg

        q_meas_rad = self._hw_deg_to_urdf_rad(joint_deg)

        # Reference state: the last commanded joint target when it still
        # tracks the measured position (measured joints lag the command under
        # load → seeding from them oscillates), the measured position after a
        # divergence (stall / external contact) or at session start.
        q_ref = q_meas_rad
        seed_resynced = False
        if self.config.seed_from_last_command and self._q_cmd_last is not None:
            if float(np.max(np.abs(self._q_cmd_last - q_meas_rad))) <= self.config.seed_resync_threshold_rad:
                q_ref = self._q_cmd_last
            else:
                seed_resynced = True
                logger.warning(
                    "[PiperEE] commanded/measured joints diverged (>%.2f rad) — re-anchoring to measured",
                    self.config.seed_resync_threshold_rad,
                )

        current_pos, current_R, _ = self._ik.fk(q_ref)

        target_pos = np.array(
            [float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])],
            dtype=np.float64,
        )
        target_R = self._ik.rpy_to_matrix(
            float(action["ee.roll"]),
            float(action["ee.pitch"]),
            float(action["ee.yaw"]),
        )

        # Projection ladder: clamp → warm IK (soft yaw + fallbacks) → scale
        # retry → last-valid (see EEActionProjector).
        result = self._projector.project(
            target_pos,
            target_R,
            current_pos=current_pos,
            current_R=current_R,
            q_seed=q_ref,
        )

        # Outer per-command safety cap, direction-preserving (the whole delta
        # is scaled, never clipped per-joint, so the motion direction is
        # kept).  The solver caps its own solutions; this protects the
        # last-valid fallback path, whose joints may be far from q_ref.
        max_step = float(
            self.config.max_joint_step_rad or math.radians(self.config.v_max_deg_s / self.config.rate_hz)
        )
        delta = result.q - q_ref
        max_abs = float(np.max(np.abs(delta)))
        outer_capped = max_abs > max_step
        if outer_capped:
            delta = delta * (max_step / max_abs)
        q_cmd_rad = np.clip(q_ref + delta, self._ik.q_min, self._ik.q_max)
        self._q_cmd_last = q_cmd_rad

        q_cmd_hw_deg = self._urdf_rad_to_hw_deg(q_cmd_rad)
        gripper_mm = self._pct_to_mm(float(action["gripper.pos"])) if "gripper.pos" in action else None
        sdk.set_joint_positions_deg(
            q_cmd_hw_deg,
            gripper_mm=gripper_mm,
            gripper_force=self.config.gripper_force_n,
        )

        if self.config.debug_dump_dir is not None:
            self._dump_command(
                action=action,
                result=result,
                q_ref=q_ref,
                q_meas=q_meas_rad,
                q_cmd=q_cmd_rad,
                outer_capped=outer_capped,
                seed_resynced=seed_resynced,
            )
        return action

    # ------------------------------------------------------------------
    # Raw-vs-projected dump
    # ------------------------------------------------------------------

    def _dump_command(
        self,
        *,
        action: dict[str, Any],
        result,
        q_ref: np.ndarray,
        q_meas: np.ndarray,
        q_cmd: np.ndarray,
        outer_capped: bool,
        seed_resynced: bool,
    ) -> None:
        """Append one JSON line: raw action, projected pose, joints, IK info."""
        try:
            if self._dump_path is None:
                dump_dir = Path(self.config.debug_dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
                self._dump_path = dump_dir / f"piper_ee_commands_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                logger.info("[PiperEE] raw-vs-projected dump → %s", self._dump_path)

            proj_rpy = self._ik.matrix_to_rpy(result.R)
            info = result.info
            record = {
                "n": self._dump_count,
                "t": time.time(),
                "action_raw_ee": [float(action[k]) for k in _EE_KEYS],
                "action_projected_ee": [*map(float, result.pos), *map(float, proj_rpy)],
                "q_ref_deg": np.round(np.degrees(q_ref), 4).tolist(),
                "q_meas_deg": np.round(np.degrees(q_meas), 4).tolist(),
                "q_cmd_deg": np.round(np.degrees(q_cmd), 4).tolist(),
                "outer_capped": outer_capped,
                "seed_resynced": seed_resynced,
                "ik": {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in info.items()
                    if isinstance(v, (bool, int, float, str))
                },
            }
            self._dump_count += 1
            with open(self._dump_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.debug("[PiperEE] command dump failed", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_sdk(self) -> PiperFullSDK:
        if not self.is_connected or self._sdk is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._sdk

    def _hw_deg_to_urdf_rad(self, joint_hw_deg) -> np.ndarray:
        """Map ``PiperFullSDK.get_joint_angles_deg()`` to URDF radians.

        Empirically verified on the real arm (FK-vs-SDK consistency check at
        <1mm / <0.02°): the SDK reports joints already in the URDF frame.
        ``dataset_joint_signs`` mirrors the *observation/action* vector into
        the dataset-training frame (see ``config_piper_full.py`` and
        ``get_observation``/``ee_anchor_q_from_observation`` for the
        piper_full analog); it is not a HW↔URDF transform, so it is not
        applied here — FK/IK always run in the true, unmirrored hw frame.
        """
        return np.deg2rad(np.asarray(joint_hw_deg, dtype=np.float64))

    def _urdf_rad_to_hw_deg(self, q_urdf_rad: np.ndarray) -> list[float]:
        """Inverse of :meth:`_hw_deg_to_urdf_rad`."""
        return list(np.rad2deg(np.asarray(q_urdf_rad, dtype=np.float64)))

    def _oriented_joint_limits_deg(self) -> tuple[list[float], list[float]]:
        sdk = self._require_sdk()
        min_deg, max_deg = sdk.joint_limits_deg
        oriented_min, oriented_max = [], []
        for i, sign in enumerate(self.config.dataset_joint_signs):
            if sign >= 0:
                oriented_min.append(min_deg[i])
                oriented_max.append(max_deg[i])
            else:
                oriented_min.append(-max_deg[i])
                oriented_max.append(-min_deg[i])
        return oriented_min, oriented_max

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
