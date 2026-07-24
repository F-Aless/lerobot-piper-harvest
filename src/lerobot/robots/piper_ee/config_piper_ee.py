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

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_ee")
@dataclass
class PiperEEConfig(RobotConfig):
    """End-effector controlled Piper (xyz + rpy + gripper).

    Action features (Cartesian, in canonical ``ordered_action_keys``):

    * ``ee.x``, ``ee.y``, ``ee.z`` — target EE position in metres.
    * ``ee.roll``, ``ee.pitch``, ``ee.yaw`` — target orientation as ``R = Rz·Ry·Rx``
      (Tait-Bryan ZYX intrinsic, identical to XYZ extrinsic), in radians.
    * ``gripper.pos`` — gripper opening in [0, 100] percent.

    Observation features: the same set as ``piper_full`` (joint angles + gripper
    + cameras) plus the FK-computed EE pose under the same ``ee.*`` keys.
    """

    # --- CAN transport (same defaults as piper_full) ---
    can_channel: str = "can0"
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000

    # --- Piper firmware variant ---
    firmware: str = "v188"

    # --- Joint conventions (URDF↔hardware sign alignment) ---
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i + 1}" for i in range(6)])
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])

    # --- Gripper (mm stroke; SDK native unit is 0.001 mm ticks) ---
    gripper_max_mm: float = 70.0
    gripper_min_mm: float = 0.0

    # --- Dynamics / runtime ---
    tau_lowpass_alpha: float = 0.1
    gripper_tau_deadband_n: float = 0.2
    speed_percent: int = 50
    enable_timeout_s: float = 5.0
    gripper_force_n: float = 1.0

    # --- Cameras ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # IK / rate / safety
    # ------------------------------------------------------------------

    # Control rate of the rollout loop (Hz).  Used to compute the per-tick
    # joint-delta safety cap.
    rate_hz: float = 25.0

    # Maximum cartesian-equivalent joint speed (deg/s) used to derive
    # ``max_joint_step_rad`` when the latter is left as None.
    v_max_deg_s: float = 80.0

    # Hard cap on per-command joint delta (radians).  If None, computed in
    # ``__post_init__`` as ``deg2rad(v_max_deg_s / rate_hz)``.
    max_joint_step_rad: float | None = None

    # --- URDF + EE frame ---
    # If None, defaults to the bundled URDF at
    # ``src/lerobot/robots/piper_ee/piper/piper.urdf``.
    urdf_path: str | None = None
    ee_frame_parent: str = "joint6"
    ee_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- DLS solver parameters ---
    ik_max_iter: int = 50
    ik_damping: float = 0.01
    ik_pos_tol_m: float = 1e-3
    ik_rot_tol_rad: float = 5e-3
    ik_step_cap_rad_per_iter: float = 0.4
    null_space_weight: float = 0.1
    # Yaw-free neutral-posture bias.
    #
    # In "free" mode the IK ignores EE yaw and solves only position + roll/pitch.
    # That removes the singular yaw task, but it also means one wrist/yaw
    # direction is underdetermined. The regular seed bias keeps each tick close
    # to the previous command, which prevents sudden wrist flips, but it does
    # not by itself bring an already-drifted wrist back home: if the seed is
    # j4=40deg, j6=-40deg, "stay near the seed" keeps that awkward posture.
    #
    # These two optional targets add a second, global pull inside the same
    # null-space. It is not a hard safety limit: if the requested EE task really
    # needs j4/j6 away from zero, the task wins. If those angles are only free-yaw
    # drift, the solver spends the redundant DOF to drain them toward neutral.
    # Set either target to None to leave that joint unbiased. Increase weight for
    # faster recovery; decrease it if it starts trading off too much EE accuracy.
    null_space_neutral_j4_deg: float | None = 0.0
    null_space_neutral_j6_deg: float | None = 0.0
    null_space_neutral_weight: float = 0.15
    null_space_neutral_tolerance_deg: float = 2.0

    # --- Yaw task handling ---
    # "full": 6-D task, yaw tracked rigidly. "soft": yaw row down-weighted by
    # yaw_weight — followed when cheap, released near the yaw/position-coupled
    # singularity at home (pitch ≈ 85°, cond(J) ≈ 2.8e4).  "free": yaw row
    # dropped, null-space bias toward the seed and neutral wrist posture.
    # Default "free": policy-emitted yaw is not necessarily consistent with
    # the commanded x/y, and on this arm a given x/y admits essentially one
    # natural yaw once the wrist (joint_4/5) is kept from large rotations —
    # so the kinematics dictate yaw, the policy's yaw channel is ignored.
    # Use "soft" only when the policy's yaw output is trustworthy.
    yaw_mode: str = "free"
    yaw_weight: float = 0.10
    # In "full" mode, fall back to "free" upfront when cond(J) at the seed
    # exceeds this (the documented 80°-wrist-jump configuration).
    condition_fallback_threshold: float = 1e3
    # Retry yaw-free when a constrained solution moves any joint further than
    # this from the seed — the guard against the slow-motion wrist flip near
    # the home singularity.  <= 0 disables.
    fallback_joint_jump_rad: float = 0.6
    # Legacy alias: True -> yaw_mode="free", False -> "full".  None = ignored.
    free_yaw: bool | None = None

    # --- Singularity-adaptive DLS damping (Chiaverini) ---
    # The boost is keyed on the σ_min of the *essential* task (pos+roll+pitch
    # for soft/free, full 6-D for full).  Healthy Piper values are ≥ 0.02;
    # near the home singularity the 6-D σ_min collapses to ~6e-5.
    ik_adaptive_damping: bool = True
    ik_damping_max: float = 0.1
    ik_sigma_threshold: float = 0.01

    # --- EE action projector ---
    # Per-step caps are derived as vel / rate_hz.
    projector_max_lin_vel_m_s: float = 0.5
    projector_max_ang_vel_rad_s: float = 2.0
    # On IK failure the clamped target is pulled toward the current pose by
    # this factor and re-solved yaw-free.
    projector_scale_retry_factor: float = 0.5

    # --- IK seeding ---
    # Seed the IK (and the projector's "current" pose) from the last commanded
    # joint target instead of the measured position: measured joints lag the
    # command under load, and mixing the two causes tick-to-tick oscillation.
    seed_from_last_command: bool = True
    # If the last command and the measured position diverge beyond this on any
    # joint (stall, external contact), re-anchor to the measured position.
    seed_resync_threshold_rad: float = 0.35

    # --- Raw vs projected logging ---
    # When set, every send_action appends one JSON line with the raw action,
    # the projected pose, the commanded joints and the full IK diagnostics.
    debug_dump_dir: str | None = None

    # ------------------------------------------------------------------
    # __post_init__
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.max_joint_step_rad is None:
            self.max_joint_step_rad = math.radians(self.v_max_deg_s / self.rate_hz)
        if self.free_yaw is not None:
            self.yaw_mode = "free" if self.free_yaw else "full"
        if self.yaw_mode not in ("full", "soft", "free"):
            raise ValueError(f"yaw_mode must be 'full'|'soft'|'free', got {self.yaw_mode!r}")
