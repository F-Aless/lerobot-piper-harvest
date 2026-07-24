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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_full")
@dataclass
class PiperFullConfig(RobotConfig):
    """AgileX Piper 6-DOF arm + parallel gripper, full joint-space control via piper_sdk.

    Observation / action semantics (stable across policies):
      joint_{1..6}.pos   — [-100, 100] normalized, degrees or radians (see ``unit``)
      gripper.pos        — normalized to [0, 100] (0 = closed, 100 = fully open at ``gripper_max_mm``)
      gripper.tau        — EMA-filtered gripper force in N (observation only)
    """

    # --- CAN transport ---
    can_channel: str = "can0"
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000

    # --- Piper firmware variant ---
    # "default" (≤ S-V1.8-2) | "v183" (S-V1.8-3..7) | "v188" (≥ S-V1.8-8)
    firmware: str = "v188"

    # --- Arm kinematics ---
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i + 1}" for i in range(6)])
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])

    # --- Normalization ---
    # Joint angle unit of observations/actions:
    #   "pct" — normalized to [-100, 100] through the oriented joint limits
    #           (the convention all our datasets/policies were trained with)
    #   "deg" — signed degrees, gripper.pos in mm
    #   "rad" — signed radians, gripper.pos in mm
    # None derives the unit from the legacy ``use_degrees`` flag.
    # WARNING: only "pct" has been validated on the real arm; "deg" and "rad"
    # are implemented but NOT hardware-tested.
    unit: str | None = None
    # Legacy alias: use_degrees=True is equivalent to unit="deg".
    use_degrees: bool = False

    # Physical max gripper stroke (mm) — the Piper gripper's usable travel is
    # 70 mm (the SDK can report up to ~100 mm raw, the last ~30 mm being a
    # mechanical dead zone).  The legacy fork called this "7.0" because its
    # tick conversion was 10× off (its "mm" were physically cm); the pct
    # mapping is unchanged, so existing datasets/policies are unaffected.
    gripper_max_mm: float = 70.0
    gripper_min_mm: float = 0.0

    # --- Dynamics / runtime ---
    tau_lowpass_alpha: float = 0.1
    gripper_tau_deadband_n: float = 0.2
    speed_percent: int = 50
    enable_timeout_s: float = 5.0
    gripper_force_n: float = 1.0

    # Optional safety: cap max |Δpos| per step. See `ensure_safe_goal_position`.
    max_relative_target: float | dict[str, float] | None = None

    # --- Cameras ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # EE-policy execution (engine-side EE→joint conversion)
    # ------------------------------------------------------------------
    # Used only when an ee.* pose policy runs through the qp_sync/qp_rtc
    # engines: each chunk is projected, batch-IK'd and joint-QP-smoothed, and
    # the arm receives plain SDK joint commands.  Kinematics come from the
    # bundled Piper URDF (requires pinocchio, lazily imported).
    ee_urdf_path: str | None = None
    # Yaw handling for the IK: "free" ignores the policy's yaw channel — the
    # realized yaw emerges from the kinematics (one natural yaw per x/y with
    # the wrist kept still).  See PiperEEConfig for the full rationale.
    ee_yaw_mode: str = "free"
    ee_yaw_weight: float = 0.10
    # Guard against the slow-motion wrist flip near the home singularity.
    ee_fallback_joint_jump_rad: float = 0.6
    # Yaw-free neutral-posture bias for EE policies executed on piper_full.
    #
    # In "free" mode the IK ignores EE yaw and solves only position + roll/pitch.
    # That removes the singular yaw task, but it also leaves one wrist/yaw
    # direction underdetermined. The seed bias keeps chunks continuous and
    # prevents sudden wrist flips, but if the command stream has already drifted
    # to j4=40deg, j6=-40deg, "stay near the seed" also keeps the awkward pose.
    #
    # These targets add a soft pull inside the same null-space. They are not hard
    # limits: the EE task wins when it genuinely needs those angles. They only
    # drain free-yaw drift when redundant motion is available. Set either target
    # to None to leave that joint unbiased. Increase weight for faster recovery;
    # decrease it if the projected chunk starts carrying too much EE residual.
    ee_null_space_neutral_j4_deg: float | None = 0.0
    ee_null_space_neutral_j6_deg: float | None = 0.0
    ee_null_space_neutral_weight: float = 0.15
    ee_null_space_neutral_tolerance_deg: float = 2.0
    # Per-step projector caps, as velocities (per-step = vel / engine rate).
    ee_projector_max_lin_vel_m_s: float = 0.5
    ee_projector_max_ang_vel_rad_s: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unit is None:
            self.unit = "deg" if self.use_degrees else "pct"
        if self.unit not in ("pct", "deg", "rad"):
            raise ValueError(f"piper_full: unit must be 'pct', 'deg' or 'rad', got {self.unit!r}")
        if self.use_degrees and self.unit != "deg":
            raise ValueError(
                f"piper_full: use_degrees=True conflicts with unit={self.unit!r} — set only one "
                "(use_degrees is a legacy alias for unit='deg')."
            )
