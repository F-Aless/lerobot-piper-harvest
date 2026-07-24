#!/usr/bin/env python

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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig


@TeleoperatorConfig.register_subclass("so_leader_piper")
@dataclass
class SOLeaderPiperConfig(TeleoperatorConfig, SOLeaderConfig):
    """Classic 5-DOF SO-ARM (100/101) leader remapped to drive a 7-DOF AgileX Piper.

    The SO-ARM lacks the Piper's forearm-roll DOF, so ``joint_4`` is held at a
    fixed value while the five real arm joints map straight through::

        shoulder_pan  -> joint_1     (base yaw)
        shoulder_lift -> joint_2     (shoulder pitch)
        elbow_flex    -> joint_3     (elbow pitch)
                      -> joint_4 = fixed_joint_4   (forearm roll, absent on SO-ARM)
        wrist_flex    -> joint_5     (wrist pitch)
        wrist_roll    -> joint_6     (end roll)
        gripper       -> gripper

    Output keys/units match :class:`PiperFull`'s action schema (``joint_*.pos`` in
    ``[-100, 100]``, ``gripper.pos`` in ``[0, 100]``), so the standard teleop
    pipeline forwards them to the Piper 1:1 — exactly like ``so101_leader_7dof``,
    minus the physical sixth motor.

    Uses its own ``so_leader_piper`` calibration directory: the leader must be
    calibrated specifically for the Piper-driving role (sweeping each joint in
    the direction/range that matches the corresponding Piper joint), so a fresh
    calibration is run rather than reusing the plain ``so_leader`` one.
    """

    # Match the Piper / so101_leader_7dof normalized convention (NOT degrees),
    # so a leader joint at ±100 maps to the matching Piper joint limit.
    use_degrees: bool = False

    # Value sent on the Piper's missing forearm-roll DOF, in the normalized
    # [-100, 100] convention; 0 = mid-range (forearm "straight").
    fixed_joint_4: float = 0.0

    # Per-joint direction for joints 1..6: set an entry to -1 to flip a joint
    # that moves opposite to the Piper. Index 3 (joint_4) is ignored (fixed).
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
