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

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower_7dof")
@dataclass
class SO101Follower7DofConfig(RobotConfig):
    """SO-ARM 101 follower variant with 7 Feetech sts3215 motors (6 joints + gripper).

    Matches the joint layout of :class:`SO101Leader7Dof` so the same action
    schema (``joint_1.pos``..``joint_6.pos``, ``gripper.pos``) can drive either
    this follower or the AgileX Piper without remapping.
    """

    # --- Bus transport ---
    port: str

    # --- Normalization ---
    # Body joints default to [-100, 100] for cross-arm policy transfer.
    use_degrees: bool = False

    # --- Safety / dynamics ---
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None

    # --- Cameras ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
