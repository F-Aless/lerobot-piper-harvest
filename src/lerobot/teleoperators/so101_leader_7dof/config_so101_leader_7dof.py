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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_7dof")
@dataclass
class SO101Leader7DofConfig(TeleoperatorConfig):
    """SO-ARM 101 leader variant with 7 Feetech sts3215 motors (6 joints + gripper).

    Designed to drive 7-DOF followers such as the AgileX Piper
    (6 arm joints + parallel gripper).
    """

    # Serial port for the Feetech bus (e.g. /dev/ttyACM0)
    port: str

    # If ``True``, body joints are exposed in degrees; otherwise normalized to [-100, 100].
    use_degrees: bool = False
