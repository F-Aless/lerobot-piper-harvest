# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Classic 5-DOF SO-ARM leader remapped onto the 7-DOF AgileX Piper schema.

This is the 5-DOF analogue of :class:`SO101Leader7Dof`: it reads the same six
physical motors as the plain SO-ARM leader (``shoulder_pan``..``gripper``) but
emits the Piper action keys (``joint_1..joint_6.pos`` + ``gripper.pos``).  The
SO-ARM has no forearm-roll joint, so the Piper's ``joint_4`` is held at a fixed
value (default mid-range) to absorb the DOF difference; the remaining five arm
joints map straight through.  Output keys/units match :class:`PiperFull`, so the
standard teleop pipeline forwards them 1:1 with no key remapping.
"""

import logging
import time

from lerobot.utils.decorators import check_if_not_connected

from .config_so_leader import SOLeaderPiperConfig
from .so_leader import SOLeader

logger = logging.getLogger(__name__)


class SOLeaderPiper(SOLeader):
    """5-DOF SO-ARM leader exposing the 7-DOF Piper action schema (``joint_4`` fixed)."""

    config_class = SOLeaderPiperConfig
    # Own calibration directory: the leader is calibrated specifically for the
    # Piper-driving role, not reusing the plain ``so_leader`` calibration.
    name = "so_leader_piper"

    # classic SO-ARM motor  ->  (Piper action key, joint_signs index)
    # joint_4 (index 3) is intentionally absent: the SO-ARM has no forearm roll.
    _ARM_MAP = (
        ("shoulder_pan", "joint_1.pos", 0),
        ("shoulder_lift", "joint_2.pos", 1),
        ("elbow_flex", "joint_3.pos", 2),
        ("wrist_flex", "joint_5.pos", 4),
        ("wrist_roll", "joint_6.pos", 5),
    )
    _PIPER_FEATURE_KEYS = (
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "gripper.pos",
    )

    config: SOLeaderPiperConfig

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._PIPER_FEATURE_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        # Bus is keyed by the classic SO-ARM motor names (shoulder_pan, ...).
        raw = self.bus.sync_read("Present_Position")
        signs = self.config.joint_signs

        action = {key: raw[motor] * signs[idx] for motor, key, idx in self._ARM_MAP}
        action["joint_4.pos"] = float(self.config.fixed_joint_4)
        action["gripper.pos"] = raw["gripper"]
        # Emit in the canonical joint_1..joint_6 + gripper order.
        action = {key: action[key] for key in self._PIPER_FEATURE_KEYS}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Drive the leader servos toward a Piper-schema pose (handover/DAgger).

        Maps the Piper keys back to the classic motor names, dropping ``joint_4``
        (no matching motor).  Requires torque to be enabled by the caller.
        """
        signs = self.config.joint_signs
        # signs are ±1, so multiplying again inverts the get_action mapping.
        goals = {motor: feedback[key] * signs[idx] for motor, key, idx in self._ARM_MAP if key in feedback}
        if "gripper.pos" in feedback:
            goals["gripper"] = feedback["gripper.pos"]
        if goals:
            self.bus.sync_write("Goal_Position", goals)
