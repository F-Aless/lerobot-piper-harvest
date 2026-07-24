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

"""SO-ARM 101 leader arm, 7-DOF variant (6 body joints + gripper).

The 7-motor layout mirrors the AgileX Piper's kinematic chain so that actions
recorded from this leader can be replayed on :class:`PiperFull` or on the
matching :class:`SO101Follower7Dof` without any key remapping.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader_7dof import SO101Leader7DofConfig

logger = logging.getLogger(__name__)


class SO101Leader7Dof(Teleoperator):
    """Seven-motor SO-ARM 101 leader: joint_1..joint_6 + gripper on IDs 1..7."""

    config_class = SO101Leader7DofConfig
    name = "so101_leader_7dof"

    _MOTOR_KEYS = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    _JOINT_FEATURE_KEYS = tuple(f"{motor}.pos" for motor in _MOTOR_KEYS)

    def __init__(self, config: SO101Leader7DofConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=OrderedDict(
                (
                    ("joint_1", Motor(1, "sts3215", norm_mode_body)),
                    ("joint_2", Motor(2, "sts3215", norm_mode_body)),
                    ("joint_3", Motor(3, "sts3215", norm_mode_body)),
                    ("joint_4", Motor(4, "sts3215", norm_mode_body)),
                    ("joint_5", Motor(5, "sts3215", norm_mode_body)),
                    ("joint_6", Motor(6, "sts3215", norm_mode_body)),
                    ("gripper", Motor(7, "sts3215", MotorNormMode.RANGE_0_100)),
                )
            ),
            calibration=self.calibration,
        )

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return OrderedDict((key, float) for key in self._JOINT_FEATURE_KEYS)

    @property
    def feedback_features(self) -> dict[str, type]:
        # Same key space as action_features: position targets for each motor.
        # Non-empty so DAgger/episodic handovers treat this leader as actuated
        # (teleop_supports_feedback) and can drive it to the follower's pose.
        return OrderedDict((key, float) for key in self._JOINT_FEATURE_KEYS)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the "
                "calibration file (or no calibration file found)"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    # ------------------------------------------------------------------
    # Teleoperation
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        raw_action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": raw_action[motor] for motor in self._MOTOR_KEYS}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def enable_torque(self) -> None:
        """Enable servo torque so the leader can be commanded via ``write_goal_positions``."""
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        """Disable servo torque so the operator can move the leader freely."""
        self.bus.disable_torque()

    def write_goal_positions(self, positions: dict[str, float]) -> None:
        """Command the leader servos to move to the given positions.

        Accepts either motor names (``joint_1``) or dataset-style keys (``joint_1.pos``).
        Values for unknown motors are silently dropped.
        """
        goal: dict[str, float] = {}
        for key, value in positions.items():
            motor_name = key.removesuffix(".pos") if key.endswith(".pos") else key
            if motor_name in self.bus.motors:
                goal[motor_name] = value
        if goal:
            self.bus.sync_write("Goal_Position", goal)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Drive the leader servos toward the given positions.

        Expects the action/feedback key space (``joint_X.pos``, ``gripper.pos``);
        unknown keys are dropped by ``write_goal_positions``.  Only moves the arm
        while torque is enabled — callers (e.g. ``teleop_smooth_move_to``) must
        call ``enable_torque()`` first.
        """
        self.write_goal_positions(feedback)
