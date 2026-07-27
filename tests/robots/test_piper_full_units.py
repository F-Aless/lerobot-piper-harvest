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

"""Hardware-free tests for PiperFull's joint-unit handling (pct | deg | rad).

A fake SDK stands in for the CAN interface, so these validate the pure
conversion layer: config resolution/validation, observation → action
round-trips in every unit, and the safety clamps of the raw-angle units.
"""

import math

import numpy as np
import pytest

from lerobot.robots.piper_full import PiperFull, PiperFullConfig

JOINT_KEYS = [f"joint_{i + 1}.pos" for i in range(6)]

# Plausible HW-frame limits (deg), same layout as PiperFullSDK.joint_limits_deg.
LIM_MIN = [-150.0, 0.0, -170.0, -100.0, -70.0, -120.0]
LIM_MAX = [150.0, 180.0, 0.0, 100.0, 70.0, 120.0]


class FakeSDK:
    """Records the last hardware command instead of talking to the CAN bus."""

    is_connected = True
    joint_limits_deg = (LIM_MIN, LIM_MAX)

    def __init__(self, joint_deg=None, gripper_mm=35.0):
        self.reported_joint_deg = joint_deg or [10.0, 45.0, -30.0, 5.0, -20.0, 60.0]
        self.reported_gripper_mm = gripper_mm
        self.last_joints_deg = None
        self.last_gripper_mm = None

    def get_joint_angles_deg(self):
        return list(self.reported_joint_deg)

    def get_gripper_width_mm(self):
        return self.reported_gripper_mm

    def get_gripper_force_n(self):
        return 0.0

    def set_joint_positions_deg(self, joints_deg, gripper_mm=None, gripper_force=None):
        self.last_joints_deg = list(joints_deg)
        self.last_gripper_mm = gripper_mm


def _robot(**cfg_kwargs) -> tuple[PiperFull, FakeSDK]:
    robot = PiperFull(PiperFullConfig(**cfg_kwargs))
    sdk = FakeSDK()
    robot._sdk = sdk  # no cameras configured → is_connected is True
    return robot, sdk


class TestUnitConfig:
    def test_default_is_pct(self):
        assert PiperFullConfig().unit == "pct"

    def test_use_degrees_legacy_alias(self):
        assert PiperFullConfig(use_degrees=True).unit == "deg"

    def test_invalid_unit_rejected(self):
        with pytest.raises(ValueError, match="unit"):
            PiperFullConfig(unit="foo")

    def test_alias_conflict_rejected(self):
        with pytest.raises(ValueError, match="use_degrees"):
            PiperFullConfig(use_degrees=True, unit="rad")

    def test_action_angle_unit_exposed(self):
        robot, _ = _robot(unit="rad")
        assert robot.action_angle_unit == "rad"


class TestUnitRoundTrip:
    """Sending back the observed pose must command the pose the SDK reported."""

    @pytest.mark.parametrize("unit", ["pct", "deg", "rad"])
    def test_obs_to_action_round_trip(self, unit):
        robot, sdk = _robot(unit=unit)
        obs = robot.get_observation()
        action = {k: obs[k] for k in [*JOINT_KEYS, "gripper.pos"]}
        robot.send_action(action)
        assert np.allclose(sdk.last_joints_deg, sdk.reported_joint_deg, atol=1e-9)
        assert sdk.last_gripper_mm == pytest.approx(sdk.reported_gripper_mm, abs=1e-9)

    def test_deg_and_rad_observations_are_raw_angles(self):
        """"deg"/"rad" are the TRUE hw/URDF frame: no ``dataset_joint_signs``
        flip. Only "pct" (and the EE-policy bridge) use the mirrored dataset
        frame — see ``config_piper_full.py``."""
        robot_deg, sdk = _robot(unit="deg")
        obs_deg = robot_deg.get_observation()
        for i, key in enumerate(JOINT_KEYS):
            assert obs_deg[key] == pytest.approx(sdk.reported_joint_deg[i])
        assert obs_deg["gripper.pos"] == pytest.approx(sdk.reported_gripper_mm)

        robot_rad, sdk_rad = _robot(unit="rad")
        obs_rad = robot_rad.get_observation()
        for i, key in enumerate(JOINT_KEYS):
            assert obs_rad[key] == pytest.approx(math.radians(sdk_rad.reported_joint_deg[i]))


class TestRawUnitSafetyClamps:
    def test_gripper_mm_clamped(self):
        robot, sdk = _robot(unit="rad")
        action = dict.fromkeys(JOINT_KEYS, 0.0)
        robot.send_action({**action, "gripper.pos": 1000.0})
        assert sdk.last_gripper_mm == pytest.approx(robot.config.gripper_max_mm)
        robot.send_action({**action, "gripper.pos": -5.0})
        assert sdk.last_gripper_mm == pytest.approx(robot.config.gripper_min_mm)

    def test_joints_clamped_to_sdk_limits(self):
        robot, sdk = _robot(unit="deg")
        action = dict.fromkeys(JOINT_KEYS, 0.0) | {"gripper.pos": 10.0}
        robot.send_action({**action, "joint_1.pos": 1000.0})
        for hw, lo, hi in zip(sdk.last_joints_deg, LIM_MIN, LIM_MAX, strict=True):
            assert lo <= hw <= hi

    def test_pct_input_clamped(self):
        robot, sdk = _robot(unit="pct")
        action = dict.fromkeys(JOINT_KEYS, 0.0) | {"gripper.pos": 50.0}
        robot.send_action({**action, "joint_2.pos": 250.0})
        # pct is clamped to +100, which maps exactly onto the oriented limit.
        for hw, lo, hi in zip(sdk.last_joints_deg, LIM_MIN, LIM_MAX, strict=True):
            assert lo - 1e-9 <= hw <= hi + 1e-9


class TestEEBridgeUnitInvariance:
    """The EE-policy bridge (``ee_anchor_q_from_observation`` /
    ``_q_signed_rad_to_action``) always works in the mirrored dataset frame
    (``dataset_joint_signs`` applied), regardless of ``config.unit`` — only
    the plain ``get_observation``/``send_action`` numeric format changes with
    ``unit``. A regression here would silently send an *_ee-trained policy's
    actions to the wrong (unmirrored) joints whenever unit != "pct"."""

    def test_anchor_q_same_across_units(self):
        signs = np.array(PiperFullConfig().dataset_joint_signs, dtype=np.float64)
        expected_rad = np.deg2rad(np.array(FakeSDK().reported_joint_deg) * signs)

        for unit in ("pct", "deg", "rad"):
            robot, _ = _robot(unit=unit)
            obs = robot.get_observation()
            q_anchor = robot.ee_anchor_q_from_observation(obs)
            assert q_anchor == pytest.approx(expected_rad, abs=1e-6), f"unit={unit}"

    def test_signed_rad_to_action_round_trips_to_same_hw_command(self):
        # Same mirrored-frame anchor, mapped to each unit's action format and
        # sent — all three must land on the identical true hw-frame command.
        signs = np.array(PiperFullConfig().dataset_joint_signs, dtype=np.float64)
        q_anchor_rad = np.deg2rad(np.array(FakeSDK().reported_joint_deg) * signs)

        commands = {}
        for unit in ("pct", "deg", "rad"):
            robot, sdk = _robot(unit=unit)
            action_vals = robot._q_signed_rad_to_action(q_anchor_rad)
            action = {key: float(v) for key, v in zip(JOINT_KEYS, action_vals, strict=True)}
            robot.send_action({**action, "gripper.pos": sdk.reported_gripper_mm})
            commands[unit] = list(sdk.last_joints_deg)

        assert commands["pct"] == pytest.approx(commands["deg"], abs=1e-6)
        assert commands["pct"] == pytest.approx(commands["rad"], abs=1e-6)
        assert commands["pct"] == pytest.approx(FakeSDK().reported_joint_deg, abs=1e-6)
