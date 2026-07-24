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

"""Thin wrapper around ``piper_sdk`` for the ``piper_full`` robot.

Mirrors the hardware-validated ``PiperFullSDKInterface`` from our previous
in-lab lerobot fork, with small adjustments to keep the same external API the rest of LeRobot
expects (``connect()`` / ``disconnect()`` / ``set_joint_positions_deg()`` etc).

All angular I/O on this interface is in **degrees** (joints) and **mm**
(gripper) to match LeRobot's robot-level convention; the unit conversions to
the SDK's native milli-degrees / 0.001-mm gripper units happen here.

Key conventions verified against the legacy interface:
  * ``MotionCtrl_2(0x01, 0x01, speed, 0x00)`` — ctrl=CAN, mode=J, **POSITION
    framing** (``is_mit_mode=0``).  The legacy stack ran 60 Hz teleop with this.
  * ``EnablePiper()`` is the high-level enable; it returns True/False so we
    don't need to poll ``GetArmEnableStatus``.
  * Joint limits come from ``GetAllMotorAngleLimitMaxSpd`` in deci-degrees
    (×10°) — falls back to hard-coded URDF defaults if the SDK call fails.
  * Gripper SDK unit: 0.001 mm per tick (1 mm = 1000 SDK ticks). NB: the
    legacy fork used 10000 ticks/mm — its "mm" were physically cm; the
    ``gripper_max_mm=7.0`` it carried is really the 70 mm stroke. The pct
    mapping (ticks per pct) is identical either way, so datasets/policies
    are unaffected by this relabelling.
  * Mode is set **once** at connect-time; we do NOT re-assert MotionCtrl_2 on
    every JointCtrl (the legacy doesn't and 60 Hz teleop works fine).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as exc:  # pragma: no cover
    C_PiperInterface_V2 = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


# MotionCtrl_2 framing: ``is_mit_mode=0x00`` = POSITION (the legacy default).
_POSITION_MODE = 0x00

# CAN channels that already had a connect() in this process — in-process
# reconnect is unsupported (see PiperFullSDK.connect / disconnect).
_CONNECTED_CHANNELS: set[str] = set()

# Fallback joint limits in degrees if ``GetAllMotorAngleLimitMaxSpd`` fails.
# These match the Piper URDF lower/upper position limits.
_DEFAULT_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-150.0, 150.0),  # joint_1
    (0.0, 180.0),  # joint_2
    (-170.0, 0.0),  # joint_3
    (-100.0, 100.0),  # joint_4
    (-70.0, 70.0),  # joint_5
    (-180.0, 180.0),  # joint_6
)


class PiperFullSDK:
    """Wraps :class:`piper_sdk.C_PiperInterface_V2`.

    External contract (unchanged from previous implementation):
      * ``connect()`` / ``disconnect(disable_motors=False)``
      * ``is_connected`` (property)
      * ``get_joint_angles_deg() -> list[float] | None``    (6 joints, deg)
      * ``get_gripper_width_mm() -> float | None``
      * ``get_gripper_force_n() -> float | None``           (low-level read; sign-flipped in robot)
      * ``get_motion_status() -> int | None``               (0 = idle)
      * ``get_end_pose() -> dict[str, float] | None``       (m + rad)
      * ``is_ok() -> bool``
      * ``set_joint_positions_deg(joints_deg, gripper_mm=..., gripper_force=...)``
      * ``go_home_slow(speed_pct=20, timeout_s=8.0)``
      * ``electronic_emergency_stop()``
      * ``reset()`` — no-op for backward compat
      * ``joint_limits_deg`` (property: ``(min_list, max_list)``)
    """

    NUM_JOINTS = 6

    def __init__(
        self,
        *,
        can_channel: str = "can0",
        can_interface: str = "socketcan",
        can_bitrate: int = 1_000_000,
        firmware: str = "v188",
        speed_percent: int = 100,
        enable_timeout_s: float = 5.0,
    ) -> None:
        # The legacy SDK ignores firmware / can_interface / can_bitrate.
        del can_interface, can_bitrate, firmware

        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. Install via:\n    pip install piper_sdk"
            ) from _IMPORT_ERROR

        self._can_channel = can_channel
        self._speed_percent = int(max(1, min(100, speed_percent)))
        self._enable_timeout_s = float(enable_timeout_s)
        self._piper: C_PiperInterface_V2 | None = None
        self._connected = False
        self._last_enable_check_s = 0.0
        self._last_enable_status: list[bool] | None = None

        self._joint_limits_deg_min = [lo for lo, _ in _DEFAULT_JOINT_LIMITS_DEG]
        self._joint_limits_deg_max = [hi for _, hi in _DEFAULT_JOINT_LIMITS_DEG]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return

        # Reconnecting within the same process is NOT supported: disconnect()
        # deliberately does not call DisconnectPort() (gs_usb fights repeated
        # open/close cycles), so the previous interface's CAN reader threads
        # and socket stay alive.  A second C_PiperInterface_V2 on the same
        # channel would race them — fail loudly instead.
        if self._can_channel in _CONNECTED_CHANNELS:
            raise RuntimeError(
                f"PiperFullSDK: '{self._can_channel}' was already connected in this process; "
                "in-process reconnect is not supported (the previous CAN reader threads stay "
                "alive by design). Restart the process to reconnect."
            )
        _CONNECTED_CHANNELS.add(self._can_channel)

        self._piper = C_PiperInterface_V2(self._can_channel)
        self._piper.ConnectPort()
        time.sleep(0.1)

        self._safe_resume()
        self._wait_for_enable()

        # Set motion mode J + POSITION framing + speed_percent. Set ONCE here;
        # the legacy stack does not re-assert this on every JointCtrl.
        self._piper.MotionCtrl_2(0x01, 0x01, self._speed_percent, _POSITION_MODE)

        # Read joint limits from the SDK; fall back to defaults on failure.
        self._read_joint_limits_from_sdk()

        self._connected = True

    def disconnect(self, *, disable_motors: bool = False) -> None:
        """Disconnect from the arm.

        No motion command is sent: in position mode a ``JointCtrl(0,...,0)``
        is a *home target*, not a latch cleanup — the arm would physically
        move on shutdown.  The last commanded target simply stays latched.
        Motors stay energised — the teleop / next session can take over
        immediately without a re-enable. Pass ``disable_motors=True`` to
        actually drop power on the joints.
        """
        if not self._connected:
            return
        if disable_motors and self._piper is not None:
            try:
                time.sleep(0.1)
                self._piper.DisableArm(7, 1)
            except Exception:
                logger.debug("DisableArm raised during disconnect", exc_info=True)
        # NB: the legacy interface does not call DisconnectPort either —
        # it relies on the process exiting to close the CAN socket. We do
        # the same here so multiple connect()/disconnect() cycles within a
        # single process don't fight the gs_usb driver.
        self._connected = False

    def electronic_emergency_stop(self) -> None:
        if self._piper is not None:
            try:
                self._piper.EmergencyStop(2)
            except Exception:
                logger.exception("EmergencyStop failed")

    def reset(self) -> None:
        # Kept for backward compat with the old wrapper.
        return None

    # ------------------------------------------------------------------
    # State readers (degrees / mm / N)
    # ------------------------------------------------------------------

    def get_joint_angles_deg(self) -> list[float] | None:
        if self._piper is None:
            return None
        try:
            msg = self._piper.GetArmJointMsgs()
        except Exception:
            return None
        if msg is None:
            return None
        js = msg.joint_state
        # joint_N is in milli-degrees in the SDK protocol.
        return [
            float(js.joint_1) / 1000.0,
            float(js.joint_2) / 1000.0,
            float(js.joint_3) / 1000.0,
            float(js.joint_4) / 1000.0,
            float(js.joint_5) / 1000.0,
            float(js.joint_6) / 1000.0,
        ]

    def get_end_pose(self) -> dict[str, float] | None:
        """End pose in metres + radians, read directly from the SDK."""
        if self._piper is None:
            return None
        try:
            msg = self._piper.GetArmEndPoseMsgs()
        except Exception:
            return None
        if msg is None:
            return None
        ep = msg.end_pose
        import math

        return {
            "x": float(ep.X_axis) / 1_000_000.0,
            "y": float(ep.Y_axis) / 1_000_000.0,
            "z": float(ep.Z_axis) / 1_000_000.0,
            "rx": math.radians(float(ep.RX_axis) / 1000.0),
            "ry": math.radians(float(ep.RY_axis) / 1000.0),
            "rz": math.radians(float(ep.RZ_axis) / 1000.0),
        }

    def get_gripper_width_mm(self) -> float | None:
        if self._piper is None:
            return None
        try:
            msg = self._piper.GetArmGripperMsgs()
        except Exception:
            return None
        if msg is None:
            return None
        # ``grippers_angle`` SDK unit: 0.001 mm per tick (1 mm = 1000 ticks).
        return float(msg.gripper_state.grippers_angle) / 1000.0

    def get_gripper_force_n(self) -> float | None:
        """Raw gripper effort in N (milli-Newtons in SDK). Sign may be inverted
        depending on direction of motion; callers may want to negate it."""
        if self._piper is None:
            return None
        try:
            msg = self._piper.GetArmGripperMsgs()
        except Exception:
            return None
        if msg is None:
            return None
        try:
            return float(msg.gripper_state.grippers_effort) / 1000.0
        except AttributeError:
            return 0.0

    def get_motion_status(self) -> int | None:
        if self._piper is None:
            return None
        try:
            msg = self._piper.GetArmStatus()
        except Exception:
            return None
        if msg is None:
            return None
        return int(getattr(msg.arm_status, "motion_status", -1))

    def is_ok(self) -> bool:
        if self._piper is None or not self._connected:
            return False
        try:
            return self.get_joint_angles_deg() is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Send commands
    # ------------------------------------------------------------------

    def set_joint_positions_deg(
        self,
        joints_deg: list[float],
        *,
        gripper_mm: float | None = None,
        gripper_force: float = 1.0,
    ) -> None:
        if self._piper is None:
            raise RuntimeError("PiperFullSDK.set_joint_positions_deg called before connect()")
        if len(joints_deg) != self.NUM_JOINTS:
            raise ValueError(f"Expected {self.NUM_JOINTS} joint commands, got {len(joints_deg)}")
        clamped = [
            max(self._joint_limits_deg_min[i], min(self._joint_limits_deg_max[i], float(joints_deg[i])))
            for i in range(self.NUM_JOINTS)
        ]
        j_mdeg = [int(round(d * 1000.0)) for d in clamped]

        self._ensure_enabled_for_command()

        # NB: deliberately NO MotionCtrl_2 here — the legacy doesn't refresh
        # it on every step, and adding the refresh just bloats the CAN frame
        # count (3-4 frames/tick → ENOBUFS at 60 Hz with txqueuelen=10).
        self._piper.JointCtrl(j_mdeg[0], j_mdeg[1], j_mdeg[2], j_mdeg[3], j_mdeg[4], j_mdeg[5])

        if gripper_mm is not None:
            # SDK gripper unit: 0.001 mm per tick (1 mm = 1000 ticks); force in mN.
            pos_sdk = int(round(float(gripper_mm) * 1000.0))
            force_sdk = int(round(float(gripper_force) * 1000.0))
            self._piper.GripperCtrl(pos_sdk, force_sdk, 0x01, 0)

    def go_home_slow(self, *, speed_pct: int = 20, timeout_s: float = 8.0) -> None:
        if self._piper is None:
            raise RuntimeError("PiperFullSDK.go_home_slow called before connect()")
        previous = self._speed_percent
        new_speed = int(max(1, min(100, speed_pct)))
        try:
            self._piper.MotionCtrl_2(0x01, 0x01, new_speed, _POSITION_MODE)
            time.sleep(0.05)
            self._piper.JointCtrl(0, 0, 0, 0, 0, 0)
            # Poll joint positions, not motion_status (which can falsely
            # report "FAILED" on transient overshoot).
            deadline = time.monotonic() + max(0.0, float(timeout_s))
            while time.monotonic() < deadline:
                joints = self.get_joint_angles_deg()
                if joints is not None and all(abs(d) < 2.0 for d in joints):
                    break
                time.sleep(0.1)
        finally:
            try:
                self._piper.MotionCtrl_2(0x01, 0x01, previous, _POSITION_MODE)
            except Exception:
                logger.debug("speed restore failed", exc_info=True)
            self._speed_percent = previous

    # ------------------------------------------------------------------
    # Limits / introspection
    # ------------------------------------------------------------------

    @property
    def joint_limits_deg(self) -> tuple[list[float], list[float]]:
        return list(self._joint_limits_deg_min), list(self._joint_limits_deg_max)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_resume(self) -> None:
        try:
            status = self._piper.GetArmStatus()
        except Exception:
            return
        if status is None:
            return
        arm_st = getattr(status, "arm_status", None)
        if arm_st is None:
            return
        motion = getattr(arm_st, "motion_status", 0)
        ctrl = getattr(arm_st, "ctrl_mode", 0)
        if motion not in (0, None) or ctrl == 2:  # 2 = TEACHING
            logger.warning("Piper not idle (motion=%s, ctrl=%s) — issuing e-stop", motion, ctrl)
            self.electronic_emergency_stop()
            time.sleep(0.2)

    def _wait_for_enable(self) -> None:
        """Enable the arm and wait until all joint drivers report enabled."""
        start = time.monotonic()
        while True:
            try:
                self._piper.EnableArm(7)
            except Exception:
                logger.debug("EnableArm failed", exc_info=True)

            enabled = self._read_enable_status()
            if enabled is not None and all(enabled):
                return

            if time.monotonic() - start > self._enable_timeout_s:
                raise TimeoutError(
                    f"Piper EnableArm timed out after {self._enable_timeout_s:.1f}s "
                    f"(last enable status: {enabled})"
                )
            time.sleep(0.05)

    def _read_enable_status(self) -> list[bool] | None:
        if self._piper is None:
            return None
        try:
            status = [bool(value) for value in self._piper.GetArmEnableStatus()]
        except Exception:
            return None
        if len(status) != self.NUM_JOINTS:
            return None
        self._last_enable_status = status
        self._last_enable_check_s = time.monotonic()
        return status

    def _ensure_enabled_for_command(self) -> None:
        # The SDK can keep streaming joint states while drivers are disabled.
        # Guard JointCtrl so data collection does not silently record commands
        # that the arm ignores.
        now = time.monotonic()
        if (
            self._last_enable_status is not None
            and all(self._last_enable_status)
            and now - self._last_enable_check_s < 0.5
        ):
            return

        enabled = self._read_enable_status()
        if enabled is not None and all(enabled):
            return

        logger.warning("Piper drivers are disabled before JointCtrl (%s); re-enabling", enabled)
        self._wait_for_enable()
        try:
            self._piper.MotionCtrl_2(0x01, 0x01, self._speed_percent, _POSITION_MODE)
        except Exception:
            logger.debug("MotionCtrl_2 after re-enable failed", exc_info=True)

    def _read_joint_limits_from_sdk(self) -> None:
        """Pull per-joint limits from ``GetAllMotorAngleLimitMaxSpd``.

        SDK returns deci-degrees (×10°). Falls back to defaults on failure.
        """
        try:
            msg = self._piper.GetAllMotorAngleLimitMaxSpd()
            if msg is None:
                return
            motors = msg.all_motor_angle_limit_max_spd.motor
            # motor[0] is unused (index 1..6 are the joints).
            mins: list[float] = []
            maxs: list[float] = []
            for i in range(1, self.NUM_JOINTS + 1):
                lo = float(motors[i].min_angle_limit) / 10.0
                hi = float(motors[i].max_angle_limit) / 10.0
                # The SDK/firmware read is occasionally garbage (a stale or
                # misframed CAN response yields e.g. min=-179.2, max=-409.6 with
                # min>max). Keep the URDF default for any joint whose read is
                # implausible, else one bad frame inverts a QP range bound
                # (lo>hi) and every chunk-smoother solve fails.
                if not (lo < hi and -360.0 <= lo <= 360.0 and -360.0 <= hi <= 360.0):
                    logger.warning(
                        "piper_full: implausible SDK joint-limit read for joint %d "
                        "(min=%.1f, max=%.1f deg); using default %s.",
                        i,
                        lo,
                        hi,
                        _DEFAULT_JOINT_LIMITS_DEG[i - 1],
                    )
                    lo, hi = _DEFAULT_JOINT_LIMITS_DEG[i - 1]
                mins.append(lo)
                maxs.append(hi)
            self._joint_limits_deg_min = mins
            self._joint_limits_deg_max = maxs
        except Exception:
            logger.debug("Could not read joint limits from SDK; using URDF defaults", exc_info=True)

    # Used by external scripts that want a raw handle to the underlying SDK.
    @property
    def inner(self) -> Any:
        return self._piper
