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

"""Thread-safe robot wrapper for concurrent observation/action access."""

from __future__ import annotations

from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    """Lock-protected wrapper around a :class:`Robot` for use with background threads.

    When RTC inference runs in a background thread while the main loop
    executes actions, both threads may access the robot concurrently.
    This wrapper serialises ``get_observation`` and ``send_action`` calls.

    Read-only properties are proxied without the lock since they don't
    mutate hardware state.
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            return self._robot.send_action(action)

    # -- Read-only proxies (no lock needed) -----------------------------------

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def joint_limits_deg(self) -> tuple[list[float], list[float]] | None:
        """Forward the wrapped robot's joint limits property (degrees)."""
        return self._robot.joint_limits_deg

    @property
    def make_ee_chunk_smoother(self):
        """EE chunk-smoother factory of the wrapped robot, or None.

        Duck-typed capability (e.g. ``PiperEE``): the QP inference engines use
        it to smooth Cartesian action chunks through joint space.  The factory
        is pure construction — no hardware I/O — so no lock is needed.
        """
        return getattr(self._robot, "make_ee_chunk_smoother", None)

    def urdf_q_from_observation(self, obs: dict[str, Any]) -> Any | None:
        """Map an observation dict to URDF joint radians (None if unsupported).

        Pure computation on an already-captured observation — no lock needed.
        """
        fn = getattr(self._robot, "urdf_q_from_observation", None)
        return None if fn is None else fn(obs)

    @property
    def ee_pose_from_observation(self):
        """FK (obs → ``ee.*`` pose dict) of the wrapped robot, or None.

        Duck-typed capability (e.g. ``PiperFull``): the QP inference engines
        use it to feed EE-state policies on joint-space robots.  Pure
        computation on an already-captured observation — no lock needed.
        """
        return getattr(self._robot, "ee_pose_from_observation", None)

    @property
    def ee_anchor_q_from_observation(self):
        """Joint anchor in the robot's EE-frame convention, or None.

        When a robot's EE pipeline works in a frame other than the true URDF
        one (e.g. ``PiperFull``'s signed frame), the qp engines must warm-start
        the chunk IK from this anchor instead of ``urdf_q_from_observation``.
        Pure computation — no lock needed.
        """
        return getattr(self._robot, "ee_anchor_q_from_observation", None)

    @property
    def inner(self) -> Robot:
        """Access the underlying robot (e.g. for connect/disconnect)."""
        return self._robot
