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

"""Action interpolation for smoother robot control.

Provides configurable Nx control rate by interpolating between consecutive actions.
Useful with RTC and action-chunking policies to reduce jerkiness.

Scalar action dimensions are interpolated linearly.  Orientation triples
(``*.roll``/``*.pitch``/``*.yaw`` groups, declared via ``rpy_triples``) are
interpolated with quaternion SLERP instead: linear interpolation of Euler
angles takes the long way around when an angle crosses the ±π branch cut and
distorts the rotation path near gimbal lock, which for EE action spaces turns
into wrist sweeps the policy never asked for.
"""

import math

import torch
from torch import Tensor


def rpy_triples_from_keys(keys: list[str]) -> list[tuple[int, int, int]]:
    """Detect ``<prefix>.roll/.pitch/.yaw`` index triples in an action key list.

    Returns one ``(roll_idx, pitch_idx, yaw_idx)`` tuple per prefix that has
    all three components (e.g. ``ee.`` → indices of ``ee.roll``, ``ee.pitch``,
    ``ee.yaw``).  Use the result as ``ActionInterpolator(rpy_triples=...)``.
    """
    triples = []
    for i, key in enumerate(keys):
        if not key.endswith(".roll"):
            continue
        prefix = key[: -len("roll")]
        try:
            triples.append((i, keys.index(prefix + "pitch"), keys.index(prefix + "yaw")))
        except ValueError:
            continue
    return triples


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)

    Args:
        multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, ...).
        rpy_triples: Optional ``(roll, pitch, yaw)`` index triples (see
            :func:`rpy_triples_from_keys`).  Those components are interpolated
            via quaternion SLERP; everything else stays linear.
    """

    def __init__(self, multiplier: int = 1, rpy_triples: list[tuple[int, int, int]] | None = None):
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self.rpy_triples = list(rpy_triples or [])
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier + 1):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                for triple in self.rpy_triples:
                    idx = list(triple)
                    interp[idx] = _slerp_rpy(self._prev[idx], action[idx], t)
                self._buffer.append(interp)
        else:
            # First step: no previous action yet, so run at base FPS without interpolation.
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        return action

    def get_control_interval(self, fps: float) -> float:
        """Get the control interval based on interpolation multiplier.

        Args:
            fps: Base frames per second.

        Returns:
            Control interval in seconds (divided by multiplier).
        """
        return 1.0 / (fps * self.multiplier)


# ---------------------------------------------------------------------------
# Quaternion SLERP on (roll, pitch, yaw) triples — ZYX intrinsic convention
# (R = Rz(yaw) · Ry(pitch) · Rx(roll)), matching the EE robots and the
# kinematics helpers.
# ---------------------------------------------------------------------------


def _slerp_rpy(rpy_a: Tensor, rpy_b: Tensor, t: float) -> Tensor:
    """Interpolate two RPY triples along the SO(3) geodesic."""
    qa = _rpy_to_quat(float(rpy_a[0]), float(rpy_a[1]), float(rpy_a[2]))
    qb = _rpy_to_quat(float(rpy_b[0]), float(rpy_b[1]), float(rpy_b[2]))
    q = _quat_slerp(qa, qb, t)
    return torch.tensor(_quat_to_rpy(q), dtype=rpy_a.dtype)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """RPY (ZYX intrinsic) → quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _quat_slerp(
    qa: tuple[float, float, float, float], qb: tuple[float, float, float, float], t: float
) -> tuple[float, float, float, float]:
    """Shortest-path spherical interpolation between two unit quaternions."""
    dot = sum(a * b for a, b in zip(qa, qb, strict=True))
    if dot < 0.0:  # take the short way around the 3-sphere
        qb = tuple(-b for b in qb)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:  # nearly parallel: lerp + renormalise is numerically safer
        q = tuple(a + t * (b - a) for a, b in zip(qa, qb, strict=True))
        norm = math.sqrt(sum(c * c for c in q))
        return tuple(c / norm for c in q)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return tuple(wa * a + wb * b for a, b in zip(qa, qb, strict=True))


def _quat_to_rpy(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) → RPY (ZYX intrinsic), principal branch."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw
