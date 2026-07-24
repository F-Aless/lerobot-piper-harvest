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

"""EE action projection for the Piper.

A policy emitting raw Cartesian targets must not be trusted to respect the
arm's local kinematic constraints — especially the yaw/position coupling near
the ill-conditioned home configuration.  The projector turns a raw target
into a kinematically acceptable one *before* it reaches the robot:

1. clamp the translation step (per-tick linear velocity cap);
2. clamp the rotation step with an SO(3) geodesic metric — never a naive
   Euler-angle difference;
3. warm-started IK with soft/free yaw (the solver has its own
   condition-number and convergence fallbacks);
4. if the IK still fails, scale the target toward the current pose and retry
   yaw-free;
5. if that fails too, fall back to the last valid projected command (or hold
   the current pose when none exists yet).

The result carries the projected pose, the IK joint solution, and a merged
diagnostics dict so callers can log raw-vs-projected targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .piper_ee_ik import PiperEEKinematics


@dataclass
class EEProjectionResult:
    """Outcome of one projection step."""

    pos: np.ndarray  # projected target position (3,)
    R: np.ndarray  # projected target rotation (3, 3)
    q: np.ndarray  # IK joint solution for the projected target (nq,)
    info: dict[str, Any] = field(default_factory=dict)


class EEActionProjector:
    """Project raw EE targets into kinematically acceptable ones.

    Parameters
    ----------
    kinematics : PiperEEKinematics
        FK/IK backend (also provides the SO(3) log/exp used for clamping).
    max_dpos_per_step : float
        Translation clamp per step, metres.
    max_drot_per_step : float
        Rotation clamp per step, radians (geodesic angle).
    scale_retry_factor : float
        On IK failure, the (already clamped) target is pulled toward the
        current pose by this factor and re-solved yaw-free.
    solve_kwargs : dict
        Keyword arguments forwarded to :meth:`PiperEEKinematics.solve`
        (yaw_mode, tolerances, damping, max_joint_step, ...).
    """

    def __init__(
        self,
        kinematics: PiperEEKinematics,
        *,
        max_dpos_per_step: float,
        max_drot_per_step: float,
        scale_retry_factor: float = 0.5,
        solve_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._kin = kinematics
        self._max_dpos = float(max_dpos_per_step)
        self._max_drot = float(max_drot_per_step)
        self._scale_retry = float(scale_retry_factor)
        self._solve_kwargs = dict(solve_kwargs or {})

        self._last_valid: EEProjectionResult | None = None

    def reset(self) -> None:
        """Clear the last-valid-command fallback (call between episodes)."""
        self._last_valid = None

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        *,
        current_pos: np.ndarray,
        current_R: np.ndarray,
        q_seed: np.ndarray,
    ) -> EEProjectionResult:
        """Run the clamp → IK → scale-retry → last-valid ladder."""
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
        current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
        current_R = np.asarray(current_R, dtype=np.float64).reshape(3, 3)

        # --- 1+2. per-step clamps -----------------------------------------
        proj_pos, clamped_pos = self._clamp_translation(current_pos, target_pos)
        proj_R, clamped_rot = self._clamp_rotation(current_R, target_R)

        # --- 3. warm-started IK (soft yaw by default) -----------------------
        q, ik_info = self._kin.solve(proj_pos, proj_R, q_seed, **self._solve_kwargs)

        scaled = False
        used_last_valid = False
        held_pose = False

        # --- 4. scale toward current + yaw-free retry -----------------------
        if not ik_info["converged"]:
            scaled = True
            s = self._scale_retry
            proj_pos = current_pos + s * (proj_pos - current_pos)
            omega = self._kin.rotation_log(proj_R @ current_R.T)
            proj_R = self._kin.rotation_exp(s * omega) @ current_R
            retry_kwargs = {**self._solve_kwargs, "yaw_mode": "free", "free_yaw": None}
            q, ik_info = self._kin.solve(proj_pos, proj_R, q_seed, **retry_kwargs)

        # --- 5. last-valid fallback -----------------------------------------
        if not ik_info["converged"]:
            used_last_valid = True
            if self._last_valid is not None:
                proj_pos = self._last_valid.pos.copy()
                proj_R = self._last_valid.R.copy()
                q = self._last_valid.q.copy()
            else:
                # No history yet: hold the current pose.
                held_pose = True
                proj_pos = current_pos.copy()
                proj_R = current_R.copy()
                q = np.asarray(q_seed, dtype=np.float64).copy()

        result = EEProjectionResult(
            pos=proj_pos,
            R=proj_R,
            q=q,
            info={
                **ik_info,
                "clamped_pos": clamped_pos,
                "clamped_rot": clamped_rot,
                "scaled": scaled,
                "used_last_valid": used_last_valid,
                "held_pose": held_pose,
                "raw_dpos": float(np.linalg.norm(target_pos - current_pos)),
                "raw_drot": float(np.linalg.norm(self._kin.rotation_log(target_R @ current_R.T))),
            },
        )

        if not used_last_valid:
            self._last_valid = EEProjectionResult(pos=proj_pos.copy(), R=proj_R.copy(), q=q.copy(), info={})
        return result

    # ------------------------------------------------------------------
    # Clamps
    # ------------------------------------------------------------------

    def _clamp_translation(self, current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        delta = target - current
        norm = float(np.linalg.norm(delta))
        if norm <= self._max_dpos or norm == 0.0:
            return target, False
        return current + delta * (self._max_dpos / norm), True

    def _clamp_rotation(self, current_R: np.ndarray, target_R: np.ndarray) -> tuple[np.ndarray, bool]:
        omega = self._kin.rotation_log(target_R @ current_R.T)
        angle = float(np.linalg.norm(omega))
        if angle <= self._max_drot or angle == 0.0:
            return target_R, False
        return self._kin.rotation_exp(omega * (self._max_drot / angle)) @ current_R, True
