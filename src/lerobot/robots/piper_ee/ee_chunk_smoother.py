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

"""Chunk-level EE→joint pipeline for the Piper.

For an EE policy the per-tick path (projector → IK → joint command) cannot
smooth across time: the joint limits live in joint space and the EE→joint map
is exactly where the wrist jumps appear.  This module implements the
chunk pipeline:

.. code-block:: text

    policy → raw EE chunk (policy action order)
           → EEActionProjector per step (state evolves along the chunk)
           → warm-started batch IK → joint chunk
           → joint-space QP smoother (the validated OSQP smoother)
           → output chunk in the *execution* action space

Two output spaces:

* ``output_space="joint"`` (the deployment path): the smoothed joint
  trajectory is emitted directly in the robot's joint action convention
  (``q_rad_to_action`` converts URDF radians → e.g. signed degrees or pct),
  so the robot receives plain SDK joint commands and the in-between
  interpolation downstream is linear on joints — no Euler representation
  issues can exist at all.

* ``output_space="ee"``: every output row is the FK of the QP-smoothed joint
  trajectory (joint-space smoothness baked into Cartesian targets, robot-side
  IK re-derives the same path).  RPY rows are kept representation-continuous
  (nearest Euler branch) so RTC blending / interpolation never see a ±π flip.

Engines obtain instances via the duck-typed robot factory
``make_ee_chunk_smoother`` (implemented by ``PiperFull`` for joint output and
``PiperEE`` for Cartesian output), which keeps the rollout layer free of
robot-specific imports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .ee_action_projector import EEActionProjector

if TYPE_CHECKING:
    from .piper_ee_ik import PiperEEKinematics

logger = logging.getLogger(__name__)

EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")


@dataclass
class _JointQPConfig:
    """Duck-typed config for the inner joint-space :class:`QPSmoother`."""

    v_max_deg_s: float
    lambda_a: float
    lambda_j: float
    rate_hz: float
    joint_keys: list[str]
    gripper_key: str | None = None
    action_unit: str = "deg"
    strict_anchor: bool = False


class EEChunkSmoother:
    """Project + batch-IK + joint-QP for EE action chunks.

    Parameters
    ----------
    kinematics : PiperEEKinematics
        FK/IK backend.
    projector : EEActionProjector
        Per-step projector (its per-step caps should be sized for the engine
        control rate ``rate_hz``).
    policy_action_keys : list[str]
        Column layout of the *incoming* chunk (the policy's action feature
        order).  Must contain the six ``ee.*`` pose keys.
    output_action_keys : list[str]
        Column layout of the *outgoing* chunk (the execution action space —
        the rollout's ``ordered_action_keys``).
    output_space : "joint" | "ee"
        See module docstring.
    joint_output_keys : list[str] | None
        For ``"joint"`` output: execution names of the six joints, in URDF
        joint order (``q[i]`` is written to column ``joint_output_keys[i]``).
    gripper_key : str | None
        Name of the gripper column, looked up in both key lists and passed
        through unchanged.
    q_rad_to_action : callable | None
        For ``"joint"`` output: maps a ``(T, 6)`` URDF-radian array to the
        robot's joint action unit.  Default: plain degrees.
    v_max_deg_s, lambda_a, lambda_j, rate_hz : float
        Joint-space QP parameters (same semantics as the joint pipeline).
    strict_anchor : bool
        If True, a missing joint anchor raises instead of warning.
    """

    def __init__(
        self,
        kinematics: PiperEEKinematics,
        projector: EEActionProjector,
        *,
        policy_action_keys: list[str],
        output_action_keys: list[str],
        output_space: str = "ee",
        joint_output_keys: list[str] | None = None,
        gripper_key: str | None = "gripper.pos",
        q_rad_to_action: Callable[[np.ndarray], np.ndarray] | None = None,
        v_max_deg_s: float,
        lambda_a: float,
        lambda_j: float,
        rate_hz: float,
        strict_anchor: bool = False,
    ) -> None:
        try:
            from lerobot.policies.chunk_smoother import QPSmoother
        except ImportError as e:
            raise ImportError(
                "EEChunkSmoother requires the 'chunk-smoother' extra. "
                "Install via: pip install 'lerobot[chunk-smoother]'"
            ) from e

        if output_space not in ("ee", "joint"):
            raise ValueError(f"output_space must be 'ee'|'joint', got {output_space!r}")

        missing = [k for k in EE_POSE_KEYS if k not in policy_action_keys]
        if missing:
            raise ValueError(
                f"EEChunkSmoother: EE pose keys missing from policy_action_keys: {missing!r} "
                f"(available: {list(policy_action_keys)!r})"
            )

        self._kin = kinematics
        self._projector = projector
        self._policy_action_keys = list(policy_action_keys)
        self._output_action_keys = list(output_action_keys)
        self._output_space = output_space
        self._strict_anchor = bool(strict_anchor)
        self._warned_no_anchor = False

        # --- input layout (policy chunk) -----------------------------------
        self._pos_idx = [policy_action_keys.index(k) for k in EE_POSE_KEYS[:3]]
        self._rpy_idx = [policy_action_keys.index(k) for k in EE_POSE_KEYS[3:]]
        self._in_gripper_idx = (
            policy_action_keys.index(gripper_key)
            if gripper_key is not None and gripper_key in policy_action_keys
            else None
        )

        # --- output layout ---------------------------------------------------
        nq = kinematics.nq
        if output_space == "joint":
            if joint_output_keys is None or len(joint_output_keys) != nq:
                raise ValueError(
                    f"EEChunkSmoother: 'joint' output needs joint_output_keys with {nq} names "
                    f"(got {joint_output_keys!r})"
                )
            missing_out = [k for k in joint_output_keys if k not in output_action_keys]
            if missing_out:
                raise ValueError(
                    f"EEChunkSmoother: joint_output_keys {missing_out!r} not in "
                    f"output_action_keys {list(output_action_keys)!r}"
                )
            self._out_joint_idx = [output_action_keys.index(k) for k in joint_output_keys]
            self._out_pos_idx = self._out_rpy_idx = None
        else:
            missing_out = [k for k in EE_POSE_KEYS if k not in output_action_keys]
            if missing_out:
                raise ValueError(
                    f"EEChunkSmoother: EE pose keys missing from output_action_keys: {missing_out!r}"
                )
            self._out_joint_idx = None
            self._out_pos_idx = [output_action_keys.index(k) for k in EE_POSE_KEYS[:3]]
            self._out_rpy_idx = [output_action_keys.index(k) for k in EE_POSE_KEYS[3:]]
        self._out_gripper_idx = (
            output_action_keys.index(gripper_key)
            if gripper_key is not None and gripper_key in output_action_keys
            else None
        )
        self._q_rad_to_action = q_rad_to_action if q_rad_to_action is not None else np.degrees

        joint_keys = [f"q{i + 1}" for i in range(nq)]
        self._qp = QPSmoother(
            _JointQPConfig(
                v_max_deg_s=float(v_max_deg_s),
                lambda_a=float(lambda_a),
                lambda_j=float(lambda_j),
                rate_hz=float(rate_hz),
                joint_keys=joint_keys,
                strict_anchor=False,  # anchor presence is handled here
            ),
            ordered_action_keys=joint_keys,
        )
        self._joint_limits_deg = np.column_stack([np.degrees(kinematics.q_min), np.degrees(kinematics.q_max)])

    def reset(self) -> None:
        """Clear cross-chunk projector state (call between episodes)."""
        self._projector.reset()

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        chunk: torch.Tensor,
        q0_rad: np.ndarray | None,
        delay: int = 0,
        *,
        q_cmd_rad: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert + smooth an EE chunk through joint space.

        Two distinct joint-space reference points feed this solve:

        * ``q0_rad`` — the *measurement-side* state: where the arm actually
          is.  It seeds the FK that initialises the projector state and
          warm-starts the first IK, so the EE→joint conversion evolves from
          reality.
        * ``q_cmd_rad`` — the *command-continuity* point: where the command
          stream is (the last action actually sent).  It pins the inner
          joint-QP anchor so consecutive chunks join without a step even when
          the arm lags the commands.  When None, the QP anchors on ``q0_rad``
          (legacy behaviour: one source for both roles).

        Parameters
        ----------
        chunk : Tensor[T, A_policy]
            EE action chunk in ``policy_action_keys`` order (positions in
            metres, orientations in radians).
        q0_rad : np.ndarray | None
            Measured joint state (URDF radians) at the first executed tick.
            When None the first IK seeds from zeros and — absent
            ``q_cmd_rad`` — the QP runs unanchored.
        delay : int
            Leading ticks the consumer will drop (RTC) — forwarded to the
            joint QP so it anchors at the first *executed* tick.
        q_cmd_rad : np.ndarray | None
            Optional command-continuity QP anchor (URDF radians).

        Returns
        -------
        (smoothed_chunk, diagnostics)
            ``smoothed_chunk`` has shape ``[T, len(output_action_keys)]``, in
            the configured output space; non-pose columns (gripper) pass
            through.
        """
        if chunk.ndim != 2:
            raise ValueError(f"EEChunkSmoother.solve expects [T, A] tensor, got {tuple(chunk.shape)}")
        if chunk.shape[1] != len(self._policy_action_keys):
            raise ValueError(
                f"EEChunkSmoother: chunk has {chunk.shape[1]} columns, policy_action_keys has "
                f"{len(self._policy_action_keys)} — the policy output layout does not match."
            )
        arr = chunk.detach().cpu().numpy().astype(np.float64)
        T = arr.shape[0]
        nq = self._kin.nq

        if q0_rad is None:
            if self._strict_anchor:
                raise RuntimeError(
                    "EEChunkSmoother: strict_anchor=True but no joint anchor available. "
                    "Ensure the observation exposes joint positions."
                )
            if not self._warned_no_anchor:
                logger.warning("EEChunkSmoother: no joint anchor — seeding IK from q=0 (loose start).")
                self._warned_no_anchor = True
            q_prev = np.zeros(nq, dtype=np.float64)
        else:
            q_prev = np.asarray(q0_rad, dtype=np.float64).reshape(nq).copy()
        # QP anchor: command continuity when available, measurement otherwise.
        if q_cmd_rad is not None:
            q_anchor_deg = np.degrees(np.asarray(q_cmd_rad, dtype=np.float64).reshape(nq))
        elif q0_rad is not None:
            q_anchor_deg = np.degrees(q_prev)
        else:
            q_anchor_deg = None

        cur_pos, cur_R, _ = self._kin.fk(q_prev)

        # --- project + batch warm-started IK --------------------------------
        q_traj = np.empty((T, nq), dtype=np.float64)
        n_fallback = n_scaled = n_capped = n_last_valid = 0
        max_pos_err = 0.0
        max_cond = 0.0
        max_null_space_target_err = 0.0
        for t in range(T):
            target_R = self._kin.rpy_to_matrix(*arr[t, self._rpy_idx])
            res = self._projector.project(
                arr[t, self._pos_idx],
                target_R,
                current_pos=cur_pos,
                current_R=cur_R,
                q_seed=q_prev,
            )
            q_traj[t] = res.q
            q_prev = res.q
            cur_pos, cur_R, _ = self._kin.fk(res.q)

            info = res.info
            n_fallback += bool(info.get("fallback_used"))
            n_scaled += bool(info.get("scaled"))
            n_capped += bool(info.get("capped"))
            n_last_valid += bool(info.get("used_last_valid"))
            max_pos_err = max(max_pos_err, float(info.get("pos_err", 0.0)))
            max_cond = max(max_cond, float(info.get("condition_number", 0.0)))
            max_null_space_target_err = max(
                max_null_space_target_err,
                float(info.get("null_space_target_err", 0.0)),
            )

        # --- joint-space QP ---------------------------------------------------
        q_raw_deg = np.degrees(q_traj)
        smoothed_deg = self._qp.solve(
            torch.from_numpy(q_raw_deg),
            joint_limits_deg=self._joint_limits_deg,
            q_anchor_deg=q_anchor_deg,
            delay=int(delay),
        )
        q_smooth = np.radians(smoothed_deg.numpy().astype(np.float64))

        # --- emit in the execution action space ------------------------------
        out = np.zeros((T, len(self._output_action_keys)), dtype=np.float64)
        if self._output_space == "joint":
            out[:, self._out_joint_idx] = self._q_rad_to_action(q_smooth)
        else:
            # FK back with representation-continuous RPY.
            ref_rpy = arr[0, self._rpy_idx]  # reference branch: the raw chunk's own first row
            for t in range(T):
                pos, R, _ = self._kin.fk(q_smooth[t])
                rpy = self._continuous_rpy(R, ref_rpy)
                out[t, self._out_pos_idx] = pos
                out[t, self._out_rpy_idx] = rpy
                ref_rpy = rpy
        if self._out_gripper_idx is not None and self._in_gripper_idx is not None:
            out[:, self._out_gripper_idx] = arr[:, self._in_gripper_idx]

        smoothed_chunk = torch.from_numpy(out).to(dtype=chunk.dtype, device=chunk.device)
        diagnostics = {
            "output_space": self._output_space,
            "q_raw_deg": q_raw_deg,
            "q_smooth_deg": np.degrees(q_smooth),
            "qp_anchor_deg": q_anchor_deg,
            "ee_projected_chunk": out if self._output_space == "ee" else None,
            "n_fallback": n_fallback,
            "n_scaled": n_scaled,
            "n_capped": n_capped,
            "n_last_valid": n_last_valid,
            "max_ik_pos_err": max_pos_err,
            "max_condition_number": max_cond,
            "max_null_space_target_err": max_null_space_target_err,
            "q_last_smoothed_rad": q_smooth[-1],
            "q_smooth_rad": q_smooth,
        }
        return smoothed_chunk, diagnostics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _continuous_rpy(self, R: np.ndarray, ref_rpy: np.ndarray) -> np.ndarray:
        """RPY representation of ``R`` closest to ``ref_rpy``.

        Considers both Euler decompositions of ``R`` — ``(r, p, y)`` and the
        equivalent ``(r+π, π−p, y+π)`` — and shifts each component by the
        nearest multiple of 2π, then keeps the candidate with the smallest
        max component distance to the reference.
        """
        principal = self._kin.matrix_to_rpy(R)
        alternate = np.array(
            [principal[0] + np.pi, np.pi - principal[1], principal[2] + np.pi], dtype=np.float64
        )
        best = None
        best_dist = np.inf
        for cand in (principal, alternate):
            shifted = ref_rpy + np.mod(cand - ref_rpy + np.pi, 2.0 * np.pi) - np.pi
            dist = float(np.max(np.abs(shifted - ref_rpy)))
            if dist < best_dist:
                best, best_dist = shifted, dist
        return best
