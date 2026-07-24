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

"""OSQP-backed chunk smoother — one independent QP per joint.

Cost on each joint trajectory ``y \\in R^T``:

.. math::

    \\min_y \\; \\Vert y - r \\Vert^2
        + \\lambda_a \\Vert \\Delta^2 y \\Vert^2
        + \\lambda_j \\Vert \\Delta^3 y \\Vert^2

subject to:

* per-tick velocity cap ``|Delta y| <= v_max / rate_hz``
* range bounds ``q_min <= y <= q_max`` (optional — omitted if no limits passed)
* anchor equality ``y[anchor_idx] = q_anchor`` (optional — omitted if no anchor passed)
* optional velocity equality ``y[anchor_idx] - y[anchor_idx - 1] = v_anchor``

The reference ``r`` is the (post-processor) action chunk in degrees.  The
solver problem geometry is built lazily on the first ``solve`` call (once
``T`` is known) and re-used afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import osqp
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    from lerobot.rollout.inference.factory import QPRTCInferenceConfig

logger = logging.getLogger(__name__)

# OSQP renamed solver settings in the 1.0 rewrite (polish->polishing,
# warm_start->warm_starting). Detect the installed version so the smoother runs
# on the Isaac sim-eval env (osqp 0.6.x) as well as modern training envs (>=1.0).
try:
    _OSQP_MAJOR = int(osqp.__version__.split(".")[0])
except Exception:  # noqa: BLE001
    _OSQP_MAJOR = 1
_OSQP_POLISH_KW = "polishing" if _OSQP_MAJOR >= 1 else "polish"
_OSQP_WARMSTART_KW = "warm_starting" if _OSQP_MAJOR >= 1 else "warm_start"


@dataclass(frozen=True)
class _PerJointGeometry:
    """Cached OSQP problem geometry that depends on ``T`` and the anchor index.

    Rebuilt only when ``T`` changes or when the anchor tick moves.
    """

    P: sp.csc_matrix
    A: sp.csc_matrix
    rate_step_deg: float
    anchor_idx: int


class QPSmoother:
    """Smooth a chunk of policy actions joint-wise via OSQP.

    Parameters
    ----------
    cfg : QPRTCInferenceConfig
        Holds smoother hyper-parameters (``v_max_deg_s``, ``lambda_a``,
        ``lambda_j``, ``rate_hz``, ``joint_keys``, ``gripper_key``,
        ``action_unit``).
    ordered_action_keys : list[str]
        Canonical order of action features used by the rollout.  Joint and
        gripper indices are resolved by name lookup in this list, never
        positionally.

    Notes
    -----
    Each joint is treated independently — this matches the offline prototype
    used for the smoothing parameter sweep (joint-wise QP) and keeps each per-joint problem tiny (T variables) so OSQP warm-starts
    cheaply.  The gripper column is passed through unchanged.
    """

    def __init__(
        self,
        cfg: QPRTCInferenceConfig,
        ordered_action_keys: list[str],
    ) -> None:
        self._cfg = cfg

        # --- Resolve indices by NAME (fail loudly on mismatch) ---------------
        missing = [k for k in cfg.joint_keys if k not in ordered_action_keys]
        if missing:
            raise ValueError(
                f"QPSmoother: joint keys not found in ordered_action_keys "
                f"(missing={missing!r}, available={list(ordered_action_keys)!r})"
            )
        self._joint_indices: list[int] = [ordered_action_keys.index(k) for k in cfg.joint_keys]

        if cfg.gripper_key is not None and cfg.gripper_key not in ordered_action_keys:
            raise ValueError(
                f"QPSmoother: gripper_key '{cfg.gripper_key}' not found in "
                f"ordered_action_keys {list(ordered_action_keys)!r}"
            )
        self._gripper_index: int | None = (
            ordered_action_keys.index(cfg.gripper_key) if cfg.gripper_key is not None else None
        )

        self._n_joints = len(self._joint_indices)
        self._rate_hz = float(cfg.rate_hz)
        self._v_max_deg_s = float(cfg.v_max_deg_s)
        self._lambda_a = float(cfg.lambda_a)
        self._lambda_j = float(cfg.lambda_j)
        self._strict_anchor = bool(cfg.strict_anchor)

        if cfg.action_unit not in ("pct", "deg", "rad"):
            raise ValueError(f"QPSmoother: action_unit must be 'pct'|'deg'|'rad', got {cfg.action_unit!r}")
        self._action_unit = cfg.action_unit

        # Lazy-built OSQP geometry, keyed by (T, anchor_idx).
        # We keep one solver per geometry — OSQP wants P/A fixed at setup time.
        self._cache: dict[tuple[int, int], tuple[_PerJointGeometry, list[osqp.OSQP]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        processed_chunk: torch.Tensor,
        joint_limits_deg: np.ndarray | None,
        q_anchor_deg: np.ndarray | None,
        delay: int = 0,
        *,
        q_anchor_velocity_deg: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Smooth a chunk in-unit and return a new tensor of the same shape.

        Parameters
        ----------
        processed_chunk : Tensor[T, A]
            Action chunk in ``cfg.action_unit`` (post-postprocessor).
        joint_limits_deg : np.ndarray[N, 2] | None
            Per-joint ``(lower, upper)`` in degrees.  If None, no range
            bounds are enforced.
        q_anchor_deg : np.ndarray[N] | None
            Position anchor in degrees.  In delay-aware mode this is applied
            at ``y[delay]`` (the first tick the robot will actually execute).
            If None, the first executed sample is free (loose start).  In
            strict_anchor mode, None triggers ``RuntimeError``.
        delay : int
            Number of leading ticks the consumer will drop.  Used in
            delay-aware mode: when ``delay > 0`` the smoother anchors at
            ``y[delay]`` (the first tick the robot will actually execute)
            and leaves the leading ``delay`` ticks free.
        q_anchor_velocity_deg : np.ndarray[N] | None
            Optional per-tick velocity anchor in degrees/tick.  When provided
            and ``delay > 0``, the solver enforces
            ``y[delay] - y[delay - 1]`` to match it.  This keeps chunk seams
            C1-continuous for RTC, where leading delayed ticks are discarded.

        Returns
        -------
        Tensor[T, A]
            Same shape/dtype/device as ``processed_chunk``.  Joint columns
            are smoothed; the gripper column (if configured) is copied
            through unchanged.
        """
        if processed_chunk.ndim != 2:
            raise ValueError(
                f"QPSmoother.solve expects 2D [T, A] tensor, got shape={tuple(processed_chunk.shape)}"
            )

        T = int(processed_chunk.shape[0])
        if T < 3:
            return processed_chunk  # too short to enforce 2nd/3rd-order diffs

        if self._action_unit == "pct" and joint_limits_deg is None:
            raise RuntimeError(
                "QPSmoother: action_unit='pct' requires joint_limits_deg "
                "(provide via QPRTCInferenceConfig.joint_limits_deg or expose "
                "Robot.joint_limits_deg)."
            )

        # --- Anchor strict mode ---------------------------------------------
        if self._strict_anchor and q_anchor_deg is None:
            raise RuntimeError(
                "QPSmoother: strict_anchor=True but no q_anchor provided. "
                "Notify the engine via notify_last_commanded_action or supply joint state in obs."
            )

        # Track original dtype/device to restore at the end.
        out = processed_chunk.detach().clone()
        joints_unit = out[:, self._joint_indices].cpu().numpy().astype(np.float64)

        # --- Convert to degrees for the QP ---------------------------------
        joints_deg = self._to_deg(joints_unit, joint_limits_deg)

        # --- Solve per joint ------------------------------------------------
        rate_step = self._v_max_deg_s / self._rate_hz
        anchor_idx = max(0, min(int(delay), T - 1))  # anchor at first executed tick
        geom, solvers = self._get_geometry(T, rate_step, anchor_idx)

        smoothed_deg = np.empty_like(joints_deg)
        for j in range(self._n_joints):
            ref = joints_deg[:, j]
            q_anchor_j = float(q_anchor_deg[j]) if q_anchor_deg is not None else None
            q_velocity_j = (
                float(q_anchor_velocity_deg[j])
                if q_anchor_velocity_deg is not None and anchor_idx > 0
                else None
            )
            if joint_limits_deg is None:
                lo, hi = -1e6, 1e6
            else:
                lo, hi = float(joint_limits_deg[j, 0]), float(joint_limits_deg[j, 1])
            smoothed_deg[:, j] = self._solve_one(
                T=T,
                j=j,
                ref=ref,
                rate_step=rate_step,
                q_anchor=q_anchor_j,
                q_anchor_velocity=q_velocity_j,
                lo=lo,
                hi=hi,
                geom=geom,
                solvers=solvers,
            )

        # --- Convert back to the input unit --------------------------------
        smoothed_unit = self._from_deg(smoothed_deg, joint_limits_deg)

        out[:, self._joint_indices] = torch.from_numpy(smoothed_unit).to(dtype=out.dtype, device=out.device)
        # Gripper passthrough — already in `out` (we only overwrote joint cols).
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_geometry(
        self,
        T: int,
        rate_step_deg: float,
        anchor_idx: int,
    ) -> tuple[_PerJointGeometry, list[osqp.OSQP]]:
        """Build (or fetch from cache) the joint-wise QP geometry + solver list."""
        key = (T, anchor_idx)
        cached = self._cache.get(key)
        if cached is not None and cached[0].rate_step_deg == rate_step_deg:
            return cached

        I = sp.eye(T, format="csc")
        # 2nd-order diff: y[t] - 2 y[t+1] + y[t+2], rows 0..T-3
        D2 = sp.diags([1.0, -2.0, 1.0], offsets=[0, 1, 2], shape=(T - 2, T), format="csc")
        # 3rd-order diff: -y[t] + 3 y[t+1] - 3 y[t+2] + y[t+3], rows 0..T-4
        D3 = sp.diags([-1.0, 3.0, -3.0, 1.0], offsets=[0, 1, 2, 3], shape=(T - 3, T), format="csc")
        # 1st-order diff: y[t+1] - y[t], rows 0..T-2 (rate bound rows)
        D1 = sp.diags([-1.0, 1.0], offsets=[0, 1], shape=(T - 1, T), format="csc")

        P = 2.0 * (I + self._lambda_a * (D2.T @ D2) + self._lambda_j * (D3.T @ D3))
        P_upper = sp.triu(P, format="csc")

        # Anchor row: a 1×T row vector with a single 1.0 at column `anchor_idx`.
        e_anchor = sp.csc_matrix(([1.0], ([0], [int(anchor_idx)])), shape=(1, T))
        # Optional velocity-anchor row. Bounds are left free unless the caller
        # supplies q_anchor_velocity; for anchor_idx=0 the row is all zeros.
        if anchor_idx > 0:
            e_velocity = sp.csc_matrix(
                ([-1.0, 1.0], ([0, 0], [int(anchor_idx) - 1, int(anchor_idx)])),
                shape=(1, T),
            )
        else:
            e_velocity = sp.csc_matrix((1, T))
        A = sp.vstack([D1, I, e_anchor, e_velocity], format="csc")

        geom = _PerJointGeometry(P=P_upper, A=A, rate_step_deg=rate_step_deg, anchor_idx=anchor_idx)
        solvers: list[osqp.OSQP] = [None] * self._n_joints  # type: ignore[list-item]
        self._cache[key] = (geom, solvers)
        return geom, solvers

    def _solve_one(
        self,
        *,
        T: int,
        j: int,
        ref: np.ndarray,
        rate_step: float,
        q_anchor: float | None,
        q_anchor_velocity: float | None,
        lo: float,
        hi: float,
        geom: _PerJointGeometry,
        solvers: list[osqp.OSQP],
    ) -> np.ndarray:
        """Solve the per-joint QP. Reuses the warm-started solver."""
        # A has (T-1) rate rows + T range rows + 1 anchor row + 1 velocity-anchor row.
        m = 2 * T + 1
        l = np.empty(m)
        u = np.empty(m)
        # rate rows: indices [0, T-1)
        l[: T - 1] = -rate_step
        u[: T - 1] = rate_step
        # range rows: indices [T-1, 2T-1)
        l[T - 1 : 2 * T - 1] = lo
        u[T - 1 : 2 * T - 1] = hi
        # anchor row: index 2T-1
        if q_anchor is None:
            l[2 * T - 1], u[2 * T - 1] = -np.inf, np.inf
        else:
            l[2 * T - 1], u[2 * T - 1] = q_anchor, q_anchor
        # velocity-anchor row: index 2T
        if q_anchor_velocity is None:
            l[2 * T], u[2 * T] = -np.inf, np.inf
        else:
            # The same finite-difference row is also rate-bounded. Clip tiny
            # external overshoots to keep the equality feasible.
            v = float(np.clip(q_anchor_velocity, -rate_step, rate_step))
            l[2 * T], u[2 * T] = v, v

        q = -2.0 * ref

        solver = solvers[j]
        if solver is None:
            solver = osqp.OSQP()
            solver.setup(
                P=geom.P,
                q=q,
                A=geom.A,
                l=l,
                u=u,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                **{_OSQP_POLISH_KW: True, _OSQP_WARMSTART_KW: True},
            )
            solvers[j] = solver
        else:
            solver.update(q=q, l=l, u=u)

        result = solver.solve()
        if result.info.status_val not in (1, 2):  # SOLVED, SOLVED_INACCURATE
            logger.warning(
                "QPSmoother joint %d: OSQP status=%s, falling back to reference.",
                j,
                result.info.status,
            )
            return ref
        return np.asarray(result.x, dtype=np.float64)

    # --- Unit conversions ----------------------------------------------------

    def _to_deg(self, x: np.ndarray, limits: np.ndarray | None) -> np.ndarray:
        if self._action_unit == "deg":
            return x
        if self._action_unit == "rad":
            return np.degrees(x)
        # pct ∈ [-100, 100] → deg = lo + (pct + 100) / 200 * (hi - lo)
        lo = limits[:, 0][None, :]
        hi = limits[:, 1][None, :]
        return lo + (x + 100.0) / 200.0 * (hi - lo)

    def _from_deg(self, x: np.ndarray, limits: np.ndarray | None) -> np.ndarray:
        if self._action_unit == "deg":
            return x
        if self._action_unit == "rad":
            return np.radians(x)
        lo = limits[:, 0][None, :]
        hi = limits[:, 1][None, :]
        # pct = (deg - lo) / (hi - lo) * 200 - 100
        span = hi - lo
        # Guard zero-span joints (should not happen for real hardware).
        span = np.where(span <= 0, 1.0, span)
        pct = (x - lo) / span * 200.0 - 100.0
        return np.clip(pct, -100.0, 100.0)
