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

"""Pinocchio-backed FK + damped-least-squares IK for the Piper 6-DOF arm.

Pinocchio is imported lazily — call sites that don't need IK (e.g. importing
``lerobot.robots.piper_ee`` for draccus registration) won't pay the import
cost or fail without the conda-forge dependency.

Solver model:

  * FK via :func:`pinocchio.framesForwardKinematics` on a reduced model where
    the gripper joints (``joint7``/``joint8``) are locked.
  * Body Jacobian in ``LOCAL_WORLD_ALIGNED`` frame, so the top 3 rows are
    ``∂pos/∂q`` in world and the bottom 3 are ``∂ω_world/∂q``.
  * Rotation error as the SO(3) logarithm of ``R_target · R_current.T``.
  * Damped pseudoinverse ``dq = J.T · (J·J.T + λ²·I)⁻¹ · e`` with optional
    singularity-adaptive damping (Chiaverini-style: boost λ when the smallest
    singular value of the task Jacobian collapses).
  * Three yaw modes — the Piper is near a yaw/position-coupled singularity at
    home (pitch ≈ 85°, cond(J) ≈ 2.8e4), so demanding exact yaw produces
    ~80° wrist jumps for cm-scale position targets:

      - ``"full"``  — 6-D task, yaw tracked with full weight;
      - ``"soft"``  — 6-D task, yaw row down-weighted by ``yaw_weight``
        (recommended for EE policies: yaw is followed when cheap, released
        near the singularity);
      - ``"free"``  — 5-D task, yaw row dropped, null-space bias toward
        ``q_seed`` and, optionally, a configured neutral posture.

    The neutral-posture bias is deliberately a *soft preference*, not a joint
    limit.  Dropping yaw leaves one redundant direction: many joint
    configurations can realize the same position + roll/pitch.  The seed bias
    keeps consecutive ticks continuous, but it is local: if the seed has
    slowly drifted to joint_4=40° and joint_6=-40°, "stay near the seed" also
    means "stay drifted".  ``null_space_target`` adds a second null-space
    drive toward comfortable finite entries, typically joint_4=0 and
    joint_6=0, while NaN entries are ignored.  Because the drive is projected
    through ``N = I - J⁺J``, it only uses motion left over after the required
    position + roll/pitch task.  If the task genuinely needs those wrist
    angles, the solver may still go there; if they are only yaw-free drift,
    the solver drains them back toward neutral.  ``max_joint_step`` still
    caps the returned motion, so the recovery rate respects the same per-tick
    speed limit as normal IK motion.

  * Automatic fallback to ``"free"``: upfront when ``"full"`` is requested at
    an ill-conditioned configuration, and post-hoc for ``"full"``/``"soft"``
    when the solve does not converge.
  * Per-iter step cap, optional direction-preserving per-joint total delta
    cap, and a final URDF joint-limit clamp.

``solve`` returns ``(q, info)`` where ``info`` carries full diagnostics
(``converged``, ``pos_err``, ``rot_err``, ``yaw_err``,
``condition_number``, ``max_abs_delta``, ``capped``, ``fallback_used``,
``yaw_mode_requested``/``yaw_mode_used``, ``iters``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # only for type hints, never imported at module-import time
    import pinocchio as pin

logger = logging.getLogger(__name__)

YAW_MODES = ("full", "soft", "free")


def _import_pinocchio():
    """Lazy import of pinocchio with an actionable error message."""
    try:
        import pinocchio as pin

        return pin
    except ImportError as e:
        raise ImportError(
            "PiperEE requires pinocchio. Install via:\n"
            "    conda install -c conda-forge pinocchio\n"
            "(or `pip install pin` where wheels are available; conda-forge is the\n"
            "recommended source on platforms without PyPI wheels.)"
        ) from e


def _default_urdf_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / "piper" / "piper.urdf")


class PiperEEKinematics:
    """Forward kinematics + DLS inverse kinematics for the Piper 6-DOF.

    Parameters
    ----------
    urdf_path : str | None
        If None, the bundled URDF at ``piper/piper.urdf`` is used.
    ee_frame_parent : str
        Name of the joint whose distal frame is the EE parent.
    ee_translation : (float, float, float)
        Translation offset (metres) from ``ee_frame_parent`` to the EE point.
    ee_rpy : (float, float, float)
        Roll-pitch-yaw offset (radians) of the EE frame, applied as
        ``R = Rz · Ry · Rx`` (Tait-Bryan ZYX intrinsic).
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        ee_frame_parent: str = "joint6",
        ee_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._pin = _import_pinocchio()
        pin = self._pin

        path = urdf_path or _default_urdf_path()
        if not Path(path).exists():
            raise FileNotFoundError(f"URDF not found: {path}")

        # Pinocchio does not interpret URDF mesh paths relative to the URDF
        # file; we must explicitly point it at the directory tree that
        # contains the mesh files.
        urdf_dir = str(Path(path).resolve().parent)
        robot = pin.RobotWrapper.BuildFromURDF(path, package_dirs=[urdf_dir])

        # Lock the gripper joints (joint7, joint8) so q has 6 entries.
        locked = [j for j in ("joint7", "joint8") if robot.model.existJointName(j)]
        self._reduced = robot.buildReducedRobot(
            list_of_joints_to_lock=locked,
            reference_configuration=np.zeros(robot.model.nq),
        )
        self.model: pin.Model = self._reduced.model
        self.data: pin.Data = self.model.createData()
        self.nq: int = self.model.nq

        # Add the EE frame as a child of `ee_frame_parent` with configurable offset.
        parent_joint_id = self.model.getJointId(ee_frame_parent)
        rot = _rpy_to_matrix(ee_rpy[0], ee_rpy[1], ee_rpy[2])
        offset = pin.SE3(rot, np.asarray(ee_translation, dtype=np.float64))
        self.model.addFrame(
            pin.Frame(
                "ee",
                parent_joint_id,
                offset,
                pin.FrameType.OP_FRAME,
            )
        )
        # Re-create data after adding the frame — pinocchio requires this.
        self.data = self.model.createData()
        self.ee_frame_id: int = self.model.getFrameId("ee")

        # Cached joint limits (radians, from URDF).
        self.q_min: np.ndarray = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.q_max: np.ndarray = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def fk(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics. Returns ``(pos[3], R[3,3], rpy[3])``."""
        pin = self._pin
        q = np.asarray(q_rad, dtype=np.float64)
        pin.framesForwardKinematics(self.model, self.data, q)
        frame = self.data.oMf[self.ee_frame_id]
        rpy = pin.rpy.matrixToRpy(frame.rotation)
        return np.array(frame.translation), np.array(frame.rotation), np.array(rpy)

    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """6x6 body Jacobian in the ``LOCAL_WORLD_ALIGNED`` frame.

        Top 3 rows: ``∂pos/∂q`` in world; bottom 3: ``∂ω_world/∂q``.
        """
        pin = self._pin
        q = np.asarray(q_rad, dtype=np.float64)
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        return np.asarray(
            pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
        )

    # ------------------------------------------------------------------
    # SO(3) helpers (public so the projector can clamp rotations without
    # importing pinocchio itself)
    # ------------------------------------------------------------------

    def rotation_log(self, R: np.ndarray) -> np.ndarray:
        """Axis-angle (3-vec) logarithm of a rotation matrix."""
        return np.asarray(self._pin.log3(np.asarray(R, dtype=np.float64)))

    def rotation_exp(self, omega: np.ndarray) -> np.ndarray:
        """Rotation matrix exponential of an axis-angle 3-vec."""
        return np.asarray(self._pin.exp3(np.asarray(omega, dtype=np.float64)))

    def rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Public alias of the module-level ZYX-intrinsic RPY → matrix helper."""
        return _rpy_to_matrix(roll, pitch, yaw)

    def matrix_to_rpy(self, R: np.ndarray) -> np.ndarray:
        """Rotation matrix → principal-branch RPY (ZYX intrinsic)."""
        return np.asarray(self._pin.rpy.matrixToRpy(np.asarray(R, dtype=np.float64)))

    def condition_number(self, q_rad: np.ndarray) -> float:
        """Condition number of the full 6-D task Jacobian at ``q_rad``."""
        return float(np.linalg.cond(self.jacobian(q_rad)))

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        q_seed: np.ndarray,
        *,
        max_iter: int = 50,
        damping: float = 0.01,
        pos_tol: float = 1e-3,
        rot_tol: float = 5e-3,
        step_cap: float = 0.4,
        yaw_mode: str = "soft",
        yaw_weight: float = 0.10,
        condition_fallback_threshold: float = 1e3,
        fallback_joint_jump_rad: float = 0.6,
        max_joint_step: float | None = None,
        null_space_weight: float = 0.1,
        null_space_target: np.ndarray | None = None,
        null_space_target_weight: float = 0.0,
        null_space_target_tolerance: float = 1e-3,
        adaptive_damping: bool = True,
        damping_max: float = 0.1,
        sigma_threshold: float = 0.01,
        free_yaw: bool | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Damped-least-squares IK solver. All inputs in URDF rad / metres.

        Parameters
        ----------
        target_pos : (3,) array
            Target EE position in world frame.
        target_R : (3, 3) array
            Target EE rotation matrix in world frame.
        q_seed : (nq,) array
            Initial joint configuration (radians, URDF convention).
        yaw_mode : "full" | "soft" | "free"
            Yaw task handling (see module docstring).
        yaw_weight : float
            Weight of the yaw task row in ``"soft"`` mode, in (0, 1].
        condition_fallback_threshold : float
            In ``"full"`` mode, if ``cond(J)`` at the seed exceeds this the
            solver falls back to ``"free"`` upfront.
        fallback_joint_jump_rad : float
            If a ``"full"``/``"soft"`` solution moves any joint further than
            this from the seed, retry yaw-free.  This is the guard against
            the yaw/position-coupled wrist flip: near the home singularity a
            yaw-constrained task can be "solved" with an ~80–100° wrist
            reconfiguration that barely moves the EE — soft weighting only
            slows that flip down, it does not forbid it.  Set <= 0 to disable.
        max_joint_step : float | None
            Per-joint cap on ``|q - q_seed|`` (radians). Applied
            direction-preserving: the whole delta vector is scaled, never
            clipped per-joint, so the commanded motion direction is kept.
        null_space_target : (nq,) array | None
            Optional neutral posture for ``"free"`` yaw mode. Finite entries are
            pulled through the task null-space; NaN entries are ignored. This
            prevents the free yaw DOF from accumulating into uncomfortable
            wrist configurations across ticks.
        null_space_target_weight : float
            Gain of the neutral-posture pull. Set <= 0 to disable.
        null_space_target_tolerance : float
            Stop the neutral-posture relaxation once the finite target entries
            are within this norm (radians), provided the essential task is met.
        adaptive_damping : bool
            Boost the DLS damping near singularities:
            ``λ_eff² = λ² + (1 - (σ_min/σ_thr)²) · λ_max²`` when
            ``σ_min < σ_thr`` (Chiaverini's numerical filtering).
        free_yaw : bool | None
            Legacy alias — ``True`` maps to ``yaw_mode="free"``, ``False``
            to ``"full"``. Ignored when None.

        Returns
        -------
        (q, info)
            ``q``: converged joint configuration, clipped to URDF limits.
            ``info``: diagnostics dict — ``converged``,
            ``iters``, ``pos_err``, ``rot_err`` (roll/pitch error norm),
            ``yaw_err``, ``condition_number`` (full J at the solution),
            ``condition_number_seed``, ``max_abs_delta``, ``capped``,
            ``fallback_used``, ``yaw_mode_requested``, ``yaw_mode_used``,
            ``null_space_target_err``.
        """
        if free_yaw is not None:
            yaw_mode = "free" if free_yaw else "full"
        if yaw_mode not in YAW_MODES:
            raise ValueError(f"yaw_mode must be one of {YAW_MODES}, got {yaw_mode!r}")

        q_seed = np.clip(np.asarray(q_seed, dtype=np.float64).reshape(self.nq), self.q_min, self.q_max)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
        ns_target, ns_target_mask = self._normalise_null_space_target(null_space_target)

        solve_kwargs = {
            "max_iter": max_iter,
            "damping": damping,
            "pos_tol": pos_tol,
            "rot_tol": rot_tol,
            "step_cap": step_cap,
            "yaw_weight": yaw_weight,
            "null_space_weight": null_space_weight,
            "null_space_target": ns_target,
            "null_space_target_mask": ns_target_mask,
            "null_space_target_weight": null_space_target_weight,
            "null_space_target_tolerance": null_space_target_tolerance,
            "adaptive_damping": adaptive_damping,
            "damping_max": damping_max,
            "sigma_threshold": sigma_threshold,
        }

        requested = yaw_mode
        fallback_used = False

        # Upfront conditioning fallback: a hard 6-D yaw task at an
        # ill-conditioned seed is the documented 80°-wrist-jump case.
        cond_seed = float(np.linalg.cond(self.jacobian(q_seed)))
        if yaw_mode == "full" and cond_seed > condition_fallback_threshold:
            yaw_mode = "free"
            fallback_used = True

        q, info = self._solve_once(target_pos, target_R, q_seed, yaw_mode=yaw_mode, **solve_kwargs)

        # Post-hoc fallback: retry yaw-free when the constrained solve did not
        # converge or asked for a joint jump (slow-motion wrist flip).
        jump_thr = float(fallback_joint_jump_rad)
        jumped = jump_thr > 0.0 and float(np.max(np.abs(q - q_seed))) > jump_thr
        if yaw_mode != "free" and (not info["converged"] or jumped):
            q_fb, info_fb = self._solve_once(target_pos, target_R, q_seed, yaw_mode="free", **solve_kwargs)
            fb_jumped = jump_thr > 0.0 and float(np.max(np.abs(q_fb - q_seed))) > jump_thr
            adopt = (
                (info_fb["converged"] and not fb_jumped)
                or (jumped and float(np.max(np.abs(q_fb - q_seed))) < float(np.max(np.abs(q - q_seed))))
                or (
                    not info["converged"]
                    and info_fb["pos_err"] + info_fb["rot_err"] < info["pos_err"] + info["rot_err"]
                )
            )
            if adopt:
                q, info = q_fb, info_fb
                yaw_mode = "free"
                fallback_used = True

        # Direction-preserving per-joint total delta cap.
        capped = False
        if max_joint_step is not None and max_joint_step > 0.0:
            delta = q - q_seed
            max_abs = float(np.max(np.abs(delta)))
            if max_abs > max_joint_step:
                q = np.clip(q_seed + delta * (max_joint_step / max_abs), self.q_min, self.q_max)
                capped = True
                # Errors at the capped configuration differ from the solve's.
                pos_err, rot_err, yaw_err = self._pose_errors(target_pos, target_R, q)
                info.update(pos_err=pos_err, rot_err=rot_err, yaw_err=yaw_err)

        info.update(
            condition_number=float(np.linalg.cond(self.jacobian(q))),
            condition_number_seed=cond_seed,
            max_abs_delta=float(np.max(np.abs(q - q_seed))),
            capped=capped,
            fallback_used=fallback_used,
            yaw_mode_requested=requested,
            yaw_mode_used=yaw_mode,
            null_space_target_err=self._null_space_target_error(q, ns_target, ns_target_mask),
        )
        return q, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pose_errors(
        self, target_pos: np.ndarray, target_R: np.ndarray, q: np.ndarray
    ) -> tuple[float, float, float]:
        """(pos_err, roll/pitch rot_err, yaw_err) of configuration ``q``."""
        pin = self._pin
        pin.framesForwardKinematics(self.model, self.data, q)
        T_curr = self.data.oMf[self.ee_frame_id]
        e_pos = target_pos - np.array(T_curr.translation)
        e_rot = np.array(pin.log3(target_R @ np.array(T_curr.rotation).T))
        return float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot[:2])), float(abs(e_rot[2]))

    def _normalise_null_space_target(
        self, target: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(target, finite_mask)`` for optional yaw-free posture bias."""
        if target is None:
            return None, None
        target_arr = np.asarray(target, dtype=np.float64).reshape(self.nq)
        mask = np.isfinite(target_arr)
        if not bool(np.any(mask)):
            return None, None
        target_arr = np.where(mask, target_arr, 0.0)
        target_arr = np.clip(target_arr, self.q_min, self.q_max)
        return target_arr, mask

    def _null_space_target_error(
        self,
        q: np.ndarray,
        target: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> float:
        if target is None or mask is None:
            return 0.0
        return float(np.linalg.norm((target - q) * mask))

    def _solve_once(
        self,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        q_seed: np.ndarray,
        *,
        yaw_mode: str,
        max_iter: int,
        damping: float,
        pos_tol: float,
        rot_tol: float,
        step_cap: float,
        yaw_weight: float,
        null_space_weight: float,
        null_space_target: np.ndarray | None,
        null_space_target_mask: np.ndarray | None,
        null_space_target_weight: float,
        null_space_target_tolerance: float,
        adaptive_damping: bool,
        damping_max: float,
        sigma_threshold: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """One DLS descent in a fixed yaw mode. Returns ``(q, partial info)``.

        Convergence is on position + roll/pitch for ``"soft"``/``"free"``
        (yaw is best-effort there) and additionally on yaw for ``"full"``.
        """
        pin = self._pin
        q = q_seed.copy()
        lam2_base = damping**2
        has_target_bias = (
            yaw_mode == "free"
            and null_space_target is not None
            and null_space_target_mask is not None
            and null_space_target_weight > 0.0
        )

        pos_err = rot_err = yaw_err = float("inf")
        iters = 0
        while iters < int(max_iter):
            iters += 1
            pin.framesForwardKinematics(self.model, self.data, q)
            T_curr = self.data.oMf[self.ee_frame_id]
            current_pos = np.array(T_curr.translation)
            current_R = np.array(T_curr.rotation)

            e_pos = target_pos - current_pos
            e_rot = np.array(pin.log3(target_R @ current_R.T))

            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot[:2]))
            yaw_err = float(abs(e_rot[2]))

            essential_ok = pos_err < pos_tol and rot_err < rot_tol
            # "free" never looks at yaw; "soft"/"full" keep descending while
            # yaw still improves (soft additionally stops on stall — below).
            if essential_ok:
                if yaw_mode == "free":
                    if (
                        not has_target_bias
                        or self._null_space_target_error(q, null_space_target, null_space_target_mask)
                        < null_space_target_tolerance
                    ):
                        break
                elif yaw_err < rot_tol:
                    break

            J = self.jacobian(q)  # (6, nq)
            if yaw_mode == "free":
                J_task = np.vstack([J[:3, :], J[3:5, :]])
                err = np.concatenate([e_pos, e_rot[:2]])
            elif yaw_mode == "soft":
                w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, float(yaw_weight)])
                J_task = J * w[:, None]
                err = np.concatenate([e_pos, e_rot]) * w
            else:  # full
                J_task = J
                err = np.concatenate([e_pos, e_rot])

            lam2 = lam2_base
            if adaptive_damping:
                # Conditioning of the directions we are *required* to track:
                # the unweighted 5-D pos+roll+pitch subtask for soft/free (the
                # soft yaw row is weak by design — its σ would always trigger),
                # the full 6-D task for "full".
                J_ess = J if yaw_mode == "full" else np.vstack([J[:3, :], J[3:5, :]])
                sigma_min = float(np.linalg.svd(J_ess, compute_uv=False)[-1])
                if sigma_min < sigma_threshold:
                    ratio = sigma_min / sigma_threshold
                    lam2 = lam2_base + (1.0 - ratio * ratio) * damping_max**2

            m = J_task.shape[0]
            JJt = J_task @ J_task.T + lam2 * np.eye(m)
            dq = J_task.T @ np.linalg.solve(JJt, err)

            if yaw_mode == "free" and (null_space_weight > 0.0 or has_target_bias):
                J_pinv = J_task.T @ np.linalg.solve(JJt, np.eye(m))
                N = np.eye(self.nq) - J_pinv @ J_task
                # Seed term: local continuity. Target term: global drift drain.
                null_drive = null_space_weight * (q_seed - q)
                if has_target_bias:
                    null_drive = null_drive + null_space_target_weight * (
                        (null_space_target - q) * null_space_target_mask
                    )
                dq = dq + N @ null_drive

            n_dq = float(np.linalg.norm(dq))

            if essential_ok and yaw_mode == "free" and has_target_bias and n_dq < 1e-5:
                break

            # Soft mode with essentials met: the only remaining drive is the
            # down-weighted yaw row.  Near the singularity its damped gain is
            # ~0 and the iterate stalls — stop instead of spinning to max_iter.
            if essential_ok and yaw_mode == "soft" and n_dq < 1e-4:
                break

            if n_dq > step_cap:
                dq = dq * (step_cap / n_dq)

            q = np.clip(q + dq, self.q_min, self.q_max)

        # "soft" converges on the essentials (yaw is best-effort by design);
        # "full" additionally requires yaw within tolerance.
        converged = pos_err < pos_tol and rot_err < rot_tol and (yaw_mode != "full" or yaw_err < rot_tol)
        return q, {
            "converged": bool(converged),
            "iters": iters,
            "pos_err": pos_err,
            "rot_err": rot_err,
            "yaw_err": yaw_err,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3×3 rotation matrix ``R = Rz(yaw) · Ry(pitch) · Rx(roll)``.

    This is the Tait-Bryan ZYX intrinsic convention used throughout the
    project (also identical to the XYZ extrinsic convention).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx
