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

"""Hardware-free tests for the Piper EE pipeline (IK, projector, chunk smoother).

These exercise the pipeline's success criteria against the bundled URDF.  They are skipped when pinocchio / osqp are not
installed (``pip install pin`` / the ``chunk-smoother`` extra).
"""

import numpy as np
import pytest
import torch

pin = pytest.importorskip("pinocchio")

from lerobot.robots.piper_ee.piper_ee_ik import PiperEEKinematics  # noqa: E402


@pytest.fixture(scope="module")
def kin() -> PiperEEKinematics:
    return PiperEEKinematics()


def _rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# IK (plan Fasi 1 + 3)
# ---------------------------------------------------------------------------


class TestPiperEEIK:
    def test_home_is_ill_conditioned(self, kin):
        """The documented problem: cond ≈ 2.8e4 full vs ≈ 31 yaw-free at home."""
        J = kin.jacobian(np.zeros(6))
        assert np.linalg.cond(J) > 1e4
        J5 = np.vstack([J[:3, :], J[3:5, :]])
        assert np.linalg.cond(J5) < 100

    @pytest.mark.parametrize("mode", ["full", "soft", "free"])
    def test_no_wrist_jump_from_home(self, kin, mode):
        """Plan success criterion: small x/y targets must not produce ~80° jumps."""
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        q, info = kin.solve(pos + np.array([0.0, 0.02, 0.0]), R, q0, yaw_mode=mode)
        max_dq_deg = np.max(np.abs(np.degrees(q - q0)))
        assert max_dq_deg < 25.0, f"wrist jump in {mode}: {max_dq_deg:.1f} deg"
        # position residual small (home sits on a joint-limit corner: a small
        # part of the target is unreachable, ~1.8 mm)
        assert info["pos_err"] < 3e-3

    def test_full_mode_falls_back_at_home(self, kin):
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        _, info = kin.solve(pos + np.array([0.0, 0.02, 0.0]), R, q0, yaw_mode="full")
        assert info["fallback_used"]
        assert info["yaw_mode_used"] == "free"
        assert info["yaw_mode_requested"] == "full"

    def test_full_mode_tracks_yaw_away_from_singularity(self, kin):
        q_far = np.array([0.3, 0.8, -0.5, 0.2, 0.4, 0.1])
        pos, R, _ = kin.fk(q_far)
        q, info = kin.solve(pos + np.array([0.0, 0.02, 0.0]), R, q_far, yaw_mode="full")
        assert info["converged"] and not info["fallback_used"]
        assert np.degrees(info["yaw_err"]) < 0.5

    def test_soft_tracks_cheap_yaw(self, kin):
        q_far = np.array([0.3, 0.8, -0.5, 0.2, 0.4, 0.1])
        pos, R, _ = kin.fk(q_far)
        q, info = kin.solve(pos, _rotz(np.radians(8)) @ R, q_far, yaw_mode="soft")
        assert info["converged"] and not info["fallback_used"]
        assert np.degrees(info["yaw_err"]) < 1.0

    def test_soft_releases_yaw_at_home(self, kin):
        """At the singularity the yaw request is dropped instead of chased."""
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        q, info = kin.solve(pos, _rotz(np.radians(8)) @ R, q0, yaw_mode="soft")
        assert np.max(np.abs(np.degrees(q - q0))) < 2.0  # barely moves

    def test_joint_jump_fallback(self, kin):
        """Interior config near home: holding yaw while translating demands a
        ~97° wrist flip — the jump guard must reroute to yaw-free."""
        q_in = np.array([0.0, 0.15, -0.15, 0.0, 0.0, 0.0])
        pos, R, _ = kin.fk(q_in)
        q, info = kin.solve(pos + np.array([0.0, 0.02, 0.0]), R, q_in, yaw_mode="soft")
        assert info["fallback_used"] and info["yaw_mode_used"] == "free"
        assert np.max(np.abs(np.degrees(q - q_in))) < 35.0
        assert info["pos_err"] < 1e-3

    def test_diagnostics_complete(self, kin):
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        _, info = kin.solve(pos, R, q0)
        for key in (
            "converged",
            "iters",
            "pos_err",
            "rot_err",
            "yaw_err",
            "condition_number",
            "max_abs_delta",
            "capped",
            "fallback_used",
            "yaw_mode_requested",
            "yaw_mode_used",
        ):
            assert key in info, f"missing diagnostic: {key}"

    def test_direction_preserving_cap(self, kin):
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        target = pos + np.array([0.15, 0.10, -0.05])
        q_unc, _ = kin.solve(target, R, q0, yaw_mode="soft", max_joint_step=None)
        q_cap, info = kin.solve(target, R, q0, yaw_mode="soft", max_joint_step=0.056)
        assert info["capped"]
        assert np.max(np.abs(q_cap - q0)) <= 0.056 + 1e-9
        d_unc, d_cap = q_unc - q0, q_cap - q0
        cos = np.dot(d_unc, d_cap) / (np.linalg.norm(d_unc) * np.linalg.norm(d_cap))
        assert cos > 0.999  # same direction, only scaled

    def test_legacy_free_yaw_alias(self, kin):
        q0 = np.zeros(6)
        pos, R, _ = kin.fk(q0)
        _, info = kin.solve(pos, R, q0, free_yaw=True)
        assert info["yaw_mode_requested"] == "free"

    def test_free_yaw_neutral_bias_drains_wrist_drift(self, kin):
        """Yaw-free has a true null-space target, not only a seed-local bias."""
        q_drift = np.radians(np.array([-15.0, 84.0, -61.0, 39.0, -27.0, -34.0]))
        pos, R, _ = kin.fk(q_drift)
        neutral = np.array([np.nan, np.nan, np.nan, 0.0, np.nan, 0.0])
        q, info = kin.solve(
            pos,
            R,
            q_drift,
            yaw_mode="free",
            null_space_target=neutral,
            null_space_target_weight=0.15,
            max_joint_step=None,
        )
        assert info["converged"]
        assert info["pos_err"] < 1e-4
        assert info["rot_err"] < 1e-4
        assert abs(q[3]) < 0.6 * abs(q_drift[3])
        assert abs(q[5]) < 0.6 * abs(q_drift[5])


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


@pytest.fixture()
def projector(kin):
    from lerobot.robots.piper_ee.ee_action_projector import EEActionProjector

    return EEActionProjector(
        kin,
        max_dpos_per_step=0.02,
        max_drot_per_step=0.08,
        solve_kwargs={"yaw_mode": "soft", "max_joint_step": np.radians(80) / 25.0},
    )


class TestEEActionProjector:
    Q0 = np.array([0.0, 0.6, -0.6, 0.0, 0.3, 0.0])

    def test_translation_clamp(self, kin, projector):
        pos, R, _ = kin.fk(self.Q0)
        res = projector.project(
            pos + np.array([0.30, 0.0, 0.0]), R, current_pos=pos, current_R=R, q_seed=self.Q0
        )
        assert res.info["clamped_pos"]
        assert np.linalg.norm(res.pos - pos) <= 0.02 + 1e-9

    def test_rotation_clamp_so3(self, kin, projector):
        pos, R, _ = kin.fk(self.Q0)
        res = projector.project(pos, _rotz(np.radians(120)) @ R, current_pos=pos, current_R=R, q_seed=self.Q0)
        assert res.info["clamped_rot"]
        angle = np.linalg.norm(kin.rotation_log(res.R @ R.T))
        assert angle <= 0.08 + 1e-9

    def test_hold_pose_when_nothing_valid(self, kin):
        """An unreachable target with no history holds the current pose."""
        from lerobot.robots.piper_ee.ee_action_projector import EEActionProjector

        proj = EEActionProjector(
            kin,
            max_dpos_per_step=10.0,  # no clamp: feed the IK garbage directly
            max_drot_per_step=10.0,
            solve_kwargs={"yaw_mode": "soft", "max_iter": 10},
        )
        pos, R, _ = kin.fk(self.Q0)
        res = proj.project(pos + np.array([5.0, 0.0, 0.0]), R, current_pos=pos, current_R=R, q_seed=self.Q0)
        assert res.info["used_last_valid"] and res.info["held_pose"]
        assert np.allclose(res.q, self.Q0)


# ---------------------------------------------------------------------------
# EE chunk smoother
# ---------------------------------------------------------------------------


class TestEEChunkSmoother:
    KEYS = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]
    Q0 = np.array([0.0, 0.6, -0.6, 0.0, 0.3, 0.0])
    RATE = 25.0

    @pytest.fixture()
    def smoother(self, kin, projector):
        pytest.importorskip("osqp")
        from lerobot.robots.piper_ee.ee_chunk_smoother import EEChunkSmoother

        return EEChunkSmoother(
            kin,
            projector,
            policy_action_keys=self.KEYS,
            output_action_keys=self.KEYS,
            output_space="ee",
            v_max_deg_s=80.0,
            lambda_a=40.0,
            lambda_j=20.0,
            rate_hz=self.RATE,
        )

    def _jerky_chunk(self, kin, T=40, seed=7):
        rng = np.random.default_rng(seed)
        pos0, _, rpy0 = kin.fk(self.Q0)
        chunk = np.zeros((T, 7))
        base = np.linspace(0, 1, T)[:, None]
        chunk[:, :3] = pos0 + base * np.array([0.10, 0.08, -0.05]) + rng.normal(0, 0.008, (T, 3))
        chunk[:, 3:6] = rpy0 + rng.normal(0, 0.15, (T, 3))
        chunk[:, 6] = 50.0
        return torch.tensor(chunk, dtype=torch.float32)

    def test_velocity_cap_anchor_consistency(self, kin, smoother):
        out, diag = smoother.solve(self._jerky_chunk(kin), self.Q0)
        q_s = diag["q_smooth_deg"]
        # joint velocity capped
        assert np.abs(np.diff(q_s, axis=0)).max() * self.RATE <= 80.0 * 1.05
        # anchored at q0
        assert np.abs(q_s[0] - np.degrees(self.Q0)).max() < 1e-6
        # served EE rows are exactly FK of the smoothed joints
        arr = out.numpy()
        for t in range(0, arr.shape[0], 5):
            p, _, _ = kin.fk(np.radians(q_s[t]))
            assert np.linalg.norm(arr[t, :3] - p) < 1e-5
        # gripper passthrough
        assert np.allclose(arr[:, 6], 50.0)

    def test_rpy_continuity_across_branch(self, kin, smoother):
        """Start orientation sits at rpy ≈ (−180°, 78°, −180°) — the ±π branch
        cut.  Output rows must stay representation-continuous anyway."""
        out, _ = smoother.solve(self._jerky_chunk(kin), self.Q0)
        rpy = out.numpy()[:, 3:6]
        assert np.abs(np.diff(rpy, axis=0)).max() < np.pi / 2


class TestEEChunkSmootherJointOutput:
    """EE chunk in, SDK-ready *joint* chunk out (the piper_full deployment path)."""

    POLICY_KEYS = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]
    JOINT_KEYS = [f"joint_{i + 1}.pos" for i in range(6)]
    EXEC_KEYS = [*JOINT_KEYS, "gripper.pos"]
    Q0 = np.array([0.0, 0.6, -0.6, 0.0, 0.3, 0.0])
    RATE = 25.0

    @pytest.fixture()
    def smoother(self, kin, projector):
        pytest.importorskip("osqp")
        from lerobot.robots.piper_ee.ee_chunk_smoother import EEChunkSmoother

        return EEChunkSmoother(
            kin,
            projector,
            policy_action_keys=self.POLICY_KEYS,
            output_action_keys=self.EXEC_KEYS,
            output_space="joint",
            joint_output_keys=self.JOINT_KEYS,
            v_max_deg_s=80.0,
            lambda_a=40.0,
            lambda_j=20.0,
            rate_hz=self.RATE,
        )

    def test_joint_output(self, kin, smoother):
        rng = np.random.default_rng(11)
        pos0, _, rpy0 = kin.fk(self.Q0)
        T = 40
        chunk = np.zeros((T, 7))
        base = np.linspace(0, 1, T)[:, None]
        chunk[:, :3] = pos0 + base * np.array([0.10, 0.08, -0.05]) + rng.normal(0, 0.008, (T, 3))
        chunk[:, 3:6] = rpy0 + rng.normal(0, 0.15, (T, 3))
        chunk[:, 6] = 42.0
        out, diag = smoother.solve(torch.tensor(chunk, dtype=torch.float32), self.Q0)

        assert diag["output_space"] == "joint"
        arr = out.numpy()
        assert arr.shape == (T, len(self.EXEC_KEYS))
        # joint columns ARE the smoothed trajectory (default unit: URDF deg)
        assert np.allclose(arr[:, :6], diag["q_smooth_deg"], atol=1e-4)
        # velocity cap holds directly on the served actions
        assert np.abs(np.diff(arr[:, :6], axis=0)).max() * self.RATE <= 80.0 * 1.05
        # anchored at q0
        assert np.abs(arr[0, :6] - np.degrees(self.Q0)).max() < 1e-6
        # gripper passthrough into the execution column
        assert np.allclose(arr[:, 6], 42.0)

    def test_custom_unit_converter(self, kin, projector):
        """q_rad_to_action maps the joint columns to the robot's action unit."""
        pytest.importorskip("osqp")
        from lerobot.robots.piper_ee.ee_chunk_smoother import EEChunkSmoother

        signs = np.array([-1, 1, 1, -1, 1, -1], dtype=np.float64)
        sm = EEChunkSmoother(
            kin,
            projector,
            policy_action_keys=self.POLICY_KEYS,
            output_action_keys=self.EXEC_KEYS,
            output_space="joint",
            joint_output_keys=self.JOINT_KEYS,
            q_rad_to_action=lambda q: np.degrees(q) * signs,  # signed-deg convention
            v_max_deg_s=80.0,
            lambda_a=40.0,
            lambda_j=20.0,
            rate_hz=self.RATE,
        )
        pos0, _, rpy0 = kin.fk(self.Q0)
        chunk = np.zeros((10, 7))
        chunk[:, :3] = pos0
        chunk[:, 3:6] = rpy0
        out, diag = sm.solve(torch.tensor(chunk, dtype=torch.float32), self.Q0)
        assert np.allclose(out.numpy()[:, :6], diag["q_smooth_deg"] * signs, atol=1e-4)


class TestResolveEEMode:
    JOINT_KEYS = [f"joint_{i + 1}.pos" for i in range(6)]
    EXEC_JOINTS = [*JOINT_KEYS, "gripper.pos"]
    EE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]

    def _resolve(self, **kw):
        from lerobot.policies.chunk_smoother.anchor import resolve_ee_mode

        defaults = {
            "ee_mode": None,
            "joint_keys": self.JOINT_KEYS,
            "ordered_action_keys": self.EXEC_JOINTS,
            "policy_action_names": None,
            "ee_factory_available": True,
            "engine_name": "test",
        }
        return resolve_ee_mode(**{**defaults, **kw})

    def test_auto_ee_policy_on_joint_robot(self):
        assert self._resolve(policy_action_names=self.EE_NAMES) is True

    def test_auto_joint_policy_on_joint_robot(self):
        assert self._resolve(policy_action_names=self.EXEC_JOINTS) is False

    def test_auto_legacy_cartesian_exec_space(self):
        assert self._resolve(ordered_action_keys=self.EE_NAMES) is True

    def test_auto_nameless_ambiguous_raises(self):
        """No names + robot supports both interpretations → demand the flag."""
        with pytest.raises(ValueError, match="ee_mode"):
            self._resolve(policy_action_names=None)

    def test_auto_nameless_without_factory_stays_joint(self):
        assert self._resolve(policy_action_names=None, ee_factory_available=False) is False

    def test_forced_off_with_ee_policy_on_joints_raises(self):
        with pytest.raises(ValueError, match="mislabelled"):
            self._resolve(ee_mode=False, policy_action_names=self.EE_NAMES)

    def test_forced_on_without_factory_raises(self):
        with pytest.raises(ValueError, match="make_ee_chunk_smoother"):
            self._resolve(ee_mode=True, ee_factory_available=False)

    def test_forced_on_with_joint_policy_raises(self):
        with pytest.raises(ValueError, match="does not emit"):
            self._resolve(ee_mode=True, policy_action_names=self.EXEC_JOINTS)


# ---------------------------------------------------------------------------
# SLERP interpolation
# ---------------------------------------------------------------------------


class TestRPYInterpolation:
    def test_rpy_triples_detection(self):
        from lerobot.utils.action_interpolator import rpy_triples_from_keys

        keys = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]
        assert rpy_triples_from_keys(keys) == [(3, 4, 5)]
        assert rpy_triples_from_keys(["joint_1.pos", "gripper.pos"]) == []

    def test_slerp_crosses_pi_branch(self):
        """yaw +176° → −176°: SLERP goes the 8° way through ±π, linear lerp
        would sweep 352° through zero."""
        from lerobot.utils.action_interpolator import ActionInterpolator

        interp = ActionInterpolator(multiplier=2, rpy_triples=[(0, 1, 2)])
        a = torch.tensor([0.0, 0.0, np.radians(176.0), 1.0])
        b = torch.tensor([0.0, 0.0, np.radians(-176.0), 2.0])
        interp.add(a)
        interp.get()
        interp.add(b)
        mid = interp.get()
        assert abs(mid[2]) > np.radians(170.0)  # short way: |yaw| stays near π
        assert abs(mid[3] - 1.5) < 1e-6  # scalar dims still linear

    def test_slerp_matches_lerp_for_small_angles(self):
        from lerobot.utils.action_interpolator import ActionInterpolator

        interp = ActionInterpolator(multiplier=2, rpy_triples=[(0, 1, 2)])
        a = torch.tensor([0.10, 0.20, 0.30])
        b = torch.tensor([0.12, 0.22, 0.34])
        interp.add(a)
        interp.get()
        interp.add(b)
        mid = interp.get()
        assert torch.allclose(mid, (a + b) / 2.0, atol=1e-3)
