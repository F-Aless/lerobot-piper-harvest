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

"""Contract tests for the qp_sync/qp_rtc EE mode (policy action-space resolution).

These pin the deployment-critical behaviours:

* an EE checkpoint without ``action_feature_names`` must be runnable with
  ``--inference.ee_mode=true`` (canonical layout assumed);
* ``--inference.policy_action_keys`` overrides the layout;
* the nameless+ambiguous case (robot supports both interpretations) must FAIL
  instead of guessing — an EE chunk mislabelled as joints moves the arm;
* ``--inference.rate_hz`` must match ``--fps``.

Skipped when pinocchio / osqp are unavailable.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("pinocchio")
pytest.importorskip("osqp")

from lerobot.robots.piper_ee.ee_action_projector import EEActionProjector  # noqa: E402
from lerobot.robots.piper_ee.ee_chunk_smoother import EEChunkSmoother  # noqa: E402
from lerobot.robots.piper_ee.piper_ee_ik import PiperEEKinematics  # noqa: E402
from lerobot.rollout.inference.factory import QPSyncInferenceConfig  # noqa: E402
from lerobot.rollout.inference.qp_sync import QPSyncInferenceEngine  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

EE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]
JOINT_KEYS = [f"joint_{i + 1}.pos" for i in range(6)] + ["gripper.pos"]
SIGNS = np.array([-1, 1, 1, -1, 1, -1], dtype=np.float64)

_KIN = None


def _kin():
    global _KIN
    if _KIN is None:
        _KIN = PiperEEKinematics()
    return _KIN


class FakePiperFull:
    """Joint-space robot with EE conversion support (piper_full-like)."""

    robot_type = "piper_full"

    @property
    def joint_limits_deg(self):
        kin = _kin()
        return (list(np.degrees(kin.q_min)), list(np.degrees(kin.q_max)))

    def make_ee_chunk_smoother(
        self,
        *,
        policy_action_keys,
        output_action_keys,
        joint_keys=None,
        gripper_key="gripper.pos",
        v_max_deg_s,
        lambda_a,
        lambda_j,
        rate_hz,
        strict_anchor=False,
    ):
        kin = _kin()
        projector = EEActionProjector(
            kin,
            max_dpos_per_step=0.5 / rate_hz,
            max_drot_per_step=2.0 / rate_hz,
            solve_kwargs={"yaw_mode": "free", "max_joint_step": np.radians(v_max_deg_s) / rate_hz},
        )
        return EEChunkSmoother(
            kin,
            projector,
            policy_action_keys=policy_action_keys,
            output_action_keys=output_action_keys,
            output_space="joint",
            joint_output_keys=[f"joint_{i + 1}.pos" for i in range(6)],
            gripper_key=gripper_key,
            q_rad_to_action=lambda q: np.degrees(q) * SIGNS,
            v_max_deg_s=v_max_deg_s,
            lambda_a=lambda_a,
            lambda_j=lambda_j,
            rate_hz=rate_hz,
            strict_anchor=strict_anchor,
        )

    def urdf_q_from_observation(self, obs):
        if any(f"joint_{i + 1}.pos" not in obs for i in range(6)):
            return None
        signed = np.array([obs[f"joint_{i + 1}.pos"] for i in range(6)])
        return np.deg2rad(signed * SIGNS)

    # Same convention for actions: joint_X.pos columns in signed degrees
    # (inverse of the q_rad_to_action above), like the real PiperFull.
    ee_anchor_q_from_observation = urdf_q_from_observation


class JointOnlyRobot:
    """Joint-space robot WITHOUT EE conversion support."""

    robot_type = "generic"
    joint_limits_deg = ([-170.0] * 6, [170.0] * 6)


class _Identity:
    def __call__(self, x):
        return x

    def reset(self):
        pass


def _make_policy(action_feature_names=None, ee_chunk=True, permute=None):
    kin = _kin()
    q0 = np.array([0.0, 0.6, -0.6, 0.0, 0.3, 0.0])
    pos0, _, rpy0 = kin.fk(q0)

    class _Cfg:
        use_amp = False

    if action_feature_names is not None:
        _Cfg.action_feature_names = action_feature_names

    class _Policy:
        config = _Cfg()
        calls = 0

        def predict_action_chunk(self, obs):
            self.calls += 1
            rng = np.random.default_rng(self.calls)
            T = 20
            chunk = np.zeros((1, T, 7), dtype=np.float32)
            if ee_chunk:
                base = np.linspace(0, 1, T)[:, None]
                chunk[0, :, :3] = pos0 + base * np.array([0.06, 0.04, -0.03]) + rng.normal(0, 0.005, (T, 3))
                chunk[0, :, 3:6] = rpy0 + rng.normal(0, 0.1, (T, 3))
                chunk[0, :, 6] = 50.0
            else:
                chunk[0] = np.cumsum(rng.normal(0, 3.0, (T, 7)), axis=0)
            out = torch.tensor(chunk)
            return out[..., permute] if permute is not None else out

        def reset(self):
            pass

    return _Policy(), q0


def _make_engine(policy, robot, qp_config, exec_keys=JOINT_KEYS):
    return QPSyncInferenceEngine(
        policy=policy,
        preprocessor=_Identity(),
        postprocessor=_Identity(),
        robot_wrapper=robot,
        dataset_features={ACTION: {"names": exec_keys}},
        ordered_action_keys=exec_keys,
        task="t",
        device="cpu",
        qp_config=qp_config,
    )


def _serve(engine, q0, n=25):
    obs = {f"joint_{i + 1}.pos": float(np.degrees(q0[i]) * SIGNS[i]) for i in range(6)}
    engine.notify_observation(obs)
    frame = {"observation.state": np.zeros(7, dtype=np.float32)}
    served = []
    for _ in range(n):
        action = engine.get_action(dict(frame))
        # The rollout loop notifies the engine after every commanded action —
        # the joint-mode QP anchors consecutive chunks on it.
        engine.notify_last_commanded_action(action)
        served.append(action.numpy())
    return np.stack(served)


class TestEEModeResolution:
    def test_nameless_ambiguous_raises(self):
        """Fail-safe: no names + robot supports both → demand the flag."""
        policy, _ = _make_policy(action_feature_names=None)
        cfg = QPSyncInferenceConfig(rate_hz=25.0)
        with pytest.raises(ValueError, match="ee_mode"):
            _make_engine(policy, FakePiperFull(), cfg)

    def test_nameless_forced_ee_uses_canonical_layout(self):
        policy, q0 = _make_policy(action_feature_names=None)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, v_max_deg_s=80.0, ee_mode=True)
        eng = _make_engine(policy, FakePiperFull(), cfg)
        assert eng._ee_mode and eng._policy_action_keys == EE_KEYS
        served = _serve(eng, q0)
        v = np.abs(np.diff(served[:, :6] * SIGNS, axis=0)).max() * 25.0
        assert v <= 80.0 * 1.05
        assert np.allclose(served[:, 6], 50.0)

    def test_nameless_forced_joint_runs_joint_mode(self):
        policy, q0 = _make_policy(action_feature_names=None, ee_chunk=False)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, v_max_deg_s=80.0, ee_mode=False, action_unit="deg")
        eng = _make_engine(policy, FakePiperFull(), cfg)
        assert eng._ee_mode is False
        served = _serve(eng, q0)
        assert np.abs(np.diff(served[:, :6], axis=0)).max() * 25.0 <= 80.0 * 1.05

    def test_explicit_policy_action_keys_layout(self):
        custom = ["gripper.pos", *EE_KEYS[:-1]]  # gripper first
        policy, q0 = _make_policy(action_feature_names=None, permute=[6, 0, 1, 2, 3, 4, 5])
        cfg = QPSyncInferenceConfig(rate_hz=25.0, v_max_deg_s=80.0, ee_mode=True, policy_action_keys=custom)
        eng = _make_engine(policy, FakePiperFull(), cfg)
        served = _serve(eng, q0, n=5)
        assert np.allclose(served[:, 6], 50.0)  # gripper routed correctly

    def test_named_ee_policy_autodetects(self):
        policy, q0 = _make_policy(action_feature_names=EE_KEYS)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, v_max_deg_s=80.0)
        eng = _make_engine(policy, FakePiperFull(), cfg)
        assert eng._ee_mode is True
        served = _serve(eng, q0, n=5)
        assert served.shape == (5, 7)

    def test_named_joint_policy_autodetects_joint(self):
        policy, _ = _make_policy(action_feature_names=JOINT_KEYS, ee_chunk=False)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, action_unit="deg")
        eng = _make_engine(policy, FakePiperFull(), cfg)
        assert eng._ee_mode is False

    def test_nameless_on_joint_only_robot_stays_joint(self):
        policy, _ = _make_policy(action_feature_names=None, ee_chunk=False)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, action_unit="deg")
        eng = _make_engine(policy, JointOnlyRobot(), cfg)
        assert eng._ee_mode is False

    def test_width_mismatch_fails_loudly(self):
        """Canonical layout assumed but the checkpoint emits a different dim."""

        policy, q0 = _make_policy(action_feature_names=None)
        policy.predict_action_chunk = lambda obs: torch.zeros((1, 20, 9))  # 9 ≠ 7
        cfg = QPSyncInferenceConfig(rate_hz=25.0, ee_mode=True)
        eng = _make_engine(policy, FakePiperFull(), cfg)
        with pytest.raises(ValueError, match="columns"):
            _serve(eng, q0, n=1)


def _ee_chunk_from(q_urdf, T=20):
    """Straight-line EE chunk starting at fk(q_urdf), gripper at 50."""
    kin = _kin()
    pos0, _, rpy0 = kin.fk(q_urdf)
    chunk = np.zeros((T, 7), dtype=np.float32)
    chunk[:, :3] = pos0 + np.linspace(0, 1, T)[:, None] * np.array([0.05, 0.03, -0.02])
    chunk[:, 3:6] = rpy0
    chunk[:, 6] = 50.0
    return torch.tensor(chunk)


class TestCommandAnchorSeparation:
    """q0 (measurement) seeds FK/projector/IK; q_cmd pins the QP start."""

    Q_MEAS = np.array([0.0, 0.6, -0.6, 0.0, 0.3, 0.0])
    Q_CMD = Q_MEAS + np.radians([0.0, 3.0, -3.0, 0.0, 1.0, 0.0])  # arm lags ~3 deg

    def _smoother(self):
        return FakePiperFull().make_ee_chunk_smoother(
            policy_action_keys=EE_KEYS,
            output_action_keys=JOINT_KEYS,
            v_max_deg_s=80.0,
            lambda_a=40.0,
            lambda_j=20.0,
            rate_hz=25.0,
        )

    def test_solver_anchors_on_command_when_given(self):
        chunk = _ee_chunk_from(self.Q_MEAS)
        _, diag = self._smoother().solve(chunk, self.Q_MEAS, delay=0, q_cmd_rad=self.Q_CMD)
        assert np.allclose(diag["q_smooth_deg"][0], np.degrees(self.Q_CMD), atol=0.05)
        assert np.abs(diag["q_smooth_deg"][0] - np.degrees(self.Q_MEAS)).max() > 2.0
        # the measurement still seeds the conversion: raw IK starts at q_meas
        assert np.allclose(diag["q_raw_deg"][0], np.degrees(self.Q_MEAS), atol=2.0)

    def test_solver_legacy_anchor_without_command(self):
        chunk = _ee_chunk_from(self.Q_MEAS)
        _, diag = self._smoother().solve(chunk, self.Q_MEAS, delay=0)
        assert np.allclose(diag["q_smooth_deg"][0], np.degrees(self.Q_MEAS), atol=0.05)

    def _rtc_engine(self, **cfg_kwargs):
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.rollout.inference.factory import QPRTCInferenceConfig
        from lerobot.rollout.inference.qp_rtc import QPSmoothedRTCInferenceEngine

        policy, _ = _make_policy(action_feature_names=EE_KEYS)

        class _Pipeline(_Identity):
            steps = ()

        return QPSmoothedRTCInferenceEngine(
            policy=policy,
            preprocessor=_Pipeline(),
            postprocessor=_Pipeline(),
            robot_wrapper=FakePiperFull(),
            rtc_config=RTCConfig(),
            hw_features={},
            task="t",
            fps=25.0,
            device="cpu",
            qp_config=QPRTCInferenceConfig(rate_hz=25.0, v_max_deg_s=80.0, **cfg_kwargs),
            ordered_action_keys=JOINT_KEYS,
        )

    def _signed_action(self, q_urdf, gripper=50.0):
        return torch.tensor([*(np.degrees(q_urdf) * SIGNS), gripper], dtype=torch.float64)

    def test_rtc_ee_anchor_prefers_last_commanded(self):
        eng = self._rtc_engine()
        assert eng._ee_mode and eng._ee_anchor_source == "command"
        eng.notify_observation(
            {f"joint_{i + 1}.pos": float(np.degrees(self.Q_MEAS[i]) * SIGNS[i]) for i in range(6)}
        )
        eng.notify_last_commanded_action(self._signed_action(self.Q_CMD))
        chunk = _ee_chunk_from(self.Q_MEAS)
        out = eng._post_process_chunk(chunk, chunk.clone(), 0)
        # first executed command continues the command stream (q_cmd), not the lagging measurement
        q_out_deg = out[0, :6].numpy() * SIGNS
        assert np.allclose(q_out_deg, np.degrees(self.Q_CMD), atol=0.05)

    def test_rtc_ee_anchor_falls_back_to_measured(self):
        # no commanded action yet → measured anchor; same with ee_qp_anchor="measured"
        for eng in (self._rtc_engine(), self._rtc_engine(ee_qp_anchor="measured")):
            eng.notify_observation(
                {f"joint_{i + 1}.pos": float(np.degrees(self.Q_MEAS[i]) * SIGNS[i]) for i in range(6)}
            )
            if eng._ee_anchor_source == "measured":
                eng.notify_last_commanded_action(self._signed_action(self.Q_CMD))
            chunk = _ee_chunk_from(self.Q_MEAS)
            out = eng._post_process_chunk(chunk, chunk.clone(), 0)
            q_out_deg = out[0, :6].numpy() * SIGNS
            assert np.allclose(q_out_deg, np.degrees(self.Q_MEAS), atol=0.05)

    def test_rtc_ee_anchor_validates_config(self):
        with pytest.raises(ValueError, match="ee_qp_anchor"):
            self._rtc_engine(ee_qp_anchor="bogus")


class TestRateValidation:
    def test_rate_mismatch_rejected(self):
        from lerobot.rollout.inference.factory import create_inference_engine

        cfg = QPSyncInferenceConfig(rate_hz=30.0)
        with pytest.raises(ValueError, match="rate_hz"):
            create_inference_engine(
                cfg,
                policy=None,
                preprocessor=None,
                postprocessor=None,
                robot_wrapper=None,
                hw_features={},
                dataset_features={},
                ordered_action_keys=[],
                task="",
                fps=25.0,
                device="cpu",
            )

    def test_rate_inherits_fps(self):
        cfg = QPSyncInferenceConfig()
        assert cfg.rate_hz is None
        # resolved by the factory before engine construction
        from lerobot.rollout.inference.factory import create_inference_engine

        policy, _ = _make_policy(action_feature_names=JOINT_KEYS, ee_chunk=False)
        engine = create_inference_engine(
            cfg,
            policy=policy,
            preprocessor=_Identity(),
            postprocessor=_Identity(),
            robot_wrapper=JointOnlyRobot(),
            hw_features={},
            dataset_features={ACTION: {"names": JOINT_KEYS}},
            ordered_action_keys=JOINT_KEYS,
            task="t",
            fps=25.0,
            device="cpu",
        )
        assert cfg.rate_hz == 25.0
        assert engine._ee_mode is False


class TestGoldenTicketPolicyCompat:
    """golden_ticket passes ``noise`` to the policy — only flow-matching heads accept it."""

    def test_kwargless_policy_without_ticket_serves(self):
        # ACT-style policy: predict_action_chunk(batch) with no extra kwargs.
        policy, q0 = _make_policy(action_feature_names=JOINT_KEYS, ee_chunk=False)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, action_unit="deg")
        eng = _make_engine(policy, JointOnlyRobot(), cfg)
        served = _serve(eng, q0, n=3)
        assert served.shape == (3, 7)

    def test_kwargless_policy_with_ticket_raises_clear_error(self, tmp_path):
        ticket = tmp_path / "ticket.pt"
        torch.save(torch.zeros(1, 20, 7), ticket)
        policy, q0 = _make_policy(action_feature_names=JOINT_KEYS, ee_chunk=False)
        cfg = QPSyncInferenceConfig(rate_hz=25.0, action_unit="deg", golden_ticket=str(ticket))
        eng = _make_engine(policy, JointOnlyRobot(), cfg)
        with pytest.raises(TypeError, match="golden_ticket"):
            _serve(eng, q0, n=1)
