from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("osqp")

from lerobot.policies.chunk_smoother import QPSmoother  # noqa: E402


def _cfg():
    return SimpleNamespace(
        v_max_deg_s=90.0,
        lambda_a=1.0,
        lambda_j=1.0,
        rate_hz=30.0,
        joint_keys=["joint_1.pos"],
        gripper_key=None,
        action_unit="deg",
        strict_anchor=True,
    )


def test_qp_smoother_velocity_anchor_constrains_takeover_velocity():
    smoother = QPSmoother(_cfg(), ["joint_1.pos"])
    chunk = torch.linspace(0.0, 14.0, 8, dtype=torch.float32).unsqueeze(1)

    out = smoother.solve(
        chunk,
        joint_limits_deg=np.array([[-180.0, 180.0]]),
        q_anchor_deg=np.array([10.0]),
        delay=3,
        q_anchor_velocity_deg=np.array([2.0]),
    )

    assert float(out[3, 0]) == pytest.approx(10.0, abs=1e-4)
    assert float(out[3, 0] - out[2, 0]) == pytest.approx(2.0, abs=1e-4)


def test_qp_smoother_fourth_positional_arg_remains_delay():
    smoother = QPSmoother(_cfg(), ["joint_1.pos"])
    chunk = torch.linspace(0.0, 14.0, 8, dtype=torch.float32).unsqueeze(1)

    out = smoother.solve(
        chunk,
        np.array([[-180.0, 180.0]]),
        np.array([10.0]),
        3,
    )

    assert float(out[3, 0]) == pytest.approx(10.0, abs=1e-4)
