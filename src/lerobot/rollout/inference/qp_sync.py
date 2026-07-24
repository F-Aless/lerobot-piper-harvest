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

"""Synchronous chunked inference with QP smoothing.

Blocking counterpart of :class:`QPSmoothedRTCInferenceEngine`: the policy is
called inline on the control thread via ``predict_action_chunk``; the chunk is
run through :class:`QPSmoother` (anchored to the last commanded action so
consecutive chunks join without velocity spikes) and served one action per
tick from a local FIFO.  When the FIFO drains, the next ``get_action`` call
blocks on a fresh inference.

Compared to ``qp_rtc`` there is no background thread and no chunk merging —
the robot simply pauses on chunk boundaries for the duration of one policy
call.  Use this when you want deterministic open-loop chunk execution (e.g.
to reproduce offline-tuned QP behaviour) or when the policy is fast enough
that inference pauses are acceptable.

Anchor strategy (most-to-least preferred), mirroring ``qp_rtc``:

1. The last action commanded to the robot (``notify_last_commanded_action``).
2. The latest observation (``notify_observation``), looking up the canonical
   joint keys directly.
3. None — the first chunk runs unanchored (loose).  In ``strict_anchor=True``
   mode this raises instead.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.policies.chunk_smoother import QPSmoother
from lerobot.policies.chunk_smoother.anchor import (
    EE_POSE_KEYS,
    QPAnchorResolver,
    ee_state_names,
    resolve_ee_mode,
    resolve_ee_state_obs,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

if TYPE_CHECKING:
    from .factory import QPSyncInferenceConfig

logger = logging.getLogger(__name__)


def _resolve_policy_action_keys(qp_config, policy) -> list[str] | None:
    """Policy chunk column layout: explicit config > policy config > None.

    Most policies (SmolVLA, ACT, Diffusion) do not persist
    ``action_feature_names`` in their config — only the pi0 family does — so
    the explicit ``--inference.policy_action_keys`` override and the canonical
    EE fallback exist for them.
    """
    names = qp_config.policy_action_keys or getattr(
        getattr(policy, "config", None), "action_feature_names", None
    )
    return list(names) if names else None


def _canonical_ee_keys(qp_config, engine_name: str) -> list[str]:
    """Canonical EE chunk layout assumed when the checkpoint exposes no names."""
    canonical = [*EE_POSE_KEYS] + ([qp_config.gripper_key] if qp_config.gripper_key else [])
    logger.warning(
        "%s: EE mode without action_feature_names in the policy config — assuming the canonical "
        "EE chunk layout %s. Override with --inference.policy_action_keys if the checkpoint "
        "differs (a width mismatch will fail loudly at the first chunk).",
        engine_name,
        canonical,
    )
    return canonical


class QPSyncInferenceEngine(InferenceEngine):
    """Inline chunked inference + QP chunk smoothing."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        *,
        qp_config: QPSyncInferenceConfig,
    ) -> None:
        if not hasattr(policy, "predict_action_chunk"):
            raise ValueError(
                f"Policy {type(policy).__name__} does not expose predict_action_chunk; "
                "the 'qp_sync' engine requires a chunk-producing policy "
                "(ACT, Pi0, SmolVLA, ...). Use --inference.type=sync instead."
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._dataset_features = dataset_features
        self._ordered_action_keys = list(ordered_action_keys)
        self._task = task
        self._device = torch.device(device or "cpu")

        self._qp_config = qp_config
        # Fixed action-head noise ("golden ticket"): None = sample fresh noise each
        # inference (default). Mirrors the rtc/qp_rtc engines.
        if getattr(qp_config, "golden_ticket", None):
            from lerobot.rollout.inference.rtc import _load_golden_ticket

            self._golden_noise = _load_golden_ticket(qp_config.golden_ticket, self._device)
        else:
            self._golden_noise = None
        ee_factory = getattr(robot_wrapper, "make_ee_chunk_smoother", None)
        self._policy_action_keys = _resolve_policy_action_keys(qp_config, policy)
        self._ee_mode = resolve_ee_mode(
            ee_mode=qp_config.ee_mode,
            joint_keys=qp_config.joint_keys,
            ordered_action_keys=self._ordered_action_keys,
            policy_action_names=self._policy_action_keys,
            ee_factory_available=ee_factory is not None,
            engine_name="qp_sync",
        )
        if self._ee_mode and self._policy_action_keys is None:
            self._policy_action_keys = _canonical_ee_keys(qp_config, "qp_sync")
        elif qp_config.ee_mode is None and not self._ee_mode and self._policy_action_keys is None:
            logger.warning(
                "qp_sync: the policy config does not expose action_feature_names — assuming a "
                "JOINT-space policy (this robot has no EE conversion support)."
            )
        if self._ee_mode:
            self._smoother = None
            self._anchor = None
            self._ee_smoother = ee_factory(
                policy_action_keys=self._policy_action_keys,
                output_action_keys=self._ordered_action_keys,
                joint_keys=qp_config.joint_keys,
                gripper_key=qp_config.gripper_key,
                v_max_deg_s=qp_config.v_max_deg_s,
                lambda_a=qp_config.lambda_a,
                lambda_j=qp_config.lambda_j,
                rate_hz=qp_config.rate_hz,
                strict_anchor=qp_config.strict_anchor,
            )
        else:
            self._ee_smoother = None
            self._smoother = QPSmoother(qp_config, self._ordered_action_keys)
            self._anchor = QPAnchorResolver(qp_config, self._ordered_action_keys, robot_wrapper)

        # EE-state observation: feed the policy an FK-computed ee.* state in
        # place of the robot's joint state (policies trained on *_ee datasets).
        self._ee_state_obs = resolve_ee_state_obs(
            ee_state_obs=qp_config.ee_state_obs,
            ee_mode=self._ee_mode,
            fk_available=getattr(robot_wrapper, "ee_pose_from_observation", None) is not None,
            engine_name="qp_sync",
        )
        self._ee_state_names: list[str] | None = None
        if self._ee_state_obs:
            orig_names = list((dataset_features.get(OBS_STATE) or {}).get("names") or [])
            self._ee_state_names = ee_state_names(orig_names, qp_config.joint_keys)
            logger.info("qp_sync: EE-state FK active — observation.state → %s", self._ee_state_names)

        self._fifo: deque[torch.Tensor] = deque()
        self._last_commanded: torch.Tensor | None = None
        self._last_obs: dict | None = None
        self._state_lock = Lock()
        # EE mode: joint configuration at the last executed tick of the
        # previous smoothed chunk — exact warm-start anchor for the next one.
        self._last_q_anchor = None

        self._dump_dir: Path | None = None
        self._dump_index = 0
        if qp_config.dump_dir:
            self._dump_dir = Path(qp_config.dump_dir)
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info("qp_sync chunk dump enabled → %s", self._dump_dir)

        logger.info(
            "QPSyncInferenceEngine initialized (v_max=%.1f deg/s, lambda_a=%.1f, lambda_j=%.1f, "
            "rate=%.1f Hz, unit=%s, execute_horizon=%s, ee_mode=%s)",
            qp_config.v_max_deg_s,
            qp_config.lambda_a,
            qp_config.lambda_j,
            qp_config.rate_hz,
            qp_config.action_unit,
            qp_config.execute_horizon,
            self._ee_mode,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("QPSyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        logger.info("QPSyncInferenceEngine stopped")

    def reset(self) -> None:
        """Clear episode-scoped state: FIFO, anchors, policy and processors."""
        logger.info("Resetting qp_sync inference state (policy + processors + FIFO)")
        self._fifo.clear()
        with self._state_lock:
            self._last_commanded = None
            self._last_obs = None
        self._last_q_anchor = None
        if self._ee_smoother is not None:
            self._ee_smoother.reset()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def notify_observation(self, obs: dict) -> None:
        with self._state_lock:
            self._last_obs = obs

    def notify_last_commanded_action(self, action: torch.Tensor) -> None:
        with self._state_lock:
            self._last_commanded = action.detach().cpu().clone()

    # ------------------------------------------------------------------
    # Action production
    # ------------------------------------------------------------------

    def _ee_state_vector(self) -> np.ndarray:
        """FK-recomputed ``observation.state`` row in the policy's EE layout."""
        with self._state_lock:
            obs = self._last_obs
        if obs is None:
            raise RuntimeError("qp_sync: ee_state_obs is active but no observation was notified yet.")
        ee = self._robot.ee_pose_from_observation(obs)
        if ee is None:
            raise RuntimeError(
                "qp_sync: ee_state_obs is active but the observation lacks joint values for FK."
            )
        merged = {**obs, **ee}
        try:
            return np.array([float(merged[k]) for k in self._ee_state_names], dtype=np.float32)
        except KeyError as e:
            raise RuntimeError(f"qp_sync: missing state key for the EE observation: {e}") from e

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next smoothed action; run a new chunk inference if drained."""
        if not self._fifo:
            if obs_frame is None:
                return None
            self._refill(obs_frame)
        if not self._fifo:
            return None
        return self._fifo.popleft()

    def _refill(self, obs_frame: dict) -> None:
        """Run one chunk inference, smooth it, and fill the FIFO."""
        observation = copy(obs_frame)
        if self._ee_state_obs:
            observation[OBS_STATE] = self._ee_state_vector()
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot.robot_type
            )
            observation["task"] = [self._task]
            preprocessed = self._preprocessor(observation)
            # Pass ``noise`` only when a golden ticket is loaded: not every
            # chunking policy accepts the kwarg (e.g. ACT's predict_action_chunk
            # takes no extra arguments — only flow-matching heads sample noise).
            if self._golden_noise is not None:
                try:
                    actions = self._policy.predict_action_chunk(preprocessed, noise=self._golden_noise)
                except TypeError as e:
                    raise TypeError(
                        f"--inference.golden_ticket requires a policy whose predict_action_chunk "
                        f"accepts a 'noise' argument (flow-matching heads such as SmolVLA/pi0). "
                        f"{type(self._policy).__name__} does not — drop the golden_ticket flag."
                    ) from e
            else:
                actions = self._policy.predict_action_chunk(preprocessed)
            processed = self._postprocessor(actions).squeeze(0)

        if processed.dim() != 2:
            raise ValueError(
                f"qp_sync expects a (T, A) action chunk after postprocessing, got {tuple(processed.shape)}"
            )

        horizon = self._qp_config.execute_horizon
        n_exec = processed.shape[0] if horizon is None else max(1, min(int(horizon), processed.shape[0]))

        diagnostics = None
        if self._ee_mode:
            # The chunk stays in the policy's own action order — the EE
            # smoother resolves its columns against policy_action_keys and
            # emits rows in the execution (ordered_action_keys) space.
            chunk = processed.cpu()
            q0 = self._last_q_anchor
            if q0 is None:
                with self._state_lock:
                    obs = self._last_obs
                q_fn = getattr(self._robot, "ee_anchor_q_from_observation", None) or getattr(
                    self._robot, "urdf_q_from_observation", None
                )
                q0 = q_fn(obs) if (q_fn is not None and obs is not None) else None
            smoothed, diagnostics = self._ee_smoother.solve(chunk, q0, delay=0)
            self._last_q_anchor = diagnostics["q_smooth_rad"][n_exec - 1]
        else:
            # Reorder every row into canonical ordered_action_keys space
            # BEFORE smoothing — QPSmoother resolves its joint columns
            # against that order.
            rows = []
            for t in range(processed.shape[0]):
                action_dict = make_robot_action(processed[t].cpu(), self._dataset_features)
                rows.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))
            chunk = torch.stack(rows, dim=0)
            smoothed = self._smoother.solve(
                chunk,
                joint_limits_deg=self._anchor.resolve_joint_limits_deg(),
                q_anchor_deg=self._compute_anchor_deg(),
                delay=0,
            )

        for t in range(n_exec):
            self._fifo.append(smoothed[t].clone())

        if self._dump_dir is not None:
            self._write_chunk_dump(chunk, smoothed, diagnostics)

    def _write_chunk_dump(
        self, chunk: torch.Tensor, smoothed: torch.Tensor, diagnostics: dict | None
    ) -> None:
        """Save one .npz per inference: raw vs smoothed chunk (+ EE joint data)."""
        try:
            data = {
                "raw": chunk.numpy(),
                "smoothed": smoothed.cpu().numpy(),
            }
            if diagnostics is not None:
                for key in (
                    "q_raw_deg",
                    "q_smooth_deg",
                    "n_fallback",
                    "n_scaled",
                    "n_capped",
                    "n_last_valid",
                    "max_ik_pos_err",
                    "max_condition_number",
                    "max_null_space_target_err",
                ):
                    if key in diagnostics:
                        data[key] = np.asarray(diagnostics[key])
            path = self._dump_dir / f"chunk_{self._dump_index:06d}.npz"
            self._dump_index += 1
            np.savez(path, **data)
        except Exception:
            logger.debug("qp_sync chunk dump failed", exc_info=True)

    # ------------------------------------------------------------------
    # Anchor resolution (same priority as qp_rtc)
    # ------------------------------------------------------------------

    def _compute_anchor_deg(self):
        with self._state_lock:
            last = self._last_commanded.clone() if self._last_commanded is not None else None
            obs = self._last_obs

        anchor = self._anchor.anchor_from_action(last)
        if anchor is not None:
            return anchor

        anchor = self._anchor.anchor_from_obs(obs)
        if anchor is not None:
            return anchor

        if self._qp_config.strict_anchor:
            raise RuntimeError(
                "QPSyncInferenceEngine: strict_anchor=True but no anchor source available "
                "(no last_commanded_action notification, and observation does not contain "
                f"all of {self._qp_config.joint_keys}). Either disable strict_anchor or "
                "ensure the observation exposes joint .pos values."
            )
        return None
