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

"""RTC inference engine with a QP chunk smoother bolted on.

Subclasses :class:`RTCInferenceEngine` and overrides ``_post_process_chunk``
to run the chunk through :class:`QPSmoother` before it is merged into the
``ActionQueue``.

Anchor strategy (most-to-least preferred):

1. The last action commanded to the robot (``notify_last_commanded_action``).
2. The latest observation, looking up the canonical joint keys directly.
3. None — the first chunk is allowed to start unanchored (loose).  In
   ``strict_anchor=True`` mode this raises instead.

EE mode separates the two roles (``QPRTCInferenceConfig.ee_qp_anchor``): the
measured state seeds FK/projector/IK (the conversion evolves from where the
arm is), while the QP anchor pins the chunk to the last *commanded* action so
the command stream stays continuous across merges — re-anchoring on the
measured state would step the command backwards by the tracking error at
every merge (sawtooth).
"""

from __future__ import annotations

import logging
from threading import Event, Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.policies.chunk_smoother import QPSmoother
from lerobot.policies.chunk_smoother.anchor import (
    QPAnchorResolver,
    ee_state_names,
    resolve_ee_mode,
    resolve_ee_state_obs,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE

from ..robot_wrapper import ThreadSafeRobot
from .qp_sync import _canonical_ee_keys, _resolve_policy_action_keys
from .rtc import RTCInferenceEngine

if TYPE_CHECKING:
    from .factory import QPRTCInferenceConfig

logger = logging.getLogger(__name__)


class QPSmoothedRTCInferenceEngine(RTCInferenceEngine):
    """RTC + QP chunk smoother. Drop-in replacement for ``RTCInferenceEngine``."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        *,
        qp_config: QPRTCInferenceConfig,
        ordered_action_keys: list[str],
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
        simulated_delay_ticks: int | None = None,
        golden_ticket: str | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=rtc_config,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=rtc_queue_threshold,
            shutdown_event=shutdown_event,
            simulated_delay_ticks=simulated_delay_ticks,
            golden_ticket=golden_ticket,
            feedforward_ticks=qp_config.feedforward_ticks,
        )

        self._qp_config = qp_config
        self._ordered_action_keys = list(ordered_action_keys)
        ee_factory = getattr(robot_wrapper, "make_ee_chunk_smoother", None)
        self._policy_action_keys = _resolve_policy_action_keys(qp_config, policy)
        self._ee_mode = resolve_ee_mode(
            ee_mode=qp_config.ee_mode,
            joint_keys=qp_config.joint_keys,
            ordered_action_keys=self._ordered_action_keys,
            policy_action_names=self._policy_action_keys,
            ee_factory_available=ee_factory is not None,
            engine_name="qp_rtc",
        )
        if self._ee_mode and self._policy_action_keys is None:
            self._policy_action_keys = _canonical_ee_keys(qp_config, "qp_rtc")
        elif qp_config.ee_mode is None and not self._ee_mode and self._policy_action_keys is None:
            logger.warning(
                "qp_rtc: the policy config does not expose action_feature_names — assuming a "
                "JOINT-space policy (this robot has no EE conversion support)."
            )
        # EE mode: QP-anchor source ("command" = continuity with the last sent
        # action; "measured" = legacy observation anchor) + tracking monitor.
        self._ee_anchor_source = str(getattr(qp_config, "ee_qp_anchor", "command"))
        if self._ee_anchor_source not in ("command", "measured"):
            raise ValueError(f"ee_qp_anchor must be 'command' or 'measured', got {self._ee_anchor_source!r}")
        self._tracking_warn_deg = float(getattr(qp_config, "tracking_warn_deg", 8.0))
        self._warned_cmd_anchor_unavailable = False
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
            engine_name="qp_rtc",
        )
        if self._ee_state_obs:
            state_ft = dict(self._hw_features[OBS_STATE])
            names = ee_state_names(list(state_ft.get("names") or []), qp_config.joint_keys)
            state_ft["names"] = names
            state_ft["shape"] = (len(names),)
            self._policy_hw_features = {**self._hw_features, OBS_STATE: state_ft}
            logger.info("qp_rtc: EE-state FK active — observation.state → %s", names)

        self._prev_commanded: torch.Tensor | None = None
        self._last_commanded: torch.Tensor | None = None
        self._last_commanded_lock = Lock()

        logger.info(
            "QPSmoothedRTCInferenceEngine initialized (v_max=%.1f deg/s, lambda_a=%.1f, lambda_j=%.1f, rate=%.1f Hz, unit=%s)",
            qp_config.v_max_deg_s,
            qp_config.lambda_a,
            qp_config.lambda_j,
            qp_config.rate_hz,
            qp_config.action_unit,
        )

    def _obs_for_policy(self, obs: dict) -> dict:
        """Augment the raw observation with FK-computed ``ee.*`` keys (EE-state mode)."""
        if not self._ee_state_obs:
            return obs
        ee = self._robot.ee_pose_from_observation(obs)
        if ee is None:
            raise RuntimeError(
                "qp_rtc: ee_state_obs is active but the observation lacks joint values for FK."
            )
        return {**obs, **ee}

    # ------------------------------------------------------------------
    # InferenceEngine ABC override
    # ------------------------------------------------------------------

    def notify_last_commanded_action(self, action: torch.Tensor) -> None:
        with self._last_commanded_lock:
            self._prev_commanded = self._last_commanded
            self._last_commanded = action.detach().cpu().clone()

    def reset(self) -> None:
        super().reset()
        # New episode: the previous episode's last command must not anchor
        # the first chunk.
        with self._last_commanded_lock:
            self._prev_commanded = None
            self._last_commanded = None
        if self._ee_smoother is not None:
            self._ee_smoother.reset()

    # ------------------------------------------------------------------
    # RTC subclass override
    # ------------------------------------------------------------------

    def _post_process_chunk(
        self,
        original: torch.Tensor,
        processed: torch.Tensor,
        delay: int,
    ) -> torch.Tensor:
        del original  # the smoothers only consume the postprocessor output

        if self._ee_mode:
            # Cartesian chunk → joint space → QP → back to the execution space.
            # Two distinct joint references (see QPRTCInferenceConfig.ee_qp_anchor):
            #   * q_meas — freshest observation: seeds FK/projector/IK so the
            #     conversion evolves from where the arm actually is;
            #   * q_cmd  — last action sent: pins the QP anchor so the command
            #     stream stays continuous across merges (no backward step when
            #     the arm lags the commands).
            with self._obs_lock:
                obs = self._obs_holder.get("obs") if self._obs_holder else None
            q_fn = getattr(self._robot, "ee_anchor_q_from_observation", None) or getattr(
                self._robot, "urdf_q_from_observation", None
            )
            q_meas = q_fn(obs) if (q_fn is not None and obs is not None) else None
            q_cmd = self._ee_command_anchor_q() if self._ee_anchor_source == "command" else None
            tracking_gap_deg = None
            if q_cmd is not None and q_meas is not None:
                tracking_gap_deg = float(np.max(np.abs(np.degrees(np.asarray(q_cmd) - np.asarray(q_meas)))))
                if tracking_gap_deg > self._tracking_warn_deg:
                    logger.warning(
                        "qp_rtc: measured joints lag the commanded ones by %.1f deg at merge "
                        "(threshold %.1f deg) — tracking failure: the arm cannot follow the "
                        "commanded speed (driver limit / gains / v_max too high?).",
                        tracking_gap_deg,
                        self._tracking_warn_deg,
                    )
            smoothed, diagnostics = self._ee_smoother.solve(
                processed, q_meas, delay=int(delay), q_cmd_rad=q_cmd
            )
            if self._dump_dir is not None:
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
                        self._pending_dump[key] = np.asarray(diagnostics[key])
                # q_anchor_deg = the QP anchor actually used (command-continuity
                # point when available); the measured state and the gap between
                # the two are dumped alongside for offline tracking analysis.
                anchor_used = q_cmd if q_cmd is not None else q_meas
                if anchor_used is not None:
                    self._pending_dump["q_anchor_deg"] = np.degrees(np.asarray(anchor_used))
                if q_meas is not None:
                    self._pending_dump["q_meas_anchor_deg"] = np.degrees(np.asarray(q_meas))
                if q_cmd is not None:
                    self._pending_dump["q_cmd_anchor_deg"] = np.degrees(np.asarray(q_cmd))
                if tracking_gap_deg is not None:
                    self._pending_dump["tracking_gap_deg"] = tracking_gap_deg
                self._pending_dump["smoother_delay"] = int(delay)
                self._pending_dump["pre_smoother"] = processed.detach().cpu().numpy()
            return smoothed

        q_anchor_deg, q_anchor_velocity_deg = self._compute_anchor_deg_and_velocity(delay)
        joint_limits_deg = self._resolve_joint_limits_deg()
        smoothed = self._smoother.solve(
            processed,
            joint_limits_deg=joint_limits_deg,
            q_anchor_deg=q_anchor_deg,
            q_anchor_velocity_deg=q_anchor_velocity_deg,
            delay=int(delay),
        )
        # Stash extra fields for the parent RTC debug dump (no-op if disabled).
        if self._dump_dir is not None:
            if q_anchor_deg is not None:
                self._pending_dump["q_anchor_deg"] = np.asarray(q_anchor_deg)
            if q_anchor_velocity_deg is not None:
                self._pending_dump["q_anchor_velocity_deg"] = np.asarray(q_anchor_velocity_deg)
            if joint_limits_deg is not None:
                self._pending_dump["joint_limits_deg"] = np.asarray(joint_limits_deg)
            self._pending_dump["smoother_delay"] = int(delay)
            # Smoother INPUT (post-postprocessor, pre-smoothing): lets the QP pass
            # be reproduced offline with different params for tuning comparisons.
            self._pending_dump["pre_smoother"] = processed.detach().cpu().numpy()
        return smoothed

    # ------------------------------------------------------------------
    # Anchor + limit resolution
    # ------------------------------------------------------------------

    def _ee_command_anchor_q(self) -> np.ndarray | None:
        """Last commanded action → EE-frame joint radians (command anchor).

        The rollout loop notifies the engine of every action it sends, in
        ``ordered_action_keys`` space.  For joint-output robots those columns
        are the joint actions in the same keys/unit convention as the
        observation, so the robot's observation→anchor converter applies to
        them unchanged.  Returns None (→ measured-state fallback in
        ``solve``) when no action was sent yet or the robot cannot map the
        commanded action to joints (e.g. Cartesian-output robots).
        """
        with self._last_commanded_lock:
            last = self._last_commanded
        conv = getattr(self._robot, "ee_anchor_q_from_observation", None)
        q = None
        if last is not None and conv is not None:
            vals = last.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if vals.shape[0] == len(self._ordered_action_keys):
                q = conv({k: float(v) for k, v in zip(self._ordered_action_keys, vals, strict=True)})
        if q is None and last is not None and not self._warned_cmd_anchor_unavailable:
            self._warned_cmd_anchor_unavailable = True
            logger.warning(
                "qp_rtc: ee_qp_anchor='command' but the commanded action cannot be mapped "
                "to joints on this robot — falling back to the measured-state anchor."
            )
        return q

    def _compute_anchor_deg(self) -> np.ndarray | None:
        """Resolve the joint-space anchor for the QP, in degrees.

        Priority:
          1. last commanded action (cached via ``notify_last_commanded_action``);
          2. latest observation joint state (looking up ``joint_keys`` by name);
          3. None — first chunk runs unanchored (or raises in strict mode).
        """
        # --- P1: last commanded ------------------------------------------------
        with self._last_commanded_lock:
            last = self._last_commanded.clone() if self._last_commanded is not None else None
        anchor = self._anchor.anchor_from_action(last)
        if anchor is not None:
            return anchor

        # --- P2: observation lookup --------------------------------------------
        with self._obs_lock:
            obs = self._obs_holder.get("obs") if self._obs_holder else None
        anchor = self._anchor.anchor_from_obs(obs)
        if anchor is not None:
            return anchor

        # --- P3: no anchor available ------------------------------------------
        if self._qp_config.strict_anchor:
            raise RuntimeError(
                "QPSmoothedRTCInferenceEngine: strict_anchor=True but no anchor source "
                "available (no last_commanded_action notification, and observation does "
                f"not contain all of {self._qp_config.joint_keys}). "
                "Either disable strict_anchor or ensure the observation exposes joint .pos values."
            )
        return None

    def _compute_anchor_deg_and_velocity(self, delay: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Resolve QP position and optional velocity anchors, in degrees.

        In RTC joint mode the first executed tick of a new chunk is
        ``y[delay]`` because earlier ticks are discarded by ``ActionQueue``.
        The legacy position-only anchor pins that tick to the last command,
        which preserves C0 continuity but can zero or flip the carried velocity
        at chunk seams.  When the previous two commands are available, anchor
        ``y[delay]`` to one extrapolated tick and constrain
        ``y[delay] - y[delay - 1]`` to the previous command velocity.  That
        gives the QP a C1-continuous seam while keeping the same velocity cap.
        """
        if int(delay) <= 0:
            return self._compute_anchor_deg(), None

        with self._last_commanded_lock:
            prev = self._prev_commanded.clone() if self._prev_commanded is not None else None
            last = self._last_commanded.clone() if self._last_commanded is not None else None

        last_deg = self._anchor.anchor_from_action(last)
        if last_deg is None:
            return self._compute_anchor_deg(), None

        prev_deg = self._anchor.anchor_from_action(prev)
        if prev_deg is None:
            return last_deg, None

        rate_step = float(self._qp_config.v_max_deg_s) / float(self._qp_config.rate_hz)
        velocity = np.clip(last_deg - prev_deg, -rate_step, rate_step)
        next_anchor = last_deg + velocity

        limits = self._resolve_joint_limits_deg()
        if limits is not None:
            next_anchor = np.clip(next_anchor, limits[:, 0], limits[:, 1])
            velocity = next_anchor - last_deg

        return next_anchor, velocity

    def _resolve_joint_limits_deg(self) -> np.ndarray | None:
        """Pull joint limits, in degrees, from config or robot, as a (N, 2) array."""
        return self._anchor.resolve_joint_limits_deg()
