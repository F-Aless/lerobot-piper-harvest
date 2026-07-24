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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|qp_rtc|qp_sync``.
Adding a new backend requires registering its config subclass and
dispatching it in :func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30
    # Fixed initial noise ("golden ticket") for the flow-matching action head,
    # used in place of freshly sampled Gaussian noise on every inference. Either
    # a path to a single ``.pt`` tensor of shape (1, chunk_size, max_action_dim),
    # or a directory containing exactly one such ``.pt`` file. None = sample
    # random noise each chunk (default behaviour).
    golden_ticket: str | None = None
    # If set, save one .npz per inference chunk to this directory for offline
    # analysis: original / processed / delay / latency / obs_state, plus any
    # subclass-specific fields (e.g. smoother anchor + pre-smoother chunk).
    dump_dir: str | None = None
    # Deterministic delay override (ticks). None = measure wall-clock latency
    # (real hardware). When set to D, the engine treats every inference as a
    # fixed D-tick latency and blocks the merge until exactly D actions have
    # been consumed since inference started, so indexes_diff == real_delay == D
    # regardless of wall-clock speed. Required for a valid RTC test in SIM,
    # where the control loop is not rate-clamped and inference contends with
    # rendering for the GPU (wall-clock delay != sim-tick delay).
    simulated_delay_ticks: int | None = None


@dataclass
class QPSmootherParams:
    """QP chunk-smoother parameters shared by the ``qp_rtc`` and ``qp_sync`` engines.

    Requires the ``chunk-smoother`` extra: ``pip install 'lerobot[chunk-smoother]'``.

    The smoother applies a quadratic-programming pass to each predicted action
    chunk, penalising acceleration/jerk and enforcing a velocity cap and
    (optional) joint range bounds.

    Defaults (``v_max_deg_s=80``, ``lambda_a=40``, ``lambda_j=20``) come from
    an offline parameter sweep on recorded chunks; the harvest-apples
    deployments run with ``lambda_a=lambda_j=80`` (see USAGE_GUIDE.md).
    """

    v_max_deg_s: float = 80.0
    lambda_a: float = 40.0
    lambda_j: float = 20.0
    # Control rate the smoother assumes (Hz). None = inherit the rollout
    # ``--fps``; setting both to different values is rejected at parse time.
    rate_hz: float | None = None
    joint_keys: list[str] = field(default_factory=lambda: [f"joint_{i + 1}.pos" for i in range(6)])
    gripper_key: str | None = "gripper.pos"
    # Column layout of the policy's action chunk, needed in EE mode when the
    # checkpoint does not expose ``action_feature_names`` (SmolVLA/ACT/...).
    # None = take it from the policy config, falling back to the canonical EE
    # layout [ee.x, ee.y, ee.z, ee.roll, ee.pitch, ee.yaw, <gripper_key>].
    policy_action_keys: list[str] | None = None
    # If None, the smoother falls back to ``robot_wrapper.joint_limits_deg``.
    joint_limits_deg: list[tuple[float, float]] | None = None
    # Unit of the actions emitted by ``postprocessor`` (and consumed by the smoother).
    #   - "pct": [-100, 100] joint percentages (piper_full default convention)
    #   - "deg": degrees
    #   - "rad": radians
    # None = auto: take the robot's ``action_angle_unit`` (e.g. piper_full's
    # ``--robot.unit``), falling back to "pct" — a robot/smoother unit mismatch
    # silently corrupts the velocity caps, so prefer leaving this unset.
    action_unit: str | None = None
    # If True, raise instead of letting the first chunk pass without an anchor.
    strict_anchor: bool = False
    # Cartesian (EE) action spaces: smooth each chunk through joint space
    # (projector → batch IK → joint QP → FK back) instead of QP-ing the raw
    # action columns.  Requires a robot exposing ``make_ee_chunk_smoother``
    # (e.g. ``piper_ee``).  None = auto-detect: enabled when ``joint_keys``
    # are absent from the action space and the robot supports it.
    ee_mode: bool | None = None
    # Feed the policy an ``ee.*`` state observation computed via robot FK
    # (joints → pose) in place of the robot's native joint state.  Policies
    # trained on ``*_ee`` datasets need this on a joint-space robot, otherwise
    # they receive joint values normalized with EE statistics (garbage).
    # None = auto: on when EE mode is active and the robot exposes
    # ``ee_pose_from_observation``; explicit True fails loudly when it can't.
    ee_state_obs: bool | None = None


@InferenceEngineConfig.register_subclass("qp_rtc")
@dataclass
class QPRTCInferenceConfig(RTCInferenceConfig, QPSmootherParams):
    """RTC with QP chunk smoothing (async inference + smoothed chunks)."""

    # EE mode: which configuration pins the first executed tick of each
    # smoothed chunk (the inner joint-QP anchor).
    #   - "command":  the last action actually sent to the robot. Keeps the
    #     command stream continuous across merges: when the arm lags the
    #     commands, the measured state sits behind the previous command and
    #     re-anchoring on it would step the command backwards at every merge
    #     (sawtooth). The measured state still seeds FK/projector/IK, so the
    #     QP blends toward the measurement-consistent trajectory within the
    #     velocity caps (distributed correction instead of a step).
    #   - "measured": legacy behaviour — anchor on the observed joint state.
    ee_qp_anchor: str = "command"
    # EE mode: warn when the measured joints lag the commanded ones by more
    # than this at merge time (max over joints, degrees) — tracking-failure
    # signal; the gap is also recorded in the chunk dump as tracking_gap_deg.
    tracking_warn_deg: float = 8.0
    # Feedforward: anticipate the arm's transport delay by N control ticks so the
    # executed command leads the policy (the RTC chunk already carries the
    # lookahead). 0 = off. ~2-3 ≈ the Piper's ~100 ms tracking delay at 30 Hz.
    feedforward_ticks: int = 0


@InferenceEngineConfig.register_subclass("qp_sync")
@dataclass
class QPSyncInferenceConfig(InferenceEngineConfig, QPSmootherParams):
    """Blocking chunked inference with QP chunk smoothing.

    The policy is called inline on the control thread; each chunk is smoothed
    (anchored to the last commanded action) and executed open-loop before the
    next inference. The robot pauses on chunk boundaries for the duration of
    one policy call.
    """

    # How many smoothed steps to execute per chunk before re-inferring.
    # None — execute the full chunk.
    execute_horizon: int | None = None
    # If set, save one .npz per inference chunk to this directory for offline
    # analysis: raw / smoothed chunk plus EE-mode joint trajectories.
    dump_dir: str | None = None
    # Fixed action-head noise ("golden ticket"): path to a .pt tensor (or a dir
    # containing exactly one) used in place of fresh Gaussian noise on every
    # inference. None = sample fresh noise. Mirrors RTCInferenceConfig.golden_ticket.
    golden_ticket: str | None = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    # QP configs: the smoother rate must equal the control-loop rate.  None
    # inherits fps; an explicit mismatch is a config error (wrong velocity
    # caps and per-step clamps), normally already rejected by RolloutConfig.
    if isinstance(config, (QPSyncInferenceConfig, QPRTCInferenceConfig)):
        if config.rate_hz is None:
            config.rate_hz = float(fps)
        elif abs(float(config.rate_hz) - float(fps)) > 1e-6:
            raise ValueError(
                f"--inference.rate_hz={config.rate_hz} differs from --fps={fps}: the QP velocity "
                "caps and the EE projector clamps are per-tick quantities and must use the control "
                "rate. Drop --inference.rate_hz (it inherits fps) or set them equal."
            )
        # action_unit=None → take the robot's declared unit (e.g. piper_full's
        # --robot.unit) so the two can never silently disagree; "pct" fallback
        # for robots that don't declare one.
        if config.action_unit is None:
            # robot_wrapper is normally a ThreadSafeRobot (unwrap via .inner),
            # but tests may pass a bare robot object.
            inner = getattr(robot_wrapper, "inner", robot_wrapper)
            robot_unit = getattr(inner, "action_angle_unit", None)
            config.action_unit = robot_unit or "pct"
            logger.info(
                "QP action_unit resolved to '%s' (%s)",
                config.action_unit,
                "from robot.action_angle_unit" if robot_unit else "default",
            )
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, QPSyncInferenceConfig):
        try:
            from .qp_sync import QPSyncInferenceEngine
        except ImportError as e:
            raise ImportError(
                "The 'qp_sync' inference engine requires the 'chunk-smoother' extra. "
                "Install via: pip install 'lerobot[chunk-smoother]'"
            ) from e
        return QPSyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            qp_config=config,
        )
    # QP variant must come BEFORE the generic RTC check (QPRTCInferenceConfig
    # subclasses RTCInferenceConfig, so isinstance matches both).
    if isinstance(config, QPRTCInferenceConfig):
        try:
            from .qp_rtc import QPSmoothedRTCInferenceEngine
        except ImportError as e:
            raise ImportError(
                "The 'qp_rtc' inference engine requires the 'chunk-smoother' extra. "
                "Install via: pip install 'lerobot[chunk-smoother]'"
            ) from e
        engine_qp = QPSmoothedRTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
            qp_config=config,
            ordered_action_keys=ordered_action_keys,
            simulated_delay_ticks=config.simulated_delay_ticks,
            golden_ticket=config.golden_ticket,
        )
        if config.dump_dir:
            engine_qp.enable_chunk_dump(config.dump_dir)
        return engine_qp
    if isinstance(config, RTCInferenceConfig):
        engine_rtc = RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
            simulated_delay_ticks=config.simulated_delay_ticks,
            golden_ticket=config.golden_ticket,
        )
        if config.dump_dir:
            engine_rtc.enable_chunk_dump(config.dump_dir)
        return engine_rtc
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
