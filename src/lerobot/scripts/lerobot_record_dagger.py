#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
DAgger-style human-in-the-loop data collection for Piper + SO Leader 7DoF.

Three-state control loop built around a "policy preview, then human correction"
workflow:

    TELEOP_IDLE  (default)
        leader torque OFF, the leader drives the follower, NO recording
    POLICY
        leader torque ON, the policy drives the follower, the leader mirrors
        the commanded action, NO recording
    TELEOP_REC
        leader torque OFF, the leader drives the follower, RECORDING active

Keyboard controls:

    s   - TELEOP_IDLE  -> POLICY     (start policy preview)
          TELEOP_REC   -> TELEOP_IDLE (save current episode)
          POLICY       -> no-op
    m   - POLICY       -> TELEOP_REC (start recording a new episode)
          otherwise    -> no-op
    c   - POLICY       -> TELEOP_IDLE (abort policy preview)
          TELEOP_REC   -> TELEOP_IDLE (discard current episode)
          TELEOP_IDLE  -> no-op
    ESC - stop everything and (optionally) push dataset to hub

Usage example::

    lerobot-record-dagger \\
        --robot.type=piper_full \\
        --robot.cameras='{image: {type: intelrealsense, serial_number_or_name: "733512070600", width: 640, height: 480, fps: 30}}' \\
        --teleop.type=so101_leader_7dof \\
        --teleop.port=/dev/ttyUSB0 \\
        --policy.path=Faless/xvla-harvest-noee-right-mix \\
        --dataset.repo_id=Faless/dagger-harvest-noee-right-mix \\
        --dataset.single_task="Pick the red apples one by one and place them into the green basket" \\
        --dataset.num_episodes=50 \\
        --dataset.fps=30 \\
        --dataset.push_to_hub=false \\
        --interpolation_multiplier=2 \\
        --display_data=true
"""

import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    is_headless,
    predict_action,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.policies import (
    ActionInterpolator,
    PreTrainedPolicy,
    make_policy,
    make_pre_post_processors,
    make_robot_action,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    piper_full,
    so101_follower_7dof,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_leader_7dof,
    so_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helpers (kept inline to avoid coupling to `lerobot_record` internals)
# ---------------------------------------------------------------------------


def _set_depth_features_to_images(features: dict[str, dict]) -> dict[str, dict]:
    """Keep depth maps as PNG images (uint16), never encode them as videos."""
    for key, feat in features.items():
        if key.startswith(f"{OBS_STR}.images.") and key.endswith("_depth") and feat.get("dtype") == "video":
            feat["dtype"] = "image"
    return features


# ---------------------------------------------------------------------------
# Teleop helpers
# ---------------------------------------------------------------------------


def _teleop_has_motor_control(teleop: Teleoperator) -> bool:
    return all(hasattr(teleop, attr) for attr in ("enable_torque", "disable_torque", "write_goal_positions"))


def _teleop_disable_torque(teleop: Teleoperator) -> None:
    if hasattr(teleop, "disable_torque"):
        teleop.disable_torque()


def _teleop_enable_torque(teleop: Teleoperator) -> None:
    if hasattr(teleop, "enable_torque"):
        teleop.enable_torque()


def _teleop_smooth_move_to(
    teleop: Teleoperator,
    target_pos: dict[str, float],
    duration_s: float = 1.5,
    fps: int = 50,
) -> None:
    """Smoothly move the leader to ``target_pos`` before handing control to the policy."""
    if not _teleop_has_motor_control(teleop):
        logger.warning("Teleop does not expose torque control — cannot mirror robot position")
        return

    _teleop_enable_torque(teleop)
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        t = step / steps
        interp = {}
        for k in current:
            if k in target_pos:
                interp[k] = current[k] * (1 - t) + target_pos[k] * t
            else:
                interp[k] = current[k]
        teleop.write_goal_positions(interp)
        time.sleep(1.0 / fps)


# ---------------------------------------------------------------------------
# Keyboard listener (S / M / C / ESC)
# ---------------------------------------------------------------------------


def _init_dagger_keyboard_listener():
    """Keyboard listener that exposes DAgger state transitions as pending events.

    The main loop consumes the ``requested_*`` booleans; only the main loop knows
    the current state, so enforcement of the "no-op" transitions lives there
    (e.g. pressing S twice from POLICY does nothing).
    """
    events = {
        "requested_start_policy": False,
        "requested_start_record": False,
        "requested_cancel": False,
        "stop_recording": False,
    }

    if is_headless():
        logger.warning("Headless environment detected — keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                if char == "s":
                    events["requested_start_policy"] = True
                elif char == "m":
                    events["requested_start_record"] = True
                elif char == "c":
                    events["requested_cancel"] = True
            elif key == keyboard.Key.esc:
                logger.info("[DAgger] ESC — stop and push dataset")
                events["stop_recording"] = True
        except Exception as e:
            logger.info(f"Key handler error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _print_controls() -> None:
    logger.info(
        "DAgger data collection controls:\n"
        "    s    - start policy (from teleop) / save episode (from recording)\n"
        "    m    - switch from policy to recording teleop\n"
        "    c    - cancel policy or discard current episode\n"
        "    ESC  - stop and push to hub"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DaggerConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: DatasetRecordConfig
    policy: PreTrainedConfig | None = None

    # Control rate multiplier used while the policy drives the robot. Higher
    # values produce a smoother leader mirror with slow policies (e.g. SmolVLA).
    interpolation_multiplier: int = 2

    # Display all cameras on screen via rerun
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False

    # Use vocal synthesis for state transitions
    play_sounds: bool = True

    # Resume on an existing dataset
    resume: bool = False

    # Optional observation-key rename map applied to dataset stats and the
    # policy preprocessor (mirrors the top-level field of ``lerobot-rollout``).
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy is required for DAgger data collection.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# DAgger control loop
# ---------------------------------------------------------------------------

_STATE_IDLE = "TELEOP_IDLE"
_STATE_POLICY = "POLICY"
_STATE_REC = "TELEOP_REC"


@safe_stop_image_writer
def _dagger_loop(
    robot: Robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[torch.Tensor, torch.Tensor],
    dataset: LeRobotDataset,
    events: dict,
    cfg: DaggerConfig,
) -> None:
    """Single long-running loop that handles all DAgger states and transitions."""
    fps = cfg.dataset.fps
    device = get_safe_torch_device(cfg.policy.device)

    # The interpolator runs ONLY in POLICY state so chunked policies (SmolVLA,
    # XVLA) produce a smooth mirror on the leader. TELEOP states always tick at
    # the dataset base fps so recorded frames map 1:1 to ``dataset.fps``.
    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    policy_interval = interpolator.get_control_interval(fps)
    teleop_interval = 1.0 / fps

    action_keys = list(dataset.features[ACTION]["names"])
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    state = _STATE_IDLE
    logger.info(f"[DAgger] Entering state {state}")
    _teleop_disable_torque(teleop)

    episode_frames = 0
    last_robot_action: dict[str, Any] = {}

    while not events["stop_recording"]:
        loop_start = time.perf_counter()

        # --- handle pending state transitions --------------------------------
        if events["requested_start_policy"]:
            events["requested_start_policy"] = False
            if state == _STATE_IDLE:
                log_say("Start policy", cfg.play_sounds)
                obs = robot.get_observation()
                robot_pos = {
                    k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
                }
                _teleop_smooth_move_to(teleop, robot_pos, duration_s=1.5, fps=50)
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                interpolator.reset()
                state = _STATE_POLICY
                logger.info(f"[DAgger] {_STATE_IDLE} -> {_STATE_POLICY}")
            elif state == _STATE_REC:
                log_say(f"Save episode {dataset.num_episodes}", cfg.play_sounds)
                dataset.save_episode()
                episode_frames = 0
                state = _STATE_IDLE
                logger.info(f"[DAgger] {_STATE_REC} -> {_STATE_IDLE} (saved)")
            else:
                logger.info("[DAgger] 's' ignored in POLICY state")

        if events["requested_start_record"]:
            events["requested_start_record"] = False
            if state == _STATE_POLICY:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                _teleop_disable_torque(teleop)
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                interpolator.reset()
                episode_frames = 0
                state = _STATE_REC
                logger.info(f"[DAgger] {_STATE_POLICY} -> {_STATE_REC}")
            else:
                logger.info("[DAgger] 'm' ignored outside POLICY state")

        if events["requested_cancel"]:
            events["requested_cancel"] = False
            if state == _STATE_POLICY:
                log_say("Abort policy", cfg.play_sounds)
                _teleop_disable_torque(teleop)
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                interpolator.reset()
                state = _STATE_IDLE
                logger.info(f"[DAgger] {_STATE_POLICY} -> {_STATE_IDLE} (aborted)")
            elif state == _STATE_REC:
                log_say("Discard episode", cfg.play_sounds)
                dataset.clear_episode_buffer()
                episode_frames = 0
                state = _STATE_IDLE
                logger.info(f"[DAgger] {_STATE_REC} -> {_STATE_IDLE} (discarded)")
            else:
                logger.info("[DAgger] 'c' ignored in TELEOP_IDLE state")

        # --- observe ---------------------------------------------------------
        obs = robot.get_observation()
        obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
        obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        robot_action: dict[str, Any] = {}

        # --- act -------------------------------------------------------------
        if state == _STATE_POLICY:
            if interpolator.needs_new_action():
                action_tensor = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                )
                robot_action = make_robot_action(action_tensor, dataset.features)
                stacked = torch.tensor([robot_action[k] for k in action_keys])
                interpolator.add(stacked)

            interp_action = interpolator.get()
            if interp_action is not None:
                robot_action = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                robot.send_action(robot_action)
                if hasattr(teleop, "write_goal_positions"):
                    teleop.write_goal_positions(robot_action)
                last_robot_action = robot_action

        else:
            # TELEOP_IDLE or TELEOP_REC — leader drives the follower, torque OFF.
            robot_action = teleop.get_action()
            robot.send_action(robot_action)
            last_robot_action = robot_action

            if state == _STATE_REC:
                action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                dataset.add_frame(frame)
                episode_frames += 1

        if cfg.display_data and last_robot_action:
            log_rerun_data(
                observation=obs_filtered,
                action=last_robot_action,
                compress_images=cfg.display_compressed_images,
            )

        # --- pace loop -------------------------------------------------------
        target_interval = policy_interval if state == _STATE_POLICY else teleop_interval
        dt = time.perf_counter() - loop_start
        sleep_time = target_interval - dt
        if sleep_time > 0:
            precise_sleep(sleep_time)

    if state == _STATE_REC and episode_frames > 0:
        logger.warning("[DAgger] Stop requested while recording — discarding in-progress episode")
        dataset.clear_episode_buffer()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_dataset_features(
    robot: Robot,
    teleop_action_processor,
    robot_observation_processor,
    use_videos: bool,
) -> dict:
    """Build dataset features from robot hardware features.

    Keeps ``.pos`` and ``.tau`` floats for ``observation.state`` (mirroring the
    rollout context filter — e.g. SmolVLA on piper_full expects the 8-D state
    with ``gripper.tau``) and tuple-typed image entries.
    """
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    all_observation_features = robot.observation_features
    state_joint_names = [
        key
        for key, value in all_observation_features.items()
        if (key.endswith(".pos") or key.endswith(".tau")) and value is float
    ]
    observation_features_hw = {name: all_observation_features[name] for name in state_joint_names}
    for key, value in all_observation_features.items():
        if isinstance(value, tuple):
            observation_features_hw[key] = value

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=use_videos,
        ),
    )
    return _set_depth_features_to_images(dataset_features)


@parser.wrap()
def dagger_collect(cfg: DaggerConfig) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="dagger_collection", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    if not _teleop_has_motor_control(teleop):
        raise TypeError(
            f"Teleop '{type(teleop).__name__}' does not expose enable_torque/disable_torque/"
            "write_goal_positions; DAgger needs an active-motor leader (e.g. so101_leader_7dof)."
        )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    del robot_action_processor  # unused: we send raw teleop/policy actions to the robot

    dataset_features = _build_dataset_features(
        robot=robot,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
        use_videos=cfg.dataset.video,
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            # The plain constructor opens the dataset read-only (no writer);
            # ``resume`` reloads the metadata AND attaches a writer.
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        robot.connect()
        teleop.connect()

        listener, events = _init_dagger_keyboard_listener()
        _print_controls()
        logger.info(f"  Policy:  {cfg.policy.pretrained_path}")
        logger.info(f"  Task:    {cfg.dataset.single_task}")
        logger.info(f"  FPS:     {cfg.dataset.fps} (interp x{cfg.interpolation_multiplier})")

        with VideoEncodingManager(dataset):
            _dagger_loop(
                robot=robot,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                events=events,
                cfg=cfg,
            )

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        with contextlib.suppress(Exception):
            _teleop_disable_torque(teleop)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    register_third_party_plugins()
    dagger_collect()


if __name__ == "__main__":
    main()
