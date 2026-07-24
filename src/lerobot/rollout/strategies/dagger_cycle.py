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

"""Single-key DAgger cycle strategy (Piper + SO-101 7-DoF leader).

A manual-episode human-in-the-loop loop built on the shared rollout machinery
(qp_sync inference engine, dataset writer, smooth handover helpers).  Designed
to pair with ``--inference.type=qp_sync``: the policy runs synchronously and is
QP smoothed, and because dataset frames are stamped by index (not wall clock)
the on-chunk-boundary inference pauses are excluded from the fixed-fps dataset.

State machine
-------------

    STANDBY
        idle at home, nothing recorded, waiting for ``save_key`` to start.
    AUTONOMOUS
        qp-smoothed policy drives the follower; leader torque OFF; every
        executed action is recorded (``intervention=False``).
    PAUSED
        policy stopped; the leader arm is driven to the follower's pose and
        holds there (torque ON); nothing recorded.
    CORRECTING
        leader torque OFF — the operator drives the follower by hand; every
        frame is recorded (``intervention=True``).

Controls (keyboard)
-------------------

    cycle_key (default space)
        AUTONOMOUS -> PAUSED -> CORRECTING -> AUTONOMOUS  (one ring)
    save_key (default s)
        STANDBY    -> start the policy (begin an episode)
        recording  -> save the episode, return home, STANDBY
    discard_key (default c)
        recording  -> discard the episode, return home, STANDBY
    ESC
        stop the session (an in-progress episode is discarded).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.common.control_utils import (
    is_headless,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import DAggerCycleStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

_STANDBY = "STANDBY"
_AUTONOMOUS = "AUTONOMOUS"
_PAUSED = "PAUSED"
_CORRECTING = "CORRECTING"
_RECORDING_STATES = (_AUTONOMOUS, _PAUSED, _CORRECTING)


# ---------------------------------------------------------------------------
# Keyboard events (latest-press-wins, consumed by the control loop)
# ---------------------------------------------------------------------------


class _CycleEvents:
    """Thread-safe single-slot request from the keyboard thread to the loop.

    Only the most recent unconsumed press matters; the main loop owns the
    current state and enforces which transitions are legal, so the listener
    just records intent.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: str | None = None  # "cycle" | "save" | "discard"
        self.stop = Event()

    def request(self, action: str) -> None:
        with self._lock:
            self._pending = action

    def consume(self) -> str | None:
        with self._lock:
            action = self._pending
            self._pending = None
            return action


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class DAggerCycleStrategy(RolloutStrategy):
    """Single-key, manual-episode DAgger loop (see module docstring)."""

    config: DAggerCycleStrategyConfig

    def __init__(self, config: DAggerCycleStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events = _CycleEvents()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        cfg = ctx.runtime.cfg
        # The dataset stamps timestamps as frame_index / dataset.fps, which is a
        # SEPARATE field from the control-loop --fps. A mismatch silently records
        # a trajectory whose claimed rate differs from the real control rate.
        if cfg.dataset is not None and abs(float(cfg.dataset.fps) - float(cfg.fps)) > 1e-6:
            logger.warning(
                "dataset.fps=%s != control --fps=%s: recorded timestamps (frame_index/%s) will NOT "
                "match the real %s Hz control rate. Pass --dataset.fps=%s so the metadata is correct.",
                cfg.dataset.fps,
                cfg.fps,
                cfg.dataset.fps,
                cfg.fps,
                cfg.fps,
            )
        # Streaming encoding is unsafe to discard mid-episode here: there is a
        # frame-less PAUSED gap and the encoder is cancelled from clear_episode_buffer
        # while still mid-container. Manual episodes don't need it.
        if cfg.dataset is not None and getattr(cfg.dataset, "streaming_encoding", False):
            logger.warning(
                "dataset.streaming_encoding=True with dagger_cycle: discarding an episode cancels the "
                "streaming encoder mid-stream and can hang/abort the process. Prefer "
                "--dataset.streaming_encoding=false for manual save/discard episodes."
            )
        teleop = ctx.hardware.teleop
        if not teleop_supports_feedback(teleop):
            logger.warning(
                "Teleop '%s' has no torque control (enable_torque/disable_torque/feedback). "
                "Leader alignment on AUTONOMOUS->PAUSED is disabled — align the leader by hand. "
                "Use an actuated leader (e.g. so101_leader_7dof) for the intended handover.",
                type(teleop).__name__ if teleop is not None else None,
            )
        self._listener = self._init_keyboard()
        self._print_controls(ctx)

    def run(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        features = ctx.data.dataset_features
        engine = self._engine
        interpolator = self._interpolator
        events = self._events
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        play = cfg.play_sounds
        control_interval = interpolator.get_control_interval(cfg.fps)
        # New episodes to collect this session (0/None -> unbounded, ESC stops).
        num_cap = self.config.num_episodes or 0

        engine.reset()
        interpolator.reset()
        state = _STANDBY
        last_action: dict[str, Any] | None = None
        saved = 0
        log_say("Standby. Press start to run the policy.", play)

        with VideoEncodingManager(dataset):
            try:
                while not events.stop.is_set() and not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    # --- handle one pending keyboard request ---------------------
                    action = events.consume()
                    if action == "save":
                        if state == _STANDBY:
                            log_say(f"Start policy. Episode {dataset.num_episodes}.", play)
                            self._enter_autonomous(ctx)
                            state, last_action = _AUTONOMOUS, None
                        else:
                            log_say(f"Save episode {dataset.num_episodes}.", play)
                            engine.pause()
                            self._release_leader(teleop)
                            dataset.save_episode()
                            saved += 1
                            self._return_home(ctx)
                            state, last_action = _STANDBY, None
                            if num_cap and saved >= num_cap:
                                log_say("Reached episode target.", play)
                                break
                    elif action == "discard":
                        if state in _RECORDING_STATES:
                            log_say("Discard episode.", play)
                            engine.pause()
                            self._release_leader(teleop)
                            dataset.clear_episode_buffer()
                            self._return_home(ctx)
                            state, last_action = _STANDBY, None
                    elif action == "cycle":
                        if state == _AUTONOMOUS:
                            log_say("Pause. Aligning leader.", play)
                            engine.pause()
                            self._align_leader_to_follower(ctx)
                            state = _PAUSED
                        elif state == _PAUSED:
                            log_say("Teleop correction.", play)
                            self._release_leader(teleop)
                            state = _CORRECTING
                        elif state == _CORRECTING:
                            log_say("Resume policy.", play)
                            self._enter_autonomous(ctx)
                            state, last_action = _AUTONOMOUS, None

                    # --- STANDBY: idle, no observation / no recording ------------
                    if state == _STANDBY:
                        dt = time.perf_counter() - loop_start
                        if (sleep_t := control_interval - dt) > 0:
                            precise_sleep(sleep_t)
                        continue

                    obs = robot.get_observation()

                    # --- AUTONOMOUS: qp policy drives, record intervention=False --
                    if state == _AUTONOMOUS:
                        obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                        if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                            continue
                        action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                        if action_dict is not None:
                            last_action = ctx.processors.robot_action_processor((action_dict, obs))
                            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                            action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                            dataset.add_frame(
                                {
                                    **obs_frame,
                                    **action_frame,
                                    "task": task_str,
                                    "intervention": np.array([False], dtype=bool),
                                }
                            )
                            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                    # --- PAUSED: hold the follower at its last commanded pose -----
                    elif state == _PAUSED:
                        if last_action is not None:
                            robot.send_action(last_action)

                    # --- CORRECTING: human teleop, record intervention=True -------
                    elif state == _CORRECTING:
                        obs_processed = ctx.processors.robot_observation_processor(obs)
                        teleop_action = teleop.get_action()
                        processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                        robot_action_to_send = ctx.processors.robot_action_processor((processed_teleop, obs))
                        robot.send_action(robot_action_to_send)
                        last_action = robot_action_to_send
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, processed_teleop, prefix=ACTION)
                        dataset.add_frame(
                            {
                                **obs_frame,
                                **action_frame,
                                "task": task_str,
                                "intervention": np.array([True], dtype=bool),
                            }
                        )
                        self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)

                    # --- pace the loop -------------------------------------------
                    dt = time.perf_counter() - loop_start
                    if (sleep_t := control_interval - dt) > 0:
                        precise_sleep(sleep_t)
                    elif state == _AUTONOMOUS:
                        # Overruns in AUTONOMOUS are the qp_sync inference pause on a
                        # chunk boundary — expected, and excluded from the dataset.
                        logger.debug("Inference pause: loop took %.0f ms this tick", dt * 1e3)
                    else:
                        logger.warning(
                            "Control loop slower (%.1f Hz) than target FPS (%s Hz) — frames may drop. "
                            "Check camera FPS / CPU load.",
                            1 / dt if dt > 0 else float("inf"),
                            cfg.fps,
                        )
            finally:
                engine.pause()
                if state in _RECORDING_STATES and dataset is not None and dataset.has_pending_frames():
                    logger.warning("Session stopped mid-episode — discarding the in-progress episode")
                    with contextlib.suppress(Exception):
                        dataset.clear_episode_buffer()

    def teardown(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        play = cfg.play_sounds
        logger.info("Stopping DAgger cycle recording")
        log_say("Stopping recording.", play)

        if self._listener is not None and not is_headless():
            self._listener.stop()

        if ctx.data.dataset is not None:
            ctx.data.dataset.finalize()
            if cfg.dataset is not None and cfg.dataset.push_to_hub:
                logger.info("Pushing dataset to hub...")
                if safe_push_to_hub(ctx.data.dataset, tags=cfg.dataset.tags, private=cfg.dataset.private):
                    log_say("Dataset uploaded to hub.", play)

        self._teardown_hardware(ctx.hardware, return_to_initial_position=cfg.return_to_initial_position)
        logger.info("DAgger cycle teardown complete")

    # ------------------------------------------------------------------
    # Transition side-effects
    # ------------------------------------------------------------------

    def _enter_autonomous(self, ctx: RolloutContext) -> None:
        """Reset + resume the engine for a fresh policy run; release the leader.

        The reset clears the qp_sync FIFO and anchors so the next chunk
        re-anchors to the current (possibly human-moved) follower pose.
        """
        self._interpolator.reset()
        self._engine.reset()
        self._engine.resume()
        self._release_leader(ctx.hardware.teleop)

    def _align_leader_to_follower(self, ctx: RolloutContext) -> None:
        """Drive the leader arm to the follower's current measured pose (torque ON)."""
        teleop = ctx.hardware.teleop
        if not teleop_supports_feedback(teleop):
            logger.warning("Leader has no torque control — align it by hand before correcting")
            return
        obs = ctx.hardware.robot_wrapper.get_observation()
        target = {k: obs[k] for k in teleop.feedback_features if k in obs}
        if not target:
            logger.warning("No follower joint keys match the leader — cannot align")
            return
        teleop_smooth_move_to(teleop, target, duration_s=self.config.align_duration_s, fps=50)

    @staticmethod
    def _release_leader(teleop) -> None:
        """Disable leader torque so the operator can move it freely."""
        if teleop is not None and teleop_supports_feedback(teleop):
            teleop.disable_torque()

    def _return_home(self, ctx: RolloutContext) -> None:
        """Smoothly interpolate the follower back to the home configuration.

        Uses ``home_position`` if set, otherwise the startup pose captured at
        connect (``ctx.hardware.initial_position`` — roughly where the arm was
        parked when the session launched).
        """
        robot = ctx.hardware.robot_wrapper
        home = self.config.home_position or ctx.hardware.initial_position or {}
        log_say("Returning home.", ctx.runtime.cfg.play_sounds)
        try:
            obs = robot.get_observation()
            current = {k: obs[k] for k in home if k in obs}
            if not current:
                logger.warning("No home target available — skipping return home")
                return
            steps = max(int(self.config.home_duration_s * 50), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {k: current[k] * (1 - t) + home[k] * t for k in current}
                robot.send_action(interp)
                precise_sleep(1 / 50)
        except Exception as e:
            logger.warning("Could not return home: %s", e)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _init_keyboard(self):
        if not PYNPUT_AVAILABLE or is_headless():
            logger.warning("Headless environment or pynput unavailable — keyboard controls disabled")
            return None

        cfg = self.config
        special_keys = {
            "space": keyboard.Key.space,
            "tab": keyboard.Key.tab,
            "enter": keyboard.Key.enter,
        }

        def resolve(key) -> str | None:
            if key == keyboard.Key.esc:
                return "esc"
            for name, pynput_key in special_keys.items():
                if key == pynput_key:
                    return name
            if hasattr(key, "char") and key.char:
                return key.char
            return None

        key_to_action = {
            cfg.cycle_key: "cycle",
            cfg.save_key: "save",
            cfg.discard_key: "discard",
        }

        def on_press(key):
            try:
                resolved = resolve(key)
                if resolved is None:
                    return
                if resolved == "esc" or resolved == cfg.quit_key:
                    logger.info("Stop key — ending session")
                    self._events.stop.set()
                    return
                if resolved in key_to_action:
                    self._events.request(key_to_action[resolved])
            except Exception as e:
                logger.debug("Key error: %s", e)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def _print_controls(self, ctx: RolloutContext) -> None:
        cfg = self.config
        logger.info(
            "DAgger cycle controls:\n"
            "    %s   - start policy (STANDBY) | save episode + home (recording)\n"
            "    %s   - cycle AUTONOMOUS -> PAUSED -> CORRECTING -> AUTONOMOUS\n"
            "    %s   - discard episode + home\n"
            "    %s / ESC - end session (saved episodes kept; unsaved one dropped)\n"
            "  Policy: %s\n"
            "  Task:   %s\n"
            "  FPS:    %s  (engine=%s, interp x%s)",
            cfg.save_key,
            cfg.cycle_key,
            cfg.discard_key,
            cfg.quit_key,
            ctx.runtime.cfg.policy.pretrained_path,
            ctx.runtime.cfg.dataset.single_task if ctx.runtime.cfg.dataset else ctx.runtime.cfg.task,
            ctx.runtime.cfg.fps,
            ctx.runtime.cfg.inference.type,
            ctx.runtime.cfg.interpolation_multiplier,
        )
