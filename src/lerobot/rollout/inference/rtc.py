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

"""Real-Time Chunking inference engine.

A background thread produces action chunks asynchronously via
:meth:`policy.predict_action_chunk`.  The main control loop polls
``get_action`` for the next ready action; observations flow the other
way via ``notify_observation``.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


def _load_golden_ticket(path: str, device: str) -> torch.Tensor:
    """Load a fixed initial-noise tensor for the action head from ``path``.

    ``path`` may be a single ``.pt`` file or a directory containing exactly one
    ``.pt`` file. The tensor is moved to ``device`` and cast to float32. Returns
    a tensor of shape (1, chunk_size, max_action_dim) — the same shape SmolVLA's
    ``sample_noise`` would produce, which is passed straight through as the
    flow-matching ``x_t`` seed.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        candidates = sorted(p.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt golden ticket found in directory {p}")
        if len(candidates) > 1:
            logger.warning(
                "Golden ticket directory %s contains %d .pt files; using the first: %s",
                p,
                len(candidates),
                candidates[0].name,
            )
        p = candidates[0]
    elif not p.is_file():
        raise FileNotFoundError(f"Golden ticket path does not exist: {p}")

    noise = torch.load(p, map_location=device, weights_only=True)
    if not isinstance(noise, torch.Tensor):
        raise TypeError(f"Golden ticket {p} must contain a torch.Tensor, got {type(noise).__name__}")
    noise = noise.to(device=device, dtype=torch.float32)
    logger.info("Loaded golden ticket %s (shape=%s) as fixed action-head noise", p, tuple(noise.shape))
    return noise


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Async RTC inference: a background thread produces action chunks.

    ``get_action`` pops the next action from the shared queue (or
    returns ``None`` if the queue is empty).  The main loop should call
    ``notify_observation`` every tick and ``pause``/``resume`` around
    human-intervention phases.
    """

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
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
        simulated_delay_ticks: int | None = None,
        golden_ticket: str | None = None,
        feedforward_ticks: int = 0,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        # Policy-side view of the hardware features. Subclasses may replace it
        # (e.g. qp_rtc swaps joint state names for FK-computed ee.* keys).
        self._policy_hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold
        # Deterministic delay override (ticks); None = wall-clock latency (real HW).
        self._simulated_delay_ticks = simulated_delay_ticks
        # Fixed action-head noise ("golden ticket"); None = sample fresh noise.
        self._golden_noise: torch.Tensor | None = (
            _load_golden_ticket(golden_ticket, self._device) if golden_ticket else None
        )
        # Feedforward: anticipate the arm's transport delay by N ticks on top of the
        # measured inference delay, so the executed command leads the policy by N ticks
        # (the chunk already carries the lookahead). 0 = off (legacy behaviour).
        self._feedforward_ticks = max(0, int(feedforward_ticks))

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None

        # Optional per-chunk debug dump (one .npz per chunk in self._dump_dir).
        # Subclasses may stash extra fields by writing to ``_pending_dump`` from
        # within ``_post_process_chunk`` — the dict is flushed alongside the
        # original/processed/delay/latency record at the end of each iteration.
        self._dump_dir: Path | None = None
        self._dump_index: int = 0
        self._pending_dump: dict[str, Any] = {}

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )

        # Processor introspection for relative-action re-anchoring.
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once torch.compile warmup is complete (or immediately if compile is disabled)."""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set()

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        self._action_queue = ActionQueue(self._rtc_config)
        self._obs_holder = {
            "obs": None,
            "robot_type": self._robot.robot_type,
        }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            else:
                logger.info("RTC inference thread stopped")
            self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread."""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        self._policy_active.set()

    def reset(self) -> None:
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_queue is not None:
            self._action_queue.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    # ------------------------------------------------------------------
    # Subclass extension point
    # ------------------------------------------------------------------

    def _post_process_chunk(
        self,
        original: torch.Tensor,
        processed: torch.Tensor,
        delay: int,
    ) -> torch.Tensor:
        """Hook for subclasses to transform the processed action chunk.

        Called inside the RTC background thread, after ``postprocessor`` and
        before ``ActionQueue.merge``. The ``delay`` argument is the number of
        leading ticks the queue will drop on merge — useful for delay-aware
        smoothing.

        Default: identity. Subclasses (e.g. ``QPSmoothedRTCInferenceEngine``)
        override to apply chunk-level smoothing or retargeting.
        """
        return processed

    # ------------------------------------------------------------------
    # Debug dump (optional)
    # ------------------------------------------------------------------

    def _obs_for_policy(self, obs: dict) -> dict:
        """Hook: the observation dict as the policy should see it (default: as-is)."""
        return obs

    def enable_chunk_dump(self, dump_dir: str | Path) -> None:
        """Save one .npz per inference chunk to ``dump_dir`` (created if missing).

        Each file ``chunk_NNNNNN.npz`` contains:
          * ``original``      — (T, A) chunk straight out of the policy
          * ``processed``     — (T, A) chunk after the postprocessor
                                (and after _post_process_chunk in subclasses)
          * ``delay_ticks``   — int, RTC delay in queue ticks
          * ``inference_s``   — float, wall-clock policy latency
          * ``idx_before``    — int, ActionQueue index at start of inference
          * ``timestamp``     — float, time.time() at dump
          * ``obs_state``     — flattened obs joint state (best-effort)

        Subclasses can add extra arrays by writing to ``self._pending_dump``
        from within ``_post_process_chunk``; those keys are merged into the
        file before the npz is written.
        """
        p = Path(dump_dir)
        p.mkdir(parents=True, exist_ok=True)
        self._dump_dir = p
        self._dump_index = 0
        logger.info("RTC chunk dump enabled → %s", p)

    def _write_chunk_dump(
        self,
        *,
        original: torch.Tensor,
        processed: torch.Tensor,
        delay_ticks: int,
        inference_s: float,
        idx_before: int,
        obs: dict | None,
    ) -> None:
        if self._dump_dir is None:
            return
        path = self._dump_dir / f"chunk_{self._dump_index:06d}.npz"
        self._dump_index += 1
        try:
            data = {
                "original": original.detach().cpu().numpy(),
                "processed": processed.detach().cpu().numpy(),
                "delay_ticks": int(delay_ticks),
                "inference_s": float(inference_s),
                "idx_before": int(idx_before),
                "timestamp": float(time.time()),
            }
            if obs is not None:
                # Snapshot scalar state values keyed by name (joint_X.pos, ...).
                scalars = {
                    k: float(v)
                    for k, v in obs.items()
                    if isinstance(v, (int, float)) and (k.endswith(".pos") or k.endswith(".tau"))
                }
                if scalars:
                    data["obs_state_keys"] = np.array(list(scalars.keys()))
                    data["obs_state_values"] = np.array(list(scalars.values()), dtype=np.float64)
            # Merge in subclass-supplied fields.
            for k, v in self._pending_dump.items():
                data[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v
            self._pending_dump.clear()
            np.savez(path, **data)
        except Exception:
            logger.debug("chunk dump failed", exc_info=True)
            self._pending_dump.clear()

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 0
            inference_count = 0
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()

                        if self._simulated_delay_ticks is not None:
                            delay = self._simulated_delay_ticks
                        else:
                            latency = latency_tracker.max()
                            delay = math.ceil(latency / time_per_chunk) if latency else 0

                        obs_batch = build_dataset_frame(
                            self._policy_hw_features, self._obs_for_policy(obs), prefix="observation"
                        )
                        obs_batch = prepare_observation_for_inference(
                            obs_batch, policy_device, self._task, self._robot.robot_type
                        )
                        obs_batch["task"] = [self._task]

                        preprocessed = self._preprocessor(obs_batch)

                        if prev_actions is not None and self._relative_step is not None:
                            # Rebase against the raw cached state so the leftover tail stays in
                            # the training-time coordinate frame.
                            raw_state = self._relative_step.get_cached_state()
                            if raw_state is not None:
                                prev_abs = queue.get_processed_left_over()
                                if prev_abs is not None and prev_abs.numel() > 0:
                                    prev_actions = reanchor_relative_rtc_prefix(
                                        prev_actions_absolute=prev_abs,
                                        current_state=raw_state,
                                        relative_step=self._relative_step,
                                        normalizer_step=self._normalizer_step,
                                        policy_device=policy_device,
                                    )

                        if prev_actions is not None:
                            prev_actions = _normalize_prev_actions_length(
                                prev_actions, target_steps=self._rtc_config.execution_horizon
                            )

                        # ``noise`` only when a golden ticket is loaded — the RTC
                        # kwargs are always required (RTC-capable policies only),
                        # but not every such policy accepts a noise override.
                        rtc_kwargs: dict[str, Any] = {
                            "inference_delay": delay,
                            "prev_chunk_left_over": prev_actions,
                        }
                        if self._golden_noise is not None:
                            rtc_kwargs["noise"] = self._golden_noise
                        actions = self._policy.predict_action_chunk(preprocessed, **rtc_kwargs)

                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        new_latency = time.perf_counter() - current_time
                        if self._simulated_delay_ticks is not None:
                            # Deterministic sim delay: block the merge until at least D
                            # actions have been consumed since inference started
                            # (emulates a fixed D-tick HW latency regardless of wall-clock
                            # sim speed), then splice using the ACTUAL consumed count so
                            # the seam is exactly continuous (indexes_diff == real_delay,
                            # no jump / no warning spam). Needs queue_threshold >= D so the
                            # buffer can supply D pops; empty() guards against deadlock.
                            target_delay = self._simulated_delay_ticks
                            while not self._shutdown_event.is_set() and self._policy_active.is_set():
                                if (queue.get_action_index() - idx_before) >= target_delay:
                                    break
                                if queue.empty():
                                    break
                                time.sleep(_RTC_IDLE_SLEEP_S)
                            new_delay = max(0, queue.get_action_index() - idx_before)
                        else:
                            new_delay = math.ceil(new_latency / time_per_chunk)

                        # Feedforward: lead the policy by N ticks WITHOUT mutating the
                        # measured delay. ``new_delay`` stays the actual inference delay
                        # (diagnostics + queue-validation); ``splice_idx = actual + N``
                        # drives the QP anchor AND the queue splice (kept in lock-step so
                        # the anchor pins the spliced point to the last command -> seam
                        # continuity holds). Skip the first chunk: no command anchor yet,
                        # so leading would start mid-trajectory (jump).
                        ff = (
                            self._feedforward_ticks
                            if (self._feedforward_ticks and inference_count >= 1)
                            else 0
                        )
                        splice_idx = min(new_delay + ff, processed.shape[0] - 1)
                        if ff and new_delay + ff > processed.shape[0] - 1:
                            logger.warning(
                                "feedforward: delay+ff (%d+%d) saturates chunk horizon T=%d; holding instead of leading",
                                new_delay,
                                ff,
                                processed.shape[0],
                            )

                        # Keep an unsmoothed copy for the debug dump.
                        processed_raw_for_dump = processed.clone() if self._dump_dir is not None else None

                        inference_count += 1
                        consecutive_errors = 0
                        is_warmup = self._use_torch_compile and inference_count <= warmup_required
                        if is_warmup:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        processed = self._post_process_chunk(original, processed, splice_idx)

                        # Optional debug dump (one .npz per chunk).
                        if self._dump_dir is not None:
                            if processed_raw_for_dump is not None:
                                # ``processed`` may have been smoothed by the
                                # subclass; record the pre-smoother snapshot
                                # under a separate key so we can diff offline.
                                self._pending_dump.setdefault(
                                    "processed_pre_smooth",
                                    processed_raw_for_dump.detach().cpu().numpy(),
                                )
                            self._write_chunk_dump(
                                original=original,
                                processed=processed,
                                delay_ticks=new_delay,
                                inference_s=new_latency,
                                idx_before=idx_before,
                                obs=obs,
                            )

                        queue.merge(original, processed, new_delay, idx_before, feedforward=ff)

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # Persistent failure: stop retrying and propagate shutdown.
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
