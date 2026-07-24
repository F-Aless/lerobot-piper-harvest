# Code overview — what this fork adds and changes

This document explains the code added on top of upstream LeRobot
(`main` @ `79c68214`, June 2026) and every modification made to upstream files.

## Module map

```
src/lerobot/
├── robots/
│   ├── piper_full/            # AgileX Piper, joint-space control (CAN / piper_sdk)
│   ├── piper_ee/              # Cartesian Piper: IK, projector, EE chunk pipeline, URDF + meshes
│   └── so101_follower_7dof/   # 7-motor SO-ARM 101 follower (Piper-compatible schema)
├── teleoperators/
│   ├── so101_leader_7dof/     # 7-motor SO-ARM 101 leader
│   └── so_leader/so_leader_piper.py   # classic 5-DOF SO-ARM remapped to the Piper
├── policies/chunk_smoother/   # OSQP joint-trajectory smoother + anchor resolution
├── rollout/
│   ├── inference/qp_sync.py   # blocking chunked inference + QP smoothing
│   ├── inference/qp_rtc.py    # RTC (async) inference + QP smoothing
│   └── strategies/dagger_cycle.py     # single-key DAgger loop
└── scripts/lerobot_record_dagger.py   # "policy preview → human correction" recorder
tests/
├── policies/test_qp_smoother.py
├── robots/test_piper_ee_pipeline.py
└── test_qp_ee_engine.py
golden_tickets/                # selected fixed-noise seeds for our checkpoints
```

## Robots

### `piper_full` — joint-space Piper

`PiperFullSDK` (`piper_full_sdk.py`) is a thin wrapper around `piper_sdk`
(`C_PiperInterface_V2`, imported lazily): degree/mm I/O at the LeRobot level,
conversion to the SDK's milli-degree / 0.001-mm units inside, position-control
framing (`MotionCtrl_2(0x01, 0x01, speed, 0x00)`) set once at connect — good for
60 Hz teleop. Joint limits are read from the SDK (`GetAllMotorAngleLimitMaxSpd`)
with URDF defaults as fallback.

`PiperFull` exposes:

- observation `{joint_{1..6}.pos, gripper.pos, gripper.tau, cameras…}`,
  action `{joint_{1..6}.pos, gripper.pos}`. The joint unit is selected with
  `--robot.unit`: `pct` (default, `[-100, 100]` through the oriented limits,
  gripper `[0, 100]` — the convention all our datasets use), `deg` or `rad`
  (raw signed angles, gripper in mm). ⚠️ `deg`/`rad` are implemented but not
  hardware-tested. `use_degrees=True` remains as a legacy alias for
  `unit="deg"`. The robot declares its unit via `action_angle_unit`, which the
  QP engines read to auto-resolve their `action_unit`.
- `joint_limits_deg` — introspectable limits used by the QP smoother.
- `go_home_slow()` — smooth homing to all-joints-zero, used by the rollout
  shutdown path and the `dagger_cycle` save/discard transitions.
- `ee_pose_from_observation()` / `urdf_q_from_observation()` — FK helpers
  (lazy pinocchio) that let EE-trained policies run on this joint-space robot.
- `make_ee_chunk_smoother(...)` — factory for the EE→joint chunk pipeline
  (see below) with joint-space output.

### `piper_ee` — Cartesian Piper

Same hardware I/O (reuses `PiperFullSDK`) but a Cartesian action space
(`ee.x/y/z/roll/pitch/yaw + gripper`). Three layers:

- **`piper_ee_ik.py`** — pinocchio-backed FK + damped-least-squares IK on a
  reduced model of the bundled URDF (gripper joints locked). The Piper sits
  near a yaw/position-coupled singularity at home (cond(J) ≈ 2.8e4), so the
  solver supports three yaw modes: `full` (track yaw), `soft` (yaw
  down-weighted), `free` (yaw dropped, null-space bias toward the seed and an
  optional neutral posture), with automatic fallback to `free` when
  ill-conditioned. Per-iteration step caps and a final joint-limit clamp keep
  every returned step executable. Pinocchio is imported lazily — registering
  the robot or parsing configs never imports it.
- **`ee_action_projector.py`** — makes a raw policy target kinematically
  acceptable *before* it reaches the robot: per-tick translation clamp,
  SO(3)-geodesic rotation clamp, warm-started IK, and a chain of fallbacks
  (scaled target → yaw-free retry → hold last valid command).
- **`ee_chunk_smoother.py`** — the chunk-level pipeline used by the QP engines
  in EE mode: raw EE chunk → per-step projection (state evolves along the
  chunk) → warm-started batch IK → joint-space QP smoothing → output either as
  joint commands (deployment path; downstream interpolation is linear on
  joints, so no Euler issues) or as FK'd EE targets with branch-continuous RPY.

## QP chunk smoother (`policies/chunk_smoother`)

One independent QP per joint over the chunk horizon `y ∈ R^T`:

```
min_y  ‖y − r‖² + λ_a ‖Δ²y‖² + λ_j ‖Δ³y‖²
s.t.   |Δy| ≤ v_max / rate_hz          (per-tick velocity cap)
       q_min ≤ y ≤ q_max               (joint limits, if available)
       y[anchor_idx] = q_anchor        (anchor equality, if available)
       y[anchor_idx] − y[anchor_idx−1] = v_anchor   (optional velocity anchor)
```

`r` is the post-processor action chunk converted to degrees. Solved with OSQP
(warm-started, problem geometry built once per chunk length; compatible with
both osqp 0.6.x and ≥ 1.0). Defaults `v_max=80 °/s, λ_a=40, λ_j=20` come from
an offline sweep; the harvest-apples deployments use `80/80/80`.

`anchor.py` holds the glue shared by both engines: joint-limit resolution
(config override → robot introspection), unit conversion (`pct`/`deg`/`rad` →
degrees), and anchor extraction from the last commanded action or the latest
observation.

## Inference engines (`rollout/inference`)

Both engines require the `chunk-smoother` extra and are selected with
`--inference.type=qp_sync|qp_rtc`. Their smoother rate is tied to the control
rate: `rate_hz` inherits `--fps` and an explicit mismatch is rejected.

**Anchor priority** (both engines): (1) last action actually commanded to the
robot — the control loop publishes it via `notify_last_commanded_action`;
(2) the latest observation's joint values; (3) none — first chunk runs
unanchored (`strict_anchor=true` raises instead).

### `qp_sync`

Blocking chunked inference: the policy runs inline on the control thread,
each chunk is QP-smoothed anchored to the last command, then executed
open-loop from a FIFO; the robot pauses on chunk boundaries. Deterministic and
simple — used for the `dagger_cycle` sessions, where the pauses are harmless
(dataset frames are stamped by index, so pauses don't corrupt the fixed-fps
dataset). Works with **any** chunking policy (ACT included): the `noise`
kwarg is only passed when a golden ticket is configured, since non-flow
policies like ACT don't accept it. The RTC engines instead require
RTC-capable policies (SmolVLA / pi0-family) by construction.

### `qp_rtc`

Subclasses the upstream RTC engine and overrides `_post_process_chunk` to
smooth each chunk before it is merged into the action queue, so inference
stays asynchronous. Extra machinery added at the RTC level (available to plain
`rtc` too):

- **`golden_ticket`** — load a fixed noise tensor (shape
  `(1, chunk_size, max_action_dim)`) passed as the flow-matching `x_t` seed on
  every inference, replacing per-chunk Gaussian sampling. Makes rollouts
  repeatable and lets you select a "good" seed offline (see
  `golden_tickets/`).
- **`simulated_delay_ticks`** — deterministic delay override for sim tests:
  the merge blocks until exactly D actions were consumed, making RTC behaviour
  independent of wall-clock speed.
- **`feedforward_ticks`** — splice each new chunk N ticks ahead of the
  measured delay so the executed command *leads* the policy, compensating the
  arm's ~100 ms transport lag; the QP anchor moves with the splice point so
  seam continuity holds.
- **`dump_dir`** — one `.npz` per chunk (original / pre-smoother / smoothed,
  delay, latency, obs state, anchor…) for offline analysis.
- **EE mode anchoring** (`ee_qp_anchor`) — the measured state seeds
  FK/projector/IK, but the QP anchor pins the chunk to the last *commanded*
  action: re-anchoring on the measured state would step the command backwards
  by the tracking error at every merge (sawtooth). `tracking_warn_deg` flags
  merges where the arm lags the commands.

In EE mode both engines optionally replace the robot's joint state observation
with FK-computed `ee.*` values (`ee_state_obs`) — required when the policy was
trained on an `*_ee` dataset, otherwise its normalisation statistics would be
applied to joint values.

## DAgger collection

### `dagger_cycle` strategy (`rollout/strategies/dagger_cycle.py`)

Manual-episode, single-operator loop (state machine documented in the module
docstring): STANDBY → (s) AUTONOMOUS → (space) PAUSED — the leader arm is
servoed to the follower's pose so the handover is seamless — → (space)
CORRECTING (leader torque off, human drives) → (space) AUTONOMOUS again;
`s` saves the episode and homes the robot, `c` discards. Autonomous frames are
tagged `intervention=False`, corrections `intervention=True`, in the same
episode. Designed to pair with `--inference.type=qp_sync`.

### `lerobot-record-dagger` script

A lighter workflow on the plain record stack: TELEOP_IDLE (not recorded) →
`s` POLICY preview (the leader mirrors the commanded action, not recorded) →
`m` TELEOP_REC (human corrects, recorded) → `s` save / `c` discard.

## Changes to upstream files

- `configs/policies.py` — `action_feature_names` (the training dataset's action
  column layout) moved into the base pretrained config so checkpoints are
  self-describing; removed from the per-policy pi0/pi05/pi0_fast configs.
  Consumed by the EE→joint conversion and action-space validation.
- `rollout/context.py` — `.tau` channels (gripper force) accepted as state
  features (SmolVLA on the Piper uses an 8-D state); a policy whose action
  space differs from the robot's is rejected unless a converting engine
  (`qp_sync`/`qp_rtc`) is selected.
- `rollout/robot_wrapper.py`, `robots/robot.py` — duck-typed capability
  forwarding: `joint_limits_deg`, `make_ee_chunk_smoother`,
  `ee_pose_from_observation`, `urdf_q_from_observation`,
  `ee_anchor_q_from_observation`.
- `rollout/strategies/core.py` — the control loop publishes every commanded
  action to the engine (`notify_last_commanded_action`, the QP anchor source);
  shutdown homes via `go_home_slow()` when the robot supports it; the action
  interpolator SLERPs RPY triples.
- `policies/rtc/action_queue.py` — `merge(..., feedforward=N)` support.
- `utils/action_interpolator.py` — quaternion SLERP for `*.roll/pitch/yaw`
  triples (linear Euler interpolation jumps across the ±π branch cut).
- `scripts/lerobot_record.py` — dataset write moved after the Rerun log so live
  viz doesn't lag behind encoding; teleop connects before the robot;
  disconnect order mirrored.
- Registration boilerplate — the new robots/teleops added to the factories in
  `robots/utils.py`, `teleoperators/utils.py` and to the imports of every
  `lerobot_*` CLI script.
- Compatibility shims (for older Python/library environments, e.g. the Isaac
  sim-eval env): PEP 695 generics replaced with explicit `TypeVar`s
  (`streaming_dataset.py`, `processor/pipeline.py`, `utils/io_utils.py`,
  `motors/motors_bus.py`); guarded `torch_compilable_check` import in
  `policies/eo1`; explicit `@dataclass` on the GR00T config; osqp 0.6/1.0 API
  detection in the smoother.
- Small fixes: NVENC/VideoToolbox options in `configs/video.py` go through
  `set_if` (no `None` values passed to ffmpeg); Rerun images no longer logged
  `static=true` (`utils/visualization_utils.py`).
- `pyproject.toml` — extras `piper`, `piper-ee`, `chunk-smoother`; the
  `lerobot-record-dagger` entry point; Piper URDF/meshes shipped as package
  data; per-path ruff exemptions for math notation.

## Dependency policy

- `piper_sdk` — behind the `piper` extra, imported lazily.
- `osqp`/`scipy` — behind the `chunk-smoother` extra; the engines raise a
  clear "install lerobot[chunk-smoother]" error if missing.
- **pinocchio** — needed only for EE kinematics (`piper_ee`, `ee_mode=true`,
  FK state observations). Imported lazily everywhere and intentionally not a
  pip dependency (no reliable wheels on all platforms): install via
  `conda install -c conda-forge pinocchio` only if you need EE poses.
