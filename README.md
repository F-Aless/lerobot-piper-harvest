# LeRobot — AgileX Piper fork (QP-smoothed VLA deployment & DAgger)

This repository is a research fork of [🤗 LeRobot](https://github.com/huggingface/lerobot),
based on upstream `main` at commit [`79c68214`](https://github.com/huggingface/lerobot/commit/79c68214070e392f04800ed092dd89e0b761f44e)
(June 2026, v0.5.2 dev). It adds real-robot support for the **AgileX Piper 6-DOF arm**
and the tooling we use to deploy and fine-tune VLA policies (SmolVLA) on a real-world
**apple-harvesting task**: QP-based action-chunk smoothing, smoothed real-time-chunking
(RTC) inference, and single-operator DAgger data collection.

Everything upstream LeRobot provides (datasets, policies, training) works unchanged —
see the [official docs](https://huggingface.co/docs/lerobot/index) for that. This README
covers only what this fork adds.

## What this fork adds

**Robots and teleoperators**

| Component | Type | Description |
|---|---|---|
| `piper_full` | robot | AgileX Piper 6-DOF + gripper over CAN (`piper_sdk`), joint-space control at up to 60 Hz |
| `piper_ee` | robot | Cartesian (end-effector) variant of the Piper: damped-least-squares IK on the bundled URDF, with yaw handling tuned for the Piper's home-pose singularity |
| `so101_follower_7dof` / `so101_leader_7dof` | robot / teleop | 7-motor SO-ARM 101 pair whose layout mirrors the Piper's kinematic chain — actions recorded from the leader replay 1:1 on the Piper |
| `so_leader_piper` | teleop | Classic 5-DOF SO-ARM leader remapped to drive the Piper (missing forearm-roll joint held fixed) |

**Inference & control**

- **QP chunk smoother** (`lerobot.policies.chunk_smoother`) — each predicted action
  chunk is passed through a per-joint OSQP quadratic program that penalises
  acceleration and jerk, caps per-tick velocity, respects joint limits, and anchors
  the chunk to the last commanded action so consecutive chunks join without spikes.
- **`qp_sync` / `qp_rtc` inference engines** — the smoother integrated into blocking
  chunked inference and into RTC (background) inference respectively. In *EE mode*
  a Cartesian chunk is projected, batch-IK'd, QP-smoothed in joint space, and emitted
  as plain joint commands.
- **Golden tickets** — optional fixed initial noise for the flow-matching action head
  (instead of fresh Gaussian noise per inference), making rollouts reproducible.
  Selected seeds for our checkpoints live in [`golden_tickets/`](golden_tickets/).
- **SLERP action interpolation** — orientation triples (`ee.roll/pitch/yaw`) are
  interpolated on SO(3) instead of linearly in Euler space.

**Data collection**

- **`dagger_cycle` rollout strategy** — single-key human-in-the-loop loop:
  the QP-smoothed policy drives the arm, one key cycles
  AUTONOMOUS → PAUSED (leader aligns to the follower) → CORRECTING (human teleop),
  and both autonomous and correction frames land in the same episode with an
  `intervention` flag.
- **`lerobot-record-dagger` script** — "policy preview, then human correction"
  recording workflow.

## Installation

Requires Python ≥ 3.12 (a conda env is recommended).

```bash
git clone <this-repo-url> lerobot-piper
cd lerobot-piper
pip install -e ".[piper,chunk-smoother]"
```

Relevant extras:

| Extra | Installs | Needed for |
|---|---|---|
| `piper` | `piper_sdk` | talking to the Piper arm (`piper_full`, `piper_ee`) |
| `chunk-smoother` | `osqp`, `scipy` | the `qp_sync` / `qp_rtc` inference engines |
| `feetech` | Feetech servo SDK | the SO-ARM leaders/followers |
| `intelrealsense` | RealSense SDK | the cameras used in our setup |
| `smolvla` | SmolVLA deps | running our harvest-apples checkpoints |

**Pinocchio is only needed for end-effector kinematics** — i.e. the `piper_ee` robot,
`--inference.ee_mode=true`, or FK-derived `ee.*` state observations. It is imported
lazily and deliberately **not** declared as a pip dependency (no reliable PyPI wheels
on all platforms). If and only if you need EE poses, install it via conda:

```bash
conda install -c conda-forge pinocchio
```

Joint-space usage (`piper_full` teleop, recording, `qp_sync`/`qp_rtc` with
`ee_mode=false`) works without it.

## Quick start

Full step-by-step commands (CAN bus activation, calibration, teleop, recording,
autonomous rollout, DAgger) are in **[USAGE_GUIDE.md](USAGE_GUIDE.md)**.
The shortest path, once the CAN interface is up and the leader is calibrated:

```bash
# Teleoperate the Piper with the 7-DOF SO-101 leader
lerobot-teleoperate \
    --robot.type=piper_full --robot.id=my_piper \
    --teleop.type=so101_leader_7dof --teleop.port=/dev/ttyACM0 --teleop.id=leader_7dof

# Autonomous rollout with QP-smoothed RTC inference
lerobot-rollout \
    --strategy.type=base \
    --policy.path=Faless/harvest_apples_smolvla_real \
    --robot.type=piper_full --robot.id=my_piper \
    --task="Pick the red apples one by one and place them into the green basket" \
    --fps=25 --inference.type=qp_rtc \
    --inference.lambda_a=80 --inference.lambda_j=80 --inference.v_max_deg_s=80 \
    --device=cuda
```

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** — operational guide with the exact commands
  used in the lab (CAN setup, calibration, teleop, dataset recording, rollout,
  DAgger sessions, experiment scripts).
- **[CODE_OVERVIEW.md](CODE_OVERVIEW.md)** — explanation of the code added by this
  fork: module map, the QP smoother formulation, the EE→joint pipeline, the
  inference engines, the DAgger strategies, and every change made to upstream files.
- [Upstream LeRobot documentation](https://huggingface.co/docs/lerobot/index) — for
  everything else (datasets, training, policies).

## Known limitations

- `tests/artifacts/` contains ~50 Git LFS *pointer* files without their content
  (inherited from how this snapshot was obtained; the repo is hosted without
  LFS). The upstream tests that read those artifacts are not runnable from a
  clone — restore the files from [upstream](https://github.com/huggingface/lerobot)
  if you need them. None of this fork's own code or tests depend on them.
- `piper_full`'s `deg`/`rad` joint units are implemented and unit-tested but
  **not yet validated on the real arm** (all our datasets and policies use the
  default normalized `pct` convention).

## License and attribution

This fork, like upstream LeRobot, is released under the [Apache 2.0 license](LICENSE).
LeRobot is developed by the Hugging Face team and contributors — see the
[upstream repository](https://github.com/huggingface/lerobot) for the original project
and citation information.
