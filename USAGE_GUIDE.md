# Usage guide — AgileX Piper setup

Step-by-step commands to use this fork, both to **run our harvest-apples
policies** and to **use the arm for your own policy** (e.g. ACT) — see
[section 7](#7-using-the-piper-with-your-own-policy-example-act) for the
latter. Ports, camera serial numbers and Hub repo ids (`Faless/...`) are those
of our setup — replace them with your own.

**Hardware assumed below**

- AgileX Piper arm connected via a USB-CAN adapter (`can0`)
- SO-ARM 101 leader, 7-DOF variant, on `/dev/ttyACM0` (or a classic 5-DOF
  SO-ARM leader via the `so_leader_piper` teleop)
- 2× Intel RealSense cameras (serials `733512070600`, `337322073539`)

---

## 0. Environment setup

Create a fresh conda env and install this repo with the extras you need:

```bash
conda create -n lerobot-piper python=3.12 -y
conda activate lerobot-piper
cd <this-repo>
pip install -e ".[piper,chunk-smoother,feetech,intelrealsense,smolvla]"
```

- `piper` → `piper_sdk` (the arm), `chunk-smoother` → OSQP smoothing,
  `feetech` → SO-ARM leader, `intelrealsense` → cameras, `smolvla` → our
  checkpoints.
- **Only if you need end-effector poses** (`piper_ee`, `--inference.ee_mode=true`,
  or FK state observations): `conda install -c conda-forge pinocchio`.
  Joint-space work doesn't need it.

## 1. CAN bus: install and activate

One-time install (Ubuntu):

```bash
sudo apt install can-utils   # candump/cansend, for debugging
```

The activation scripts ship inside the `piper_sdk` package you just installed.
After every boot / adapter replug:

```bash
conda activate lerobot-piper
cd "$(python -c 'import piper_sdk, pathlib; print(pathlib.Path(piper_sdk.__file__).parent)')"
bash find_all_can_port.sh          # lists the USB-CAN adapters found
bash can_activate.sh can0 1000000  # bring can0 up at 1 Mbit/s (asks for sudo)
```

Verify it works: `ip link show can0` should say `state UP`, and with the arm
powered `candump can0` should stream frames (Ctrl-C to stop). If `can0` does
not appear, replug the adapter and rerun `find_all_can_port.sh`.

## 2. Calibrate the SO-ARM leader

```bash
lerobot-calibrate \
  --teleop.type=so101_leader_7dof \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_7dof
```

The `so_leader_piper` teleop (classic 5-DOF SO-ARM driving the Piper) uses its
own calibration directory — calibrate it separately with
`--teleop.type=so_leader_piper` if you use that leader.

## 3. Teleoperate the Piper

With the 7-DOF leader:

```bash
lerobot-teleoperate \
    --robot.type=piper_full \
    --robot.id=my_piper \
    --teleop.type=so101_leader_7dof \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_7dof \
    --display_data=true \
    --robot.cameras='{camera1: {"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30}, camera2: {"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}'
```

With the classic 5-DOF SO-ARM leader (forearm roll `joint_4` held fixed):

```bash
lerobot-teleoperate \
  --robot.type=piper_full \
  --robot.can_channel=can0 \
  --robot.id=my_piper \
  --teleop.type=so_leader_piper \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=so_classic \
  --display_data=true \
  --robot.cameras='{camera1: {"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30}, camera2: {"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}'
```

**Joint units.** By default `piper_full` reports/accepts joints normalized to
`[-100, 100]` and the gripper to `[0, 100]` — this is the convention all our
datasets and policies use. `--robot.unit=deg` (signed degrees) or
`--robot.unit=rad` (radians) expose raw angles instead, with the gripper in mm.
⚠️ `deg`/`rad` are implemented but **not hardware-tested** — if you record with
them, deploy with the same unit end-to-end.

## 4. Record a teleop dataset

The configuration used for the 100-episode harvest-apples dataset:

```bash
lerobot-record \
  --robot.type=piper_full \
  --robot.id=my_piper \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --teleop.type=so101_leader_7dof \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_7dof \
  --dataset.repo_id=Faless/harvest_apples_real_100ep \
  --dataset.num_episodes=100 \
  --dataset.single_task="Pick the red apples one by one and place them into the green basket" \
  --dataset.fps=30 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=30 \
  --dataset.streaming_encoding=false \
  --dataset.camera_encoder.vcodec=libsvtav1 \
  --dataset.encoder_threads=2 \
  --display_data=true
```

## 5. Autonomous rollout (inference)

### Choosing the inference engine

`lerobot-rollout` selects the engine with `--inference.type`:

| Type | Inference | Smoothing | Works with | Notes |
|---|---|---|---|---|
| `sync` | blocking, one call per chunk | none | any chunking policy (ACT, SmolVLA, …) | robot pauses at chunk boundaries |
| `qp_sync` | blocking | **QP** | any chunking policy (ACT, SmolVLA, …) | what we use for DAgger sessions |
| `rtc` | async (background thread) | RTC blending | RTC-capable policies only (SmolVLA, pi0-family) | no pauses |
| `qp_rtc` | async | RTC + **QP** | RTC-capable policies only | what we use for deployment |

Flags by engine — mixing them up is rejected at parse time:

- QP params (`qp_sync` and `qp_rtc` only): `lambda_a`, `lambda_j`,
  `v_max_deg_s`, `ee_mode`, `ee_state_obs`, `action_unit`, `strict_anchor`,
  `joint_keys`, `joint_limits_deg`. `action_unit` is auto-resolved from the
  robot's unit; it only tells the QP how to *interpret* chunk values (for
  velocity caps and limits) — it converts nothing. Policy, dataset and
  `--robot.unit` must use the same unit end-to-end, so normally leave it unset.
- RTC params (`rtc` and `qp_rtc` only): `rtc.*` (e.g.
  `rtc.execution_horizon`, `rtc.prefix_attention_schedule`),
  `queue_threshold`, `simulated_delay_ticks`; `feedforward_ticks` is
  `qp_rtc` only.
- `qp_sync` only: `execute_horizon` (smoothed steps executed per chunk before
  re-inferring; default = whole chunk).
- `golden_ticket` (`rtc`, `qp_rtc`, `qp_sync`): fixed action-head noise —
  **flow-matching policies only** (SmolVLA / pi0-family). With ACT or other
  non-flow policies leave it unset (you get a clear error otherwise).
- `dump_dir` (`rtc`, `qp_rtc`, `qp_sync`): one `.npz` per chunk for offline
  analysis.
- `sync` takes no extra flags at all.

### Joint-space policy, QP-smoothed RTC (our deployment config)

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=Faless/harvest_apples_smolvla_real \
  --device=cuda \
  --robot.type=piper_full \
  --robot.id=my_piper \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --task="Pick the red apples one by one and place them into the green basket" \
  --fps=25 \
  --duration=120 \
  --inference.type=qp_rtc \
  --inference.ee_mode=false \
  --inference.dump_dir=outputs/qp_rtc_dump \
  --inference.queue_threshold=19 \
  --inference.rtc.execution_horizon=19 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.lambda_a=80 \
  --inference.lambda_j=80 \
  --inference.v_max_deg_s=80 \
  --inference.golden_ticket=golden_tickets/harvest_apples_smolvla_real \
  --display_data=true
```

### End-effector policy on the joint-space robot (EE mode)

Requires pinocchio (`conda install -c conda-forge pinocchio`). Same command,
with an `*_ee` checkpoint and `--inference.ee_mode=true`; the engine converts
each EE chunk to joints (projector → batch IK → joint QP) before execution:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=Faless/harvest_apples_smolvla_real_ee \
  --device=cuda \
  --robot.type=piper_full \
  --robot.id=my_piper \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --task="Pick the red apples one by one and place them into the green basket" \
  --fps=30 \
  --duration=120 \
  --inference.type=qp_rtc \
  --inference.ee_mode=true \
  --inference.dump_dir=outputs/ee_rtc_dump \
  --inference.queue_threshold=19 \
  --inference.rtc.execution_horizon=19 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.lambda_a=80 \
  --inference.lambda_j=80 \
  --inference.v_max_deg_s=80 \
  --inference.golden_ticket=golden_tickets/harvest_apples_smolvla_v2_base_bs96_ee \
  --display_data=true
```

Note: `--inference.rate_hz` inherits `--fps` — don't set it separately.

## 6. DAgger data collection

### `dagger_cycle` (recommended): single-key loop with the qp_sync engine

Wrapped in [`dagger_harvest_cycle.sh`](dagger_harvest_cycle.sh):

```bash
bash dagger_harvest_cycle.sh \
  --dataset.repo_id=Faless/rollout_harvest_apples_dagger \
  --inference.golden_ticket=golden_tickets/harvest_apples_smolvla_real/ticket_00002.pt
```

Keyboard flow per episode:

| Key | Effect |
|---|---|
| `s` | from STANDBY: start the policy (begin an episode) |
| `space` | AUTONOMOUS → PAUSED (leader slides to the follower pose) → CORRECTING (teleop, torque off) → AUTONOMOUS |
| `s` | save the episode, return home (all joints to 0), back to STANDBY |
| `c` | discard the episode, return home |
| `ESC` | end the session (saved episodes are kept) |

Policy frames are recorded with `intervention=False`, correction frames with
`intervention=True`, in the same episode at a fixed 25 fps. Add `--resume=true`
to append to an existing dataset. The scripts run inside the `lerobot-piper`
conda env by default — set `LEROBOT_ENV=<your-env>` to override.

[`dagger_harvest_qprtc.sh`](dagger_harvest_qprtc.sh) is the measurement variant:
same flow but with the `qp_rtc` engine, no live viz, full-speed arm and per-chunk
dumps enabled — used to rebuild the chunk-overlap analysis from real data.

### `dagger` strategy (episode-count driven)

```bash
lerobot-rollout \
  --strategy.type=dagger \
  --strategy.num_episodes=10 \
  --policy.path=Faless/harvest_apples_smolvla_v2_base_bs96 \
  --device=cuda \
  --robot.type=piper_full \
  --robot.id=my_piper \
  --robot.cameras='{"image":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"image2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --teleop.type=so101_leader_7dof \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_7dof \
  --dataset.repo_id=Faless/rollout_harvest_dagger \
  --dataset.single_task="Pick the red apples one by one and place them into the green basket" \
  --dataset.fps=25 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=30 \
  --task="Pick the red apples one by one and place them into the green basket" \
  --fps=25 \
  --duration=0 \
  --inference.type=qp_rtc \
  --inference.ee_mode=false \
  --inference.queue_threshold=19 \
  --inference.rtc.execution_horizon=19 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.lambda_a=80 \
  --inference.lambda_j=80 \
  --inference.v_max_deg_s=80
```

### `lerobot-record-dagger` (policy preview, then correction)

An alternative recording workflow without the rollout machinery: `s` starts a
policy preview (not recorded), `m` switches to human correction with recording,
`s` saves / `c` discards, `ESC` ends. See the docstring of
`src/lerobot/scripts/lerobot_record_dagger.py` for a full example.

## 7. Using the Piper with your own policy (example: ACT)

Nothing here is harvest-specific — the robot, teleop and recording stack work
for any task/policy. Typical loop for a new user with ACT:

1. **Record** a dataset of your task by teleoperation (section 4), with your
   own `--dataset.repo_id` and `--dataset.single_task`. Keep the default joint
   unit (normalized pct) unless you have a reason not to.

2. **Train** ACT on it with the standard upstream trainer (see the
   [LeRobot training docs](https://huggingface.co/docs/lerobot/il_robots)):

   ```bash
   lerobot-train \
     --dataset.repo_id=<you>/<your_dataset> \
     --policy.type=act \
     --output_dir=outputs/train/act_piper \
     --job_name=act_piper \
     --policy.device=cuda \
     --policy.repo_id=<you>/act_piper
   ```

3. **Deploy** with the blocking engines — ACT is not an RTC-capable policy, so
   use `sync` (raw) or `qp_sync` (QP-smoothed, recommended on the real arm),
   and no `golden_ticket` (that's for flow-matching policies only):

   ```bash
   lerobot-rollout \
     --strategy.type=base \
     --policy.path=<you>/act_piper \
     --device=cuda \
     --robot.type=piper_full \
     --robot.id=my_piper \
     --robot.cameras='{...same cameras as recording...}' \
     --task="<your task>" \
     --fps=30 \
     --duration=120 \
     --inference.type=qp_sync \
     --inference.lambda_a=40 \
     --inference.lambda_j=20 \
     --inference.v_max_deg_s=80
   ```

   `--inference.execute_horizon=N` re-infers after N executed steps instead of
   running each chunk to the end (more reactive, more pauses). For DAgger-style
   corrections on your own policy, everything in section 6 works unchanged —
   just point `--policy.path` at your checkpoint.

## 8. Experiment scripts

- [`measure_vmax.sh`](measure_vmax.sh) — autonomous (sentry) run at a given
  `v_max`, recording both the dataset and the per-chunk dumps, for the
  "delay vs velocity limit" sweep: `bash measure_vmax.sh 80`.
