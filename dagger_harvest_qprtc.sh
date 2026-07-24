#!/usr/bin/env bash
# MEASUREMENT run — DAgger cycle with the qp_rtc engine, NO live viz, recording
# everything needed to rebuild the chunk-overlap images from REAL data:
#   * dataset (--dataset.repo_id, passed as arg) -> action ACTUALLY SENT per tick
#     (the true executed trajectory) + measured state per tick;
#   * --inference.dump_dir -> per-chunk raw / pre_smoother / smoothed + timestamp
#     + delay + idx (align to the dataset by timestamp).
#
# Copy of dagger_harvest_cycle.sh. Changes vs the original (all flagged):
#   inference.type        qp_sync -> qp_rtc
#   display_data          true    -> false        (no live rerun)
#   + inference.dump_dir                          (save chunks for the images)
#   + robot.speed_percent=100                     (was default 50)
#   + lambda_a/lambda_j/v_max = 80/80/80          (golden-ticket config; dagger
#                                                   default would be 40/20/80)
#   + queue_threshold=19, rtc.execution_horizon=19, prefix_attention=EXP
#                                                 (golden-ticket config)
#   - hardcoded dataset.repo_id removed -> pass it as an arg (no dup override)
#
# Same single-key flow as the original (s / space / s|c / ESC). Keypresses still
# work with display_data=false. Pass --dataset.repo_id=... and
# --inference.golden_ticket=... as extra args. Add --resume=true to append.
set -euo pipefail

conda run -n "${LEROBOT_ENV:-lerobot-piper}" --no-capture-output python -u -m lerobot.scripts.lerobot_rollout \
  --robot.type=piper_full \
  --robot.can_channel=can0 \
  --robot.id=my_piper \
  --robot.speed_percent=100 \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --teleop.type=so101_leader_7dof \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_7dof \
  --policy.path=Faless/harvest_apples_smolvla_real \
  --strategy.type=dagger_cycle \
  --inference.type=qp_rtc \
  --inference.ee_mode=false \
  --inference.dump_dir=outputs/qp_rtc_measure_dump_30hz \
  --inference.queue_threshold=19 \
  --inference.rtc.execution_horizon=19 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.lambda_a=80 \
  --inference.lambda_j=80 \
  --inference.v_max_deg_s=80 \
  --dataset.single_task="Pick the red apples one by one and place them into the green basket" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --fps=30 \
  --device=cuda \
  --display_data=false \
  "$@"
