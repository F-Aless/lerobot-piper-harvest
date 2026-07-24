#!/usr/bin/env bash
# measure_vmax.sh <vmax_deg_s> [extra rollout args]
#
# Autonomous (sentry) measurement at a given v_max, for the "delay vs velocity
# limit" sweep. Records:
#   * dataset Faless/rollout_qp_rtc_v<vmax>  -> action ACTUALLY SENT + measured state per tick
#   * dump    outputs/qp_rtc_dump_v<vmax>    -> per-chunk raw/pre_smoother/smoothed + delay
#                                               + q_anchor + q_anchor_velocity (velocity anchor ON)
#
# NO teleop (pure policy+smoother command -> clean state<->action lag).
# qp_rtc, lambda 80/80, fps 30, speed_percent 100, EXP, queue/horizon 19 (golden-ticket config).
# Stop with Ctrl-C (SIGINT) -> graceful shutdown -> homes to all-joints-0deg (latch holds).
#
# Usage:  ./measure_vmax.sh 40        # then Ctrl-C after ~2-3 min of motion
set -euo pipefail
VMAX=${1:?usage: ./measure_vmax.sh <vmax_deg_s> [extra rollout args]}
shift || true
SUF=${TAG:+_$TAG}                     # optional run label via TAG env var (e.g. TAG=ff2)
pkill -x rerun 2>/dev/null || true    # kill any leftover rerun viewer (CPU hog -> jerky ~13Hz loop)

conda run -n "${LEROBOT_ENV:-lerobot-piper}" --no-capture-output python -u -m lerobot.scripts.lerobot_rollout \
  --robot.type=piper_full \
  --robot.can_channel=can0 \
  --robot.id=my_piper \
  --robot.speed_percent=100 \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --policy.path=${POLICY:-Faless/harvest_apples_smolvla_real} \
  --strategy.type=sentry \
  --inference.type=qp_rtc \
  --inference.ee_mode=false \
  --inference.dump_dir=outputs/qp_rtc_dump_v${VMAX}${SUF} \
  --inference.queue_threshold=19 \
  --inference.rtc.execution_horizon=19 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.lambda_a=80 \
  --inference.lambda_j=80 \
  --inference.v_max_deg_s=${VMAX} \
  --dataset.repo_id=Faless/rollout_qp_rtc_v${VMAX}${SUF} \
  --dataset.single_task="Pick the red apples one by one and place them into the green basket" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --fps=30 \
  --device=cuda \
  --display_data=false \
  "$@"
