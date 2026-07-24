#!/usr/bin/env bash
# DAgger cycle recording — SmolVLA (joint) harvest-apples on AgileX Piper.
#
# Single-key flow per episode:
#   s      start the policy (from STANDBY)
#   space  AUTONOMOUS -> PAUSED (leader slides to Piper pose) -> CORRECTING
#          (teleop, leader torque off) -> AUTONOMOUS  (one ring)
#   s      save the episode + return home (all joints to 0) -> STANDBY
#   c      discard the episode + return home -> STANDBY
#   ESC    stop the session
#
# Both the qp-smoothed policy frames (intervention=False) and the teleop
# corrections (intervention=True) are recorded into the same episode at a
# fixed 25 fps; the synchronous-inference pauses are excluded automatically.
#
# Add --resume=true to append to an existing rollout_* dataset.
set -euo pipefail

conda run -n "${LEROBOT_ENV:-lerobot-piper}" --no-capture-output python -u -m lerobot.scripts.lerobot_rollout \
  --robot.type=piper_full \
  --robot.can_channel=can0 \
  --robot.id=my_piper \
  --robot.cameras='{"camera1":{"type":"intelrealsense","serial_number_or_name":"733512070600","width":640,"height":360,"fps":30},"camera2":{"type":"intelrealsense","serial_number_or_name":"337322073539","width":640,"height":360,"fps":30}}' \
  --teleop.type=so101_leader_7dof \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_7dof \
  --policy.path=Faless/harvest_apples_smolvla_real \
  --strategy.type=dagger_cycle \
  --inference.type=qp_sync \
  --inference.ee_mode=false \
  --dataset.repo_id=Faless/rollout_harvest_apples_dagger_2 \
  --dataset.single_task="Pick the red apples one by one and place them into the green basket" \
  --dataset.num_episodes=50 \
  --dataset.fps=25 \
  --dataset.push_to_hub=false \
  --fps=25 \
  --device=cuda \
  --display_data=true \
  "$@"
