#!/usr/bin/env bash
# Eval hopex_realsensor checkpoints deterministically.
# Run: distrobox enter grasping -- bash scripts/eval_realsensor_hopex.sh
# Output: /tmp/eval_realsensor_model{N}.log  (inside distrobox /tmp == host /tmp)
set -euo pipefail

WBT_DIR="/home/dongc1/workspace/HOPE/hope_training/whole_body_tracking"
RUN_DIR="$WBT_DIR/logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_14-12-42_hopex_realsensor_scratch"
FH="$WBT_DIR/artifacts/hope_forehand_hopex/motion.npz"
BH="$WBT_DIR/artifacts/hope_backhand_hopex/motion.npz"

cd "$WBT_DIR"
source setup_train_env.sh

for CKPT_NUM in 5000 5400; do
    CKPT="$RUN_DIR/model_${CKPT_NUM}.pt"
    LOG="/tmp/eval_realsensor_model${CKPT_NUM}.log"
    echo "====== Evaluating model_${CKPT_NUM} -> $LOG ======"
    hope_isaac_py scripts/eval_deterministic.py \
        task=HOPEPingPongRealSensor algo=ppo headless=true \
        num_envs=256 +steps=1200 +tail=400 \
        +noise_scales='[0.0,0.05]' \
        checkpoint="$CKPT" \
        "motion_file=[$FH,$BH]" 2>&1 | tee "$LOG"
    echo "====== model_${CKPT_NUM} done ======"
done
