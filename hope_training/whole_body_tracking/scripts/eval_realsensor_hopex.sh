#!/usr/bin/env bash
# Eval realsensor/deploy-parity checkpoints deterministically.
#
# Portable: resolves the whole_body_tracking dir from its own location; every
# machine-specific value is an env override (defaults document the 2026-07-01
# hopex_realsensor_scratch run):
#
#   HOPE_EVAL_RUN_DIR=<run dir containing model_*.pt>
#   HOPE_EVAL_CKPTS="5000 5400"                    checkpoint numbers to eval
#   HOPE_EVAL_TASK=HOPEPingPongRealSensor          (or HOPEPingPongDeployParity)
#   HOPE_EVAL_FH / HOPE_EVAL_BH=<motion .npz>      default: artifacts/hope_{forehand,backhand}_hopex
#   HOPE_EVAL_LOG_DIR=/tmp                         per-checkpoint logs
#
# Run inside the GPU/Isaac environment: bash scripts/eval_realsensor_hopex.sh
set -euo pipefail

WBT_DIR="${HOPE_WBT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${HOPE_EVAL_RUN_DIR:-$WBT_DIR/logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_14-12-42_hopex_realsensor_scratch}"
CKPTS="${HOPE_EVAL_CKPTS:-5000 5400}"
TASK="${HOPE_EVAL_TASK:-HOPEPingPongRealSensor}"
FH="${HOPE_EVAL_FH:-$WBT_DIR/artifacts/hope_forehand_hopex/motion.npz}"
BH="${HOPE_EVAL_BH:-$WBT_DIR/artifacts/hope_backhand_hopex/motion.npz}"
LOG_DIR="${HOPE_EVAL_LOG_DIR:-/tmp}"

for f in "$FH" "$BH"; do
    [ -f "$f" ] || { echo "[FATAL] motion file not found: $f (set HOPE_EVAL_FH/HOPE_EVAL_BH)"; exit 1; }
done
[ -d "$RUN_DIR" ] || { echo "[FATAL] run dir not found: $RUN_DIR (set HOPE_EVAL_RUN_DIR)"; exit 1; }

cd "$WBT_DIR"
source setup_train_env.sh

for CKPT_NUM in $CKPTS; do
    CKPT="$RUN_DIR/model_${CKPT_NUM}.pt"
    [ -f "$CKPT" ] || { echo "[FATAL] checkpoint not found: $CKPT (set HOPE_EVAL_RUN_DIR/HOPE_EVAL_CKPTS)"; exit 1; }
    LOG="$LOG_DIR/eval_realsensor_model${CKPT_NUM}.log"
    echo "====== Evaluating model_${CKPT_NUM} (task=$TASK) -> $LOG ======"
    hope_isaac_py scripts/eval_deterministic.py \
        task="$TASK" algo=ppo headless=true \
        num_envs=256 +steps=1200 +tail=400 \
        +noise_scales='[0.0,0.05]' \
        checkpoint="$CKPT" \
        "motion_file=[$FH,$BH]" 2>&1 | tee "$LOG"
    echo "====== model_${CKPT_NUM} done ======"
done
