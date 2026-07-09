#!/usr/bin/env bash
# =============================================================================
# G2 in-sim RECOVERY eval (rally v3, 2026-07-08).
#
# The normal rally metric base_heading_abs_at_swing_start is diluted ~0 by near-square wrap holds.
# This eval CONCENTRATES yawed holds — stand_start_prob=1.0 + stand_start_yaw_range=[-0.9,0.9] + a
# short episode — so nearly every hold expiry is a yawed-stand recovery, making the spawn-conditioned
# heading_recovery_expiry_yaw (added to eval_deterministic REPORT_ROWS) the load-bearing number.
#
# READ THE OUTPUT:
#   heading_recovery_spawn_yaw (cond)   ~ mean |yaw| the holds START at (should be ~0.45 rad given +-0.9)
#   heading_recovery_expiry_yaw (GATE)  ~ mean |yaw| when the swing ARMS after the hold.
#   PASS (recovery works): expiry << spawn AND expiry < ~0.30 rad (deploy engage gate is 0.35 rad / 20 deg).
#   FAIL: expiry ~= spawn (no turning) or expiry > 0.30.
# NB: Isaac does not reproduce the deploy over-rotation, so this is a PROXY — the decisive gate is
#     pp_gate3_rally.sh (rescues==0). Also sanity-check composite here isn't wrecked by the yawed starts.
#
# Usage:  bash scripts/pp_recovery_eval.sh  <checkpoint.pt>  [num_envs]
#   run from .../hope_training/whole_body_tracking, inside the `grasping` distrobox.
# =============================================================================
set -euo pipefail
CKPT="${1:?usage: pp_recovery_eval.sh <checkpoint.pt> [num_envs]}"
NENV="${2:-256}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup_train_env.sh >/dev/null 2>&1

hope_isaac_py scripts/eval_deterministic.py \
  task=HOPEPingPongHitterPureRallyV3 algo=ppo headless=true \
  num_envs="${NENV}" +steps=1500 +tail=1000 +noise_scales=0.0 \
  task.env.episode_length_s=5.0 \
  task.motion.stand_start_prob=1.0 \
  'task.motion.stand_start_yaw_range=[-0.9,0.9]' \
  checkpoint="${CKPT}"
