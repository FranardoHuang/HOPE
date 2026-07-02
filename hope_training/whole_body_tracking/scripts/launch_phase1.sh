#!/usr/bin/env bash
# Phase 1 (widen racket target range) — warm-start from Phase-0 model_13000.
# Run INSIDE the `grasping` distrobox. Widens y/z so racket_progress has distance
# to close and footwork is forced (Phase 0 collapsed to a planted stand-and-strike:
# leg_moving_prestrike 1.0 -> 0.004). Overrides use the `task.racket.*` Hydra path
# (the YAML header's `racket.*` comment is missing the `task.` prefix).
set -euo pipefail
cd "$HOME/workspace/HOPE/hope_training/whole_body_tracking"
source setup_train_env.sh

CKPT=logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_05-00-44/model_13000.pt

hope_isaac_py scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
  checkpoint_path="$CKPT" \
  run_name=phase1_widerange_ws13000 \
  task.racket.racket_pos_y_abs_range='[0.05,0.30]' \
  task.racket.pos_z_range='[0.75,1.05]' \
  task.racket.pos_range_per_clip.forehand.y='[-0.35,-0.12]' \
  task.racket.pos_range_per_clip.forehand.z='[0.78,1.00]' \
  task.racket.pos_range_per_clip.backhand.y='[0.12,0.35]' \
  task.racket.pos_range_per_clip.backhand.z='[1.00,1.25]'
