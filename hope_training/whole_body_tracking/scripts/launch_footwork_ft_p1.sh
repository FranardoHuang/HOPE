#!/usr/bin/env bash
# Footwork fine-tune, Phase 1 — WARM-START from the 2026-07-01 champion (model_5400 of the
# hopex_realsensor scratch run) and WIDEN the per-clip racket-target boxes so the dense
# racket_progress reward starts paying for body/leg shift as arm-only reach runs out.
#
# All curriculum changes are passed on the Hydra CLI (task.racket.*) — the YAML is NOT edited.
# Same task/reward/DR/obs as the champion (HOPEPingPongRealSensor, footwork reward cfg already active);
# same explicit hopex motion clips it was trained on (so registry 'latest' can't swap to v4).
#
# Run INSIDE the `grasping` distrobox:
#   distrobox enter grasping -- bash scripts/launch_footwork_ft_p1.sh
#
# Boxes keep the SAME centers as Phase 0 (strike reference unchanged -> warm-start stays valid),
# widened outward. Lateral Y is the primary footwork axis (0.10 -> 0.36 wide, 3.6x).
set -euo pipefail
cd "$HOME/workspace/HOPE/hope_training/whole_body_tracking"
source setup_train_env.sh

RUN_DIR="logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_14-12-42_hopex_realsensor_scratch"
CKPT="$RUN_DIR/model_5400.pt"
FH="$HOME/workspace/HOPE/hope_training/whole_body_tracking/artifacts/hope_forehand_hopex/motion.npz"
BH="$HOME/workspace/HOPE/hope_training/whole_body_tracking/artifacts/hope_backhand_hopex/motion.npz"

# NOTE: train.py resolves clips from registry_name/registry_name_2 (NOT motion_file). _resolve_clip
# accepts a LOCAL motion.npz path and bypasses the wandb registry — this is how we pin the exact hopex
# clips the champion trained on (so registry 'latest' cannot swap to v4).
hope_isaac_py scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
  num_envs=6144 \
  checkpoint_path="$CKPT" \
  run_name=footwork_ft_p1_ws5400 \
  registry_name="$FH" \
  registry_name_2="$BH" \
  'task.racket.pos_range_per_clip.forehand.x=[0.40,0.56]' \
  'task.racket.pos_range_per_clip.forehand.y=[-0.56,-0.20]' \
  'task.racket.pos_range_per_clip.forehand.z=[0.78,0.96]' \
  'task.racket.pos_range_per_clip.backhand.x=[0.44,0.60]' \
  'task.racket.pos_range_per_clip.backhand.y=[-0.22,0.14]' \
  'task.racket.pos_range_per_clip.backhand.z=[0.96,1.14]'
