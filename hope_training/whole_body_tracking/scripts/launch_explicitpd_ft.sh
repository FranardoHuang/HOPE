#!/usr/bin/env bash
# EXPLICIT-PD sim2real fine-tune (2026-07-02) — WARM-START from the p4 deploy-parity champion
# (model_12100) so the fine-tuned policy survives AGI's EXPLICIT clipped-PD MuJoCo (== real).
#
# What changed in the training code for this fine-tune (already applied, NOT via CLI):
#   CHANGE 1  arms/waist/feet -> IdealPDActuatorCfg (explicit clipped PD, effort_limit set so it CLIPS);
#             legs/head stay implicit. (source/.../robots/agibot_a3.py)
#   CHANGE 2  arm_torque_saturation reward: penalize over-limit COMPUTED effort on arm+waist joints.
#   CHANGE 3  prestrike_upright reward: penalize forward base tilt (proj_grav_xy) through the approach.
#   CHANGE 4  PD-gain DR +/-15% enabled (pd_gain_range) + link mass +/-15%. (HOPEPingPongDeployParity.yaml)
#
# The p4 checkpoint was trained with WIDER per-clip target boxes than the YAML PHASE-0 defaults, so we
# re-pass those exact boxes on the CLI (train.py otherwise applies the narrow YAML boxes and the target
# distribution collapses -> warm-start invalid). Velocity boxes already match the YAML.
#
# Halved LR (0.001 -> 0.0005) for a gentle fine-tune restart; ~0.5-1.5M steps recommended.
# Run INSIDE the training distrobox (e.g. `distrobox enter grasping -- bash scripts/launch_explicitpd_ft.sh`).
# DO NOT run unattended without watching the metrics below.
set -euo pipefail
cd "$HOME/workspace/HOPE/hope_training/whole_body_tracking"
source setup_train_env.sh

CKPT="logs/rsl_rl/agibot_a3_hope_realsensor/2026-07-01_23-12-20_footwork_ft_p4_swingreach/model_12100.pt"
FH="$HOME/workspace/HOPE/hope_training/whole_body_tracking/artifacts/hope_forehand_hopex/motion.npz"
BH="$HOME/workspace/HOPE/hope_training/whole_body_tracking/artifacts/hope_backhand_hopex/motion.npz"

hope_isaac_py scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
  num_envs=6144 \
  checkpoint_path="$CKPT" \
  run_name=explicitpd_ft \
  registry_name="$FH" \
  registry_name_2="$BH" \
  algo.algorithm.learning_rate=0.0005 \
  task.rewards.arm_torque_saturation_weight=-0.5 \
  task.rewards.prestrike_upright_weight=-1.0 \
  'task.racket.pos_range_per_clip.forehand.x=[0.35,0.62]' \
  'task.racket.pos_range_per_clip.forehand.y=[-0.95,-0.10]' \
  'task.racket.pos_range_per_clip.forehand.z=[0.72,1.02]' \
  'task.racket.pos_range_per_clip.backhand.x=[0.40,0.66]' \
  'task.racket.pos_range_per_clip.backhand.y=[-0.30,0.52]' \
  'task.racket.pos_range_per_clip.backhand.z=[0.90,1.20]'

# WATCH (eval_deterministic + W&B): strike_composite >= 0.9 and racket_pos_error_exact_strike <= 0.04 m
#   (the strike must NOT regress); eplen recovering toward ~500; arm_torque_sat_frac falling toward 0
#   (the swing moving inside the torque envelope); proj_grav_xy / base tilt staying low pre-strike.
