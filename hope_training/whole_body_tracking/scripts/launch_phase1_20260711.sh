#!/usr/bin/env bash
# Reproducible launcher for the 2026-07-11 Phase-1 causal controls and fresh v3 seeds.
# This is simulation-only. It never invokes a robot/deploy command path.

set -euo pipefail

role=${1:-}
case "$role" in
  smoke|pod1|pod2|pod1_scaleout_2|pod1_scaleout_3|pod1_scaleout_4|pod2_scaleout_2|pod2_scaleout_3|pod2_scaleout_4) ;;
  *)
    echo "usage: $0 {smoke|pod1|pod2|pod1_scaleout_{2,3,4}|pod2_scaleout_{2,3,4}}" >&2
    exit 2
    ;;
esac

artifact_root=${PHASE1_ARTIFACT_ROOT:-/workspace/codexschema/phase1_fresh_20260711}
repo_root=${PHASE1_REPO_ROOT:-/workspace/codexschema/nohope}
wbt_root="$repo_root/hope_training/whole_body_tracking"
python_bin=${HOPE_ISAAC_PYTHON:-/workspace/hope_isaac_venv/bin/python}
env_file=${PHASE1_ENV_FILE:-/workspace/codexschema/env.sh}
locked_launcher="$wbt_root/scripts/launch_kit_training_locked.sh"

[[ -f $env_file ]] || { echo "missing environment file: $env_file" >&2; exit 1; }
[[ -x $python_bin ]] || { echo "missing Isaac Python: $python_bin" >&2; exit 1; }
[[ -x $locked_launcher ]] || { echo "missing locked launcher: $locked_launcher" >&2; exit 1; }

# shellcheck disable=SC1090
source "$env_file"
cd "$wbt_root"
if [[ -n $(git status --porcelain) ]]; then
  echo "refusing Phase-1 launch from a dirty Pod checkout" >&2
  git status --short >&2
  exit 1
fi

training_commit=$(git rev-parse HEAD)
case "$role" in
  pod1_scaleout_*|pod2_scaleout_*)
    expected_training_commit=6d93bcb16c422a2f42748c2dc99432559653480b
    if [[ $training_commit != "$expected_training_commit" ]]; then
      echo "refusing scale-out from the wrong frozen training checkout" >&2
      echo "expected=$expected_training_commit actual=$training_commit" >&2
      exit 1
    fi
    ;;
esac

mkdir -p "$artifact_root/runs"
printf '%s\n' "$training_commit" >"$artifact_root/runs/code_commit.txt"

verify_file() {
  local path=$1 expected=$2 actual
  [[ -f $path ]] || { echo "missing launch input: $path" >&2; exit 1; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ $actual != "$expected" ]]; then
    echo "launch input hash mismatch: $path" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 1
  fi
}

fresh_dir="$artifact_root/assets/v4rg_runtime_order_v3"
fresh_fh="$fresh_dir/hope_forehand_v4rg_cal.npz"
fresh_bh="$fresh_dir/hope_backhand_v4rg_cal.npz"
fresh_bank="$fresh_dir/s1_v4rg_runtime_order_schema3_train.npz"
verify_fresh_inputs() {
  verify_file "$fresh_fh" f2cb2d9f5d27cefbcee0b790000fcd979abaf02894d4fcad061ebca27f141687
  verify_file "$fresh_bh" 1722553375cd28f9b2d567c01b1a5fc6bcd149fa12cadb20e5202a9153367534
  verify_file "$fresh_bank" 2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700
}

# Shared recipe pins. The causal pairs differ only in face_command_pairing within each family.
base_recipe=(
  task=HOPEPingPongVirtualBall
  algo=ppo
  headless=true
  logger=tensorboard
  video=false
  device=cuda:0
  task.actor_obs_contract=deploy_parity_face179
  task.env.episode_length_s=10.0
  task.sim.dt=0.005
  task.sim.decimation=4
  task.actions.qdes_clamp=true
  task.motion.wrap_teleport=false
  task.motion.stand_start_prob=0.25
  "task.motion.hold_steps_range=[0,100]"
  task.motion.stand_start_min_hold=25
  task.motion.post_swing_start_prob=0.25
  task.motion.post_swing_buffer_size=4096
  task.motion.post_swing_min_fill=256
  task.motion.post_swing_min_hold=25
  task.motion.clip_switch_prob=0.0
  "task.motion.speed_scale_range=[1.0,1.0]"
  ++task.motion.rsi_skip_settle_frames=0
  ++task.motion.rsi_hold_root_stand_z=false
  ++task.motion.stagger_initial_clock=false
  ++task.motion.stagger_hold_max_steps=150
  ++task.racket.question_bank_allow_legacy=false
  ++task.racket.face_command=true
  ++task.racket.face_command_obs=true
  ++task.racket.station_obs=false
  "++task.racket.mount_normal_sign_per_clip=[1.0,-1.0]"
  task.racket.target_mode=uniform
  task.racket.normal_mode=velocity
  task.racket.adaptive_sigma=true
  task.racket.target_delay_steps=2
  task.racket.target_jitter_pos_per_s=0.0
  task.racket.target_jitter_vel_per_s=0.0
  task.racket.target_noise_white=0.0019
  task.racket.target_noise_ar1_sigma=0.0052
  task.racket.target_noise_ar1_rho=0.717
  task.racket.target_dropout_prob=0.0
  task.racket.target_post_strike_dropout_s=0.0
  task.racket.target_bias_per_swing=0.0
  task.racket.midswing_resample_prob=0.0
  task.racket.midswing_resample_tts_floor=0.3
  task.racket.vb_spin_mode=minimize
  task.racket.vb_metrics_only=true
  ++task.racket.rally_legacy_metrics=true
  task.racket.achieved_target_mix_prob=0.30
  task.racket.achieved_buffer_size=4096
  task.racket.achieved_min_fill=256
  task.racket.achieved_jitter_pos=0.03
  task.racket.achieved_jitter_vel=0.15
  task.racket.achieved_clamp_inflate=0.20
  task.racket.clean_reference_strike_velocity=true
  task.racket.clean_strike_vel_window=2
  task.racket.strike_window_s=0.12
  task.racket.strike_success_pos_thresh=0.075
  ++task.physical_ball=true
  task.rewards.racket_position_weight=14.0
  task.rewards.racket_position_std=0.2
  task.rewards.racket_velocity_weight=10.0
  task.rewards.racket_velocity_std=1.0
  task.rewards.racket_normal_weight=5.0
  task.rewards.racket_normal_std=0.3
  task.rewards.foot_orientation_weight=-0.3
  task.rewards.base_decel_weight=0.0
  task.rewards.free_wrist_ori_mimic=true
  ++task.rewards.free_wrist_vel_mimic=false
  ++task.rewards.racket_face_guidance_weight=0.0
)

launch_arm() {
  local arm=$1 gpu=$2 marker=$3
  shift 3
  local run_dir="$artifact_root/runs/$arm"
  local log_file="$run_dir/run.log"
  if [[ -e ${log_file}.launch ]]; then
    echo "refusing duplicate launch; state already exists: ${log_file}.launch" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  if [[ ${PHASE1_DRY_RUN:-0} == 1 ]]; then
    printf 'PHASE1_DRY_RUN arm=%q gpu=%q log=%q command=' "$arm" "$gpu" "$log_file"
    printf '%q ' env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      "$python_bin" scripts/train.py "${base_recipe[@]}" "$@"
    printf '\n'
    return 0
  fi
  KIT_BOOT_MARKER="$marker" KIT_BOOT_TIMEOUT_S=900 KIT_BOOT_POLL_S=5 \
    "$locked_launcher" "$log_file" \
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    "$python_bin" scripts/train.py "${base_recipe[@]}" "$@"
}

launch_fresh_variant() {
  local arm=$1 gpu=$2 seed=$3 iterations=$4 marker=$5 pairing=$6 zero_friction=$7
  verify_fresh_inputs
  launch_arm "$arm" "$gpu" "$marker" \
    seed="$seed" num_envs="$([[ $role == smoke ]] && echo 32 || echo 4096)" \
    max_iterations="$iterations" run_name="$arm" \
    checkpoint_tolerant=false checkpoint_allow_missing_contract=false \
    checkpoint_allow_contract_mismatch=false \
    task.plant.zero_joint_friction="$zero_friction" \
    ++task.motion.allow_legacy_link_origin_velocity=false \
    motion_file="$fresh_fh" motion_file_2="$fresh_bh" \
    "task.racket.strike_phase_per_clip=[0.471,0.338]" \
    ++task.racket.question_bank="$fresh_bank" \
    ++task.racket.face_command_pairing="$pairing" \
    ++task.rewards.racket_guidance_weight=0.0
}

launch_fresh() {
  launch_fresh_variant "$1" "$2" "$3" "$4" "$5" shared_plus_y true
}

launch_causal_seeded() {
  local arm=$1 gpu=$2 seed=$3 checkpoint=$4 fh=$5 bh=$6 bank=$7 phases=$8 pairing=$9
  local guidance=${10}
  shift 10
  verify_file "$checkpoint" "$1"
  verify_file "$fh" "$2"
  verify_file "$bh" "$3"
  verify_file "$bank" "$4"
  launch_arm "$arm" "$gpu" "Learning iteration" \
    seed="$seed" num_envs=4096 max_iterations=4000 run_name="$arm" \
    checkpoint_path="$checkpoint" checkpoint_tolerant=false \
    checkpoint_allow_missing_contract=true checkpoint_allow_contract_mismatch=false \
    task.plant.zero_joint_friction=false \
    ++task.motion.allow_legacy_link_origin_velocity=true \
    motion_file="$fh" motion_file_2="$bh" \
    "task.racket.strike_phase_per_clip=$phases" \
    ++task.racket.question_bank="$bank" \
    ++task.racket.face_command_pairing="$pairing" \
    ++task.rewards.racket_guidance_weight="$guidance"
}

launch_causal() {
  local arm=$1 gpu=$2
  shift 2
  launch_causal_seeded "$arm" "$gpu" 1 "$@"
}

stagger_scaleout() {
  if [[ ${PHASE1_DRY_RUN:-0} != 1 ]]; then
    sleep "${PHASE1_STAGGER_S:-75}"
  fi
}

launch_m3_causal_seeded() {
  local arm=$1 gpu=$2 seed=$3 pairing=$4
  launch_causal_seeded "$arm" "$gpu" "$seed" \
    "$artifact_root/parents/M3c/model_16999.pt" \
    "$artifact_root/assets/legacy/swing/motions/hope_forehand_swing_cal.npz" \
    "$artifact_root/assets/legacy/swing/motions/hope_backhand_swing_cal.npz" \
    "$artifact_root/assets/legacy/swing/s1_swing_schema3_train.npz" \
    '[0.371,0.495]' "$pairing" -0.95 \
    46f0050589f3343d96f2e5c261b92224079b379e4da473be342b1bd0f0cf7ff1 \
    1c53405f19733a34db4a2f10bd61b4768002afce0face83c492e12171bc3ddc9 \
    875c41db05f964e57221617b5b1ade6d64d161511e4107e3dc5fab726c52c15b \
    96329c79f13e659c035bf65bafedf84123d23a3219b02ac78ae654aed930cf60
}

launch_m2_causal_seeded() {
  local arm=$1 gpu=$2 seed=$3 pairing=$4
  launch_causal_seeded "$arm" "$gpu" "$seed" \
    "$artifact_root/parents/M2f/model_16999.pt" \
    "$artifact_root/assets/legacy/v4rg/motions/hope_forehand_v4rg_cal.npz" \
    "$artifact_root/assets/legacy/v4rg/motions/hope_backhand_v4rg_cal.npz" \
    "$artifact_root/assets/legacy/v4rg/s1_v4rg_schema3_train.npz" \
    '[0.471,0.338]' "$pairing" 0.0 \
    0ab05144ec1792db91d6e1e3c2ce79f46dae9507ed267b1470838bf998f0f012 \
    34f130101a539981e83545af83c27f5b5b512d2f2b475d6c6ee44161a3bd21e7 \
    d98a337a4131fb8073433e646e8a74c1e9782e04e9afde2a3879cda259b55ee7 \
    dc67326f5cf0e1a3e3a6ae89f43cbd8a9a3785c341ab2ae998e3edbd40f8d30a
}

case "$role" in
  smoke)
    launch_fresh phase1_fresh_v3_179_smoke 2 1 0 "[train.py] hard training contract:"
    ;;
  pod1)
    m3_parent="$artifact_root/parents/M3c/model_16999.pt"
    m3_fh="$artifact_root/assets/legacy/swing/motions/hope_forehand_swing_cal.npz"
    m3_bh="$artifact_root/assets/legacy/swing/motions/hope_backhand_swing_cal.npz"
    m3_bank="$artifact_root/assets/legacy/swing/s1_swing_schema3_train.npz"
    launch_causal phase1_M3_old_pairing 0 "$m3_parent" "$m3_fh" "$m3_bh" "$m3_bank" \
      '[0.371,0.495]' legacy_signed_vs_A -0.95 \
      46f0050589f3343d96f2e5c261b92224079b379e4da473be342b1bd0f0cf7ff1 \
      1c53405f19733a34db4a2f10bd61b4768002afce0face83c492e12171bc3ddc9 \
      875c41db05f964e57221617b5b1ade6d64d161511e4107e3dc5fab726c52c15b \
      96329c79f13e659c035bf65bafedf84123d23a3219b02ac78ae654aed930cf60
    launch_causal phase1_M3_S1_pairing 1 "$m3_parent" "$m3_fh" "$m3_bh" "$m3_bank" \
      '[0.371,0.495]' shared_plus_y -0.95 \
      46f0050589f3343d96f2e5c261b92224079b379e4da473be342b1bd0f0cf7ff1 \
      1c53405f19733a34db4a2f10bd61b4768002afce0face83c492e12171bc3ddc9 \
      875c41db05f964e57221617b5b1ade6d64d161511e4107e3dc5fab726c52c15b \
      96329c79f13e659c035bf65bafedf84123d23a3219b02ac78ae654aed930cf60
    launch_fresh phase1_fresh_v3_S1_seed1 2 1 17000 "Learning iteration"
    ;;
  pod2)
    m2_parent="$artifact_root/parents/M2f/model_16999.pt"
    m2_fh="$artifact_root/assets/legacy/v4rg/motions/hope_forehand_v4rg_cal.npz"
    m2_bh="$artifact_root/assets/legacy/v4rg/motions/hope_backhand_v4rg_cal.npz"
    m2_bank="$artifact_root/assets/legacy/v4rg/s1_v4rg_schema3_train.npz"
    launch_causal phase1_M2_old_pairing 0 "$m2_parent" "$m2_fh" "$m2_bh" "$m2_bank" \
      '[0.471,0.338]' legacy_signed_vs_A 0.0 \
      0ab05144ec1792db91d6e1e3c2ce79f46dae9507ed267b1470838bf998f0f012 \
      34f130101a539981e83545af83c27f5b5b512d2f2b475d6c6ee44161a3bd21e7 \
      d98a337a4131fb8073433e646e8a74c1e9782e04e9afde2a3879cda259b55ee7 \
      dc67326f5cf0e1a3e3a6ae89f43cbd8a9a3785c341ab2ae998e3edbd40f8d30a
    launch_causal phase1_M2_S1_pairing 1 "$m2_parent" "$m2_fh" "$m2_bh" "$m2_bank" \
      '[0.471,0.338]' shared_plus_y 0.0 \
      0ab05144ec1792db91d6e1e3c2ce79f46dae9507ed267b1470838bf998f0f012 \
      34f130101a539981e83545af83c27f5b5b512d2f2b475d6c6ee44161a3bd21e7 \
      d98a337a4131fb8073433e646e8a74c1e9782e04e9afde2a3879cda259b55ee7 \
      dc67326f5cf0e1a3e3a6ae89f43cbd8a9a3785c341ab2ae998e3edbd40f8d30a
    launch_fresh phase1_fresh_v3_S1_seed2 2 2 17000 "Learning iteration"
    ;;
  pod1_scaleout_2)
    launch_m3_causal_seeded phase1_M3_S1_pairing_seed2 0 2 shared_plus_y
    stagger_scaleout
    launch_m3_causal_seeded phase1_M3_old_pairing_seed2 1 2 legacy_signed_vs_A
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_SP_seed1 2 1 17000 "Learning iteration" shared_plus_y false
    ;;
  pod1_scaleout_3)
    launch_fresh_variant phase1_fresh_v3_SZ_seed3 0 3 17000 "Learning iteration" shared_plus_y true
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_SP_seed3 1 3 17000 "Learning iteration" shared_plus_y false
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LZ_seed1 2 1 17000 "Learning iteration" legacy_signed_vs_A true
    ;;
  pod1_scaleout_4)
    launch_fresh_variant phase1_fresh_v3_LP_seed3 0 3 17000 "Learning iteration" legacy_signed_vs_A false
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LZ_seed3 1 3 17000 "Learning iteration" legacy_signed_vs_A true
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LP_seed1 2 1 17000 "Learning iteration" legacy_signed_vs_A false
    ;;
  pod2_scaleout_2)
    launch_m2_causal_seeded phase1_M2_S1_pairing_seed2 0 2 shared_plus_y
    stagger_scaleout
    launch_m2_causal_seeded phase1_M2_old_pairing_seed2 1 2 legacy_signed_vs_A
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_SP_seed2 2 2 17000 "Learning iteration" shared_plus_y false
    ;;
  pod2_scaleout_3)
    launch_fresh_variant phase1_fresh_v3_SP_seed4 0 4 17000 "Learning iteration" shared_plus_y false
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_SZ_seed4 1 4 17000 "Learning iteration" shared_plus_y true
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LZ_seed2 2 2 17000 "Learning iteration" legacy_signed_vs_A true
    ;;
  pod2_scaleout_4)
    launch_fresh_variant phase1_fresh_v3_LZ_seed4 0 4 17000 "Learning iteration" legacy_signed_vs_A true
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LP_seed4 1 4 17000 "Learning iteration" legacy_signed_vs_A false
    stagger_scaleout
    launch_fresh_variant phase1_fresh_v3_LP_seed2 2 2 17000 "Learning iteration" legacy_signed_vs_A false
    ;;
esac
