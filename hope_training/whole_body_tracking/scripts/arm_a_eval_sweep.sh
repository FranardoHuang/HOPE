#!/usr/bin/env bash
# ARM-A verification sweep: deterministic (pure-mean = deployed) eval across the
# retrain checkpoint spread + the pre-retrain baseline (model_11400). Proves the
# deterministic policy did NOT regress despite the std-inflated W&B rollout curve.
# Each checkpoint = one Isaac boot (~3-4 min). Sequential; logs per checkpoint.
set -u
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh >/dev/null 2>&1
FH=$PWD/artifacts/hope_forehand_hopex/motion.npz
BH=$PWD/artifacts/hope_backhand_hopex/motion.npz
RUN=logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-04_19-09-21
BASE=logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-03_13-32-07
OUT=/tmp/arm_a_eval; mkdir -p $OUT
declare -A CK=(
  [11400_baseline]=$BASE/model_11400.pt
  [13300]=$RUN/model_13300.pt
  [14300]=$RUN/model_14300.pt
  [14600]=$RUN/model_14600.pt
  [15600]=$RUN/model_15600.pt
)
for name in 11400_baseline 13300 14300 14600 15600; do
  ck=${CK[$name]}
  echo "===== eval $name ($ck) =====" | tee -a $OUT/summary.txt
  hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPongDeployParity algo=ppo \
    headless=true num_envs=128 +steps=1200 +tail=400 +noise_scales=0.0 \
    checkpoint="$ck" "motion_file=[$FH,$BH]" >$OUT/$name.log 2>&1
  # pull the EVAL block
  sed -n '/^EVAL |/,/^====/p' $OUT/$name.log | tee -a $OUT/summary.txt
  echo "" | tee -a $OUT/summary.txt
done
echo "ALL DONE" | tee -a $OUT/summary.txt
