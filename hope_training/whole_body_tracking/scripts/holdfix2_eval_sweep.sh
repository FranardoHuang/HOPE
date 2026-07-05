#!/usr/bin/env bash
# holdfix2 sweep: deterministic eval across the 2026-07-05_13-04-02 plateau
# (resume of model_16400 w/ entropy_coef=0.003; std DEflated 0.32->0.26, so late
# checkpoints are candidates). model_16400 = already-gated baseline for reference.
set -u
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh >/dev/null 2>&1
FH=$PWD/artifacts/hope_forehand_hopex/motion.npz
BH=$PWD/artifacts/hope_backhand_hopex/motion.npz
BASE=logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-05_02-41-10
RUN=logs/rsl_rl/agibot_a3_hope_deploy_parity/2026-07-05_13-04-02
OUT=/tmp/holdfix2_eval; mkdir -p $OUT
declare -A CK=(
  [16400_baseline]=$BASE/model_16400.pt
  [19400]=$RUN/model_19400.pt
  [21400]=$RUN/model_21400.pt
  [23000]=$RUN/model_23000.pt
  [23700]=$RUN/model_23700.pt
)
for name in 16400_baseline 19400 21400 23000 23700; do
  ck=${CK[$name]}
  echo "===== eval $name ($ck) =====" | tee -a $OUT/summary.txt
  hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPongDeployParity algo=ppo \
    headless=true num_envs=128 +steps=1200 +tail=400 +noise_scales=0.0 \
    checkpoint="$ck" "motion_file=[$FH,$BH]" >$OUT/$name.log 2>&1
  sed -n '/^EVAL |/,/^====/p' $OUT/$name.log | tee -a $OUT/summary.txt
  echo "" | tee -a $OUT/summary.txt
done
echo "ALL DONE" | tee -a $OUT/summary.txt
