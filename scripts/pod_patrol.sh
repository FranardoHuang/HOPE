#!/bin/bash
# 两 pod 训练巡检:每臂 iteration 前进、窗口 legal/strike、摔率、GPU 僵死。
# 人话:只报异常和一行摘要;给 cron 每小时跑。异常判据:
#   - 训练进程存在但 10 分钟内日志无新 "Learning iteration"(卡死)
#   - GPU 有显存占用但 util<5% 且该卡无 <10min 活跃日志(僵尸)
#   - 最新窗口 legal/strike 相比上次巡检下降超过一半(崩塌预警,状态存 /tmp/pod_patrol_state)
# 用法:bash scripts/pod_patrol.sh [--full]   (--full 打印全部臂而非只报异常)
set -u
FULL=${1:-}
STATE=/tmp/pod_patrol_state; mkdir -p $STATE
declare -A PODS=( [pod1]="-p 18333 root@162.43.172.171" [pod2]="-p 13146 root@162.43.172.181" )
for pod in pod1 pod2; do
  ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519_runpod ${PODS[$pod]} bash -s "$pod" "$FULL" <<'REMOTE'
POD="$1"; FULL="$2"
echo "== $POD $(date -u +%H:%MZ) =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | while IFS=, read -r idx util mem; do
  u=$(echo $util | tr -dc 0-9); m=$(echo $mem | tr -dc 0-9)
  [ "${m:-0}" -gt 1500 ] && [ "${u:-0}" -lt 5 ] && echo "WARN gpu$idx: ${mem} 占用但 util ${util}(可能僵尸/卡死)"
done
ps -eo pid,etime,args | grep "scripts/train.py" | grep -v grep | while read -r pid etime rest; do
  rn=$(echo "$rest" | grep -o "run_name=[^ ]*" | cut -d= -f2)
  log=$(ls -t /workspace/codexschema/*/*.log /workspace/codexschema/*/*/*.log 2>/dev/null | head -40 | xargs grep -ls -- "$rn" 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    it=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$log" | tail -1)
    js=$(grep -F HOPE_EXACT_BEHAVIOR_UPDATE_JSON "$log" | tail -1)
    legal=$(echo "$js" | grep -oE '"virtual_legal_return_per_strike":[0-9.null]+' | head -1 | cut -d: -f2)
    fall=$(echo "$js" | grep -oE '"post_strike_physical_fall_rate":[0-9.null]+' | head -1 | cut -d: -f2)
    line="$rn pid=$pid $it legal/strike=${legal:-?} fall=${fall:-?} log_age=${age}s"
    if [ "$age" -gt 600 ]; then echo "WARN $line ← 日志 10 分钟未动";
    elif [ "$FULL" = "--full" ]; then echo "OK   $line"; fi
    prev_f="/tmp/patrol_${rn}_legal"; prev=$(cat "$prev_f" 2>/dev/null || echo "")
    echo "${legal:-}" > "$prev_f"
    if [ -n "$prev" ] && [ -n "${legal:-}" ] && [ "$legal" != "null" ] && [ "$prev" != "null" ]; then
      awk -v p="$prev" -v c="$legal" 'BEGIN{ if (p>0.02 && c < p/2) exit 0; exit 1 }' && echo "WARN $rn legal/strike 腰斩: $prev → $legal"
    fi
  else
    echo "WARN $rn pid=$pid 找不到日志"
  fi
done
REMOTE
done
