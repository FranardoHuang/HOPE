# 直接运行 0.5 秒 K100

这是一条简单的行为测试：100 道固定题都从动作第 0 帧的零速度状态开始，给策略
0.5 秒（50 Hz 下 25 个控制周期）完成触球。正手、反手各 50 题；未触球、来不及、
摔倒和未上台都保留在分母。

当前结论：2026-07-18 在 Pod2 对 `taskrev_p2_equal_reward@model_5700` 的第一次完整直跑已经结束。
100 题全部完成，触球 `0/100`、回台 `0/100`、物理摔倒 `0/100`，两侧各有 50 次超时。
这说明该策略在这条严格 0.5 秒诊断卷上来不及击球；结果是 Isaac 诊断，不替代 vendor MuJoCo。

## 原则

- 不再为一次测试制作 v1/v2/v3、activation、receipt 或人工 SHA 对拍。
- 直接运行 evaluator；每次只换一个新的输出目录，旧输出不覆盖。
- 启动前只确认三件事：checkpoint 存在、题表存在、目标 GPU 没有别人的进程。
- 成功与否看输出 JSON 的 `status` 和 100 道题是否完整，不把 Kit 关闭阶段的 shell 返回码当成行为结论。
- 不发真机命令，不按进程名宽泛停止；若需要停止，只管理本次记录的 PID/PGID。

## Pod2 直接命令

下面命令不要求操作者提供任何 SHA。`OUT` 必须是新目录，`GPU` 必须是实际空闲卡。

```bash
SRC=/workspace/codexschema/nohope_eval_simple_20260718
RUN=/workspace/codexschema/nohope_task_revision_b1f5a38/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope_virtualball/2026-07-16_20-04-47_phase1_taskrev_p2_equal_reward_seed3_20260716
BANK=/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_exam_bank_rebind_v1/s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz
SCHEDULE=/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json
TIMING=/workspace/codexschema/phase1_task_revision_supercombo_20260716/papers/timing_exam_0p5_k100.schedule.json
OUT=/workspace/codexschema/simple_exact_0p5_20260718/next_attempt
GPU=1

test ! -e "$OUT"
mkdir -p "$OUT"
source "$SRC/hope_training/whole_body_tracking/setup_train_env.sh"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$HOPE_WBT_PYTHONPATH" \
  /workspace/hope_isaac_venv/bin/python \
  "$SRC/hope_training/whole_body_tracking/scripts/isaac_bank_exam.py" \
  task=HOPEPingPongVirtualBall headless=true device=cuda:0 \
  "+run_dir=$RUN" "+checkpoint=$RUN/model_5700.pt" \
  "+exam_bank=$BANK" "+schedule_json=$SCHEDULE" \
  +per_clip_quota=50 +schedule_seed=0 +noise_scale=0.0 \
  +allow_inexact_contract=true "+timing_paper=$TIMING" \
  "+output_dir=$OUT" +output_stem=isaac_timing_0p5 \
  >"$OUT/evaluator.log" 2>&1
```

需要后台运行时，用普通的 `nohup setsid` 记录本次 PID；不要再包一层版本化 launcher。

## 最小验收

```bash
python3 - "$OUT/isaac_timing_0p5.json" <<'PY'
import json, sys

result = json.load(open(sys.argv[1]))
attempts = result["attempts"]
assert result["status"] == "valid"
assert len(attempts) == 100
assert sum(bool(row["finalized"]) for row in attempts) == 100
assert sum(bool(row["censored"]) for row in attempts) == 0
print(result["summary"])
PY
```

结果只回答“这个 checkpoint 在这 100 道 0.5 秒题上做到了什么”。后续训练应直接针对
准备时间分布、动作加速和同一拍中的目标更新，不再为这份负结果重复建 harness。

权威实验记录见
[EXP-P1-TASK-REVISION-0P5-K100](../experiments/2026-07/EXP-P1-TASK-REVISION-0P5-K100.md)。
