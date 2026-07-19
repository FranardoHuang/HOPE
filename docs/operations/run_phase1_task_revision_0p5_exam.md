# 直接运行 0.5 秒 K100

这是一条简单的行为测试：100 道固定题都从动作第 0 帧的零速度状态开始，给策略
0.5 秒（50 Hz 下 25 个控制周期）完成触球。正手、反手各 50 题；未触球、来不及、
摔倒和未上台都保留在分母。
这里的 [`K100`](../DEFINITIONS.md#q50-and-k100) 是固定顺序的 100 道同卷题；完整人话合同见
[0.5 秒时序卷](../DEFINITIONS.md#timing-exam-0p5)。

当前结论：2026-07-18 在 Pod2 对 `taskrev_p2_equal_reward@model_5700` 的第一次完整直跑已经结束。
100 题全部完成，触球 `0/100`、回台 `0/100`、物理摔倒 `0/100`，两侧各有 50 次超时。
这说明该策略在这条严格 0.5 秒诊断卷上来不及击球；结果是 Isaac 诊断，不替代 vendor MuJoCo。

2026-07-19 当前边界：`W`（拍心优先 × 自由非击球臂）和 `Y`（拍心优先 × 触球窗老师静音）
只是厂商 MuJoCo 同卷的演示优先候选，`U`（拍心优先 × 强准备）是稳定备选。现有路径尚不能运行
W/Y 厂商行为卷；三轮只读定位仍未唯一找到两份 `model_6700`，导出 preflight（导出前置检查）尚未开始。

### W/Y checkpoint 只读定位边界

三轮 Pod1 只读连接均 exit `0` 且未重连。W/Y 各有唯一完整 `run_name` 的 wrapper `run.log`，但日志
没有可唯一解析的 RSL/checkpoint 绝对路径；cached-source 受限根中也没有能由相邻材料精确归属的
`model_6700.pt`。`train.py` 以 launcher 启动工作目录生成 `logs/rsl_rl/...`，Hydra 不改变它，而 sprint
配置没记录该目录。第三轮确认每臂各有唯一 regular `run.sh`，但静态解析得不到可接受的绝对 cwd。
下一轮不再猜 cwd：只在 `/workspace/codexschema` 单一文件系统内枚举 regular `model_6700.pt`，并按
parent basename 精确以 `_<完整 run_name>` 结尾筛选。仍不唯一就保持 `UNKNOWN`，不得按“最新”猜。

现有 `standalone_onnx_export.py` 没有真正的 `--plan`/`--dry-run`：完整调用会创建输出目录并替换
`policy.onnx`，`--help` 与 import smoke 也不验证 W/Y 材料。checkpoint 唯一闭合前不得运行；闭合后先
补零写入 plan，或把正式 W/Y 导出分别写到全新独立目录。

## 为什么当前不能直接跑 W/Y 厂商同卷

- 下方 Isaac 命令用 `isaac_bank_exam.py` 直接驱动 policy，绕过生产规划器（planner），因此只能复现
  已完成的 `model_5700` 诊断。
- Python `mujoco_eval_onnx.py` 支持 179 维模型和 bank（固定题库），但不消费逐题 25 周期的 timing
  paper（时序卷），也不把每题送进生产 planner。
- Gate3 假球入口只接扁平的 `N × 6` 发球列表，每题为初始位置与初始速度六元组；当前缺少
  “K100 时序卷 → 发球 → [同球实时任务修订](../DEFINITIONS.md#planner-task-revision) → 生产 planner”
  适配器。
- 旧 `pp_gate3_rally.sh` / `pp_rally_conductor.py` 保持隔离禁用，禁止用其启动或清理逻辑拼接本卷。

在适配器落地前，不启动厂商行为。适配器必须保持同一 100 题、每侧 50 题、每题第 0 帧零速度、
50 Hz 下 25 个控制周期、正手倍率 `2.64`、反手倍率 `1.8`，并贯穿同一生产 planner、同一 MuJoCo
XML 场景模型（MJCF）和同一执行 plant（执行器、比例微分控制与时间步配置）。输出必须逐题包含
`attempt`（尝试/题号）、`completion`（动作完成）、`hit`（物理触球）、`return`（合法回台）、
`fall`（摔倒）与 `deadline`（截止）字段，失败题不删。

当前详细边界见[半秒冲刺记录](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md)，Gate 状态见
[G06](../gates/G06_isaac_to_mujoco.md)。G05/G06 保持 `Partial`，`Gate3-D0` 保持 `Open`；没有厂商
演示结果。

## 已完成 Isaac 诊断的原则（当前不发射）

- 不再为一次测试制作 v1/v2/v3、activation、receipt 或人工 SHA 对拍。
- 直接运行 evaluator；每次只换一个新的输出目录，旧输出不覆盖。
- 启动前只确认三件事：checkpoint 存在、题表存在、目标 GPU 没有别人的进程。
- 成功与否看输出 JSON 的 `status` 和 100 道题是否完整，不把 Kit 关闭阶段的 shell 返回码当成行为结论。
- 不发真机命令，不按进程名宽泛停止；若需要停止，只管理本次记录的 PID/PGID。

## 已完成的 Pod2 Isaac 复现命令（当前阶段不执行）

下面命令只保留已完成的 `model_5700` Isaac 诊断复现方法；当前阶段不连接 Pod2，也不把它改成
W/Y 厂商命令。`OUT` 必须是新目录，`GPU` 必须是实际空闲卡。

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
