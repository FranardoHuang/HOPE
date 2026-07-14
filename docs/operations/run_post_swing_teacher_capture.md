# 生成随挥结束教师状态制品

状态：**exact capture preregistered / training NO-LAUNCH。** 本操作只用于仿真推理和后续训练 cold-start，
绝不下发真机命令。源码专项与攻击负测通过；首个 exact 实例见
[`phase1_post_swing_teacher_capture_prereg_20260715.json`](../../configs/phase1_post_swing_teacher_capture_prereg_20260715.json)，
只授权一次 Pod2 GPU1 inference-only capture。4096-environment capture、attestation 和首 reset readback
probe 尚未完成，因此 scientific trainer、第二 seed、judge 与 promotion 仍未授权。

合同真源见 [随挥结束教师状态接口](../interfaces/post_swing_teacher_artifact.md)，当前科学动机见
[base-deceleration measurement rerun](../experiments/2026-07/EXP-P1-V1V2-BASE-DECEL-MEASUREMENT-RERUN.md)。
以下每个命令行参数（flag）旁边都写明人话用途；通用缩写见 [定义表](../DEFINITIONS.md)。

## 前置条件

- 使用已经合入 `main` 的 clean detached source；不得在 archive checkout 或活 trainer source 上修改文件。
- teacher checkpoint 必须有相邻 `params/training_contract.json`、原始 no-clobber queue claim，且 checkpoint
  内嵌 schema=`3`、fresh lineage=`1` 和同一 claim digest。
- ordered motion paths 必须与 checkpoint hard contract 完全一致。
- capture output directory 必须是新建空目录；`natural_wrap_capture.claim.json`、`natural_wrap_states.npz`、
  `natural_wrap_capture.json` 或 `teacher_receipt.json` 任一已存在都禁止复用。
- 先在实验记录中冻结 teacher checkpoint、target count、root linear/angular velocity limit、4096 environments、
  最大 inference steps 和 GPU 槽。不要用 timeout 状态、失败 reset 或 clip-switch 中止状态补数量。

## 1. inference-only 自然 wrap 采集

下例中的 `+task.motion.post_swing_capture_output_dir` 表示“raw 结果写到哪个全新目录”；
`+task.motion.post_swing_capture_target_count` 表示“必须收满多少条自然 wrap 状态”；
`post_swing_capture_max_steps` 是只允许推理多少步的硬预算，不是成功条件。

```bash
cd /ABS/CLEAN/NOHOPE/hope_training/whole_body_tracking
mkdir /ABS/NEW/CAPTURE_DIR

/ABS/HOPE_ISAAC_PY scripts/play.py \
  task=HOPEPingPongVirtualBall algo=ppo headless=true device=cuda:GPU \
  num_envs=4096 checkpoint=/ABS/RUN/model_ITER.pt \
  motion_file=/ABS/MOTION_0.npz motion_file_2=/ABS/MOTION_1.npz \
  task.motion.wrap_teleport=false task.motion.post_swing_start_prob=0.25 \
  +task.motion.post_swing_capture_output_dir=/ABS/NEW/CAPTURE_DIR \
  +task.motion.post_swing_capture_target_count=4096 \
  post_swing_capture_max_steps=20000
```

capture mode 不做 PPO update，也跳过 ONNX export。环境建立后先用当前 runtime 重建 schema-3 hard contract；
只有它与 checkpoint 相邻合同逐字段相同，`MotionCommand` 才向自己用 `O_EXCL` 占有的 claim fd 写入冻结绑定，
并允许 natural-wrap 分支接收第一条 live state。不存在可被外部 caller 喂任意 arrays 的 writer API。目标数未在最大步数前收满会非零退出，
不得把 partial memory 或临时文件晋级。

## 2. one-shot attestation

`--root-linear-limit-mps` 是 floating-base 线速度范数上限（m/s）；
`--root-angular-limit-radps` 是角速度范数上限（rad/s）；两个值必须来自预注册，而不是看完数据后调大。
每个 `--motion` 按 hard contract 顺序重复一次。

```bash
cd /ABS/CLEAN/NOHOPE
python3 scripts/attest_post_swing_teacher.py \
  --capture-result /ABS/NEW/CAPTURE_DIR/natural_wrap_capture.json \
  --checkpoint /ABS/RUN/model_ITER.pt \
  --hard-contract /ABS/RUN/params/training_contract.json \
  --launch-claim /ABS/RUN/queue_claim.json \
  --capture-source-checkout /ABS/CLEAN/NOHOPE \
  --motion /ABS/MOTION_0.npz --motion /ABS/MOTION_1.npz \
  --root-linear-limit-mps ROOT_LINEAR_LIMIT \
  --root-angular-limit-radps ROOT_ANGULAR_LIMIT \
  --output-receipt /ABS/NEW/CAPTURE_DIR/teacher_receipt.json
```

attestor 使用 `torch.load(..., weights_only=True)`；不兼容 restricted unpickler 的 checkpoint 直接 fail closed，
不得切回任意 pickle 执行。成功只输出 receipt 路径、SHA 和 count；重复运行因 no-clobber 失败。SSH timeout 时只读检查四个固定文件，
不要自动重发 capture 或 attestor。

## 3. 4096-environment 首 reset probe（尚未执行）

后续 probe 的两臂必须引用同一 receipt SHA，并显式设置：

- `post_swing_teacher_root_linear_velocity_limit_mps` / `...angular...`：与 attestor 完全相同的 root 上限；
- `post_swing_first_reset_min_adopted_count`：预注册 count 下限；
- `post_swing_first_reset_min_adopted_fraction`：预注册比例下限；
- `post_swing_first_reset_selection_tolerance`：相对配置概率的绝对偏差上限；
- `post_swing_first_reset_require_readback=true`：两次 simulator write 后复核实际 root/joint state。

probe 需要在首个 PPO rollout/update 前通过，并证明两臂 hard contract 只在科学自变量上不同；否则保持
`Partial`，不得发正式 pair、第二 seed、judge 或 promotion。

## 源码验证

```bash
pytest -q \
  tests/test_attest_post_swing_teacher.py \
  hope_training/whole_body_tracking/tests/test_post_swing_teacher.py

python3 -m py_compile \
  scripts/attest_post_swing_teacher.py \
  hope_training/whole_body_tracking/scripts/play.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/post_swing_teacher.py \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py

git diff --check
```
