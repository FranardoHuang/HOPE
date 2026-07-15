# 生成随挥结束教师状态制品

状态：**v1/v2 均已阻断且 namespace 不重发；v3 已自然收满 4096 条 finite 状态，唯一授权的 attestor
attempt-2 已 rc0 并发布 exact receipt；first-reset trainer-consumer probe 尚未运行。** 本操作只用于仿真推理和后续训练 cold-start，
绝不下发真机命令。源码专项与攻击负测通过；首个 exact 实例见
[`phase1_post_swing_teacher_capture_prereg_20260715.json`](../../configs/phase1_post_swing_teacher_capture_prereg_20260715.json)，
但它在 Hydra compose 阶段因保留 train-only checkpoint 兼容键而 fail closed；capture directory、claim、
process 和 GPU work 均未创建，v1 永久不重发。schema-v2 的只读 compose 与所有合同复核随后通过，但正式
launch 在零 inference step 读取初始 observation 时命中 IsaacLab 版本差异：wrapper 返回合法
`(actor_observation, extras)`，旧 `play.py` 却直接调用 tuple 的 `.to()`。v2 只留下 claim，没有 states/result/
receipt；exact PGID teardown 后永久花掉。机器证据见
[`phase1_post_swing_teacher_capture_attempt_v2_result_20260715.json`](../../configs/phase1_post_swing_teacher_capture_attempt_v2_result_20260715.json)。
successor v3 在同一 Pod2 GPU2 上自然退出，`natural_clip_wrap` 状态 `4096/4096` 且 arrays finite；但旧
attestor 把“完整 JSON 文档的末尾换行”错误计入 queue claim 的嵌入式 `content_sha256`，以 rc2
`training launch claim canonical digest mismatch` 假拒绝。`teacher_receipt.json` 从未创建；完整机器证据见
[`phase1_post_swing_teacher_capture_attempt_v3_result_20260715.json`](../../configs/phase1_post_swing_teacher_capture_attempt_v3_result_20260715.json)。
attempt-1 在 `_claim()` 即停止：它当时只证明 digest 是第一个观测到的 blocker；checkpoint restricted load、
fresh lineage、相邻 hard contract、producer/attestor source、motion/order 与 velocity limits 均未执行、未证明。
capture arrays finite 是单独的 capture 证据，不能代替完整 attestation；后续 attempt-2 已补齐这些门。
不得重跑 capture 或 attestor。首 reset readback probe 尚未完成，因此 scientific trainer、第二 seed、judge
与 promotion 仍未授权。

2026-07-15 的 seed-parity 修复已让 `play.py` 只接受 plain uint32 seed，并在 `gym.make` 前把同一个值
写入 environment config 与 PPO runner config；三个 train-only checkpoint 键也都有真实 Hydra compose
负测。一次性 controller/builder 随后通过独立 source-only 红队，v3 又实际闭合 plan/compose/launch/capture；
attestor 的 content/document canonical 语义和同一 immutable v3 capture 的唯一 attestation 现已闭合；
当前缺口只剩独立的首 reset probe。不是 seed、controller、capture 或 receipt lineage 缺口。

successor 的 `play.py` 必须通过共享 `policy_observation_tensor` 只接受三种明确结构：actor tensor、exact
`(actor tensor, extras mapping)`，或含 `policy` 的 observation mapping；mapping 同时含 `critic` 时也只把
`policy` 交给 actor，critic-only/坏 tuple/非 tensor 一律 fail closed。初始 observation 与每个 step 输出都走
同一 adapter。环境一旦由 `gym.make` 创建，正常退出、初始 observation 异常和 step 异常都必须 exactly once
关闭最终 wrapper；wrapper 构造前失败才关闭 base env，并保留原异常不被 teardown 异常覆盖。该修复只有
host focused/Hydra compose 证据，仍需新 commit、新 plan 和新 namespace 的 Pod runtime 才能晋级。

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

## successor 必须使用的一次性 controller

v1 的手工 argv 派生已被证伪；successor 不再手拼命令。源码工具
[`run_preregistered_post_swing_capture.py`](../../scripts/run_preregistered_post_swing_capture.py) 只在仿真 Pod
本机运行，读取 schema-2 机器预注册，并提供三个模式：

```bash
python3 scripts/run_preregistered_post_swing_capture.py \
  --plan /ABS/FROZEN_PLAN.json \
  --expected-plan-sha256 PLAN_SHA256 plan

python3 scripts/run_preregistered_post_swing_capture.py \
  --plan /ABS/FROZEN_PLAN.json \
  --expected-plan-sha256 PLAN_SHA256 launch

python3 scripts/run_preregistered_post_swing_capture.py \
  --plan /ABS/FROZEN_PLAN.json \
  --expected-plan-sha256 PLAN_SHA256 status
```

`plan` 只读复算 source、ignored A3 tree、checkpoint/claim/binding/milestone receipt、动作、题库、Pod2
hostname/machine-id/boot-id、physical GPU2 UUID、冻结 Python symlink chain、五个实际 source/Isaac import root
和最终 argv；它随后用 launch 完全相同的 absolute cwd、安全环境、argv 与 timeout 跑只读
`--cfg job --resolve`，记录 compose output SHA/bytes/elapsed，再次复算所有输入。失败或前后漂移均不会创建
launch/capture namespace、claim 或 capture process。`launch` 持有共享 `/tmp/hope_lean_queue_gpu2.lock`，仍会用
exact `/usr/bin/git`、`/usr/bin/nvidia-smi` bytes/SHA 和同一 helper 重复 compose、再次复算所有输入。
只有仍 exact 才创建 capture directory；controller 随后以同一 PID `execve` 交棒，不产生可能成为 orphan 的 child。
`status` 只按 immutable exec intent 的 PID/PGID/SID/starttime/cmdline 和四个固定制品读取，symlink、zombie、
PID reuse 或 teacher receipt 重绑都不能报绿。
工具没有 stop/retry/SSH/trainer 子命令。compose 或二次复算失败会留下 no-clobber failure evidence，但不会
创建 capture claim/process；该 plan 仍视为花掉，必须新预注册。play 的 plain-uint32 seed parity 已合入并
通过真实 Hydra compose 正负例。当前工具仍只是 source gate；必须用下面 builder 在 Pod2 本机生成并 review
一个全新 schema-v2 plan，再过 `plan` 模式，才可单次运行。

### schema-v2 计划生成

[`build_post_swing_capture_plan_v2.py`](../../scripts/build_post_swing_capture_plan_v2.py) 只做本机 byte snapshot，
不创建 launch/capture namespace，也不启动 Hydra/Isaac。输出必须在 immutable capture checkout 之外；五项
`--runtime-tree` 已收缩为 controller 实际使用的五个 source/Isaac Python import root，不声称覆盖 venv
site-packages、stdlib、native ELF 或整个 rootfs。

```bash
python3 scripts/build_post_swing_capture_plan_v2.py \
  --template-plan /ABS/V1_EVIDENCE_TEMPLATE.json \
  --capture-source-checkout /ABS/CLEAN/DETACHED/SOURCE \
  --plan-id NEW_DIRECT_NAMESPACE \
  --gpu-uuid GPU-feee9e1f-7663-06f6-fa29-62fca6a9b1a4 \
  --git /usr/bin/git --nvidia-smi /usr/bin/nvidia-smi \
  --runtime-tree capture_pythonpath=/ABS/CAPTURE_SOURCE_ROOT:on \
  --runtime-tree isaaclab=/ABS/ISAACLAB_ROOT:on \
  --runtime-tree isaaclab_tasks=/ABS/ISAACLAB_TASKS_ROOT:on \
  --runtime-tree isaaclab_assets=/ABS/ISAACLAB_ASSETS_ROOT:on \
  --runtime-tree isaaclab_rl=/ABS/ISAACLAB_RL_ROOT:on \
  --env PATH=/usr/bin:/bin --env HOME=/root \
  --output /ABS/OUTSIDE_SOURCE/PLAN.json
```

路径占位符必须替换成 Pod2 实际 exact root；用 `--help` 查看完整参数。Pod 当前没有
`RUNPOD_POD_ID`，所以 schema 不伪造该字段，而是固定 hostname + `/etc/machine-id` + boot-id；Pod 重启后
boot-id 改变会让旧 plan fail closed。

威胁模型是 trusted root operator 下的误配置、字节漂移和并发合法 job；不声称抵御同机恶意 root 在微秒级
替换 path/inode，也不把五个 import root 冒充完整 Python/native dependency closure。该剩余限制必须保留在
review 记录中。

## 1. inference-only 自然 wrap 采集

下例仅解释 controller 最终生成的 argv 语义，不再允许人工直接执行。`+task.motion.post_swing_capture_output_dir`
表示“raw 结果写到哪个全新目录”；
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
cd /ABS/CLEAN/ATTESTOR_A38B7E9
python3 /ABS/CLEAN/ATTESTOR_A38B7E9/scripts/attest_post_swing_teacher.py \
  --capture-result /ABS/NEW/CAPTURE_DIR/natural_wrap_capture.json \
  --checkpoint /ABS/RUN/model_ITER.pt \
  --hard-contract /ABS/RUN/params/training_contract.json \
  --launch-claim /ABS/RUN/queue_claim.json \
  --capture-source-checkout /ABS/CLEAN/ORIGINAL_CAPTURE_SOURCE_906A3C3 \
  --motion /ABS/MOTION_0.npz --motion /ABS/MOTION_1.npz \
  --root-linear-limit-mps ROOT_LINEAR_LIMIT \
  --root-angular-limit-radps ROOT_ANGULAR_LIMIT \
  --retry-authorization /ABS/CLEAN/MAIN_WITH_AUTH/configs/phase1_post_swing_teacher_capture_v3_attestor_retry_authorization_20260715.json \
  --expected-retry-authorization-sha256 87fd1c7136d9a6d54546f26367cc83d5dad20e4d8b0f57fba94e6cc207e2dfda \
  --output-receipt /ABS/NEW/CAPTURE_DIR/teacher_receipt.json
```

attestor 使用 `torch.load(..., weights_only=True)`；不兼容 restricted unpickler 的 checkpoint 直接 fail closed，
不得切回任意 pickle 执行。成功只输出 receipt 路径、SHA 和 count；重复运行因 no-clobber 失败。SSH timeout 时只读检查四个固定文件，
不要自动重发 capture 或 attestor。

queue claim 的 `content_sha256` 只覆盖 `json.dumps(..., separators=(",", ":"), sort_keys=True)` 产生的
compact UTF-8 **content bytes**，不含末尾换行；claim/receipt 落盘时才在完整 JSON **document bytes** 后追加
恰好一个 `\n`。两类 bytes 必须用不同 helper，不能再共享一个“总是加换行”的 canonicalizer。v3
attempt-1 已证明混用会在任何 receipt 写入前假拒绝；该失败实例永久不重放。

attempt-2 的 running attestor 与 `--capture-source-checkout` 故意不是同一个 checkout。running attestor
必须是 clean detached `a38b7e9e693db407795d9a5f3af144b8f8e293cf`，其脚本 SHA-256 必须为
`03611b565a539fa81811ac76c4631484a60679adfd11c1f1e07599081f46310f`；authorization 文件来自已经包含它的
后续 clean main，不得改成 main HEAD 直接运行 attestor。后者（`--capture-source-checkout`）必须是原始
v3 producer 的 clean `906a3c3...` source，只写入 receipt 的
`capture_source={commit,clean,producer_source_sha256}`；running script 则从修复后 clean main 自己的 repo
root 派生 `attestor_source={commit,clean,attestor_source_sha256}`。receipt attestation sub-schema=`2`，trainer
hard-contract summary 保留两者和 retry authorization。`--retry-authorization` 必须是 main 中固定、tracked
的 attempt-2 授权，逐字节绑定 v3 plan/capture/teacher/output 与唯一 attestor commit/SHA；命令行还必须给
它的 exact file SHA `87fd1c71...dfda`。controller `status` 会把 capture source 对回 immutable v3 plan，并只从自身 clean
checkout 读取这份 tracked authorization，再核对 receipt 的 attestor source 与 authorization；不得用任意
post-fix clean HEAD 自签自验、不得用 post-fix HEAD 冒充 capture source，也不得交换两个对象。

2026-07-15 唯一 attempt-2 已按上述合同执行：running source=`a38b7e9...293cf`，authorization source=
`main@ff9a253...0d54`，PGID `403786` 自然 rc0 后 absent。receipt 为 4103 bytes / SHA-256
`e20a6989a43f3a0725b5973e8675f2a25e72d2fe705d3fc0c914cd7da2d2aba4`、count=`4096`；merged-main
controller `status` 返回 `teacher_receipt_binding_exact=true`。state/log/rc SHA-256 分别为
`7a7e74ab...d0a24` / `1becc72e...e77c3` / `9a271f2a...86aa`。该 receipt 和 attestor 都不得重发或覆盖。

## 3. 4096-environment 首 reset probe（尚未执行）

后续 probe 的两臂必须引用同一 receipt SHA，并显式设置：

- `post_swing_teacher_retry_authorization`：指向 main tracked 的
  [随挥教师重签授权](../DEFINITIONS.md)，当前只能是
  `configs/phase1_post_swing_teacher_capture_v3_attestor_retry_authorization_20260715.json`；
- `post_swing_teacher_retry_authorization_sha256`：上述文件的 exact byte SHA-256，当前只能是
  `87fd1c7136d9a6d54546f26367cc83d5dad20e4d8b0f57fba94e6cc207e2dfda`；
- `post_swing_teacher_root_linear_velocity_limit_mps` / `...angular...`：与 attestor 完全相同的 root 上限；
- `post_swing_first_reset_min_adopted_count`：预注册 count 下限；
- `post_swing_first_reset_min_adopted_fraction`：预注册比例下限；
- `post_swing_first_reset_selection_tolerance`：相对配置概率的绝对偏差上限；
- `post_swing_first_reset_require_readback=true`：两次 simulator write 后复核实际 root/joint state。

probe 需要在首个 PPO rollout/update 前通过，并证明两臂 hard contract 只在科学自变量上不同；否则保持
`Partial`，不得发正式 pair、第二 seed、judge 或 promotion。

receipt 路径/SHA 与 authorization 路径/SHA 四项必须同时存在；缺任一项都在建环境时 fail closed。trainer
loader 会从 authorization 内容派生原始 producer `906a3c3...` / `496a7fa2...` 与 attestor
`a38b7e9e...` / `03611b56...` 两个 tuple，并逐字段对 receipt；不能用另一组格式合法的 hex 值重绑。
解析后的完整 authorization 进入 schema-3 training hard contract，保证后续 checkpoint lineage 也带着这条
外部信任根。

## 源码验证

```bash
pytest -q \
  tests/test_run_preregistered_post_swing_capture.py \
  tests/test_attest_post_swing_teacher.py \
  tests/test_post_swing_play_runtime_compose.py \
  hope_training/whole_body_tracking/tests/test_post_swing_teacher.py \
  hope_training/whole_body_tracking/tests/test_isaac_bank_exam_adapter.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py

python3 -m py_compile \
  scripts/build_post_swing_capture_plan_v2.py \
  scripts/run_preregistered_post_swing_capture.py \
  scripts/attest_post_swing_teacher.py \
  hope_training/whole_body_tracking/scripts/play.py \
  hope_training/whole_body_tracking/scripts/train.py \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/post_swing_teacher.py \
  hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py

git diff --check
```

2026-07-15 在可导入 Hydra 的本地环境，controller/attestor/teacher 原四文件门曾复现为 `41 passed`；
加入 observation adapter、v2 result 负测与 content/document canonical 回归后的旧五文件命令曾复现为
`96 passed`。补齐 authorization-backed consumer 与 YAML override 后，上述六文件命令复现为
`181 passed`（另有一个既有 duplicate-ZIP warning），其中 attestor 专项为 `11 passed`。这仍只是 host
source gate，不能替代 v3 receipt attestation 或首 reset probe。
