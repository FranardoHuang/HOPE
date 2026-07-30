# Fresh ActionBall N5：V3 双 GPU no-clobber 课程启动

本页是 fresh（从零开始）、upper（上肢）、`no_move`（base 不移动）的 exact N=5
[按动作条件化 Ball-first](../DEFINITIONS.md#action-conditioned-ball-first) 正式启动工序。专用入口是：

```text
hope_training/whole_body_tracking/scripts/launch_action_ball_curriculum.py
```

这里的 [`no-clobber`](../DEFINITIONS.md#no-clobber) 表示一次尝试的 namespace、收据和训练输出
只能首次创建，已有路径永不覆盖；`run_name` 是 trainer 日志目录后缀和这次尝试的人类可读名字。
`plan` 是只做静态复核、打印 exact argv 和 claim 的子命令；`launch` 是持双 GPU 锁后重新复核并
实际启动的子命令。两者都没有任意 Hydra 参数入口。

## 当前结论（2026-07-29）

**尚未有 V3 ActionBall 长训或通过收据。** 当前代码侧以下正式 trust set 仍为空：

- motion promotion certificate；
- frozen evaluator V4 launch receipt；
- GPU1 evaluation sidecar 的 code receipt 与 launch receipt；
- drain/reset runtime receipt。

空 trust set 必须让 `plan` fail-closed，不能用 diagnostic flag、spec 自报或运行时注入填充。正式
fresh N5 的五件 motion bytes、动作专属 fitted-ball profile、MuJoCo fitted-ball gate、Isaac
filtered-contact table smoke、motion admission、prelaunch safety attestation、runtime inventory
和 evaluator receipts 也必须来自本次 exact clean commit 的真实工件；旧 N4/N3、文件名写 N5
但内容不是 exact N5 的配置、placeholder digest 和 ignored/local 未钉资产都不能顶替。

因此本页记录的是 **schema V3 的发射合同与命令**，不是“已跑”。只有本页列出的输入和阶段门
闭合后，才允许按 `smoke → canary → long` 顺序推进。

formal N5 trainer argv 还固定拥有三项 fresh 语义，caller 不能从 `extra_overrides` 改写：

```text
algo.policy.init_noise_std=0.02
action_ball_shared_ready_bootstrap=true
+task.racket.reference_guard_mode=metrics_only
```

第三项必须带 Hydra 的 `+`，因为 task YAML 不预声明该键；写成普通 override 会在 Kit 前的
Hydra compose 直接失败。shared-ready bootstrap 仅适用于这里的 exact N=5 共用 ready 动作集，
不是 N=73/N=93 的默认。

## 唯一动作身份

正式顺序逐字固定为：

```text
bh_loop_c
v12_forehand_block
bh_block
s0_highpress
fh_loop_high
```

launcher 会交叉核对 launch spec、manifest、upper prototype、canonical registry、motion
admission、frozen evaluator、fitted-ball profile pins、committed `FRESH_N5_ORDER` 和最终 trainer
argv。数量、顺序、stable action UID、motion bytes 或 family 任一漂移都拒绝。旧 `fh_loop` 和
`fh_block_syn` 不属于这次 fresh N5。

动作、UID、球/物理/solver pins 的完整含义见
[`action-ball manifest`](../DEFINITIONS.md#action-ball-manifest)；actor 形状必须是
[`action_ball_table_pose_twist_heading_task_n5`](../DEFINITIONS.md#action-ball-table-pose-twist-heading-task-contract)
（frame 一致的相对击球 task + 绝对桌体 pose/twist，总宽 198），不能把旧
`action_ball_n5`、混合 frame 的同宽 checkpoint、其他 N 的 checkpoint 或动作列表续成 N5。

## V3 spec：字段必须恰好一致

spec 必须是绝对路径指向的 UTF-8 JSON；重复键、NaN/Infinity、未知字段都失败。顶层键必须恰好为：

```text
schema_version, kind, launch_profile, source, action_set, inputs,
policy_contract_sha256, train, gpus, stages
```

固定值：

```text
schema_version = 3
kind = action_ball_no_clobber_launch_spec_v3
launch_profile = fresh_upper_nomove_n5_v3
action_set.expected_n = 5
action_set.scope = upper
action_set.mobility_mode = no_move
```

`source.checkout` 是 exact Git worktree 根的绝对路径；`source.commit_sha` 是完整 40 位 commit。
HEAD 必须等于该 commit，staged、unstaged、untracked 任一改动都会在拿锁和建 namespace 前拒绝。
运行时源码、科学输入和 motion bytes 必须能从该 commit 的普通 blob 重读；“worktree clean”不能
替代 commit-tree 证明。

`inputs` 必须恰好包含：

```text
manifest
prototype
motion_admission_receipt
evaluator_launch_receipt
sidecar_launch_receipt
drain_reset_launch_receipt
canonical_registry
promotion_certificate
fitted_ball_profile_pins
fitted_ball_launch_trust_spec
fitted_ball_launch_trust_root
fitted_ball_gate_receipt
isaac_table_smoke_receipt
stage_evaluator_authority
prelaunch_safety_attestation
```

repo 内输入使用 repo-relative path 与 exact 文件 SHA-256；fitted-ball gate、Isaac table smoke、
runtime inventory 和签名 safety attestation 使用 checkout 外绝对路径与 exact 文件 SHA-256。
`canonical_registry` 还必须同时钉 alignment、canonical-ready 和 ready-FK 三个 SHA-256。

`evaluator_launch_receipt` 必须是 V4 正式 receipt，`sidecar_launch_receipt` 必须绑定 committed
sidecar bytes、backend/runtime/policy-evaluation 合同和下文的窗口/heartbeat 合同；
`drain_reset_launch_receipt` 必须与同一动作顺序、solver、policy 和 evaluator identity 相同。
promotion、evaluator、sidecar 和 drain/reset 的 digest 只能来自 committed source 中各自唯一的
literal trust set；空集合、多余 digest、环境变量或 spec 注入全部失败。

`train` 必须恰好包含：

```text
isaac_python
runtime_inventory
seed
extra_overrides
ground_plant_contract_sha256
effective_reward_recipe_sha256
ppo_recipe_sha256
```

`isaac_python` 钉绝对路径、最终 executable bytes SHA、Python version、cache tag 和有序
import roots。`runtime_inventory` 钉下面生成的外部 receipt。`extra_overrides` 当前只允许逐字
`logger=tensorboard`，也就是只把指标写本地 TensorBoard；scientific keys、bank/exam、
checkpoint/resume 和 launcher-owned keys 均不能从这里注入。seed、plant、实际生效 Reward 和
PPO 共同组成 `training_recipe_sha256`；三个阶段必须逐字相同。

## Pod runtime overlay 与 inventory

不得向共享 Isaac venv 做 `pip install/uninstall`。为 exact clean checkout 创建一份新的、从未
存在过的 overlay venv；overlay 通过 `--system-site-packages` 只读复用共享 Isaac/torch，
而 `whole_body_tracking` editable distribution 指向这份 exact Git checkout。下面的
`ACTIONBALL_*` 变量只是路径占位，使用前替换为本次真实绝对路径：

```bash
ACTIONBALL_CHECKOUT=/absolute/clean/detached/nohope
ACTIONBALL_BASE_PY=/workspace/hope_isaac_venv/bin/python
ACTIONBALL_OVERLAY=/absolute/no-clobber/runtime-overlay
ACTIONBALL_ISAACLAB=/absolute/clean/IsaacLab
ACTIONBALL_CONTROL=/absolute/no-clobber/operator-control

test ! -e "$ACTIONBALL_OVERLAY"
"$ACTIONBALL_BASE_PY" -m venv --system-site-packages "$ACTIONBALL_OVERLAY"
"$ACTIONBALL_OVERLAY/bin/python" -m pip install --no-deps pin-pink==3.1.0
"$ACTIONBALL_OVERLAY/bin/python" -m pip install --no-deps --editable \
  "$ACTIONBALL_CHECKOUT/hope_training/whole_body_tracking/source/whole_body_tracking"

"$ACTIONBALL_OVERLAY/bin/python" -I -B \
  "$ACTIONBALL_CHECKOUT/hope_training/whole_body_tracking/scripts/action_ball_runtime_inventory.py" \
  mint \
  --python "$ACTIONBALL_OVERLAY/bin/python" \
  --isaaclab-checkout "$ACTIONBALL_ISAACLAB" \
  --output "$ACTIONBALL_CONTROL/runtime-inventory.json"
```

`--python` 是要被冻结的 overlay interpreter；`--isaaclab-checkout` 是实际 IsaacLab Git
checkout；`--output` 是首次创建的外部 receipt。定义与 no-clobber 规则见
[`DEFINITIONS.md`](../DEFINITIONS.md#no-clobber)。

inventory 会递归验证 Python/IsaacLab import closure、PEP 508 `Requires-Dist`、关键
`carb/omni` RECORD、overlay 的 `pyvenv.cfg`、`.pth`/editable direct URL、exact Git bytes 和
解释器身份。当前共享环境缺少的 distribution 只能安装进新 overlay；不得修补共享 base。
launcher 的 `plan` 和持锁重验都会以 committed verifier 重新跑完整 inventory。

trainer 实际建好 env/agent 后会先持久化 `env.pkl`、`agent.pkl`、training contract 和 frozen
evaluation runtime identity，再以 `O_EXCL`（目标已存在即失败）发布
`params/action_ball_runtime_bootstrap_receipt.json`。runner 在第一次 checkpoint 前绑定该 receipt
的 content SHA、文件 SHA 和 location-free lineage；没有 bootstrap receipt 的 checkpoint 不可
作为正式阶段结果。

## 双 GPU、双 UUID、双锁

`gpus` 必须恰好包含 `trainer` 和 `evaluator`：

| role | physical GPU | 固定共享锁 | 作用 |
| --- | ---: | --- | --- |
| trainer | `0` | `/tmp/hope_lean_queue_gpu0.lock` | PPO rollout/update |
| evaluator | `1` | `/tmp/hope_lean_queue_gpu1.lock` | frozen policy canary/heldout |

两项都必须给 explicit GPU UUID、真实人名 owner、`require_empty=true` 和共同的 Pod boot lock
`/workspace/.kit_boot.lock`；UUID 与 GPU lifetime lock 必须互异。每个 stage 另有
trainer/evaluator 两份 owner receipt，逐字绑定 role、physical index、UUID、owner、lock path、
stage、namespace 和 source commit。

launcher 只打开已存在的 regular lock file，再 non-blocking `flock`；不创建、删除、truncate、
chmod 或清理未知锁。它按 lock path 排序拿齐两把锁，锁内重跑完整 plan，再用受信任路径解析的
`nvidia-smi` 核对 index→UUID 和两卡 compute-process empty。任一卡 occupied、输出不可解析或
owner/UUID 漂移都拒绝；不会迁移到别的卡。两把 lock fd 会传给 exact supervisor/children，
避免 trainer 活着而另一边提前释放。

## Frozen evaluator 窗口不可缩

正式 V4 evaluator 的每个决策窗口恰好是：

| 用途 | proposal 数 | 最少 safe-closed 数 | 能否授权 frontier |
| --- | ---: | ---: | --- |
| rolling scheduler | `100` | 不替代正式门 | 只选下一候选 arm |
| frozen canary | `320` | `256` | 否，只作晋级前门 |
| disjoint heldout | `960` | `768` | 是，且仍须其余安全/统计门 |

三类窗口都必须 `optional_stopping=false`；不能看见好结果后提前停。每个 proposal tape 固定
`20% center / 60% interior / 20% frontier`，并由 authority 分配互斥、连续的
seed/sample/birth ranges。rolling-100 只决定下一条
[`curriculum arm`](../DEFINITIONS.md#action-ball-arm-catalog)，不能直接扩大域；正式 safe-policy
failure 的分母见[定义](../DEFINITIONS.md#safe-policy-failure)。

sidecar 每 `5 s` 发布一次 heartbeat；`120 s` 无新 heartbeat 即 stale；一个 frozen evaluation
request 的 deadline 固定为 `7200 s`。heartbeat 必须保持 owner/run/PID、sidecar/launch/backend
SHA、request identity、进度和 monotonic time 一致。缺 heartbeat、进度倒退、deadline 漂移、
sidecar 提前退出或 trainer/evaluator policy generation 不一致都会关闭本次 stage。

## 三阶段预算

每阶段使用不同、从未存在过的绝对 namespace、双 owner receipts 和 evaluation inbox：

| stage | 预算硬门 | 前序门 |
| --- | --- | --- |
| `smoke` | 恰好 `1 env × 2 updates`，`save_interval=1`，frozen-eval interval=`2` | `null` |
| `canary` | 至少 `2 env × 3 updates`；`updates // frozen_eval_interval >= 5` | exact passed smoke receipt |
| `long` | 至少 `4096 env × 20001 updates`，`save_interval<=100`，且规模不小于 canary | exact passed canary receipt |

`frozen_eval_interval_updates` 是每隔多少 PPO update 发一次冻结评估请求，不是窗口样本数；窗口样本
数只能是上一节的 `100/320/960` 合同。初次 smoke spec 可把以后阶段
`predecessor_receipt=null`；准备 canary/long 时再把签名的前一阶段通过收据填入。不能预写
“预计通过”或拿 `supervisor_ready` 代替 passed stage receipt。

三个阶段都 fresh-from-random；checkpoint/resume 不能通过 spec 或 `extra_overrides` 注入。这里的
“exact resume”只是在阶段结束后验证 checkpoint 能无损恢复，不表示下一阶段从该 checkpoint
续训。

## 先跑 plan

只在拟运行的 clean checkout 上执行；`--spec` 是 operator spec 的绝对路径，
`--stage` 是本次要验证的 `smoke/canary/long`：

```bash
python3 hope_training/whole_body_tracking/scripts/launch_action_ball_curriculum.py \
  plan \
  --spec /absolute/operator-control/action-ball-fresh-n5-v3.json \
  --stage smoke
```

`plan` 不拿 GPU 锁、不查询 GPU、不创建 namespace、不生成子进程。成功输出完整 canonical payload、
trainer/evaluator argv 和本次唯一 `launch_claim_sha256`。人工逐项复核 source commit、N5 order、
全部输入 pins、runtime inventory、Reward/PPO recipe、双 UUID/owner/locks、stage budget、
evaluation window/heartbeat 和 argv。输入未闭合时正确结果是
`ACTION_BALL_LAUNCH_REFUSED`，不要为了绿色 plan 改用旧资产或 diagnostic bypass。

host 回归：

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_launch_action_ball_curriculum.py \
  hope_training/whole_body_tracking/tests/test_action_ball_stage_supervisor.py \
  hope_training/whole_body_tracking/tests/test_action_ball_runtime_inventory.py \
  hope_training/whole_body_tracking/tests/test_action_ball_exact_resume_verifier.py \
  hope_training/whole_body_tracking/tests/test_action_ball_stage_evidence.py
```

## 正式 launch 与接受事务

只有 plan 完全一致时，才把它输出的完整 64 位 claim 原样带回：

```bash
python3 hope_training/whole_body_tracking/scripts/launch_action_ball_curriculum.py \
  launch \
  --spec /absolute/operator-control/action-ball-fresh-n5-v3.json \
  --stage smoke \
  --confirm-claim-sha256 <plan输出的launch_claim_sha256>
```

`--confirm-claim-sha256` 是人工确认的 exact plan 摘要；spec、stage、namespace 或任一 byte 变化都会
换摘要。不要提前创建 namespace。

事务顺序固定为：

```text
静态 plan
→ 获取 GPU0/GPU1 两把 lifetime lock
→ 锁内重验 clean source、pins、owners、UUID 和两卡 empty
→ atomic mkdir(namespace) + exclusive launch_claim/live_gpu_admission
→ boot GPU1 frozen evaluator，等 exact ready + heartbeat
→ boot GPU0 trainer，等第一条 Learning iteration
→ supervisor_ready
→ launcher 写 accept-intent 并发控制 token
→ supervisor 写 accept-ACK
→ launcher 写 launch_accepted
→ supervisor 写 launch commit-ACK
→ launch 命令才返回 accepted；supervisor 继续持锁值守
→ trainer 自然退出且无残余
→ 精确停止 GPU1 evaluator
→ GPU0 独立 exact-resume verifier 自然退出并验 receipt
→ supervisor_terminal
```

只看到 Kit boot、sidecar ready、trainer PID、`Learning iteration` 或 `supervisor_ready` 都不算 launch
已接受。commit-ACK 前任一异常，launcher 必须只针对本 claim 的 exact supervisor/process groups
发取消并验证 reap；不能按命令行模式 kill，也不能碰外部进程。若 exact closure 无法证明，写
`launch_unresolved.json`，namespace 仍永久 spent，且不得启动同名重试。

接受后 evaluator 必须在 trainer 整个生命周期保持活跃并持续 heartbeat。trainer 只有自然
exit=`0` 且 process group 无残留才算完成运行；随后 supervisor 精确停止 evaluator、验证其退出且
未使用 forced kill，在仍持有 GPU0/GPU1 lifetime locks 时选择 terminal checkpoint，并在 GPU0
启动下节的独立 exact-resume verifier。verifier 自然通过后才写 `supervisor_terminal.json`。
任一 child 非零、提前退出、heartbeat stale、NaN/identity 漂移、exact-resume 失败或残余进程都写
closed failure，而不是 stage PASS。

## 训练结束后的 exact-resume 与阶段签名

V3 supervisor 会在 trainer 退出且 GPU1 evaluator 关闭后自动选择唯一 terminal
`model_<N>.pt`，再持有 Pod boot lock 启动 committed verifier。boot flock 覆盖 verifier 的整个
Kit lifetime，而不只覆盖 fork；lock fd 也继承给 child，避免 supervisor 意外退出后新 Kit
误入同一启动窗口。等价的 exact argv 结构如下，
**仅用于审计日志；正式 stage 不得手工再跑**，因为输出路径已经由 supervisor no-clobber 占用：

```bash
"$ACTIONBALL_OVERLAY/bin/python" -I -B \
  "$ACTIONBALL_CHECKOUT/hope_training/whole_body_tracking/scripts/action_ball_exact_resume_verifier.py" \
  --claim /absolute/stage-namespace/launch_claim.json \
  --checkpoint /absolute/rsl-log-dir/model_<N>.pt \
  --out /absolute/stage-namespace/exact_resume_verification.json
```

`--claim` 指向本 stage 的 exact launch claim；`--checkpoint` 指向该 stage 的最终 checkpoint；
`--out` 必须是 stage namespace 内首次创建的固定 receipt 路径。它们同样服从
[`no-clobber`](../DEFINITIONS.md#no-clobber)，不能拿另一 stage 的 claim、预建输出或覆盖失败
证据。verifier 使用 GPU0、继续继承两把 lifetime locks，并有固定 `3600 s` 完成上限；非零退出、
超时、残余 process group 或 receipt 漂移都会让 supervisor 写 closed failure，不能落
`supervisor_terminal.json`。

verifier 构造真实 runtime，严格恢复 policy、optimizer、actor/critic normalizers、RNG 和完整
ActionBall environment state；不 reset、不 step、不 update，使用 runner 的
`save_exact_resume_roundtrip` 写一次 no-step checkpoint，然后从该 checkpoint 再构造并恢复一次。
两次 runtime 都自然关闭，simulator step 总数必须为 0，core/exact-resume state digest 必须相同。
只检查 checkpoint 字段结构不算通过。

`supervisor_terminal.json` 会绑定 trainer、evaluator、verifier 三个 exact process identity 和
已经复核的 exact-resume receipt。最后由 committed stage evaluator 消费 claim、terminal、
frozen-evaluator append-only
记录、Reward activation audit、runtime bootstrap、final checkpoint 和上面的 exact-resume receipt，
写签名 stage result：

```bash
"$ACTIONBALL_OVERLAY/bin/python" -I -B \
  "$ACTIONBALL_CHECKOUT/hope_training/whole_body_tracking/scripts/action_ball_stage_evidence.py" \
  attest-stage \
  --claim /absolute/stage-namespace/launch_claim.json \
  --authority /absolute/clean/checkout/path/to/stage-evaluator-authority.json \
  --private-key /absolute/operator-control/stage-evaluator-private-key \
  --out /absolute/operator-control/smoke-passed.json
```

`--authority` 是 committed 公钥/源码身份；`--private-key` 是 operator 控制目录中的签名私钥；
`--out` 是新的 no-clobber stage receipt。只有 `status=passed`、checkpoint finite、
`exact_resume_passed=true`、正式 `320+960` 统计门与所有身份/安全/Reward 账本都闭合的签名收据，
才能作为下一阶段 `predecessor_receipt`。

## namespace 与训练输出

launcher 固定：

```text
task.experiment_name=agibot_a3_hope_action_ball_fresh_n5
run_name=<stage namespace basename>
```

trainer 的真实目录应唯一匹配：

```text
<checkout>/hope_training/whole_body_tracking/logs/rsl_rl/
  agibot_a3_hope_action_ball_fresh_n5/
  YYYY-MM-DD_HH-MM-SS_<namespace basename>/
```

stage evaluator 从本 namespace 的 `train.log`、claim 和 runtime bootstrap 反向证明这一路径；目录及
祖先不得是 symlink。namespace 或相同 RSL suffix 已存在时均视为 spent。atomic claim 后失败也
保留全部证据，只能使用新的 attempt namespace 和新的双 owner receipts。

## 常见 fail-closed 结果

- `checkout is dirty`：把代码与科学输入提交后，使用新的 clean detached checkout。
- `code-owned trust set ... is empty`：完成真实证据审查，把 exact digest 进入对应 committed
  trust set；禁止 spec/runtime 自签。
- `exact N=5` / `action order mismatch`：重建 manifest→prototype→registry→admission→evaluator
  整条 ordered lineage，不能只改文件名。
- `runtime inventory ... differs`：新建另一份 overlay，补齐 exact dependency 后重新 mint；不得
  修改共享 venv 或覆盖旧 receipt。
- `GPU lifetime lock is already owned`：等待 owner；不得删除锁或 signal 未知进程。
- `GPU UUID mismatch` / occupied / `nvidia-smi` parse failure：不发射；新卡要新 spec 与 owner
  receipts。
- `sidecar heartbeat ... stale` / request deadline：按 exact stop 规则关闭本 stage；不得临时缩窗、
  改超时或用 trainer 自报结果替代。
- `namespace ... spent` / RSL suffix spent：保留旧证据，换全新 attempt 名。
- `exact-resume ... refused`：阶段不得签 PASS，也不得进入 canary/long；先修 checkpoint/runtime
  可恢复性，再从新的 fresh smoke lineage 开始。
- `result receipt training recipe differs`：不能只改下一阶段 spec；plant、Reward、PPO、seed 或
  overrides 任一改变都要新的 smoke lineage。
