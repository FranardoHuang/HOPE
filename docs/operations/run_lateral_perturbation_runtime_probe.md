# 运行横向平衡扰动的 Isaac full-scene probe

状态：**source-only、尚未运行**。本操作只验证默认关闭的 Isaac adapter/hook 候选能否在 exact
Isaac Lab `v2.1.0` full scene 中为每个 physics substep 显式计算/提交当前 torso COM、同步 direct-setter
boundary、strike/window 清零和 reset 对账。它不是 trainer，不生成 checkpoint，也不授权现役训练、真机或
部署。实验真源见
[EXP-P1-LATERAL-BALANCE-PERTURBATION](../experiments/2026-07/EXP-P1-LATERAL-BALANCE-PERTURBATION.md)，
接口边界见
[lateral perturbation adapter contract](../interfaces/lateral_perturbation_adapter_contract.md)。

## 为什么必须单独 probe

Isaac Lab `v2.1.0` 的内建 external-wrench 路径最终以 `position_data=None` 调用 PhysX；这个参数表示作用在
link transform/origin，不是 COM。首格科学合同要求 torso COM，因此 probe 绕过该内建命令路径：每个
physics substep 都读取当前 link pose 与 local COM offset，再用
`apply_forces_and_torques_at_position(position_data=<explicit COM>, is_global=true)` 提交 WORLD-Y 力。内建
force/torque buffer 始终全零且 unowned，只作为 same-tick/non-torso 竞争 writer 哨兵。subset reset 的额外
full-scene write 前也显式提交**全 batch** zero。

Isaac Lab 没有 getter 能读回 PhysX solver 实际消费的 wrench。本 probe 即使通过，也只产生“同步 direct
setter 已收到显式 COM/WORLD command，且私有 command readback 与内建 buffer 哨兵一致”的 E2 候选证据；输出永远固定
`solver_execution_readback_available=false`、`launch_authorized=false` 和
`training_authorized=false`。之后还需要独立 dynamics-response probe 与 throughput/no-host-sync 重设计。

## 前置条件

1. 使用一个 clean、detached/reviewed 的 `nohope` checkout；记录其完整 commit 为 `SOURCE_COMMIT`。
2. Isaac Lab checkout 必须 clean exact
   `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`（tag `v2.1.0`）。
3. 使用已通过上游 motion 结构门的 exact schema NPZ；每个路径必须是普通文件而不是 symlink。probe 全程
   保留 stable read-only fd，MotionLoader 只收到 `/proc/self/fd/<fd>`（或平台等价 `/dev/fd/<fd>`）而非可
   替换的公开路径；environment load 后和写结果前仍重新核对公开路径 inode/metadata/SHA，同 bytes 换 inode
   也拒绝。
4. 输出 parent 必须人工预建、全路径无 symlink 且位于 source/IsaacLab checkout 之外；probe 在 Kit boot 前
   打开 stable parent dirfd，最后只用 `openat(O_CREAT|O_EXCL|O_NOFOLLOW)` 创建 basename。目标 JSON（包括
   dangling symlink）必须不存在。L0/L1 使用不同的新路径，禁止删除旧回执重跑。
5. 只在 simulator-only 隔离环境运行；不要启动 trainer、worker、judge、vendor stack 或机器人命令。

probe 会把所有 active EventManager term 的 EventTermCfg 行为字段、plain function source identity 与 exact
typed 参数写入 manifest；pinned `SceneEntityCfg` 绑定 selector/resolved ids。未知 config、decorated/method func、
非有限/callable/opaque 参数与任何 interval term 都拒绝，每步前后重验漂移，不使用带地址的 `repr`。reset/startup
term 若留下 external wrench，也会在 adapter attach 或 reset readback
处 fail closed。它不仅关闭已知的 `push_robot`，也不允许另一个未列名的周期 writer 与本 adapter 争 buffer。

本 feature 不在 Pod 上执行以下命令。review 后的首次 full-scene canary 可用：

```bash
cd /path/to/clean/nohope
SOURCE_COMMIT=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=normal)"
test "$(git -C /path/to/IsaacLab rev-parse HEAD)" = \
  21f7136325136ca3f6ca4e0a8125edffe5c24f7e
test -z "$(git -C /path/to/IsaacLab status --porcelain=v1 --untracked-files=normal)"

source hope_training/whole_body_tracking/setup_train_env.sh
hope_isaac_py \
  hope_training/whole_body_tracking/scripts/probe_lateral_perturbation_runtime.py \
  --task HOPE-PingPong-VirtualBall-AgibotA3-v0 \
  --motion-file /absolute/content-addressed/motion.npz \
  --num-envs 32 \
  --steps 100 \
  --cell L1 \
  --source-root "$PWD" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --isaaclab-root /path/to/IsaacLab \
  --output /absolute/no-clobber-evidence/lateral_l1_explicit_com_probe.json \
  --headless \
  --device cuda:0 \
  --execute \
  --confirm SIM_ONLY_PROBE_ONE_LATERAL_WRENCH_RUNTIME
```

对照 `L0` 必须用相同 task/motion/环境数/步数和另一个全新输出路径，只把 `--cell` 改为 `L0`。不要在
一次命令中隐式发两格，也不要因 SSH timeout 自动重放。

## 接受条件

自然退出后，只读解析 JSON 并逐项确认：

- source/IsaacLab/motion SHA 与执行前记录一致，实际执行的 probe script 以及 import 的
  `lateral_probe_artifacts`、`isaac_lateral_perturbation`、`isaaclab`、`isaaclab_tasks` 和
  `whole_body_tracking` module 路径都落在对应 exact clean checkout，而不是 ambient install；这些 checkout、
  module origin 与 motion stable/public identity 会在输出前再验证一次；
- `policy_steps == --steps`，`physics_substeps == steps × env.cfg.decimation`；
- `post_rollout_terminal_zero_overwrite_succeeded=true`、`post_rollout_hook_terminal=true` 且
  `post_rollout_hook_dirty_unknown=false`；这个 clean terminal zero 必须发生在所有回执验收、
  source/module/motion 重验、`openat` 落盘、stdout 打印和 `env.close` 之前。scene hook 恢复失败或 terminal-zero
  setter 异常/错误返回都必须无成功 JSON，且下一 step 被拒绝；
- 每个 direct explicit-COM call 与 scene write 都同步完成，private WORLD command readback exact，内建
  full-articulation force/torque buffers 全零且 `has_external_wrench=false`；每个 substep receipt 的
  `submitted_position_w_full` 不能缺失，torso row 必须等于用该 substep 当前 pose/local COM 复算的 WORLD COM。
  `lifecycle_coverage` 必须同时记录 reset scene
  write、strike-window row 和“active pulse 被 strike 中断后同 step 清零”。没发生的路径必须为
  `observed=false`，对应 reset-zero 字段必须是 `null`，不能用“所有已观察 reset 都通过”的空集合冒充；
- `receipt_transcript_schema=typed_dataclass_tensor_bytes_v1` 且有 64-hex transcript SHA；该摘要覆盖每步
  episode index/step、eligible/strike/safe-window、scheduler/application ledger、所有 physics substep 与 reset，
  不能用只含总数的日志替代；
- 两格都必须出现 eligible/selected opportunity；L0 的非零 backend-submitted application 为 `0`，L1 必须
  有非零 backend-submitted application。该回执只证明 direct setter 接受提交，绝不证明 solver consumed。
  未激活时 probe fail closed，保留日志，不得把零 wrench readback 冒充 treatment 通过；
- result 仍明确 no solver readback/no training/no launch。任何缺字段、非 finite、异常退出、输出已存在、
  checkout/asset 漂移或 scene-write 次数不符都 fail closed，并保全日志与已有输出。

只有 **L1** 的
`status=explicit_com_direct_setter_full_lifecycle_probe_pass_solver_readback_unavailable` 且
`lifecycle_coverage.full_lifecycle_coverage=true` 才能说 full-scene lifecycle 已观察。普通零动作 canary 若没有
自然 reset 或**非零** active-pulse strike interruption，仍可保存为 explicit-COM source 证据，但 status 必须是
`...lifecycle_paths_uncovered...`；不得自动重放直到碰巧通过。后续应预注册独立、确定性的 reset 与 strike
interruption probes，而不是事后修改现有门。

## 本 probe 之后仍然开放的门

- 用至少一个**非零 torso local-COM offset** 初态做“零 wrench vs 非零 wrench”的同 tick
  dynamics-response 配对，证明物理状态变化来自 solver，并排除 link-origin 与 COM torque 混淆；
- exact source review 证明 scene-write closure 没有第二个绕过 Isaac buffer 的 direct PhysX setter；API 没有
  setter-owner/readback，不能只靠内建 buffer 全零哨兵宣称这类 direct competitor 不存在；
- 把 correctness-first host synchronizations 改成 reviewed handoff，并在相同 GPU/scene/seed 下通过
  `>=0.95×` environment-steps/s、`<=1.05×` p95 step time 且 hot path 无 host sync；
- trainer wrapper、标量 activation ledger 与 checkpoint hard contract 已有 E1 source 接线；仍需让本 probe
  的 exact runtime receipt 与训练 source/claim 绑定，并冻结内容寻址 ball-arrival-bin × action-family 留出题表；
- vendor MuJoCo 复核。以上未闭合前，机器预注册必须继续保持 `launch_authorized=false`。
