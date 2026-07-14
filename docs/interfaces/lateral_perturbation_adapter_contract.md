# 横向平衡扰动 adapter 事务接口

Status: `Partial`（已有默认关闭的 Isaac Lab 2.1 adapter/hook 候选与 mock 回归；没有 full-scene runtime 证据）

## 目的与诚实边界

本接口约束恢复/等待窗横向扰动从 scheduler 到 simulator 外力 buffer 的一次应用事务。它防止伪造回执
解锁下一 tick、CUDA 异步失败晚于物理写入、坏回执留下非零 backend，以及同一步 cache 被重放到另一个
live backend。当前仓库有一个不改变现役 task registration 的 probe-only Isaac adapter 候选；机器预注册
仍为 `launch_authorized=false`。Isaac Lab 2.1 只能读回待提交的 articulation command buffer，不能读回
PhysX solver 真正消费的 wrench，因此不得把 buffer readback 写成物理执行 ACK。

源码、测试和机器合同分别在：

- [`lateral_perturbation.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/lateral_perturbation.py)
- [`isaac_lateral_perturbation.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/isaac_lateral_perturbation.py)
- [`test_lateral_perturbation.py`](../../hope_training/whole_body_tracking/tests/test_lateral_perturbation.py)
- [`test_isaac_lateral_perturbation.py`](../../hope_training/whole_body_tracking/tests/test_isaac_lateral_perturbation.py)
- [`phase1_lateral_balance_perturbation_prereg_20260715.json`](../../configs/phase1_lateral_balance_perturbation_prereg_20260715.json)

## 唯一允许的状态机

```text
PLANNED
  -> PREWRITE_VALIDATED
  -> STAGED_NO_SIDE_EFFECT
  -> PRECOMMIT_VALIDATED
  -> FULL_BUFFER_COMMIT_AND_READBACK
  -> COMMITTED_LEDGER
```

- `PREWRITE_VALIDATED`：public step 已与 scheduler 私有 canonical step 完整一致；质量、dtype cast、最终
  WORLD wrench、force 上限、X/Z 零 force 和零 torque 都已 host-visible 地通过。
- `STAGED_NO_SIDE_EFFECT`：adapter 可准备变换或 staging buffer，但不得改变 live backend buffer。
- `PRECOMMIT_VALIDATED`：typed preflight receipt 必须精确回显 source 生成的一次性 token、总质量、WORLD
  force/torque、active mask、transform SHA 和 backend SHA；所有 tensor predicate 必须在 commit 前可见。
- `FULL_BUFFER_COMMIT_AND_READBACK`：完整 overwrite 后同步 exact readback；成功时只返回 `None`。Python/CUDA
  路径不能自证 memory-atomic 或 noexcept，任何异常/非 `None` 都必须在写 application ledger 前把 backend
  标为 terminal `DIRTY/UNKNOWN`，禁止 retry、advance 或下一次 simulator step。
- `COMMITTED_LEDGER`：只有 commit 返回后，scheduler 才能写 application cache、计数并解锁下一 tick。

preflight 拒绝时必须无副作用 discard staging，回到 `PLANNED`，允许同 step token 的 canonical command
用一个全新 preflight nonce 重新 dispatch。commit 一旦进入后若抛异常或返回非 `None`，backend 标记为
`DIRTY/UNKNOWN`；普通 retry、同 tick 重发和下一 tick 都必须拒绝，直到独立审核的全零 clear+readback 或
直接终止该 run。

## 身份与 capability

- 不存在公开的 `acknowledge_application`。scheduler bookkeeping 只接受模块内 dispatch identity
  capability，所有 expected command/mask/count 只从 scheduler 私有 canonical step 与已验证实际总质量
  推导。这个 Python object identity 是防误用接口，不是抵抗同进程恶意 introspection 的安全边界。
- preflight token 由 source 为每次事务生成；adapter 只能原样回显并消费同一个 object。旧 token、foreign
  token 或 `None` 在 commit 前拒绝。
- application cache 同时绑定 transform SHA、adapter/backend SHA 与 live backend object token。同 SHA 的
  新 adapter 实例不能拿旧 cache 冒充已写。

## Adapter 必须提供的合同

adapter 必须声明并实现：

- `preflight_side_effect_free=true`
- `commit_failure_is_terminal=true`
- `discard_is_noexcept=true`
- 稳定的 `application_backend_identity_sha256` 与当前 live buffer 的 object token
- side-effect-free `preflight_world_wrench_at_body_com(...)`
- `commit_preflighted_world_wrench_at_body_com(...) -> None`
- `discard_preflighted_world_wrench_at_body_com(...) -> None`

这些布尔声明和 source mock 不能证明真实 Isaac adapter 满足合同；特别是不得把两次 tensor copy、CUDA sync
与 readback 自称 atomic/noexcept。当前 candidate 已实现以下 source 路径：

- attach 时拒绝接管任何已有非零 robot wrench owner；随后每步覆盖整台 articulation buffer，只有
  `torso_link` 可非零；即使 buffer 全零，只要 `has_external_wrench=true` 也视作已有 owner。每次 preflight
  还会把 live force/torque object identity 和完整 bytes 与上一次自身 readback 对账，其他 writer 在 policy
  steps 之间改写也会 terminal fail closed；strict probe 另外绑定全部 EventManager terms 并拒绝 interval term；
- 从 `root_physx_view.get_masses()` 读取随机化后的全部 body masses，并在 preflight 再次逐值绑定 caller 的
  total mass；
- 由于 Isaac v2.1 buffer 是 BODY frame，每个 physics substep 都用当下 `body_quat_w` 重新做
  WORLD→BODY 变换，然后在 `scene.write_data_to_sim()` 后同步 CUDA 并逐值读回完整 articulation
  force/torque command buffer；任一非 torso row 或 buffer object identity 漂移都将 backend 标成
  `DIRTY/UNKNOWN`；
- subset reset 的额外 full-scene write 前，先确认 reset rows 已被 articulation reset 清零，再主动清空**全
  batch** wrench buffer并在 write 后同步读回全零；这样非 reset rows 不会在 decimation 之外多受一次力，
  scheduler 在下一 policy tick 重施仍有效的 pulse，并用 episode index 变化保存 reset rows 的
  sampled/commanded/applied/abandoned 账；
- `enabled=false` 直接返回原始 `env.step(action)`，不读取 scene、command 或 episode state，mock 测试验证
  返回对象 identity 不变。

但 Isaac Lab 2.1 的 `ArticulationView` 没有 solver-consumed external-wrench getter。同步 enqueue + command
buffer readback 只能证明提交边界与输入 buffer，不能证明 solver 的实际 wrench。因此必须按
[runtime probe 操作页](../operations/run_lateral_perturbation_runtime_probe.md) 做 exact full-scene 运行，再用独立
dynamics-response probe 补 solver 执行证据；pulse 结束/strike/window/reset 清零、同 GPU throughput 与
no-host-sync 验证也仍待运行。通过前 G05 保持 `Partial`，训练不得启动。

## Isaac Lab 2.1 substep 时序

现役 policy tick 是 50 Hz，内部 `decimation=4`，physics substep 是 200 Hz。Isaac v2.1 的
`set_external_force_and_torque` 只改 BODY-frame buffer；`InteractiveScene.write_data_to_sim()` 才在每个
physics substep 前把 buffer 送给 PhysX。若只在 policy tick 起点变换一次，躯干在后续 3 个 substep 旋转时，
WORLD force 会跟着 BODY frame 偏转，已经违反首格的 WORLD-Y 合同。

因此显式 probe hook 的顺序固定为：

```text
policy scheduler + side-effect-free preflight
  -> first command-buffer commit/readback
  -> for each physics substep:
       fresh torso quaternion
       -> WORLD-to-BODY transform
       -> full robot wrench-buffer overwrite
       -> scene.write_data_to_sim
       -> CUDA synchronize
       -> exact command-buffer readback
  -> termination/reset
  -> verify reset rows zero
  -> full-batch zero overwrite
  -> reset-only scene write + synchronized full-batch zero readback
  -> episode/strike/window impulse reconciliation
```

现有 task ID 没有被替换。只有显式构造 `IsaacLateralPerturbationRuntimeHook(..., enabled=true)` 并通过
该 hook 的 `step` 才进入候选路径；这是为了让 default-off 保持原行为，也避免在 full-scene gate 前偷偷把
probe 变成训练功能。

## 中断冲量恒等式

reset、strike 和 safe-window closure 三类原因必须先逐环境保存账，再清 active pulse。每一类都满足：

```text
sampled = commanded + abandoned_uncommanded
commanded = applied + abandoned_unapplied
```

runtime-ack 路径中，已成功 commit 的 tick 应有 `applied == commanded`；plan-only 源码测试允许
`applied=0`，此时 `abandoned_unapplied == commanded`。strike/window 中断 tick 仍必须 commit 一次全零
buffer，不能只清 scheduler 内存。
