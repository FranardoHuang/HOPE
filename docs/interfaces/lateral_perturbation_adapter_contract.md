# 横向平衡扰动 adapter 事务接口

Status: `Partial`（已有默认关闭的 Isaac Lab 2.1 adapter/hook、trainer/hard-contract 候选与 mock 回归；
没有 full-scene runtime、solver-response 或 throughput 证据，仍禁止点火）

## 目的与诚实边界

本接口约束恢复/等待窗横向扰动从 scheduler 到 simulator 外力命令的一次应用事务。它防止伪造回执
解锁下一 tick、CUDA 异步失败晚于物理写入、坏回执留下非零 backend，以及同一步 cache 被重放到另一个
live backend。当前仓库有一个默认关闭的 Isaac adapter 候选，并有显式 opt-in trainer wrapper；现役 task
默认路径不包装 `env.step`，机器预注册仍为 `launch_authorized=false`。Isaac Lab 2.1 没有 PhysX
solver-consumed wrench getter，因此不得把 direct
setter 成功或 adapter 私有 command readback 写成物理执行 ACK。

源码、测试和机器合同分别在：

- [`lateral_perturbation.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/lateral_perturbation.py)
- [`isaac_lateral_perturbation.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/isaac_lateral_perturbation.py)
- [`lateral_probe_artifacts.py`](../../hope_training/whole_body_tracking/scripts/lateral_probe_artifacts.py)
- [`test_lateral_perturbation.py`](../../hope_training/whole_body_tracking/tests/test_lateral_perturbation.py)
- [`test_isaac_lateral_perturbation.py`](../../hope_training/whole_body_tracking/tests/test_isaac_lateral_perturbation.py)
- [`phase1_lateral_balance_perturbation_prereg_20260715.json`](../../configs/phase1_lateral_balance_perturbation_prereg_20260715.json)

## 唯一允许的状态机

```text
PLANNED
  -> PREWRITE_VALIDATED
  -> STAGED_NO_SIDE_EFFECT
  -> PRECOMMIT_VALIDATED
  -> PRIVATE_FULL_COMMAND_COMMIT_AND_READBACK
  -> COMMITTED_LEDGER
  -> PER_SUBSTEP_EXPLICIT_COM_DIRECT_SUBMIT
```

- `PREWRITE_VALIDATED`：public step 已与 scheduler 私有 canonical step 完整一致；质量、dtype cast、最终
  WORLD wrench、force 上限、X/Z 零 force 和零 torque 都已 host-visible 地通过。
- `STAGED_NO_SIDE_EFFECT`：adapter 可准备变换或 staging buffer，但不得改变 live backend buffer。
- `PRECOMMIT_VALIDATED`：typed preflight receipt 必须精确回显 source 生成的一次性 token、总质量、WORLD
  force/torque、active mask、transform SHA 和 backend SHA；所有 tensor predicate 必须在 commit 前可见。
- `PRIVATE_FULL_COMMAND_COMMIT_AND_READBACK`：完整 overwrite adapter 私有 WORLD command 后同步 exact
  readback；成功时只返回 `None`。Python/CUDA
  路径不能自证 memory-atomic 或 noexcept，任何异常/非 `None` 都必须在写 application ledger 前把 backend
  标为 terminal `DIRTY/UNKNOWN`，禁止 retry、advance 或下一次 simulator step。
- `COMMITTED_LEDGER`：只有 commit 返回后，scheduler 才能写 application cache；它仍不是 solver ACK。
- `PER_SUBSTEP_EXPLICIT_COM_DIRECT_SUBMIT`：runtime hook 每个 physics substep 都读取当前 link pose 与 PhysX
  local COM offset，计算 WORLD torso COM，并调用
  `ArticulationView.apply_forces_and_torques_at_position(position_data=<explicit COM>, is_global=true)`。任一调用、
  scene write 或随后验证失败都必须 terminal zero-overwrite 并禁止下一 simulator step。

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

## Trainer 入口与 metric 合同

`train.py` 只暴露 `task.lateral_perturbation.enabled/cell/seed`。启用时 `cell` 只能是 `L0`（零推力但占用
相同随机机会/虚拟 pulse）或 `L1`（冻结的 `0.04–0.08 m/s` 横向冲量）；body 固定 `torso_link`、frame
固定 WORLD、作用点固定 link COM、X/Z force 与 torque 固定为零。`enabled=false`/缺失不附加 env cfg 字段、
不产生 lateral hard-contract key、不构造 hook；disabled 同时给 cell/seed、未知 key、非 uint32 seed 或非
L0/L1 cell 都 fail closed。

启用的 checkpoint hard contract 绑定 resolved tick schedule、共同随机题与 hard-safety SHA、Isaac backend/
transform identity、全部 active EventManager term 的 exact typed 参数值与 manifest SHA，以及 metric schema。
唯一非 JSON container 白名单是 pinned Isaac Lab v2.1.0 `SceneEntityCfg`：其完整 dataclass schema、name/names、
resolved ids 与 `preserve_order` 全绑定；EventTermCfg 的 mode/interval/global-time/reset-throttle、plain module
function 的 module/qualname/函数片段 SHA/模块文件 SHA/defaults/closure 也全绑定。未知 configclass、decorated/
method/callable-instance func、非有限/callable/opaque 参数或任一 interval term 都在首次 submit 前 fail closed；
每个 env.step 前后重新 canonical/hash，抓 attach 后 mutation。编码不用对象 `repr`，不会把进程地址写进身份。
trainer 模式不累积完整 per-step receipt；每个成功 step 只复制输出
`extras['log']` 并加入下列标量，不改 environment-owned extras：

- 整数：opportunity、eligible opportunity、selected、nonzero commanded pulse、backend-accepted pulse、
  full-zero overwrite step；
- 浮点：sampled/commanded/backend-accepted 与 abandoned-uncommanded/not-backend-accepted 归一化冲量和；
- 浮点：该 step 实际随机化后 articulation 总质量 min/mean/max。

scheduler/ledger schema v2 已统一使用 `backend_accepted_*`/`not_backend_accepted_*`；preflight 尚未 commit 的
mask 叫 `scheduled_nonzero_force_mask`。scheduler 的 backend-accepted 层只表示 private full-command commit 与
exact readback 成功；trainer 只在该 step 所有 synchronous direct-setter/scene-write submission 完成后发布
这些 metric。两者都绝不表示 PhysX solver 已消费或积分该 wrench；solver-executed 层当前明确 unavailable。

metric key 已存在、输出不是五元 Gym tuple、extras/log 类型错误、T1 event timing、竞争 writer 或 terminal
zero 失败均中止 run。这个 source 接口不等于 launch 授权；full-scene 和 throughput 门仍在下节。

## Adapter 必须提供的合同

adapter 必须声明并实现：

- `preflight_side_effect_free=true`
- `commit_failure_is_terminal=true`
- `discard_is_noexcept=true`
- 稳定的 `application_backend_identity_sha256` 与当前 PhysX articulation view 的 object token
- side-effect-free `preflight_world_wrench_at_body_com(...)`
- `commit_preflighted_world_wrench_at_body_com(...) -> None`
- `discard_preflighted_world_wrench_at_body_com(...) -> None`

这些布尔声明和 source mock 不能证明真实 Isaac adapter 满足合同；特别是不得把 tensor copy、direct setter、
CUDA sync 与私有 readback 自称 atomic/noexcept。当前 candidate 已实现以下 source 路径：

- attach 时要求 Isaac 内建 force/torque buffer 的所有 environment/body row 全零且
  `has_external_wrench=false`；candidate 从不借用该 origin-based buffer，而把它作为竞争 writer 哨兵。每次
  private command copy、每个 substep direct submit 前后、scene write 后和 reset clear 前后都检查完整 buffer
  identity/zero bytes 与 owner flag；同 tick 或 non-torso writer 都 terminal fail closed。strict probe 与
  trainer 都绑定全部 EventManager terms 并拒绝 interval term；
- 从 `root_physx_view.get_masses()` 读取随机化后的全部 body masses，并在 preflight 再次逐值绑定 caller 的
  total mass；
- pinned Isaac Lab `write_data_to_sim()` 使用 `position_data=None`，官方 PhysX tensor 语义是作用在 link
  transform/origin，**不是 COM**。candidate 因此每个 physics substep 读取 `body_pos_w`、scalar-first
  `body_quat_w` 和 `robot.data.com_pos_b` 的 local COM offset，计算当前 WORLD torso COM；pinned IsaacLab
  的 raw `root_physx_view.get_coms()` 不保证与 articulation 同 device，而 `ArticulationData.com_pos_b` 已显式
  `.to(self.device)`，所以合同禁止直接强制 raw tensor device；随后 direct
  setter 使用 WORLD force、显式 WORLD position 与 `is_global=true`。`position_data=None` 被合同禁止；
  输入 pose/COM finite 还不够，旋转/加法后的 torso COM 和 full setter positions 必须再次 finite；float overflow
  必须在 direct setter 前 terminal fail closed；

仓库 [`shadow_ball.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/shadow_ball.py)
与 [`physical_ball.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/physical_ball.py)
中“Isaac Lab 2.1 默认在 COM 施力”的既有注释与固定 PhysX API 文档不符：`position_data=None` 是 link transform
origin。这里不顺手改那两条球路径；球体 COM 近似 origin，真实行为影响需另做独立审计，不能据注释推断。
- subset reset 的额外 full-scene write 前，先检查竞争 writer，再用同一显式 position API 提交**全 batch**
  zero；这样非 reset rows 不会在 decimation 之外多受一次力，
  scheduler 在下一 policy tick 重施仍有效的 pulse，并用 episode index 变化保存 reset rows 的
  sampled/commanded/backend-accepted/abandoned 账；
- dispatch 成功后若 scene write 抛错、setter 返回类型异常、environment output 不合法或后续任何验证失败，
  hook 都进入 terminal guard；若没有竞争 writer，它用最后可信/当前显式 positions 提交全零并禁止继续。
  若发现竞争 writer，则不覆盖对方 bytes，保全证据且在下一 physics step 前终止；
- scene hook 恢复失败也必须向 caller 抛错并进入同一 terminal guard，不能因为发生在 `finally` 就吞掉；正常
  rollout 则必须在读取/校验回执、重验 source、创建输出、打印或关闭环境之前调用一次 clean terminal zero。
  只有 zero setter 同步成功且 hook 仍非 `DIRTY/UNKNOWN` 才能继续发布；后续 step 永久拒绝；
- `enabled=false` 直接返回原始 `env.step(action)`，不读取 scene、command 或 episode state，mock 测试验证
  返回对象 identity 不变。

但 Isaac Lab 2.1 的 `ArticulationView` 没有 solver-consumed external-wrench getter。同步 direct setter +
私有 command readback 只能证明提交边界与输入，不能证明 solver 的实际 wrench。因此必须按
[runtime probe 操作页](../operations/run_lateral_perturbation_runtime_probe.md) 做 exact full-scene 运行，再用独立
dynamics-response probe 补 solver 执行证据；pulse 结束/strike/window/reset 清零、同 GPU throughput 与
no-host-sync 验证也仍待运行。通过前 G05 保持 `Partial`，训练不得启动。

还有一个不可隐瞒的 ownership 边界：PhysX tensor API 没有“direct setter owner”或 setter-command getter。
内建 Isaac buffer 的任意 same-tick/non-torso writer 能被全量哨兵抓住，但若 exact task source 在同一
`scene.write_data_to_sim()` 内绕过 buffer、再次直接调用同一个 PhysX setter，当前 adapter 无法从 API 读回并
区分。strict probe 以 exact clean source closure、active EventManager manifest 和拒绝 interval term 缩小该面，
但 full-scene/source review 前不能宣称 direct setter 独占已获运行证明。

## Isaac Lab 2.1 explicit-COM substep 时序

现役 policy tick 是 50 Hz，内部 `decimation=4`，physics substep 是 200 Hz。更关键的是 pinned Isaac
`Articulation.write_data_to_sim()` 把 `position_data=None` 传给 PhysX，实际作用点是 link origin。把该路径
叫做 torso COM 会改变转矩和科学问题。candidate 改为每个 200 Hz substep 显式重算并传入当前 COM；既不依赖
BODY-frame command buffer，也不允许用 origin treatment 冒充 COM treatment。

因此显式 probe hook 的顺序固定为：

```text
policy scheduler + side-effect-free preflight
  -> private full WORLD command commit/readback
  -> for each physics substep:
       guard full built-in force/torque buffers remain unowned and zero
       -> fresh link origin + quaternion + local COM offset
       -> explicit current torso COM in WORLD
       -> direct PhysX full-articulation submit(position_data != None, is_global=true)
       -> scene.write_data_to_sim
       -> CUDA synchronize
       -> private command exact readback + built-in full-zero guard
  -> reset (if observed)
  -> direct full-batch zero submit at explicit positions
  -> reset-only scene write + synchronized zero/owner guard
  -> episode/strike/window impulse reconciliation
  -> end-of-rollout clean terminal full-batch zero submit
  -> only then receipt validation / source re-attestation / no-clobber publication
```

现有 task ID 没有被替换。probe 只有显式构造
`IsaacLateralPerturbationRuntimeHook(..., enabled=true)` 才进入候选路径；trainer 只有显式 Hydra
`enabled=true + cell + seed` 才构造同一个 hook 的 non-retaining wrapper。default-off 保持原行为；在
full-scene gate 前，源码可 compose 不表示允许点火。

## 中断冲量恒等式

reset、strike 和 safe-window closure 三类原因必须先逐环境保存账，再清 active pulse。每一类都满足：

```text
sampled = commanded + abandoned_uncommanded
commanded = backend_accepted + abandoned_not_backend_accepted
```

runtime-ack 路径中，已成功 commit 的 tick 应有 `backend_accepted == commanded`；plan-only 源码测试允许
`backend_accepted=0`，此时
`abandoned_not_backend_accepted == commanded`。这仍不是 solver ACK。strike/window 中断 tick 仍必须
commit 一次全零 command，不能只清 scheduler 内存。
