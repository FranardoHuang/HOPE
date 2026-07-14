# 横向平衡扰动 adapter 事务接口

Status: `Partial`（只有 E1 源码与单测；没有 Isaac runtime adapter）

## 目的与诚实边界

本接口约束恢复/等待窗横向扰动从 scheduler 到 simulator 外力 buffer 的一次应用事务。它防止伪造回执
解锁下一 tick、CUDA 异步失败晚于物理写入、坏回执留下非零 backend，以及同一步 cache 被重放到另一个
live backend。当前仓库没有实现 Isaac adapter；机器预注册仍为 `launch_authorized=false`。

源码、测试和机器合同分别在：

- [`lateral_perturbation.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/lateral_perturbation.py)
- [`test_lateral_perturbation.py`](../../hope_training/whole_body_tracking/tests/test_lateral_perturbation.py)
- [`phase1_lateral_balance_perturbation_prereg_20260715.json`](../../configs/phase1_lateral_balance_perturbation_prereg_20260715.json)

## 唯一允许的状态机

```text
PLANNED
  -> PREWRITE_VALIDATED
  -> STAGED_NO_SIDE_EFFECT
  -> PRECOMMIT_VALIDATED
  -> ATOMIC_COMMIT
  -> COMMITTED_LEDGER
```

- `PREWRITE_VALIDATED`：public step 已与 scheduler 私有 canonical step 完整一致；质量、dtype cast、最终
  WORLD wrench、force 上限、X/Z 零 force 和零 torque 都已 host-visible 地通过。
- `STAGED_NO_SIDE_EFFECT`：adapter 可准备变换或 staging buffer，但不得改变 live backend buffer。
- `PRECOMMIT_VALIDATED`：typed preflight receipt 必须精确回显 source 生成的一次性 token、总质量、WORLD
  force/torque、active mask、transform SHA 和 backend SHA；所有 tensor predicate 必须在 commit 前可见。
- `ATOMIC_COMMIT`：只接受单次完整 buffer overwrite；函数必须不抛异常并返回 `None`。
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
- `commit_is_atomic_and_noexcept=true`
- `discard_is_noexcept=true`
- 稳定的 `application_backend_identity_sha256` 与当前 live buffer 的 object token
- side-effect-free `preflight_world_wrench_at_body_com(...)`
- `commit_preflighted_world_wrench_at_body_com(...) -> None`
- `discard_preflighted_world_wrench_at_body_com(...) -> None`

这些布尔声明和 source mock 不能证明真实 Isaac adapter 满足合同。必须另做 backend buffer readback、pulse
结束/strike/window 中断后的全零检查、随机化后总质量绑定、同 GPU throughput 与 no-host-sync 验证；通过前
G05 保持 `Partial`。

## 中断冲量恒等式

reset、strike 和 safe-window closure 三类原因必须先逐环境保存账，再清 active pulse。每一类都满足：

```text
sampled = commanded + abandoned_uncommanded
commanded = applied + abandoned_unapplied
```

runtime-ack 路径中，已成功 commit 的 tick 应有 `applied == commanded`；plan-only 源码测试允许
`applied=0`，此时 `abandoned_unapplied == commanded`。strike/window 中断 tick 仍必须 commit 一次全零
buffer，不能只清 scheduler 内存。
