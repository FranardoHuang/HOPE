# ActionBall 单动作双后端：唯一执行 TODO

> 状态：`ACTIVE / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-18
> `origin/main:docs/NOW.md` 仍是项目优先级唯一权威；本页只维护这条分支的依赖、证据和下一条命令，不建立影子队列。

## 1. 目标

在 `shot_slot_capacity=1`、同一 `action_slot=0` 下，先让 Isaac family A 真实运行，再在同一进程跑到
1000 个 PPO update 并读取中间趋势；随后按相同环境、seed 和训练设置运行 family C。只有 Isaac 的真实
opportunity/contact/outcome/recovery 分母与学习趋势可解释后，才让 MuJoCo GPU A/C 消费同一 MDP
语义。全部运行均为 [`diagnostic_unauthorized`](../DEFINITIONS.md)，不授权 formal promotion、export、
deployment、真机或物理安全。

## 2. HANDOFF §3 的执行规则

| 不接受 | 本线采用的替代 |
| --- | --- |
| fixture 自己造 expected、self-SHA、同一 writer 双写互证 | 真实 writer + 独立 consumer，或能区分实现的固定反例 |
| zero-callpoint、没人消费的 counter、按构造必真的 gate | 删除；只有真实调用点和失败动作接通后才称 gate |
| 比较两个本来没有规格要求相等的量 | 先追两个量各自服务的目的；目的不同则不设任意阈值 |
| 用 rollout 证明确定性几何 | 击球帧 FK、接触状态与弹道直接计算 |
| 要求未训练 policy 先不碰桌、不跌倒、先会击球 | 保留真实机械 hard limit；可学习规避的终止进入 reason、分母和 Reward |
| 用 2/5/14/60 update 判断可学性 | 只作工程 smoke；学习早期趋势至少看同一进程 1000 update |
| 新增 owner/receipt/journal 解决一致性 | 优先复用 upstream loop，删并行语义；一个 root、一个事件边界、一个 WAL |

阻断只剩两类：环境结构使目标不可学，或运行证据与实际状态不一致。其余不完美先记账再训练。

## 3. 当前采用、延后和拒绝

### 采用

- Jiayi/build_2 精确栈：Isaac Sim `5.1.0-rc.19`、IsaacLab `8320e0be…`、Python 3.11.13、
  PyTorch 2.7/cu128、RSL-RL 3.1.2、TensorDict 0.10。
- Git 管理代码；忽略的机器人 USD/大模型只作为外部资产，以源路径和 SHA-256 绑定。
- FullMDP 使用 upstream RSL3 `OnPolicyRunner.learn()`；只在零参数 `alg.update()` 外包一层
  `PENDING fsync -> owner ACK -> EPOCH_ACK fsync -> stdout`。
- 第一条真实训练已按 Franco 新调度切到 Pod1 空卡；源码仍由 Git exact commit 下载，family A
  `N=2 × 2 update` 成功后才启动一个不重启的 A1000。
- A1000 在 20/50/100/200/500/1000 只读里程碑，不因“看起来不好”自动停止。

### 延后

- family C：A 的工程链和第一个 A1000 可解释后再跑，避免同时改变 runtime 与 Reward family。
- checkpoint/restore：当前只存在 owner 结构事务，不存在完整 plant/manager/trainer/RNG fresh-process restore。
  A1000 必须一次自然运行完成，不能声称可恢复长跑。
- 多动作：单动作非零 opportunity/contact/outcome 分母和趋势之前不扩。
- MuJoCo GPU：等真实 211/319 observation、Reward、termination、masked-reset lineage producers 闭合。

### 拒绝

- 旧 Isaac Sim 4.5 / IsaacLab 2.1 / Python 3.10 / RSL-RL 2.3 作为等价环境。
- 把 14.6k 行旧 runner 整体移植到 RSL3；FullMDP 已绕过它，旧路径只保留 legacy 兼容债务。
- 再加一个没有真实 `step/reset` 成功路径的 MuJoCo root 骨架。
- 为了固定动作 V7/V8 的 `robot_hit_table` 终止而改 production reset；这是可学习行为，不是启动门。

## 4. 当前证据与依赖

| 顺序 | 状态 | 事实 | 下一步 |
| ---: | --- | --- | --- |
| 0 | `PASS` | 附件环境合同与 Pod2 实机匹配；build_2 `6144 × 1` 完成，RSL3 ABI 为 `act(TensorDict)` / `process_env_step(TensorDict,...)` | 使用同一栈 |
| 1 | `PASS` | FullMDP 源码已提交并通过 Git 拉到 clean clone；执行 HEAD `758e88e…` | 外部 USD 单独验 SHA |
| 2 | `PASS` | FullMDP lifecycle 重基线到 IsaacLab 8320；exact Kit cfg、train wiring 和 focused union 通过 | 不再维护 2.1 lifecycle |
| 3 | `PASS-live-N2` | 真实 `gym.make -> reset -> forced selected reset`：generation `[1,1] -> [2,1]`，selected row reset、peer row不变、obs/reward finite | 进入 PPO smoke |
| 4 | `PASS-direct` | 397 行 RSL3 adapter direct test：成功顺序、optimizer exception、PENDING fsync failure；Pod host `3 passed` | real `alg.update()` |
| 5 | `HOLD-first-code` | 三次Pod1 fresh canary均穿过完整env/Reward构造后在PPO/WAL前停止：8320 manager cfg dict（`40c6631…5051b9f`）；legacy schema-3误拒FullMDP finite-`q_des`（`dba73962…e5e99a1`）；修复后 exact validator 又揭示 runtime facts 从未把已安装的399-D FullMDP critic写入hard contract（`9e6b066c…35b8a2`）。三个namespace均已消费且不重试 | 只在真实ActionEpoch actor下从live critic manager物化399-D事实 |
| 6 | `NEXT-fresh-fix4` | 第三窄修已提交`f53143f7`并在Pod1 host复跑联合wiring=`283 passed, 1 deselected`。第一份fix3 wrapper因磁盘在检查与执行之间跌破20GiB，guard/log/run均未创建但按no-retry废弃 | 使用同一代码的新docs commit、新Git root、新namespace再跑A `2×2` |
| 7 | `HOLD-learning` | contact/flight/R06 outcome/R07 recovery、per-shot family attribution需要 live v10 分母 | A1000内观察，不作启动门 |
| 8 | `HOLD` | portable restore 缺 Motion/Racket/Physical/R03/R06/R07、plant/manager/action history、trainer/optimizer/RNG和pre-gym reader | 不声称 resume |
| 9 | `HOLD-producers` | MuJoCo 已有 Plant/R05→M04 packed boundary，但缺真实 vector observation/reward/termination/reset lineage 13类producer | producer-first实现后才 `learn(1)` |
| 10 | `PASS-cleanup / HOLD-retention` | 第一轮只删无live-ref cache和旧verify scratch。fix3 preflight又暴露磁盘回落；逐项验证realpath/mount/special file和全`/proc`引用后，删除6个已由Git commit与`/tmp`主日志替代的可重建checkout，实收`2,729,152,512 B`，free=`23,742,586,880 B`。未碰foreign PID、checkpoint、主日志或资产 | fresh canary仍重验20GiB；A1000前再做retention与增长预算 |

## 5. 下一条命令

最近执行、按no-retry废弃的 Pod1 GPU1 wrapper：

```text
.codex-tmp/run_full_mdp_isaac51_rsl3_a_env2_iter2_fix3_pod1_gpu1_20260818.sh
SHA256=be9add821ce665675257b426d7d9a753b4c07c0a75a2e0e896738519f73db792
```

远端副本：

```text
/tmp/run_full_mdp_isaac51_rsl3_a_env2_iter2_fix3_pod1_gpu1_20260818.sh
```

它要求 exact approval、clean Git HEAD、Isaac/IsaacLab/asset identity、至少 20 GiB workspace、GPU1 queue
与 Kit 双锁；拿锁后再次确认物理 GPU UUID、显存和 zero compute PID，才以 no-clobber 创建 guard/log。
它在任何guard/log/run创建前因`/workspace < 20 GiB`以RC70退出；虽然没有消费科学namespace，仍按
一次性命令规则不重试。空间清理只移除6个可由Git重建的旧checkout；三个真实失败主日志继续保留在
`/tmp`。下一份wrapper必须用新的Git root、namespace和approval，并在拿锁后重验磁盘与GPU。

## 6. A1000 同进程里程碑

每个节点只读同一个日志/WAL/TensorBoard，不重启、不改 Reward、不换 seed：

| update | 要回答的问题 |
| ---: | --- |
| 20 | optimizer、Reward20、WAL/ACK、finite、episode reason 和 opportunity 分母是否真实运转 |
| 50 | ready/swing dense income 与 termination rate 是否开始偏离随机初始化 |
| 100 | selected/defer/reject/accept 的比例与 episode length 是否出现方向性变化 |
| 200 | actor-pair contact、flight、outcome、recovery 是否出现非零且可归因的 numerator |
| 500 | 模仿收入、机会率、终止率是否稳定改善而非短暂噪声 |
| 1000 | 才裁决“有早期学习趋势 / Reward经济失衡 / 环境不可学 / 仍未测” |

每个节点至少记录：completed updates/steps、mean episode length、termination reason counts、20个 Reward term
的 signed income、opportunity transactions/due/selected/accept/censor/reject/defer、contact/flight/outcome/
recovery 的 eligible denominator 和 numerator、按 `stroke_family`/side 分组的结果、policy std/LR/KL、
非有限数与 wall time。零分母写 `未测`，不能写 0% 或塞进平均数。

Reward 比例只在 update1000 后按因果证据改：先判断是触发率、权重还是分母问题；A/C 除 observation 和
post-contact family Reward 外保持相同。历史 C 长跑曾看到 action penalty 负正比约
`10.39 -> 3.45`、episode length 谷底后恢复；这只作为需要观察的模式，不直接抄权重。

## 7. 架构减法

当前复杂度不是乒乓任务本身需要，而是历史上叠加了 RSL2/RSL3、legacy/full-MDP、多个 owner、receipt、
journal、checkpoint schema 和重复证明 gate。核心闭包约 14.9 万行，热路径约 11.8 万行；同类参考仓库
的共同训练闭包约 1.1k--1.6k 行。这个差距说明当前实现有真实技术债，不说明任务天然需要 100 倍代码。

减法顺序必须由真实运行证明保护：

1. FullMDP 已绕过 14.6k 行旧 runner，只保留 397 行 RSL3 boundary adapter；
2. A `2×2` 与 A1000 通过后，删除旧 runner 内 FullMDP/R10/resume 分支及其重复 validator；
3. 把五角色 structural carry 与没有生产 restore callpoint的 checkpoint ground-work隔离或删除；
4. MuJoCo 只实现缺失 producer，不再新建平行 root/receipt/schema；
5. 每次删除必须保持真实 writer/consumer、live telemetry和反例，不用 self-SHA 代签。

## 8. 存储边界

“超过 500 GB”是 Pod `/workspace` 配额，不是本机 repo；本机 `nohope` 约 1.56 GiB。本机已按
quarantine→`git fsck` 回收约 470 MB。2026-08-18 Pod1切回执行后，只删除6,902个可重建且无live-ref的
`__pycache__/.pytest_cache`与旧`fullmdp-verify2.pdvCeV` scratch；观测free从约16.53GB升到24GB级。
外部资产、foreign run、checkpoint、canonical logs和单副本证据仍不能按大小直接删；每次fresh run
前重验磁盘增长和至少20GiB余量。
