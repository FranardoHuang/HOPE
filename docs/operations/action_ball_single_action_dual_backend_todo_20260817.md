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
- MuJoCo GPU：消费当前FullMDP ActionEpoch `229/399` portable observation；历史A211/C211 `211/319`
  只保留为旧诊断证据。等真实Reward、termination、masked-reset lineage producers闭合。

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
| 5 | `PASS-live-A2x2` | Pod1 exact 5.1/8320/RSL3、GPU1、`N=2 × 2 update`自然RC0：exact 2个optimizer update；WAL 4行严格`PENDING0/ACK0/PENDING1/ACK1`；229/399-D obs、Reward20全有限，无poison/nonfinite | 进入同代码、同N、单进程A1000 |
| 6 | `PASS-A200 / RUNNING-A1000` | A200=`9600 steps/400 WAL/9600 finite/0 poison`；118次selected中105次admitted后均defer/not-ready、13次unknown reject，r03/physical=`0/105`，R06/R07=`未测`；116个episode均tilt，mean length=`81.56` | 继续500/1000；并行只做readiness因果审计，不热改现役run |
| 7 | `HOLD-learning` | contact/flight/R06 outcome/R07 recovery、per-shot family attribution需要 live v10 分母 | A1000内观察，不作启动门 |
| 8 | `HOLD` | portable restore 缺 Motion/Racket/Physical/R03/R06/R07、plant/manager/action history、trainer/optimizer/RNG和pre-gym reader | 不声称 resume |
| 9 | `PASS-live-reset / HOLD-step` | fresh Git `495a0870`真实N=1 reset RC0；`d28a7eac`新namespace又直接断言compiled `physics_dt=0.001`、20 substeps、control dt0.02、registered Warp deviation `noslip=0`，result SHA=`9b40214f…e11eddc` | 接最小WAIT transition：真实action/step、Reward20、五termination和selected reset；RSL3.1.2隔离前禁止`learn(1)` |
| 10 | `PASS-cleanup / PASS-A1000-margin` | 两轮只删无live-ref cache、旧verify scratch和6个Git可重建checkout，后者实收`2,729,152,512 B`；外部并发清理后末次只读free=`31,737,970,688 B`。未碰foreign PID、checkpoint、主日志或资产 | A1000仍在拿锁后重验；不按目录表观大小删硬链接/资产 |

## 5. 下一条命令

已执行且不可复用的 Pod1 GPU1 engineering canary：

```text
.codex-tmp/run_full_mdp_isaac51_rsl3_a_env2_iter2_fix4_pod1_gpu1_20260818.sh
SHA256=9dcb0bcb54e163eb2bcd1b2b22da48e4a371580e875cbf59cebe1d6696109a08
```

远端副本：

```text
/tmp/run_full_mdp_isaac51_rsl3_a_env2_iter2_fix4_pod1_gpu1_20260818.sh
```

它在 fresh clean HEAD `86e36270…`、exact asset/环境、双锁和空物理GPU1上自然RC0；主日志
`/tmp/action-ball-20260818T080000Isaac51Rsl3FullMdpAEnv2Iter2Fix4Pod1GPU1CST.log` SHA256=
`51b26a37d624124dac03542ec35e4d3ecbb83717ba0242bbf9ab82f3333317ed`。WAL SHA256=
`ba87926c8dd9beaf9df5c12a161d6624cdfc66f63e2b0f56d34fec75206acbe5`，4行newline-terminated JSONL
严格闭合两个update的pending/ack key与step/commit/op/drain frontier。下一份A1000 wrapper必须使用新的
Git root、namespace、approval和锁；`max_iterations=1000`且不生成checkpoint、不声称resume。

A1000现役运行（禁止修改、复用或重启）：Git HEAD=`4d374ca8…`，namespace=
`20260818T082100Isaac51Rsl3FullMdpAEnv2Iter1000Pod1GPU1CST`，wrapper SHA256=
`9a6fc9194be7099e4c1ee0c0db8fdc782a763f594316e33de1c3fdd3a30707ed`，物理GPU1 UUID=
`GPU-a8f7dd24-1162-15d4-2f22-7552ce2a6cb6`。它通过GitHub fresh clone取得代码；外部USD仍按路径与
SHA-256绑定。运行期间只读日志/WAL/TensorBoard，不在active checkout热补丁。

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

update100的WAL前缀闭合到ACK99：`1,297,132 B`，SHA256=
`fdce022ba481fe7848619c3fa59244485629900b59df3de5ffda94e0d4606ed9`。累计100个update中，
Reward=`4800/4800` finite、conservation violation=`0`、actual sum=`276.6504531`；58次due/selected中
54次forehand construction/key admitted后全部defer/not-ready，4次unknown reject，仍为0 ACCEPT、
0 playback/launch/contact/outcome/recovery。50→100窗口为30次selected、29次admitted、29次defer、
1次reject；30个episode均base tilt，窗口mean length=`84.47`。六个dense income累计=`276.6504594`，
约`0.05764/sample`，相比update20/50没有上升。TensorBoard step99仅作learner辅证：value loss=
`5.34370`、surrogate=`-0.07934`、noise std=`0.0200275`、LR=`1e-5`；RSL-RL 3.1.2没有落KL。
这仍只说明readiness入口尚未出现，不足以区分课程触发、模仿梯度和权重经济，故继续update200。

update200前缀闭合到ACK199：`2,597,276 B`，SHA256=
`6132cf21fd971f437554fc5e2070ff9f66bd60614a54ce88b35f2a8e71c82045`。累计9600个Reward sample
全部finite、0 conservation violation；118次selected中105次forehand admitted仍全部not-ready，13次
unknown reject，ACCEPT/playback/contact保持0。100→200窗口60次selected、51次admitted、51次defer、
9次reject；60个episode全tilt，窗口mean length=`78.02`，累计mean=`81.56`。dense income累计=
`547.6872`，约`0.05705/sample`；窗口约`0.05647/sample`，继续缓慢下降。TensorBoard step199的
value loss=`0.01449`、surrogate=`-0.03678`、std=`0.0200528`、LR=`1e-5`。运行保持可信并继续500；
但连续200 update为0 ACCEPT已足以并行启动只读readiness producer/阈值因果审计，不能靠调权重猜根因。

## 7. 架构减法

当前复杂度不是乒乓任务本身需要，而是历史上叠加了 RSL2/RSL3、legacy/full-MDP、多个 owner、receipt、
journal、checkpoint schema 和重复证明 gate。核心闭包约 14.9 万行，热路径约 11.8 万行；同类参考仓库
的共同训练闭包约 1.1k--1.6k 行。这个差距说明当前实现有真实技术债，不说明任务天然需要 100 倍代码。

减法顺序必须由真实运行证明保护：

1. FullMDP 已绕过 14.6k 行旧 runner，只保留 397 行 RSL3 boundary adapter；
2. A `2×2` 与 A1000 通过后，删除旧 runner 内 FullMDP/R10/resume 分支及其重复 validator；
3. 把五角色 structural carry 与没有生产 restore callpoint的 checkpoint ground-work隔离或删除；
4. MuJoCo 不迁移73文件/约12.4万行未跟踪WIP；从tracked plant的真实reset/step callpoint只实现当前
   FullMDP 229/399所需producer，不再新建平行root/receipt/schema；
5. 每次删除必须保持真实 writer/consumer、live telemetry和反例，不用 self-SHA 代签。

## 8. 存储边界

“超过 500 GB”是 Pod `/workspace` 配额，不是本机 repo；本机 `nohope` 约 1.56 GiB。本机已按
quarantine→`git fsck` 回收约 470 MB。2026-08-18 Pod1切回执行后，只删除6,902个可重建且无live-ref的
`__pycache__/.pytest_cache`与旧`fullmdp-verify2.pdvCeV` scratch；观测free从约16.53GB升到24GB级。
外部资产、foreign run、checkpoint、canonical logs和单副本证据仍不能按大小直接删；每次fresh run
前重验磁盘增长。fix4结束后曾为`22,794,698,752 B`；外部并发清理后末次只读值为
`31,737,970,688 B`，已满足本次A1000的20GiB启动门和增长余量。审计仍无法证明额外3GiB可安全删除，
所以不把`codexschema`、asset、日志或有live ref的目录包装成清理候选。
