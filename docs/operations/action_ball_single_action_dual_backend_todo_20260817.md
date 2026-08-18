# ActionBall 单动作双后端：唯一执行 TODO

> 状态：`ACTIVE / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-19
> `origin/main:docs/NOW.md` 仍是项目优先级唯一权威；本页只维护这条分支的依赖、证据和下一条命令，不建立影子队列。

## 1. 目标

在 `shot_slot_capacity=1`、同一 `action_slot=0` 下，先让 Isaac family A 以 `4096 env` 真实运行，
在同一进程跑到1000 个 PPO update并读取中间趋势；随后按相同环境、seed 和训练设置运行 family C。
`N=2`只允许回答构造、reset、ABI和optimizer调用点，不作为学习效果或长跑证据。只有 Isaac 的真实
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
- 第一条可信学习 run 使用 Git exact commit 和 `4096 env × 1000 update`；前5个update只是同一进程的
  construction/capacity/finite观察窗，通过后自然继续，不另起或重跑smoke。
- A1000 在 5/20/50/100/200/500/1000 只读里程碑，不为里程碑重启，也不因“看起来不好”自动停止。
- Pod1 按物理 GPU 与 CPU locality 调度：每卡最多两个进程；每个新进程固定独立 CPU 核组，
  不再用一个全 Pod 生命周期 Kit 锁把三张卡串成一张卡。共卡只要求显存余量、同卡进程上限和
  独立 namespace，不把共卡 wall time 当单进程性能证据。
- 在 portable FullMDP MuJoCo A 尚未闭合前，先运行已有真实 court/ball/contact/reward 的 native A
  长跑取得 Reward 比例、episode length、`action_rate_l2` 和 contact 趋势；它是工程/学习基线，
  不能冒充 229/399-D portable A。

### 延后

- family C：A 的工程链和第一个 A1000 可解释后再跑，避免同时改变 runtime 与 Reward family。
- checkpoint/restore：当前只存在 owner 结构事务，不存在完整 plant/manager/trainer/RNG fresh-process restore。
  A1000 必须一次自然运行完成，不能声称可恢复长跑。
- 多动作：单动作非零 opportunity/contact/outcome 分母和趋势之前不扩。
- MuJoCo portable GPU A：消费当前FullMDP ActionEpoch `229/399` portable observation；历史A211/C211
  `211/319`只保留为旧诊断证据。真实Reward、termination、masked-reset lineage producer仍待闭合；
  这不再阻塞独立的native A学习基线。

### 拒绝

- 旧 Isaac Sim 4.5 / IsaacLab 2.1 / Python 3.10 / RSL-RL 2.3 作为等价环境。
- 把 14.6k 行旧 runner 整体移植到 RSL3；FullMDP 已绕过它，旧路径只保留 legacy 兼容债务。
- 再加一个没有真实 `step/reset` 成功路径的 MuJoCo root 骨架。
- 为了固定动作 V7/V8 的 `robot_hit_table` 终止而改 production reset；这是可学习行为，不是启动门。

## 4. 当前证据与依赖

先把“对齐”拆开，禁止用一个宽度或一次update代签整套MDP：

| 轴 | Isaac FullMDP A | portable MuJoCo WAIT | native MuJoCo A | 当前裁决 |
| --- | --- | --- | --- | --- |
| 环境数 | 下一条必须4096；旧证据N=2 | N=2 learn(1) | N=1024×1000 | 未对齐 |
| actor/critic/action | 229/399/31 | 229/399/31 | 114/114/31 | native未对齐 |
| RSL/PPO | RSL3.1.2 thin adapter | RSL3.1.2 upstream | native trainer | 只对齐前两列训练ABI |
| reset/plant | FullMDP generation/selected reset | WAIT selected reset已过 | native episode reset | lifecycle未对齐 |
| question/reveal/launch | 完整producer | 全部缺席，`idle_wait_only` | 简化serve | portable未对齐 |
| Reward | Reward20 | 仅6项dense有真实值 | 10-term简化Reward | 未对齐 |
| contact/outcome/recovery | R03/physical/R06/R07 | 尚未接 | 简化binary contact | 未对齐 |
| table终止 | component-OBB SAT | SAT已接 | table contact多为telemetry | native未对齐 |
| terrain | nominal plane | nominal plane | plane | baseline对齐；rough尚未live |

因此小N和native长跑都不能回答最终policy质量；它们只分别关闭工程调用点和提供下一代吞吐/Reward趋势。

| 顺序 | 状态 | 事实 | 下一步 |
| ---: | --- | --- | --- |
| 0 | `PASS` | 附件环境合同与 Pod2 实机匹配；build_2 `6144 × 1` 完成，RSL3 ABI 为 `act(TensorDict)` / `process_env_step(TensorDict,...)` | 使用同一栈 |
| 1 | `PASS` | FullMDP 源码已提交并通过 Git 拉到 clean clone；执行 HEAD `758e88e…` | 外部 USD 单独验 SHA |
| 2 | `PASS` | FullMDP lifecycle 重基线到 IsaacLab 8320；exact Kit cfg、train wiring 和 focused union 通过 | 不再维护 2.1 lifecycle |
| 3 | `PASS-live-N2` | 真实 `gym.make -> reset -> forced selected reset`：generation `[1,1] -> [2,1]`，selected row reset、peer row不变、obs/reward finite | 进入 PPO smoke |
| 4 | `PASS-direct` | 397 行 RSL3 adapter direct test：成功顺序、optimizer exception、PENDING fsync failure；Pod host `3 passed` | real `alg.update()` |
| 5 | `PASS-live-A2x2 / historical-engineering` | Pod1 exact 5.1/8320/RSL3、GPU1、`N=2 × 2 update`自然RC0：exact 2个optimizer update；WAL 4行严格`PENDING0/ACK0/PENDING1/ACK1`；229/399-D obs、Reward20全有限，无poison/nonfinite | 只保留调用点证据；不再续成小N学习run |
| 6 | `PASS-engineering-N2 / FAIL-scientific-long` | `N=2 × 2`工程门闭合；随后`N=2` A1000只到ACK470并因LM device assert停滞。可信前缀全部finite，但296个episode全tilt、260个admitted opportunity全not-ready、ACCEPT=0；环境数和任务入口都不足以形成学习结论 | 不再重复小N smoke；fresh同一进程直接`4096 × 1000`，前5次只读观察 |
| 7 | `PASS-host-chain / HOLD-fresh-live` | bootstrap两拍ready仍授权Motion，但不再以neutral key写R07 ActionEpoch telemetry；CPU真实fact→owner projection→Motion reveal→production D05 settle已得到两行ACCEPT、Epoch无overflow。contact/flight/R06 outcome/R07 recovery仍需下一条fresh live分母 | 在4096长跑同一进程观察首个ACCEPT与真实分母；不再为它单开N=2 run |
| 8 | `HOLD` | portable restore 缺 Motion/Racket/Physical/R03/R06/R07、plant/manager/action history、trainer/optimizer/RNG和pre-gym reader | 不声称 resume |
| 9 | `PASS-live-step / PASS-live-SAT` | commit `61887b43…` 的fresh Pod1 GPU0共卡门真实完成`19 passed,0 skipped`：N=1 reset/step、N=2 masked reset、float32 device SAT及attached `robot/*` authority全过。run结束仍只有原1个peer，queue lock自然释放 | 保持同一SAT producer；下一步接真实A lifecycle，不再扩identity gate |
| 10 | `PASS-cleanup / PASS-A1000-margin` | 外部清理后Pod1约`249.8 GiB` free；A1000 ACK417时run目录仅约6.6MB，预计到1000新增日志不足约10MB，即使终点单checkpoint也远低于空间余量。未碰foreign PID、checkpoint、主日志或资产 | 不再为本run清理；只读监控实际增长，不按表观du删除硬链接/资产 |
| 11 | `PASS-live-WAIT-learn1 / HOLD-full-A` | 同一fresh checkout在19-test后直接调用upstream RSL-RL3.1.2：`N=2 × 24`完成1次PPO update、48 transitions、229/399宽度，`final_rc=0`；result SHA=`322592ce…f07a`。lifecycle仍明确`idle_wait_only`，没有question/contact/outcome | 复用该VecEnv/runner接真实A reveal→flight→outcome；先短live再启动MuJoCo A长跑 |
| 12 | `PASS-live-native-A1000 / HOLD-portable-A` | native A以`1024 env × 1000`自然完成：后500-update窗口binary racket-ball contact=`4078/87546=4.658%`，明显高于50--100窗口`6/8452=0.071%`；但robot-table episode fraction仍约97%，ABI为114/114、10-term简化Reward，缺WAIT/ActionEpoch/outcome/recovery | 只作下一代吞吐与Reward经济参考；不可写成portable 229/399 FullMDP A成功 |
| 13 | `PASS-Pod-CUDA` | LM code 8/9 已统一为`lm_solve_info_nonzero/lm_solve_nonfinite`；`dq`非有限、`solve_ex info!=0`以及有限`q+dq`溢出都在任何物理forward前逐行拒绝，正常peer保持不变。exact Pod1 Git `2c8ef444…`、Jiayi Python3.11/Torch2.7-cu128在GPU0得到三参数`3 passed`，每次后续kernel与synchronize正常 | 进入唯一4096同进程；真实异常仍需写row/iteration/try/info以继续收窄 |
| 14 | `PASS-structural / HOLD-live-business` | RSL3薄adapter升级独立v11 telemetry：compact joint-safety在optimizer前prepare/validate，随后`PENDING fsync -> Epoch ACK -> EPOCH_ACK fsync -> durable latch -> safety ACK`；4096×5单元反例得到5组pair和5份post-ACK receipt | 4096同进程前5次必须独立看到5份v11 safety receipt，并要求真实D05 producer/counter被调用；单元fixture的D05全零不能代签 |
| 15 | `PASS-live-contact-slice / HOLD-portable-A` | portable MuJoCo新增诚实的`full_a_slice_attempted`纵切片：逐行reveal、真实ball state launch、20-substep plant、live contact、bounded terminal与selected reset；receipt固定`full_a_complete=false`。exact Pod1 Git `2c8ef444…`、GPU1节点从measured racket site构造真实ball-racket pair并穿production step/latch，`1 passed`。R03、R06 landing outcome、R07 recovery及Reward项0--13仍明确`not_produced` | 并行逐项接通缺失producer；未闭合前不得称MuJoCo A，但不再阻塞Isaac 4096 A1000 |
| 16 | `FAIL-pre-PPO / ROOT-CAUSE-FIXED-HOST` | first 4096 one-shot消费commit `5ee1ffa6…`、wrapper `022c13f5…`和fresh namespace；GPU preexec、sealed archive及真实Kit Python身份通过，但wrapper在`AppLauncher`前导入Torch/RSL，Kit startup后约0.34秒segfault。Hydra已解析4096，scene/PPO/WAL均零调用；这不是容量或Reward失败 | runtime identity改成两阶段：AppLauncher前只验不可变解释器/archive，AppLauncher成功后同一Kit进程才导入并核Torch/RSL，再进入`_run`；新commit/namespace/wrapper可直发同一4096长跑，不复用本次证据 |
| 17 | `FAIL-pre-App / ROOT-CAUSE-NARROWED` | second 4096 one-shot消费commit `d341931c…`、wrapper `60cfdeed…`和fresh namespace；preexec通过、v2 receipt为空，Kit仍在约2秒内segfault，scene/PPO/WAL零调用。pre-App状态机已证明Torch/RSL未提前导入，所以它不是唯一根因；两次失败共同剩下的异常入口是`python.sh -P -S -B -c runpy(...)` | successor回到成功N2使用的direct script入口，仅保留`-P/-B`和production内pre/post-App attestation；删除Kit Python的`-S/-c/runpy`，仍以fresh namespace直发4096长跑，不插smoke |
| 18 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | third 4096 one-shot消费commit `356f706b…`、wrapper `00e340b7…`和fresh namespace；normal `python.sh -P -B train.py` 已稳定进入AppLauncher并完成约10秒Kit启动，随后production post-App gate拒绝AppLauncher合法加载的Torch。preexec通过、v2 receipt/scene/PPO/WAL仍为零；GPU与锁自然释放 | 不新增门；把post-App门缩到真正必须未加载的RSL/TensorDict，App前仍拒绝Torch/RSL/TensorDict，并继续对App-owned Torch核exact版本与venv来源。新commit/namespace直发同一4096长跑，不插smoke |

## 5. 下一条发射协议

不再生成`N=2×2 -> 停 -> 修 -> N=2×2`的重复流水账。下一条Isaac A只允许一个fresh namespace：

1. exact clean Git、Isaac5.1/IsaacLab8320/RSL3、资产SHA、空物理GPU和独立CPU affinity一次性闭合；
2. `num_envs=4096`、`max_iterations=1000`直接启动，update0--4就是同一进程的scale/finite门；
3. update5健康即自然继续，不重启、不换seed、不改Reward；只读20/50/100/200/500/1000；
4. 仅在进程失败、证据不可信或结构性不可学被直接证明时停止；普通tilt/table/fall只记telemetry；
5. 长跑期间并行完成portable MuJoCo A lifecycle、rough课程和2.0删除清单，不热补正在运行的源码或scene。

第一份一次性4096 wrapper已经消费且自然RC1：commit=`5ee1ffa6…`、wrapper SHA256=`022c13f5…`、
namespace=`20260819T044500FullMdpA4096Iter1000Git5EE1FFA6Pod1GPU0CST`。preexec receipt和旧v1 runtime
receipt都落盘，但`run.log`只有identity marker和Kit segfault；无`Learning iteration`、无WAL、无GPU
compute驻留。根因是identity代码在`AppLauncher`之前导入Torch/RSL，违反训练入口原有“先启动Isaac Sim
再导入runtime模块”的顺序。该namespace不得复用、不得重试，也不能写成4096容量失败。

successor保持`4096×1000`、fresh namespace、GPU0/CPU32--47、最多一个已知peer、20 GiB余量和同进程
update0--4；唯一变化是两阶段identity门。pre-App阶段验证sealed bytes/interpreter且拒绝Hydra预载
Torch/RSL/TensorDict；opt-in precheck一经尝试即不可重试，成功证明连同五项attestation值以
`unchecked→checked→consumed`一次性交接，post-App
入口先消费、再核值未漂移并拒绝AppLauncher预载，然后在
真实Kit进程中导入并核Torch/TensorDict/RSL class、fsync v2 receipt，然后才调用`_run`。新ignored
one-shot仍不进Git；最终source commit、wrapper SHA、namespace和result由发射件与remote保留证据共同记录。

second one-shot在上述pre/post-App状态机完整生效后仍于AppLauncher startup内segfault：preexec receipt已fsync，
v2 receipt、scene、PPO和WAL均未出现。它推翻了“pre-App Torch/RSL是唯一根因”，但没有提供4096容量信息。
下一件只改启动形状：不再用Kit Python的`-S/-c/runpy`代理，直接以
`python.sh -P -B scripts/train.py`进入同一production attestation；不是重跑已消费namespace。

third one-shot验证direct script入口确实消除了前两次segfault：AppLauncher正常打印设备、experience、系统与
P2P信息并存活约10秒。随后首个Python错误来自post-App门把AppLauncher合法加载的Torch误归为policy runtime；
这同样发生在scene/PPO/WAL之前，不是4096容量或学习反例。下一件只缩窄这道门：pre-App继续证明Torch、
TensorDict、RSL都未加载；post-App允许Torch并在既有ABI/venv来源门中验证，只要求TensorDict/RSL仍未加载。

旧失败run只有在以下条件同时成立时才能按已绑定的唯一PID链停止：LM exact Pod异常路径零skip、v11
adapter真实callpoint可被前5次消费、portable MuJoCo缺失项已被明确列出且没有被成功receipt掩盖、最终
commit/wrapper经独立红队。旧PID随后已自然消失，因此本轮没有发signal/kill。第一份4096 one-shot也已
自然失败并释放GPU/锁；successor只剩post-App identity回归、final commit/wrapper终审和临门资源重验。
通过后仍直接启动长跑，不能插入小N或独立5-update smoke。
下面保留旧命令与失败证据，防止误复用。

### 已消费旧命令

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

A1000已在可信边界ACK470后失败并停滞（仍禁止修改、复用、signal或重启）：Git HEAD=`4d374ca8…`，namespace=
`20260818T082100Isaac51Rsl3FullMdpAEnv2Iter1000Pod1GPU1CST`，wrapper SHA256=
`9a6fc9194be7099e4c1ee0c0db8fdc782a763f594316e33de1c3fdd3a30707ed`，物理GPU1 UUID=
`GPU-a8f7dd24-1162-15d4-2f22-7552ce2a6cb6`。它通过GitHub fresh clone取得代码；外部USD仍按路径与
SHA-256绑定。最后完整WAL SHA256=`ba80b955…bcceb7`，log SHA256=`505751f4…ab293f`；根device assert
来自`stroke_adapt_torch.py`的fixed-try LM守卫，证明至少一个active candidate的`solve_ex info!=0`或
`dq`非有限，但未落具体row/类别。其后的PhysX view与`torch.isfinite(reference_hit)`只是坏CUDA context
的异步观察面，不能冒充根kernel。fresh源码已经把该条件改成普通的construction rejection：无效candidate
不更新`q/r/cost`，逐行记录`lm_solve_info_nonzero`或`lm_solve_nonfinite`，不再调用`_assert_async`。
旧进程、active checkout和锁继续只读保留；下一条fresh run必须使用新Git root、namespace、approval和真实空卡。

native A canary执行commit=`08d17b74…`、namespace=
`20260818T135500MujocoNativeACanary08D17B74Fix1Pod1GPU2SharedCST`，result SHA256=
`8847a1b5…e09c`。它完成2 update，capacity=`PASS_NO_OVERFLOW`，binary contact仍为0；短跑只证明链路。
同一输入/seed/Reward/PPO的A1000随后以fresh namespace
`20260818T140500MujocoNativeA1000_08D17B74Pod1GPU2SharedCST`启动，GPU2已有1个co-resident，
本run固定CPU48--63并持GPU2 queue lock。它不是portable FullMDP A。

## 6. A1000 同进程里程碑

每个节点只读同一个日志/WAL/TensorBoard，不重启、不改 Reward、不换 seed：

| update | 要回答的问题 |
| ---: | --- |
| 5 | 4096 scene、显存、obs/Reward/optimizer/WAL是否finite且没有scale-only错误；通过后同进程继续 |
| 20 | optimizer、Reward20、WAL/ACK、finite、episode reason 和 opportunity 分母是否真实运转 |
| 50 | ready/swing dense income 与 termination rate 是否开始偏离随机初始化 |
| 100 | selected/defer/reject/accept 的比例与 episode length 是否出现方向性变化 |
| 200 | actor-pair contact、flight、outcome、recovery 是否出现非零且可归因的 numerator |
| 500 | 模仿收入、机会率、终止率是否稳定改善而非短暂噪声 |
| 1000 | 才裁决“有早期学习趋势 / Reward经济失衡 / 环境不可学 / 仍未测” |

每个节点至少记录：completed updates/steps、mean episode length、termination reason counts、20个 Reward term
的 signed income、opportunity transactions/due/selected/accept/censor/reject/defer、contact/flight/outcome/
recovery 的 eligible denominator 和 numerator、按 `stroke_family`/side 分组的结果、policy std/LR；RSL3
当前没有落KL producer，因此KL固定写`not_produced`而不是伪造数值，
非有限数与 wall time。零分母写 `未测`，不能写 0% 或塞进平均数。

Reward 比例只在 update1000 后按因果证据改：先判断是触发率、权重还是分母问题；A/C 除 observation 和
post-contact family Reward 外保持相同。历史 C 长跑曾看到 action penalty 负正比约
`10.39 -> 3.45`、episode length 谷底后恢复；这只作为需要观察的模式，不直接抄权重。

fresh `fcfe9918…` A1000 的update20/50均已从完整ACK读取。累计actual Reward sample=`960/2400`，
nonfinite=`0/0`、conservation violation=`0/0`；mean episode length约`87.6/86.4`。D05 selected=
`12/28`，其中not-ready defer=`11/25`、reject=`1/3`、ACCEPT=`0/0`。链路可信但task producer仍未ready，
所以继续100/200，不把0 ACCEPT归咎于Reward比例。

fresh update100累计4800个actual Reward sample全finite、0 conservation violation，mean episode length=
`85.36`；58次selected为54 not-ready defer、4 reject、0 ACCEPT。它与旧run到update100的已记录数值
精确一致，说明LM row-reject修复没有改变此前正常前缀；继续200。

native A1000 的update20/50从完整JSONL读取：累计episode length约`145.45/145.56`，节点min racket-ball
distance约`0.454/0.448 m`，binary contact=`0/0`，action-rate income约`-3.65e-5/-3.78e-5`，累计
负/正Reward比约`0.00270/0.00286`；capacity overflow和nonfinite均为0。当前未见负惩罚压垮正收入，
更像接触/可达性尚未学到；按预注册继续100/200，不因0 contact早停。

native的50--100窗口首次出现真实binary racket-ball contact：`6/8452=0.071%`；100--200窗口为
`32/17025=0.188%`，约2.65倍。同期mean min distance仍约`0.461/0.462 m`，episode length约
`145.22/144.47`，所以只称“稀少接触率上升”，不称整体命中。action-rate income均值约
`-3.96e-5/-4.34e-5`，负/正Reward比约`0.00307/0.00326`，不是当前主导惩罚。继续500。

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

只读因果审计进一步区分了“当前还没ready”和“ready后会坏”两件事。现役HEAD截至ACK437仍是
R07 first-ready=`0`、242次admitted均not-ready，因此sticky overflow尚未发生；但同一冻结源码在
bootstrap两拍首次ready时会把neutral/empty shot key送入只接受current full key的Epoch writer，随后
sticky overflow并使下一次D05 CENSOR或drain fail。该问题不能靠Reward比例解决。下一条fresh源码只把
`reference_kind=bootstrap`的first-ready从Epoch telemetry中mask，仍保留R07 owner-private dwell、
source step和Motion next-tick ready；completed shot和错误key继续fail closed。该A1000在ACK470后被
异步PhysX CUDA device assert截断；所以它只保留为0--469的负证据，不能再产生500/1000里程碑，也不能
用于裁决Reward权重。bootstrap修复仍只进入下一条fresh源码。

## 7. 架构减法

当前复杂度不是乒乓任务本身需要，而是历史上叠加了 RSL2/RSL3、legacy/full-MDP、多个 owner、receipt、
journal、checkpoint schema 和重复证明 gate。核心闭包约 14.9 万行，热路径约 11.8 万行；同类参考仓库
的共同训练闭包约 1.1k--1.6k 行。这个差距说明当前实现有真实技术债，不说明任务天然需要 100 倍代码。

2026-08-18重新计数：整个production Python树约305,720行；`hope_commands/commands/landing-device/旧runner/
physical-device/runtime`六个单体分别约32.6k/20.1k/16.3k/14.6k/13.3k/11.3k，合计超过108k。
`commands`与`hope_commands`还由同一`mdp/__init__.py`双 wildcard导入。这是历史兼容与并行语义堆叠，
不是物理任务固有复杂度；它也解释了为何每次改动都要跨多份一致性证明。

减法顺序必须由真实运行证明保护：

1. FullMDP 已绕过 14.6k 行旧 runner，只保留 397 行 RSL3 boundary adapter；
2. A `2×2` 与 A1000 通过后，删除旧 runner 内 FullMDP/R10/resume 分支及其重复 validator；
3. 把五角色 structural carry 与没有生产 restore callpoint的 checkpoint ground-work隔离或删除；
4. MuJoCo 不迁移73文件/约12.4万行未跟踪WIP；从tracked plant的真实reset/step callpoint只实现当前
   FullMDP 229/399所需producer，不再新建平行root/receipt/schema；
5. 每次删除必须保持真实 writer/consumer、live telemetry和反例，不用 self-SHA 代签。

2.0瘦身不在活跃长跑中热改。两边各有长跑后，先生成production callpoint census，再按依赖从外向内删：
零引用observation、旧RSL2 runner的FullMDP/R10分支、无生产restore callpoint的carry/checkpoint groundwork、
重复receipt/registry/journal。目标是一个训练root、一个typed event log、一个packed backend boundary；
任何新增“安全层”若不能指出独立事实源、真实消费者和失败动作，默认删除而不是继续堆叠。

## 8. 地形修正（代码现在修，课程用fresh run）

旧10 cm网格逐顶点独立`±2 cm`高度是白噪声地面，不是可解释的真实地形。producer现已改成固定seed的
空间相关场：四次box smoothing、出生圆岛exact flat、桌侧exact flat、两处smoothstep过渡，并用固定
IID反例证明相邻格相关而非白噪声；plant identity改为`robot_side_correlated_spawn_flat_v2`，禁止旧rough
checkpoint静默resume。host性质测试`13 passed`，training-contract回归`145 passed`。

当前FullMDP nominal配置本来就是plane；上述修改只有显式`terrain_rough_height_range`时才生效，所以
不会解释或改变旧N=2失败。下一条4096 nominal长跑仍用plane；rough必须用新namespace、新plant identity，
按`plane -> ±5 mm -> ±10 mm -> ±20 mm`独立阶段启用。启用前还要用live A3双脚包络验证0.20 m出生平地
半径，并做Isaac 2-env足底穿透/桌体对齐与4096吞吐；不把共享clone mesh伪装成逐env动态curriculum。

## 9. 存储边界

“超过 500 GB”是 Pod `/workspace` 配额，不是本机 repo；本机 `nohope` 约 1.56 GiB。本机已按
quarantine→`git fsck` 回收约 470 MB。2026-08-18 Pod1切回执行后，只删除6,902个可重建且无live-ref的
`__pycache__/.pytest_cache`与旧`fullmdp-verify2.pdvCeV` scratch；观测free从约16.53GB升到24GB级。
外部资产、foreign run、checkpoint、canonical logs和单副本证据仍不能按大小直接删；每次fresh run
前重验磁盘增长。fix4结束后曾为`22,794,698,752 B`；外部并发清理后末次只读值为
`31,737,970,688 B`，已满足本次A1000的20GiB启动门和增长余量。审计仍无法证明额外3GiB可安全删除，
所以不把`codexschema`、asset、日志或有live ref的目录包装成清理候选。
