# ActionBall 单动作双后端：唯一执行 TODO

> **当前执行已迁移：** 本页保留为2026-08-17到2026-08-19的详细历史账。最新依赖、阻塞、
> A200证据与`4096 × 25000`完成条件只看
> [`action_ball_dual_backend_longrun_todo_20260819.md`](action_ball_dual_backend_longrun_todo_20260819.md)。
> 不再在本页追加命令流水。

> 状态：`ACTIVE / branch-scoped / diagnostic_unauthorized`
> 人类负责人：Franco
> 执行者：Codex
> 更新：2026-08-19
> `origin/main:docs/NOW.md` 仍是项目优先级唯一权威；本页只维护这条分支的依赖、证据和下一条命令，不建立影子队列。

## 1. 目标

在 `shot_slot_capacity=1`、同一 `action_slot=0` 下，先让 Isaac family A 以 `4096 env` 真实运行，
在同一进程跑到25000 个 PPO update；1000只是早期趋势节点，不停机。随后按相同环境、seed 和训练设置运行 family C。
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
- 现役单动作family A/C只走
  [`FullMDP Phase-A direct-lean runtime`](../DEFINITIONS.md#fullmdp-phase-a-direct-lean)：exact lean owner
  是唯一可构造top owner；formal runtime选择、partial installer和逐leaf getter已从现役路径退休。
  `action_ball_full_mdp_runtime_owner.py`及其专属适配面已由Phase-B branch candidate物理退役，不再提供
  安全、执行或checkpoint权威；详细证据见
  [FullMDP hot-path实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md)。保留的是真实独立边界：source/API pin、lease/seal、component identity、reset
  generation/overflow、finite/sticky poison，以及仍禁止exact resume的FullMDP `R10 HOLD`。同一writer
  生成的依赖有向无环图（DAG）/hash/receipt不能自证安全；这项结构减法也不证明`6 s/update`或任何
  Gate Done。
- 第一条可信学习 run 使用 Git exact commit 和 `4096 env × 25000 update`；前5个update只是同一进程的
  construction/capacity/finite观察窗，通过后自然继续，不另起或重跑smoke。
- FullMDP Motion只消费与runtime 0807 A3P plant同源重解的73条measured bank；frame-0 grounding检查
  articulation root `pelvis_link`，不再把合法挥拍腰转造成的`torso_Link` yaw误报成未ground。该bank仍为
  `diagnostic_unauthorized`且机械admission=`0/73`，这项身份修正不授权formal/deployment。
- A25000 在 5/20/50/100/200/500/1000/2500/5000/10000/25000 只读里程碑，不为里程碑重启，也不因“看起来不好”自动停止。
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
| 环境数 | 现役4096×25000；旧证据N=2 | N=2 learn(1) | N=1024×1000 | portable仍未对齐 |
| actor/critic/action | 229/399/31 | 229/399/31 | 114/114/31 | native未对齐 |
| RSL/PPO | RSL3.1.2 thin adapter | RSL3.1.2 upstream | native trainer | 只对齐前两列训练ABI |
| reset/plant | FullMDP generation/selected reset | WAIT selected reset已过 | native episode reset | lifecycle未对齐 |
| question/reveal/launch | 完整producer | host纵切片已逐行reveal并真写launch state | 简化serve | portable仅窄切片 |
| Reward | Reward20 | host已消费R03项0--9与6项dense；10--13仍零 | 10-term简化Reward | 未对齐 |
| contact/outcome/recovery | R03/physical/R06/R07 | host R03 + live generic racket contact；selected-rubber/R06/R07缺失 | 简化binary contact | 未对齐 |
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
| 6 | `PASS-engineering-N2 / FAIL-scientific-long` | `N=2 × 2`工程门闭合；随后`N=2` A1000只到ACK470并因LM device assert停滞。可信前缀全部finite，但296个episode全tilt、260个admitted opportunity全not-ready、ACCEPT=0；环境数和任务入口都不足以形成学习结论 | 不再重复小N smoke；现役同一进程已直接进入`4096 × 25000` |
| 7 | `PASS-host-chain / HOLD-fresh-live` | bootstrap两拍ready仍授权Motion，但不再以neutral key写R07 ActionEpoch telemetry；CPU真实fact→owner projection→Motion reveal→production D05 settle已得到两行ACCEPT、Epoch无overflow。contact/flight/R06 outcome/R07 recovery仍需下一条fresh live分母 | 在4096长跑同一进程观察首个ACCEPT与真实分母；不再为它单开N=2 run |
| 8 | `HOLD` | portable restore 缺 Motion/Racket/Physical/R03/R06/R07、plant/manager/action history、trainer/optimizer/RNG和pre-gym reader | 不声称 resume |
| 9 | `PASS-live-step / PASS-live-SAT` | commit `61887b43…` 的fresh Pod1 GPU0共卡门真实完成`19 passed,0 skipped`：N=1 reset/step、N=2 masked reset、float32 device SAT及attached `robot/*` authority全过。run结束仍只有原1个peer，queue lock自然释放 | 保持同一SAT producer；下一步接真实A lifecycle，不再扩identity gate |
| 10 | `PASS-cleanup / PASS-A1000-margin` | 外部清理后Pod1约`249.8 GiB` free；A1000 ACK417时run目录仅约6.6MB，预计到1000新增日志不足约10MB，即使终点单checkpoint也远低于空间余量。未碰foreign PID、checkpoint、主日志或资产 | 不再为本run清理；只读监控实际增长，不按表观du删除硬链接/资产 |
| 11 | `PASS-live-WAIT-learn1 / HOLD-full-A` | 同一fresh checkout在19-test后直接调用upstream RSL-RL3.1.2：`N=2 × 24`完成1次PPO update、48 transitions、229/399宽度，`final_rc=0`；result SHA=`322592ce…f07a`。lifecycle仍明确`idle_wait_only`，没有question/contact/outcome | 复用该VecEnv/runner接真实A reveal→flight→outcome；先短live再启动MuJoCo A长跑 |
| 12 | `PASS-live-native-A1000 / HOLD-portable-A` | native A以`1024 env × 1000`自然完成：后500-update窗口binary racket-ball contact=`4078/87546=4.658%`，明显高于50--100窗口`6/8452=0.071%`；但robot-table episode fraction仍约97%，ABI为114/114、10-term简化Reward，缺WAIT/ActionEpoch/outcome/recovery | 只作下一代吞吐与Reward经济参考；不可写成portable 229/399 FullMDP A成功 |
| 13 | `PASS-Pod-CUDA` | LM code 8/9 已统一为`lm_solve_info_nonzero/lm_solve_nonfinite`；`dq`非有限、`solve_ex info!=0`以及有限`q+dq`溢出都在任何物理forward前逐行拒绝，正常peer保持不变。exact Pod1 Git `2c8ef444…`、Jiayi Python3.11/Torch2.7-cu128在GPU0得到三参数`3 passed`，每次后续kernel与synchronize正常 | 进入唯一4096同进程；真实异常仍需写row/iteration/try/info以继续收窄 |
| 14 | `PASS-structural / HOLD-live-business` | RSL3薄adapter升级独立v11 telemetry：compact joint-safety在optimizer前prepare/validate，随后`PENDING fsync -> Epoch ACK -> EPOCH_ACK fsync -> durable latch -> safety ACK`；4096×5单元反例得到5组pair和5份post-ACK receipt | 4096同进程前5次必须独立看到5份v11 safety receipt，并要求真实D05 producer/counter被调用；单元fixture的D05全零不能代签 |
| 15 | `PASS-live-generic-contact / PASS-host-R03 / HOLD-portable-A` | portable MuJoCo的`full_a_slice_attempted`已有逐行reveal、真实ball launch、20-substep plant、live generic racket contact、bounded terminal与selected reset；receipt固定`full_a_complete=false`。当前host又把同一postphysics racket site FK发布为R03 fact，并让engine-neutral Reward20 kernel消费项0--9；MuJoCo focused=`28 passed,9 GPU skipped`。generic contact不冒充selected-rubber；73-action lineage/mount sign、selected-rubber、R06/R07及Reward10--13仍明确缺失 | 下一批先接共享73-action identity和selected-rubber authority，再接R06/R07；fresh GPU R03门通过前不发portable长跑 |
| 16 | `FAIL-pre-PPO / ROOT-CAUSE-FIXED-HOST` | first 4096 one-shot消费commit `5ee1ffa6…`、wrapper `022c13f5…`和fresh namespace；GPU preexec、sealed archive及真实Kit Python身份通过，但wrapper在`AppLauncher`前导入Torch/RSL，Kit startup后约0.34秒segfault。Hydra已解析4096，scene/PPO/WAL均零调用；这不是容量或Reward失败 | runtime identity改成两阶段：AppLauncher前只验不可变解释器/archive，AppLauncher成功后同一Kit进程才导入并核Torch/RSL，再进入`_run`；新commit/namespace/wrapper可直发同一4096长跑，不复用本次证据 |
| 17 | `FAIL-pre-App / ROOT-CAUSE-NARROWED` | second 4096 one-shot消费commit `d341931c…`、wrapper `60cfdeed…`和fresh namespace；preexec通过、v2 receipt为空，Kit仍在约2秒内segfault，scene/PPO/WAL零调用。pre-App状态机已证明Torch/RSL未提前导入，所以它不是唯一根因；两次失败共同剩下的异常入口是`python.sh -P -S -B -c runpy(...)` | successor回到成功N2使用的direct script入口，仅保留`-P/-B`和production内pre/post-App attestation；删除Kit Python的`-S/-c/runpy`，仍以fresh namespace直发4096长跑，不插smoke |
| 18 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | third 4096 one-shot消费commit `356f706b…`、wrapper `00e340b7…`和fresh namespace；normal `python.sh -P -B train.py` 已稳定进入AppLauncher并完成约10秒Kit启动，随后production post-App gate拒绝AppLauncher合法加载的Torch。preexec通过、v2 receipt/scene/PPO/WAL仍为零；GPU与锁自然释放 | 不新增门；把post-App门缩到真正必须未加载的RSL/TensorDict，App前仍拒绝Torch/RSL/TensorDict，并继续对App-owned Torch核exact版本与venv来源。新commit/namespace直发同一4096长跑，不插smoke |
| 19 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | fourth one-shot消费commit `d2cd7911…`、wrapper `41cd2d2a…`和fresh namespace；Torch范围修正生效，随后Kit Python在同一post-App门因未导出`F_SEAL_*`符号而拒绝，preexec通过但v2 receipt/scene/PPO/WAL仍为零，进程与GPU自然释放 | seal验证继续保留，但直接使用Linux uapi的`F_GET_SEALS=1034`与四个seal位；exact Pod父进程创建/封印memfd→Kit继承读取已PASS。不把Kit Python可选符号别名当训练ABI；下一条直接扩成同进程`4096×25000`，不是另跑smoke |
| 20 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | fifth one-shot消费commit `6b3078bc…`和fresh 25k namespace；真实RSL serializer contract、GPU preexec、AppLauncher约10秒启动均通过，随后`sys.path[0]`位置断言拒绝；preexec=1、runtime receipt/scene/PPO/WAL=0，GPU与锁自然释放 | AppLauncher合法前插不提供`rsl_rl`的extension path；改为root、四个package与四个真实leaf在执行前逐一核live resolver的loader/archive/prefix/origin精确指向fd18，执行后仍逐module/class核source；旧namespace不复用 |
| 21 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | sixth one-shot消费commit `00cc5425…`、wrapper `edb7fec4…`和fresh 25k namespace；GPU preexec与AppLauncher通过，但post-App把Isaac Sim 5.1从`omni.isaac.ml_archive`加载的Torch 2.7/cu128误拒为“escaped frozen venv”。runtime receipt/scene/PPO/WAL均为0，result自然RC1 | 只允许两条exact Torch entrypoint：冻结venv，或由已验签Kit Python推导的Isaac Sim 5.1 ML bundle；live `torch.*`必须全在同一selected root，TensorDict仍只能来自venv。错误携带module与resolved path；旧namespace不复用 |
| 22 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | seventh one-shot消费commit `73d607f0…`、wrapper `324d36ad…`和fresh 25k namespace；exact Torch entrypoint已越过，但closure把注册在`sys.modules['torch.ops']`的动态`_Ops`对象当成file-backed module，其`__file__`访问触发operator namespace并产生伪路径。runtime receipt/scene/PPO/WAL仍为0，result自然RC1 | closure只检查`types.ModuleType`的file-backed Python/extension模块；dynamic namespace不参与origin claim。`torch.optim/_C` required modules与parent attribute identity硬门保持不变；旧namespace不复用 |
| 23 | `FAIL-post-App-gate / ROOT-CAUSE-EXACT` | eighth one-shot消费commit `5db486cd…`、wrapper `a1d97aae…`和fresh 25k namespace；`torch.ops`本身是`ModuleType`子类，因此上一版`isinstance`窄修仍把它当普通module并重现同一伪路径拒绝。runtime receipt/scene/PPO/WAL仍为0，result自然RC1 | 删除没人消费的blanket `torch.*` scan；只保留top-level exact origins、实际消费的`torch.optim/_C` origins、parent/sys.modules identity与PPO wiring。dynamic namespace不再被错误升级成训练规格；旧namespace不复用 |
| 24 | `FAIL-pre-scene-asset-gate / ROOT-CAUSE-EXACT` | ninth one-shot消费commit `f919ff1d…`、wrapper `d307dc29…`和fresh 25k namespace；GPU preexec、AppLauncher及真实Kit runtime v2 attestation全部通过，随后cfg构造把run-private资产快照路径误拒为“不是code-owned固定路径”。scene/PPO/WAL仍为0，result自然RC1。源目录与快照13个文件逐字相同，`model.usd` SHA均为`a3cd3829…8140` | 删除路径字符串相等这一错对象gate；无参production consumer改为对`HOPE_AGIBOT_A3_USD_PATH`实际指向的canonical、non-symlink目录运行同一个tracked producer，以代码内固定URDF/STL/asset-hash重建并验实际bytes。几何clone消费同一已验快照；旧namespace不复用 |
| 25 | `FAIL-pre-PPO / ROOT-CAUSE-EXACT` | commit `53156327…`首次完成4096 scene、229/399 observation、Reward20与真实Kit/RSL runtime attestation；约25分钟后runner构造发现live `joint_pos`没有通过新compact producer identity门。`preexec_passed=1`、`runtime_attested=1`，但PPO/WAL均为0，result自然RC1；GPU和锁已自然释放 | exact IsaacLab源码证明live class继承`ManagerTermBase(ABC)`，旧adapter却要求metaclass恰好为`type`，形成必错gate；compact仍由`train.py` runtime binding唯一写。successor只修为type实例+exact module/name+live object identity，并保留env0 source审计与逐env binding，直接fresh 4096×25000 |
| 26 | `FAIL-pre-scene / WRONG-OWNER-FIX-REJECTED` | commit `75b18ba2…`、wrapper `29986b0f…`、fresh namespace在runtime v2后、scene前自然RC1；task cfg预写compact被`train.py`唯一writer门精确拒绝。result SHA=`993031e5…ae2ed`，PPO/WAL为0、GPU/锁释放 | 封存且不重试；撤掉task duplicate writer。该失败反证“多处复制同一安全状态”会增加一致性成本，不把它算容量或学习证据 |
| 27 | `RUNNING-live-A4096x25000 / A100-ENGINEERING-PASS / BUSINESS-ZERO` | fresh commit `b64cb944…`在Pod1 GPU0、独立CPU32--47运行同一4096×25000进程。A100有100组完整v11 pair、9,830,400 transitions，Reward全部finite、0 nonfinite、0 conservation violation。0--49→50--99窗口每transition Reward=`0.057989→0.058106`，episode return=`5.076→5.251`、length=`88.02→90.29`；但累计106,645个episode全部base tilt，110,664次D05 selected=`97,328 not-ready defer + 13,336 reject`，ACCEPT/launch/R03/physical/R06/R07全0 | 不停机、不换seed、不热改；继续200/500/1000/.../25000。轻微dense趋势不能掩盖业务分母为零；下一代并行修ready producer而非继续加smoke |

## 5. 下一条发射协议

不再生成`N=2×2 -> 停 -> 修 -> N=2×2`的重复流水账。下一条Isaac A只允许一个fresh namespace：

1. exact clean Git、Isaac5.1/IsaacLab8320/RSL3、资产SHA、空物理GPU和独立CPU affinity一次性闭合；
2. `num_envs=4096`、`max_iterations=25000`直接启动，update0--4就是同一进程的scale/finite门；
3. update5健康即自然继续，不重启、不换seed、不改Reward；只读20/50/100/200/500/1000/2500/5000/10000/25000；
4. 仅在进程失败、证据不可信或结构性不可学被直接证明时停止；普通tilt/table/fall只记telemetry；
5. 长跑期间并行完成portable MuJoCo A lifecycle、rough课程和2.0删除清单，不热补正在运行的源码或scene。

第一份一次性4096 wrapper已经消费且自然RC1：commit=`5ee1ffa6…`、wrapper SHA256=`022c13f5…`、
namespace=`20260819T044500FullMdpA4096Iter1000Git5EE1FFA6Pod1GPU0CST`。preexec receipt和旧v1 runtime
receipt都落盘，但`run.log`只有identity marker和Kit segfault；无`Learning iteration`、无WAL、无GPU
compute驻留。根因是identity代码在`AppLauncher`之前导入Torch/RSL，违反训练入口原有“先启动Isaac Sim
再导入runtime模块”的顺序。该namespace不得复用、不得重试，也不能写成4096容量失败。

该阶段successor当时保持`4096×1000`、fresh namespace、GPU0/CPU32--47、最多一个已知peer、20 GiB余量和同进程
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

fourth one-shot证明上述范围修正已越过；新的首错是Isaac Sim Kit Python的`fcntl`模块有`fcntl()`，却没有
系统Python提供的`F_SEAL_*`符号别名。memfd本身及preexec seal验证未漂移，修复只使用Linux uapi固定值读取
同一seal bitmask。Franco同时澄清长跑目标：此前`1000`是PPO update（98,304,000 transitions），不是
1000个step；它只作早期趋势节点。下一条唯一进程改为`4096×25000`，即2,457,600,000 transitions，
update1000不停机；25k终点仍不等于formal promotion。

fifth one-shot（`6b3078bc…`）已越过serializer contract、preexec、AppLauncher约10秒启动，但post-App门把
“sealed archive必须仍为`sys.path[0]`”当成身份。Kit/AppLauncher会合法前插不提供`rsl_rl`的extension path，
所以在scene/PPO/WAL前RC1；preexec=1、runtime receipt=0。successor不放宽实际来源：对实际消费的root、
四个package与四个leaf逐一核live resolver的loader/archive/prefix/origin精确指向fd18，再import并核
module/class source与wiring。门只防确定性的依赖来源漂移；不把已经执行的可信Kit代码假设成会恶意并发改写
Python全局import状态的对手。旧namespace不复用。

sixth one-shot（`00cc5425…`，wrapper SHA256=`edb7fec4…`，namespace=
`20260819T074000FullMdpA4096Iter25000Git00CC5425Pod1GPU0CST`）自然RC1。它证明完整26-module sealed
RSL resolver已经越过，但同一post-App gate仍把AppLauncher合法选择的Isaac Sim 5.1 bundled Torch当成
foreign，因为旧条件只接受venv目录。Pod只读核验表明 bundled与venv入口SHA相同、版本均为
`2.7.0+cu128`，且bundled root固定在exact Kit tree。successor只增加这一条由已验签Kit解释器推导出的
allowlist并拒绝同一top-level下混入foreign `torch.*`或parent attribute alias；不放宽TensorDict/RSL，不复用旧namespace，也不把
这次零PPO失败记为4096容量证据。

seventh one-shot（`73d607f0…`，wrapper SHA256=`324d36ad…`，namespace=
`20260819T080000FullMdpA4096Iter25000Git73D607F0Pod1GPU0CST`）又在post-App、scene/PPO/WAL前自然RC1。
首错精确为`torch.ops=<run-root>/_ops.py`；`torch.ops`实际是Torch注册到`sys.modules`的动态`_Ops`对象，
不是`ModuleType`，它的`__file__`是动态operator namespace而非来源字段。successor只把closure范围收回到真实
file-backed modules；实际消费的`torch.optim`、`torch._C`及parent/sys.modules identity门不变。

eighth one-shot（`5db486cd…`，wrapper SHA256=`a1d97aae…`，namespace=
`20260819T081000FullMdpA4096Iter25000Git5DB486CDPod1GPU0CST`）证明`torch.ops`是`ModuleType`子类，
所以仅按Python类型做blanket closure仍会重复同一错误。按HANDOFF §3，successor删除这道未消费全包扫描，
只验RSL真实使用的top-level Torch/TensorDict、`torch.optim/_C`与parent/PPO identity；这是删错层gate，
不是放宽实际训练依赖。

ninth one-shot（`f919ff1d…`，wrapper SHA256=`d307dc29…`，namespace=
`20260819T081500FullMdpA4096Iter25000GitF919FF1DPod1GPU0CST`）首次完整落下
`trainer_runtime_attested_v2`，证明normal entry、AppLauncher、Torch/TensorDict/RSL实际消费identity均已越过。
新的首错发生在env cfg构造、scene之前：wrapper按设计把完整split-rubber目录复制到fresh run私有目录并在
launch边界重验inventory，consumer却只比较路径字符串是否等于旧共享目录。只读核验源目录和快照13个文件
逐字相同，`model.usd`均为`2018444 B`、SHA256=`a3cd382943ff9f70beecf88c729a6cc1c052a3c0a0cbffe91003ec319ab78140`。
tracked producer本来已经用代码内固定pins对实际`output_root`重建URDF/STL/asset hash与USD语义，所以固定
pathname不增加事实，只拒绝更强的私有快照。successor保留无参consumer和全部重建检查，仅把`output_root`
改成实际选中的canonical、non-symlink目录；live geometry clone也从同一目录读取。host asset/factory回归=
`77 passed,37 skipped`；fresh Pod仍待新commit、wrapper和namespace，旧run不重试。

25k不能只留下曲线。lean graph尚无完整plant/owner/RNG restore合同，所以仍禁止`load/resume`；但runner现在
每1000 update及自然终点调用upstream RSL save，把policy、optimizer、iteration和normalizer所属model state
写成`model_N.diagnostic_nonresumable.pt`。文件内同时写
`checkpoint_authority=false`与`resume_authority=false`，文件名也不冒充可恢复checkpoint。长跑final consumer
必须看到exact 26份（0、1000…24000、24999）regular non-empty snapshot及同数目的no-clobber
`*.receipt.json`；每份receipt绑定iteration、snapshot size/SHA、payload kind及两项authority=false，final consumer
逐份交叉验证后才发布稳定inventory SHA。receipt生成前还会从同一已打开fd以`weights_only=True`重新读取真实
Torch payload，要求`model_state_dict/optimizer_state_dict/iter/infos`四键、iteration一致、两类state非空且全部
tensor finite；prelaunch再用冻结RSL3的真实`OnPolicyRunner.save`做一次host serializer contract。这个检查不是
学习smoke，只保证训练产物未丢且确由diagnostic save边界签收，不能关闭TODO row8的完整restore缺口。

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

## 6. A25000 同进程里程碑

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
| 2500 | 检查早期趋势是否延续，而不是一次短窗口波动 |
| 5000 | 检查contact/outcome分母和Reward经济是否进入稳定区间 |
| 10000 | 检查策略是否继续改善或出现退化/遗忘 |
| 25000 | 长跑终局；裁决可学习性、Reward比例与下一代配置，不自动授权promotion |

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
checkpoint静默resume。host性质测试现在为`14 passed`，training-contract历史回归`145 passed`。

当前FullMDP nominal配置本来就是plane；上述修改只有显式`terrain_rough_height_range`时才生效，所以
不会解释或改变旧N=2失败。下一条4096 nominal长跑仍用plane；rough必须用新namespace、新plant identity，
按`plane -> ±5 mm -> ±10 mm -> ±20 mm`独立阶段启用。exact Pod1 CPU FK已对固定MJCF
`70c4fd65…`与ready pose `ab6b7e41…`计算四个踝部collision mesh：最大XY半径为
`0.470508504 m`。producer据此把出生exact-flat半径从错误的`0.20 m`改为`0.60 m`，外接到`0.80 m`
的smooth blend；这覆盖全部足部包络再加一个`0.10 m`地形cell guard。仍需fresh Isaac 2-env足底穿透/
桌体对齐和4096 rough吞吐；不把共享clone mesh伪装成逐env动态curriculum，也不热改现役plane run。

## 9. 存储边界

“超过 500 GB”是 Pod `/workspace` 配额，不是本机 repo；本机 `nohope` 约 1.56 GiB。本机已按
quarantine→`git fsck` 回收约 470 MB。2026-08-18 Pod1切回执行后，只删除6,902个可重建且无live-ref的
`__pycache__/.pytest_cache`与旧`fullmdp-verify2.pdvCeV` scratch；观测free从约16.53GB升到24GB级。
外部资产、foreign run、checkpoint、canonical logs和单副本证据仍不能按大小直接删；每次fresh run
前重验磁盘增长。fix4结束后曾为`22,794,698,752 B`；外部并发清理后末次只读值为
`31,737,970,688 B`，已满足本次A1000的20GiB启动门和增长余量。审计仍无法证明额外3GiB可安全删除，
所以不把`codexschema`、asset、日志或有live ref的目录包装成清理候选。
