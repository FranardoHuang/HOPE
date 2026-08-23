# EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822

> 问题：为什么两条 [FullMDP](../../DEFINITIONS.md#fullmdp-ppo-v3) H48 长跑已经学会站立，却没有进入可学习的 mimic/击球链；怎样用最少状态和单一真源修复后 fresh 重启？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`V4 evidence frozen / V5 branch candidate / no authorized formal run`
> 证据边界：`diagnostic_unauthorized=true`；本记录不授权 resume、promotion、export、部署或真机。

> **阅读边界（2026-08-23）：**§1--§8保留课程故障的发现与被替换实现史，不是现役执行真源；当前契约、
> V4最终证据和V5未闭合项只认§9。尤其是旧文中“hidden都用reset-ready”和“11项row fault”已被§9明确
> supersede，不能再拿来描述现役代码。

## 1. 旧 live 事实与裁决

- Isaac 到约 update 500 已从早期倾倒变为整段 1500-tick timeout，说明 balance 已学出；但 update 305
  起每个 due row 都是 CENSOR，task Reward/击球/落点分母保持零。这不是“学习还慢”，而是账与实际不一致。
- 最早 CENSOR 在 update 116，28 行恰是同一 control-step 的整批 due。按该历史旧 cadence，首个活过
  当时 second due tick `295` 的 row 触发 chronology bit 50，随后同批 fault fanout；这不是把现役
  first due `295` 称为第二拍。此前终止 row 都没活到该 tick，CENSOR 为零。
- 根因是 Motion 每个 due 都推进 ordinal，而 D05 旧 afterimage 只在 ACCEPT 时推进。第一次
  DEFER/REJECT 后两个 cursor 永久错一位，所以第二次 due 被错误归为 producer fault。
- MuJoCo hidden 阶段的 joint teacher 取 reset/default，而 14-body teacher 却取 action frame 0。两套 FK
  body/order/quaternion 审计均正确；冲突来自 target authority。reset-ready 到 frame 0 的 14-body orientation
  RMS 为 `50.353 deg`，旧 `std=.4 rad` 指数核初值仅约 `0.008`，梯度严重饱和。
- R07 的 13 项 all-of 是恢复质量诊断，不是 plant safety 独立事实。把它用于首次 task reveal 或 actor phase，
  等价于要求策略先完成尚未获得训练分母的动作，违反
  [`HANDOFF_TO_CODEX_20260808.md` §3](../../operations/HANDOFF_TO_CODEX_20260808.md)。

## 2. Adopt / defer / reject

采用：

- 单一自然事件课程，不新增热切 Stage：前 295 policy tick 只有 reset-ready balance；能存活到 tick 295 的
  row 获得首次 task/motion 暴露。mimic、击球和落点随后由同一 task 的 teacher、确定性击球几何和真实事件
  自然获得分母；不以未训练 policy 的 rollout 成败作下游启动门。
- `295` 只由 portable catalog 定义，Isaac cadence 消费同一常量；episode 内能完成的四个 due 为
  `295, 588, 881, 1174`。`1467`是第四拍的retirement boundary，不是第五个机会；从它再开一拍会越过
  1500-tick episode horizon。
  首次`295`是catalog课程常量；重复间隔`293`则由当前action timing上界
  `max close tick 214 + recovery window 77 + 2`派生。两者来源不同，不能把293反写成first-reveal真源。
- D05 的 cadence cursor 按所有 settled due 前进；target generation、selected cell 与现役
  task/outcome/ball identity 仍只按 ACCEPT 前进。checkpoint 合同允许“cadence 已消费但尚无 accepted task”，
  并统一要求 `outcome_shot_index = scheduled_ordinal + 1`。
- **HISTORICAL / SUPERSEDED by §9.2：**本轮当时把所有hidden概括为同一reset-ready tuple；现役合同已
  收窄为“首次ACCEPT前reset-ready、ACCEPT同tick selected measured frame0、recovery使用completed-action frame0”。
  旧证据仍表明人为35-step插值不是存活主因，因此不增加另一套blend状态机。
- body orientation 使用同一误差的 fine `.4 rad` 与 coarse `1.0 rad` 等权核；它只在 reference 修正后
  恢复大误差区梯度，不用放宽阈值掩盖错 target。
- R07 保留 recovery Reward、critic plant state和可读 telemetry，但退出 fresh reveal、训练 liveness 与
  actor phase authority。nonfinite/overflow、joint envelope、跌倒/过低、真实桌碰等 plant fail-stop不变。
- **HISTORICAL / SUPERSEDED by §7：**当时保留了H48、GAE lambda `.98`、E5/MB8、fresh
  `sigma=.02`、log-std、entropy `.01`、adaptive KL；后续真实checkpoint已证明永久entropy使std无界上升，
  当前V3改为`entropy=0`，仍不叠加std decay、clamp或LR改动。
- 不新增 actor observation。现有 203-D 已包含deployment-intended且因果可观测的root/velocity、teacher、
  task residual、phase与倒计时；真实table/root pose、heading和marker-to-COM velocity producer仍由G07的
  OptiTrack校准、同步、stale/dropout合同闭合，不能把sim truth写成已部署。R07误差重复喂给actor只会扩大
  合同并制造shortcut。

延后：

- `solver_solve_many`/`cq_n_iters` 只在当前 profiler 与 fixed-tape parity 后单独裁决；旧 profile 不代签新热路。
- critic 中既有 support/dwell 是否进一步删除，等 fresh 证据后作为独立 Observation ABI 变更处理，不与本轮
  actor/task unblock 混合。

拒绝：

- 不以 `ACCEPT>0`、R07 all-of、早期 hit/landing 成功率作启动或安全 Gate。
- 不靠 Reward 总权重、rollout 缩短、增加 env 数或新 receipt 掩盖零 denominator/错误 reference。
- 不增加 owner/gate 去证明同一 writer 的自洽；变更只保留能构造独立反例的 plant/identity边界。

## 3. 当前实现与验证

- portable catalog 首次 due 改为 295；Isaac 与 MuJoCo 都在 policy tick 295 曝光，R07 false 不再 defer。
- MuJoCo hidden/prepare/completed teacher 已同源；Isaac fresh canonical/legacy actor phase不读取R07授权。
- D05 due-settlement、carry invariant和 second-due bit50 回归已加入。
- Isaac one-shot launcher新增 `--profile-updates 0..50` 和显式 `--cpu-affinity`。前者只做有界归因且自动卸载；
  后者将训练 child 置于 GPU-local CPU set，不是性能 Gate。fresh namespace现在拥有自己的
  `training/` checkpoint、WAL与contract，不再写入 source checkout。
- host launcher/catalog为`18 passed`；exact Pod把11个相关文件逐个放进独立Python进程，结果为
  `292 passed, 36 skipped`。skip来自轻量MuJoCo venv没有IsaacLab/Hydra或专有资产的既有边界；真实Isaac
  构造另走下述launcher。`py_compile`与`git diff --check`通过。
- 首次GPU1 profiler-on real使用fresh namespace
  `fullmdp-a-h48-v2-isaac-unblock-019d9a6f-20260822`，在PPO update 0前fail-closed：profiler还绑定已经退役的
  `_assert_owner_binding_current`，而现役环境入口是`_assert_step_may_start`。该失败root冷弃、process/GPU/lock
  自然释放，不resume、不复用。profiler已改为只包装现役真实callpoint，host profiler+launcher=`29 passed`；
  同次失败还发现Hydra把三份metadata写到source checkout；launcher现将`hydra.run.dir`一并固定到fresh
  `training/hydra/`，旧文件移入失败root留证，不用`.gitignore`掩盖。下一fresh GPU实跑仍待复验。
- 修复后的Isaac successor `fullmdp-a-h48-v2-isaac-unblock-333f9490-20260822`已在GPU1启动并连续取得durable
  ACK；profiler 5轮后自动卸载，源码保持clean。旧Isaac在update631仍把全部618个due行CENSOR，随后按
  PID/startticks精确TERM，process/GPU app/lock均已释放，completion仍缺席并明确为operator stop。
- 第一次MuJoCo GPU0 successor跑到ACK7且Reward/storage finite，但底层生成的`MUJOCO_LOG.TXT`落入source cwd；
  该run按精确child PID终止并冷弃，日志移入失败root。launcher现把child cwd固定为fresh run root，避免用
  `.gitignore`逐个追赶未知fallback文件；新的fresh namespace仍待启动。

## 4. fresh 验收

1. exact Pod 先跑 D05/cadence/carry、Motion phase、reward、MuJoCo lifecycle、launcher 聚焦回归。
2. 独立 fresh namespace 做短验：连续 ACK、Reward/observation finite、fault 0；首次 due 可为
   DEFER/REJECT，但 second due 不得再出现 bit50 chronology CENSOR。
3. Isaac 短验在同GPU比较明确 CPU-local affinity，并把 profiler-on窗口与profiler-off成绩分开报告。
4. successor 可用后才按精确 PID/start-ticks停止旧run；旧root、checkpoint与日志只读封存，不伪造completion。
5. 两条正式 fresh run从零开始，逐项报告 balance→mimic→strike→landing 的 numerator/denominator；零分母写
   `未测`，不做平均稀释。

## 5. 2026-08-22 22:50 UTC fresh运行快照

- Isaac使用commit `333f9490ed388025fd1a49e1a978a6f5107c156a`、namespace
  `fullmdp-a-h48-v2-isaac-unblock-333f9490-20260822`在GPU1运行，launcher/child为
  `2213515/2213532`，CPU affinity为`48-63,112-127`。已连续取得durable ACK至update52；每轮
  `196,608/196,608` actual Reward finite、nonfinite为0。最近可结算episode均为base tilt，平均长度约
  `91--93 tick`，尚未存活到首次due 295，因此task reveal、strike、landing分母都为0并记`未测`；这仍是
  初始balance阶段，不是CENSOR复发。
- Isaac前5轮有界profile按约定自动卸载：collection中位约`14.616 s`，其中
  `post_physics_publish` inclusive中位约`9.206 s`，当前主要墙在post-physics owner数据流，不是solver。
  profiler-off最近20轮wall中位约`16.92 s/H48`，约`11,620 transitions/s`，H24-equivalent约`8.46 s`；
  尚未达到约`12 s/H48`（旧尺度`6 s/H24`）的量级目标，性能工作继续，但不应停车或改课程。
- portable MuJoCo使用commit `23c0f6c8923a6f602c9b2f40fb926d5bb47f0ee0`、namespace
  `fullmdp-a-h48-v2-mujoco-unblock-23c0f6c8-20260822`在GPU0运行，launcher/child为
  `2219700/2219718`，child cwd已钉到run root。已连续取得ACK至update18；Reward/storage全finite且
  conservation fault为0。最近10轮wall中位约`9.36 s/H48`，约`21,005 transitions/s`，H24-equivalent
  约`4.68 s`。episode平均约`105--112 tick`，仍未到due 295，所以下游三阶段同样为`未测`。
- reference修复已在活数据可见：Isaac新run的`motion_body_ori`单轮configured income约`643--800`，旧run
  update631只有约`79`；MuJoCo最近窗口约`2,526--3,831`。这证明大误差区不再被错误target/单一窄核压到
  近零，但不是mimic/击球成功声明；当前teacher仍是hidden reset-ready。
- 被替换的旧Isaac/MuJoCo分别精确停在ACK631/3023。两者的PID、目标GPU compute app与lock均absent/free，
  completion文件都缺席；admin-stop pre/post证据独立封存，旧root和snapshot未删除、未resume。

### 22:55 UTC阶段首穿补充

- Isaac early update0--9与recent update60--69的完成episode平均长度从`87.65`升到`94.93 tick`；recent
  仍为`due=0`，`motion_body_ori`均值从约`692.4`升到`789.6`。最近20轮wall中位约`15.775 s/H48`
  （H24-equivalent约`7.89 s`）。这是balance早期改善信号，不足以声明balance或mimic成功；recent终止中
  table-hit已占约`11.0%`，必须继续按原因与分母观察。
- MuJoCo early update0--9与recent update38--47的完成episode平均长度从`105.61`升到`149.75 tick`。
  update45首次真实出现`due=1 / reveal=1 / deferred=0`，随后同一rollout有phase-2 task row `27`个、
  `R03 physically valid=26`且task Reward0--9已产生非零梯度；这直接验证上一阶段有一个row活到due时，
  下一阶段立即开始可学，不再被旧R07门清零。selected contact与landing仍为0，所以击球和上台继续记
  `未测`。最近20轮wall中位约`9.634 s/H48`（H24-equivalent约`4.82 s`）。

## 6. 2026-08-22第二次successor：CUDA故障、几何闭环与热路减法

### 6.1 旧Isaac不是“继续等就会学”

现役Isaac在durable ACK257后触发匿名CUDA device-side assert；因为旧`torch._assert_async`先毒化context、
仍继续执行，异常直到下一次PhysX write才显现，进程随后约2小时45分无新ACK。最后完整ACK、日志SHA、
PID/start-ticks与run root已经写入旧namespace下独立`administrative_failed_stop_20260822T0252Z`；只对精确
child先TERM、两次无效后KILL，leader与GPU1 compute app均确认absent。该run不resume、不复用namespace。

最接近故障因果的active-path是R06 settlement与retire矛盾处的异步assert：旧代码即使条件为false仍继续
修改R06，符合“在后一PhysX调用才显错”的时序。本候选先mask无效行，Physical把
`accept_not_launchable / due_identity_lost / r06_retire_mismatch`三类跨owner矛盾保留在device-local bit，
并复用既有每control一次host reduction具名失败；删除scene对Physical私有projection和rubber echo的
same-writer异步assert。finite、key/generation、contact、joint、fall/table等真实独立安全源不变。

### 6.2 上一阶段完成时下一阶段现在有确定性入口

对slot0做构造期反例后发现两个真实问题。portable MuJoCo把profile contact offset与未retarget的measured
teacher独立拼接；完美mimic在strike时刻与球心约差`13 cm`。共同Isaac D05又把
`continuous_questions.p_contact`（源码合同明确为球心arrival）初始化为selected face centre，少一个球半径。
两处都不是需要新Observation的问题，而是题目本身不闭合。

修复后两端都从measured official racket site、reference root displacement、command quaternion和共享
`ball_center_from_site_local(sign)`唯一构造球心；base goal同时减去同一ball-centre reach，不再由第二份
profile offset制造冲突。确定性测试逐项证明：`ball_center = site + R*local_offset`、球心到selected face
法向距离精确为球半径、切向残差为0、`pre_wait + scaled_t_hit = ttc`。这意味着balance row活到295后立即
有mimic分母，而完美mimic在strike tick几何上已经是一枚selected-face接触；真实引擎contact仍必须由fresh
run实测，几何证明不冒充物理成功率。

### 6.3 共用q_des与postphysics减法

- 新的`action_ball_qdes_guard`只拥有纯tensor变换，不拥有backend、receipt、counter或Observation。Isaac和
  portable MuJoCo都调用同一函数，覆盖finite fallback、soft/hard inset、20 ms q/qdot brake与terminal bit；
  alignment不再靠声明字符串，而以AST反例验证两边唯一call和同一参数。
- Physical→Epoch/R06在同一Python调用栈内同步消费的postphysics packet不再先复制一份完整`[N,K,*]`
  snapshot，再由Epoch复制一次输入。ActionEpoch只读借用在operation返回前有效；legacy长租约/abort路径仍
  保留独立snapshot。跨owner identity/fault join仍执行，删除的只是同writer、无并发mutation窗口的副本。
- 当前host分进程结果：question/D05=`71 passed,2 skipped`，Physical/Epoch/R06=`112 passed,7 skipped`，
  MuJoCo action/transition/runner=`74 passed,7 skipped`，shared q_des/Isaac joint safety=`122 passed`；完整
  exact Pod、fixed parity与profiler-off matched H48仍待执行，因此本节不给速度GO。

取舍保持：rollout=`48`、lambda=`.98`，不缩rollout冒充提速；不新增actor observation；resume、CUDA
Graph、solver iteration、physics dt与C++均延后。第二次fresh只有在exact Pod通过后才替换仍只读运行的
MuJoCo旧run。

### 6.4 空业务热路与显式运行输入候选

本节记录`661ff84b`形成前后的历史候选；其中“no-key仍每control读取plant/readiness并安装Motion”的描述
已被§6.6的`aa42418b`实测裁决取代，不是当前合同。

`dd82bb7b`的fresh Isaac前五轮profile仍显示无shot时`post_physics_publish`约`9 s/update`；同一窗口中
Physical已经走idle，但R03和R07每control仍为不存在的key生成空ActionEpoch写入。候选不新增stage、gate、
owner或第二次D2H：Physical现有packed host summary在同一`.item()`中区分`transport_work`与
`keyed_epoch_work`。R03只消费前者；R07 keyed facts只消费后者；plant read、balance/readiness私有状态、
Motion projection与install每control继续执行。球已retire而recovery仍active时是
`transport=false/keyed=true`，因此会跳R03但保留完整R07。全空路径还在构造keyed grid与`[N,S,32]`
values前退出；现有Epoch snapshot clone暂时保留，避免为未实测的小收益暴露live tensor alias。该路径不生成
receipt，也不把idle当安全判决。

首次MuJoCo `dd82bb7b`启动在PPO前自然RC1，原因不是physics或EPA，而是one-shot launcher仍隐式依赖外层
shell预设`ACTIONBALL_READY_POSE`；其dry-run没有暴露这个输入。候选把ready pose改为required
`--ready-pose`，在创建run root前核canonical regular file与固定SHA，并显式写入child env。失败namespace
保留、不复用；新的Pod测试、性能数值和fresh ACK尚未产生，故本节仍是branch candidate，不能声称提速成功。

### 6.5 `661ff84b`双fresh实测与下一性能问题

exact Pod回归先在`8aa3…`分进程得到`546 passed,32 skipped`；旧N2 landing fixture另有2个与本变更无关的
`replicate_physics`构造缺口。随后第一次fresh Isaac在ACK97后的首个active-path暴露
`Physical.flight_slot`来自`expand`的noncontiguous tensor；该run自然失败且namespace保留。最终
`661ff84b…`把固定grid在构造时一次性`.contiguous()`，重跑直接覆盖的Physical/Epoch为
`92 passed,7 skipped`，没有每control复制。

`661ff84b…`的portable MuJoCo已连续取得ACK0以上；ACK1--28 collection约`9.11--9.61 s/H48`，Reward/storage
全finite、conservation fault=0。early episode均长从约67 tick升至111 tick；`joint_qdes_forbidden`终止在
update22后已降为0，因此它是balance早期可学习过渡，不是持续的q_des阻塞。尚无due/reveal/contact/landing，
后三阶段按零分母记`未测`。

同commit的Isaac已连续取得ACK0--10；fault/CENSOR/nonfinite为0，Epoch commit由旧约`1104--1152`降为
`816--824/update`。但前5轮profile自动卸载后的ACK5--10仍约`14.94--15.91 s/H48`；profile内
`post_physics_publish`为`6.53--9.12 s/update`，而`sim_step`仅`1.75--1.84 s/update`。所以当前实测支持
“空R03/keyed写减法正确且有局部收益”，不支持“已达约12秒”。下一诊断只在已有bounded profiler增加
Physical、R07 plant/reference/reward/readiness和Motion install的嵌套host wall，不增加CUDA同步、gate、owner、
receipt或actor observation；测完5轮自动卸载。只有测得主墙后才做下一结构刀。

第一次嵌套profile commit `8cbccad8…`在runner构造时自然RC1，尚未进入PPO：R07 bundle是frozen dataclass，
实例方法不可替换；即使强行替换，也会破坏LeanRuntime要求的class-bound exact method identity。失败namespace
只读封存，GPU/lock自然释放。successor保留原方法对象不动：profiler只在LeanRuntime实例安装一个非authority
host-clock callback，LeanRuntime仍先用现有`_bound_plain_method`认证真实Physical/R07/Motion方法，再把该
bound method交给callback计时。自动卸载时callback一并删除；未启profile时生产调用序列不变。

### 6.6 `aa42418b`：no-key contact read删除、性能闭合与双fresh切换

`2b590889…`的细分profile把旧Isaac主墙收窄到一个具体独立事实读取：no-key阶段每个H48 update中，
`ContactSensor.data.net_forces_w`的两脚support读取吸收约`7.41--8.03 s` host wall；ActionEpoch chronology
snapshot只有约`2 ms`，coordinator store约`5 ms`。这两个support bit和ready dwell只在critic
`[216:219]`，actor 203-D从未看到；在没有admitted shot key时也没有recovery事件，dwell按定义为0。

因此本轮采用最小语义减法：no-key路径只观察独立source step、ActionEpoch reset generation和Motion
cadence chronology，不触碰ContactSensor；critic宽度仍保留219，`[216:219]`按N/A填零。keyed R07路径不变，
仍读取真实contact并计算foot slip/support deficit、recovery reward和readiness。延后把critic物理缩到216，
因为那会同时破坏checkpoint/optimizer和双后端ABI；拒绝全局关闭contact sensor。这里删除的不是安全事实，
而是在没有shot/recovery业务事件时为三个critic bit重读整份contact buffer。finite、joint/table/contact、
key/generation和optimizer后durable ACK边界全部保留。

虽然shape不变，idle真实foot bits变成N/A zero已经改变critic数值合同，所以只能fresh lineage，不能称为旧
checkpoint exact resume。`postphysics_valid=true`只说明该control step的neutral chronology有效，不是
"已测得双脚无支撑"。同批只删除R07 bundle对已绑定ActionEpoch `snapshot_idle_*` getter的重复方法身份复核；
LeanRuntime的production component callpoint binding与callee R07 owner核验仍保留，Observation继续以独立
Motion/ActionEpoch chronology反例检查consumer事实。测试也删除了把Motion的true输入
手造回D05 ACCEPT的self-proof fixture，保留catalog-owned first reveal tick 295与cadence 293的独立seam。

exact Pod在clean detached source `aa42418b187e8f3edf49d5757868fe0215e62d42`按文件隔离得到
`304 passed, 6 skipped`；其中sentinel令任意no-key ContactSensor `.data`访问直接失败，且keyed recovery、
Motion bridge、Physical hot lane、env install与D05 row-wise transaction均通过。H48 diagnostic namespace
（一次性、[仅全新训练且禁止续跑](../../DEFINITIONS.md#fresh-only-no-resume)的性能归因运行身份）
`fullmdp-a-h48-v2-isaac-idle-zero-aa42418b-profile-20260822`的5个profile-on collection为
`6.086/4.910/5.898/6.255/5.949 s`，`r07_idle_support_read=0 calls`、idle stamp仅
`5.8--7.3 ms/update`；profiler自动卸载后的匹配5轮为
`6.346/5.999/6.400/5.892/6.408 s`，中位`6.346 s/H48`。前9个ACK的fault/CENSOR/nonfinite/
conservation均为0。相对上一同卡profiler-off中位`14.194 s/H48`下降约55%；这是真实数据流删除，不是缩H48、
改PPO或加一层gate。diagnostic随后按PID/start-ticks精确停止，GPU2与lock均free。

双fresh切换随后完成：

- Isaac namespace `fullmdp-a-h48-v2-isaac-idle-zero-aa42418b-20260822`在物理GPU2运行；快照到ACK67时
  source clean、Reward全finite且fault/CENSOR/nonfinite/conservation为0，recent20 H48 collection中位
  `6.758 s`。first10→recent10 episode均长约`87.73→93.57 tick`；reset-ready anchor/body orientation
  imitation reward均有上升，但所有episode仍以fall/table结束，尚不能写balance或action mimic成功。due=0，
  所以reveal/launch/contact/landing均为`未测`。
- portable MuJoCo前两个namespace在首个ACK前分别因新checkout遗漏EPA48 build receipt、遗漏RSL3 wheel
  自然失败并封存；均未进入trainer/PPO、训练业务路径或首个ACK，namespace未复用。按三文件manifest恢复并
  重核固定SHA后，r3 namespace
  `fullmdp-a-h48-v2-mujoco-idle-zero-aa42418b-r3-20260822`在物理GPU1运行。update0--19 H48 collection中位
  `9.391 s`，storage/reward finite且lifecycle/conservation fault为0。早期qdes forbidden在update25降到3、
  update26起为0，复现了旧lineage的短暂balance过渡；随后主要终止转为robot-table contact，episode均长
  到update36已约`122.3 tick`。这仍是policy可学习的balance行为，不是应绕过的安全门，也不是成功。
  due/reveal/launch/contact/landing仍为0分母，统一记`未测`。

新Isaac连续ACK后，旧Isaac `661ff84b…`才按精确child identity停止；新MuJoCo连续ACK后，旧MuJoCo才停止。
最终只保留两条`aa42418b` fresh lineage运行。两条都仍为`diagnostic_unauthorized=true`，没有resume、promotion、
export、physics parity或部署授权。该切换快照的阶段判断是“早期balance + hidden reset-ready imitation信号，
尚未到task/action mimic reveal”；符合balance→action mimic→hit→landing顺序，但还没有证据验收
action mimic→hit或hit→landing交接。

后续同进程自然推进已经给出第一次课程交接反例：Isaac update106出现
`due/selected/ACCEPT=1/1/1`且defer/reject/CENSOR/fault全0；MuJoCo update49--59累计
`due/reveal=9/9`、deferred=0，每个有task的update都产生非零且相等的R03 present/physically-valid rows。
这证明上一阶段只要有row活过tick295，task/action-mimic分母立即打开，没有被R07 readiness再次阻断。Isaac尚未
开始playback，两个backend的launch/contact/landing仍为0，所以击球/上台继续`未测`，不能用R03几何有效
冒充真实接触。

只读刷新到Isaac ACK176 / MuJoCo update118时，累计分母分别为
`due/selected/construction/key=2/2/2/2`和`due/reveal=15/15`、deferred=0、R03
present/physically-valid=`174/174`。Isaac recent10 collection中位`8.238 s/H48`、episode均长
`148.93 tick`；MuJoCo对应为`9.585 s/H48`、`151.17 tick`。两端fault/nonfinite/conservation仍为0，
但Isaac playback与两端launch/contact/landing仍为0。

### 6.7 side-session建议的独立裁决

- **采用且已完成：**保留既有100-step fused RK4，不重复实现；现役Isaac日志在task construction出现后仍无
  eager-fallback warning。共享engine-neutral q-des guard也已在本source中由两端消费，alignment ledger为
  `ALIGNED`，不是本轮待办。
- **采用更窄实现：**没有先造一个新的巨型`ActionBallState`或C++层，而是从实测主墙删除no-key
  ContactSensor读取和同writer重复自证；同卡H48匹配中位从`14.194 s`降到`6.346 s`。这是状态/数据流减法，
  不是用gate包住旧结构。
- **延后：**219→216的critic ABI收缩、完整single-state合并、optimizer-boundary numerical resume、
  CUDA Graph/Warp capture。它们不解决本轮已定位的首墙，并会扩大checkpoint/optimizer/双后端语义面；后续
  只按新profile或明确恢复需求独立实现。
- **拒绝：**C++优先重写、缩H48、减少MuJoCo 20个contact substep、热补现役run或增加stage/safety gate。
  这些要么没有当前因果证据，要么改变训练/接触问题。

## 7. 2026-08-23 R03 exact-strike与PPO探索修复

### 7.1 当前V2不是“再多等一些step”

- Isaac现役源码在R03 arm时要求`REVEAL_COMMITTED`，但exact-strike one-shot只会在真实发射后出现；因此
  production chronology下R03不可能发布。MuJoCo又用`expected_step <= current_step`反复重采当前FK，既不是
  exact问题，也没有保留首次回答。两处是实现错误，不是balance→mimic→hit→landing课程设计错误。
- MuJoCo V2的learned `log_std[31]`没有decay/clamp；永久`entropy_coef=.01`对每维每minibatch施加固定
  `-.01` loss梯度。真实checkpoint的post-update mean std从update0的`.020076`升到update500的`.094639`、
  update1000的`.216543`、update2000的`.481180`和update2500的`1.148104`；现役只读刷新已超过`2.8`。
  这与任务质量无关，并已破坏balance，所以现役V2不得resume。

### 7.2 adopt / defer / reject

采用：

- Isaac只在`LAUNCH_SETTLED && exact-one-shot`时冻结R03问题；publish只核冻结的identity/source，即使同一
  post-physics tick随后真实进入OUTCOME也不得重新按phase拦截。MuJoCo同样只接受
  `armed && expected_step == current_step && LAUNCH_SETTLED`，exact tick一次采样；event下一tick归零，owner
  fact/valid/source保持到task reset。wrong phase或错过exact tick都不补发。
- PPO V3只把FullMDP typed recipe的`entropy_coef`从`.01`改为`0`。learned std仍由真实advantage更新，
  adaptive KL仍约束actor mean/std总变化；H48、lambda `.98`、E5/MB8、LR与fresh sigma `.02`不变。
- MuJoCo launcher把Warp kernel cache绑定到fresh `<run-root>/warp_cache`，只解决已由Warp 1.16.0源码确认的
  root-cache写入路径；它不是新Gate，也不冒充Isaac的Omniverse/Triton root-cache问题已关闭。

延后：

- actual-KL统计只可作为optimizer摘要；本轮不做promotion Gate。若fresh 20-update仍出现同量级的各维std
  一致上涨，再定位surrogate机制，届时才讨论上限。
- R03 sticky fact是否只在event当tick计Reward、219→216 critic ABI、两后端单一巨型状态与CUDA Graph均不
  混入本轮实现修复。

拒绝：

- 不增加phase owner、receipt、poison gate、entropy decay状态机或std clamp；不在publish时二次检查phase。
- 不把真实contact、R07 readiness或rollout成功作为R03许可；它们会重新制造“先学会下游才开放上游”的环。
- 不热补现役V2进程、不复用namespace、不从已膨胀std的checkpoint resume。

### 7.3 实现与验收状态

- Isaac/MuJoCo都增加production chronology正例与IDLE/REVEAL/OUTCOME/RETIRED、wrong-phase/no-catch-up、
  one-shot/sticky/reset反例；另覆盖LAUNCH时arm后经真实R06进入OUTCOME仍可publish，防止错误phase gate回归。
- launcher dry-run不得创建root，real root预建0700 cache且ambient `WARP_CACHE_PATH`不得泄漏。
- clean detached commit `46800617c505f9cc10ff72f9228a60ece2bb0293`在exact Pod分进程聚焦回归为
  launcher `15 passed`、PPO/train wiring `139 passed,1 skipped`、Isaac R03 `27 passed`、MuJoCo
  lifecycle/RSL `61 passed,6 skipped`、扩展Epoch/runtime `103 passed,6 skipped`，合计
  `345 passed,13 skipped`；skip仍是既有可选依赖/CUDA边界。
- fresh MuJoCo namespace `fullmdp-a-h48-v3-mujoco-exact-r03-46800617-20260823`绑定GPU0，Warp首次编译
  产物已确认只写run-owned cache。到ACK35，mean std从`.01999999`降至`.01958219`，最近10轮
  pre-ACK中位`9.524 s/H48`；Reward/storage finite、conservation fault=0。
- fresh Isaac namespace `fullmdp-a-h48-v3-isaac-exact-r03-46800617-20260823`绑定GPU1。到ACK23，日志
  mean std保持`.02`，最近10轮中位`6.915 s/H48`；Reward nonfinite和attributed fault均0。两端都通过
  20-update optimizer canary，没有复现V2的统一std上漂。
- successor分别连续8个MuJoCo ACK和11个Isaac ACK后，才按旧process的PID/start-ticks/argv精确替换V2；
  MuJoCo一次TERM退出，Isaac leader一次TERM退出、Kit child第二次TERM退出，未用KILL。旧root/checkpoint/log
  保留，completion仍缺席；GPU2已释放。当前只有两条V3 lineage运行，均`diagnostic_unauthorized=true`。
- 后续只读刷新到MuJoCo update790 / Isaac update694：前10→recent10完成episode均长分别从
  `68.05→211.05 tick`和`87.53→176.72 tick`，说明balance继续改善但尚未基本成功。MuJoCo累计
  `due/reveal=1802/1802`、deferred=0；Isaac累计`due/selected/accepted=233/233/218`且playback started=`42`，
  证明row一活到first reveal tick295就立即获得mimic/task输入，没有R07或新Gate阻断。
- 后半段仍未闭合：MuJoCo只有`launch=1`，该row只在launch phase活了2个control tick，没到20 tick后的
  exact-strike，因此R03=0；Isaac launch=0。两端contact/landing也都是0分母和`未测`。这不是旧R03 phase
  bug复发，但说明“mimic基本成功→开始击球”尚未发生，不能把入口打开写成阶段成功。
- exploration因果修复持续成立：MuJoCo mean std `.02000→.01640`，Isaac TensorBoard精确
  `.02000495→.01445465`；Reward nonfinite、owner/conservation fault均0。recent10 raw wall分别为
  `9.833/10.380 s/H48`，H24-equivalent约`4.916/5.190 s`；相比旧约22秒已大砍，但raw H48仍未达到约6秒，
  后续性能刀必须从active/mimic profiler找真实主墙，不能缩rollout或加Gate冒充提速。

## 8. 2026-08-23 V3阶段复核、reference对账与V4实现纠错

### 8.1 V3当前不是“训练失败”，也不是可继续忽略实现错误

2026-08-22T21:17Z只读快照如下。两条run均仍绑定clean `46800617…` source、进程存活、finite且
`diagnostic_unauthorized=true`；本节没有hot-patch、signal、resume或复用namespace。

- portable MuJoCo到update1256、`247,136,256` transitions。early100→recent100 episode均长约
  `116.63→230.73 tick`，recent100平均回报约`21.25`；累计`due/reveal/deferred=10648/10648/0`，但
  `launch=1 / R03=0 / selected contact=0 / R06=0 / landing=0`。recent100 raw H48 mean/median=
  `11.269/11.262 s`，不是约6秒目标。
- Isaac到durable ACK1024、`201,523,200` environment steps。early100→recent100 episode均长约
  `91.06→183.18 tick`，recent100平均回报约`12.39`；累计`due/accepted/playback=656/601/101`，
  playback/accepted约`16.8%`，但physical launch、R03、contact、R06、landing仍全0。recent100
  collection mean/median=`14.495/13.794 s`，完整iteration=`15.790/14.810 s`；该V3错误使用远离GPU1
  NUMA-local的CPU affinity，当前迭代速度不能称正常。
- 两端timeout终止均为0，因此episode变长只证明balance/survival与reset-ready imitation继续改善，不能写
  balance“基本成功”。task入口已经自然打开，但绝大多数accepted/revealed row在playback/launch前由fall、
  table或旧Mu qdes语义结束；mimic尚未基本形成，hit与landing仍是零分母`未测`。

这个顺序与预期的balance→mimic→hit→landing一致：允许前一阶段花很多step；验收点不是固定iteration，
而是上一阶段基本形成时下一阶段已经有真实非零exposure。当前只验证了balance存活row会立即得到mimic题，
尚未验证mimic基本形成后会得到launch/contact，也没有资格判断hit→landing交接。

### 8.2 reference与Observation重新对账

本轮没有把“launch=0”重新包装成一个未经证明的reference偏移。对账
[`action_ball_dual_backend_longrun_todo_20260819`](../../operations/action_ball_dual_backend_longrun_todo_20260819.md)
与[`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)
后三层结论仍一致：

- **HISTORICAL / SUPERSEDED by §9.2：**当时的概括是physical reset与hidden WAIT使用同一reset-ready
  joint/body、reveal切measured frame0。现役更精确的边界是只有首次ACCEPT前用reset-ready；ACCEPT同tick
  joint/body都切frame0，后续recovery/ready也持有completed-action frame0。direct frame0 physical birth
  已被exact Pod `0/73`拒绝，不能因当前训练慢就复活该方案。
- 当前fresh slot0（`mount_normal_sign=+1`）里，portable teacher/contact旧组合错位与Isaac少一个球半径已经在
  §6.2用唯一official racket site、共享ball-centre offset与构造期几何反例修掉；perfect mimic在exact strike
  边界可触球。该结论不得外推给全部73动作：negative-face动作的raw +Y mimic/R03 normal与selected physical
  face符号必须另有反例闭合。若fixed-tape回归失败，应归类实现回归，不应添加actor观测或准入Gate。
- 203-D actor已经有deployment-intended root/gravity/gyro、q/dq/last action、teacher q/dq/anchor、motion phase、
  task位置/速度/法向误差、base goal、三个clock、epoch phase与task-valid；219-D critic才保留物理/contact/
  support事实。HITTER、SMASH、BeyondMimic对照支持这份分层。本轮没有找到现有actor状态相同却要求不同未来动作
  的具体alias反例，因此拒绝冗余、不可部署的新Observation；真实producer仍须完成G07的OptiTrack calibration/
  synchronization/stale-dropout合同，只有出现alias反例时才重新讨论future-teacher preview。

### 8.3 已采用的实现纠错

V3的下游零事件尚未触发多数错误，但第一枚真实hit/outcome会立刻污染经济，因此不能继续等到事件出现才修：

- sticky R03、Physical contact与R07只在各自fresh source step付钱；R06两个ordinal共享settlement前冻结图并
  完整支付一次，payment mailbox用full key/publication/settlement/payment chronology闭合。sticky fact仍留给
  Observation与审计，不能重复变成Reward。
- Isaac/portable MuJoCo launch都只接受row自己的exact Motion clock equality；missed tick不得catch-up后继续沿用
  旧contact/deadline。Isaac用既有packed host reduction报具名`missed_launch_tick`并禁止scene write；MuJoCo用
  `full_a_missed_launch_event`在PPO前ledger具名失败，不增加逐step D2H。
- 共享cadence从五个due收成episode内能完成的四个`295/588/881/1174`。第五个1467最早launch已超过1500 horizon，
  不是课程机会；保留它只会制造永远无法完成的分母。
- MuJoCo invalid racket-contact classification结算invalid outcome并退休shot，但不伪造Gym Done/reset generation；
  q-des predicted/current inner-envelope intervention与actual hard-edge作为逐update evidence，只有nonfinite q-des保留
  双后端相同的qdes Done语义。
- portable evidence schema 4在已有一次PPO前reduction中检查policy/critic observation、actions、values、
  log-prob、mu/sigma、reward/return/advantage finite，done二值且sigma为正；consumer按exact schema拒绝旧V3
  冒充兼容。两端launcher把CUDA/XDG/Warp/TMP/pycache放到fresh run root；Isaac ready后直接核child继承同一
  GPU lifetime flock，不能再用`lslocks`缺行误判。
- R05删除把一行结构故障广播到全batch的两个Python O(N)逐行reduce；故障保持row-local。Physical未来launch
  pending只读Motion per-env clock并走零scene/R06 fastpath，R07只在真实transport或recovery业务存在时执行完整
  keyed路径。这些是状态/数据流减法，不是新增Gate。

### 8.4 safety与结构裁决

按`HANDOFF_TO_CODEX_20260808.md`§3，继续硬失败的是来自独立事实源且会让环境不可学或证据不可信的边界：
nonfinite、full-key/generation/chronology、plant fall/table、scene writer、optimizer/storage domain、exact process/
GPU lock。task成功、每update必须出事件、R07 readiness、同writer回声/hash、actual finite hard-edge立即reset、
late catch-up都不是安全Gate；它们只做telemetry、课程结果或直接删除。

结构审计也确认当前代码过于臃肿：Physical约万行、Epoch/R06各有多份sticky fact/payment镜像，production测试还会因
同一源码被不同import alias加载而出现exact-type假失败。过多owner、reverse ACK与same-writer self-proof既拖慢热路，
也让真正跨owner边界藏在噪声里；旧匿名CUDA assert延迟到下一PhysX write才报错就是实际代价。V4前只实现已有
因果证据的纠错与减法；不先造巨型`ActionBallState`、C++重写、第二套receipt或新stage。V4运行后再按
“一条lifecycle spine + backend StepFeatures adapter + 纯fact/payment kernels + 一个telemetry funnel”分批重构，
每次删除self-proof都保留独立mutation与fixed-tape/RNG/reason/counter parity。

仍保留的结构债不冒充本轮blocker：R06 launch admission/payment的真正跨owner谓词应从匿名async assert迁成
写前preflight+具名fault；Observation最终finite与R07↔Motion chronology也应迁到已有host boundary。它们的正常
production after-image当前可达且本轮反例未触发，但后续不能继续用async assert当写前安全屏障。MuJoCo的family
字段也仍由launcher自陈，当前单一slot0/forehand只能按该分母报告，其他动作/侧继续`未测`。

### 8.5 host验收与下一边界

修改文件逐进程host回归合计`533 passed,23 skipped`；另有alignment聚焦`12 passed,23 deselected`。
完整alignment本机因未安装pinned `mjlab`有`30 passed,5 failed`，五项都是同一
`ModuleNotFoundError: mjlab`导致axis不可验证，不是把失败吞成PASS；它必须在exact Pod runtime重跑。
全部修改Python已通过`py_compile`，`git diff --check`通过。

当前状态仍是branch candidate：exact Pod clean detached回归、ignored EPA48/RSL3资产恢复、两个one-shot
dry-run、GPU-local affinity、fresh V4连续ACK和真实wall尚未完成。完成前不替换V3、不宣布速度恢复、也不更新
`docs/NOW.md`或任何formal Gate为Done。

## 9. 2026-08-23 V4固定前缀与V5最终候选边界

### 9.1 V4真实训练复核：MuJoCo继续，Isaac因匿名实现故障停止

clean commit `9e26afd3342e1da8643b225c987d4a3c91a3ff2f`已在exact Pod得到
`589 passed, 12 skipped, 0 failed`；skip为明确的real-GPU边界，两个one-shot launcher dry-run也通过。
随后两个fresh namespace分别在GPU1/GPU0运行，均保持`diagnostic_unauthorized=true`，没有复用V3
checkpoint或namespace，也没有对现役源码hot-patch。MuJoCo仍在只读推进；下列是本轮文档收敛时临时固定
前缀，精确停止前将只替换数值，不追加新的时间流水账：

- Isaac最终连续durable ACK为`0..748`（749轮、`147,259,392` transitions）；update749在optimizer前以
  `epoch drain decoded a terminal overflow`停止，没有ACK749，进程已退出。V4只有单一generic bool，历史
  证据不能唯一反推出具体row/cause。recent50 episode均长/return=`179.802/12.157`，wall mean/median=
  `9.128/9.100 s/H48`；D05 due/ACCEPT=`300/270`，playback=`61`，launch=`1`，contact/R03=`0/0`。
  R06虽有一次settlement，但contact/net/cross/table/common均为0；R07/retire均为0，不能写成有效落点样本。
- portable MuJoCo固定前缀连续durable ACK `0..4509`（4510轮、`886,702,080` transitions），launcher/child
  仍以原PID/PGID/start-ticks存活。recent20 wall mean/median=`14.299/14.301 s/H48`、episode均长=
  `378.583 tick`。recent20 public due/reveal/defer=`10,487/10,487/0`、launch=`6,809`、R03
  physically-valid=`2,176`，racket/selected contact与landing均为0；累计reveal/launch/R03-valid=
  `1,488,260/426,606/112,669`且contact仍为0。旧schema-4的`racket_contact_eligible`只是launch的同写者
  别名，不作为接触机会或分母；actor已有Motion phase，当前缺的是独立durable
  playback denominator，故playback成功分母继续`未测`。

两端全部durable Reward/storage都finite，nonfinite/conservation为0；MuJoCo lifecycle fault为0，Isaac则由
上述未解码实现故障终止。MuJoCo episode已稳定越过295，只证明survival-to-due与task exposure
无需硬Stage便可达，不证明balance已基本成形；目前也没有因果证据要新增balance Reward或R07
readiness Gate。大量launch与valid R03之后仍然0 contact，说明
旧实现的mimic→hit交接不符合预期，不能再用“还在balance早期”解释。结合§9.2的hidden-reference回归和§9.3的Physical
typed/legacy identity回归，
V4结果已被实现错误污染，不能用来证明课程设计成功或失败。hit→landing没有合格分母，继续`未测`；
V5必须fresh重跑；在没有独立durable playback denominator前，playback成功继续`未测`，
不能由reveal替代。

### 9.2 reference/Observation结论不因稀少事件反转

这次快照继续按
[`action_ball_dual_backend_longrun_todo_20260819`](../../operations/action_ball_dual_backend_longrun_todo_20260819.md)
和[`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)
对账。没有出现新的same-observation/different-required-action反例，也没有current slot0的构造期位置几何
回归；所以不能把launch/contact稀少重新解释为新offset，或据此增加冗余/不可部署actor Observation。

本次代码自查反而抓到一个与既定文档相反的实现回归：旧Isaac fresh路径把generic `in_hold`当作reference
选择器，造成ACCEPT后joint仍取runtime default、body却已取frame0。现役权威不再用“hidden”这个模糊词，而是：

1. `fresh_bound && policy_opportunities_created==0`的首次ACCEPT前行，joint/body都取reset-ready；
2. D05在post-transition ACCEPT边界原子安装selected measured action frame0 joint/body到返回的`obs_{t+1}`，
   第一笔task-conditioned action/reward发生在下一transition，再按公开clock进入playback；
3. playback结束后的recovery/ready取上一条completed action frame0，不回退到reset-ready。

V5只修这个selector与同tickbody cache，不新增offset、owner、stage或Observation。current slot0
`mount_normal_sign=+1`的official racket site/一个球半径位置几何仍成立；N73中
`mount_normal_sign=-1`的动作则必须把raw +Y mimic/R03 normal与selected physical face分离，这是现有
法向合同实现错，不是新obs、课程或安全Gate。

现役203-D actor与219-D privileged critic边界保持不变；`history_length=0`是当前实现事实，对照系统的
`history=8`只保留为deferred候选，除非先构造出alias反例。真实table/root状态producer仍由G07闭合，
不能把sim truth称为已部署。actor同时有两个不同的5-way phase：Epoch learning phase描述shot业务账的
`IDLE/REVEAL/LAUNCH/OUTCOME/RETIRED`，Motion phase描述teacher/reference的
`prepare_visible/swing/follow_through/recover_hidden/ready_hold`。合法物理launch可先推进Epoch，
而Motion仍可在frame0/prepare-visible边界；
`task_valid`由Motion可见性决定，不能用Epoch retained payload替代。把两种phase硬要求同步，正会制造
本轮要删除的伪chronology fault。

当前科学裁决不是“再加东西”，而是先恢复已经选定的最小合同：

| 项 | 裁决 | 证据与理由 |
| --- | --- | --- |
| PPO | 采用H48、GAE `lambda=.98`；其余V3配方不变 | rollout是学习取舍，不代签速度 |
| Observation | 保持actor/critic `203/219`，不造237-D | 203-D已含`base_position_table(3)+base_heading_table_xy(2)+base_com_lin_vel_heading(3)`这8维；无alias反例 |
| History | 当前`history_length=0`；`history=8`延后 | 扩大部署/normalizer合同前先要因果反例 |
| Balance | 不新增Reward或Stage | V4只证明survival-to-due与下游曝光可达；尚无因果证据需增机制，也不等于balance或安全存活毕业 |
| Reference | 采用三段selector修复 | 首次ACCEPT前reset-ready、post-transition ACCEPT边界安装frame0到`obs_{t+1}`、recovery completed-action frame0 |
| Curriculum boundary | scheduled due与public due分开；DEFER只表示结算后仍busy | 同tickterminal不能假报actor看见reveal；自然RETIRED可立即ACCEPT，R07不是admission |

### 9.3 V5只做可证伪的结构减法与plant身份收口

[fullmdp-a-h48-v5-*](../../DEFINITIONS.md#fullmdp-optimization-lineage-v5)当前仍是未形成最终clean commit、
未发车、未完成exact Pod验收的branch candidate。它保持H48、PPO、自然课程、Reward经济、Observation和
backend physics；termination只修pure-timeout/canonical reason分区。学习配方继续为H48、GAE
`lambda=.98`、E5/MB8、entropy 0、fresh learned `log_std`/init sigma `.02`；“6秒”只是继续大砍
墙钟的方向，不是Gate。下表是最终候选应同时满足的合同，不是已验收清单：

| 边界 | V5唯一现役合同 |
| --- | --- |
| transition | 冻结scheduled due candidate → 结算既有launch/park → 用`obs_t` teacher跑physics/terminal/facts/reward → 结算outcome/recovery → 只对存活row分类public due → 安装selected frame0到`obs_{t+1}`。 |
| due/lifecycle | `scheduled_due_rows`记schedule在本transition到达；`due_terminal_overlap_rows`记due与terminal重叠且永不actor-visible；`reveal_due_rows`只记存活的public due。结算后仍busy才`DEFER`；起始busy但当界自然`RETIRED`可立即`ACCEPT`。R07只提供recovery telemetry/reward，不是admission。 |
| legal overlap | launch与outcome可在同tick都为true；outcome与natural recovery互斥。Motion与Epoch是两个独立phase clock；合法payment/launch/outcome不必等待Motion close或teacher离开frame0。 |
| Reference | 首次ACCEPT前joint/body共同使用reset-ready；post-transition ACCEPT安装selected measured frame0到`obs_{t+1}`；首个task-conditioned action/reward从下一transition开始；recovery/ready使用completed-action frame0。不新增offset。 |
| fault owner | Isaac共享ActionEpoch owner有36个single-bit row cause：bit0--26的27项旧原因、R03 identity/stale/nonfinite的bit27--29、Physical/R06六项bit41--46。MuJoCo另有四项per-transition fact-integrity：R03 nonfinite、R06 source-invalid、R07 sequence、R07 nonfinite。两个命名空间不混计；Mu R03 stale因逐tick调用与同step消费按构造不可达，不保留恒0 Gate。 |
| fault drain | 每个backend的坏row在业务写前先latch并neutral/freeze，raw audit保留、健康peer继续；所有原因只进入既有唯一packed pre-optimizer drain，不增加第二owner或逐step D2H。 |
| typed identity | direct scene、postphysics与R06只使用ActionEpoch typed 8-field key/publication，Physical唯一推进ordinal和previous/current-center continuity；legacy plane保持隔离，不用假digest安抚旧Gate。 |
| terminal/reason | `pure_timeout = raw_timeout & ~plant_terminal`；只有pure timeout使用RSL bootstrap和canonical timeout reason。horizon与tilt/table/qdes重叠不bootstrap。Mu `robot_hit_table` bit表示keepout或resolved-table，resolved只是子fact。`invalid_contact + Gym done`是真reset而非retire。 |
| Observation/face | actor/critic保持`203/219`、`history_length=0`；无alias反例不加Observation/history。raw mount A-frame `+Y`只服务teacher/actor/R03 normal；`mount_normal_sign`只选physical face、ball-centre offset与contact classifier。 |

证据wire最终只发布`evidence/update schema=6`、`completion schema=5`、`summary schema=5`。
上一个已提交版本是`5/5/4`；本轮未发布的中间版本不引入第二次schema bump。旧V4的
`racket_contact_eligible`与launch逐行exact相等，只是已删除的冗余别名，不是contact opportunity。
新consumer只对fresh prefix验证可证明的因果边界：`launch<=reveal`、`racket contact<=launch`、
`selected contact<=racket contact`、`R03 present<=launch`、`flight outcome<=launch`、
`landing crossing<=selected contact`、`shot retired<=launch`。R03 exact-strike与contact使用不同
时钟，`selected_contact/R03_valid`只能作描述比；真实selected-contact rate的分母是launch。
summary中的`r06_common_per_eligible`才是closed task-landing成功率；
`opponent_landing_per_crossing`只表示已crossing条件下的比例，不得代称总成功率，也不新增event或Gate。

性能减法只保留两个可审计方向：MuJoCo合并base metric与Full-A的contact-buffer遍历，Isaac consumer
改读ActionEpoch窄投影而不复制dense `current()`。它们必须在最终源码冻结后用
fixed-tape/RNG/reason/counter/safety parity和profiler-off matched-strata wall验收；host微基准、
理论调用数或旧Pod证据都不代签新V5速度。

MuJoCo runtime身份保持两个正交真源：`runtime_stack`原子绑定exact EPA48、RSL3.1.2与MJLab1.5.3 tree；
`source_plant`/`runtime_attach` v2绑定vendor base receipt、exact geometry、`decimation=20`、
`step_dt=.02`、capacity、owner-local frame与court/ball augmented MJB。runner先cold verify，再
private-stage→hash/fsync→no-clobber发布run-owned `runtime.mjb`；consumer在ACK前独立验证并加载。
这些是防止训练不可学或证据不可信的边界，不是业务成功Gate。

按`HANDOFF_TO_CODEX_20260808.md` §3，只有独立的plant/finite/full-key/optimizer/durable WAL/ACK事实
可在未满足时阻断学习或拒绝证据；task success、R07 ready、same-writer echo和stdout都不是安全Gate。
stdout payload可在transaction前序列化，但提交顺序是optimizer → WAL/fsync → owner ACK →
EPOCH_ACK/fsync，最后才best-effort写stdout marker；short write、`BrokenPipe`或flush错误只写
structured stderr warning，不能污染或撤销已提交训练。结构只逐步收敛到
`PlantFacts → ActionBallState transition → StepTelemetry`，不以一次巨型重写取代每次抽取的对拍。

最终clean commit SHA、final host测试数、exact Pod GPU direct/RSL one-update、双launcher dry-run、
fixed-tape、fresh 1/5 ACK、profiler-off wall、fresh双后端长跑与contact/landing分母全部
**待源码冻结后填入**。旧V4和a103等旧lineage的GPU/ACK证据不迁移；在这些闭合前，V5不得写成
已发车、已提速、已验证或Gate完成。
