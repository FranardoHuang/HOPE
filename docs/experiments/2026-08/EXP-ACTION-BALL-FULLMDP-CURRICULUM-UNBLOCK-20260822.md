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

## 9. 2026-08-23 V4双后端实时复核与V5候选边界

### 9.1 V4真实训练复核：MuJoCo继续，Isaac因匿名实现故障停止

clean commit `9e26afd3342e1da8643b225c987d4a3c91a3ff2f`已在exact Pod得到
`589 passed, 12 skipped, 0 failed`；skip为明确的real-GPU边界，两个one-shot launcher dry-run也通过。
随后两个fresh namespace分别在GPU1/GPU0运行，均保持`diagnostic_unauthorized=true`，没有复用V3
checkpoint或namespace，也没有对现役源码hot-patch。下列证据刷新于2026-08-23T11:39:47Z：

- Isaac最终连续durable ACK为`0..748`（749轮、`147,259,392` transitions）；update749在optimizer前以
  `epoch drain decoded a terminal overflow`停止，没有ACK749，进程已退出。V4只有单一generic bool，历史
  证据不能唯一反推出具体row/cause。recent50 episode均长/return=`179.802/12.157`，wall mean/median=
  `9.128/9.100 s/H48`；D05 due/ACCEPT=`300/270`，playback=`61`，launch=`1`，contact/R03=`0/0`。
  R06虽有一次settlement，但contact/net/cross/table/common均为0；R07/retire均为0，不能写成有效落点样本。
- portable MuJoCo固定快照连续durable ACK `0..3995`（3996轮、`785,645,568` transitions），launcher/child
  仍以原PID/PGID/start-ticks存活。recent20/50 wall约=`14.344/14.346 s/H48`；episode length recent20/50=
  `375.28/375.70`，已越过first due 295，但recent20全部`10,450/10,450` completed episode仍由tilt/table
  原始信号闭合，不能称安全存活毕业。累计reveal=`1,219,808`、launch/contact-eligible=
  `265,378/265,378`、R03 valid=`63,684`，但racket/selected contact仍为0。当前版本没有发布独立
  playback字段，所以该格写`未测`，不能由reveal或launch反推；R06 present/eligible=`379`全为
  `flight_expired`，post-hit landing/recovery因hit分母为0继续`未测`。

两端全部durable Reward/storage都finite，nonfinite/conservation为0；MuJoCo lifecycle fault为0，Isaac则由
上述未解码实现故障终止。MuJoCo的episode-length越过295且recent50均长到`375.70`，已经证明balance可学且task
exposure无需硬Stage便会自然打开；因此不新增balance Reward或R07 readiness Gate。但在265,378个
contact-eligible launch及63,684个valid R03之后仍然0 contact，且mimic收入基本横盘，mimic→hit交接已
明确不符合预期，不再能用“还在balance早期”解释。结合§9.2的hidden-reference回归和§9.3的Physical
typed/legacy identity回归，
V4结果已被实现错误污染，不能用来证明课程设计成功或失败。hit→landing没有合格分母，继续`未测`；
V5必须fresh重跑并发布独立playback/typed-fault telemetry。

### 9.2 reference/Observation结论不因稀少事件反转

这次快照继续按
[`action_ball_dual_backend_longrun_todo_20260819`](../../operations/action_ball_dual_backend_longrun_todo_20260819.md)
和[`EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802`](EXP-ACTION-BALL-MUJOCO-NATIVE-READINESS-20260802.md)
对账。没有出现新的same-observation/different-required-action反例，也没有current slot0的构造期位置几何
回归；所以不能把launch/contact稀少重新解释为新offset，或据此增加冗余/不可部署actor Observation。

本次代码自查反而抓到一个与既定文档相反的实现回归：旧Isaac fresh路径把generic `in_hold`当作reference
选择器，造成ACCEPT后joint仍取runtime default、body却已取frame0。现役权威不再用“hidden”这个模糊词，而是：

1. `fresh_bound && policy_opportunities_created==0`的首次ACCEPT前行，joint/body都取reset-ready；
2. D05 ACCEPT同tick原子安装selected measured action frame0 joint/body，再按公开clock进入playback；
3. playback结束后的recovery/ready取上一条completed action frame0，不回退到reset-ready。

V5只修这个selector与同tickbody cache，不新增offset、owner、stage或Observation。current slot0
`mount_normal_sign=+1`的official racket site/一个球半径位置几何仍成立；N73中
`mount_normal_sign=-1`的动作则必须把raw +Y mimic/R03 normal与selected physical face分离，这是现有
法向合同实现错，不是新obs、课程或安全Gate。

现役203-D actor与219-D privileged critic边界保持不变；`history_length=0`是当前实现事实，对照系统的
`history=8`只保留为deferred候选，除非先构造出alias反例。真实table/root状态producer仍由G07闭合，
不能把sim truth称为已部署。actor同时有两个不同的5-way phase：Epoch learning phase描述shot业务账的
`IDLE/REVEAL/LAUNCH/OUTCOME/RETIRED`，Motion phase描述teacher/reference的
`pre-reveal/active/suffix/recovery/ready`。合法物理launch可先推进Epoch而Motion仍在frame0/active；
`task_valid`由Motion可见性决定，不能用Epoch retained payload替代。把两种phase硬要求同步，正会制造
本轮要删除的伪chronology fault。

当前科学裁决不是“再加东西”，而是先恢复已经选定的最小合同：

| 项 | 裁决 | 证据与理由 |
| --- | --- | --- |
| PPO | 采用H48、GAE `lambda=.98`；其余V3配方不变 | rollout是学习取舍，不代签速度 |
| Observation | 保持actor/critic `203/219`，不造237-D | 203-D已含`base_position_table(3)+base_heading_table_xy(2)+base_com_lin_vel_heading(3)`这8维；无alias反例 |
| History | 当前`history_length=0`；`history=8`延后 | 扩大部署/normalizer合同前先要因果反例 |
| Balance | 不新增Reward或Stage | V4 episode已穿过tick295且recent50均长`375.70`，说明balance可学、下游曝光已自然打开；不等于安全存活毕业 |
| Reference | 采用三段selector修复 | 首次ACCEPT前reset-ready、ACCEPT同tick切到selected measured frame0、recovery completed-action frame0 |

### 9.3 V5只做可证伪的结构减法与plant身份收口

[`fullmdp-a-h48-v5-*`](../../DEFINITIONS.md#fullmdp-optimization-lineage-v5)当前仍只是未commit、
未发车、未完成exact Pod验收的branch candidate。它不改H48、PPO、课程、Reward经济、Observation、
physics或termination；学习配方继续明确为H48、GAE `lambda=.98`、E5/MB8、entropy 0、fresh
learned `log_std`/init sigma `.02`，不能把H48误写成性能修复。自查后保留的候选减法是：

- 本轮性能减法只有两项进入当前patch。MuJoCo让一次contact-buffer遍历同时服务base metric与Full-A，保留
  per-world row count、全局counter、selected/opposite/invalid、robot-table、net/landing；final forward的
  census不重复累计base metric。扫描从`41→21/control`、H48从`1968→1008/update`，host联合=
  `335 passed,10 skipped`；CPU microbenchmark `22.248→14.395 ms`（ratio `.647`）不代签Pod CUDA wall。
  Isaac热consumer改读ActionEpoch窄投影，避免为了少量字段复制dense `current()`；N4096静态payload从
  `5.953→0.316 MiB/call`（raw bytes降`94.7%`），cached/dense-empty/retained lane每control分别调用
  `0/4/8`次（decimation=4）。invalid slot走既有`DUE_IDENTITY_LOST`且健康peer仍launch；六文件相邻回归=
  `147 passed,16 skipped,0 failed`。真实wall仍待Pod profiler，其余热路审计候选没有因此自动采用。
- 合法payment可以早于Motion close；payment assertion只核付款key/publication/settlement/payment chronology，
  真正retire仍要求close debt。合法launch/outcome也可以早于teacher离开frame0，Motion playback/close因此只认
  shared open-shot phases `REVEAL/LAUNCH/OUTCOME`，不把已物理推进的shot误判成坏账。
- Physical launch、Epoch launch consumer与postphysics consumer分别独立核exact publication ordinal；匿名
  `_undecoded_overflow`换成单个device `int64[N] row_fault_bits`。不是手抄旧“11项”：从现役
  `ACTION_EPOCH_ROW_FAULT_NAMES`动态核验为33个唯一single-bit原因。既有core/cross-owner bit0--10共11项、
  R06 runtime bit11--22共12项、Motion runtime bit23--26共4项；稀疏ingress bit41--46再加入Physical的
  `postphysics_producer/nonfinite`和R06 owner的`producer_contract/engine_overflow/nonfinite/other`六项。
  ingress在任何业务写前先latch，坏row冻结但raw owner fault仍原样进record/journal，健康peer继续；unknown/
  compound都fail closed，compound保留每个已知公共位。ActionEpoch全文件=`81 passed,7 skipped`，新增P0
  parameter case共12项。R07自己的17项local sticky ledger是另一编号空间，不能混报为ActionEpoch schema。
  packed row faults复用已有drain，多传约28 KiB/update且不增加逐step D2H；异常报告reason、row count和
  首批env id。V4旧generic bool不能
  反向证明任一具体根因。R07 bit15只拒cadence rollback/skip（idle允许hold或`+1`，keyed recovery要求
  精确`+1`），bit16只拒`int64` successor overflow；坏row neutral/freeze、健康peer继续。九文件
  fresh-process回归=`429 passed,17 skipped,0 failed`。
- direct Physical链另发现一处同等级P0：ActionEpoch launch写的是typed 8-field shot key/publication，旧
  scene/postphysics仍读legacy `outcome_sha/generation/observation` plane；fresh direct下legacy digest为0且
  ordinal不推进，身份与previous/current-center continuity都会错。V5不制造一份假digest来让旧门通过，
  而是让direct scene、postphysics和R06全程只用typed key/publication，并由Physical唯一推进observation
  ordinal与球心continuity；legacy API保持独立。参数化legacy digest=`0/23`的launch→3 substeps→retire
  反例单文件=`2 passed`：ordinal `[0,1,2]`、相邻previous/current闭合、publication恒8，retire清typed
  identity/continuity且peer逐字节不变；九文件host联合=`203 passed,17 skipped`。landing diagnostic N2的
  stale fixture仅补生产合同已要求的`replicate_physics=True/env_prim_paths`。17项skip需要CUDA/SimulationApp，
  Pod exact Isaac/CUDA N=2仍未测。
- Physical/R06曾把producer fault写进owner journal，却没有进入唯一pre-optimizer row verdict；这是典型
  “有人写、无人读”的证据错误。V5把这些原因映射进ActionEpoch具名、不冲突的ingress bits，由已有packed
  drain唯一消费。selected-rubber异步identity冲突则不再由producer self-assert炸整批：真实scene/Racket
  相邻反例中，stale typed identity只让对应row写`_binding_fault`、`face=-1`；Physical bit41映射为
  `physical_postphysics_producer`，保留raw audit但不写业务fact，lean在optimizer前终止，健康peer继续。
  lean named-fault+真实binding相邻反例=`3 passed`；Pod CUDA未测。
- Motion exact-resume现在持久化首次ACCEPT前reset-ready body tuple和pending mask，恢复后仍执行三段reference
  selector；顶层checkpoint schema保持5，Motion child schema由6升7。legacy Reward flag改为只在真实legacy
  调用时lazy import，FullMDP import不再依赖无关旧Reward图；Reward flags=`427 passed`、train wiring+flags=
  `562 passed`。Motion 6个关键文件逐fresh=`127 passed,0 failed`，11文件联合=`222 passed,1 skipped`，
  R07 live=`43 passed,7 skipped`；Pod exact resume仍未测。
- negative-face合同已收窄：raw mount A-frame `+Y`只服务teacher/actor/R03 normal，`mount_normal_sign`只选
  物理胶面、球心offset与contact classifier。`sign=-1` perfect-mimic反例要求actor residual/R03 normal error
  均为0，同时球心仍位于负面外精确一个球半径；没有新增Observation或Gate。

上述受影响的40个test文件已全部用fresh Python逐文件运行：`1724 passed,66 skipped,0 failed`；所有修改/新增
Python `py_compile`和whole-diff `git diff --check`通过。66项skip包含设备条件，不能代签exact Pod/CUDA、
fixed-tape、launcher或真实wall。

MuJoCo runtime身份同时收成两个正交、单值的合同。唯一`runtime_stack` v1原子替换旧
`mujoco_warp_runtime`字段：除exact EPA48 wheel/runtime外，还钉`rsl-rl-lib==3.1.2` wheel SHA
`406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d`，以及`mjlab==1.5.3`
选定193文件、`1,399,177` bytes、tree SHA `88c9725d...`。runner在Torch/MJLab首次import前cold verify两份
wheel和MJLab树，env构造后再核所有loaded `mjlab.*` module；consumer在读取`runtime.mjb`和ACK前独立执行
同一cold verification。这里没有保留新旧双真源。

plant_model是另一条纯标准库合同。`source_plant`只描述预注册vendor base闭包与实际
verification receipt；`runtime_attach` v2则同时绑定exact checkout geometry source、
policy clock（`decimation=20`、`step_dt=0.02 s`）、MuJoCo-Warp每world `njmax=572/nconmax=128`容量、
owner-local-frame digest与真实
MJLab court/ball augmented [`MJB（MuJoCo compiled model binary）`](../../DEFINITIONS.md#mjb)。runner把live
`env.mj_model`经private stage→hash/fsync→no-clobber hardlink发布为run-owned `runtime.mjb`。runner在env
构造后复验clean source；persist helper先要求private-stage receipt exact匹配expected
`final_augmented_mjb`，才允许hardlink，漂移时`runtime.mjb`不存在且stage无残留。consumer在ACK前独立hash并
用MuJoCo加载。plant contract=`13 passed`，runner identity=`12 passed,39 deselected`。该最终MJB诊断预注册为
`1ef4bb9e52b0b46afd422d2fe712ae38628853a1704b324b20a8ec3f26030c0b` / `72,260,546` bytes；启动器
清除ambient `HOPE_GEOMETRY_PY`，runner核实实际构造court的geometry源与合同相同。该候选针对“同一base
receipt却可换掉实际court/runtime代码”的不可信证据缺口；只有exact Pod producer→ACK→consumer闭合后才算
关闭，不是新业务Gate。

预注册来自隔离的CPU-only probe：`N=1`、`N=2`与两个不同绝对checkout路径产生同一MJB bytes；
单独改cone、contact pairs或ball spawn都会改变digest。这一正三反证明身份对路径和并行世界数稳定、
又能感知真实runtime plant变异；它不代签physics parity或训练成功。runner与consumer仍分别从提供的
locator调用同一canonical full verifier，耐久身份不含绝对XML路径，也不再把base MJB误称为augmented
runtime model。这减少的是重复验证、路径偶然性和same-writer自证，不删除独立plant/optimizer/finite/
chronology事实。formal N=1/N=2与semantic-mutation registration receipt仍未闭合，因此MJB摘要不是formal
plant authority。上述hot-path仍要求fixed-tape/reason/counter与现有测试相同，不能用host微基准代签
真实update秒数。

按`HANDOFF_TO_CODEX_20260808.md`§3，V5不增加“每update必须成功”“R07 ready才可学习”之类业务Gate；
也不以一次巨型state重写取代验证。方向只逐步收敛到single
`PlantFacts → ActionBallState transition → StepTelemetry`中心链；backend-local adapter和既有WAL/ACK留在
边界，backend physics保持分离，每次抽取都做fixed-tape parity。per-ordinal Reward snapshots、每control至少3次
D2H、canonical/flat双import和三层stamp/receipt echo已经审计确认，但先defer到Pod matched-state profile，
不冒充本轮已完成。在clean commit、exact Pod parity、fresh namespace
连续ACK与profiler-off wall产生前，V5不得写成已提速、已验证或Gate完成。

Pod首轮GPU反例进一步固定了课程边界：clean `67612c41…`的CPU合同为
`457 passed,6 skipped,0 failed`，EPA48 GPU direct五项为`4 passed,1 failed`；唯一失败的CUDA-only断言仍把
旧R07 readiness当成ACCEPT权威，而production与既有host mutation均是`due & (IDLE | RETIRED)`直接曝光，
R07只提供recovery telemetry/reward。故采纳的是删除tests-only ready注入和旧断言，让真实GPU测试直接覆盖
“balance survival→mimic question”的无额外Gate交接；不是把production改回R07 admission。并行审计还把
Mu child从numeric GPU index改为已核UUID，并让它继承同一GPU flock open-file-description，去掉枚举偶然性和
parent-only lifetime假设。更新后的host launcher=`21 passed`、transition=`62 passed,6 skipped`；必须在新的
clean commit上重跑GPU direct/RSL update和fresh 1/5 ACK后才能采用为运行证据。
