# EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822

> 问题：为什么两条 [FullMDP](../../DEFINITIONS.md#fullmdp-v9-candidate) H48 长跑没有形成可学习的
> balance→mimic→hit链；怎样用最少状态和单一真源修复后fresh重启？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`V8 stopped read-only / V9 dual-fresh running / no authorized formal run`
> 证据边界：`diagnostic_unauthorized=true`；本记录不授权 resume、promotion、export、部署或真机。

> **阅读边界（2026-08-28）：**§1--§10保留课程故障发现、旧运行和被替换实现史；当前结论只认§11。
> 尤其是旧文中tick295首次曝光、混源Take058/Take061和“V8只需更多step”均已supersede，不能反向覆盖
> V9同源ready→teacher、tick48重叠课程或当前证据边界。

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
  `R03 physically valid=26`且task Reward0--9已产生非零income/sample；这直接验证上一阶段有一个row活到due时，
  下一阶段立即开始可学，不再被旧R07门清零。selected contact=0，landing因没有selected-contact分母为
  `未测`，故击球和上台仍未闭合。最近20轮wall中位约`9.634 s/H48`（H24-equivalent约`4.82 s`）。

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
开始playback，两个backend的launch=0，故contact/landing为`未测`；击球/上台继续`未测`，不能用R03几何有效
冒充真实接触。

只读刷新到Isaac ACK176 / MuJoCo update118时，累计分母分别为
`due/selected/construction/key=2/2/2/2`和`due/reveal=15/15`、deferred=0、R03
present/physically-valid=`174/174`。Isaac recent10 collection中位`8.238 s/H48`、episode均长
`148.93 tick`；MuJoCo对应为`9.585 s/H48`、`151.17 tick`。两端fault/nonfinite/conservation仍为0，
但Isaac playback=0；两端launch=0，故contact/landing为`未测`。

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

## HISTORICAL / FROZEN — 9. 2026-08-23 V4最终封存与V5 fresh诊断边界

### 9.1 V4真实训练最终封存：两端均已停止，不再续写前缀

V4两条lineage均保持`diagnostic_unauthorized=true`，没有resume、promotion、export、部署或真机授权；本节
数字现已最终冻结，不再把只读刷新追加成时间流水账：

- Isaac最终连续durable ACK为`0..748`（749轮、`147,259,392` transitions）；update749在optimizer前以
  `epoch drain decoded a terminal overflow`停止，没有ACK749，进程已退出。V4只有单一generic bool，历史
  证据不能唯一反推出具体row/cause。recent50 episode均长/return=`179.802/12.157`，wall mean/median=
  `9.128/9.100 s/H48`；D05 due/ACCEPT=`300/270`，playback=`61`，launch=`1`，contact/R03=`0/0`。
  R06虽有一次settlement，但contact/net/cross/table/common均为0；R07/retire均为0，不能写成有效落点样本。
- portable MuJoCo最终连续durable ACK为`0..4798`（4799轮、`943,521,792` transitions）。recent20 H48
  wall mean/median=`14.146/13.994 s`。累计public due/reveal=`1,637,789/1,637,789`、defer=`0`、
  launch=`527,957`、R03 physically-valid=`145,814`，racket contact、selected contact与landing均为`0`。
  累计完成episode=`3,376,589`；terminal原因计数为base tilt=`1,521,819`、
  `robot_hit_table=1,860,583`、`base_low=383`，terminal bits允许同一episode重叠，因而不能把原因计数
  相加冒充episode总数。全部durable
  Reward/storage finite，nonfinite、lifecycle/fact fault与conservation均为`0`。该run已安全停止，completion
  缺席；旧lineage仍只是diagnostic，不伪造正常完成。

旧MuJoCo虽有大量reveal、launch与R03-valid，但独立durable mimic/playback denominator仍未测；不能以reveal
代替mimic，也不能把`0` contact直接归因为策略击球失败。结合§9.2的reference回归与§9.3的typed identity、
terminal/reason纠错，V4结果已被实现错误污染，不能证明课程设计成功或失败。V4的hit→landing与recovery
同样没有合格分母，统一记`未测`；只有fresh V5可以提供新的因果证据。

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

### 9.3 V5 clean冻结、exact Pod验收与fresh双后端诊断

[fullmdp-a-h48-v5-*](../../DEFINITIONS.md#fullmdp-optimization-lineage-v5)已冻结到clean source
`39f9481950a660e198dedac7fd402806d648906b`。它保持H48、PPO、自然课程、Reward经济、Observation和backend
physics；termination只修pure-timeout/canonical reason分区。学习配方继续为H48、GAE `lambda=.98`、
E5/MB8、entropy 0、fresh learned `log_std`/init sigma `.02`。下表是该source的现役合同：

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

证据wire只发布`evidence/update schema=6`、`completion schema=5`、`summary schema=5`。
上一个已提交版本是`5/5/4`；未发布的中间版本没有引入第二次schema bump。旧V4的
`racket_contact_eligible`与launch逐行exact相等，只是已删除的冗余别名，不是contact opportunity。
新consumer只对fresh prefix验证可证明的因果边界：`launch<=reveal`、`racket contact<=launch`、
`selected contact<=racket contact`、`R03 present<=launch`、`flight outcome<=launch`、
`landing crossing<=selected contact`、`shot retired<=launch`。R03 exact-strike与contact使用不同
时钟，`selected_contact/R03_valid`只能作描述比；真实selected-contact rate的分母是launch。
summary中的`r06_common_per_eligible`才是closed task-landing成功率；
`opponent_landing_per_crossing`只表示已crossing条件下的比例，不得代称总成功率，也不新增event或Gate。
`business_chain_complete`目前仍是producer逐row attestation加consumer边际聚合一致性，不是独立
same-env/same-epoch重放；晋级前要用keyed carry-state或可重放trace闭合。

性能减法只保留两个可审计方向：MuJoCo合并base metric与Full-A的contact-buffer遍历，Isaac consumer
改读ActionEpoch窄投影而不复制dense `current()`。数学与性能裁决仍要求fixed-tape/RNG/reason/counter/safety
parity和profiler-off matched-strata wall；host微基准、理论调用数或旧Pod证据都不能代签。fresh H48数值
尚未完成matched-strata对照，当前不能正式归因，且仍高于约`6 s/H48`的优化方向。

MuJoCo runtime身份保持两个正交真源：`runtime_stack`原子绑定exact EPA48、RSL3.1.2与MJLab1.5.3 tree；
`source_plant`/`runtime_attach` v2绑定vendor base receipt、exact geometry、`decimation=20`、
`step_dt=.02`、capacity、owner-local frame与court/ball augmented MJB。runner先cold verify，再
private-stage→hash/fsync→no-clobber发布run-owned `runtime.mjb`；consumer在ACK前独立验证并加载。
这些是防止训练不可学或证据不可信的边界，不是业务成功Gate。

按`HANDOFF_TO_CODEX_20260808.md` §3，只有独立的plant/finite/full-key/optimizer/durable WAL/ACK事实
可在未满足时阻断学习或拒绝证据；task success、R07 ready、same-writer echo和stdout都不是安全Gate。
stdout不是authority；提交顺序是optimizer → WAL/fsync → owner ACK → EPOCH_ACK/fsync。live `39f94819`
的stdout为regular file且未失败，但审计发现两处post-durable裸print；下一source `a3c528f1…`已改为
best-effort structured warning并增加closed-pipe反例，不hot-patch或重启live。结构只逐步收敛到
`PlantFacts → ActionBallState transition → StepTelemetry`，不以一次巨型重写取代每次抽取的对拍。

clean source的exact Pod验收分层如下，不能把不同环境的PASS混成同一个integration声明：

- broad回归为`792 passed, 57 skipped, 0 failed`；clean runtime重跑为`77 passed, 0 failed`。
- MuJoCo GPU direct为`5 passed, 0 failed`，RSL H48 one-update为`1 passed, 0 failed`。
- Isaac GPU检查为`32 passed, 0 failed`，但其中runner drain使用的是非Isaac integration，不能单独代签真实Kit路径；下述fresh
  Isaac live ACK为真实Kit边界补上了活证据。以上仍不把diagnostic提升成formal Gate。

两个fresh namespace都绑定clean `39f94819…`、从零训练且
`diagnostic_unauthorized=true`，没有迁移旧V4/a103 checkpoint或ACK：

- Isaac fresh诊断namespace
  `fullmdp-a-h48-v5-isaac-chronology-39f94819-20260823T144237Z`的首个验收前缀为durable ACK `0..63`
  （64轮、`12,582,912` transitions）；最后5轮H48 wall依次为
  `7.78/7.47/8.17/6.79/8.19 s`。累计完成episode=`139,264`且全部为tilt；所有D05与shot事件均为`0`，
  nonfinite、fault与conservation均为`0`。只读推进到ACK97时，early10→recent10完成episode平均长度
  `87.606→97.776 tick`、平均return `7.052→8.705`，说明balance已经开始学习，但尚不能写成基本成功，
  也尚无public due。
- portable MuJoCo fresh诊断namespace
  `fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z`的验收前缀为durable ACK `0..8`
  （9轮、`1,769,472` transitions）；recent5 H48 wall mean/median=`8.727/8.777 s`，约
  `22,528.7 transitions/s`。累计完成episode=`16,378`、平均长度=`105.213 tick`，全部以
  `robot_hit_table`终止；所有due、reveal、defer、launch及其下游shot事件均为`0`，Reward/storage finite，
  fact fault与conservation均为`0`。

学习轨迹现原位冻结到Isaac ACK450 / MuJoCo ACK385；详细窗口与分母只认
[双后端TODO §0.4](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)。
Isaac早期回退后已恢复并超过旧峰值；最新20窗episode mean length/return=`170.249/15.625`，累计
due/public=`20/20`、ACCEPT/reject=`17/3`、playback=2。MuJoCo最新20窗=`186.693/17.771`，累计
scheduled/public/terminal-overlap=`307/297/10`、natural launch/missed=`6/0`、R03
present/physically-valid=`1/1`。这证明两条自然课程链已经分别到达playback与launch/R03。Mu contact=
`0/6 launch`是diagnostic negative；Isaac contact因launch=0为`未测`，两端landing因selected contact=0为
`未测`，因此仍不能判mimic成功或launch→contact闭合。最新冻结wall为Isaac stdout
辅助mean/median=`9.488/9.465 s`、Mu durable mean/median=`9.235/9.223 s`；约6秒方向
尚未达到。没有resume、promotion、export、physics parity、部署或真机授权，V5不得写成formal run、已完成
提速或Gate完成。

## 10. 2026-08-25 current：V5反例收口与V6最小桥修复

本节取代§9全节作为本实验的current裁决；执行顺序只认
[双后端TODO §0.5](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#fullmdp-v6-todo-current)。
它区分已运行的branch诊断结果与仍未完成的formal/promotion，不把diagnostic写成authorized或formal。

### 10.1 V5回答了什么、没有回答什么

旧MuJoCo V5 source=`39f9481950a660e198dedac7fd402806d648906b`已经精确TERM并只读冻结，最终ACK=
`0..9046`（`1,778,712,576` transitions），没有伪造completion。最终recent50 wall
窗口固定为ACK8997..9046，mean/p50/p90=`13.314/13.301/13.328 s/H48`；`14,818`个episode的mean length/return=
`663.851/48.624`。该窗scheduled/public/terminal-overlap=`24,845/24,827/18`，launch=
`24,121`、R03 physically-valid=`23,833`，raw racket edge与selected contact均为
`0/24,121 launch`；recovery success/fail/timeout=`0/7,415/12,402`。

全历史scheduled/public/terminal-overlap=`3,484,435/3,481,891/2,544`，launch=`3,260,695`、
R03 physically-valid=`3,108,092`，raw racket edge只有2次，selected contact始终为0。因为没有
selected contact，landing没有eligible分母，必须写`未测`；recovery fail/timeout也不是selected-shot
recovery rate，不能把landing补成零。因此：

可复算输入是只读root
`/workspace/franco/runs/fullmdp-a-h48-v5-mujoco-chronology-39f94819-20260823T144237Z/evidence.jsonl`
（最后修改`2026-08-24T22:47:42Z`，SHA-256=
`138a1aed62e239713b843385972cd231b2d19638d7874178e05e0f2415091a11`）。全历史取全部9047条ACK；
recent50取末50条（ACK8997..9046）；计数逐条求和，episode mean按`Σlength或return / Σcompleted episode`
计算。

- balance/survival和return继续改善，task、launch与精确击球目标的分母也足够大；
- mimic→真实拍球的连续控制桥没有按设计出现，不能再用“训练步数还少”解释；
- 这个negative只裁决V5实现，不裁决balance→mimic→hit→landing的自然课程。修正后的fresh lineage仍须
  重新给逐阶段分母；上一阶段开始成功时下一阶段应已能取得样本，而不是等上一阶段形式毕业后才开门。

### 10.2 为什么采用measured-paddle连续prior

V5的common mimic由两个anchor项和四个body-average项构成；后四项曾把持拍腕与其他body混在平均量里，
即使整体姿态改善，policy仍可能用错误腕部姿态取得相近收入。V6的
[`Reward24`](../../DEFINITIONS.md#fullmdp-reward24)保留14项lifecycle与两个anchor项，只从四个
body-average项排除持拍腕，再用同一份官方实测动作的拍心位置、点速度、selected physical
hitting-face法向和长轴四个Cauchy prior
直接训练拍的6-DoF轨迹。四项weight=`1/1/1/.5`、width=`.70 m / 4 mps / pi rad / 1 rad`，strike window
不降权；它们的总cap为3.5，相对六个common mimic项cap 5.0足够形成桥，但仍让one-shot ball lifecycle决定击球与
落点质量。Reward没有新增隐藏oracle或课程Gate；actor侧另以Observation V3只暴露同producer、同clock的
最小teacher-achieved残差，不能把“Reward本身不加观测”误写成actor仍保持V2。

这里的mimic法向不是actor task tail里的raw-A目标残差：动作选定后，playback用该动作的
`mount_normal_sign`把achieved raw `+Y`变成selected physical face，并与同一sign语义的measured teacher
对拍；只有reset-ready没有selected动作时才使用canonical raw `+Y`。slot0的sign=`+1`不能代签未来负sign
动作。

### 10.3 从Build4学原则，不抄证明与数值

可复现参考冻结为`origin/build_4@324e60d1`。启动合同**强制**从Build1 `model21800` bit-exact
warm-start actor mean，缺checkpoint会直接拒绝，然后才把sigma重置为`.19`；这是Build4相对当前fresh run
最直接的已证配置差异，所以Build4不能被当作fresh-from-zero实验。仓内又没有actual model0 initialization
receipt；加之cadence、replay、Reward和优化器设置同时变化，不能把早期行为独立归因给继承，也不能拿两者
的早期动作量作因果对比。

Build4同时只有2条clip、约`1--1.3 s`一次的高频任务曝光、actor直接看desired拍心位置/速度与TTS，并从
proprioception推断actual拍状态；reward在strike window连续支付位置/速度/法向`14/14/5`，另有replay与
双learning-rate。这些机制可能保持或细化挥拍，但在同一配方中混杂，现有证据不能排序。它只支持
“mimic目标必须直接约束拍且policy至少能够推断纠偏状态”；没有独立actor normal或achieved-residual，
不能拿它证明Observation V3必要，也不能把某个曝光频率或reward常数写成因果答案。

Build4操作文档记录了本地model3440 checkpoint、ONNX与deploy YAML的hash；对应字节不在
`origin/build_4@324e60d1` Git tree，本轮未复算。即使接受这份历史identity记录，selected candidate的
promotion仍为`NOT_PROVEN`，没有逐侧/逐动作evaluation receipt、fresh schema22 long或matched Gate3，
artifact身份不能代签行为质量。当前V6 live仍只有single slot0；Build4的2条clip与最终73动作目标之间的差距只限制最终外推，
不能解释当前single-slot early-learning差异。Build4 virtual landing项在选中配方中为零。故本轮：

- **adopt：**Build4的direct-paddle objective/correction-information原则与逐阶段诚实分母；Reward24是本轮
  直接连续拍面实现，Observation V3 residual是本轮独立的最小实现，其必要性仍待paired control；
- **defer：**`1--1.3 s`曝光频率、warm-start、replay、双LR、sigma `.19`与`14/14/5`数值，留给独立可归因实验；
- **reject：**直接续Build4 checkpoint、把它称为physical/fullMDP formal成功，或把2条clip外推成73动作。

### 10.4 V6合同与课程时钟

learner最终采用[`PPO V5`](../../DEFINITIONS.md#fullmdp-ppo-v5)：`2048 env × H48 × U25000`、save500、
E5/MB4；继承V4的`gamma=.99`、GAE`lambda=.98`、entropy0、learned`log_std`、fresh sigma`.05`且没有
按iteration强制std decay。相对4096/U12500/MB8，总transition=`2,457,600,000`、每minibatch=`24,576`
sample和optimizer step=`500,000`保持；但policy刷新、GAE/KL、WAL和checkpoint的样本边界改变，所以这是
明确的算法/迭代性取舍，不是语义等价热路优化。exact rate已实测Mu p50/p90=
`5.468/5.526 s/H48`、Isaac=`7.81/8.38 s/H48`；前者达到约6秒方向，后者未达到严格6秒。

Observation最终采用[`V3`](../../interfaces/policy_observation_action.md#current-portable-fullmdp-semantic-observation-v3-actor-215--critic-231)：
actor/critic=`215/231`。它在旧common183与task tail之间加入4×3 heading-frame residual：同一measured
Motion teacher减去achieved paddle的拍心位置、点速度、signed physical face和长轴。四块全phase、
不按`task_valid` mask，直接复用Reward24 producer；它关闭的是Reward source未直接actor-visible的
representation gap与sample-complexity负担，不宣称已枚举strict Markov alias、V5零接触唯一根因、完整
Markov或收敛。V2 `203/219`只保留旧checkpoint ABI及后续
paired control，不能作为fresh fallback。拒绝加入raw ball/aim/rate/history/action ID；实机builder尚未
闭合，不能称deployment已验证。

四个真实due为tick`295/588/881/1174`；tick1467是第四球settlement boundary，不是第五次机会。V3 actor
clock列`[208]`（V2历史位置`[196]`）在仍有机会时
给到下一次due的时间，第四次due消费后给raw `-1` exhausted sentinel；Isaac与MuJoCo必须同义，不能分别
指向1467和episode horizon。terminal恰好跨due时，MuJoCo从pre-physics schedule直接记overlap；Isaac由
独立`ResetTelemetry`与D05 scheduled-due在既有CPU pre-optimizer drain求交集。该跨writer合成没有每步
Gate、D2H、新owner或actor字段。

### 10.5 验收边界

最终clean/pushed source=`caddecb76727ea55b0ce089453eea91cb5a9f8ea`。exact Pod host为
`1,036 passed, 11 skipped`；两端PPO V5 rate probe均自然完成61 update，Mu p50/p90=
`5.468/5.526 s/H48`，Isaac=`7.81/8.38 s/H48`。失败分类、recipe身份与receipt只认
[热路实验§16](EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md#fullmdp-v6-rate-current)。

两个[`FullMDP V6`](../../DEFINITIONS.md#fullmdp-v6-candidate) `2048×H48×25000`
[`fresh/no-resume`](../../DEFINITIONS.md#fresh-only-no-resume)长跑已于`2026-08-24T22:48:31Z`从同一checkout同时启动：

- MuJoCo namespace=`fullmdp-a-h48-v6-mujoco-reward24-obsv3-caddecb7-20260824T224808Z`，GPU0；
- Isaac namespace=`fullmdp-a-h48-v6-isaac-reward24-obsv3-caddecb7-20260824T224808Z`，GPU1。

`ACK 0..101`固定启动快照在`observed_at=2026-08-24T23:10:34Z`只读提取；两端各完成
`10,027,008` transitions，进程、PGID、exact GPU UUID、
CPU affinity与lifetime flock仍在。Mu累计`73,051`个episode，mean length/return=
`134.964/21.730`；first10→recent10固定为ACK0..9→ACK92..101，结果为
`104.337/16.043 → 159.622/26.015`。episode mean均按`Σlength或return / Σcompleted episode`计算；common mimic与新增
paddle-prior的income/sample分别`0.08751→0.09350`、`0.06794→0.06948`；scheduled/public/
terminal-overlap=`14/13/1`，launch=0，故contact与landing均为`未测`。Isaac累计`93,776`个episode，mean
length/return=`105.427/16.195`；first10→recent10为`87.502/12.956 → 129.229/20.532`。
common mimic与paddle-prior分别`0.08173→0.09008`、`0.06768→0.06887`；尚无row活到首次due，
所以task/playback/launch/contact/landing均为`未测`。income/sample均按`Σconfigured income / Σreward sample`
计算，不是policy gradient。可复算输入分别是Mu root
`/workspace/franco/runs/fullmdp-a-h48-v6-mujoco-reward24-obsv3-caddecb7-20260824T224808Z/evidence.jsonl`
首102行（SHA-256=`32f6b4ea4a21d5480a023e89445059ee72fe39647d2c996e0a3d582df5593b99`）和Isaac root
`/workspace/franco/runs/fullmdp-a-h48-v6-isaac-reward24-obsv3-caddecb7-20260824T224808Z/run.log`
首102条`HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=`行（SHA-256=
`78b8fca9f7105c72ffdde25dc959bfd6e272bda4d049e3dbb9af4a3ec4ffb520`）。

两端Reward nonfinite/conservation、attributed/fact fault与qdes-forbidden terminal均为0。recent10 raw wall
mean为Mu `5.518 s/H48`、Isaac `8.332 s/H48`。这只支持“balance/survival与common mimic/
paddle-prior income/sample同步提高；能活到295 tick的Mu row已自然打开task入口”。当前尚不能称balance或
mimic基本成功，也不能把launch=0、contact未测
解释成hit失败；后续判据保持上一阶段开始基本成形时下一阶段已有自然非零分母。25000 completion、
selected contact/landing、formal physics parity与transfer均未完成；G04/G05/G06继续`Partial`，
`diagnostic_unauthorized=true`。

### 10.6 首次双侧launch与关节遥测裁决（固定诊断窗）

`2026-08-24T23:30:13Z`只读学习快照不覆盖§10.5启动窗，而是冻结首次双侧launch这个结论变化点：

- MuJoCo到durable ACK326，共`32,145,408` transitions；recent10 ACK317..326的episode mean
  length/return=`354.184/55.818`，相对first10的`104.337/16.043`明显提高。common mimic
  income/sample=`0.09025`，略高于first10的`0.08751`；paddle-prior=`0.06725`，相对`0.06794`
  基本持平且略低，不能写成所有mimic项同步改善。累计scheduled/public/terminal-overlap=
  `24,934/24,841/93`，`59`个launch中有`3`个R03 present且`3/3` physically-valid；selected contact=
  `0/59 launch`，是有分母的diagnostic negative，landing因selected contact=0仍为`未测`。
- Isaac到durable ACK294，共`28,999,680` transitions；recent10 ACK285..294的episode mean
  length/return=`173.915/27.968`，相对first10的`87.502/12.956`提高。common mimic与paddle-prior
  income/sample=`0.09148/0.06935`，均高于first10的`0.08173/0.06768`。累计scheduled/public/
  terminal-overlap=`30/29/1`，public中accepted/rejected=`26/3`；已有`4`次playback、`2`次physical
  launch、`2/2` R03 present/physically-valid与`2`次settlement/payment，selected contact=`0/2 launch`，
  landing仍为`未测`。逐row复核的`24`个accepted-terminal样本全部因`base_fell_tilt`在episode tick
  `297..367`结束；它解释了大多数accepted尚未走到playback，而非证明accepted/playback实现断链。

两条进程在`2026-08-24T23:31:40Z`继续增长到Mu ACK335与Isaac ACK304。上述证据证明四阶段没有被硬Gate
串行化：balance尚未称基本成功时，两边已经自然打开playback/launch/R03；当前瓶颈转为继续存活和
launch后的击球学习。仍然没有contact/landing成功证据，也没有量化的“基本成功”预注册阈值；后者只按
[当前TODO的未来窗口证据判读合同](../../operations/action_ball_dual_backend_longrun_todo_20260819.md#fullmdp-v6-todo-current)
补齐，不能回看本窗后再调阈值。live recent10 wall约为Mu`9.23 s/update`、Isaac`8.9 s/update`；随着row存活
更久、task路径打开，live wall不再等于空任务占多数的fixed rate probe，因此Mu的约6秒rate方向不能冒充
全程学习wall承诺。

另在`2026-08-24T23:34:21Z`/`23:34:04Z`冻结关节遥测审计。两个后端口径不同，禁止合并平均：

- Mu ACK0..352共`34,701,312` env-policy rows；actual-hard-edge=
  `1,940,999/34,701,312=5.5934%`，qdes-guard-intervention=
  `2,060,531/34,701,312=5.9379%`。recent10 ACK343..352分别为
  `51,317/983,040=5.2202%`与`60,159/983,040=6.1197%`，latest row为`3.1016%/3.8300%`；
  first10曾为`31.5741%/33.1323%`，方向显著下降。`joint_qdes_forbidden=0`。
- Isaac ACK0..319共`31,457,280` env-policy rows。policy crossing的分母是
  `transition×31 joint`，累计`5,625,017/975,175,680=0.576821%`；substep hard crossing与actual hard
  edge的分母是`transition×31 joint×5 readback`，累计分别为
  `28,523,732/4,875,878,400=0.584997%`与`26,145,870/4,875,878,400=0.536229%`。recent10三项依次为
  `0.017175%/0.021422%/0.007762%`；first10 actual hard edge=`1.84254%`，近期下降约`237×`。
  qdes joint count、qdes-forbidden terminal与Reward nonfinite均为0。

可复算输入仍是§10.5的两个run root。Mu `evidence.jsonl`原始前353行已核update index=`0..352`，
`head -n 353 evidence.jsonl | sha256sum`为
`0bb0f818e4dc4f98e957d3eaff58eb01e50f0bf467d1ff3b594b76f344fabc21`；Isaac `run.log`筛选canonical
ACK整行后前320行已核`ppo_update=0..319`，
`grep '^HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=' run.log | head -n 320 | sha256sum`为
`46db8f7efdbd835670deab59433acdc22b6536accebdd62fe59cae9a3fec9225`。

语义必须分开：qdes-guard-intervention包含有限投影/预测越inner envelope后的可恢复制动；
actual-hard-edge则是真实模拟关节在任一readback到达或越过机械硬边，不能称为“无害投影”。现役
`diagnostic_unauthorized`配置允许finite策略通过制动恢复并继续optimizer，符合
[`HANDOFF_TO_CODEX_20260808` §3.5](../../operations/HANDOFF_TO_CODEX_20260808.md#35-一个反向的平衡franco-08-08-的纠正)
“可由policy学会回避则记账继续”的范围；两端近期显著下降且nonfinite/完整性故障为0，所以当前不停车、
不重启。但这批非零actual-hard-edge明确阻断formal、promotion、deployment与真机安全声明；若qdes
nonfinite变为非零、证据/守恒失真，或balance成熟后actual hard edge不再下降甚至反升，必须停下做根因修复。

### 10.7 V6结论翻转：生存形成，但mimic→hit桥与joint经济失败

`observed_at=2026-08-25T16:15:41Z`固定前缀已经触发§10.6最后一句的既有根因检查条件；这不是看完曲线后
补一个新Gate。两条进程仍是exact source=`caddecb76727ea55b0ce089453eea91cb5a9f8ea`、
`diagnostic_unauthorized=true`，completion均不存在：

- Mu ACK0..6846共`673,087,488` transitions。first10→recent10 episode mean length/return从
  `104.337/16.043`升到`1463.64/202.27`，但scheduled/public/terminal-overlap=
  `1,640,751/1,640,328/423`、launch=`1,439,028`、R03 physically-valid=`1,337,501`之后，
  selected contact仍是`0/1,439,028 launch`；landing因contact分母为0继续是`未测`。
- Isaac ACK0..2508共`246,644,736` transitions。first10→recent10 episode mean length/return从
  `87.502/12.956`升到`1441.81/177.55`；scheduled/public/overlap=`517,350/517,102/248`、
  accepted/rejected=`432,670/84,432`、playback=`424,472`、launch=`409,414`、R03 valid=
  `403,060`，但selected contact=`0/409,414 launch`，`377,513`次R06 settlement也没有common legal。

以上分母全部属于single slot0 forehand；其余action/side继续`未测`。接近1500 tick的survival说明balance
已经形成可用行为，但两端巨大launch分母下exact zero contact使hit成为明确negative，不再是“还没走到击球”。
common mimic/paddle income per transition也从first10的Mu`0.08751/0.06794`、Isaac
`0.08173/0.06768`降到recent10的`0.07508/0.06200`与`0.06289/0.06017`；收入只能说明经济，且当前wire
没有输出teacher-achieved物理误差，所以不能把mimic写成基本成功。

关节趋势同时反转。Mu actual-hard-edge累计`135,158,801/673,087,488=20.08%`，recent10=
`51.57%`、末段500-update窗=`58.42%`；qdes guard recent10=`52.52%`。Isaac按各自真实分母计算，
actual-hard-edge从first10=`1.843%`升到recent10=`3.366%`，近期policy/substep crossing均约`4.95%`。
qdes joint/forbidden terminal、Reward nonfinite/conservation、fact/attribution fault仍全0，故这是可信的策略/
plant行为，不是ledger损坏。

源码对账给出一个无需猜测的目标漏洞：FullMDP materialized reward只含14 lifecycle + 6 common mimic +
4 paddle prior；旧ActionBall中已经验证的`action_rate_l2`、`qdes_projection_penalty`、`qdes_limit_barrier`和
actual `joint_limit`没有进入该图。FullMDP termination又只对nonfinite qdes、fall/low/table终止，
actual-hard-edge特意只记账不Done。因此“允许policy从可恢复边缘学习”的设计前提——平滑动作和连续joint
cost——在现役图中实际缺席，policy可以用抖动/硬边换survival而不付目标成本。修复应把已有纯tensor代价
接进shared reward/PlantFacts，而不是新增阻断Gate。

第二个可审计缺口是paddle prior只存kernel income，没有实际误差；当前宽度`.70 m / 4 mps / pi rad / 1 rad`
适合远场capture，却没有证据证明达到球拍接触所需的精度。下一source先输出真实误差，再用一个固定
coarse+precision kernel同时保留capture梯度和物理精度，不新增actor observation；perfect-mimic几何闭合仍然
成立，所以不复活offset/reference猜修。

性能也随active business翻转：Mu recent10约`9.10 s/H48`，Isaac约`32.06 s/H48`（collection
`29.73 s`、learning`2.33 s`），不再等同于空任务为主的rate probe。Isaac drain在PPO边界逐CommitEntry
Python重放，并对event/row做`.item()`/`.tolist()`后把完整opportunity/shot/reset列表写进每条ACK；这不是
训练、聚合分层或durability所需的最小状态。下一性能刀应换成device compact counts + bounded fault sample，
保留optimizer/WAL/fsync/ACK独立边界；不能把整个owner graph原样移进C++或再加receipt。

V6可复算前缀：Mu `head -n 6847 evidence.jsonl` SHA-256=
`e27dc113e792c797ad83c04162c68e5039adda8cd9d5c54b791cb60e4159107f`；Isaac筛选前2509条
`HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=`整行 SHA-256=
`92c452506e2e9702cfc3f1c995d31022d1d2c292fc5854d51992df40fbdebff9`。这两条V6 root现已停止并保持
只读；V7现役exact checkout与两个fresh namespace同样禁止hot-patch、resume或复用。

### 10.8 V7 replacement合同：修学习目标，不加课程Gate

V7保留`balance → mimic → hit → landing`的自然重叠课程、四个due和Observation V3 `215/231`；没有把
“上一阶段成功”做成runtime admission。实现只修V6已测出的两个学习目标漏洞：

- `Reward28`在原14 lifecycle + 6 common mimic + 4 paddle之上恢复四项连续成本：31维
  `action_rate_l2=-.1`、processed-qdes soft barrier=`-10`、pre-clamp projection distance=`-1`、actual
  joint soft barrier=`-10`。双backend共用纯tensor kernel；actual-hard-edge继续作plant telemetry，不新增
  Done或所谓安全Gate。
- 四项paddle prior固定为50/50 precision-exp + coarse-Cauchy，position/velocity/face/long-axis的
  `precision/coarse`分别为`.075/.30 m`、`.50/2.0 mps`、`15/60 deg`、`10/40 deg`，weight=`1/1/1/.5`。
  同一个teacher-achieved误差producer同时服务Reward和telemetry；inactive不进分母，active nonfinite只降低
  finite denominator，不污染sum/sumsq。Isaac N1先报告aggregate，Mu由pinned slot0/forehand identity精确
  归属；其他action/side仍`未测`。

wire因新增误差字段显式升版：Isaac milestone schema8，Mu evidence/completion/summary=`10/5/6`。
本地聚焦回归=`606 passed, 34 skipped`，Pod1受影响文件逐进程隔离=`706 passed, 32 skipped`。Pod1 fixed
synthetic action-shaped tape拒绝把`cq_n_iters`从12降到8/4：两者均改变admission reason、selected identity
和target residual，故本轮保持12。

旧V6已经在精确进程身份复核后停止。V7 source=`1d33130ba07288918aa73d1323e1106303b7cad1`现以fresh Mu
namespace `fullmdp-a-h48-v7-mujoco-reward28-obsv3-1d33130b-20260825T180216Z-r2`和fresh Isaac namespace
`fullmdp-a-h48-v7-isaac-reward28-obsv3-1d33130b-20260825T180216Z`运行。启动窗Mu ACK0..16、Isaac
update0..61均为28项finite且conservation fault 0；paddle error字段存在，但因为仍是balance-only而active
分母为0，所以mimic仍`未测`，不是0误差。两端due/contact/landing也尚无分母；按自然重叠课程继续等待，
不能把启动窗变成success Gate。Mu recent10 mean/p50=`5.360/5.366 s/H48`，Isaac=
`8.423/8.380 s/H48`；active-strata wall和各阶段预注册判据仍未闭合。

<a id="109-v7结论固定lr在mimic曝光前耗尽"></a>

### 10.9 V7结论：固定LR在mimic曝光前耗尽

`observed_at=2026-08-26T19:27Z`冻结Mu ACK0..9815与Isaac update0..4459。两条仍是exact source=
`1d33130ba07288918aa73d1323e1106303b7cad1`、`diagnostic_unauthorized=true`，进程与root未热改：

- Mu共`964,952,064` transitions；累计launch=`1,748,621`、selected contact=`0/1,748,621`。最近50窗
  episode mean length/return=`1499.389/164.890`，`3,294/3,298`个terminal为timeout，故balance已基本
  形成。playback=`1,273,020` row，teacher-achieved position/velocity/face/long-axis mean=
  `.568/1.123/1.192/.898`（单位依次为m、m/s、rad、rad），远非contact尺度。
- Isaac共`438,435,840` transitions；累计launch=`430,393`、selected contact=`0/430,393`。最近50窗
  episode mean length/return=`494.914/68.581`，仍以base tilt结束，但比first10的`87.639/10.655`明显
  学会更多生存。playback=`1,047,071` row，四项真实误差mean=`.704/.919/1.087/1.063`。两端landing均因
  selected-contact分母为0写`未测`；全部分母只属于slot0 forehand，其余action/side也写`未测`。

Reward28的连续成本并非无效：Mu最近50 actual-hard-edge row约`0.155%`，显著低于启动窗；Isaac最近50按
`joint sample`约`3.6%`，仍阻断formal但没有qdes-forbidden/nonfinite/conservation/fact fault。失败不应再
归因为安全Gate缺失，也不能把aggregate survival当mimic成功。

真正的learner顺序错误由checkpoint直接闭合。Mu evidence显示LR约update500起反复贴`1e-5`下限，
update9787..9836的median仍为`1e-5`；Mu `model_9500.pt`与Isaac
`model_4000.diagnostic_nonresumable.pt`的optimizer param-group LR均精确为`1e-5`。这个时间点正与真实due/
playback开始打开重合：per-minibatch adaptive KL先在balance-only分布上耗尽步长，等paddle任务有大分母后
learner已几乎不能适应。Build4使用fixed/分离LR虽受warm-start混杂、不能直接复制行为结论，却进一步支持
“不要让早期阶段消费掉后期任务的学习率控制权”这一原则。

性能也不符合迭代要求：Mu最近50 mean/p50=`9.457/9.400 s/H48`；Isaac console最近50=
`25.638/25.505 s/H48`。启动balance-only的`5.36/8.42 s`不能代表active业务。Isaac每update又写约
`293 kB`完整shot/opportunity/reset JSON，累计`run.log`与WAL各约`1.3 GB`；这证明offline consumer分离是
真实结构债，但它不解释全部20+秒collection，不能把删日志夸成已证加速。

V8最小因果包因此只采用[`PPO V6`](../../DEFINITIONS.md#fullmdp-ppo-v6)：fixed LR`1e-4`以及
`512×H48×U100000/MB1/save2000`。总transition、24,576-row minibatch、总optimizer step与按transition
save cadence保持；刷新快4倍是显式算法取舍。Reward28、Observation V3、课程、plant、physics和
`cq_n_iters=12`不变。先验证LR不再被early balance吞掉及迭代wall；若未来固定窗paddle真实误差仍不降，
才把Build4的强direct-paddle权重作为独立下一轴，不在同一版本混入warm-start/replay/sigma/新obs或Stage。

首次V8 Isaac rate启动在environment construction前又暴露一个materialization遗漏：typed recipe的fixed
schedule正确给出`desired_kl=None`，shared `runner_kwargs()`仍执行`float(None)`。这证明recipe和launcher
identity测试不能代替最终RSL cfg构造。修复只落在唯一转换边界：null保持null，非null仍显式转float；失败
namespace永久封存，新source必须重跑两端exact rate，不能沿用修复前Mu receipt。

修复后的clean source=`0ad85ae1dfae13f617dc102a15bf99dba6b9ebf6`在Pod1目标矩阵为
`305 passed, 4 skipped`。两端production dry-run与61-update profiler-off rate均绑定learning SHA
`5a39b660…321f8`；Mu p50/p90=`3.796/3.999 s`、约`6,455 transitions/s`，Isaac=
`6.835/7.612 s`、约`3,598 transitions/s`。这证明iteration latency显著下降；Isaac仍未达到严格6秒，且
rate不代签学习质量。

replacement ready后只按exact PID/startticks/PGID/cwd/source/namespace停止V7，旧root保持只读且没有
伪造completion。V8 fresh namespace为
`fullmdp-a-h48-v8-mujoco-fixedlr512-0ad85ae1-20260826T203329Z`与
`fullmdp-a-h48-v8-isaac-fixedlr512-0ad85ae1-20260826T203329Z`。启动快照Mu ACK0..35、Isaac ACK0..51均
连续durable；Mu每条LR=`1e-4`，Isaac `model_0` optimizer param-group LR也精确为`1e-4`。两端episode
mean length仍约百步、未到首个due tick295，因此launch/contact/paddle-error/landing当前均是零eligible的
`未测`，符合balance起点，不能提前写成hit negative。下一结论只看预注册未来窗中的survival-to-due、
paddle真实误差与contact/landing分母。

## 11. 2026-08-29 current：V8/V9反例、R8出生修复与自然重叠课程

### 11.1 不是“Pod装坏了”，而是可执行训练配方与已验证范围被写错

V8两条进程继续保持只读；合计已观察`1,089,548`次physical launch而selected contact仍为0。这个结果不能
再归因为step不足。源码、artifact与runtime input交叉审计发现：teacher动作、physical-ready、dynamic-ready
artifact和最终YAML override并非同一动作/同一owner，Take058 teacher与Take061 ready被混入一个N1谱系；
同时bootstrap构造出的ready→frame0 bridge会被legacy override再次覆盖。另一个独立错误是catalog attachment
只安装motion/family/phase/sign，遗漏`clip_names_per_clip`，造成motion catalog `N=1`而action identity
catalog `N=0`。两处都是实现/结构错误，不是课程目标本身有问题。

当前固定Isaac执行边界`/opt/IsaacLab-8320e0be`、Kit Python、sealed RSL wheel和A3 USD在真实启动、训练与
durable ACK中工作；没有证据支持“Pod安装损坏”。`/workspace/IsaacLab`是另一个可变checkout，不能混进
当前run身份，也不能用它与Jiayi电脑的行为作未经控制的比较。若要裁决“机器人动作在sim里对不上”，最小
实验是同一Take061、同一joint order、同一physical birth/teacher frame0桥、同一policy clock的固定轨迹逐帧
对比；现在的零contact只能证伪旧配方，不能单独证明engine或Pod安装错。

“老师第0帧已经调过”也不能推出raw teleport可执行。现有nominal收据明确区分physical birth与teacher
frame0，只证明从physical-ready出发的桥在60 policy tick内名义可维持；收据末尾`waist_roll=-0.3205 rad`、
速度约`-1.19 rad/s`，距`-0.3491 rad`限位已很近。把这个1.2秒收据外推为tick295（5.9秒）前都只学
balance没有证据，旧fresh policy又普遍在tick77--82倾倒，所以它永远看不到任务。

### 11.2 最小修复与采用/延后/拒绝

采用：Take061的physical-ready→teacher frame0同源bridge成为唯一bootstrap owner；N1 catalog的motion与
action identity一次原子安装；首次due改为tick48，六次机会为`48/233/418/603/788/973`，185-tick间隔仍
覆盖task close、77-tick recovery和2-tick hidden gap。tick48恰是一轮H48 rollout且位于60-tick实测窗内，
所以fresh policy先收到完整balance梯度，随后自然获得mimic分母；mimic基本形成后同一任务链自然走到launch/
contact，hit形成后自然进入landing，不加入硬Stage或learned-success Gate。

延后：warm-start、replay、双LR、sigma`.19`、Build4 `14/14/5`权重、新observation和DR。本轮没有出现
same-observation/different-required-action反例，Observation V3已直接给出同clock的teacher-achieved最小
残差；继续加可推导量、raw ball/aim、action ID或部署不可观测oracle只会增加ABI与policy负担。

拒绝：放松base/joint/table Done以“让任务出现”、把task success变成启动/安全Gate、把60-tick收据冒充
长期稳定、继续等待V8、或用aggregate return遮掉contact零分母。真正安全边界仍是独立plant finite、joint/
table/contact几何、full-key/generation、optimizer成功与WAL/fsync/ACK；它们不替代清晰的状态机。

### 11.3 当前Pod证据与下一裁决

当前裁决已从下文V9早期前缀继续推进。R8用固定13项reserve重做physical birth，并在真实PhysX完成
`60 policy/240 physics/1.2 s`无terminal/hard-edge prefix；同一`512×H48×31` tape两端initial q/dq
逐位一致、qdes最大差`5.96e-8 rad`，但首个20 ms后q/dq已差`.00973/.89084`，故剩余首差属于backend
plant/controller/integrator/contact response，不是joint order、shared decoder或“Pod整体装坏”。R8
61-update rate中Mu p50/p90=`6.962/7.018 s`、contact=`0/6,523 launch`；Isaac=
`21.455/27.455 s`、due/playback/launch/contact=`14,221/13,555/461/0`。它证明自然课程入口已开且Isaac
迭代速度不合格，不授权把短窗零contact外推为长期不可学。

该response层已有具体源码差异：Isaac把qdes/Kp/Kd交给PhysX implicit drive在`.005 s`步内求解，Mu每个
`.001 s`子步显式计算、clamp PD torque；Isaac无量纲PhysX joint friction与Mu常值Coulomb
`frictionloss`又被源码明确标为未校准的逐数值移植。因此same qdes并不蕴含same torque/state response，
也不支持用Jiayi未附runtime收据的本机sim2sim结论宣告Pod安装损坏。

性能归因也已修正。逐轮未决row compaction虽保持业务输出逐位相等，却因小batch和动态launch使D05
question从旧dense的`18.16 s/12 updates`回归到`98.72 s`，已完整撤回。Pod exact CUDA确认
`virtual_ball.coarse_landing`的100-step Triton融合真实启用，`N=512`仅`.0526 ms/call`；真正热点是
Physical horizon对同一contact state先做`30×4=120`次eager reverse RK4 discovery，再重做最多`120`次
finalize，`[512,3,3]`直接实测=`202.2+54.1 ms`，与旧dense D05 question约`245 ms/call`匹配。
第一步trajectory cache/gather已在exact Kit/Torch2.7 RTX5090通过：`4,608`行全部admit、final batch全字段
逐bit相等，matched reference/cache总耗时=`101.263/50.534 ms`，其中finalize=`51.471/.317 ms`；retained
cache=`3.164 MiB`、该段peak增量=`7.321 MiB`。Pod逐文件隔离矩阵=`201 passed,5 skipped`，广合跑失败另归为
test-module alias/import-order污染。固定`30×4` discovery随后融合为一个Triton launch；actual `4,608`行、
边界/非有限/重复identity probe及production record都逐bit相等，reference/fused/production=
`49.591/.401/.416 ms`，约`124×` leaf speedup，fused peak增量=`3.551 MiB`，最终Pod隔离矩阵=
`203 passed,5 skipped`。保持H48、三轮、solver迭代、RNG、candidate identity、reason/fault/counter/safety和
CPU reference；下一步直接做完整D05 profile和profiler-off rate，不新增成功/安全Gate，也不用leaf倍数代签
iteration倍数。

2026-08-29的active V9只读窗已经不是“阶段未开”：Mu updates `5143..5192`的
due/reveal/launch/selected contact=`6,198/6,185/6,072/0`、episode mean=`202.53 tick`、wall median=
`6.475 s/H48`；Isaac updates `1543..1592`的due/accept/playback/launch/selected contact=
`8,441/8,437/8,420/7,477/0`、episode mean=`146.10 tick`、wall median=`22.10 s/H48`。两端都有高eligible分母而selected hit仍为0，应在R8精确候选通过后替换；V9 Mu使用
legacy plant，故这项negative不能移签R8。

source链为`651c305e`（原子action identity）→`cbf0aae3`（tick48）→`e3cbe9fc`（保持generic future-tick
fixture中性）。本地/Pod active-schedule聚焦矩阵为`53 passed`；更广矩阵的4个失败是旧Reward ordinal与fake
Motion epoch字段fixture不一致，未吞成production失败或伪装为PASS。tick295的真实Isaac 61-update rate probe
自然完成，p50/p90=`6.635/7.153 s/H48`且due=0，说明速度接近约6秒目标但课程仍死锁。

tick48真实Isaac到update4已有`due/selected/accepted=512/512/511`，Take061 action UID
`5527597793770800`为`511/511`，unknown 1 row按合同拒绝，playback started=`248`。完整61-update窗累计
`due/selected/accepted/rejected=18,432/18,432/18,419/13`、playback=`9,120`；`18,431`个terminal全部
base tilt，first10→last10 episode mean length=`79.310→81.916`，physical launch=0。paddle position
误差`.464→.496 m`而face/long-axis从`.995→.914`、`1.282→1.080 rad`，短窗趋势混合，不能称mimic成功。
active-task rate p50/p90=`10.470/13.863 s/H48`，因此空业务接近6秒不能代表现役iteration速度，task热路
必须继续profile和做结构减法。逐update对账进一步得到wall与due row数相关系数约`.771`：due=`512`的20轮
mean/p50=`12.10/12.35 s`，due=`0`的9轮=`6.85/6.85 s`，满任务cohort约多`5.25 s`，但相关性不能替代
callpoint attribution。

source `eefa5f5a`的自然退出profile诊断已在Pod1完成12/12轮，receipt SHA-256=
`a1dc0a17b8b698cfabd66f6075bd5846fec3786d67daa6868d8ce131cb9c05ed`；无signal/stop path，GPU2 lock随
child退出。profile-on median collection/learning=`8.042/.334 s`，inclusive reward/sim/observation=
`1.519/1.076/.529 s`，command-compute约`.941 s`；command结束到observation开始的gap中位`2.050 s`，
与due rows相关系数`.885`，而与terminal reset为`-.550`。因此下一刀只把既有profiler细分为D05 total、
question compose、preview、epoch settle和Motion publish；不新增runtime owner/Gate，也不把嵌套span相加。

第一次D05细分件`2a91062e`在首个environment step、PPO update前自然RC1：诊断wrapper替换了
`advance_action_ball_full_mdp_rows`这个被LeanRuntime按class function identity认证的方法，现有保护正确拒绝
`construction-bound component lacks exact advance_action_ball_full_mdp_rows`。该root冷封存且不复用；这不是
physics、asset或learner反例。successor不删除/放宽身份检查，只在认证后的原bound method调用外使用既有
opt-in runtime host-clock；内部prepare/question/preview/build/settle保持可逆计时，并新增“R05 advance与
Motion publish的class/bound identity不变”测试。Motion publish不再单独替换方法，其wall包含在D05 total
残差中。

successor exact source=`39569c492ccc94f76f83be1ff9c7451f4a2c6bc3`随后在Pod1 GPU2自然完成12/12轮，
无signal/stop且lock随child退出。namespace=
`fullmdp-a-h48-v9-isaac-profile12-d05-39569c49-20260828T1450Z`，receipt/log SHA-256=
`aa4892c3e39053479c17374bdba16b5842f3f8c2e7322152b376fa51c886b970`/
`f807955a3f35f247675ba463e66a7dab1033af8af212fc905889d5cdff51cb23`。D05 total/prepare/question
compose中位=`1.999/1.830/1.724 s`，12轮累计=`22.394/20.065/18.444 s`；question compose占D05
累计约`82.4%`。preview/build/epoch settle累计仅`.001/.092/.719 s`，不值得先动。下一候选是保留三轮
RNG和candidate identity、只对上一轮尚未出现success/fault的行继续solver/exact/Physical计算；但当前
round journal仍保存三轮reason/fault，且prepare会核全部round chronology，故在证明selected identity、
reason/counter/safety与可消费journal parity前不实现，也不直接降低100-step RK4或`cq_n_iters`。

为量化算法上限，successor仅在显式profile模式下、prepare已经完成之后对`rng_advance_mask`真实construction
行的`rounds_attempted`做一次device→host histogram，并把`>=1/2/3`行数写进profile segment的`env_count`；
它不进入生产ACK、D05 journal或任何Gate，
含该同步的wall一律不作速度证据。若第二/三轮消费者很少，下一步仍须处理现有journal保存全三轮reason/fault、
prepare核全round chronology这一语义事实；不能把未求解行伪装成已求解的producer结果。

exact source=`ad29312ae31ce017c37a04780398e819990bf3b5`的round-density件随后自然完成12/12轮，
receipt/log SHA-256=`950a3d07…f6e9`/`26019698…c70`。初版profile计数包含每次full-N transaction的
inactive行；本窗74次compose、due=`3,584`、terminal overlap=`0`，所以从三档raw
`37,888/34,594/34,326`逐档扣除`74×512-3,584=34,304`，真实construction attempted=
`3,584/290/22`。即91.9%在第一轮结束、99.4%在第二轮结束；固定dense三轮计算`10,752` env-round，
实际消费`3,896`，未决行incremental compose最多可少约`63.8%`数值row-round。该结果采用incremental
unresolved-row作为下一算法候选，明确拒绝把三轮直接降成一轮（会丢约8.1%现有后轮题）。实现前必须让
D05把previous-cell作为同owner ephemeral context交给composer，并让未尝试suffix在journal/chronology中
有诚实语义；selected identity、最终reason、RNG/draw/generation counters、producer fault和safety结果须与
dense fixed-tape逐位相同。profile计数器已修为直接过滤`rng_advance_mask`，后续不再靠扣除法。

修正计数后的exact source=`34cd7af8717b3463c519f306072e90f632f92344`又以fresh namespace
`fullmdp-a-h48-v9-isaac-profile12-rounds2-34cd7af8-20260828T1545Z`在Pod1 GPU2自然完成12/12轮。
probe/run/runtime-receipt SHA-256=`8dee501c475e1e84e0e0bbd8518e6e69eeb3ee624011f83d36c8da5465010fc1`/
`79896bc62fda9969be02ecbfe363121e413ad79cfcc2b2312cb445920ae75a4b`/
`5c588f65d71053a6d56b29b1652b93a27419a197758b66bbea0ffd530a52a9c4`。这次不做inactive减法，直接得到
round1/2/3 active attempted=`3,584/290/22`，精确复现上一段结论。ACK0..11对账为
due/selected/accepted/rejected=`3,584/3,584/3,581/3`、terminal overlap=`0`、playback=`1,642`、
launch=`0`；所有`3,584`个terminal均为base tilt。该窗继续是`diagnostic_unauthorized=true`、profile-on、
formal evidence=false，只授权热点密度事实，不授权速度或学习。

同源码MuJoCo 61-update窗也已自然完成，p50/p90=`6.644/6.854 s/H48`，scheduled/reveal=
`10,861/10,860`、launch=`6,658`、missed launch=0、R03 present/physically-valid=`5,107/5,107`，
selected contact=`0/6,658`、landing=0。episode mean first10→last10=`135.78→139.98`；paddle position/
velocity误差`.373→.378 m`、`1.106→1.149 m/s`没有改善，face/long-axis仅微降`1.189→1.176`、
`1.128→1.112 rad`，仍不能称mimic成功。首个fresh Mu root因ignored EPA48/RSL3 bundle未恢复在首ACK前
fail-closed；按`setup_local_sync`固定三文件SHA恢复后r2成功，故这是clean-checkout同步缺口，不解释后续
学习行为。

两端已证明同一任务合同可执行，却没有证明相同plant行为：Isaac约82 tick、全base tilt且launch=0；Mu约
140 tick、table/tilt混合且launch非零。该分叉既不能直接归咎“Pod安装坏了”，也不能以“原生后端不同”豁免；
下一固定实验必须从同一physical-ready、同一joint order、同一q/dq与同一固定31-D action tape逐policy tick
比较q、dq、base、racket和terminal first divergence。当前只能说balance→mimic交接实现闭合；mimic、hit、
landing与physics parity均未闭合。该有限窗当时不单独授权长期replacement；随后fixed-action根因闭合与
双fresh发射事实见§11.2，V8现已停止并只读保留。所有结果继续`diagnostic_unauthorized=true`。

`observed_at=2026-08-28T07:12Z`的旧V8只读长跑已进一步坐实negative，而不是启动balance-only：
Mu ACK0..20563累计`505,380,864` transitions，scheduled/launch/R03/contact=
`1,118,460/855,183/585,784/0`；Isaac ACK0..7171累计`176,259,072` transitions，
due/accepted/launch/R03/contact=`345,310/288,403/234,365/221,036/0`。合计selected contact=
`0/1,089,548`，common-legal与landing仍因
selected contact为零而`未测`。recent50 episode mean已经分别为`886.74/1,308.91 tick`，说明survival确实
学习；但Mu/Isaac recent50四项paddle误差仍为`.473/.974/.949/.435`与`.416/.811/1.180/.750`，没有形成接触尺度
mimic。这份结果只证伪混源/覆盖的V8谱系，不能反向证伪V9的Take061单一bootstrap与tick48修复。

为裁决“起始动作错是否就是Pod/sim错”，本轮不增加新的成功Gate，而是实现同一tracked
`512×H48×31` raw action tape的Isaac/Mu first-divergence诊断。两端共同记录reset后的physical root、
runtime joint-order q/dq、实际拍面几何，以及48 tick逐步状态/terminal；比较器把initial mismatch与
post-step divergence分开，并只给exact首差和数值包络。若q/dq在tick0前已不同，优先查asset/reset/joint
order；若初态一致而第一动作后分叉，才进入decoder/plant/backend。host targeted=`41 passed`，exact Pod
首轮记录已由clean source=`981327de58d5c72b3ffcf3c3ebf1bfe981ce0292`自然生成；两端joint order和tape
SHA完全相同，首差却在`initial_joint_pos[0,12]`。初始joint max/mean absolute差=`1.5199/.2427 rad`，
root position=`.1778/.1101 m`，racket position=`.5376/.3561 m`；Isaac root local为
`[0,0,1.0684]`且右肩pitch/elbow为`.3/.8`，Mu root local为`[.15259,-.17777,1.0684]`且右肩pitch为
`-1.2199`，后者精确对应Take061 physical-ready。tick7开始done mismatch，Mu总数`3,072=512×6`、Isaac为0。

源码因果链已闭合：RSL wrapper构造时确实调用一次canonical reset，所以不是“完全没reset”；真正错误是唯一
reset Event明确写`Articulation.default_joint_pos/default_root_state`，从未消费Motion已经绑定的
dynamic-ready physical birth。此前catalog、teacher bridge和actor-head修复只闭合老师、课程内部状态与q_des，
并未成为Isaac plant birth；这就是用户看到“老师第0帧调过但sim起始动作仍不对”的实现根因，不是Pod安装坏。

首版候选曾在wrapper前再调用一次reset；exact Kit在第二次reset处因没有terminal事实正确fail-closed，证明
现有wrapper已经是genesis reset owner。失败root保留、不复用，该重复调用已删除。最终修复不添加下游Gate：
reset Event仍是唯一sim writer，但只读取Motion已验证的`dynamic_ready.physical_ready`窄projection并写root/q/dq。
teacher frame0继续只是mimic authority，`default_joint_pos`继续只是affine action decoder offset，二者都不得
冒充physical birth。修后clean `179148e3`的Isaac/Mu记录已自然完成：initial 31-D q/dq逐位相同，root
position最大差`4.1e-7 m`、拍心position最大差`0.467 mm`，done/time-out `0` mismatch；所以初态实现错误已闭合。

但这次反例也审出了诊断本身的错问法：v1 tape围绕raw action `0`取`±.02`，而生产fresh actor mean是
Take061 dynamic-ready normalized action，最大绝对值`16.3001`、平均绝对值`1.6631`。因此v1第一步把两端都从
physical-ready拉向asset-default q_des，`512/512`行在tick`7/15/23/31/39/47`同步终止并reset；它能证明birth
修复，不能代表fresh policy/teacher mimic的backend动力学。v2不加readiness Gate，只把同一个tracked
Take061 actor mean作为tape center、叠加`±.02`确定性扰动，并要求两个backend的live bootstrap action与center
逐位一致。clean `3343fe90` centered record中两端48 tick均无terminal，initial q/dq仍exact；但Isaac相对
初态的joint/root/racket最大漂移为`.349 rad/.092 m/.170 m`，Mu只有`.012 rad/.008 m/.010 m`，跨端joint
最大差从tick0 `.0059 rad`扩大到tick47 `.3480 rad`。v3 record进一步证明tick0--34实际executable
`joint_qdes`跨端最大只差`5.96e-8 rad`，但q在首个20 ms步已经分叉；tick35后Isaac腰滚先碰hard-inner
才让guard改写q_des。故decoder/动作顺序已排除，差异在backend plant/controller response。

这也翻转旧“nominal hold PASS”的人话：原收据60 tick末已经是`waist_roll=-.3205 rad, dq=-1.19 rad/s`，
PASS只表示尚未触发terminal，不是“保持在ready”。它足以证明tick48前没有硬终止，却不能充当balance/mimic
readiness Gate。V9允许两端分别从该prefix学习balance并在tick48自然公开mimic；因此physics parity不再
冒充长训启动门，同时继续禁止用该差异声称Pod安装损坏或transfer成立。

### 11.2 V9双fresh长期启动与首个未来前缀

clean exact source=`eb57233b4522d527455a0cbd7c547eb2ec49a68c`已发射MuJoCo/Isaac双fresh长期
replacement；namespace分别为`fullmdp-a-h48-v9-mujoco-genesisfix-eb57233b-20260828T093350Z`与
`fullmdp-a-h48-v9-isaac-genesisfix-eb57233b-20260828T094243Z`。两端先取得连续durable ACK，随后旧V8才按
exact身份停止；旧root/checkpoint保留、未伪造completion。

`observed_at=2026-08-28T09:48:33Z`的首个未来前缀中，Mu ACK0..83=`2,064,384` transitions，首10→近10
episode mean=`137.31→141.96 tick`，近10 launch/R03/contact=`1,210/855/0`；Isaac ACK0..18=
`466,944` transitions，episode mean=`67.34→73.55 tick`，近10 D05 due=`3,414`，但physical
launch/R03/contact=`0/0/0`。两端reward finite、conservation与fact/attribution fault均为0。Mu近10
wall mean=`6.57 s/H48`，Isaac=`16.76 s/H48`，后者仍不满足可迭代速度方向。

阶段解释保持预注册顺序而不设硬Stage：Mu已有早期hit分母但`0/launch`，Isaac还主要在balance，因未活到
physical launch而hit/landing为`未测`；两端mimic真实误差方向尚混合，均不能称基本成功。该前缀不能被用来
事后发明阈值，也不授权physics parity、promotion、部署或真机安全。
