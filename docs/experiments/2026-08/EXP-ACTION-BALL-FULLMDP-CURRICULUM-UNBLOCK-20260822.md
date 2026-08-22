# EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822

> 问题：为什么两条 [FullMDP](../../DEFINITIONS.md#fullmdp-ppo-v2) H48 长跑已经学会站立，却没有进入可学习的 mimic/击球链；怎样用最少状态和单一真源修复后 fresh 重启？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`branch candidate / host+exact Pod focused PASS / dual fresh successors RUNNING`
> 证据边界：`diagnostic_unauthorized=true`；本记录不授权 resume、promotion、export、部署或真机。

## 1. 旧 live 事实与裁决

- Isaac 到约 update 500 已从早期倾倒变为整段 1500-tick timeout，说明 balance 已学出；但 update 305
  起每个 due row 都是 CENSOR，task Reward/击球/落点分母保持零。这不是“学习还慢”，而是账与实际不一致。
- 最早 CENSOR 在 update 116，28 行恰是同一 control-step 的整批 due。第一个活过 second due tick 295 的
  row 触发 chronology bit 50，随后同批 fault fanout；此前终止 row 都没活到 295，CENSOR 为零。
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
- `295` 只由 portable catalog 定义，Isaac cadence 消费同一常量；后续 due 为
  `295, 588, 881, 1174, 1467`。
- D05 的 cadence cursor 按所有 settled due 前进；target generation、selected cell 与现役
  task/outcome/ball identity 仍只按 ACCEPT 前进。checkpoint 合同允许“cadence 已消费但尚无 accepted task”，
  并统一要求 `outcome_shot_index = scheduled_ordinal + 1`。
- hidden joint/body 都取同一 reset-ready tuple；reveal 后 joint/body 原子切到 action frame 0。旧证据已经表明
  人为 35-step 插值不是存活主因，因此不再增加 blend 状态机。
- body orientation 使用同一误差的 fine `.4 rad` 与 coarse `1.0 rad` 等权核；它只在 reference 修正后
  恢复大误差区梯度，不用放宽阈值掩盖错 target。
- R07 保留 recovery Reward、critic plant state和可读 telemetry，但退出 fresh reveal、训练 liveness 与
  actor phase authority。nonfinite/overflow、joint envelope、跌倒/过低、真实桌碰等 plant fail-stop不变。
- PPO 保持 H48、GAE lambda `.98`、E5/MB8、fresh `sigma=.02`、log-std、entropy `.01`、adaptive KL；
  没有 actual-KL 证据前不叠加 std decay、clamp 或 LR 改动。
- 不新增 actor observation。现有 203-D 已包含可部署的 root/velocity、teacher、task residual、phase 与倒计时；
  R07 误差重复喂给 actor只会扩大合同并制造 shortcut。

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
