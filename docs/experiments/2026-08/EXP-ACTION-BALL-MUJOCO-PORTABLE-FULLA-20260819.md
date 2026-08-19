# EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819

> 问题：portable MuJoCo怎样在不复制Isaac owner graph的前提下，逐纵切片消费同一slot0题目、measured teacher、真实事件与Reward20，直到可以做`4096×25000`长跑？
>
> 人类负责人：Franco
> 执行者：Codex
> 状态：`action-affine-and-ledger-host-validated / qdes-guard-divergent / live-25k-not-started`
> 证据等级：E1源码与host反例

## 1. 采用、延后、拒绝

采用：

- fixed action严格沿用现役Isaac cadence的slot0/UID `6907688916670928`，不把已cold-load的73行bank冒充73动作训练。
- cold load一次校验manifest/motion SHA、31-joint order、tracked-body order、mount sign与measured timing；hot step只消费device tensor。
- question由slot0 manifest center、live base yaw、shared Physical reverse-integration和integer control tick生成；world origin只在qpos launch写入时恢复。
- fixed cadence沿用shared 30 s / 1500-control-tick schedule：due opportunity固定为
  `2,295,588,881,1174,1467`（`2 + 293k`）；每行在due时按live readiness得到
  `ACCEPT`或`DEFER`。`DEFER`是zero-write，不在tick 3补试，下一机会仍是下一个冻结due。
- true Gym reset写`runtime_plant.default_joint_pos_rad`、配置default root加env origin、零joint/root
  velocity和零current/previous action history；`take061/q_ready`只保留输入provenance，不是physical birth authority。
- pre-swing HOLD期间public joint teacher是runtime default、joint velocity为0；body reference与R07 target
  使用measured frame0。只有`active_motion_s > 0`后才公开measured sampler，即使rounded frame ordinal仍为0。
- raw action affine的proposal、runtime/schema-2关节顺序、scale与
  `runtime_plant.default_joint_pos_rad` offset已与active Isaac闭合；`take061/physical-ready`只保留
  provenance，不是reset或affine-offset authority。
  executable q-des guard仍明确`DIVERGENT_DECLARED`：MuJoCo机械hard clamp没有复制Isaac的
  soft-inset与state-dependent brake，因此只授权MuJoCo-only诊断，不授权transfer/matched结论。
- 直接目标是同一upstream RSL-RL 3.1.2进程`4096×24×25000 = 2,457,600,000`
  transitions；`1000`只是只读早期节点，不停车。
- thin update ledger只在optimizer boundary事务性记一条ACK；rate/window/per-side表全部
  由独立offline consumer计算。

延后：

- contact census合并、scratch/reward预分配和20-substep CUDA graph；先闭正确语义，再做4096
  profiler-off同源吞吐优化。
- C family与per-side非零分母；当前这一代只训slot0 forehand，backhand必须诚实为0，
  不为了表格对称额外发一条run。

拒绝：

- midpoint serve、`x+vt-0.5gt²`、`normal=-incoming`与generic-contact冒充selected-rubber。
- 把true reset改成take061或take058 frame0；fresh FullMDP birth authority是runtime default plant。
- 把diagnostic qdes bridge写入Motion observation或Reward teacher。它只表达历史命令consumer递推，
  不是fresh Motion public teacher或动作执行authority。
- 用WAIT `N=2×learn(1)`或native 114/114-D速度冒充portable Full-A长跑。
- 要求未训练zero-action policy在真实rollout中活过R07 age `10..77`才允许开训。
  68格连续性仍是确定性host lifecycle反例，但不是随机策略表现门。
- 保留一套额外keepout witness生产代码。现有exact keepout terminal不删；table/fall/bit16
  在训练中记telemetry，只有独立确定性证据证明frame/guard错误时才升为环境阻塞。
- 在hot runner内计算窗口表、成功率或每update向stdout打大JSON；这些派生项留给
  独立consumer。

## 2. 本轮纵切片

新增dependency-light `mujoco_full_mdp_portable_question.py`：

- 读取fixed slot0的sealed measured motion，验证`96×31` joint、tracked body、50 Hz、strike frame52、joint-order contract和mount sign；
- 从live env-local base pose构造action center，调用与Isaac同源的`physical_ball.back_integrate_incoming`做discover-then-integer-tick reverse flight；
- 生成45-D task与13-D launch state，并让contact tick严格命中measured strike frame；
- 提供无production consumer的`step_diagnostic_split_ready_qdes_bridge`，仅逐字表达Isaac oracle命令递推，不导出body/teacher接口。

MuJoCo Full-A env现在：

- reveal只安装env-local task/launch；prelaunch球保持park，launch才加对应env origin；
- phase只允许`IDLE=0 / REVEAL=2 / LAUNCH=5 / OUTCOME=6 / RETIRED=8`；
- true Gym reset使用default plant/zero action history；HOLD joint teacher保持default/zero velocity，body/R07
  独立使用measured frame0；
- natural recovery success或window-timeout发布`shot_retired`并进入phase8；它不发Gym done，不改robot、
  current/previous action、episode length或`reset_generation`，直到后续due真正ACCEPT才替换shot；
- 只有真实Gym done才发布`selected_reset`并使`reset_generation`恰增1；
- R03 expected source延到真实contact tick，不再在reveal后第一拍伪造strike fact。
- R06/R07、Reward11--13、shot-retire/Gym-reset和20-term守恒已进production extras，不再
  以`not_produced`应付runner；它们的真实次数可为0，但schema、事件和reset generation
  必须一致。
- thin ledger在device上累计26个lifecycle/event、5个terminal bit、classification、Reward20和
  reset/identity/integrity计数；每update只用一次`torch.cat(...).cpu()`把该固定向量送到host。
- 第26个`completed_action_epoch`不是边际推导：只有同一env行保留的launch、selected contact、R03、
  R06、无fault的68格R07与自然RETIRE同时闭合才发布；跨env拼接这些里程碑必须保持业务未完成。
- optimizer前要求24步、storage reward/return/advantage finite、done只能为`0/1`、timeout与bit1
  一致、done与terminal-or-resolved-table一致、`selected_reset == done`且generation delta只等于done，并验
  Reward20守恒和独立钉住的slot0/UID/sign/family。optimizer前只prepare冻结payload；只有upstream
  optimizer成功返回后才写snapshot并append+fsync ACK；optimizer/write/fsync失败不记假ACK。
- stock RSL serializer在update `0,1000,...,24000,24999`留26份no-clobber快照；它们均
  `diagnostic_unauthorized=true`、`checkpoint_authority=false`、`resume_authority=false`。

## 3. 反例与结果

question/teacher历史focused host=`22 passed, 7 skipped`。当前MuJoCo env/action/outcome卷为
`59 passed, 7 skipped`，alignment当前两轴反例=`14 passed, 21 deselected`；runner/ledger/consumer集成为
`96 passed, 1 skipped`，其中production writer生成的真实prefix直接由独立consumer读取。另有
`py_compile`与`git diff --check`通过。反例覆盖：

- 改action center/incoming center会改变task与launch，旧midpoint shortcut不能假绿；
- 两个env origin只影响world qpos写入，不污染env-local task；
- true reset的joint/root birth与action history逐项来自default/zero authority，不再从take061推导；
  HOLD joint teacher仍为default/zero velocity，而body/R07 measured-frame0通道保持独立；
- 任意中间bridge teacher会同时改变actor teacher字段和dense body reward，测试直接拒绝；
- contact tick97精确采到strike frame52；true-reset clear、phase8 retain/defer、后续ACCEPT
  与peer preservation逐行验证。
- 历史raw `[-4,+4]` clip使slot0 frame0的五个关节需求
  `-4.074456/16.459160/-5.387479/6.098073/-31.731944`结构上不可达，而对应decoded
  q-des均仍在现有joint envelope内；去掉raw clip后该确定性不可学阻塞关闭。
- timeout错位、`done=2`、shot-retire误增generation、Gym done不增generation、common step卡住、Reward20不守恒、
  同源UID自证、optimizer/fsync失败与JSONL重复/缺口均会拒绝；event次数为0、
  table/fall多或recovery failure多不会被误拒。
- 独立consumer要求完整25k文件有25000行、index `0..24999`无缺口/重复、每行
  `4096/24/98304`与最终累计精确，再另验26份有限model/optimizer快照。零分母的rate输出
  `null/未测`，不输出假0%。
- consumer分别输出`engineering_run_complete`与slot0 `business_chain_complete`。25000个ACK可关闭
  工程长跑，但业务事件全零时仍不得称slot0链出现；又73动作、双侧与科学窗口
  报告未闭合，本代`full_a_complete`固定为`false`。

## 4. 仍未闭合

1. 上述host runner/ledger/consumer尚未在exact Pod的同一fresh checkout、RSL-RL 3.1.2隔离安装和
   真实MuJoCo-Warp GPU上跑过调用点；host PASS不代签GPU。
2. portable `4096×25000`尚未实际启动；因而25000 ACK、26份快照、终点consumer和学习
   趋势均为`未测`。
3. 当前只是slot0 forehand A；C family、backhand分母、第二seed、promotion/export/deploy均未由
   本工程件授权。

这些是真实剩余的证据边界。zero-action age16/bit16、普通table/fall、零contact或低
recovery率只是未训练策略的telemetry，不再作为发车阻塞；现有keepout termination和证据完整性
检查仍保留。因此本轮是host完成的长跑工程件，不是MuJoCo Full-A已长跑。
