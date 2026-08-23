# Run Training

Status: Draft

For GPU runs on the shared team pod (3× RTX 5090, per-user clones and folders), see
[run_on_runpod.md](run_on_runpod.md); the commands below are identical there after
`source /workspace/<name>/env.sh`.

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

<a id="task-first-prelaunch"></a>
<a id="action-ball-prelaunch"></a>

### Action-conditioned Ball-first prelaunch（2026-07-27，尚未授权）

候选 executor 的顺序是
`action → time-to-contact/incoming ball/base/aim → fixed-action task+teacher-rate solve → atomic install`，
完整定义见[训练合同](../interfaces/action_conditioned_ball_first_contract.md)。训练期 selector 关闭；
旧 `task-first` 只保留为历史消融。fresh upper/no-move N5 的唯一正式入口是
[V3 双 GPU no-clobber 工序](run_action_ball_curriculum_no_clobber.md)：它固定 exact N5 顺序，
GPU0 只跑 trainer、GPU1 只跑 frozen evaluator，并要求 `smoke → canary → long` 的签名前序收据。
当前 code-owned trust sets 与真实动作/球路/桌碰证据尚未闭合，因此正确结果仍应是
`ACTION_BALL_LAUNCH_REFUSED`；本节不提供可绕过 V3 launcher 的 raw 长训命令。

`plan` 是只静态复核 spec、打印 exact argv 与 claim 的子命令；`launch` 是拿齐双 GPU lifetime
lock 后重验并实际启动的子命令。两个 flag 的定义和完整命令见
[V3 工序](run_action_ball_curriculum_no_clobber.md#先跑-plan)；`no-clobber` 表示已有 namespace
永不覆盖，见[术语](../DEFINITIONS.md#no-clobber)。

首轮 fresh N5 的 plant 固定为**平地 + no-move**。2026-07-29 的
[零均值 rough patch 候选](../experiments/2026-07/EXP-ROUGH-GROUND-FRICTION-FIX-20260729.md)
只有 host E1 证据；不要在 `extra_overrides` 中临时加入摩擦或 rough 键。corrected-friction 先过
2-env Isaac 材质 readback，rough/move 再过 clone/contact/raycast/seed/初始穿插与 4096-env
性能门，并作为新的内容寻址 scientific recipe fresh-from-random 发射。

compose 后必须同时满足：

- `racket.target_mode=action_ball`；
- `racket.action_ball_manifest_path` 与 `racket.action_ball_manifest_sha256`：操作者预先钉住 exact
  启动清单路径和文件字节 SHA-256；二者缺一即拒绝；
- `racket.action_ball_policy_contract_sha256`：观测/动作、runner、Reward 与 evaluator recipe 的固定
  identity；它不是会随 PPO 更新的 checkpoint SHA；
- `racket.action_ball_fixed_direction=true`；solver 不能改动作或在内部重采样；
- 旧 CQ producer/distribution/buffer/seed 必须为空；只允许 fixed-action solver 的
  `racket.cq_overdraw / cq_n_iters / cq_tol_m / cq_speed_budget / cq_max_redraw_rounds`，并由实际
  solver payload SHA 认证。每个额外 proposal 都保留在 `P` 与 reject-reason ledger；
- compose 输入 `racket.clip_names` 与 manifest `action_order` 完全相同；`train.py` 再由它派生
  runtime `racket_target.clip_names_per_clip`。motion 文件数、顺序与逐文件 SHA 必须完全相同；
- fresh N1 训练使用
  `task.actor_obs_contract=action_ball_table_pose_twist_heading_task_teacher_start_v2`；actor 为
  frame-consistent
  `hitter_footwork(177) + table pose(9) + base linear velocity(3) +
  face/rho(4) + time_to_teacher_start_s(1)`，固定 194-D，不含 `action_one_hot`。
  `time_to_teacher_start_s=max(pre_swing_wait_s-task_age_s,0)` 直接读取同一 Motion
  phase governor，避免 policy 重建老师何时离开 ready。UID/slot 只留在 sampler/solver/
  curriculum/receipt；formal N5/N73 发射前必须切到固定宽 teacher-trajectory/ball/task/
  validity/history successor 并过 N2/N3 共享策略验证，不另加 motion ID/intent。旧
  `action_ball_table_pose_twist_heading_task_n<N>`（`193+N`）与历史
  teacher-start `194+N` one-hot 合同只为已运行 checkpoint/receipt 保留解析，不能
  exact-resume 到新合同；
- `motion.balanced_clip_sampling=true` 及
  `motion.balanced_clip_sampling_seed=<内容绑定的整数 seed>`，使任意前缀的逐动作样本数最多差一；
- action manifest、sampler、solver profile、physics profile、motion admission、policy contract 和
  effective Reward receipt 都进入 training hard contract；solver/physics SHA 必须由实际 runtime
  canonical payload 重算，不能接受 cfg 或 manifest 自报；
- 首轮 `mobility_mode=no_move`；birth receipt 只在 true reset 冻结 action/base spawn，per-swing task
  receipt 再携带 base goal、`time_to_contact`、`teacher_rate`、scaled `t_hit/t_cycle` 和 ready wait。
  WRAP 不写 root；
- `level=0` 使用每动作、每侧 profile 的 non-zero initial std；exact arm catalog 总数为 32，
  `no_move` 禁四个 base-travel arm 后有效 28 个。课程按 per-arm marginal → joint `rho` 扩张，
  不把多个单臂 10% 直接做乘积分布；
- `teacher_rate=required_racket_site_speed/reference_racket_site_speed` 必须在动作认证范围内且不得
  clip。Motion 先保持 ready，再满足
  `pre_swing_wait + scaled_t_hit == time_to_contact`；额外等待必须在 `[0,1] s`，且
  `pre_swing_wait + scaled_t_cycle + policy_dt` 不得超出 episode horizon；其中额外一个
  `policy_dt=sim.dt*decimation` tick 用于 attempt 闭合；
- rolling-100 只选择下一个候选 arm 并强制防饥饿探索。每次 frozen evaluation 必须完整生成
  `320` 个 canary proposals（至少 `256` 个 safe-closed）和互斥的 `960` 个 heldout proposals
  （至少 `768` 个 safe-closed），三类样本都固定 `20% center / 60% interior / 20% frontier`；
  canary 只作前门，只有 heldout 连同其余统计/安全门才可改变正式 frontier；
- action-ball 的 `runner.learn(..., init_at_random_ep_len=False)` 是 hard contract；不得用随机首
  episode 截短来做相位去同步。
- action-ball formal resume 另需独立预钉的 resume receipt，绑定 raw checkpoint SHA、
  shared action-ball state root、run/training contract 与 iteration；runner 禁止在同一次 load
  现算 expected 值。缺 pin 时只允许 fresh launch/diagnostic load。

下列残留会改变问题，必须在 `gym.make` 前失败：

- question/CQ/exam bank、HER achieved-target mix，以及任一第二 task producer；
- shadow/physical ball driver；analytic virtual ball只允许消费 action-ball receipt 安装的同一球；
- planner revision、mid-swing resample、clip switch；
- target delay、jitter、white/AR(1) noise、dropout、per-swing bias；
- legacy/random motion retiming、per-clip speed scaling 或 event timing；唯一允许的时间律是 exact
  task receipt 内、由动作认证 envelope 约束的 teacher rate 与 ready wait；
- 旧 station tail、缺 face observation、没有 N-way action identity，或只有 metadata 没有
  code-rooted motion admission。

这里 `AR(1)` 是一阶自回归噪声；首轮将它关闭。在线 rollout 可累计训练 ledger，但 curriculum
晋级只能消费 frozen evaluator 生成的 `policy_checkpoint_sha256 + policy_generation` 一次性窗口；
缺 evaluator authority 时必须 hold，不能拿当前 actor 的自报 SHA 冒充。训练只在
[action-ball 实验](../experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md)把 exact
manifest、动作证书、host union、Pod 两迭代 smoke、WARN 摘要和 receipt 全部落账后开放。

V3 runtime 还要求一份不修改共享 Isaac venv 的 per-run overlay、可现场重算的 runtime inventory
receipt，以及 trainer 在 `env.pkl/agent.pkl` 和 runtime identity 落盘后发布的 bootstrap receipt。
trainer 自然结束后，supervisor 先精确停止 GPU1 evaluator，再在仍持双锁时用 GPU0 自动启动独立
real Isaac zero-step restore→save→restore verifier；只有 policy、optimizer、两个 normalizer、
RNG 与完整 ActionBall state 无损恢复，才写 terminal，并允许 stage evaluator 签
`exact_resume_passed=true`。不要在正式 stage 手工重复 verifier；完整顺序见
[V3 工序](run_action_ball_curriculum_no_clobber.md#训练结束后的-exact-resume-与阶段签名)。

#### FullMDP Phase-A runtime边界（2026-08-20）

现役单动作family A/C的FullMDP训练只能使用
[`Phase-A direct-lean runtime`](../DEFINITIONS.md#fullmdp-phase-a-direct-lean)。没有可选formal runtime mode，
也不得调用已退休的partial installer或逐leaf getter；旧`action_ball_full_mdp_runtime_owner.py`及其专属
适配面已由Phase-B branch candidate物理退役，不是发射、安全、checkpoint或resume权威；详细证据见
[FullMDP hot-path实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-HOTPATH-20260819.md)。运行入口应继续验证真实独立边界：
source/API pin、环境lease与construction seal、exact component identity、selected-reset
generation/overflow、finite与sticky poison/fail-stop；FullMDP全局checkpoint/restore仍是`R10 HOLD`，
所以现役A/C继续fresh-only。

不要为了替代已退休的formal路径新增同一writer自造的依赖有向无环图（DAG）/hash/receipt门；那类
self-proof没有独立事实源。这份Phase-A合同没有新增运行命令，也不表示已达到约`6 s/update`的方向目标、学会回球、
获得formal authority或完成任何Gate。性能结论仍须使用exact Pod、profiler-off、matched-strata墙钟证据。

#### FullMDP PPO V3执行配方（2026-08-23 branch candidate）

FullMDP A/C与portable MuJoCo只能消费code-owned typed recipe：`num_steps_per_env=48`、
`max_iterations=12500`、`save_interval=500`、`num_learning_epochs=5`、`num_mini_batches=8`、
`gamma=.99`、`lambda=.98`、fresh `init_noise_std=.02`、learned `log_std`、`entropy_coef=0`。不要在Hydra argv、task YAML或MuJoCo CLI复制/覆盖这些值；FullMDP冲突
override必须在Kit启动前失败。旧H24 checkpoint与本配方的policy refresh/GAE/update grouping不同，不能resume。

V2长跑的永久`entropy_coef=.01`在RSL-RL 3.1.2中对31维`log_std`各自产生固定`-.01` loss梯度；真实
MuJoCo V2 checkpoint的mean std已从`.02`单调涨到大于`1`并伴随balance退化。V3只删除这个已测因果项，
保留advantage对learned std和adaptive-KL的真实控制；不叠加std clamp、decay或课程状态机。V2 checkpoint
已经带有放大的std，故同样禁止resume，必须fresh。

learning SHA只描述影响学习的配方；execution SHA还绑定iteration budget与save cadence。
[fullmdp-a-h48-v5-*](../DEFINITIONS.md#fullmdp-optimization-lineage-v5)最终wire只接受exact
`evidence/update schema=6`、`completion schema=5`、`consumer summary schema=5`；consumer必须按exact schema
分流，旧V3/V4 evidence不得拿新consumer伪装兼容。两条live run的V5 source
`39f9481950a660e198dedac7fd402806d648906b`已经完成exact Pod验证、双launcher dry-run、双后端fresh launch并
观察到连续durable ACK；这只是一次稳定启动验收，不在本页追加瞬时ACK数字。现态与后续证据唯一看
[双后端长跑TODO §0.4](action_ball_dual_backend_longrun_todo_20260819.md#04-2026-08-23-v4最终冻结与v5第一性原理自查)。
schema 6在同一次PPO前host reduction中继承对Reward/return/advantage、
policy/critic observation、action、value、action log-probability、old mu和old sigma的finite/domain检查，
新增`scheduled_due_rows`、`due_terminal_overlap_rows`、`reveal_due_rows`与具名
fact-integrity字段，并把独立验过的path-free
[run_identity.runtime_stack](../DEFINITIONS.md#mujoco-fullmdp-runtime-stack)与
[run_identity.plant_model](../DEFINITIONS.md#mujoco-fullmdp-plant-binding)带入每条ACK、completion和consumer
重建身份；旧`mujoco_warp_runtime`被原子替换，不保留双真源。done仍严格二值，old sigma仍严格为正。
旧V4的`racket_contact_eligible`与launch完全相同，只是已删除的冗余别名，不是contact opportunity或分母。
H48并不是性能豁免：验收统一报告transitions/s、原始wall和`wall_s * 24 / H`的H24-equivalent；约6秒只是
“继续大砍迭代时间”的方向目标，不是发车、rate probe或safety Gate。启动验收也不闭合正式matched-strata
速度、学习收益、contact/landing、12500 completion或physics/transfer parity，整条lineage继续
`diagnostic_unauthorized`。

learner transition只允许一个因果顺序：freeze scheduled due → 结算existing launch/park →
physics/terminal/facts/reward on `obs_t` teacher → 结算outcome/recovery → 只对survivor分类
public due → selected measured frame0进入返回的`obs_{t+1}`。首次ACCEPT前joint/body共同使用
reset-ready，第一笔task-conditioned action/reward从下一transition开始，recovery使用completed-action
frame0，不新增offset。结算后仍busy才DEFER；起始busy但当界自然RETIRED可立即ACCEPT；due+terminal不public；
launch+outcome同tick合法，outcome与natural recovery互斥。

完整性cause分开编号：Isaac共享ActionEpoch owner共36项row fault（bit0--26的27项既有原因、R03
identity/stale/nonfinite bit27--29、Physical/R06六项bit41--46）；MuJoCo只有四项per-transition packed
cause（R03 nonfinite、R06 source-invalid、R07 sequence、R07 nonfinite）。Mu R03 stale因逐tick调用与同step
消费按构造不可达而删除，不影响Isaac的R03 stale bit，两个namespace不混计。所有cause只进入各自既有唯一
packed pre-optimizer drain。`pure_timeout = raw_timeout & ~plant_terminal`；只有pure timeout获得RSL
bootstrap与canonical timeout reason，horizon与tilt/table/qdes重叠不bootstrap。Mu `robot_hit_table`表示
keepout或resolved-table，resolved只是子fact。`invalid_contact + done`是真reset而不是retire。

MuJoCo one-shot launcher在child任何import前把`WARP_CACHE_PATH`、`CUDA_CACHE_PATH`、`TMPDIR`和
`PYTHONPYCACHEPREFIX`分别绑定到fresh `<run-root>/warp_cache`、`cuda_cache`、`tmp`和`pycache`；这些是
run-owned编译/临时缓存位置，不是训练Gate或第二份physics authority，也不得回退到用户根目录cache。

为得到可迭代的墙钟数据，两后端只增加一个固定`10 warm-up + 50 measured + 1 tail`的H48 rate diagnostic。
MuJoCo使用`--full-a --diagnostic-rate-probe`；Isaac只可把task中的
`action_ball_full_mdp_rate_probe=false`显式改为`true`。两者都必须profiler-off、fresh process、4096 env，
并保持`diagnostic_unauthorized`。Isaac root `max_iterations`仍只能缺席或等于typed `12500`；传61仍在Kit前
拒绝，实际61预算由code-owned diagnostic flag在runner边界安装。该入口不写学习结论、不授权resume/
promotion/export/deploy，也不得在完成后自动转成12500；先报告50个measured update的逐项wall、p50/p90、
transitions/s和H24-equivalent，再由人决定继续优化还是另开fresh长跑。约6秒不是probe
通过线，probe也不代替corrected V5的fresh训练证据。

#### FullMDP semantic Observation V2与snapshot边界（2026-08-21 branch candidate）

family A只接受actor contract `action_ball_full_mdp_semantic_actor_v2`（203-D）和critic contract
`action_ball_full_mdp_semantic_critic_v2`（219-D）；observation kind为
`action_ball_full_mdp_semantic_observation_v2`。Full-A不得回退到旧229/399 V1；V1符号只供明确声明的
historical WAIT consumer使用。family C将另用202/218合同，不补零伪装成A。

203-D actor只含目标传感链可观测的robot/teacher/anchor、table-relative root XYZ、continuous heading、
heading-frame COM velocity和raw A/+Y task-normal residual；contact/support/spin/fault/reward ledger只允许出现在critic或
telemetry。actor task-valid必须来自Motion-visible mask，不能用Epoch retained row把RETIRED task重新暴露；
target必须使用actor-visible delayed planner tuple，不能读live truth。所有scale是V2 ABI内的静态常量；
不得另开running normalization或CLI clipping改变语义。现役`history_length=0`；没有
same-observation/different-required-action alias反例时，不新增history、冗余观测或无法部署的oracle。

Isaac在尚无admitted shot key时没有R07 recovery业务事件。该no-key路径只核独立的source step、
ActionEpoch reset generation与Motion cadence chronology，不读取ContactSensor；critic `[216:219]`的
双脚support与ready dwell按N/A语义填零。此时`postphysics_valid=true`只表示neutral chronology已在本control
step发布，不表示测得“双脚无支撑”。一旦存在keyed recovery，R07仍从同一真实post-physics plant sample
读取support/slip并计算完整reward/readiness。selected reset发生在post-physics之后、返回observation之前；
仅reset generation相对R07 publication精确`+1`的keyed行清零，未reset peer仍使用同一publication并严格
对齐tick，下一次真实keyed post-physics恢复。跳代、回退或整数wrap必须拒绝；不要通过Observation重读
plant、伪造R07 capability或全batch清空来“修”这个边界。

虽然critic宽度仍为219，idle真实foot bits变成N/A zero已经改变数值语义；这只保留shape/load兼容，不保留
旧checkpoint的exact-resume语义。使用该合同必须fresh namespace/fresh lineage，禁止把旧snapshot续成同一
lineage。actor 203-D没有变化。

launcher在构造Kit/runner前必须对exact actor/critic contract、203/219宽度和training contract schema/hash
fail closed。禁止通过Hydra、task YAML、MuJoCo CLI或snapshot metadata覆盖contract、width、layout、scale、
PPO recipe或execution cadence；出现冲突就启新代码合同与fresh namespace，不做运行时“兼容修正”。

每个V2 snapshot及其receipt必须绑定同一`training_contract_schema_version`和
`training_contract_sha256`；该SHA已经覆盖actor/critic contract与维度及PPO配方，不再复制一套同writer
observation摘要作自证。缺字段、旧schema、hash mismatch、路径/iteration不一致或文件未完成验证时，consumer
必须拒绝。snapshot仍明确为`diagnostic_nonresumable`，并写
`checkpoint_authority=false / resume_authority=false`；runner `load()`继续硬拒。receipt只证明已持久化文件与
本次training contract相同，不授权resume、promotion、export、部署或真机安全。

当前真实IMU、table/root定位、marker→COM因果速度和planner producer尚未接通；simulation V2通过也只能称
host/Pod diagnostic。V5已经从上述final clean SHA在exact Pod分别fresh启动；任何后续重启仍只能使用fresh
exact checkout、fresh run root与fresh namespace，禁止resume、hot-patch或迁移旧V4/a103运行证据。

portable MuJoCo Full-A的branch候选单次入口是`scripts/launch_mujoco_full_mdp_successor.py`。它只接受
clean Git checkout、absent `/workspace/.../<namespace>`、canonical Python、显式GPU index/UUID与已有lock file，
并要求[`--ready-pose`](../DEFINITIONS.md#mujoco-fullmdp-ready-pose)与
[`--plant-xml`](../DEFINITIONS.md#mujoco-fullmdp-plant-binding)分别绑定canonical regular ready JSON和root MJCF。
launcher在建run root前核ready SHA及plant root filename/SHA，再把两条locator显式写入child环境；runner在环境构造
前后各核完整source closure，构造中的canonical verifier另核manifest、portable identity、base MJB/toolchain/model
合同。launcher会从child环境删除ambient `HOPE_GEOMETRY_PY`；绕过launcher直调Full-A runner时，只要该变量
存在，runner也会在court/MJLab/Warp import前拒绝，不能再用诊断override替换实际构造geometry。

runner还在自身首次Torch/MJLab import前建立并cold verify唯一
[`runtime_stack` v1](../DEFINITIONS.md#mujoco-fullmdp-runtime-stack)：actual EPA48与RSL3.1.2 wheel bytes、
distribution/import winner，以及MJLab1.5.3选定193文件树必须同时命中；环境构造后再复核tree和每个loaded
`mjlab.*` module origin，然后才允许形成run identity/ledger。pre/post payload逐字段相同，不能留下旧
EPA-only wire或从已import module自报身份。

live env构造完成后，runner核实际geometry source SHA，把已经augmented的live `env.mj_model`经private
stage→hash/fsync→no-clobber hardlink发布为run-owned `runtime.mjb`，并逐字节核诊断预注册MJB
`1ef4bb9e…30c0b / 72,260,546 bytes`，再独立绑定policy clock、Warp capacity与从verified base派生的
owner-local-frame digest；完整字段见
[`runtime_attach v2`](../DEFINITIONS.md#mujoco-fullmdp-plant-binding)。这些path-free事实进入schema-6
update/evidence与schema-5 completion。dry-run只打印固定H48 argv/env、plant locator与expected identity，
不查GPU、建run root或
冒充live augmented-model verification。
真实模式先按`nvidia-smi index→UUID`核选卡空闲，再把**同一UUID**直接写入child
`CUDA_VISIBLE_DEVICES`，不假定numeric CUDA enumeration与`nvidia-smi` index偶然一致。GPU flock的同一
open-file-description通过`pass_fds`由child继承：parent等待自然rc并原样返回，即使parent异常退出，仍存活的
唯一child也继续持有lifetime lock，直到自身退出；child的cwd固定为fresh run root，使`MUJOCO_LOG.TXT`等
底层fallback产物不能污染source checkout；
没有monitor、retry、resume、signal或`ACCEPT`门。当前live source `39f94819`已按该入口完成exact Pod双dry-run和MuJoCo fresh
launch并观察到连续durable ACK；这不把旧V4/a103的host、dry-run或ACK升级为发车授权。完整证据边界见
[portable Full-A实验§0](../experiments/2026-08/EXP-ACTION-BALL-MUJOCO-PORTABLE-FULLA-20260819.md#epa48-fresh-runtime-binding-20260821)。

自然结束后的独立`mujoco_full_mdp_longrun_consumer.py`必须显式接收
[--expected-plant-xml](../DEFINITIONS.md#mujoco-fullmdp-plant-binding)，从locator重新执行canonical full
verification，重建path-free `run_identity.plant_model`，再逐条对账schema-6 update/evidence、
schema-5 completion与snapshot；summary固定schema 5。consumer独立重算base receipt与owner-local-frame
digest，逐字段要求geometry SHA、policy clock、Warp capacity与actual augmented MJB identity，并在读取
`runtime.mjb`或ACK前cold verify同一runtime stack、重新hash并用MuJoCo加载run-owned MJB。

consumer还必须在fresh prefix上重验`launch<=reveal`、`racket contact<=launch`、
`selected contact<=racket contact`、`R03 present<=launch`、`flight outcome<=launch`、
`landing crossing<=selected contact`、`shot retired<=launch`，以及Mu四项fact-integrity、pure-timeout与
canonical table reason。R03与contact clock不同，`selected_contact/R03_valid`只作描述比；真实
selected-contact rate以launch为分母。`r06_common_per_eligible`是closed task-landing成功率，
`opponent_landing_per_crossing`只是crossing条件比例，不得代称总成功率。consumer必须拒绝已删除的
`racket_contact_eligible`假分母，不得信任producer绝对路径或把旧V4证据升级解释。当前live source `39f94819`的exact checkout、
Pod验证与fresh run已经完成启动验收，但formal plant registration、contact/landing、12500 completion和
physics/transfer parity仍未闭合；launcher、run、completion与consumer均为fresh
`diagnostic_unauthorized`，不形成formal readiness。

consumer对runtime stack、plant/MJB bytes、snapshot与ABI的核验来自独立读取；但业务event仍来自同一producer
的逐row ledger。summary中的`business_chain_complete`目前只能解释为`producer-attested +
consumer aggregate-consistent`，不能代称独立same-env/same-epoch replay。formal promotion前必须增加keyed
carry-state重算或可重放trace；跨env边际不能互相拼成闭环。

训练authority顺序固定为optimizer → WAL/fsync → owner ACK → EPOCH_ACK/fsync；stdout不是authority。live
`39f94819`的stdout为regular file且未失败，但post-launch审计发现snapshot/completion后的两处裸print；下一
source `a3c528f1b4c9b0a60f5cd3aeec28a11e990044b3`已clean/push，并用closed-pipe与stdout+stderr双失败反例
证明best-effort marker不会回滚或否定durable训练；exact Pod fresh checkout全文件=`52 passed, 1 skipped`
（显式real-GPU skip）。该修复不改变学习状态，不hot-patch或重启live run。

Isaac Full-A的对应单次入口是`scripts/launch_isaac_full_mdp_successor.py`。它复用仓库现有Kit boot
owner，不再建立第二套supervisor：从clean Git固定同一typed H48 argv，核exact IsaacLab、Kit Python、
RSL wheel与A3 USD，先取得目标GPU的nonblocking lifetime lock，再核index→UUID和empty compute-app，最后
才创建fresh run root。fd16 runtime receipt、fd18 sealed RSL archive与GPU lock由唯一child继承；训练child
通过`/usr/bin/env -i`获得窄runtime环境，避免Kit launcher的环境清洗丢失attestation输入。
当前exact Pod调用中，`--isaac-python`是`/workspace/isaacsim-5.1.0/python.sh`，而
`--kit-python`必须是`/workspace/isaacsim-5.1.0/kit/python/bin/python3`；两者用途不同，不能用前者路径
替代后者的Kit runtime attestation。

fresh root还必须拥有`home`、`cuda_cache`、`xdg_cache/config/data/state`、`tmp`、`pycache`与`training`；
launcher/child分别把`HOME`、四个`XDG_*`、`CUDA_CACHE_PATH`、`TMPDIR`和`PYTHONPYCACHEPREFIX`钉到这些目录。
这防止Kit/Omniverse/NVIDIA JIT把大缓存写入共享且容量很小的root overlay，也防止不同namespace混用缓存；
它不是新的学习或安全Gate。不得未经共享Pod owner授权清理`/root`或他人的cache来替代正确的run-owned边界。

该入口不把matched timing、`ACCEPT>0`、已有学习表现或短跑成功当作启动门；这些都不是防止错误写卡、
错误资产、错误进程或数值故障的独立安全事实。它也不monitor/retry/resume/signal，ready marker后只验证
exact PID=PGID、runtime receipt与live non-zombie进程并返回。真正的overflow/nonfinite、joint/table/contact
边界和optimizer后durable ACK仍在训练路径内保持fail-stop。当前live source `39f94819`已经完成exact Pod dry-run与Isaac fresh
launch并观察到连续durable ACK；这只验证one-shot启动链，不代签profiler-off matched-strata wall、
contact/landing、12500 completion或physics/transfer parity，也不得把host测试、旧V4/a103证据或dry-run
升级为正式训练证据。Isaac的36项fault、post-transition reference、pure-timeout与stdout non-authority均按
本节共享合同执行。

2026-08-22课程解阻后的successor必须额外显式给出GPU-local `--cpu-affinity`。CPU list必须在每次launch前
从live PCI/NUMA topology重新取得，不能把历史GPU0的`32-47,96-111`外推到另一张卡；当前Pod1实测GPU1/2为
`48-63,112-127`。launcher会核该list属于自身allowed cpuset，再以`taskset`包住唯一child。短期归因
可加`--profile-updates 5`，但该五轮固定为`speed_evidence_eligible=false`；正式墙钟窗口必须省略该flag或
传`0`。launcher还把`HOPE_ACTION_BALL_FULL_MDP_LOG_ROOT`固定到fresh run root的`training/`，并把
`hydra.run.dir`固定到同一目录下的`hydra/`，所以checkpoint、WAL、contract、TensorBoard和Hydra metadata
都不再散落到source checkout。三个参数都不新增学习或安全Gate。

典型新增片段为：

```bash
python3 scripts/launch_isaac_full_mdp_successor.py \
  ...既有exact Isaac/asset/GPU/fresh namespace参数... \
  --cpu-affinity <live-target-GPU-local-cpu-list> \
  --profile-updates 5
```

ready marker后还必须由launcher直接检查该exact leader的`/proc/<pid>/fd/<gpu-lock-fd>`与canonical lock file
具有相同device/inode，且`fdinfo`唯一lock行包含`FLOCK ADVISORY WRITE`。`lslocks`或`/proc/locks`可能把parent
已经退出、但child仍继承open-file-description flock的owner显示为PID 0或直接漏掉，不能据此误判lifetime lock
已丢。该检查只证明同一训练进程仍独占已批准GPU lane，不证明学习、physics parity或promotion。

课程、D05 chronology、teacher和fresh验收见
[2026-08-22课程解阻实验](../experiments/2026-08/EXP-ACTION-BALL-FULLMDP-CURRICULUM-UNBLOCK-20260822.md)。

若FullMDP在base manager构造中途失败，当前进程必须视为cold-discard：环境会按pinned顺序单次
best-effort清理已存在manager与simulator；任何terminal simulator清理失败都会sticky拒绝后续资源操作并
要求进程退出。不得在同一进程捕获该异常后重新构造环境、重试旧sim、复用namespace或把它当成可恢复
checkpoint；重新运行必须从fresh process开始。

<a id="effective-reward-truth"></a>

### Effective Reward truth（不要再按 pack 名字猜）

现役 v2 的球拍位置/速度/拍面默认真值是 `4/0.5/0.5`。旧名义表
`393.4/295.1/229.5` 已从默认路径删除；v2 下显式复活会 fail-closed，避免单项每步收入重新形成
数量级悬崖。日志里出现 `reward_pack=v2` 仍不代替 effective recipe：必须按下述 receipt 读取实际
term/weight/params。v1 历史路径不因这次清债改写。

每个科学 run 都必须从已经 compose 的环境配置生成
[`effective Reward recipe`](../DEFINITIONS.md#effective-reward-recipe)，内容覆盖所有 active
callable、weight 和 params：

1. 构造 simulator 前生成并与可选 expected SHA 对账；
2. 环境实例化后从 runtime config 重算，必须与 pre-gym receipt 完全相同；
3. 写入 `params/effective_reward_recipe.json`，并把同一 payload/SHA 嵌进 training hard contract；
4. checkpoint resume、A/B、export 和 capability artifact 都按这个 SHA 拒绝配方漂移。

预注册臂若声称“v2 名义冻结表”，还必须使用 strict compose guard，确保显式 task key 不能静默压包；
普通历史兼容路径只能 WARN，不能据此追认旧 run。当前因果结论与 paired 设计见
[effective Reward 审计](../experiments/2026-07/EXP-EFFECTIVE-REWARD-CAUSALITY-20260727.md)。

2026-07-01 update:

- `HOPEPingPong` now defaults to unified forehand+backhand HITTER training: `registry_name_2` enabled, `target_mode: uniform`, per-clip 3-D blade-centered position and velocity target boxes (`pos_range_per_clip` / `vel_range_per_clip`; this supersedes the earlier fixed hit plane `x=0.4` with (y,z)-only sampling), actor `swing_type`, and no actor racket-normal observation.
- Local Step 9-12 motion products are first-class training inputs. Pass `motion_file=<forehand.npz>` plus optional `motion_file_2=<backhand.npz>` to skip WandB entirely; omit local files to use `registry_name` / `registry_name_2`.
- `setup_train_env.sh` is portable again: it reads optional `setup_train_env.local.sh` and overridable `HOPE_ISAAC_*` paths, with auto-detection for known `/workspace/...` Isaac layouts when present.

2026-07-02 update (verified on the current shared RunPod):

- Local two-clip `HOPEPingPong` smoke and a registry-backed WandB pipeline smoke both ran on the copied Agibot A3 URDF asset (31 actuated DOF). The registry smoke run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, but the `hope_forehand:v4` / `hope_backhand:v4` motion artifacts it used were later verified to face world +Y rather than HOPE +X. Treat that run as pipeline-only evidence.
- Until future verified registry artifacts are uploaded, pass the corrected local `_hopex.npz` clips explicitly via `motion_file=` / `motion_file_2=` instead of relying on the v4 registry aliases. The local v5 clips are R15 ablation inputs only, not product/default replacements.
- HOPE task YAMLs set `motion.wrap_teleport: false` (also the code default): a mid-episode clip wrap resamples the reference clip/time and racket target without teleporting the simulated robot. Episode reset still uses RSI.

2026-07-03 realignment:

- The default training task is now `HOPEPingPongDeployParity` (gym id `HOPE-PingPong-DeployParity-AgibotA3-v0`); `HOPEPingPongRealSensor` is a backward-compat alias for the same task. Its actor observation is 175-D deploy-parity: it removes `motion_anchor_pos_b` (3) and `base_target_pos_b` (2) and reframes `racket_target_pos_b` racket-FK-relative. Layout reference: `scripts/realsensor_obs_reference.py`; checks: `scripts/verify_realsensor.py`.
- `task=HOPEPingPong` remains available only as the legacy 180-D full-obs comparison path; it is NOT deploy-honest and cannot deploy.

2026-07-05 R15 v5 correction + strike-annotation registry:

- Product/default train and replay configs remain on the hopex/registry route: `cfg/train.yaml`
  and `cfg/play.yaml` keep `motion_file: null`, and `HOPEPingPongDeployParity.yaml` keeps
  `strike_phase_per_clip: [0.47, 0.333]`. Do not edit those defaults to v5 for R15.
- `cfg/strike_annotations.yaml` is the contact-phase source of truth for reference clips (all 6
  adjudicated 2026-07-05; the hopex values are a speed-peak CONVENTION — its source videos are
  ball-less dry swings). `scripts/analyze_strike_phase.py` applies annotations first and reports
  the speed peak only as a diagnostic candidate (known trap: post-contact whip / pre-contact
  pull-up). R15 overrides are now `strike_phase_per_clip=[0.673,0.362]` with the regenerated
  backhand boxes (see NOW.md).
- `scripts/play.py` uses the same local motion resolver as training when `motion_file` or
  `motion_file_2` is set, so R15 replay/export honors both local clips.

2026-07-04 update (deploy-parity robustness flags, all default OFF):

- `motion.clip_switch_prob` (default `0.0`; try `0.002` ≈ one switch per 3-4 swings): each control
  step that fraction of envs aborts its swing operator-style — the reference jumps to a random
  clip's FIRST frame with a fresh pre-swing hold and a fresh target; the robot is untouched (no
  teleport). This is deploy parity for `pp_reference_clock.hpp`, which flips `clip_id` mid-swing
  whenever the planner re-sides the target; training previously only switched at clip END — the
  root cause of the venue falls at 准备/正手/反手 switches. Aborted swings do NOT enter the A8
  post-swing buffer and slightly deflate completion-rate metrics. Watch: `clip_switch_count`.
- P2.4 `base_decel` reward (PACE-style pre-strike base-speed shaping, default OFF via
  `rewards.base_decel_weight: 0.0`): see Reward Shaping below.
- `motion.speed_scale_range` (R14 retiming, default `[1.0, 1.0]` = OFF; ablation trial
  `[0.8, 1.2]`): per-swing reference playback speed, resampled at every swing entry (wrap,
  clip switch, reset). At speed s the clip clock advances s frames per control step (float shadow
  clock, round() indexing = deploy-clock parity), reference joint/body/anchor velocities read ×s,
  `time_to_strike` runs ÷s (computed from the float clock so the exact-strike detector still fires
  once per swing), and provisional racket velocity targets scale ×s (uniform boxes,
  `reference_perturbed`, and the HER clamp box). A schema-3 question bank is different: its
  demanded racket velocity is an absolute inverse-physics answer for the unchanged incoming ball,
  so the final bank assignment deliberately overwrites the provisional speed-scaled target. This
  makes `question_bank` compatible with fixed `motion.speed_scale_per_clip` or a speed range without
  silently slowing the required return. Positions/normals are speed-invariant. Retiming is TRAIN-ONLY:
  play/eval force it back to `[1.0, 1.0]`. Deploy note: the runner's `swing_speed` knob retimes
  the clock but does NOT scale reference/target velocities — enabling it for an R14-trained
  policy requires adding those two scalings. Watch metric: `playback_speed`.
- A1v2 actor-view sensor defects (`racket:` block; modeled on the venue mocap fit — occlusion gaps
  concentrate at contacts, re-lock after contact carries a fresh bias): `target_dropout_prob`
  (per-step frame loss, hold-last), `target_post_strike_dropout_s` (forced hold-last window after
  each strike; venue ~0.03 s), `target_bias_per_swing` (3-D Gaussian position bias resampled at
  each strike edge, held constant within a swing). They degrade ONLY the actor-visible target
  view; rewards, critic, and metrics keep the true target. Same block as the A1 latency/noise
  family (`target_delay_steps`, `target_jitter_pos_per_s`, `target_jitter_vel_per_s`,
  `midswing_resample_prob`, `target_noise_white`, `target_noise_ar1_sigma`,
  `target_noise_ar1_rho`) — every one of these defaults off.
- `racket.target_delay_tts_mode` makes the actor-visible planner tuple time-coherent when
  `target_delay_steps > 0`. `live` is the historical/default behavior: delayed position, velocity,
  face and sign are paired with the live countdown. `source_timestamp_compensated` delays all five
  fields atomically and subtracts `target_delay_steps * policy_dt` from that delayed countdown;
  `uncompensated` delays the same tuple but intentionally leaves its old countdown untouched as a
  matched negative control. Only the policy observation switches to this actor countdown; critic,
  Reward gates and truth metrics retain live `time_to_strike`. Reset backfills the complete tuple,
  while dropout holds the complete tuple. The selected mode and delay are checkpoint hard-contract
  fields. See [atomic planner tuple](../DEFINITIONS.md#atomic-planner-tuple-timing).
- `task.planner_revision` is the opt-in replacement for the old “freeze target after engage” and
  `midswing_resample_prob` ablation. When enabled it configures both motion and racket command
  terms together; a half-configured clock-only or target-only path is rejected. One physical ball
  keeps immutable question/Reward/critic truth while the actor receives atomic position, velocity,
  signed-normal and TTS revisions. A checkpoint-bound
  [`phase governor`](../DEFINITIONS.md#phase-governor) may change reference rate without reversing
  phase or exceeding its frozen rate, acceleration, target-delta or deadline-delta limits. The
  legacy hold clocks are forced to zero because the revision task's initial TTS is the sole
  preparation clock. For `phase_governor_v1`, `racket.target_delay_steps` must remain `0`.
  Positive delay fails before launch because the legacy ring delays only the actor, not the motion
  governor; it becomes legal only after both consume one coupled transport tuple in the same tick.
- An enabled revision block must include a complete `initial_tts_mixture`, not merely a broad min/
  max range. The intended launch family has four explicit strata: a sub-0.5-second stress band,
  an exact 0.5-second point mass, a 0.5–0.9-second fast-deployment band and a longer-arrival band.
  The four weights sum exactly to one and their support exactly equals
  `initial_tts_range_s`. Thus 0.5 seconds is a separately counted required baseline, not the
  minimum preparation time. Per-component, below/exact/above-0.5 and total sample counts must
  partition exactly in every accepted full-scene probe and checkpoint contract. See
  [`initial TTS mixture`](../DEFINITIONS.md#initial-tts-mixture).
- This path does not claim a 0.5-second return from source tests. Behavior must be measured with
  the immutable K100 [`0.5-second timing exam`](../DEFINITIONS.md#timing-exam-0p5): frame 0 has
  zero reference velocity, every attempt remains in the denominator and a fixed clock multiplier
  is explicitly inexact. A separate TOPP run must also certify the chosen reference/action path;
  the current safe heuristic upper bound is not a proof of a global minimum or a 0.5-second
  feasible trajectory.
- A task-revision run is pruneable only when exactly one command term provides the behavior
  ledger and the runner emits one canonical `HOPE_EXACT_BEHAVIOR_UPDATE_JSON=...` record per PPO
  update. Completion is `swing_completion_count / swing_outcome_count`: both counters close on the
  same attempt-end event, so a start in one 100-update window and an outcome in the next cannot
  invalidate either window. `swing_start_count` and `strike_opportunity_count` remain raw
  diagnostics. Physical fall accepts only exact boolean termination reasons and is split into
  mutually exclusive pre/post counts; guard or timeout reset is separate. A numeric truthy tensor,
  duplicate provider, missing update, duplicate update or zero denominator makes the behavior
  decision unavailable and the trainer continues—it never manufactures zero or stops the arm.
- In planner-revision mode, ready eligibility is exactly the first metrics sample after a new
  active `(control_epoch, task_id)` is installed. It is not the install function itself and is not
  `motion.in_hold OR new task`: legacy hold clocks are zero because time-to-strike (TTS, remaining
  time before contact) is the sole preparation clock. Same-ball `task_revision` updates therefore
  do not duplicate ready samples. The log must emit `ready_phase_sample_count`,
  `ready_planner_task_entry_sample_count`, `ready_planner_legacy_hold_violation_count` and
  `ready_foot_sensor_unavailable_sample_count`; a nonzero planner legacy-hold violation or an
  unexplained zero denominator blocks pruning. Missing foot sensors are unavailable measurements,
  never fabricated zero contact/slip. Historical receipts created without these witnesses cannot
  be backfilled.
- Before launching a pruneable successor, run a clean detached full-scene probe that proves a
  finite checkpoint, exact source/contract binding, nonzero conserved task-entry and ready
  denominators, zero planner legacy-hold violations and explicit sensor availability. The Pod2
  CPU-only direct probe (`4/4` on exact `0ebd14a6…a8dd`) checks source mechanics only. It does not replace the full-scene
  probe or the two complete disjoint 100-update windows required for any ranking or stop.
- `rewards.free_wrist_ori_mimic` (R16, default `false`): drop `right_wrist_yaw_Link` (the racket
  mount) from the `motion_body_ori` / `motion_body_ang_vel` body lists — the wrist's ORIENTATION
  stops being imitated while position/linear-velocity mimic keep the swing path. Rationale
  (franco): the video pipeline's wrist orientation is unreliable (GVHMR), so mimicking it caps
  face quality; freed, the face is shaped by the `racket_normal` reward (and by ball-outcome
  rewards on the VirtualBall stack — the arm with a real learning signal for the face). Note this
  codebase has NO joint-level imitation rewards — body-level `motion_body_ori` on the wrist link
  IS the face mimic, so the flag is config-level (body-list filtering in `train.py`).
- ⚠ Override-whitelist rule: task-yaml keys under `task.motion` / `task.racket` are translated
  through explicit whitelists (`_MOTION_KEYS` / `_RACKET_KEYS` in `scripts/train.py`) and any
  unconsumed key RAISES at startup. Adding a new key to a task yaml therefore requires extending
  the whitelist in the SAME commit — 018467a added `clip_switch_prob` to the yaml only and broke
  every task-yaml startup until the 74c129e hotfix.

`TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied Agibot A3 URDF asset (31 actuated DOF), including WandB logging, checkpoint save, and ONNX export. This proves the pipeline can run; it is NOT an accepted quality baseline. G04/G05 remain Partial, and G06/G07 are not accepted until sim-to-sim and dry-run deployment gates record verification.

## Training-critical change barrier

Before changing any contract that affects a live trainer, checkpoint consumer, evaluator, pruning
rule, planner-policy tuple, motion timing law, or recurring Pod command, first set the existing
training automation to `PAUSED` and verify that state. Pausing the automation does not itself stop
trainers; it prevents a stale recurring turn from inspecting, attesting, pruning, launching or
signalling against a contract while that contract is being changed.

Keep the automation paused while implementation, independent review, source tests, documentation,
and any required one-shot runtime gate are incomplete. Resume the **same** automation only after the
verified change is on `main`, the operation document contains the exact command, and every in-flight
one-shot mutation has an unambiguous no-clobber state. Never create a second automation to work around
this barrier. Read-only operator inspection remains allowed, but it cannot publish receipts, signal a
process, retry a run, or be reported as a behavior verdict.

This branch adds:

- a scrubbed `setup_train_env.sh` as the training shell setup source of truth (site paths are now overridable env vars).
- source-first `HOPE_WBT_PYTHONPATH` ordering, so local `whole_body_tracking` edits beat stale installed copies.
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.
- `HOPE-TableTennis-AgibotA3-v0`, a first-pass Isaac Lab table/net/ball/A3 physics scene for G04 visualization and future G08 returner/spin experiments, now with a tracked Purdue PACE USD table/net visual overlay and Purdue-style table/ball contact materials.
- updated `HOPEPingPong` target/reward defaults with unified forehand/backhand sampling, per-clip blade-centered uniform racket target boxes, per-clip strike timing, conditional exact-strike metrics, and debug reward logging hooks.
- canonical WandB motion uploads where every motion artifact contains `motion.npz`, regardless of the source filename.
- HOPE +X motion alignment in `scripts/csv_to_npz.py --robot agibot_a3` (`--hope_frame auto`) before local save/upload.
- `scripts/check_motion_target_alignment.py`, a no-Isaac gate for frame-0 yaw, +X-dominant strike velocity, and target/reference center alignment.
- `motion.wrap_teleport` (default `false`; kept explicit in the HOPE task YAMLs) controlling the mid-episode RSI teleport on clip wrap, plus a `racket_progress` resample-spike fix. (The branch's original `rsi_on_wrap` knob was dropped 2026-07-03 in favor of main's equivalent `wrap_teleport`.)
- explicit `wandb.finish()` before Isaac `simulation_app.close()`, so WandB runs finish and sync before Isaac can hard-exit the process.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/scripts/play_table_tennis.py`
- `hope_training/whole_body_tracking/scripts/probe_metric.py`
- `hope_training/whole_body_tracking/cfg/train.yaml`
- `hope_training/whole_body_tracking/cfg/play.yaml`
- `hope_training/whole_body_tracking/cfg/strike_annotations.yaml`
- `hope_training/whole_body_tracking/setup_train_env.sh`

## Environment Setup

This runs in the GPU/Isaac environment (Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, CUDA GPU), not the ROS environment. `grasping` is the maintainer's EXAMPLE distrobox name — substitute your own box.

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

The script must be SOURCED (not executed) in every new GPU/Isaac terminal. It defines the `hope_isaac_py` launcher, sets `HOPE_WBT_PYTHONPATH`, and exports placeholder WandB variables for optional registry/logging use.

The script is scrubbed of site-specific paths. It reads overridable env vars with placeholder defaults:

- `HOPE_ISAAC_PYTHON` — the Isaac Lab Python interpreter `hope_isaac_py` wraps.
- `HOPE_ISAACLAB_ROOT` — your Isaac Lab checkout.
- `HOPE_ISAAC_VENV_SITE` — optional extra `site-packages` to inject (e.g. to provide `hydra`/`omegaconf`).

Set these for your machine in a git-ignored `setup_train_env.local.sh` next to the script; `setup_train_env.sh` auto-sources it if present and auto-detects known `/workspace/...` Isaac layouts.

On the current shared RunPod (verified 2026-07-02), the actual Isaac install is the venv at `/workspace/hope_isaac_venv` with Isaac Lab at `/workspace/IsaacLab`; point the `setup_train_env.local.sh` overrides at that install. The legacy `/workspace/isaacsim/python.sh`, `/opt/drone_venv`, and `hope-motion-py310` paths are not used for Isaac training. If another machine has different paths, update the local override and this doc together.

A from-scratch Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Python 3.10 install is NOT documented here and is the single biggest reproducibility gap. Follow the official [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) first, then point `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` at it.

### Blackwell (RTX 50-series, sm_120) torch fix

Isaac Sim 4.5.0 ships `torch 2.5.1+cu124`, which has **no sm_120 kernels**. On a Blackwell GPU
(e.g. RTX 5090) `import torch` "works" and `torch.cuda.is_available()` is `True`, but any real CUDA op
fails with `no kernel image is available for execution on the device`, so training crashes after Kit
startup. Fix (verified 2026-06-26 on RTX 5090):

```bash
# inside the Isaac env (hope-isaac-py310)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4    # Isaac Sim 4.5 requires numpy<2; the torch upgrade may pull numpy 2.x
```

`isaaclab*` carry a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the
upgrade does not break them. Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124
--index-url https://download.pytorch.org/whl/cu124`. Verify with a real kernel, not just `is_available()`:

```bash
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())"
```

### EULA

The first Kit launch needs the NVIDIA Omniverse EULA. Accept it non-interactively for headless runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### No-WandB training (local motion override)

`scripts/train.py` can load the motion clip from a local `.npz` instead of the WandB registry. Pass
`motion_file=<path>` (and `motion_file_2=<path>` for a unified forehand/backhand policy); when set it
skips WandB entirely, so use it with `logger=tensorboard` for an account-free run. Resolution is
LOCAL-FIRST (`resolve_motion_sources` in `train.py`): explicit `motion_file=` / `motion_file_2=` always
win and bypass the registry; only when no local files are given are `registry_name` / `registry_name_2`
downloaded from WandB. Back-compat: a local `.npz` path (or a directory containing `motion.npz`) passed
as `registry_name=` / `registry_name_2=` is rewritten to `motion_file=` and stays registry-free. If you
have no motion data at all, generate a placeholder "stand at default pose" clip (pipeline proof only,
not a real swing):

```bash
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50

hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

For a local unified HITTER smoke after the video pipeline has produced both clips:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_local_unified_smoke
```

### Motion kinematics schema 2 preflight

Fresh formal runs require each NPZ to bind three facts: body positions are
link origins, body linear velocities are COM-point velocities, and
`body_names` gives the complete articulation column order. Every clip must
also have one finite positive scalar FPS, all clips must share it, and it must
equal the policy rate (`1 / env.step_dt`, currently 50 Hz). Schema 1 and
untagged clips remain loadable only for diagnostic checkpoint compatibility;
they cannot produce an exact schema-v3 checkpoint/ONNX.

For MuJoCo conversion, discover the body order once against a trusted Isaac
reference, then reuse the emitted file. The GMR source `dof_pos` and runtime
`joint_pos` orders are intentionally different; the converter validates their
content-bound bijection and requires complete donor ONNX `joint_names`,
`articulation_joint_names`, and identity `action_joint_ids` metadata:

```bash
python scripts/csv_to_npz_mujoco.py \
  --mjcf /path/to/a3_pingpong.xml --donor /path/to/policy.onnx \
  --joint-order-contract configs/a3_joint_order_bijection_v1.json \
  --discover-map /path/to/trusted_isaac_motion.npz \
  --body-order /path/to/body_order.txt
```

Run `python scripts/a3_joint_order_contract.py` first. A successful source gate
still prints `schema2_materialization_authorized=false`; each new private motion
family needs its own content-bound/no-clobber conversion preregistration.

Migrate a legacy V5/MuJoCo clip whose stored velocity is the derivative of
link position:

```bash
python scripts/migrate_motion_kinematics.py \
  --input /path/to/legacy.npz --output /path/to/migrated_comv.npz \
  --source-point link_origin --mjcf /path/to/a3_pingpong.xml \
  --body-order /path/to/body_order.txt
```

`--body-order` describes the source NPZ columns; it is not automatically the
current articulation order. If a real Kit preflight reports a different
runtime body order, capture that current order and rerun with
`--target-body-order /path/to/current_runtime_body_order.txt`. The migration
then permutes all four body pose/velocity arrays by body name before converting
link-origin velocity to COM velocity. Never relabel the metadata without
reordering the arrays.

For a legacy Isaac clip already carrying COM velocity, use
`--source-point center_of_mass --body-order ...` and omit `--mjcf`. Never
infer the point or body order from a filename. Interpolation-only retiming
outputs are explicitly tagged `link_origin` and are not formal training
inputs; use the FK output mode to regenerate COM velocity.

Fresh schema-v3 training must also choose the joint-friction plant explicitly.
The checked-in A3 actuator config preserves historical, uncalibrated PhysX
coefficients. Because those dimensionless/load-dependent values have no exact
MuJoCo `frictionloss` equivalent, they remain a diagnostic control. Launch the
cross-engine-exact zero-friction control from scratch with:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongVirtualBall algo=ppo \
  task.plant.zero_joint_friction=true \
  motion_file=/abs/path/forehand_schema2_comv.npz \
  motion_file_2=/abs/path/backhand_schema2_comv.npz \
  ++task.racket.question_bank=/abs/path/schema3_train.npz \
  headless=true
```

The flag is absent/false by default and never changes an existing checkpoint.
The saved `training_contract.json` records the expanded per-joint coefficients;
any legacy warm-start remains exact-ineligible even when the new run selects
zero friction. Use a paired fresh `zero_joint_friction=true` versus
as-configured run when measuring the plant effect, and label the as-configured
cell diagnostic until a physically calibrated PhysX/MuJoCo mapping exists.

Do not use that binary flag to launch the future calibrated `SC` cell. The
unit-explicit plant-contract v1 preparation and its evidence checklist are in
[`prepare_semantics_correct_plant.md`](prepare_semantics_correct_plant.md).
Current training has no `SC` adapter hook. A reviewed future launch must bind
the prepared contract into the hard contract before `gym.make`; its final
MuJoCo evidence must instantiate the adapter in the Agibot vendor
Gate3/Gate3B runtime and bind the vendor MJCF/runtime/31-joint report. A
generic MuJoCo wrapper or current `contract-proxy` result cannot fill that
role.

### Serialized Kit boot for multi-GPU hosts

Do not let several Isaac/Kit processes initialize concurrently on one Pod.
`scripts/launch_kit_training_locked.sh` holds the host boot lock only until a
reliable log marker appears, launches the child in its own process group, and
records the exact PID/PGID and command in `<log>.launch`. After the marker, the
lock is released so already-booted training jobs may run concurrently:

```bash
source /workspace/codexschema/env.sh
KIT_BOOT_MARKER='Learning iteration' KIT_BOOT_TIMEOUT_S=900 \
  scripts/launch_kit_training_locked.sh /abs/path/arm/run.log \
  env CUDA_VISIBLE_DEVICES=0 /workspace/hope_isaac_venv/bin/python \
  scripts/train.py task=HOPEPingPongVirtualBall algo=ppo device=cuda:0 \
  headless=true logger=tensorboard run_name=arm
```

For a `max_iterations=0` mechanism smoke, use a marker such as
`[train.py] hard training contract:` instead. A process that exits before its
required marker is a failed boot even when its exit code is zero. A boot
timeout sends TERM, then KILL if necessary, only to the recorded arm PGID;
never replace that cleanup with a broad `pkill`.

The frozen 2026-07-11 Phase-1 recipes use
`scripts/launch_phase1_20260711.sh`. It verifies every parent checkpoint,
motion and train-bank SHA before invoking the locked launcher. Run the exact
179-D construction gate first, then start the three lanes assigned to each
Pod:

```bash
scripts/launch_phase1_20260711.sh smoke   # Pod 1; inspect contract, then wait for clean exit
scripts/launch_phase1_20260711.sh pod1    # M3 old/S1 + fresh schema-v3 seed 1
scripts/launch_phase1_20260711.sh pod2    # M2 old/S1 + fresh schema-v3 seed 2
```

Set `PHASE1_DRY_RUN=1` to validate inputs and print shell-escaped commands
without starting Kit. The causal continuations deliberately use the legacy
motion diagnostic flag and remain `training_contract_exact=0`; the two fresh
seeds use runtime-order schema-2 motion, a strict schema-v3 train bank, no
checkpoint and `zero_joint_friction=true`.

Those first six processes occupy the six cards but do **not** fill the measured
breadth capacity. The 2026-07-08 rule is three to four 4096-env jobs per GPU;
the 2026-07-11 Phase-1 target is 24 jobs. The scale-out roles are deliberately
layered so each host can be audited at two, three and four jobs per card:

```bash
# Run this launcher from a detached/current control worktree, but point every
# training command at the clean frozen 6d93bcb checkout.
export PHASE1_REPO_ROOT=/workspace/codexschema/nohope
export PHASE1_STAGGER_S=75
EVAL=/workspace/codexschema/nohope_eval_08e438e  # historical directory name; verify the live HEAD
test -z "$(git -C "$EVAL" status --porcelain)"
git -C "$EVAL" rev-parse HEAD

bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_2
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_3
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_4

bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_2
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_3
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_4
```

The two Pods may launch the same layer in parallel; one Pod still serializes its
own three Kit boots. Verify all six new first iterations/contracts before
starting the next layer. The authoritative assignment and run names are in
`configs/phase1_scaleout_matrix_20260711.json`. Scale-out roles refuse a dirty
checkout or a training commit other than
`6d93bcb16c422a2f42748c2dc99432559653480b`.
If a layer stops on its second/third boot, preserve that failed arm and rerun
the same role: a still-live, command-identical ready arm is verified and
skipped. If an earlier arm has already completed, use
`PHASE1_ONLY_ARM=<exact_run_name>` to launch only the reviewed remaining arm.

Do not wait for terminal checkpoints to discover whether an ablation works.
The initial missing curves are frozen in
`configs/phase1_checkpoint_curve_initial_pod{1,2}_20260711.json`. The following
command is a **historical record only**; those manifests predate the mandatory
screen-policy/job-contract schema and must not be passed to the checked-in new
worker. Their successful/failed states are already preserved:

```bash
python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
  --manifest "$EVAL/configs/phase1_checkpoint_curve_initial_pod1_20260711.json" \
  --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
  --state-dir /workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/initial_pod1 \
  --max-active-cpu 9
```

The live 2026-07-11 paper deliberately remains on the clean detached evaluator
commit and judge SHA below. The current branch has since hardened both worker
and judge; mixing that latest `judge.sh` with manifests frozen to the old SHA
is correctly rejected. Do not update the live manifest SHA in place.

```bash
EVAL=/workspace/codexschema/nohope_eval_08e438e
RUNTIME_MANIFESTS=/workspace/codexschema/phase1_fresh_20260711/runtime_manifests
test "$(git -C "$EVAL" rev-parse HEAD)" = 46a0ce24524fdb843e55fe82ba4c045f2adc090f
test -z "$(git -C "$EVAL" status --porcelain)"
test "$(sha256sum "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" | awk '{print $1}')" = \
  1a00702935096b063435c3f0bd23e75f76f13e1298c87310d1cec3c26cca8529
```

The runtime manifest copies are the corrected 20998/split/SP-inexact files
from this repository; copy and hash-check them **before** launching a worker,
never while that worker is alive. This lets the historical clean evaluator run
the corrected queue without editing its worktree.

The worker starts the next judge only after the prior judge reaches its CPU-only
MuJoCo phase. It sets OpenMP/MKL/OpenBLAS/NumExpr to one thread, records exact
checkpoint and evaluator commit/hashes, requires clean frozen training/eval
worktrees, refuses stale failed state, and never signals a training process.
`judge.sh` shares the training launcher's Kit boot lock; only CPU exam phases
overlap. Observation-normalizer sidecars preserve finite zero std
dimensions only because the bound runtime divisor is `std + eps` with
`eps=0.01`; negative/non-finite std or a non-positive divisor remains fatal.
Before taking the Kit lock, `judge.sh` now activates the CPU evaluator and
requires both the graph loader and runtime. The current Pods use
`onnx==1.22.0` and `onnxruntime==1.27.0`:

```bash
/workspace/hope_mjeval_venv/bin/python -m pip install 'onnx==1.22.0'
/workspace/hope_mjeval_venv/bin/python - <<'PY'
import onnx, onnxruntime
print(onnx.__version__, onnxruntime.__version__)
PY
```

`onnxruntime` alone is insufficient because formal normalization preflight
inspects and checks the graph through `onnx`. Exact A3 plant comparison uses
no arbitrary fixed tolerance: exact float64 equality passes; otherwise the
bound metadata must be a canonical finite float32 value and the MJCF value
must map to the same float32 grid point. This accepts serialization-only
armature (`2.71e-9`) and ankle-effort (`3.0517578e-6`, `118.2` versus
`118.199996948...`) residues while a neighboring float32 value still fails.
Do not put report-only formatting changes in `venue_ball_sampler.py`: that
module's complete SHA is part of every schema-v3 bank physics contract. Final
artifact exactness is overlaid by `mujoco_eval_onnx.py` while rendering the
denominator report, leaving the bank/scorer source bytes immutable.
Long-run milestones, paired stopping rules and peak
density are specified in
`docs/research/phase1_ablation_acceleration_2026-07-11.md`.
The first historical repair manifests intentionally produced a full clean plus
5%-noise record. Do not copy that cost into every milestone. The fresh
preflight retry manifests
`configs/phase1_checkpoint_curve_fresh_retry_pod{1,2}_20260711.json` use one
fixed clean (`ns=0`) schedule with `K=20` (10 questions per side) to establish
direction. A stop or promotion still requires a separately pre-registered
50-per-side clean paper; noise and full-paper cells are reserved for survivors.

The ongoing original-arm milestones are frozen in
`configs/phase1_checkpoint_curve_cadence_pod{1,2}_20260711.json`. A cadence
worker may be started before later files exist:

```bash
python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
  --manifest "$RUNTIME_MANIFESTS/phase1_checkpoint_curve_cadence_pod1_20260711.json" \
  --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
  --state-dir /workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/cadence_pod1 \
  --max-active-cpu 6 --wait-for-checkpoints
```

In wait mode each pre-registered path must appear and keep the same size/mtime
for five seconds before hashing. Before every launch the worker rechecks the
judge SHA and both clean commits; it never scans arbitrary `model_*.pt` files
or changes a running trainer. Jobs are ordered so paired causal milestones are
consumed together and future fresh milestones follow every 2000 iterations.

The additional 18 scale-out arms have deterministic manifests generated from
the actual run-directory bindings rather than hand-copied paths:

```bash
python3 hope_training/whole_body_tracking/scripts/generate_phase1_scaleout_curve_manifests.py --check
```

Run two independent wait queues per Pod so a causal terminal checkpoint cannot
block an already-ready fresh milestone:

```bash
for queue in causal fresh; do
  state="/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/scaleout_${queue}_pod1"
  mkdir -p "$state"
  nohup setsid python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
    --manifest "$RUNTIME_MANIFESTS/phase1_checkpoint_curve_scaleout_${queue}_pod1_20260711.json" \
    --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
    --state-dir "$state" --max-active-cpu 6 --wait-for-checkpoints \
    >"$state/worker.log" 2>&1 </dev/null &
  pid=$!
  printf 'queue=%s pid=%s pgid=%s\n' "$queue" "$pid" \
    "$(ps -o pgid= -p "$pid" | tr -d ' ')"
done
```

Do not add `wait` to that loop: both workers must remain independent. The
original seed-1/2 cadence is split the same way. Its causal manifest remains
`phase1_checkpoint_curve_cadence_podN_20260711.json`; the Pod1 fresh-only
manifest starts at 4000, while Pod2 starts at 6000, each with a separate state
directory. This prevents an original causal terminal from blocking later
original `SZ` milestones.

Use the matching `pod2` manifests on Pod 2. The four files cover exactly the
18 newly launched arms and 142 clean q10 jobs: causal seed 2 at
`18000/19000/20000/20998`, and fresh at `2000/4000/.../16000/16999`.
They are milestone-major direction screens only. Their metadata explicitly
sets `screen_only=true` and never authorizes stop/promotion; use a separately
frozen q50 schedule for decisions. `SZ` is the only formal target, `SP` is an
inexact non-target plant diagnostic (non-zero PhysX friction has no exact
MuJoCo `frictionloss` equivalent), and causal plus `LZ/LP` remain inexact
diagnostics. Generated inexact jobs carry only the whitelisted
`--exam-extra --allow-inexact-contract` escape.

2026-07-13 runtime note: after these curves existed, the human owner separately authorized an
operational resource prune. Eight repeatedly collapsed trainer runs were stopped after checkpoint/
contract/log preservation, while eight continued. This does **not** reinterpret the manifest as a
q10 stopping protocol and does not change either `screen_only=true` or any q50
`whole_arm_stop_allowed=false` field. The exact runtime decision and retained artifacts are in
[EXP-P1-FACE-PLANT-SCALEOUT](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md); process ownership
and exact-PGID procedure are in
[run_on_runpod.md](run_on_runpod.md#已登记-phase-1-实验臂的算力释放).

The checked-in curve worker requires `screen_policy` on every manifest,
requires `schedule_k == 2 * attempts_per_side`, compares that schedule (and
optional seed/noise constants) with every job, and records both the complete
manifest SHA and canonical screen-policy-plus-job contract SHA in state. Only
the latter gates per-job reuse, so appending an unrelated later job does not
invalidate a completed result; any change to that job or its screen policy is
rejected rather than silently skipped. The historical `initial`/`fresh_retry`
manifests predate this
schema discipline; do not restart them with the new worker without first
migrating them to an explicit screen policy and a new state directory. Current
live workers remain on their pinned clean eval checkout until they exit; never
edit that checkout underneath them.

For continuations that resume at iteration 16999 and execute 4000 updates, the
runner's terminal checkpoint is `model_20998.pt`; `model_20999.pt` is never
written. Do not infer a terminal filename by adding 4000 to the resume label.
The checked-in manifests and their deterministic generator encode 20998 and a
regression rejects 20999.

### Causal-triangle slot refill (2026-07-11)

The four second-wave followups are frozen in
`configs/phase1_causal_followups_20260711.json`. They fill only naturally idle
trainer slots and do not edit the original 24 recipes: Pod1 GPU1/GPU0 run M3
S1-only guidance-0 seed1/2; Pod2 GPU0/GPU1 run M2 S1+guidance-`-0.95`
seed1/2. Pod1 M3 seed2 additionally requires exact predecessor PGID `1310472`
to be absent plus a stable M3-old 20998 terminal. The gate is read-only and
never signals that predecessor.

Deploy the config and launcher under the external control root, never inside
the live training checkout, then verify the bytes explicitly:

```bash
CONTROL=/workspace/codexschema/phase1_fresh_20260711/control/causal_followups_v1
CONFIG="$CONTROL/phase1_causal_followups_20260711.json"
LAUNCHER="$CONTROL/launch_phase1_causal_followups_20260711.py"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = \
  050d6047fee280feb5754ec568c043fb20e468f81ef049b7420f90ec81a0efc8
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = \
  ca69e1cb90668060f150a518d9cee254f3883a80a07683c4fdfe1f3e4e071b08
```

Run read-only validation first, one exact arm at a time:

```bash
/usr/bin/python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 050d6047fee280feb5754ec568c043fb20e468f81ef049b7420f90ec81a0efc8 \
  --expected-launcher-sha256 ca69e1cb90668060f150a518d9cee254f3883a80a07683c4fdfe1f3e4e071b08 \
  --pod pod1 --arm phase1_M3_S1_only_guidance0_seed1 validate
```

Only after validation passes, replace the final `validate` with `launch`.
Repeat with the arm's registered Pod; do not edit its GPU from the command
line. The launcher rechecks clean train `6d93bcb...` and eval `46a0ce2...`,
all artifact/tool SHAs, GPU compute/trainer count and free memory, atomically
claims a never-used run directory, starts one isolated trainer PGID, validates
the emitted hard-contract, materializes the five q10 jobs and starts one
isolated checkpoint worker. On a post-start failure it may signal only those
new, sidecar-and-`/proc`-bound PGIDs. It contains no broad kill, checkout
mutation or real-robot path.

On the first read-only Pod validation, this capacity gate correctly prevented
all writes but exposed a driver reporting detail: `nvidia-smi` returned every
compute PID twice. Launcher `ca69e1cb...` de-duplicates PID rows before
counting unique compute/trainer processes; three unique trainers still allow
the fourth slot, while four unique processes still fail closed. Do not deploy
or authorize the superseded `dca9b9df...` launcher.

The original `model_16999.pt` is only an SHA-bound parent reference. Never copy
it into the new training run beside the new hard-contract sidecar: doing so
would launder checkpoint lineage. New cadence starts at 17000. Every q10 job is
screen-only and cannot stop/promote; the generated q50 file has no jobs and
remains inactive until its preregistered paired-evidence trigger is recorded.

The first followup 17k states were produced by eval `46a0ce2`'s legacy worker
SHA `8b980359...`. Their commands/results are correct, but that worker predates
the screen-policy/job-contract state binding. Replace only these four idle
workers with the external hardened worker; do not switch or edit either Git
worktree:

```bash
CONTROL=/workspace/codexschema/phase1_fresh_20260711/control/causal_followups_v1
HARD_CONFIG="$CONTROL/phase1_curve_worker_hardening_20260711.json"
HARD_TOOL="$CONTROL/replace_phase1_curve_workers_20260711.py"
HARD_WORKER="$CONTROL/phase1_checkpoint_curve_worker_hardened_21e3015.py"
test "$(sha256sum "$HARD_CONFIG" | awk '{print $1}')" = \
  d270ebb2d2e3fe45510cc1638f64841e9715f0cdccdd9fc983a61e42d5655a58
test "$(sha256sum "$HARD_TOOL" | awk '{print $1}')" = \
  d0678af285af42e16ec133e8d739ff3ce3cec0e8e3e4e39a5a973c0cc1a621ad
test "$(sha256sum "$HARD_WORKER" | awk '{print $1}')" = \
  21e301533328cad2a6684acced85fec6bb6854225eb18ca673247386f059f0eb

/usr/bin/python3 "$HARD_TOOL" \
  --config "$HARD_CONFIG" \
  --expected-config-sha256 d270ebb2d2e3fe45510cc1638f64841e9715f0cdccdd9fc983a61e42d5655a58 \
  --expected-tool-sha256 d0678af285af42e16ec133e8d739ff3ce3cec0e8e3e4e39a5a973c0cc1a621ad \
  --pod pod1 validate
```

Use `pod2` separately. `validate` is read-only and must show both exact legacy
worker PGIDs as single-member and childless. If either has a judge child, stop:
the tool does not wait for or signal it. Only then replace the final word with
`replace`. The Pod transaction rechecks both workers before its first signal,
sends TERM only to those two exact worker PGIDs (never KILL), freezes the old
17k state/sidecar/final log, starts the SHA-pinned standalone worker with the
same manifest and a never-used state directory, and rejudges 17k. Completion
requires rc=0 plus exact manifest/job/job-contract SHAs. It never manages a
trainer or judge; old evidence remains immutable beside a correction sidecar.

This one-time correction completed on 2026-07-11. Hardened worker PGIDs are
Pod1 `1416771/1416784` and Pod2 `198759/198771`; correction-sidecar SHAs are
`2faf88de...ffe3`, `1d6f8ba3...bae9`, `0dd02fae...d165`, and
`45f4334d...0ad`. All four 17k jobs were rejudged rc=0 with manifest, job spec
and job contract SHAs present. Do **not** rerun `replace`: its legacy-worker
precondition is intentionally no longer true. For current monitoring, read
each `checkpoint_cadence_q10.worker.hardened.launch.json` and manage only its
recorded PGID.

The six older global workers were separately replaced under
`configs/phase1_global_curve_worker_hardening_result_20260711.json`. Their
current PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`. Do not rerun that replacement transaction either;
monitor the recorded launch sidecars and signal only an exact worker PGID if a
later, separately authorized repair requires it.

Before copying or launching any curve manifest, run:

```bash
python3 scripts/validate_phase1_queue_governance.py
```

For a separately supplied milestone-major manifest, validate it explicitly:

```bash
python3 scripts/validate_phase1_queue_governance.py \
  --manifest /absolute/path/to/manifest.json \
  --require-readiness-barrier
```

The validator requires q10 K=20/10 per side, screen-only/no-stop/no-promotion,
ordered milestones and barriers. It rejects q50 from the generic worker; q50
must use a preregistered paired runner.

### Paired terminal q50 runner

Do not turn a q10 trigger into an ad-hoc `judge.sh` command. The M3 terminal
paper is executed by `scripts/run_phase1_paired_bank_q50.py`, which separates
`prepare` (materialize one immutable schedule, start nothing) from `run`
(require the prepared runtime-contract SHA and validate both complete ledgers).
The accepted v2 bytes are runner
`095e476fd36fb68d500cb39ea7f71f6fee9b729209187d51599582c72c22198b`
and execution config
`550ca88988c88e94e626aed3e489cbedf981d2b32cde1bab9601ebacae05988b`.
It forces causal/inexact/non-formal semantics, K=100, 50 per side, one shared
schedule JSON, exact question order and zero censored attempts. It never
signals a process.

The 2026-07-11 M3 paper is already complete; do not rerun its no-clobber state
root. Schedule file SHA is `69f73458...7f25`, semantic schedule SHA is
`949eb196...8fc0`, runtime-contract SHA is `ca7a688a...17b2`, and paired-result
SHA is `e9bb07d3...f56e`. M3-old versus M3-S1 FH/BH/aggregate return was
`0.62/0.22/0.42` versus `1.00/1.00/1.00`, with 9 versus 0 physical falls.
The result selects S1 only in this same legacy swing-family causal paper and
the completed Isaac companion does not reproduce the ranking: both old and S1
score `0.99` aggregate on the same order. Therefore no cross-engine selection
gate closes. Full paths, the preserved fail-closed attempts and all hashes are
in `configs/phase1_M3_terminal_q50_result_20260711.json` and
`configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The fresh exact wrapper is
`scripts/run_phase1_fresh_exact_paired_bank_q50.py`. It additionally requires
fresh lineage, a shared schema-3 hard-contract SHA and no inexact escape. The
accepted seed1 model-2000/model-4000 state root is already complete and must
not be reused. Runtime-contract SHA is `a756023d...4661`, schedule semantic
SHA is `7dc6af82...ff3e`, and paired-result SHA is `b95ba6c4...0478`.
Returns were `0.66/1.00/0.83` versus `0.00/1.00/0.50`; retain model 2000 but
continue the arm. Both cells' post-strike guard resets mean this is not a
continuity/deploy gate. Full paths and hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`.

The completed fresh/exact Isaac companion consumed that same schedule file
and semantic SHA. Both checkpoints scored `0.99` aggregate (`0.98/1.00` by
side), one guard reset and zero physical falls; it does not reproduce the
MuJoCo separation. Do not interpret the earlier-checkpoint final tie-break as
cross-engine validation. Runtime-contract SHA is `63580328...b8120`, paired
result SHA is `65c08723...c18e`, and the full bindings are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

The current 10-second, no-wrap-teleport task does carry the robot state between
clips, but its complete-clip timing is slower than the conservative venue
A-B-A intervals. Do not claim that this pool proves arbitrary-time continuous
play, and do not change its live recipe. The offline reproduction command,
timing gap and separate `T0/T1` event-driven design are in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`.

Schema 3 has two validation levels. Structural validation is sufficient to
export a hash-bound diagnostic checkpoint whose motion is explicitly inexact;
it never promotes the metadata exact flag. A checkpoint whose embedded lineage
flag claims exactness must additionally pass the formal schema-2 motion/body
order gate. Removing or moving `params/training_contract.json` is not an
escape: a checkpoint that claims a binding while its adjacent sidecar is
missing is rejected. `judge.sh` likewise reads only that adjacent sidecar to
restore zero friction and the actor layout.

For a diagnostic sidecar (`motion_kinematics_exact=false` or the explicit
legacy face pairing), `judge.sh` adds `--allow-inexact-contract` to MuJoCo and
prints that decision in its preflight. A fresh exact candidate receives no
escape. Legacy schema-1/2 or missing-contract runs are also diagnostic. The
Isaac export subprocess activates the requested venv and then sources this
checkout's `setup_train_env.sh`, replacing `PYTHONPATH` with
`HOPE_WBT_PYTHONPATH`; never let a user-specific Pod env select another
checkout's task package.

`hydra`, `omegaconf`, and `rsl_rl` are NOT in the package `setup.py` `install_requires`; they must be importable from the Isaac Lab Python (provide via Isaac Lab itself or `HOPE_ISAAC_VENV_SITE`). Install the package into that Python:

```bash
hope_isaac_py -m pip install -e source/whole_body_tracking
```

Quick sanity check (expect `hydra` 1.3.2):

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Fresh Machine Entry

For a new computer, start from `docs/START_HERE.md`, then use this operation doc for Isaac training setup and [setup_local_sync.md](setup_local_sync.md) for ignored/private assets. Use [../../reimplement.md](../../reimplement.md) only as the long-form runbook when a gate or operation doc points at a specific step, such as the A3 URDF copy in Step 12.7 or the motion pipeline in steps 9-12.

Minimum order for a training machine:

1. Install Isaac Sim/Lab and set `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` in `setup_train_env.local.sh`.
2. Source `hope_training/whole_body_tracking/setup_train_env.sh`.
3. Restore or create the A3 Isaac URDF asset and motion references listed below.
4. Run the smoke commands in this doc before starting a long training run.

## WandB Setup

WandB is optional for local motion-file training, but useful for shared/internal run logging and registry-backed motion distribution. If you use the registry, WandB needs two DISTINCT identities; they MUST differ or motion-registry reads fail with `Unable to find organization for entity ...`.

Current shared RunPod values (verified 2026-07-02) — export them in the git-ignored `setup_train_env.local.sh` so sourcing `setup_train_env.sh` picks them up:

- `WANDB_ENTITY=BerkeleyPingPong` — team/entity for run logging.
- `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org` — org for the motion registry.
- `WANDB_PROJECT=hope_wbc` — training project.
- `WANDB_MOTION_PROJECT=csv_to_npz` — motion upload project.
- `WANDB_DIR=/workspace/yikang/nohope/hope_training/wandb` — local W&B cache.

On any other machine:

```bash
wandb login
export WANDB_ENTITY=your-wandb-team
export WANDB_REGISTRY_ORG=your-wandb-org
export WANDB_PROJECT=hope_wbc
```

Run `wandb login` before registry-backed training. The API key is stored outside git (observed in `/root/.netrc`); never write it into repo files. No WandB account or testing on a fresh box? Pass `logger=tensorboard` and `motion_file=...`; this explicit smoke path needs no login or registry.

## Local Assets Needed For This Task

Before smoke tests or training, the A3 Isaac asset must exist at the path expected by `robots/agibot_a3.py`:

```text
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

If it is missing, create it from the tracked Agibot ping-pong URDF package:

```bash
cd ~/workspace/HOPE
python3 scripts/prepare_a3_isaac_asset.py --force
python3 scripts/prepare_a3_isaac_asset.py --check
```

The script copies meshes/config from `agi/URDF/A3T2.5-URDF-std-pingpang/`, rewrites `package://.../meshes` URDF references to `../meshes/...`, and checks that the generated URDF references existing meshes including `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.

Motion references are also task-local setup. Registry-backed runs can use the WandB names from `cfg/task/*.yaml`, while local generated `.npz` files take precedence when passed explicitly:

- Internal registry paths such as `registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand"`.
- Local ignored `.npz` motion files under `hope_training/motions/preprocessed/` or `hope_training/whole_body_tracking/artifacts/`, passed as `motion_file=...` and optional `motion_file_2=...`, with the exact paths recorded in G05 when used.
- To create those files from manual videos, run the `reimplement.md` Step 9-12 flow: raw video -> GVHMR -> GMR (`--robot agibot_a3`) -> `scripts/csv_to_npz.py --robot agibot_a3 --output_file ../motions/preprocessed/<name>.npz`. Use `--upload_wandb` only if you also want registry artifacts.

The currently verified clips (2026-07-02) are the corrected HOPE +X local files
`hope_training/motions/preprocessed/hope_forehand_hopex.npz` and
`hope_training/motions/preprocessed/hope_backhand_hopex.npz`, passed as `motion_file=` / `motion_file_2=`. The older `hope_forehand:v4` / `hope_backhand:v4` registry artifacts face world +Y and fail the alignment gate below.

Optional R15 v5 ablation clips live on the team RunPod at `/workspace/shared/motions/hope_forehand_v5.npz` and `/workspace/shared/motions/hope_backhand_v5.npz`. Copy them into `hope_training/motions/preprocessed/` only for the R15 arm, then pass all phase and box changes on the CLI:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand_v5.npz \
  motion_file_2=../motions/preprocessed/hope_backhand_v5.npz \
  'task.racket.strike_phase_per_clip=[0.673,0.345]' \
  'task.racket.pos_range_per_clip.forehand.x=[0.29,0.49]' \
  'task.racket.pos_range_per_clip.forehand.y=[-0.63,-0.43]' \
  'task.racket.pos_range_per_clip.forehand.z=[0.74,0.94]' \
  'task.racket.vel_range_per_clip.forehand.x=[0.74,1.74]' \
  'task.racket.vel_range_per_clip.forehand.y=[0.71,1.71]' \
  'task.racket.vel_range_per_clip.forehand.z=[1.20,2.20]' \
  'task.racket.pos_range_per_clip.backhand.x=[0.60,0.80]' \
  'task.racket.pos_range_per_clip.backhand.y=[0.12,0.32]' \
  'task.racket.pos_range_per_clip.backhand.z=[0.81,1.01]' \
  'task.racket.vel_range_per_clip.backhand.x=[2.60,3.60]' \
  'task.racket.vel_range_per_clip.backhand.y=[0.50,1.50]' \
  'task.racket.vel_range_per_clip.backhand.z=[1.66,2.66]' \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=r15_v5_local_smoke
```

Before copying these values into any config, rerun `python scripts/analyze_strike_phase.py --clip forehand:../motions/preprocessed/hope_forehand_v5.npz --clip backhand:../motions/preprocessed/hope_backhand_v5.npz`; it should print `task.racket.strike_phase_per_clip=[0.673,0.345]`. Video/GVHMR face normals are wrist +Y proxies and are marked unreliable.

Use registry paths only after `scripts/check_motion_target_alignment.py --clip ...` passes for those downloaded artifacts. Do not commit generated logs, checkpoints, WandB caches, or motion artifacts unless the asset policy changes.

## Video-To-Motion Doc Map

Use this order when generating new reference clips from manually imported videos:

1. Restore local-only motion tooling and model/checkpoint assets: [setup_local_sync.md](setup_local_sync.md) steps 6-8.
2. Run the long-form command sequence: [../../reimplement.md](../../reimplement.md) steps 9-12.
3. Confirm local outputs exist:
   `hope_training/motions/preprocessed/hope_forehand.npz` and
   `hope_training/motions/preprocessed/hope_backhand.npz`.
4. Scrub the source video frame-by-frame for ball contact, record the result in `cfg/strike_annotations.yaml`, and rerun `scripts/analyze_strike_phase.py`; do not promote the speed peak by itself.
5. Replay with `scripts/replay_npz.py --motion_file ...`, then train with `motion_file=... motion_file_2=...`.
6. Add `--upload_wandb` / `registry_name=...` only for shared registry runs.

## Smoke Test

`TrackingFlat` needs a reference motion but no motion registry and no WandB account, so it is the cleanest smoke test once you have a local `.npz`:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success means the env builds, PPO prints learning iterations, and rewards remain finite.

Before any HOPE task smoke, run the no-Isaac motion/target gate (it defaults to the local `_hopex` clips; pass `--clip name:path.npz` to check other files, e.g. registry downloads):

```bash
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPongRealSensor.yaml
```

Both commands passed on 2026-07-02 using `hope_forehand_hopex.npz` / `hope_backhand_hopex.npz` against the pre-merge branch YAML values; re-run them against the merged uniform-target config before the next long run. The same check intentionally fails on the old v4 registry downloads because frame-0 yaw is 82.03/85.92 deg and strike velocity is +Y-dominant.

For R15 v5, use `scripts/analyze_strike_phase.py` with `cfg/strike_annotations.yaml` instead of the +X-dominance gate: the forehand contact is hand-verified at frame 37 / phase 0.673, while the later speed peak is the known whip trap.

Local corrected-clips smoke for the unified HOPE task:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand_hopex.npz \
  motion_file_2=../motions/preprocessed/hope_backhand_hopex.npz \
  num_envs=32 max_iterations=1 logger=tensorboard run_name=smoke_hopex_local
```

Registry + WandB smoke `6xus13ga` on 2026-07-02 finished and synced `model_0.pt`, ONNX, config, diff, output log, and summary, but used the later-rejected v4 +Y-facing motions. Keep it as pipeline evidence only, not motion-quality evidence.

## Table-Tennis Physics Scene Smoke Test

This is a G04 scene/physics check, not the accepted G05 WBC baseline. It loads
`HOPE-TableTennis-AgibotA3-v0`, serves a ball from the P2 half toward the P1-side A3, and verifies the
table/net/ball frame plus drag/Magnus hooks.

```bash
hope_isaac_py scripts/play_table_tennis.py
hope_isaac_py scripts/play_table_tennis.py --num_envs 9
hope_isaac_py scripts/play_table_tennis.py --fix_base
hope_isaac_py scripts/play_table_tennis.py --enable_aero
hope_isaac_py scripts/play_table_tennis.py --magnus 0.1
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
```

Expected behavior: by default the ball arcs under PhysX gravity/contact only, bounces on the table, and travels toward the robot. With `--enable_aero` or `--magnus`, the HOPE drag/Magnus callback also affects flight. With `--fix_base`, the pelvis stays pinned for a stable visualization. Without it, the robot may drift or fall because no balance/return policy exists yet.

Default table-tennis scene behavior follows Purdue PACE parity: the ball uses PhysX gravity plus contacts with ball mass `3.4 g`, ball/table restitution/friction `0.9/0.1` and `0.95/0.4`, and aero drag off. Pass `--enable_aero` to use the HOPE-calibrated drag callback; `--magnus` also enables aero and adds spin.

## Baseline Training Commands

Plain tracking first:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  logger=tensorboard run_name=forehand_tracking
```

HOPE racket task, unified forehand+backhand policy from local Step 9-12 motions:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  logger=tensorboard run_name=hope_local_unified
```

Registry-backed equivalent:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  registry_name_2="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand" \
  run_name=hope_registry_unified
```

Useful overrides:

```bash
num_envs=4096 max_iterations=20000 seed=1
```

`HOPEPingPong` defaults to a unified policy: clip 0 comes from `registry_name` / `motion_file`, clip 1 comes from `registry_name_2` / `motion_file_2`, and the actor receives `swing_type`. The HOPE task YAMLs also set `motion.wrap_teleport: false` (the code default), so a mid-episode clip wrap resamples the reference clip/time and racket target without teleporting the simulated robot; episode reset still uses RSI.

Resume / curriculum hand-off (added on `train_1`): `checkpoint_path=<model.pt>` loads weights + optimizer
from a prior run and CONTINUES training (the iteration counter resumes). Use it to apply a staged config
change — e.g. tightening `racket_velocity_std` — without throwing away progress:

```bash
checkpoint_path=logs/rsl_rl/agibot_a3_hope/<run>/model_2000.pt
```

Single-swing policy, if you deliberately want only one clip:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=null logger=tensorboard run_name=hope_forehand_local_smoke
```

At startup, `scripts/train.py` prints:

- the imported `whole_body_tracking` path,
- the composed env-cfg source file,
- every applied task override from `cfg/task/<name>.yaml`,
- the post-override racket reward stds and target-sampling knobs.

If a YAML key targets a missing env-cfg attribute, training now raises instead of silently ignoring the
override. Treat these printed lines as part of the G05 verification record.

### ppo.yaml deltas on this branch

- `max_iterations: 25000` is now a finite safety default. Formal launchers still bind an explicit
  per-run budget; direct Hydra launches may override it deliberately, but an omitted override no
  longer starts an effectively unbounded run.
- `save_interval` 500 -> 100.
- `entropy_coef` is 0.01; treat `cfg/algo/ppo.yaml` as the source of truth for the current value.

### Racket Target Sampling

`HOPEPingPongDeployParity.yaml` defaults to `racket.target_mode: uniform`, matching the HITTER
structure. The command term samples per-clip 3-D blade-centered position AND velocity boxes
(`pos_range_per_clip` / `vel_range_per_clip`); the earlier fixed hit plane `x=0.4` with (y,z)-only
sampling is superseded. The imitated clip supplies the motion prior and, in the unified policy, the
swing type.

```yaml
target_mode: uniform
pos_range_per_clip:
  forehand: {x: [0.58, 0.78], y: [-0.64, -0.24], z: [0.72, 0.92]}
  backhand: {x: [0.56, 0.76], y: [-0.07,  0.33], z: [0.93, 1.13]}
vel_range_per_clip:
  forehand: {x: [1.05, 2.05], y: [ 0.96, 1.96], z: [0.31, 1.11]}
  backhand: {x: [1.61, 2.61], y: [-1.21, -0.21], z: [0.00, 0.71]}
strike_phase_per_clip: [0.47, 0.333]
```

Strike phases are blade-speed-peak detected per clip version (`scripts/analyze_strike_phase.py`). The
values above come from the 2026-07-02 blade re-detect on the re-grounded `_hopex` v3 clips; the old
`[0.36, 0.74]` values were for the v1 clips.

For two local clips, `MotionLoader` concatenates the files in order: clip 0 is forehand and clip 1 is
backhand. Keep `motion_file` / `motion_file_2` ordered the same way as `strike_phase_per_clip`.

`target_mode: reference_perturbed` remains available as a NON-default option (it was the pre-merge
default on the `rsi-on-wrap-progress-fix` branch). It centers the initial target on each imitated clip's
own PER-CLIP strike-frame racket FK state (selected by `motion.clip_id`), so the teacher action and
training target start aligned, and widens the distribution only when the success-gated exact-strike
metric advances `ref_perturb_scale` (`ref_perturb_curriculum_start: 0.05`,
`ref_perturb_pos: [0.15, 0.20, 0.15]`, `ref_perturb_vel: [1.0, 1.0, 0.8]`). Use it only for controlled
comparisons against the uniform default.

For the real-sensor footwork variant, `racket_progress` is zeroed on motion/target resample steps and its previous-distance baseline is reset. This prevents clip wrap or reset from contributing a fixed progress penalty/reward that the policy cannot control.

## Live Training Telemetry

`MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) logs a `Live/...` dashboard to WandB/TensorBoard every PPO iteration. Namespaces:

- `Live/<command_term>/<metric>` — per-axis command tracking (reference vs robot anchor pos/vel per x/y/z, joint error mean/max, `motion_phase`, racket pos/vel/normal per axis, `time_to_strike_s`, `pre_strike_flag`, `strike_window_flag`, `racket_speed`, ...).
- `Live/Reward/<term>` — per-reward-term contributions.
- `Live/Termination/*`, `Live/Action/*`, `Live/Env/*`.

The real "is it learning to hit" signal is the exact-strike metric group:
`strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`,
`strike_composite_success_exact`, and `exact_strike_sample_count_decayed`. These are conditional
sample-weighted pass rates at the exact strike frame. `strike_success` and the `*_at_strike` metrics are
still useful, but episode-wide errors are diluted by the long non-strike phase, so do not judge progress
from them.

## Reward Shaping (strike_success=0 fix)

The reward kernel is `exp(-||err||^2 / std^2)`. With `std` set to the final acceptance tolerance, the reward is ~0 for any early error (a 50 cm error gives `exp(-44) ~ 0`), so there is no gradient and `strike_success` stays stuck at 0. The target-sampling fix above handles unreachable targets; the reward shaping here handles too-narrow early rewards.

The current `HOPEPingPongDeployParity.yaml` values try to keep useful gradients in the observed error
band while preventing non-hit rewards from dominating:

- `racket_position_weight: 14.0`, `racket_position_std: 0.20`
- `racket_velocity_weight: 10.0`, `racket_velocity_std: 1.0` (curriculum-tightened from 1.8 as the observed velocity error fell; the plan is 1.0 -> 0.8 -> 0.5)
- `racket_normal_weight: 5.0`, `racket_normal_std: 0.30`
- `base_position_weight: 2.5`, `base_position_std: 0.25` — legacy `HOPEPingPong` task only; the deploy-parity default REMOVES the base_position term entirely (base-free footwork: dense `racket_progress` plus pre-strike stability penalties)
- regularization: `joint_torques_weight: -0.00003`, `action_rate_weight: -0.10`, `joint_limit_weight: -10.0`, and `undesired_contacts_weight: -0.1`
- `base_decel_weight: 0.0` (P2.4, OFF by default; trial weight 1.0) — PACE-style pre-strike base
  speed shaping: `exp(-(||v_base_xy|| - v_des)^2 / base_decel_std^2) * pre_strike` with
  `v_des = clamp(base_decel_v_gain * planar_dist(racket→target), 0, base_decel_v_max)` (defaults
  `v_gain 2.0 /s`, `v_max 1.6 m/s`, `std 0.4 m/s`). Uses racket→target planar distance, NOT base
  position (deploy-parity obs and the base-free reward structure stay untouched); gated dead at
  and after the strike frame so it never commands a speed-up toward the swung-through old target.
  Speed-magnitude-only v1; the v2 spec (fitted accel/decel envelope, direction term, time budget,
  stroke-amplitude coupling) is `docs/motion_and_contract_v3.md` §5. Watch:
  `base_speed_xy_prestrike`.
- `joint_velocity_limit_hinge_weight: 0.0` and `joint_velocity_limit_hinge_margin: 0.85`
  ([关节速度限位铰链惩罚](../DEFINITIONS.md#qdot-limit-hinge)，VirtualBall only，默认关闭)：
  `mean(relu(abs(qd)/joint_velocity_limits - margin)^2)`。启用 weight 必须 `<= 0`；实现读取
  `robot.data.joint_vel` 和同一 31-joint articulation order 的实际 `joint_vel_limits`，不是
  `action_rate_weight` 的别名。启动会打印两个 applied marker；关节重排/缺失、零/非有限 limit 或
  每个 environment 的 limit 不一致都 fail closed。
- `racket_face_conditional_guidance_weight: 0.0`
  ([不逃离就绪区的固定预算 Reward](../DEFINITIONS.md#conditional-face-guidance)，默认关闭)：
  只在 wide strike window 内收费；位置误差用 `9.5→7.5 cm`、完整拍速向量误差用
  `1.0→0.5 m/s` 形成连续就绪门。未就绪时成本固定为 1；进入门后按就绪度把这份成本换成拍面误差
  （15° 内为 0，180° 为 1）。因此位置或拍速越就绪，成本只会不变或下降，不能靠故意退到门外免罚；
  门外拍面梯度为零。函数输出 `[0,1]`，weight 必须 `<=0`，所以 `|weight|` 是每个时间窗 step 的
  硬预算。开启时会记录 `face_conditional_guidance_gate`、
  `face_conditional_guidance_error_fraction` 与 `face_conditional_guidance_cost_fraction`；`+200` 若
  gate 全程为零，说明没有真正获得拍面纠偏信号。公式、配对与 `+200/+500/+1000` 门见
  [实验卷宗](../experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)。

  首轮只能通过 paired lean YAML 点火：control/treatment 均显式固定历史 static guidance 为 `0`，
  treatment 只设置 `++task.rewards.racket_face_conditional_guidance_weight=-0.4`，control 设 `0.0`；
  两格必须使用包含该 hard-contract 字段的同一 source。不要拿旧 source checkpoint 当 control，
  也不要同时扫 gate 或 weight。

qdot-limit treatment 尚未选择采用的负 weight，也没有 machine prereg，因此不要直接把下面写成临时
CLI 点火。未来 paired manifest 必须让 control/treatment 从同一父 checkpoint 各自启动，并同时冻结：
exact `task.rewards.joint_velocity_limit_hinge_weight`、margin、完整 argv、source commit、outer
`training_launch_claim_sha256`，以及 emitted hard contract 的
`joint_velocity_limit_hinge_reward` 和 31 项 `joint_names/joint_velocity_limits`。不得从 treatment 的
中间 checkpoint 再派生 control，也不得用 action-rate 代替这一轴。

<a id="恢复期腿腰-processed-q_des-slew-wave-a"></a>
### 恢复期腿腰 processed-q_des slew（Wave A）

这里的 [Wave A](../DEFINITIONS.md#balance-stability-waves) 是 action-slew 单变量波；静态 v4rg 下半身参考或
non-demo stability constraint 属于另行冻结的 Wave B。M0 四份横移 exact-GMR 虽存在，但 4/4
`stance_passed=false` 且 formal/schema2/training/hardware 全部 false，本轮禁止把它们当 moving teacher。

现役 [`action_rate_l2`](../DEFINITIONS.md#raw-action-rate-l2) 权重是 `-0.10`：50 Hz 下每 tick 计算
`sum((action[t]-action[t-1])^2)`，再由 RewardManager 乘 `0.02 s`。它只读 affine transform/clamp 之前的
31 维 raw action，但每步连续连接，不是只发生一次。历史 `-0.05` 对 `0` 已证明它能显著压低 action delta、
qdot max 和 base pitch；完成/回台/摔倒有交叉取舍，详见
[半秒冲刺 action-rate 回收](../experiments/2026-07/EXP-P1-HALF-SECOND-SPRINT.md#2026-07-20-action-rate-证据回收)。

新 [`processed_qdes_slew_hinge`](../DEFINITIONS.md#processed-qdes-slew-hinge) 默认关闭。它读取 affine
transform 和 train=deploy clamp 之后的 q_des，仅在同一拍触球后 `0.20–1.55 s` 对 exact 3 腰+12 腿关节
计算：

```text
u_j    = abs(q_des[t,j] - q_des[t-1,j]) / (qdot_limit[j] * 0.02)
tail_j = 1 - exp(-(relu(u_j - 0.85) / 0.15)^2)
value  = mean(tail_j over 15 joints)
```

reset 后首步无有效 previous q_des，必须 mask 为零。预注册 treatment weight `-0.25`；`value<1` 且
RewardManager 乘 `0.02 s`，所以满激活时每 tick 幅值小于 `0.005`、每 eligible 秒小于 `0.25`。这允许
击球臂快速变化，也不在击球前/触球窗收费。控制频率不是严格 50 Hz、31-joint order/15-joint 集合漂移、
速度上限非法或 probe 与 Reward 参数不同都会 fail closed。

四个 Hydra key 首次出现时的人话如下，统一链接上面的术语定义：

- `task.rewards.processed_qdes_slew_hinge_weight`：启用/关闭腿腰执行目标尾部惩罚，必须非正；
- `task.rewards.processed_qdes_slew_hinge_margin`：不收费区的归一化阈值，本轮固定 `0.85`；
- `task.rewards.processed_qdes_slew_hinge_recovery_start_s`：同一拍触球后开始收费时刻，本轮 `0.20 s`；
- `task.rewards.processed_qdes_slew_hinge_recovery_end_s`：同一拍停止收费时刻，本轮 `1.55 s`。

不要手写长 Hydra 命令。Wave A 的 W/V×C/N/H 六格、parent、资产、槽位、预算与 intentional
[`checkpoint_allow_contract_mismatch=true`](../DEFINITIONS.md#checkpoint-contract-mismatch)都只从
[`configs/phase1_balance_action_slew_20260720.yaml`](../../configs/phase1_balance_action_slew_20260720.yaml)
读取。远端训练 checkout 必须是 clean detached
`54c9a62656f0e60e5bb41cbcfa0e5a972b793906`；所有 child 是 diagnostic continuation、永久
formal-ineligible。

先做 dependency-light 检查和默认 **NO-LAUNCH** plan：

```bash
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$BALANCE_REPO_ROOT"

python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_reward_flags_mdp.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py \
  tests/test_run_phase1_balance_action_slew_queue.py

python3 scripts/run_phase1_balance_action_slew_queue.py --stage probe
```

`--stage probe` 的人话是“选择 4096-env×24-step/env×2-update 非科学合同探针”；默认调用只验证并打印六格计划，
`commands_emitted=false`，不会 SSH，并明确报告 manifest gate 仍 blocked。当前独立生成并复核的
[`launch manifest`](../DEFINITIONS.md#balance-launch-manifest)：它必须绑定 exact source、当前 queue
config/runner、A3 asset tree、`model.usd` 及完整 6-file sibling bundle tree、两份动作、题库、W/V checkpoint
与 parent contract 的 SHA-256。只复制或只哈希 `model.usd` 会漏掉它依赖的 `configuration/`，必须拒绝。
当前 fresh v9/probe10 的 exact 文件为
`configs/phase1_balance_action_slew_launch_manifest_20260720.json`；文件/content SHA-256 分别为
`664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a` /
`36ceb3c77dc056f4565378a92b03da58865378d86c5849085ba066631cea456c`，所绑定的 config/runner
SHA-256 分别为 `3bf5085ea8396513d162b9cce249dfb761b39b2827ec722959343c953683e59e` /
`0fff4515cbe7e62798e8c39f701851c46e68287c7321e7618161fa9dde4789ce`。新 no-clobber 根是
`/workspace/codexschema/phase1_balance_action_slew_v9_20260720`。这些 bytes 当前只表示
**source ready / preregistered / not launched**：没有 probe10 命令制品、receipt 或 GPU
runtime。只有它们进入最新 `origin/main` 后才能按下文重新渲染；不得用占位 hash，也不得把
清单存在误写成 probe 已启动。

旧 manifest `d7e95130…a2e47` 只启动过 Pod1 W-C probe2；trainer 自然退出，但 outer verifier 错把
首个 `0.48 s` rollout 的 recovery-eligible=`0` 当成失败，未发布 receipt。旧 namespace 与旧 manifest
都冻结为历史证据，不得补写或重发。manifest `2d3e7955…3bae17` 的 probe3 随后自然完成 W-C/W-N/V-C；
W-C 的旧 receipt 虽有真实 `98304/update`，但 verifier 未把 24-step rollout 写成硬门，三格都不能解锁
train。probe4 W-C 已因错写 `algo.num_steps_per_env=24` 而在建 run-dir/Kit 前被真实 Hydra compose 门
fail closed，没有运行产物或 GPU compute。

probe5 使用正确的
[`algo.runner.num_steps_per_env=24`](../DEFINITIONS.md#ppo-num-steps-per-env)，在 v4 no-clobber 根中发布了
W-C、V-C、W-N、V-N、V-H 五份 exact receipt。W-H 的旧 supervisor 在 `Popen` 后单次读取 child 身份，
恰好命中 fork→exec 的过渡 argv，因而在 trainer binding 前 fail closed；该格没有 first iteration、RSL
checkpoint、receipt 或 GPU training compute，失败后 exact leader group/child 也均 absent。这是启动取证
竞态，不是 H 机制负例。probe5 v4 manifest 的文件/content SHA-256 分别是
`6bfa73587968f8f0af71b5617e8c324f75114b304bbe1452d0b0e4617d1f51bc` /
`093a4cc7a0ce91aad74948ed39b581e5f4a0693ba114f3144062b6cc4386a462`；v4 根与五份 receipt 只作历史，
不能补写或混入当前轮。

probe6 只发射 W-C；`2026-07-20T03:01:10Z`，其 supervisor Python 在 locked launcher child bind 前以
`IndentationError: unexpected indent` 退出。failure transaction helper 曾在已经 shlex-quoted 的 multiline
program 每个换行后加两个空格，因而改变 transaction body bytes。该格没有 leader/child evidence、binding、
terminal marker、checkpoint 或 receipt；PID=PGID `2712318` 经 stable double-scan 确认 absent，Pod1 GPU0
empty、Kit/cache locks free，审计没有发 signal。其余五格未发射，所以这不是机制负例。v5 no-clobber 根
`/workspace/codexschema/phase1_balance_action_slew_v5_20260720` 与 manifest 文件/content SHA-256
`718bbee0a556cc3640ee636e20f8eb2adb293cee8d0bb1820afceebf5ce1a267` /
`3cbb019e9d315abdc687d1635c9deffc13eae9577ee2d820b0e3c30ba9b7cfd8` 冻结为历史，禁止补写或重发。

probe7 的 W-C、V-C、V-N 自然退出并通过 exact verifier receipt。W-N 在
`2026-07-20T03:22:50Z` 到达 RSL 后冻结于 `Starting the simulation`，没有 learning iteration、trainer
binding、terminal checkpoint 或 receipt；其 run.log SHA-256 是
`9c46896bbc2a12324a209635374556ab8a4100cd96acd5e9ad01595cbfaa0e3b`、大小 `22601 B`、最后 mtime
`2026-07-20T03:23:06.408757Z`。locked `180 s` watchdog 只向 exact 身份发送 TERM/KILL 并返回 rc `125`；
随后 exact groups、Pod1 GPU1、Kit/cache locks 全部闭合。V-N 在 W-N 失败被确认前 `6 s` 已经发射，后来仍
自然退出并验证通过；W-H/V-H 未发射。这是 transient infrastructure failure，不是 `action_rate=0`、W/V
parent 或任何机制的负例。v6 根 `/workspace/codexschema/phase1_balance_action_slew_v6_20260720`、三份
receipt、以及 config/runner 和 manifest 文件/content SHA-256
`912bd8d212791d99ce6a6851a8f05c12d182cdfa9d5566e02381f1b4703b8f3c` /
`3fbaf23f97fdb40e05a448f9f769267b21c7cca3bd767aa082c0d5b965ecd7d7` /
`4552fe23abd551d8959a9de05cc5f9d761d0da25eed88138d61fa45cc6558e9e` /
`6e3518d97d48fad550e7971a5178b1f11c15895696f03d30d5a62d1e27741640` 冻结为历史，禁止重试、补写或混用。

probe8 的 v7 只发射了 W-N。它在 `2026-07-20T03:44:19.857Z` 于 Pod1 物理 GPU1 到达
`Starting the simulation`（`sim.reset()` 边界），随后因 `malloc(): invalid size (unsorted)` 触发
`SIGABRT`，并于 `03:44:32.112Z` 结束；trainer terminal status 记录 exit `-6`，外层 transaction 返回 rc `134`。它没有 first
iteration、trainer binding、terminal checkpoint 或 receipt，其余 W-C/W-H/V-C/V-N/V-H 五格从未发射。
失败后 exact leader/child 均 absent、Pod1 全部 GPU 无 compute context、Kit/cache lock 无 holder；可访问的
系统日志与 telemetry 快照未见 Xid/OOM。closure 已闭合，但这不授权重试。v7 根
`/workspace/codexschema/phase1_balance_action_slew_v7_20260720`、失败证据与 config/runner 和 manifest 文件/content
SHA-256 `0c84613f05439237f6e36d37e0c9210984465d928b9c0cba50999bd8995145f9` /
`2bc9d59e21413a812a742529c6f3291f5710c384e0f4af7ee7098f33b25ba17d` /
`887c0b9e097e50300d83eef27e587d112f70132958e6e8d9b68af74437fa7231` /
`13f92d5eda71e90abd6a14a1498c2afd98d3cb825cb26e2f5b74958b5a795f84` 全部冻结为不可变历史，禁止补写、
同 namespace 重发或与后续收据混用。该失败发生在 RewardManager/action reward 生效前，不是 N 机制负例。

probe9 使用 fresh no-clobber 根
`/workspace/codexschema/phase1_balance_action_slew_v8_20260720`，run name 固定为
`phase1_balance_slew_probe9_{job}_seed3_20260720`，仍显式 override `algo.runner.num_steps_per_env=24`，要求
processed-q-des/qdot observed 逐 update 精确为 `98304`，恢复资格只要求两步合计非零。其 supervisor 在
`Popen` 后、首次 `/proc` 读取前先不可覆盖地发布 child evidence，再只等待其中同一个 child PID：最多
`5 s`、每 `10 ms` 一次；每个 identity sample 双读 `/proc/<pid>/stat` 的 PID/PGRP/starttime tuple，
并要求两次结果相同、PGRP 与 `getpgid` 相同。第一次可读 starttime 之后必须不变，所有可读身份必须保持
exact supervisor PGID，且只接受 exact final trainer argv。identity 失败会 no-clobber 持久化最后身份摘要，
任意身份读取异常也必须走该 failure path。实际发射前该 manifest 已进入当时最新 `origin/main`。
该历史 manifest 文件/content SHA-256 是
`688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353` /
`97c36e471fb8fc6b93fe212f20846de6697db518192e7a45c6618e5924947e28`，不得与 fresh v9 混用。

probe9 按 W-N→W-C→W-H→V-C→V-N→V-H 全局严格串行完成；六格均 natural exit，且每格都在
下一格前通过 verifier、不可覆盖 receipt 和 exact process/GPU/lock closure。六份 receipt 文件 SHA-256 为：
W-N `ee8c53780d6bb5f0ce5b0c31032cd8c336bfb63f922ff54b5c7d86fe5788c5ff`，
W-C `b948a4d8c701263a7805768ca7821b4257b07f008500e2aac7a5a237415018d5`，
W-H `a80502c9d9968601e844cc475ec287181d51821c88dc5a1c6728a621943a8111`，
V-C `c3db6c38446028dafe6a95455a65d0a2c12e9ab488be72fcf57546012f10edc1`，
V-N `06919a60dea539dda46357db39867f73cf9d4b019ab5313b72076d786e974bc7`，
V-H `b7a24015cc939cd19054be0710b5eead2dfb9954a8d4a9e8669b8980bdeb1ec2`。六格共用 verifier program SHA-256
`d736a205b10f1a68375df4fc51af0df547d1c0cf8096e8fa1d1867dabf590ebc`；本地 fresh v8 receipt set 重验通过，set
SHA-256=`cc9ff5910992c46b9020654a78d8473ceb376bb5d9dc4adc984b90f454b3d9c8`。

probe9 六收据解锁后，于 `origin/main=d5c08bb91728edfa75801a630531527aeb2ae06c` 渲染的 v8
long-train 命令制品 `/tmp/phase1_balance_slew_train_commands_d5c08bb9.json` 文件 SHA-256 为
`be346f94cf6bf738da36804bf59f6a60bc5249f3c6bf5474abf617358db4b42a`。它只发射了 W-N
（Pod1 物理 GPU0）；日志停在 `Starting the simulation` / `sim.reset()`，从未出现
`Learning iteration`。locked launcher 的 `180 s` stale watchdog 只收敛 exact 绑定组并以 rc `125`
结束；外层最终变成 rc `121`，原因是旧 failure audit 误把 probe-only
`trainer_child_evidence.json` 也当成 train 必需，不是又一个 trainer 错误。人工双重 stable
快照已确认 leader/exact PGID absent、GPU compute=`0`、Kit/cache lock 无 holder；W-C/W-H/V-C/V-N/V-H
五格未发射。详细证据见
[`phase1_balance_action_slew_train_v8_attempt1_result_20260720.json`](../../configs/phase1_balance_action_slew_train_v8_attempt1_result_20260720.json)。
这是 pre-first-iteration infrastructure failure，没有 Reward 结果；v8 根、命令制品和 W-N 运行证据全部
冻结，禁止重试或继续发射其余五格。

fresh v9 已将失败审计改为 stage-aware：probe 的 supervisor 是 leader，因而必须同时稳定读取
leader 与 mandatory trainer-child evidence；train 的 locked trainer 本身就是 leader，因而 child/identity-failure
evidence 必须 absent。审计对 exact PGID（probe 再加 exact child PID）连续两扫，只在 stable empty
且 train 第二扫后 child evidence 仍 absent 时才报 closure。leader、child 与可选 identity-failure JSON
均要求 canonical newline、exact producer keys 和严格数值类型；probe 初扫与成功前复扫还必须绑定同一个
identity-failure snapshot，文件出现、替换或消失都会 fail closed。审计不发任何 signal；它只证明
closure，不代替 locked launcher 自身有界、exact-target 的 watchdog。任一 transaction 失败仍停止整批，
禁止 automatic retry 和 broad kill。

以下是 probe9 已使用的历史命令渲染流程。[`--authorize-launch`](../DEFINITIONS.md#balance-command-render-latch)
在这一步仍只渲染命令、不执行 SSH；probe9/v8 现已完成并冻结，不得重新执行下面的 probe 命令：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353"
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git cat-file -e origin/main:configs/phase1_balance_action_slew_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "phase1_balance_action_slew_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/branch not bound in one NOW entry")
PY

python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage probe --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  > /tmp/phase1_balance_slew_probe9_commands.json
```

实际 probe9 执行时，`origin/main` exact commit、统一队列认领、执行分支与 tracked manifest 四道门均通过，并由操作者按
[RunPod 启动纪律](run_on_runpod.md#2026-07-20-action-slew-wave-a-启动前状态与发射纪律)
执行。实际执行严格遵守 W-N（Pod1 GPU0）→W-C（Pod1 GPU1）→W-H→V-C→V-N→V-H 全局
串行；每格的 natural exit、verifier、receipt 和 closure 均在下一格之前完成，因而没有触发 v8 freeze。
只有 verifier 才能在
远端不可覆盖地发布 `probe_receipt.json`。把六份收据逐字节复制到任务专用本地目录
`BALANCE_PROBE_RECEIPTS_DIR=/tmp/phase1_balance_action_slew_probe_receipts_20260720_v8`，布局必须是
`DIR/{w_c,w_n,w_h,v_c,v_n,v_h}/probe_receipt.json`。
收据目录是 [trusted-operator capability](../DEFINITIONS.md#balance-receipt-trust-boundary)：重验会阻止损坏、
旧 identity 和误配，但不提供抵御恶意本地 root 的数字签名。必须保留 verifier 生成的远端
bytes 不变并逐字节复制；任何人工重写都使该格失效。

六份 [`probe receipt`](../DEFINITIONS.md#balance-probe-receipt-set)会重验 absolute milestone `[6701]`、
terminal checkpoint、policy/value/full optimizer/two normalizers、C/N/H exact 参数与 applied markers、
lineage=`0`、claim/binding，以及 6700/6701 两步的 processed-q_des、completion/fall/legal-return、ready-tilt、
qdot tag 和守恒账；processed-q_des/qdot observed 必须逐 update 精确等于 `4096×24=98304`，
processed-q_des recovery-eligible 可逐步为零但两步合计必须非零，其他预注册分母仍逐步非零。最后再
检查 fatal、进程组和 GPU 释放。没有
`--probe-approved` 人工捷径；train 只接受 probe9 manifest 下收齐的六份 fresh v8 exact 收据，本轮已全部通过本地重验。probe5
的五份 v4 receipt、probe6 的 v5 失败记录、probe7 的三份 v6 receipt 和 probe8 的 v7 失败证据都不能补齐 v8 或与
probe9 收据混用，并且脚本只生成命令：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="688599c2e01653bbb703553223a58e53656da1fe83d76aa7bcaa9f8a3ee75353"
BALANCE_PROBE_RECEIPTS_DIR="/tmp/phase1_balance_action_slew_probe_receipts_20260720_v8"
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git cat-file -e origin/main:configs/phase1_balance_action_slew_launch_manifest_20260720.json
python3 - <<'PY'
import subprocess

text = subprocess.check_output(
    ["git", "show", "origin/main:docs/NOW.md"], text=True
)
title = "- **[11｜P1] 稳定机制 Wave A/B。**"
start = text.find(title)
if start < 0:
    raise SystemExit("Wave A/B claim missing from origin/main NOW")
end = text.find("\n- **[", start + len(title))
entry = text[start:] if end < 0 else text[start:end]
required = (
    "责任人 franco；执行者 Codex；执行分支",
    "Franco_codex/balance-ablation-round-20260720",
    "phase1_balance_action_slew_20260720",
)
if any(value not in entry for value in required):
    raise SystemExit("Wave A/B owner/executor/branch not bound in one NOW entry")
PY

python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage train --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  --probe-receipts-dir "$BALANCE_PROBE_RECEIPTS_DIR" \
  > /tmp/phase1_balance_slew_train_commands.json
```

上述 probe9/v8 命令只作历史重现。最后用于 v8 long-train 的 exact 制品是前述
`be346f94…b42a`；它只发了 W-N 并已冻结失败，不得执行旧 JSON、重发 W-N 或继续其余五格。

fresh v9/probe10 必须从默认 **NO-LAUNCH** plan 开始。只有 config/runner/manifest 与 `NOW` 认领全部进入
当时最新 `origin/main`、且这四份 authority 在 worktree 中无 tracked 改动时，才允许渲染：

```bash
set -euo pipefail
BALANCE_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$BALANCE_REPO_ROOT"
python3 scripts/run_phase1_balance_action_slew_queue.py --stage probe \
  > /tmp/phase1_balance_slew_probe10_plan.json

git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
BALANCE_LAUNCH_MANIFEST="$BALANCE_REPO_ROOT/configs/phase1_balance_action_slew_launch_manifest_20260720.json"
BALANCE_LAUNCH_MANIFEST_SHA256="664375cb08263e6e7cdd82a3b8dd59e9e9ae6a9756333371677417ec4aa60c4a"
test "$(sha256sum "$BALANCE_LAUNCH_MANIFEST" | awk '{print $1}')" = "$BALANCE_LAUNCH_MANIFEST_SHA256"
python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage probe --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  > /tmp/phase1_balance_slew_probe10_commands.json
```

当前不得执行上面生成的 SSH；probe10 仍是 **source/preregistered/not launched**。后续获得运行
authority 后，操作者只能从 JSON 取单条 `ssh_argv`，不得把整份 JSON pipe 给 shell。probe 必须全局严格
W-N→W-C→W-H→V-C→V-N→V-H 串行；每格 natural exit、verifier、fresh v9 receipt 与
exact process/GPU/lock closure 都必须在下一格前完成。不得复用 probe9/v8 receipts。

只有六份 probe10 receipt 均按 exact bytes 收齐到
`/tmp/phase1_balance_action_slew_probe_receipts_20260720_v9/{job}/probe_receipt.json` 后，才可在同一最新
`origin/main` authority 下渲染 train：

```bash
BALANCE_PROBE_RECEIPTS_DIR="/tmp/phase1_balance_action_slew_probe_receipts_20260720_v9"
python3 scripts/run_phase1_balance_action_slew_queue.py \
  --stage train --authorize-launch \
  --launch-manifest "$BALANCE_LAUNCH_MANIFEST" \
  --expected-launch-manifest-sha256 "$BALANCE_LAUNCH_MANIFEST_SHA256" \
  --probe-receipts-dir "$BALANCE_PROBE_RECEIPTS_DIR" \
  > /tmp/phase1_balance_slew_train_v9_commands.json
```

probe/train 都保持 swap mapping：W-N/W-C/W-H 是 Pod1 GPU0/1/2，V-C/V-N/V-H 是 Pod2 GPU0/1/2。
long-train 每个 Pod 内 Kit boot 必须串行；前一格进入真实 `Learning iteration` 且 boot lock 释放后才能启动
同 Pod 下一格。无 automatic retry，任一启动失败立即停止新发射并闭合 exact 身份/GPU/lock；禁止
broad signal。完整量尺与 Wave B 限制见
[实验记录](../experiments/2026-07/EXP-P1-BALANCE-ACTION-SLEW-20260720.md)。

### 恢复/等待窗随机横向躯干推力（source 已接线，当前 `NO-LAUNCH`）

这和 qdot-limit 完全不同：qdot-limit 是关节速度惩罚；
[恢复窗随机横向躯干推力](../DEFINITIONS.md#lateral-balance-perturbation)只在 post-strike recovery 或
pre-swing hold 且非 strike window 时，给 `torso_link` COM 施加 WORLD-Y 外力。Hydra 入口只有：

```text
+task.lateral_perturbation.enabled=true
+task.lateral_perturbation.cell=L0|L1
+task.lateral_perturbation.seed=<exact uint32>
```

`L0` 是同随机机会的零推力对照；`L1` 是冻结的 `0.04–0.08 m/s` 归一化冲量、`0.10 s` pulse、每
`0.50 s` 一个机会、eligible 后选择概率 `0.5`。命令行不能改 body/frame/XZ force/torque/强度/时长；
disabled 时不能同时提供 cell/seed。启用后的 checkpoint hard contract 会绑定 cell/seed、resolved tick、
共同随机题、hard-safety、Isaac backend/显式 COM transform、全部 active EventManager term 的 exact typed
参数值与 manifest SHA，以及 metric schema。pinned `SceneEntityCfg` 的 selector/resolved ids、EventTermCfg 全
行为字段与 plain module function source identity 会绑定；未知 config、decorated/method func、非有限/callable/
opaque 参数或任一 interval term 在首次 submit 前拒绝，每步前后重验 attach 后漂移。日志输出 opportunity/
selected/commanded/backend-accepted/zero-overwrite 整数、
abandoned 与三本 impulse 账、实际整机质量 min/mean/max。`backend_accepted_*` 只证同步 setter/scene-write
提交边界，不是 solver-consumed 证据。

目前不要把上述三行加入任何 queue：exact Isaac full-scene、solver dynamics response、同 GPU throughput 与
no-host-sync 门尚未通过，机器预注册仍为 `launch_authorized=false`。现在只允许运行 dependency-light source
回归：

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_isaac_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py
```

full-scene 首次 canary 必须改走
[专用 probe 操作页](run_lateral_perturbation_runtime_probe.md)，不能用 trainer 偷跑。

These stds are DECOUPLED from acceptance thresholds: the position metric still reports true success only
below `strike_success_pos_thresh = 0.075 m`, velocity below `0.5 m/s`, and racket-normal error below
`15 deg`.

Optional later precision pass: once the exact-strike pass rates are non-trivial, tighten the racket
position/velocity stds and resume from the checkpoint.

## Domain Randomization (deploy-parity task)

2026-08-01 current authority: tonight's plant is frozen as the non-conflicting parkour table plus
three task/SKU fallbacks: waist-yaw Kp=`85`, waist-pitch effort=`118`, wrist-pitch/yaw
`Kp20 / effort6 / armature0.0008100893338`, while wrist-roll remains
`Kp30 / effort24 / armature0.004968`. Pod verification remains required. A future exact-SKU
confirmation of 24 N m creates a new plant identity; it does not hot-edit tonight's run. MuJoCo P0
and architecture P1/P2 stay outside tonight's prelaunch source.

The split gain-DR spelling remains a future contract: Kp `log_uniform(0.8,1.2)` and Kd
`log_uniform(0.7,1.3)`. The legacy `pd_gain_range` parser spelling remains compatibility-only and
cannot be combined with either split key. Current stable-ready vendor N1 disables both gain axes,
link mass and torso CoM; they return only behind a later healthy-baseline restore gate.

The inherited selector is currently `joint_names=[".*"]`, hence it perturbs all 31 training joints,
including two head joints absent from the vendor 29-DoF table. Nominal head constants remain the
repository values; describing the DR extension as a vendor 31-DoF recommendation would be false.
Link-mass/material/CoM recipes retain their own task-specific settings.

[`HOPEPingPongActionBallA3VendorV1`](../DEFINITIONS.md#a3-vendor-v1-profile) additionally owns two
settings that no caller may override: `[0,2]`
[control-step action delay](../DEFINITIONS.md#control-step-action-delay) sampled once per episode,
and ungated [`axis_box_6d_v2`](../DEFINITIONS.md#axis-box-6d-v2) push every `1–3 s`. Push is
velocity-only with `force_push=false` and `combined_exclusive=false`; the old `68 N × 0.3 s`
force push is not installed. Delay is applied
at the policy-action boundary before affine q_des conversion, not in a physics-substep actuator
buffer. Push may occur in the strike window in this version; no recovery gate is claimed.

### Vendor runtime guard receipts

Every fresh vendor diagnostic must preserve these stdout records from
[`vendor runtime JSON markers`](../DEFINITIONS.md#vendor-runtime-json-markers):

- `HOPE_RSL_RL_RUNTIME_ABI_JSON=` once before learning, after actor/critic empirical normalizer
  ABI, state shapes and finite moments have been validated;
- `HOPE_POLICY_STD_UPDATE_JSON=` once after every optimizer update, with realized std
  min/mean/max, learning rate and LR-floor flag; any non-finite or non-positive std is fatal;
- `HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=` after the first true reset and before the first
  rollout, binding the training-contract SHA, ordered active action terms, initialized environment
  count and lag histogram. Histogram counts must sum to `num_envs`.

Missing/invalid normalizer state, ABI drift, malformed policy std/LR or incomplete delay
denominator is a launch/run failure, not a warning. The global `ppo.yaml` fallback budget is finite
at `max_iterations=25000`; the vendor diagnostic's reviewed `long` stage explicitly pins `20001`
updates and does not inherit that fallback.

#### Current integrated-probe dependency graph

The only live stages are `smoke=1×2×save1`, integrated
`probe=4096×5×save1`, and `long=4096×20001×save100`. A/B/C respectively mean
`bh_loop_c` static, `bh_block` static, and `bh_loop_c` monotonic adaptive-sigma; all three are
fresh-only. `push_evidence` is retired and its old specs/receipts are spent history.

```text
plant freeze -> exact identity/authority/hold/bundle/pin chain -> Pod focused suites
  -> optional 1x2 fail-fast smoke
  -> one integrated 4096x5 probe
  -> one n1_vendor_probe_gate_receipt_v3 with stages={probe}
  -> tracked long scientific skeleton -> 4096x20001 long
```

The probe receipt hard-gates exact source/plant, 194-D actor and 318-D critic real-runner
normalizer save→second-runner load roundtrip, finite checkpoints, finite/positive std and finite LR,
control-step delay, zero joint actual-hard/qdes/nonfinite, natural completion, nonzero velocity-push
event/application counts, and finite/in-range extrema for x/y/z/roll/pitch/yaw. Table/fall frequency,
strike-window distribution, episode age and recovery are telemetry across the long run's first 100
updates, not five-update receipt blockers.

After one lane's integrated probe exits naturally, materialize its sole receipt without starting
Kit or training:

```bash
python hope_training/whole_body_tracking/scripts/materialize_n1_vendor_probe_gate_receipt.py \
  --gate-checkout "$SOURCE_ROOT" \
  --gate-source-commit "$SOURCE_COMMIT" \
  --evidence-source-commit "$SOURCE_COMMIT" \
  --probe-namespace <absolute-probe-namespace> \
  --probe-run-dir <absolute-probe-run-directory> \
  --receipt-repo-path configs/n1_vendor_probe_gate_20260731/<lane>.probe_gate.v3.json \
  --long-spec-repo-path configs/n1_vendor_launch_20260731/<lane>.long.scientific.json \
  --output <absolute-fresh-receipt.json>
```

The output is no-clobber and must have `kind=n1_vendor_probe_gate_receipt_v3`, `verdict=PASS`, and
exactly `stages.probe`. Long pins that single tracked receipt; there is no second live push receipt.
All focused and real-runner acceptance is Pod-only. Local work does not run tests.

#### Historical superseded Stage-A evidence

The following exact SHAs and failures are retained only as historical/spent evidence. Their old
`smoke + probe + push_evidence` schedule and plant values are not current launch instructions.

Stage A ran at exact source
`5665963e96bf75c677e7669efc58c449e0c04876`. Its recipe-only stage and `1 env×2`
[`A3 vendor identity smoke`](../DEFINITIONS.md#a3-vendor-identity-smoke) passed with schema-3
training-contract SHA `98fa3239daba825f07d3997fb28f4564c92967536f2552e6bdc0f8772781366f`.
`model_0.pt` and `model_1.pt` are finite; observed delay/ABI/std marker counts are exactly
`1/1/2`. The authority live-order defect exposed during Stage A is fixed. The emitted policy
recipe SHA `27bf405e5677fe2e7bab6fcc15c166901734048dd334b8b0abc3a8ffef3ce416`
is **shared-ready only** and must not be supplied where a dynamic-ready recipe is required.

The materialized `bh_loop_c` evidence is cross-bound as follows:

- dynamic-ready candidate SHA
  `c831a4e6d1c03519181efb090120a881702d113e95ebcf22f745a3a2ca4fc794`;
- nominal-hold receipt SHA
  `11c025dc25cba93c7d0d9894bac75da05a1a7aff11f797e9a35f9b2906f67740`, PASS for
  `0.8 s / 40` steps, feet-contact `1`, no terminal;
- bundle SHA `9881c52ca035bbdee0a3e1d0c0689eb7592b2a73b5442866a9a6e9480cbaae03` at
  `configs/n1_contact_vendor_a3_20260731/bh_loop_c.bundle.v2.9881c52ca035.json`;
- actual-authority receipt SHA
  `f66a9e59f441c22c465d3236d717c95354393d04c5975f58ece3e7612a65461a` at
  `configs/a3_vendor_runtime_authority_20260731/bh_loop_c.vendor_runtime_authority.v1.json`;
- materialized required-identity SHA
  `240f3757e45006de9dc5f4ecabcfc40071058009751fd1f0b8eb92656e1801ff`, binding the
  `98fa3239…` contract and only `bh_loop_c` as dynamic-ready action.

The launcher in this batch pins both required-identity and actual-authority SHAs, tracks their
materialized files, and passes `90` focused non-Torch tests. It has not run the distinct
dynamic-ready recipe-only path. The code-owned wrapper now passes `56` adjacent host tests. The
next gate is therefore to track and run that wrapper on a clean Pod commit, then run the
`bh_loop_c` vendor diagnostic `smoke` (`1 env×2`). Only a
passing smoke may unlock same-seed `probe` (`4096×5`). `bh_block` and current-revision `long` are
mechanically rejected; `long` also requires an actual probe-produced `vendor_probe_gate_receipt`.
Formal training, promotion, export, judge, deployment and hardware remain unauthorized.

On the selected clean Pod checkout, first render and inspect the fixed recipe-only spec; the
wrapper has no operator policy SHA, action, seed, environment-count, or PPO-budget axis:

```bash
python hope_training/whole_body_tracking/scripts/materialize_n1_vendor_dynamic_ready_recipe.py \
  template \
  --checkout /workspace/franco/nohope-a3-vendor-final \
  --commit-sha <exact-40-hex-commit> \
  --isaac-python /workspace/isaaclab/_isaac_sim/python.sh \
  --gpu-index 0 \
  --gpu-uuid GPU-889b1712-8d89-0536-5c9e-e79aae30523d \
  --owner Franco \
  --namespace /workspace/franco/runs/a3vendor-dynamic-recipe-<fresh-id> \
  > /workspace/franco/specs/a3vendor-dynamic-recipe-<fresh-id>.json

python hope_training/whole_body_tracking/scripts/materialize_n1_vendor_dynamic_ready_recipe.py \
  plan --spec /workspace/franco/specs/a3vendor-dynamic-recipe-<fresh-id>.json

python hope_training/whole_body_tracking/scripts/materialize_n1_vendor_dynamic_ready_recipe.py \
  launch \
  --spec /workspace/franco/specs/a3vendor-dynamic-recipe-<fresh-id>.json \
  --confirm-claim <exact-plan-claim-sha256>
```

The result must report `ppo_update_count=0`, no checkpoints, a new
`policy_training_contract_sha256` different from `27bf405e…e416`, and all launch/export/judge/
hardware authorization booleans false. A spent namespace is never reused.

Recipe r1 at source `2430fbb2` / claim `e37f8169…e32` is a permanently spent failure record. It
passed schema-v2 pre-scene validation, then the MotionCommand consumer rejected the v2 kind because
that consumer still encoded schema-v1 only. It emitted no recipe and ran no PPO; its exact PGID was
terminated and GPU0 returned to 18 MiB. Do not edit or reuse its spec/namespace. The next attempt
must use a later clean commit whose consumer preserves v1 and validates the complete v2 plant,
timing and delay payload, plus a fresh namespace and claim.

The clean `e7787e25` retry succeeded: recipe claim `75f28f24…490c` materialized policy
`e408b845…c65d` with zero PPO/checkpoints. Vendor diagnostic smoke claim `be783ab7…ad54` then
completed `1 env×2`; model 0/1 each contain 83 finite tensors, and ABI/delay/std-LR marker counts
are `1/1/2`. One waist-roll actual-hard termination occurred at episode age 25; this is recorded
for the same-seed `4096×5` probe rather than used to relax the hard edge or change Reward. The next
authorized stage is probe only; long/formal/export/deploy/hardware remain prohibited.

That same-seed probe later completed naturally with five finite checkpoints and valid ABI/delay/std
receipts, but it is a failed long gate: `14,086` actual-hard terminations were dominated by
waist-roll/waist-pitch after the vendor adapter accidentally omitted the adopted stable-ready
override. Its 100 strike-window-entry samples also measured `97% >0.20 m` (mean `0.4339 m`), so
the successor vendor leaf adds an independently paid `std=0.30 m` coarse position kernel while
retaining the `std=0.075 m` precision kernel. The corresponding expected effective-Reward SHA is
`8220f3397cb07a143149353d13f21914a90ac7be874169d519ebf5b2b9154dc3`.

The successor diagnostic always appends exactly one `stable_ready_plant=true`, advances the
plant-state safety trigger from 2% to 5% of hard travel without relaxing the raw hard-edge DoneTerm,
and provides a third exact stage, `push_evidence = 4096 env × 32 update × save8`. That stage
covers 15.36 seconds of policy time and pins the installed IsaacLab interval scheduler and
velocity-push source SHAs; under the pinned `[5,15) s` timer semantics, natural completion proves
every environment executed at least one push. Accept the rematerialized successor only after
`dynamic recipe → {smoke, probe, push_evidence}` all pass. The old probe cannot authorize long.

#### Historical superseded simulator fast path

This graph reproduces the retired standalone-push revision only. It is preserved to interpret its
spent namespaces and must not be used to render current commands:

```text
identity recipe -> identity smoke -> live training contract
                                      |-> runtime authority ------------------|
                                      `-> dynamic-ready candidate             |
                                           -> nominal hold -> bundle ----------|
runtime authority + required identity + bundle -> clean artifact/pin commit
                                                   -> dynamic recipe
                                                        |-> smoke -----------|
                                                        |-> probe -----------|-> all pass -> long gate receipt -> long
                                                        `-> push_evidence ---|
```

The following shortened schedule preserves the current claim and receipt semantics:

1. Keep `identity recipe -> identity smoke` serial: the smoke spec consumes the recipe's exact
   policy-contract SHA.  Once the live contract exists, materialize the runtime authority and the
   dynamic-ready candidate in parallel.  Start nominal hold as soon as the candidate exists; it
   need not wait for the authority materializer.  Bundle publication still waits for nominal hold.
2. Do not repeat the identity pair for a later artifact/document-only descendant when
   `materialize_a3_vendor_runtime_authority.py` validates that every bound scientific source blob
   is byte-identical to the authority source commit.  Any bound task, robot, training-contract,
   action, runner, environment or training-entrypoint drift reopens the identity pair.
3. After the clean artifact/pin commit and dynamic recipe have produced the exact policy SHA,
   `smoke`, `probe` and `push_evidence` may run concurrently on distinct empty GPUs (or distinct
   Pods).  They have no predecessor-receipt field in this launcher revision.  Give every job its
   own fresh namespace, owner-held GPU lock and exact claim; use the same checkout, bundle, runtime
   contract, policy SHA, action and seed.  On one Pod, serialize only Kit startup until its boot
   lock is released, then overlap rollout work.
4. Accept the fast path only after all three jobs finish naturally and independently satisfy their
   existing finite-checkpoint, runtime-ABI, positive-std, delay-histogram, actual-hard and push
   evidence checks.  A failure invalidates the affected evidence; it never relaxes a threshold.
   Long still waits for the named gate receipt and remains finite.  No-clobber publication, exact
   identity, owner lock and empty-GPU admission are unchanged.

In that historical revision, `smoke`, `probe` and `push_evidence` had no predecessor-receipt field.
The current revision instead uses the single integrated-probe graph above. Do not replace the
dynamic-ready nominal hold with a training diagnostic: it remains the exact-plant certificate for
the mathematical hold candidate.

The stage-evidence v4 consumer/fixtures pass `51 passed`; the combined vendor evaluation,
canonical-admission and formal-launcher suite passes `128 passed`. These receipts and host tests
establish identity mechanics, not learning quality or formal launch authority. The 2026-07-31
diligence/vendor setting is the current fresh-training authority; earlier audits bound to the old
repository constants are historical only.

### Vendor A3 evaluation profiles

Vendor-task evaluation must select and report one of the two
[`A3 vendor eval profiles`](../DEFINITIONS.md#a3-vendor-eval-profiles):

- `play.py` applies `vendor_play_v1`: disable startup plant DR and interval push, retain policy
  observation corruption and episode-sampled `[0,2]` control-step delay;
- `eval_deterministic.py` applies `deterministic_ranking_v1`: additionally zero observation
  corruption, delay and reset-state noise for reproducible checkpoint ranking.

Both apply after task composition and before `gym.make`, emit `VENDOR_A3_EVAL_PROFILE_JSON`, and
fail closed if the exact vendor task surface is incomplete. Never compare or average their scores
without naming the profile; deterministic ranking is not vendor Play robustness evidence. The
tensor/action semantics are frozen in the
[policy/action interface](../interfaces/policy_observation_action.md).

## Evaluate And Export

`play.py` exports the policy to `<checkpoint_dir>/exported/policy.onnx`.

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="../motions/preprocessed/hope_forehand.npz" \
  motion_file_2="../motions/preprocessed/hope_backhand.npz" \
  headless=false
```

For a formal 179-D face actor, keep the exact train bank configured in the export environment.
The native exporter refuses to create a 179 ONNX unless the live bank is exact schema 3, split
`train`, checkpoint/SHA-bound, `shared_plus_y`, and `mount_plusY_A`; it derives and hashes the
per-clip raw-A demanded-normal envelope from the bank bytes during metadata attachment. The
checkpoint contract, live export configuration and envelope payload must all agree on exact
`mount_normal_sign_per_clip=[+1,-1]`. A raw-A row is wire-representable only when
`sign[clip] * raw_A.x > 1e-6`: the external schema-2 physical-B normal remains positive-x, while
the backhand actor/bank raw-A normal is negative-x. Every row must also satisfy
`raw_A_row · reference_A > 1e-6`, identical to the deploy runtime gate; a merely positive
near-boundary dot fails export. An older 179
ONNX that lacks the envelope metadata is intentionally rejected by the current C++ loader.

The Isaac-free standalone exporter has the same rule and cannot copy the envelope from its donor.
Pass the exact train NPZ explicitly:

Use `--plan` for a genuine zero-write preflight. Plan mode loads the checkpoint with
`weights_only=True`, requires a non-negative integer `checkpoint_iteration`, checks actor and
normalizer finiteness, validates the donor ONNX, motions, harvest, train bank, training contract and
formal face-179 envelope, then exits before creating `--out`, a temporary file, an ONNX graph or an
artifact. `--help` and `--contract-import-smoke` remain lightweight commands and are not export
preflights.

```bash
python scripts/standalone_onnx_export.py \
  --ckpt /abs/run/model_<N>.pt \
  --fh /abs/forehand.npz --bh /abs/backhand.npz \
  --donor /abs/same-config-donor/policy.onnx \
  --harvest /abs/same-donor-harvest.npz \
  --train-bank /abs/s1_<family>_v3_train.npz \
  --out /abs/run/exported --run-path <label> --bake-obs-norm --plan
```

Plan success prints one JSON object. Require its `checkpoint_iteration` to equal the requested
checkpoint and require `artifact_written=false`, `graph_export_not_executed=true`, `input_dim=179`,
`output_dim=31`, `materials_validated=true`, `train_bank_validated=true`, and
`formal_face179_materials_validated=true`. The `would_write` path is descriptive only. Verify that a
missing output remains absent or an existing output (including `policy.onnx`) remains byte-identical.
Remove `--plan` only for the subsequent real export, and keep W/Y in separate new output directories.

Focused source regression (`97 passed in 0.38s` on 2026-07-19):

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_standalone_onnx_export_plan.py \
  hope_training/whole_body_tracking/tests/test_export_obs_norm_contract.py \
  hope_training/whole_body_tracking/tests/test_export_planner_task_revision_contract.py \
  hope_training/whole_body_tracking/tests/test_stage1_normal_envelope.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py
```

This path runs the same schema-3 bank loader and motion/anchor validation before deriving the
envelope, then verifies the bank SHA against the checkpoint-side training contract. Do not pass an
exam bank, omit `--train-bank`, or reuse a donor's `stage1_*` labels for a 179 artifact.
`--contract-import-smoke` is a dependency-light probe that exits before ONNX/Torch imports and
asserts that neither `whole_body_tracking/__init__.py` nor Isaac modules were loaded.

The standalone exporter validates the checkpoint contract/binding, donor, both motions, harvest,
train bank and derived envelope before producing a graph. It writes a same-directory owned temp,
checks the ONNX and metadata round trip, fsyncs, and atomically replaces `policy.onnx`. Any
validation, export, checker or save failure leaves an existing final model byte-identical and
removes the temp; do not replace this path with a direct write to the final filename.

Headless video:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="../motions/preprocessed/hope_forehand.npz" \
  motion_file_2="../motions/preprocessed/hope_backhand.npz" \
  headless=true video=true
```

From a WandB run:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  wandb_path="$WANDB_ENTITY/hope_wbc/<RUN_ID>" headless=false
```

### Shared schema-v3 BankExam (Isaac + MuJoCo)

Do not pass an exam bank through a training Hydra override.  The saved
`RacketTargetCommand` continues to own its train-split bank; the evaluator loads
the exam split independently and installs only the current immutable questions.

Materialize one balanced paper first.  Both simulator cells must consume this
exact JSON (same schedule SHA, question order, hold values and attempt seeds):

```bash
python scripts/materialize_bank_exam_schedule.py \
  --exam-bank /abs/path/s1_<family>_v3_exam.npz \
  --per-clip-quota 10 --schedule-seed 0 --hold-range 0 100 \
  --output /abs/path/canary.schedule.json
```

Isaac single-ball cell (`K=20` creates one environment per immutable item):

```bash
hope_isaac_py scripts/isaac_bank_exam.py \
  task=HOPEPingPongVirtualBall headless=true device=cuda:0 \
  +run_dir=/abs/path/to/training_run \
  checkpoint=/abs/path/to/training_run/model_16999.pt \
  +exam_bank=/abs/path/s1_<family>_v3_exam.npz \
  +schedule_json=/abs/path/canary.schedule.json \
  +per_clip_quota=10 +schedule_seed=0 +noise_scale=0.0 \
  +output_dir=/abs/path/isaac_canary
```

Historical M3f/M2/G1 checkpoints are ruler canaries, not formal lineage; add
`+allow_inexact_contract=true` to Isaac and `--allow-inexact-contract` to
MuJoCo, and keep the resulting `evaluation_contract_exact=false` label.
Re-exporting or resuming an old checkpoint cannot turn it exact.

For a raw normalized ONNX, the sidecar must reproduce the saved runner formula
`(obs - mean) / (std + eps)`. Zero std entries are valid constant-feature
statistics only when `eps` makes every divisor strictly positive. The loader
therefore requires finite `std>=0`, finite `eps>=0`, and elementwise
`std+eps>0`; never delete the sidecar or feed raw observations to get around a
normalization error.

The BankExam entry point resolves the current checkout's dependency-light
`stage1_question_bank.py` automatically, exports that exact path to the
sampler process and binds the loader SHA into the execution contract. Do not
install Isaac packages into the MuJoCo environment or rely on a stale
`HOPE_STAGE1_QB` value.

MuJoCo consumes the same paper and uses the same authoritative NumPy scorer:

```bash
python scripts/mujoco_eval_onnx.py \
  --onnx /abs/path/to/exported/policy.onnx \
  --motion-files /abs/path/forehand.npz /abs/path/backhand.npz \
  --target-source bank --exam-bank /abs/path/s1_<family>_v3_exam.npz \
  --exam-schedule-json /abs/path/canary.schedule.json \
  --noise-scales 0.0 --seed 0 --qdes-clamp --hold-ref stand \
  --allow-inexact-contract \
  --out-dir /abs/path/mujoco_canary
```

Omit both inexact flags for a fresh schema-v3 checkpoint/ONNX. The evaluator
hashes the exam bank before and after loading; any mid-load replacement is
fatal. The shared schedule installer is formal/fail-closed by default and
accepts an inexact sampler only through the explicit historical diagnostic
flag above.

For a valid cell, the raw ledger must contain all `K` rows in schedule order.
Physical falls, guard resets and episode timeouts remain failed attempts in the
same denominator; an external step-cap truncation invalidates the whole cell.
The versioned hold contract is `H` ready-stand policy actions followed by raw
clip frame 0; it is part of the schedule SHA, not an evaluator-local guess.
Before comparing rates, assert that Isaac and MuJoCo report the same bank SHA,
schedule SHA and ordered question IDs.  Only the `noise_scale=0` canary
survivors advance to 50 questions per side, continuous play and 5% action
noise.

For the separate carry-state ruler, add `--exam-continuity-diagnostic` to the
MuJoCo command and keep `--allow-inexact-contract`. It consumes the same finite
paper but does not reset robot/action state between questions, always stamps
the result inexact, and reports `continuity.return_and_recover_rate`. The
denominator excludes only the terminal paper row, which has no scheduled next
opportunity. Do not call the one-environment-per-question Isaac adapter a
continuous test; its physical next-ball/serve timeline is a separate pending
implementation.

### T1 post-strike event mode is source-ready, not launch-ready

Commit `be5d7cf` adds the training-side scheduler and hard-contract fields;
it does not authorize a T1 run. Do not invent a schedule JSON or point a live
trainer at `event_timing_mode=post_strike_t1`. The frozen preregistration must
continue to fail launch validation until a reviewed materializer, immutable
screen/decision schedules, continuous judges, self-hit gate, fresh baseline
and semantics-correct plant are rebound in a new launch preregistration.

Dependency-light verification is:

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_event_timing_scheduler.py

sha=$(shasum -a 256 configs/phase1_event_timing_t0_t1_prereg_20260711.json | awk '{print $1}')
python3 scripts/validate_phase1_event_timing_prereg.py \
  --prereg configs/phase1_event_timing_t0_t1_prereg_20260711.json \
  --expected-prereg-sha256 "$sha" --mode design-check
```

Running the same validator with `--mode launch-check` must return 1 for the
frozen preregistration. The runtime field/schema contract is documented in
`docs/interfaces/t1_event_training_contract.md`.

### Signed-face single-seed rescue funnel

Do not reuse the old 24-arm launcher or replicate four seeds after the signed-face failure. The
machine-preregistered A/B/C/D hot-start/fresh × face-guidance-off/on funnel has its own clean source
commit, no-clobber L1 completion record, exact parent/asset bindings and SSH-interruption recovery
rules. Run it only through
[`run_phase1_signed_face_rescue_funnel.md`](run_phase1_signed_face_rescue_funnel.md).

L1 is a `512 env × 25 update` launch-integrity smoke on one seed. Hot cells must save lineage `0`,
fresh cells lineage `1`, and all four must emit one common hard-contract SHA. L2 is designed as
`4096 env × 1001 update`, but v6 rejects every L2 validate/plan/launch before runtime writes until a
separate immutable signed-face directional checkpoint paper path/SHA and reviewed v7 activation
exist. This launcher starts no judge, promotes no checkpoint and buys no additional seed.

The first production preflight (`control/v1`) was rejected before any run claim because its audit
looked only for top-level tensors and top-level contract keys. RSL-RL stores weights recursively and
the contract tuple under checkpoint `infos`. The v2 launch then preserved a pre-learning A-cell
failure because it did not pass the detached worktree's source-first Python environment to the
child. v3 then overreached by importing IsaacLab before `SimulationApp`; v4 proved the ignored A3
asset does not follow a detached worktree. v5 then proved the old train bank was bound to a different
physics/source-family contract and failed before learning. Preserve v1-v5; run only `control/v6`, whose
recursive audit, deterministic environment, module-origin, restored-asset tree and no-clobber rebound
train-bank report closure are recorded in the dedicated operation page.

### Wave-B 下肢稳定六格队列

W/V 两个 `model_6700` parent 内比较 upper-only control、静态 `v4rg` 十二腿软模仿和无参考
stance/qdot bundle 时，只使用
[`run_phase1_lower_body_stability_wave.md`](run_phase1_lower_body_stability_wave.md)。该队列默认不生成 SSH；
六条 long 只有在六份 `4096 env × 24 steps × 2 update` 自然退出 probe receipt 全部验证后才可渲染。M0 横移老师
不在本轮，且没有真机授权。

## First-Loop Rule

Before setting a baseline quality target, record:

1. Isaac asset path.
2. Environment start command, including `source setup_train_env.sh`.
3. Random rollout result.
4. First training command.
5. Checkpoint path, ONNX export path, and WandB run ID when WandB logging is used.
6. Failure mode or first metric.
7. Whether the run is only pipeline viability or an accepted quality baseline.

把完整的设计/run/结果/决定写入
[`../experiments/`](../experiments/README.md) 下对应的实验记录，并更新 G05；随后只在
[`../PROGRESS.md`](../PROGRESS.md) 追加一条带链接的简短记录。如果已采用 setting 或逐动作
成绩表发生变化，还要更新 [`../NOW.md`](../NOW.md)。
