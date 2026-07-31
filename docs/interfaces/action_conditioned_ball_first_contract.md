# 按动作条件化的 Ball-first 训练合同

Status: Draft / source candidate。本文定义当前候选训练语义；它不授权训练、部署或真机。
正式运行仍须通过动作 admission、桌面安全、Pod smoke 和 exact-resume 门。

共享术语见[术语与人话对照](../DEFINITIONS.md)。这条链的核心不是让训练期 selector 猜动作，
而是先由均衡训练日程冻结动作，再从该动作自己的来球域采球并解 task：

```text
action
  -> action-specific arrival-time / ball / base / landing-aim sample
  -> fixed-action ball-to-task solve
  -> certified teacher-rate + ready-wait solve
  -> atomic ball + task install
  -> policy attempt
  -> per-action capability evidence
```

训练回答的是
`P(合法且安全回球 | action, incoming ball, outgoing aim, base context)`；训练后 planner 才读取
独立留出能力工件做 selector。训练时的 action schedule 不是部署 selector。

## 1. 为什么采用这条顺序

`task-first` 会先声明拍速/拍面，再寻找或忽略来球，容易生成“照命令打就过不了网”的冲突题。
无条件 `ball-first` 虽保证球与 task 自洽，却允许 solver 或 selector 总挑容易动作，无法知道每个
动作到底学会了什么。

当前合同同时保留两点：

1. **球先于 task。** 每个 sample 都由物理解算产生 task，解不出的球不安装；
2. **动作先于球且在 attempt 内冻结。** solver 必须使用被选动作的速度方向、速度域和有符号拍面，
   不能在解题时改选容易动作。

因此训练期无需先设计在线 selector，又能得到 selector 以后真正需要的逐动作条件能力。

## 2. 身份与信任边界

启动 manifest 只描述分布和 exact identity，不得自报训练授权。它必须以 exact 文件 SHA-256 绑定：

- 非空、有序、无重复的 action catalog；
- 每动作的 stable action UID、motion path/SHA、strike phase、family、mount face sign；
- prototype、solver、physics profile 的 exact SHA；
- 每动作来球、base、落点域；
- mobility profile、课程统计参数和冻结 holdout split。

holdout 的 `samples_per_action` 至少覆盖 manifest 自己声明的 `max(min_proposals,
min_safe_closed)`；若另启 formal 768 卷，则必须使用独立、同样内容绑定且不少于 768 的 split，
不能拿 512 行 canary split 冒充 formal window。

manifest loader 可校验 repo 内 regular-file bytes、拒绝 symlink escape，并生成不可变 asset receipt；
这仍不等于 motion admission。训练 launch boundary 必须另持 code-rooted opaque motion capability，
逐项交叉绑定 ordered motion bytes、upper/full scope、shared-ready、evidence chain 和 promotion
certificate。metadata、文件存在、diagnostic PASS 或手写布尔值都不能 mint capability。

stable action UID、motion loader slot、manifest order 和 checkpoint control-plane contract
必须完全一致。dense slot 只用于当前数组索引；UID/slot 都不得作为 fresh actor 的 categorical
observation。fresh N1 actor 使用固定 194-D
`action_ball_table_pose_twist_heading_task_teacher_start_v2`，把旧 N1 的常量 one-hot 槽替换为
`time_to_teacher_start_s`。旧 N-dependent one-hot 合同只允许历史 checkpoint/receipt 解析。

N5/N73 不能简单删掉 one-hot 后共用 N1 v2：shared-ready 会使不同 teacher intent 在同一 actor
输入下发生 observation aliasing。正式多动作训练必须先冻结一份固定宽、由动作内容导出的
continuous future-motion intent/preview 合同；在此之前 fail closed。

候选 upper/no-move N5 control-plane view 的 exact ordered action ID 是：

```text
bh_loop_c
v12_forehand_block
bh_block
s0_highpress
fh_loop_high
```

旧 `fh_loop` 与 `fh_block_syn` 只保留为编译 provenance，禁止进入这份 N5 view、motion loader、
profile、考卷或 checkpoint。该 ordered view 仍只属于控制面，不授权 actor launch；N5 的通过条件
只由这五件、本次 N5 运行规模和已冻结的 continuous future-motion intent 决定。尚不存在的 N73
合同、资产或压力测试不得反向阻塞 N5 自己的准备工作。

## 3. 每动作采样域

每个动作有一个版本化、训练中不漂移的 center profile。换 center 等同换 profile SHA 和新实验，
不能在同一 domain epoch 内边训练边重估。

### 3.1 来球与目标

- time to contact：从 true reset/本拍 ready 到触球的可用时间；中心两侧分别有
  lower/upper initial/max std 与硬 bounds，不能用一个对称 std 代替。最早到球必须晚于该动作在
  最慢认证 teacher rate 下的 `t_hit + reaction margin`，额外 ready 等待不得超过 `1 s`；
- contact offset：相对 `base_goal` 的 `B_yaw` 三维位置；x/y/z 每一维的 lower/upper
  initial/max std 独立。x 是触球前后/时序敏感轴，最大宽度应小于 y/z；
- incoming speed magnitude：中心两侧独立扩张，硬下限固定为该 profile 中心速度的 `0.4×`；
- incoming direction：以动作中心来球方向为法向，在显式、正交、右手的 tangent `u/v` 上各自
  扩正负两侧；整个最大 support 都必须仍从对面来，不能只验证中心；
- spin magnitude：中心两侧独立扩张；
- spin direction：以中心旋转方向的 tangent `u/v` 各自扩正负两侧；无旋中心可在 profile 中
  使用确定的 canonical direction，不能让零向量进入单位方向合同；
- outgoing landing aim：`W` 平面 XY 的 lower/upper 两侧独立扩张。aim 是能力查询维度，不能只
  训练一个固定点却宣称 planner 可处理任意击球目标。

所有连续标量和 tangent 角都只作**一次** open-interval uniform draw，直接覆盖当前完整不对称
support；不先 `50/50` 选侧，也不采 half-normal：

```text
x ~ Uniform(center - width_lower, center + width_upper)
P(x < center) = width_lower / (width_lower + width_upper)
```

因此两侧课程 level 仍独立进退，但某侧变宽会按其物理区间长度取得更多样本，这是“在当前题域均匀
覆盖”的设计语义。实现中遗留的 `*_std` 字段名表示 **support half-width**，不是高斯标准差。
time-to-contact 同样在裁剪后的整数 policy-tick support 上一次均匀抽取；不能用 clamp 制造边界
原子。`level=0` 使用该侧 `initial width`，不是零扰动：

```text
width_side(level) =
    width_side_initial + level * (width_side_max - width_side_initial)
```

schema v3 的 authoritative arm catalog 一共 32 臂：

```text
time_to_contact: 2
contact x/y/z: 6
incoming speed: 2
incoming direction tangent u/v: 4
spin magnitude: 2
spin direction tangent u/v: 4
landing aim x/y: 4
base spawn x/y: 4
base travel x/y: 4
```

`no_move` 只禁用四个 base-travel 臂，所以有效臂为 28；`move` 使用全部 32。catalog 的 exact
顺序与 `ARM_CATALOG_SHA256` 是 sampler、curriculum、evaluator、runtime receipt 和 checkpoint 的
共同身份，不允许各模块自己重排或换一套名字。

### 3.2 Base

- base spawn：true episode reset 时在 `W` 平面 XY 采一次；root Z、quaternion、31 个 joints 和全部
  速度仍逐位取 canonical ready；
- base travel：`B_yaw` 平面 XY 的潜在位移。

`no_move` 与 `move` 是两份不同 manifest/profile SHA，不允许运行时覆盖。两臂可复用逐字相同的
latent travel 分布以保证消融可比：

```text
no_move: base_goal == base_spawn
move:    base_goal == base_spawn + R_yaw * sampled_travel
```

首个真实训练只允许 `no_move`。同一 episode 内 action 和 birth base 冻结；WRAP 只换同动作的
ball/task，不能重写 root。birth receipt 只带 episode 的 `base_spawn`/yaw；`base_goal` 属于每个
swing 的 task receipt，因此 `move` 可以在同一 episode 的后续 swing 学不同目标站位，`no_move`
则逐次强制 `base_goal == base_spawn`。`move` 只有在 base 到达时序、桌面/地面碰撞和 recovery
generation ledger 通过后才开放。

sampler identity 必须包含 seed。它对每个签发过的 birth 保存 exact transcript，后续 `sample()`
不仅重算自哈希，还必须命中本 sampler 真正签发的 `<action_uid, birth_index, birth_id>`；任意调用方
用公开公式重算一个不同 `base_spawn` 的 birth 不得通过。Motion 不读取 Racket 私有 curriculum，
也不自报 epoch/levels；birth broker 绑定 code-rooted frozen domain-claim authority，在 true-reset
batch 内按 action UID 取得 claim。课程只在 reset barrier 原子发布下一版 snapshot。

多环境 reset 必须使用一次 `reserve_many → 写全部 root → commit_many`。provider 是有
`sampler_contract_sha256 + state_dict/load_state_dict` 的 stateful object；任一行失败时 broker 与
provider RNG/tape 一起回滚。scalar 循环留下前缀 pending/committed 不算原子 reset。

## 4. Fixed-action 解题与原子安装

solver 输入必须是 exact proposal：

```text
action UID/slot
time to contact
contact position
incoming velocity and spin
landing aim
action prototype
birth base quaternion
physics/scorer pins
```

solver 不得内部重采样、改 action、翻转 physical face 或覆盖 proposal。最终 scorer 必须用同一
contact/flight 参数重放，明确校验 approach、动作速度 envelope、有符号拍面、过网、网高余量、
首落点和 flight horizon。失败行保持不可安装，并进入具名 reject ledger。

task normal 是 runtime raw +Y/A-frame 命令；接触重放必须再用 manifest 的 `mount_normal_sign`
得到 physical striking face。physical face 的 normal closing speed 必须在 venue fit
`[1.4, 7.2] m/s`，fixed-action 拍速上界取
`min(action prototype max, global speed budget)`；不能依赖 contact oracle 自动翻面或忽略全局
speed cap 后仍宣称 fixed-action 可行。

task 解出所需拍速后，teacher 时间律必须由内容绑定的动作元数据确定：

```text
teacher_rate = required_racket_speed / reference_racket_site_speed
scaled_t_hit = reference_t_hit / teacher_rate
scaled_t_cycle = reference_t_cycle / teacher_rate
pre_swing_wait = time_to_contact - scaled_t_hit
```

`reference_racket_site_speed` 必须是正式 physical racket site 的值，不得用 wrist COM 速度代替。
`teacher_rate` 必须落在该动作已认证的 `[rate_min, rate_max]`；实现不得 clip，越界就是具名 solver
reject。Motion 先保持 canonical ready `pre_swing_wait`，再按 rate 播放，使
`pre_swing_wait + scaled_t_hit == time_to_contact`。这些字段属于 per-swing task receipt，不属于
episode birth；WRAP 换球时可同时换到球时间与 rate。Motion 只能通过 Racket 暴露的只读 opaque
task receipt reference 再 resolve 取得它们，不能读取 curriculum 私有状态。发射/安装还必须保证
`pre_swing_wait + scaled_t_cycle + policy_dt <= episode horizon`；`t_cycle` 已含 shared-ready
recovery，额外一个 exact policy control tick 用于闭合 attempt。否则长到球时间会把 recovery
截在 episode 外。`policy_dt=sim.dt*decimation` 与 episode length 都必须进入 hard contract，不能
只改 `run_name`。

旧 CQ producer/distribution/buffer/seed 仍必须关闭；`cq_*` 前缀中只有 fixed-action solver 自身的
五个数值 knob 可以显式配置：`cq_overdraw`、`cq_n_iters`、`cq_tol_m`、`cq_speed_budget`、
`cq_max_redraw_rounds`。这里 overdraw/redraw 表示从 action sampler 取得更多**独立 proposal**，
每一行都保留在 `P` 与 reject-reason ledger；它不是在 continuous solver 内替换同一行。五值必须
进入由 runtime canonical payload 重算的 solver SHA，不能因沿用旧字段名就把旧 CQ producer 放回。

ball、base plan、task、action、domain epoch、sample ID 和所有 digest 只有在全部校验通过后才一次性
写入 command/ball buffers。solver 完成前不得先写一半 target，也不得用 reset 前缓存的
`base_pos_w/base_quat_w` 补 receipt。

## 5. Attempt 账本

每个 action/curriculum cell 强制记录：

```text
P proposed
A solver_admitted
I installed
S started
C closed
L legal_return
F safe_nonreturn
U_table / U_fall / U_collision
U_joint_qdes / U_joint_actual
X infrastructure_invalid
```

并满足：

```text
P >= A >= I >= S >= C
U_unique = C - L - F
max(U_table, U_fall, U_collision, U_joint_qdes, U_joint_actual)
    <= U_unique
    <= U_table + U_fall + U_collision + U_joint_qdes + U_joint_actual
N_safe = L + F
safe_policy_failure = F / N_safe
solver_admit = A / P
unsafe = U_unique / C
```

`U_*` 是逐 attempt sticky 的原始传感器维度，可以重叠：同一个 closure 可以同时撞桌并触发实际
关节硬限位，但在 `C` 中仍只闭合一次。单一 primary terminal 只用于把 closure 唯一分进
`L / F / U_unique`，绝不能用 precedence label 抹掉次级原始安全信号，也不能把 `sum(U_*)`
当作 unique unsafe。V4 scheduler attempt 必须保留 exact `terminal_signals`；curriculum state
schema 10 将这些 scheduler rows 连同 evidence 写入 checkpoint，load 时重放并重新核对上述闭包。
旧 schema 或丢失 raw signals 的正式 V4 state fail closed。

`A-I`、`I-S`、`S-C` 必须可见，不能只让容易闭环的幸存 attempt 进入分母。solver rejection 不算
policy failure；`X>0` 使该证据窗口无效。所有动作的 table hit 都是零容忍；新正手另有站位与整轨
clearance 门，不能塞进允许百分比的 aggregate unsafe 桶。

用户所说的“约 10% 回不了”严格解释为 **safe closed attempt 的 policy non-return**，不是 10%
物理无解题，也不是 10% 撞桌/摔倒。`20%` 只作为单独消融。

## 6. 异步课程

每个 `(action_uid, mobility_profile)` 独立运行：

```text
CENTER_TRAIN
  -> MARGINAL_PROBE(each arm / side)
  -> JOINT_TUNE(rho)
  -> STEADY
  -> rollback to last_certified on regression
  -> QUARANTINED when center is unsafe/unsolved
```

### 6.1 Marginal 与 rolling-100 调度

32 个 arm（`no_move` 为 28 个）分别寻找边界；探一个 arm 时其他 arm 保持 level 0。lower/upper
必须分开定界，因为“减慢一点容易、加快一点困难”正是 selector 以后需要保留的信息。

每动作最近 100 个**同 arm、同 candidate level、正常闭合且安全**的 attempt 只用于决定下一步先
考哪个 arm。优先考对 safe-policy failure 影响最小的方向，但调度必须带确定性的 forced
exploration：样本不足先补最少样本臂，固定轮次强制最久未选臂，并设置最大 starvation age。
solver reject、安装失败、unsafe 或 infrastructure invalid 不得被包装成“这个方向学得差/好”。

rolling-100 只能排候选，不能批准扩张；否则同一训练流既选题又判题，会反复窥视到偶然好窗口。
每个 marginal arm 的约 10% 只表示单变量截面，不能直接相乘成联合结论。

### 6.2 Joint

得到所有 marginal frontier 后，用共同缩放 `rho` 调联合域：

```text
joint_width_k = rho * marginal_width_k
```

在联合分布上把总 safe-policy failure 控制到目标带。稳定训练建议按预注册比例混合：

- 20% level-0 anchor；
- 60% joint interior；
- 20% frontier/marginal probes。

课程回退不能删除历史能力证据；planner 只读独立 frozen heldout 的最新能力工件。

### 6.3 统计窗口

rolling-100 只作 live scheduler。冻结 policy 的 256 个独立 attempt 只够 canary；正式
`10% ±2.5pp` 窗口默认至少 768，且使用与 canary 互斥的 heldout seed namespace。扩张用 failure
UCB `< 7.5%`，回退用 LCB `> 12.5%`，重叠则补样；solver admission、安装/启动/闭合率、unsafe 和
完整性另设硬门。

晋级证据必须来自冻结 policy snapshot 和互斥 eval seed block。每个窗口绑定：

```text
action/profile/domain/stratum/exact 32-arm levels or rho
arm catalog SHA / scheduler contract SHA / selected arm / selection round
policy contract SHA
frozen policy checkpoint SHA + monotonic global generation
sampler/solver SHA
seed block
action sample-index 半开区间
exact ordered sample receipt root
monotonic window sequence
window SHA / hash chain
```

重复、重叠、旧 epoch、generation 回退、同 generation 多 checkpoint 或 digest 漂移的窗口拒绝
消费。整数区间只负责查重，不能替代 sampler 的 SHA sample identity；窗口必须同时绑定 exact
ordered receipt root。generation 是全局 frozen-eval 版本：一旦开始消费新 generation，旧
generation 不再接受新窗口；大 N 可以在同一 generation 内异步分批提交。训练 rollout 只能触发
code-rooted frozen evaluator authority，不能直接把 `BallDomainEvidence.create(...)` 的调用方自报
对象交给 controller，也不能自报 checkpoint SHA 或反复窥视同一批 Bernoulli 样本后冒充固定覆盖率。
缺 evaluator authority 时课程必须 hold，不能“先信一次”。

## 7. Exact resume

checkpoint 必须保存并严格恢复：

- manifest 与全部 contract digests；
- per-action independent RNG counters；
- birth broker generation、stateful provider/domain-authority state SHA 与未消费 single-use receipt；
- lazy per-action solved pools 的完整未消费 rows、cursor 和 reject ledger；
- exact arm catalog 与 scheduler contract；
- curriculum marginal/joint/frontier/last-certified 状态，以及可重放的 canary/scheduler event；
- 已消费 evidence window、sample ranges 和 hash chain；
- active attempt 的闭合/作废记账；
- Python/NumPy/Torch/CUDA RNG 与 runner optimizer state。

2026-07-31 的两个新状态不能被上述“完整 ActionBall state”一句含糊带过：

1. 启用 [control-step action delay](../DEFINITIONS.md#control-step-action-delay) 时，runner
   `environment_resume_state` 从旧 schema 3 升到 schema 4，新增 ordered
   `active_action_term_names` 与逐 term `action_terms`。delay term 必须保存 config/term
   identity、每 env lag、initialized bit 和全部 history queue；恢复分两阶段，先无突变地
   检查全部 schema/形状/范围/顺序，再按 ActionManager 顺序原子 load。delay enabled
   拒绝 environment schema 1/2/3，必须 fresh 或使用 schema 4 checkpoint；delay off
   继续生成原 schema-3 四键状态，不添加 queue 或 clone。
2. [入击球窗拍距](../DEFINITIONS.md#strike-window-entry-distance)增加了每 env
   `strike_window_entry_armed` latch，所以 Racket command exact-state 为 schema 6。latch
   防止 checkpoint 恢复后同一拍重记入窗；只有 true reset 或 natural wrap 会为选中 env
   重新 arm。这不把 TaskReceipt/solver state 的其他 schema 统称为 6。

action-ball 的首次 `runner.learn` 必须使用 `init_at_random_ep_len=false`，保证首批统计从完整
true-reset/native cycle 开始；相位去同步由独立 hold-stagger 负责。该布尔值进入 policy/preflight
recipe SHA。随机截短首 episode 会污染 `C/F/close-rate`，不能沿用其他模式的历史默认。

load 只恢复数据，不采样、不写 simulator。curriculum 的 phase/frontier/certificate 都是派生值，
恢复时必须从 genesis evidence receipts 重放 deterministic reducer 后逐字段一致，不能只校验外层
JSON digest。恢复后的中途 active attempt 必须显式记为 infrastructure invalid 或按预注册规则
作废，再由首次 true reset 创建新 generation；不能悄悄丢掉。

runtime 不得为了防伪永久复制全部 retired full task receipt；N73/4096 环境会让 checkpoint 线性膨胀
到不可训练。实测 sampler 在 `4096 birth + 4096 sample` 时已有 `6,070,936 B` JSON，按 100 轮
线性外推约 `607 MB`，还没有计算 broker/pool/provider/Racket 的重复历史。issued task 必须可由
sampler + fixed solver 确定性批量重放，生命周期用完整连续索引覆盖和 compact 状态 ledger
记录，再与累计分账和 root 对账。

长期运行必须在 rollout/reset barrier 做跨组件原子
[`action-ball compaction segment`](../DEFINITIONS.md#action-ball-compaction-segment)：只给 active、
pending 与尚未消费后缀保留完整 receipt；已退休连续前缀折成逐动作 high-water、守恒计数与同一个
append-only segment head。sampler、broker、pool、provider 和 Racket 任一方没有提交相同 head，
整次 compaction 回滚。不能只做 JSON 压缩/RLE 后继续永久保存全部行，也不能让某组件先删历史。
正式开跑前须按**本次 exact manifest 的 N、环境数 E 和预注册轮数 R**完成真实分批增长、压缩、
保存/加载、roundtrip、篡改负例、峰值内存和时延硬门。fresh N5 只消费 N5 对应的
`N=5 × E × R` 压力凭据；N73 的 `N=73 × E × R` 压力门只约束真正的 N73 发射，不得拿 N73
缺资产或未压测阻断 N5。checkpoint 及 formal resume receipt 都绑定最终 segment head。

内部 hash/replay 只能证明各组件对同一状态的 cross-view consistency，并防部分丢字段、rollback、
错 birth 和普通损坏；在没有外部信任根时，它不可能抵抗攻击者协调改写全部组件并重算全部 hash。
因此 formal resume 还必须消费一份**事先独立钉住**的
[`action-ball formal resume receipt`](../DEFINITIONS.md#action-ball-resume-receipt)，至少绑定 raw
checkpoint SHA、shared action-ball state root、compaction segment head、run/training contract、
iteration 和 receipt authority。runner
先验证该 expected receipt，再重读 checkpoint bytes；不得在同一次 resume 中现算现填。缺该外部
pin 时只可 fresh launch 或 diagnostic load，不能称 formal exact resume。

### 7.1 非正式诊断跑的 safety aggregate

`action_ball_diagnostic_unauthorized=true` 明确没有 formal promotion、exact-resume 或部署
authority，因此不得在每个 reset / policy step 复制正式 forensic transcript 来拖慢训练。它仍
必须逐 physics substep 执行同一 qdes clamp、receding brake、q/qdot timestamp freshness、
actual-hard/nonfinite Done，并在 device 上累计每环境/每关节的 apply/post/timestamp count、
qdes/crossing/actual-hard count 和 minimum hard gap。

每个 PPO update 的唯一消费顺序是：

1. action term 冻结 update aggregate，后续 physics/reset mutation fail-closed；
2. runner 一次 D2H 验证 rollout step 数、apply/post 数、timestamp、finite gap、latch/count
   守恒和连续 policy sequence；
3. optimizer 成功后才 acknowledge/clear；optimizer 异常必须让原 aggregate 保持冻结；
4. 只打印一条 `formal_authority=false` 的 compact JSON，不生成 formal `.prepared.pt`、
   optimizer-commit 收据、逐 step action identity 或 per-reset terminal archive。

fixed action/ball/task identity、proposal/admission/reject、RNG 与 PPO sample 仍由 command/
broker/training contract 约束；compact safety 不能改这些值。formal ActionBall 完整证据路径保持
不变，未来把 formal 收据降到 checkpoint 粒度必须单独 bump schema，并证明旧式 receipt 可重建。

当前 vendor delay runtime JSON producer 与 stage-evidence v4 consumer 已接通，focused
`51 passed`；但 stdout 本身仍不能独立构成长跑收据。vendor smoke/probe 只能在 tracked
authority manifest 完成 runtime materialization 后作机械诊断；`long` 另须实际 probe 产出的
命名 `vendor_probe_gate_receipt`，formal 仍受自身 receipt 合同约束。

## 8. 训练后 selector

能力工件的查询至少包含 incoming ball、outgoing aim、base context、exact policy/catalog/profile
identity。planner 对任意目标按以下顺序：

1. motion、安全和 solver feasibility 硬门；
2. support/OOD；域外直接 abstain；
3. 校准成功率 lower confidence bound；
4. 在通过最低成功率的动作中结合 priority、准备/恢复成本做 tie-break；
5. 全部不通过则明确 abstain。

生产 ROS/C++ 当前仍按正反手 sign 折叠成两个 clip。训练可以先不接 selector，但部署前必须把 stable
action ID 作为控制面字段跨 Python wire、revision gate、ONNX catalog 和 C++ policy 全链保留下来；
不得因此把 categorical UID/slot 重新塞回 actor observation。

## 9. 发射、继续与当前阻塞

开正式长跑前以及运行中继续扩域/晋级时必须依次通过：

1. host：manifest/sampler/curriculum/solver/runtime/exact-resume contract tests；
2. 新正手 `[0,-5,-10] cm` station sweep；upper/full 共同取最近通过档，正式报告 action-specific
   `t_hit`、`t_cycle`、physical `right_racket` site speed、全周期 table clearance；
3. exact N5 teacher fitted-ball MuJoCo 门：每件动作的对应来球必须在真实桌、网、球、拍面接触和
   两档 `dt` 下由该动作的 exact 击球帧/时间律合法过网落台；解析 task 可解或 virtual landing
   不能替代这条物理门；
4. 先冻结 N5 fixed-width continuous future-motion intent/preview，再允许构造 actor；N1 v2
   不得直接复用到 shared-ready 多动作。N5 exact motion view 严格使用
   `bh_loop_c, v12_forehand_block, bh_block, s0_highpress, fh_loop_high`，排除旧
   `fh_loop / fh_block_syn`，五件均持 opaque training admission；
5. Pod1 双 GPU：GPU0 只跑 trainer，GPU1 只跑冻结 policy evaluator；两边独立持 no-clobber
   identity/lock，evaluator 不做 PPO update。依次过 CPU 回归、table scene、1 env × 2 update、
   单动作 center、N5 center-only canary 和本次 N5 规模的压力门；
6. center-only canary checkpoint 必须先过 learned-policy
   [`MuJoCo ActionBall policy fitted gate`](../DEFINITIONS.md#mujoco-action-ball-policy-fitted-gate)，
   证明 policy 实际执行时仍能把同一对应来球安全打上台；teacher PASS 不替 policy PASS。通过后
   才能把 center-only canary 续成动态长跑，此后每个预注册 milestone 重复本门；
7. 动态 marginal → joint 课程；
8. Pod2 只有在 exact ordered 73 件 manifest、逐件 compiler/安全/admission 证书和固定宽
   continuous future-motion actor 合同全部存在时才可 formal N73。此前只允许 CPU inventory
   或独立 N8/N12 canary，不得把它续写成 N73；
9. GPU 必须现场确认空闲，不杀别人的进程、不清未知 lock。

源码、host test 或 teacher receipt 都不等于 learned-policy gate 已通过。截至 2026-07-29，
fresh N5 尚无一份 exact checkpoint/ONNX 的 policy fitted-ball formal PASS receipt；该门明确为
**未通过**，不得据此宣称长跑结果可部署或接真机。生产 N-action selector 也仍是训练后独立工作。
