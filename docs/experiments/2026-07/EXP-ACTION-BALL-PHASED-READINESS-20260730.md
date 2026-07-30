# EXP-ACTION-BALL-PHASED-READINESS-20260730 — ActionBall 分阶段训练准备账本

- 状态：`preregistered`
- 阶段/轴：AgiBot A3，动作条件 Ball-first，N=1 诊断长训 → 正式 N=5 → N=73 → 部署
- 集成小目标：先产出一份可迭代的 N=1 policy，同时把不应阻塞首跑、但必须在后续阶段关闭的工程债写清楚
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E2`
- 创建日期/最后复核日期：2026-07-30 / 2026-07-30

共享缩写按[术语与人话对照](../../DEFINITIONS.md)解释。本文是依赖门和技术债账本，
**不是新的全项目优先级队列**；运行顺序、算力认领和当前采用 setting 仍只认
[`origin/main` 的 `NOW`](../../NOW.md)。本文只回答一件事：某项工作最迟应在首个 N=1
长训、1000 update 检查、正式 N=5、N=73 或部署中的哪一个边界前闭合。

## 1. 第一性原理裁定

### 1.1 先产 policy，但不能在错误合同上产

首个 N=1 训练只需要证明：

1. 动作身份在一拍内冻结；
2. 来球先采样，再由同一动作的 fixed-action solver 生成相对 task；
3. 出生、初始 `qdes`、上一动作、actor 观测和 teacher frame 0 描述同一 ready；
4. episode 能活过该动作 `t_hit`，checkpoint finite，table/fall/nonfinite/raw-hard 分账可信；
5. 训练设置、policy recipe、spec 和结果记录可复算。

它不需要等待正式 N=5 的完整审计基础设施、最大课程 support、N=73 动作 admission、
MuJoCo/C++ deploy consumer、full-body 对照或最终 command governor。反过来，观测列错位、
task 变绝对坐标、桌碰传感器恒零、出生姿态与控制目标冲突会直接改变 policy 学到的问题，
这些不能用“先跑再说”绕过。

### 1.2 观测合同：相对 task + 绝对桌体上下文

当前候选是 `action_ball_table_pose_twist_n<N>`；N=1 actor 宽度为 **194**：

| 区间 | 项 | 维度 | 语义 |
| --- | --- | ---: | --- |
| `[0,177)` | `hitter_footwork` prefix | 177 | reference、机器人本体状态、上一动作和相对击球 task |
| `[177,180)` | `base_position_table` | 3 | base 在桌面中心坐标系中的 XYZ |
| `[180,186)` | `base_orientation_table_6d` | 6 | base 相对桌体的完整 roll/pitch/yaw，连续 6D 旋转表示 |
| `[186,189)` | `base_lin_vel_heading` | 3 | base/root COM 在 base yaw-heading frame 的三轴线速度 |
| `[189,193)` | demanded face + reserved scalar | 4 | 有符号拍面 task 与保留标量 |
| `[193,194)` | `action_one_hot` | 1 | 冻结的 N=1 动作身份 |

N 动作总宽度为 `193+N`。三个角度没有被删掉：6D 旋转用旋转矩阵前两列表示完整
SO(3)，经正交化后第三轴唯一确定；它不是 yaw-only。采用 6D 而不是 Euler 三角或裸 quaternion
是表示合同修复，不是 Reward 假设，不做学习 A/B。依据是 Zhou 等人在 CVPR 2019 的
[连续旋转表示结果](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)。

task 位置继续保持机器人相对坐标：

- `base_target_pos_b` 是当前 base 到 base goal 的相对二维残差；
- `racket_target_pos_b` 是当前球拍 FK 到目标击球点的相对三维残差；
- 桌体 9 值回答“机器人相对球桌站在哪里、朝向如何”，不把 task 改回 world absolute。

这种拆分同时满足泛化和可观测性：相同相对 task 可以在不同站位复用，而 policy 仍知道自己与桌边、
桌面的绝对几何关系。

### 1.3 传感器权威

部署侧按物理量拆分，不做“整套 OptiTrack 或整套 IMU”二选一：

| 量 | 维度 | 首选权威 | 原因 |
| --- | ---: | --- | --- |
| table-relative base position | 3 | 360 Hz OptiTrack rigid-body pose | 无积分漂移，直接给外部位置 |
| table-relative base orientation | 6 | OptiTrack orientation 经 marker→`base_link` 与 table SE(3) 标定 | 直接给全局姿态和 yaw，无 IMU yaw 漂移 |
| `projected_gravity` | 3 | 同一 OptiTrack orientation 派生 | 与绝对姿态使用同一权威，避免静默混源 |
| `base_ang_vel` | 3 | pelvis IMU 三轴陀螺仪 | 直接测角速度；mocap 姿态差分会放大噪声并引入滤波延迟 |
| `base_lin_vel_heading` | 3 | 因果状态估计器：OptiTrack 位置作无漂移锚，可融合 IMU 加速度 | 单帧 mocap 不直接给速度，纯差分噪声大；纯 IMU 积分会漂移；最终旋入与 task 相同的 yaw-heading frame |

所以 `base_ang_vel` 是 **3 值**，`base_lin_vel_heading` 也需要 **3 值**。线速度在当前无 history 的
feed-forward actor 中能区分“同一姿态、正在倒/正在平移”和“静止”，因此值得进入 actor；
但部署 producer 必须处理 marker 到 root/COM 的位移以及旋转刚体的
`omega × offset`，不能把 marker 速度冒充 COM 速度。

训练中这些量使用 simulator rigid-body truth 的同语义 counterpart。真实 producer 尚未闭合
marker rotational extrinsic、时间同步、丢帧/延迟和 causal velocity estimator，所以 194-D
训练合同目前不授权真机。

### 1.4 joint 和 task 信息是否够

194-D 合同中已经有：

- 31 个 joint position（相对 default）；
- 31 个 joint velocity；
- 31 个上一 policy action；
- 三轴 base angular velocity、三轴 base linear velocity、projected gravity；
- motion anchor orientation、62-D teacher command；
- 相对 base goal、相对 racket target、目标拍速、time-to-strike、swing identity；
- demanded signed face、完整 table-relative base pose、冻结 action identity。

这里的 `HITTER` 是 177-D prefix 的来源；仓库没有一份独立、可替代上述布局的 “SMASH actor
contract”。SMASH 相关论文/实现可提供 reward、噪声或训练方法启发，但不能凭名字推断另一组列。
这也符合两篇论文的系统边界：
[HITTER](https://arxiv.org/abs/2508.21043)由上层 planner 生成击球位置、速度和时序，再让
whole-body controller 执行；[SMASH](https://arxiv.org/abs/2604.01158)把 onboard egocentric
perception 和 scalable whole-body skill 作为自己的系统组成。后者的感知输入不能在没有
frame、latency、deploy producer 和 checkpoint metadata 的情况下直接拼进 HITTER-derived prefix。
当前不再为首个 N=1 添加 raw ball state、角加速度、接触历史或 observation history：
fixed-action solver 已把每颗来球编译成当前相对 task；在真实 PPO 证明存在不可观测别名以前，
继续加列只会扩大接口和部署债。

### 1.5 bang-bang：先量化，再加约束

现役合同已有 raw/processed action 平滑项、实际 joint acceleration/limit 账、
finite qdes projection、投影前超出量 penalty 和 actual soft/hard 分账。首个 N=1 long
**不再临时提高动作变化惩罚，也不加入 EMA、velocity/acceleration/jerk governor**，原因是：

1. 快速击球依赖短 contact window 内的高拍速；未按 A3 电机和 teacher 导数标定的硬限速会直接
   延迟 `t_hit` 或削掉拍速；
2. 更强软 penalty 可能压过模仿/击球收入，却不能给硬安全保证；
3. stateful governor 会改变 action→executed command 合同；若 actor 不看 executed state，
   同一个 actor action 会对应不同后继状态，制造新的部分可观测性；
4. 当前最大问题仍是先取得 healthy strike denominator，而不是凭视频印象调平滑权重。

1000 update 检查必须新增或确认以下量尺：逐关节 executed-command 换向率、归一化一阶/二阶差分、
torque saturation、qdes projection/saturation、实际关节 hard/soft 驻留、击球窗拍速和 table/fall。
只有这些量证明 bang-bang 持续且与参考无关，才按以下次序开小预算 canary：

1. 首选低剂量、按每个 A3 电机能力归一化的 executed-`qdes` 一阶/二阶差分 penalty；
2. 若仍有 policy 独有的持续高频抖动，再比较 CAPS temporal regularization；
3. EMA 只作分部位诊断，必须量相位延迟和拍速损失；
4. 最后才评估逐电机 velocity/acceleration governor；击球手臂与腿/腰使用不同限制，actor
   同时加入 executed `q_cmd`/`qd_cmd`，若限制 jerk 再加 `qdd_cmd`，并要求 Isaac、MuJoCo、
   C++ 逐字同合同。

每项都同时比较 intervention rate、teacher phase delay、strike/return、racket speed 和
imitation，不以总 Reward 单独晋级。当前两个 dynamic-ready recipe 的 raw actor bias 约落在
`[-5.87, 16.30]`，所以不能套用 `[-1,1]` clip；那会让部分动作专属 ready 本身不可达。

EMA 只可作为诊断对照，不是最终安全边界；Ruckig/jerk 级在线轨迹器是部署候选，不阻塞首个
policy。[Ruckig](https://arxiv.org/abs/2105.04830)确实能对多自由度轨迹施加 velocity、
acceleration 和 jerk 三阶约束，但这只证明该类 governor 有严格约束实现，不证明未经 A3
参数标定的 governor 对快速击球无损。

## 2. 已完成证据（截至本分支 2026-07-30）

| 项 | 证据 | 当前边界 |
| --- | --- | --- |
| dynamic-ready 出生与 hold | loop/block 在 Pod 各闭环保持 `0.8 s / 40` policy steps，双脚接触率 `1.0`，table/fall/hard/nonfinite 零；可见 raw-reset 图证明原生 reset 直立 | 证明出生与 nominal hold，不证明 teacher 全轨、strike 或 long |
| dynamic-ready trainer 接线 | candidate/hold receipt 双 pin；physical state、初始 qdes、last action、actor bias 和 motion frame 0 原子一致；Pod focused `63 passed` | 当前只授权 exact N=1 diagnostic |
| 170-update overflow | diagnostic joint-safety summary 已改为每 update 按事务排空 | 旧 overflow checkpoint 不续；fresh run 重新开始 |
| qdes 安全语义 | finite request 投影到合法包络，penalty 读取投影前超出量；nonfinite/actual hard 仍终止 | Reward 剂量仍需 healthy baseline 后 canary |
| reset 诊断热路径 | diagnostic per-reset 逐 env 完整转录已移出；不可变 receipt SHA、strike timing、metrics D2H 等已做确定性优化 | formal 路径仍保留完整 per-reset 仪式；每步 ledger/broker 税仍开放 |
| 194-D actor source | `203b2d92` 实现 table pose + base twist；Pod dependency-light 合同回归 `290 passed, 9 skipped, 1 deselected` | C++/ONNX/MuJoCo producer 未实现，旧 182/191-D checkpoint 不可复用；policy recipe 与 observation 分层，现有 dynamic-ready recipe 可复用 |
| table contact source | Pod1 `eb2799b1` r26：五件 kinematic collider、32-body×5-role matrix、四子步与五 role 真实正控均通过；五 probe reset 后零泄漏，unsupported/Traceback/FAIL 均为 0，出现 `main_completed` | E2 receipt 已保存；训练仍逐 run 监测 table counter |

这些都是功能分支证据；进入 `main` 前不改变当前采用 setting。

## 3. 分阶段最迟闭合项

### 3.1 首个 N=1 long 前

直接修/验证，完成即开跑，不等待后续工程完美化：

1. table-contact Pod smoke 已在 `eb2799b1` 完成：五个 role 均有真实正控，32 个 matrix
   shape/order 正确，`robot_hit_table` 会触发，reset/settle 后零泄漏，日志无 unsupported
   filtered target；receipt 见
   [`table_smoke_eb2799b1_gpu1_r26.receipt.json`](../../../configs/n1_contact_dynamic_ready_20260730/table_smoke_eb2799b1_gpu1_r26.receipt.json)；
2. 用当前 dynamic-ready policy recipe 构造 **194-D** actor；recipe 只绑定 PPO/decoder/ready，
   observation name/width/term order 由训练 hard contract 单独绑定，所以无需为 194-D 重物化
   recipe；旧 182/191-D checkpoint 一律不续；
3. 两动作各跑 `1 env × 2 updates`，确认真实 PPO update、finite checkpoint、q/qdes/last-action/
   ready 一致；
4. 两动作各跑 `4096 env × 5 updates` 构造 probe，记录每 update 秒数、episode 是否跨 `t_hit`、
   strike opportunity、raw-hard/table/fall/nonfinite 和 checkpoint；
5. 只要一条动作过构造门且没有硬错误，立即发该动作 fresh `1001 updates`（0 起数，确保生成
   `model_1000.pt`）；另一动作可串行排在同一空闲槽，不因它阻塞先过门者；
6. 把 exact spec、recipe、run record 和复现命令纳入 repo；所有 namespace fresh/no-clobber。

下列项**不阻塞**首个 diagnostic N=1 long：

- formal per-reset receipt checkpoint compaction；
- generic formal N5 launcher；
- full-body、Reward、reference guard、课程 failure target、entropy/sigma/RSI 消融；
- 8192 env；
- MuJoCo/C++/真机 194-D producer；
- EMA 或 command governor。

### 3.2 1000 update 检查

千轮不是任意调参点。按以下顺序判读：

1. checkpoint 是否持续 finite、exact identity 是否未漂移、有无 Traceback/summary overflow；
2. 动作是否仍像 teacher：逐动作看 motion/pose error、击球窗相位、关节抽动、table/fall/hard；
3. 是否产生足够 strike opportunity；五轮没有 strike 不能判学习失败，1000 轮仍无 strike 才进入
   根因分支；
4. 长期 strike/return/landing 与四组 realized Reward income；
5. 课程 proposed/admitted/rejected、safe denominator 和每 arm/side frontier 是否开始推进；
6. bang-bang 量尺是否显示持续的 executed-command 高频换向或饱和。

如果 dynamic-ready/episode 已健康跨 `t_hit`，但 200–1000 update 仍无 strike，下一嫌疑是
official low-gain waist plant 下 teacher 动态不可跟踪；先比较部署一致的腰 gain 或 reference
retiming，不回头反复改出生，也不先放宽 hard limit。

只有 activation denominator 非零后，才允许小预算 canary：

- tracking 权重/Reward 剂量；
- reference guard 的 metrics-only、phase-gated 或连续约束方案；
- qdes projection penalty 剂量；
- curriculum safe-policy failure 目标（默认约 10%，20% 仅对照）；
- upper/full-body；
- entropy/sigma/RSI；
- bang-bang governor/EMA。

这些是经验选择，必须 canary；数学等价、合同修复和删除重复同步不做科学 A/B，但仍做 Pod
parity/吞吐验证。

### 3.3 正式 N=5 前

以下可以晚于首个 N=1 policy，但**不得晚于 formal N=5**：

1. **正式收据粒度改造。** 目前只有 diagnostic 快路径；formal 仍在每个 env-reset 做完整
   Python transcript/receipt 仪式。改为热路径紧凑 device event/assignment tape，
   checkpoint/hourly 批量物化完整 receipt；保留 proposal 分母、reject reason、action/domain/
   birth/sample/task/lifecycle/outcome，不接受仅 `seed+config`；
2. **重建与 exact resume。** fixed tape 下从紧凑日志重建旧式完整 receipt，逐字段 parity；
   checkpoint save/load/no-step roundtrip、篡改负例、segment head 和外部 resume pin 全过；
3. **分段 profiler 与 all-mode 热路径。** 分开 physics rollout、metrics/D2H、termination、
   safety archive、birth/retire、sampling/solver、state write、PPO；据实修每步 ledger clone/
   string 重建、残余 D2H、broker per-env Python。`~7 ms/env-reset` 只作为混合上界，未经 profiler
   不写成精确归因；
4. **generic formal launcher 升级。** 任意 N launcher 识别 194-D family、dynamic-ready/
   table-pose-twist identity、真实 Reward/PPO/plant/solver SHA 和 exact resume；
5. exact ordered N=5 manifest、五件 motion bytes/admission、动作专属 ball center/support、
   new-forehand `t_hit/t_cycle/site speed/table clearance/recovery` 和非空 trust set；
6. formal Reward causal receipt、frozen evaluator canary/heldout、table smoke、stage evidence 和
   checkpoint identity 全闭合；
7. 运行规模压力证据只按 N=5 的 N/E/R 合同给 N=5，不让 N=73 缺口反向阻塞。

正式路径的 per-reset receipt 改造是**有意留到这里而不是漏忘**：它不改变 N=1 学习问题，
却决定 N=5/N=73 是否能在短 episode 下接近正常吞吐和可用 checkpoint 大小。该改造不做学习
A/B；验收是 numerical/state parity、旧收据重建、exact resume 和固定工作量 Pod 吞吐。

### 3.4 N=73 前

1. exact ordered 73 manifest 与每件 motion/compiler/safety/admission 证书逐项闭合；
2. 每动作自己的 ready、`t_hit/t_cycle/site speed`、ball center/support 和 solver profile，
   不把 N=5 的 shared-ready 假设扩散到 73；
3. full-body actor/reference 和 action identity one-hot 的任意 N 路径通过；
4. N=73 对应的 sampler/pool/broker/curriculum/checkpoint 压力、compaction 和 exact resume；
5. 逐动作独立 frontier、强制覆盖/starvation、center/interior/frontier 混合和 frozen heldout；
6. 若启用 base move，先证明真实 spawn 与 no-move/move goal 语义、preparation window 和移动恢复；
7. N=73 仍在 Pod 独立 smoke/canary，不能由 N=5 run 续成或更换 action order。

### 3.5 部署前

1. C++/ONNX/MuJoCo 全部支持 exact 194-D ordered terms、normalization、metadata 和 stale/dropout
   语义；
2. OptiTrack marker cluster→`base_link`、venue→table 的完整 SE(3) 外参、时间同步、Motive
   smoothing/端到端 latency 和遮挡处理进入部署合同；
3. 三轴 gyro 与 OptiTrack pose 的 frame/extrinsic 对齐；base linear velocity estimator 完成
   mocap anchor、可选 accelerometer 融合、marker→COM 刚体速度修正；
4. 用真实噪声、延迟和 dropout 重新做 Isaac/MuJoCo observation parity；不能把 simulator truth
   直接当部署证据；
5. 若采用 governor，Isaac/MuJoCo/C++ 使用同一逐电机 velocity/acceleration/jerk 参数、executed
   state observation 和冲突处理；未采用时也必须保留 hard q/effort/velocity 安全边界；
6. 同一 checkpoint 在 MuJoCo 验证动作方向、teacher fidelity、table/ball/racket 物理和无桌碰；
7. G06/G07 dry-run、joint order、scale、stop 和 no-publish safety gate 通过前不得接真机。

## 4. 决策账本

| 项 | 分类 | 最迟边界 | 是否需要学习 A/B | 当前决定 |
| --- | --- | --- | --- | --- |
| 6D table-relative orientation | 直接修 | N=1 long | 否；做 tensor/recipe parity | 已实现，保留完整三角信息 |
| base linear velocity 3-D | 直接修 | N=1 long | 否；做 observation parity | 已进入 194-D actor |
| OptiTrack pose + gyro angular velocity | deploy contract | 部署 | 否；做传感器延迟/噪声实测 | 采用分量级最优源，不整套弃用 IMU |
| dynamic-ready 原子合同 | 直接修 | N=1 long | 否 | 已实现，待 194-D 新 recipe/smoke |
| table-contact filtered truth | 直接修 | N=1 long | 否 | Pod r26 E2 已过；训练继续逐 run 分账 |
| diagnostic receipt 快路径 | 直接修 | N=1 long | 否 | 已实现 |
| formal checkpoint 粒度 receipt | 直接修 | formal N=5 | 否；做 parity/吞吐 | 开放，不能拖到 N=73 |
| ledger/D2H/broker 热路径 | profiler 后直接修 | formal N=5 | 否；做固定工作量吞吐 | 开放 |
| Reward 权重与负项剂量 | canary | 1000 update 后 | 是 | 不阻塞首跑 |
| reference guard/CaT | canary | healthy baseline 后、formal N5 采用前 | 是 | 继续分账，不热改 |
| curriculum 10%/20% | canary | healthy baseline 后 | 是 | 10% 默认，20% 只作对照 |
| full-body | canary | N=73 前 | 是 | 不阻塞 upper N=1 |
| EMA | canary | bang-bang 量尺异常后 | 是 | 仅诊断，不作最终安全边界 |
| executed-qdes 归一化 penalty / CAPS | canary | 1000 update 后且 bang-bang 量尺异常 | 是 | 优先于 EMA/governor，仍须验收击球相位与拍速 |
| velocity/acceleration governor | canary + deploy parity | 部署前；可在 1000 后试 | 是 | 不阻塞首个 N=1；需逐电机标定和 executed-command observation |
| jerk/Ruckig | 暂不修 | 部署后续候选 | 是 | 当前收益证据不足，接口成本高 |
| 8192 env | 性能 canary | formal N5 吞吐健康后 | 是 | 不用更多 env 掩盖热路径问题 |

## 5. 收口判据

### 可发首个 N=1 `milestone1000`

- table smoke receipt 可信；
- 当前 dynamic-ready recipe 的 SHA 已验证，194-D hard contract 已由 smoke 实例化；
- `1 env×2` 与 `4096×5` 有真实 PPO update、finite checkpoint；
- episode 能跨动作 `t_hit`，没有 NaN/identity 漂移/持续 table/fall/raw-hard 爆炸；
- exact spec、recipe、run record 已保存。

### 可从 1000 update 转 reviewed long

- checkpoint 和训练连续性正常；
- teacher imitation/strike denominator 可解释；
- unsafe 与 Reward income 分组未失效；
- 若 strike 仍为零，teacher dynamic feasibility 根因已判明，而不是靠临时改 Reward 掩盖。

### 可发 formal N=5

- formal receipt checkpoint compaction、重建、exact resume 与 fixed-workload throughput 通过；
- generic 194-D launcher、exact N=5 action/admission/ball/support、Reward/table/evaluator 全闭合；
- 使用新的 clean/no-clobber lineage，不把 N=1 diagnostic 结果冒充 formal 证据。

## 6. 证据入口

- 当前训练 Gate：[G05 Isaac training first loop](../../gates/G05_isaac_training_first_loop.md)
- ActionBall 语义：[按动作条件化 Ball-first 合同](../../interfaces/action_conditioned_ball_first_contract.md)
- actor/传感器接口：[Policy observation/action](../../interfaces/policy_observation_action.md)
- N=1 发射步骤：[消融与 dynamic-ready 发射工序](../../operations/run_ablation_wave_launch.md)
- formal N=5 工序：[no-clobber ActionBall 发射](../../operations/run_action_ball_curriculum_no_clobber.md)
- table truth：[ActionBall 桌体安全 smoke](../../operations/run_action_ball_table_safety_smoke.md)
- Reward truth：[ActionBall 发射前 Reward 因果审计](../../operations/run_action_ball_reward_causal_prelaunch.md)
- 设计和吞吐审计：[N1 设计背书审计与训练加速尽调](../../research/design_audit_and_speedup_20260729.md)

本文不自行声明 Gate `Done`，不更新 `NOW`，也不授权真机。
