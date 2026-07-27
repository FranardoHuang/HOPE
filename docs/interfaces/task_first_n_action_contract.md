# Task-first 任意动作数训练合同

Status: Superseded / ablation-only。本文保留 task-first 历史语义，不再是候选 executor，也不授权
正式训练、部署或真机。当前候选合同是
[按动作条件化 Ball-first](action_conditioned_ball_first_contract.md)。

共享术语见[术语与人话对照](../DEFINITIONS.md)：本文的
[`task-first`](../DEFINITIONS.md#task-first)是“先定义动作应完成的球拍任务，再学该动作附近的任务
泛化”，不是先采来球再反解动作；[`task-first manifest`](../DEFINITIONS.md#task-first-manifest)
绑定动作和课程范围；[`Wilson 置信界`](../DEFINITIONS.md#wilson-confidence-bounds)用于晋级与回退；
[`stable action UID`](../DEFINITIONS.md#stable-action-uid)是跨工件身份，本地 slot 只用于数组索引。

## 1. 范围与非目标

训练 executor（执行策略）时没有球：

```text
动作身份 + 该动作中心任务
  -> 在当前课程范围内采样球拍/站位目标
  -> policy 执行
  -> 按球拍位置、速度、拍面、base 误差和安全事件记一次 attempt
  -> 逐动作扩大或回退任务范围
```

这一阶段不读取来球，不发解析落台 Reward，不运行“球 → selector → adapter”，也不让 planner 在
挥拍中换动作。它只回答：**给定一个动作，这个 policy 在该动作中心附近能可靠完成多大的 task
区域？**

训练完成后再用冻结的 ball-conditioned 留出卷测能力，并把结果交给
[动作能力 selector 合同](action_capability_selector_contract.md)。训练内课程统计不能直接冒充
planner 的实时成功率。

## 2. 任意 N 动作身份

一份启动 manifest 必须以 exact bytes 的 SHA-256 钉住，并包含：

- 非空、有序且无重复的 `action_order`；
- 每动作的 `action_id`、内容派生 `action_uid`、motion 相对路径与 SHA-256、strike phase、
  正反手 family sign 和球拍 mount normal sign；
- 完整任务包络、课程 Gate、冻结留出 seed/split/sample count；
- 明确的 `training_authorized`。

动作数为 `N` 时，actor 合同必须写成 `task_first_n<N>`，不能只靠 tensor 宽度猜。布局是
`hitter_footwork` 的 177 维前缀，加 `racket_target_normal_cmd(4)`，再加
`action_one_hot(N)`，总维数为 `181 + N`。manifest 顺序、motion loader 顺序、one-hot 顺序和
checkpoint hard contract 必须完全一致。

`action_uid` 由动作字符串、正反手 family 和 motion 内容 SHA 派生，范围为
`1..2^53-1`；换 motion bytes 或换 family 必须换 UID。dense slot 是本次 catalog 的本地
`0..N-1` 索引，可随 catalog 重排，不得写进跨版本能力身份。

缺字段、未知字段、非有限数、重复 JSON key、路径逃逸、manifest SHA 不符、motion SHA 不符、
`training_authorized=false` 或动作顺序不符都必须在构造环境前失败。

## 3. 每动作任务中心

每个动作先有自己的 reference strike 中心：

- 球拍位置 `p_ref`；
- 球拍速度方向 `v_ref / |v_ref|` 与标量速度中心 `|v_ref|`；
- 有符号拍面中心 `n_ref`；
- base 平面中心 `b_ref`。

`station_center_shift_xy_m` 是整个动作/任务中心的平移，不是只改 base Reward 的补丁。设
`station = (sx, sy)`，本动作的一次 task 为：

```text
racket_position = p_ref + [sx, sy, 0] + delta_position
racket_speed    = |v_ref| + delta_speed
racket_velocity = normalize(v_ref) * racket_speed
racket_face     = sample_in_cone(n_ref, face_cone)
base_xy         = b_ref + station + delta_position_xy + delta_base
```

因此 level 0 也会使用选定的 station center；后续 `delta_base` 才教同一球拍点下身体相对站位的
容差。HOPE 轴向中 `+X` 朝球台/对手，故 `sx < 0` 是整个人、整段动作和 task 一起远离球台。
完整 frame 语义见[坐标合同](frames_and_coordinates.md#task-first-station-center)。

`delta_speed` 只扩标量速度大小，首轮不同时转动速度方向。若以后扩速度方向，必须新增 manifest
schema 和独立课程轴，不能把它藏进现有 `speed_delta_mps`。

## 4. 逐动作课程

每个动作独立按下列固定顺序扩范围：

```text
position -> scalar speed magnitude -> face cone -> base residual
```

每一轴只取 `0, 0.25, 0.5, 0.75, 1` 五档；当前轴到 `1` 后下一轴才从 `0` 开始。某动作晋级不会
替另一个动作晋级，也不能把五个动作的成功率平均后一起放宽。

level 0 是严格的**中心 warm-up**：四个 curriculum delta 全为零，只重复动作中心 task；它不是
“默认已有一点泛化”。第一次真正的 task 泛化发生在 position 从 `0` 晋到 `0.25` 时。固定的
`station_center_shift_xy_m` 属于中心定义，因此不算 level-0 泛化。

每次课程决策同时使用：

- 最少 attempt 数；
- 成功率 Wilson 下置信界；
- unsafe 率 Wilson 上置信界；
- 晋级阈值、回退阈值、连续决策 dwell 和 stall policy。

晋级与回退使用不同阈值形成 hysteresis（滞回），每次最多移动一档。一次 PPO rollout
（Proximal Policy Optimization，一种批量强化学习更新）结束时才关闭本轮证据并做原子决策；
任一动作缺最少 attempt 时不得拿其他动作的样本补足。balanced sampler（均衡动作采样器）保证
任意前缀内各动作样本数最多差一，并把 permutation/cursor 写入 checkpoint。

## 5. Attempt 与成功语义

attempt 在动作 wrap、真实 reset 或已登记的 unsafe 终止时恰好关闭一次。计数必须使用闭合前锁存的
动作身份，不能把 reset 后的新 action slot 记给上一拍。

一次成功必须同时满足：

1. 在 strike opportunity 到达 exact sample；
2. 球拍位置误差低于配置阈值；
3. 球拍**标量速度**误差低于配置阈值；
4. 有符号拍面角误差低于配置阈值；
5. base 平面误差低于 `task_first_base_success_thresh_m`；
6. 从出发到恢复没有桌面碰撞或 physical fall。

table hit 与 physical fall 是互斥 unsafe 分类；成功和 unsafe 也必须互斥。缺 contact sensor、
非有限值、counter 不可达或重复 provider 都失败封闭，不能制造零值或继续晋级。所有阈值、分母、
逐动作 counts、置信界和课程变更都写入每个 rollout 的 canonical receipt。

## 6. 新正手与 station 选择

候选新正手 `fh_loop_high` 只允许比较 `[0,0]`、`[-0.05,0]`、`[-0.10,0] m` 三个整动作
station center。选择规则是：对 upper/full 两个 scope，取**离原站位最近且同时通过**以下门的一档：

- post-retime 行为/contact authority 给出的动作特定 `t_hit` 范围；
- `t_cycle`、共同 ready 回位与恢复正常；
- 厂商 MuJoCo `right_racket` physical site 的击球速度达标，不能用 wrist COM 速度替代；
- 整轨无机器人/球拍撞桌、无地面或身体安全失败；
- 参考回球 Gate 和后续 Isaac filtered contact sensor 冒烟通过。

source anchor 只作 compiler diagnostic，不是正式 `t_hit` 真值；不存在通用
`t_hit <= 0.5 s` 硬门。动作时序定义见
[动作预处理合同](motion_preprocessing_contract.md#behavior-timing-authority)。

当前新正手只有 source 候选，upper/full 正式输出、grounded collocation trace 和训练证书均缺失，
所以 `training_authorized=false`。当前 certifier 只产内容绑定的未授权 reference checks；
`diagnostic_smoke_authorized` 恒为 false，既不能授权 simulator smoke，也不能把 manifest 改成可训练。

## 7. 恢复与收据

checkpoint 必须绑定 manifest 文件 SHA、canonical SHA、action order/UID、课程 Gate、每动作档位与
dwell/stall 状态、balanced sampler 状态、命令 term 的 distribution-affecting buffer，以及
Python/NumPy/Torch/CUDA 随机数状态。字段、shape、dtype、device 数或 identity 不符时拒绝恢复。

这里的“严格恢复”只指课程、采样器和随机状态的持久状态不被静默丢失。runner 仍会重建并 reset
仿真环境；在 simulator state 没有序列化前，不得声称中断恢复与不中断运行逐物理 step 完全相同。

## 8. 发射门

正式 task-first run 必须同时满足：

1. manifest `training_authorized=true`，文件 SHA 与所有 motion bytes 对上；
2. 当前五动作 training view 明确排除旧 `fh_loop`，新 `fh_loop_high` upper/full 证书齐；
3. host contract tests、Pod Isaac 构造/两迭代 smoke 和 filtered table-contact negative/positive
   controls 全过；
4. composed effective Reward receipt 与预注册 SHA 相同；
5. 没有 ball/question-bank/planner revision/target delay-noise/retiming/event-timing 或 mid-swing
   switch 的残留配置；
6. 无未审 WARN、NaN、动作 starvation、重复/缺失 rollout receipt。

任一项未过时只允许静态检查。若另立 simulator smoke，必须由独立预注册与操作权限明确授权，不能
引用当前 certifier 的 reference check 充当授权；正式长训命令仍不提供。
