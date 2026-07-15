# 稀疏 Reward 资格账本接口

本接口定义“某个 milestone 的 Reward 是否真的得到过学习机会”，不判断哪个策略更好。共享术语见
[术语与人话对照](../DEFINITIONS.md)。实现由训练源码发出每 PPO update 的整数计数，再由只写 receipt
的 classifier 验证累计 milestone 窗口。它不停止、重启、晋级 trainer，也不购买第二 seed。

## 训练侧唯一计数源

`RacketTargetCommand` 在解析 virtual-ball 的同一步、同一 mask 上记录非衰减整数：

| TensorBoard tag 后缀 | 人话分母/分子 |
| --- | --- |
| `strike_opportunity_count_<action>` | 该动作到达 exact-strike 的次数 |
| `virtual_capture_count_<action>` | 上述机会中通过 signed face、拍心距离和接近速度门的次数 |
| `virtual_net_clear_count_<action>` | 已捕获后解析轨迹过网的次数 |
| `virtual_landing_valid_count_<action>` | 已捕获后解析 rollout 得到有限落点的次数 |
| `virtual_legal_return_count_<action>` | 同时过网并落在对面台内的解析回球次数 |

现役两动作 emitter 的 `<action>` 为 `forehand/backhand`。新五动作不能借用这两个名字；动作配置必须先
把自己的完整动作族写入 classifier contract，并让 command emitter 发出同名计数。classifier 要求输入
动作集合与冻结合同精确相等，少一类或多一类都记 `MEASUREMENT_INVALID`。

qdot 通过一个 RewardManager-stage 零返回探针记录：

| TensorBoard tag | 人话 |
| --- | --- |
| `Live/qdot/observed_sample_count` | 用 runtime 31 关节速度/上限实际算过的 environment sample 数 |
| `Live/qdot/hinge_active_sample_count` | 实际非零权重 hinge Reward 路径执行的 sample 数；对照必须为 0 |
| `Live/qdot/excess_sample_count` | 至少一个关节超过 `margin × runtime limit` 的 sample 数 |
| `Live/qdot/normalized_excess_square_sum` | 同步公式输出之和，只作方向量，不代替前三个整数分母 |

探针 manager weight 为 `1.0`，但返回逐环境严格零；当配方显式写 qdot weight/margin 时，Hydra 同时把
探针参数绑定到实际 hinge。probe 与 treatment 用 `common_step_counter` 去重：同一 environment sample
只记一次 observed，treatment 另记一次 active。

## Milestone measurement JSON

每个输入是从训练开始累计到一个 checkpoint milestone 的完整窗口，必须包含：

- `run_id/source_commit/training_claim_sha256/checkpoint_sha256`；
- `counter_window.start_update_exclusive=-1` 与 `end_update_inclusive=milestone`；
- 冻结动作族的五个非负整数；
- qdot 三个整数与 finite 非负 excess sum；
- `same_step_virtual_ledger=true`、`runtime_qdot_limits_bound=true`、窗口完整/reset-at-start；
- `virtual_outcome_semantics=analytic_virtual_contact_phase_a` 且
  `physical_contact_phase_b_observed=false`。

计数必须满足 `legal<=net<=capture<=opportunity`、`legal<=landing<=capture`、
`active<=observed`、`excess<=observed`。qdot treatment 要求 `active==observed`；对照要求 `active==0`。
任一闭包失败都不是“Reward 差”，而是 `MEASUREMENT_INVALID`。

## 五态分类

| 状态 | 含义 | trainer 动作 |
| --- | --- | --- |
| `NO_OPPORTUNITY_CONTINUE` | 当前窗 exact-strike 总机会为 0 | 原样继续 |
| `CENSORED_CONTINUE` | 有机会但总数 `<100`、任一动作 `<50`、hit-conditioned capture 不足，或 active qdot 没出现 excess | 原样继续 |
| `MEASUREMENT_INVALID` | schema、身份、动作集合、计数闭包或仪表合同不成立 | 原样继续并修仪表；不得解释 Reward |
| `DIRECTION_ONLY` | 一个 milestone 的资格分母完整 | 原样继续；只可看方向 |
| `DECISION_ELIGIBLE` | 连续两个合同 milestone 都完整 | 原样继续；只授权外部预注册决策读取，不自动采纳 |

schema 1 固定最少总机会 `100`、每动作 `50`、每动作至少一次 virtual capture 和连续两个 milestone。
一次 capture 只证明 net/landing/legal-return 通道曾被触发，不证明其比例可靠；真正 winner 仍须匹配对照、
immutable q50 与 vendor MuJoCo 判分。

随附合同只登记新的 `*_eligibility_successor` namespace；它刻意不登记 2026-07-15 已经在跑的旧
namespace，阻止用新格式给旧 EMA 日志补写资格。

## 物理边界

这些 virtual 计数读取 achieved 球拍状态，但接触与回球是解析推演。当前 PhysicalBall Phase A 的球禁用
机器人碰撞，不提供 physical hit/net/landing。classifier 会把
`physical_contact_phase_b_observed=true` 视为不兼容输入，并在 receipt 永久写
`physical_contact_phase_b_measured=false`；Phase B 必须另建物理事件账本，不能给本 receipt 改名。
