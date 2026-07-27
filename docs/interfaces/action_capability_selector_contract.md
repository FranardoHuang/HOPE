# 动作能力工件与 planner selector 合同

Status: Draft / pure-core source candidate。Python 排序核心与 host tests 可以存在，但当前生产 planner、
flat wire 和 C++ runner 仍是二动作 side selector；本文不声称任意 N 动作已经接入运行时。

共享术语见[术语与人话对照](../DEFINITIONS.md)：
[`capability artifact`](../DEFINITIONS.md#capability-artifact)是指定 checkpoint 在冻结留出题域上的
逐动作能力工件；[`OOD`](../DEFINITIONS.md#ood)表示超出该证据支持的分布；
[`LCB`](../DEFINITIONS.md#selector-lcb)是校准成功率下置信界；
[`abstain`](../DEFINITIONS.md#selector-abstain)表示没有动作获准执行。

## 1. 责任边界

对任意一个击球目标，选择链必须是：

```text
planner 为 catalog 中每个动作生成物理候选
  -> 硬安全/可达性
  -> 留出支持与 OOD
  -> 校准成功率 LCB
  -> 只在 delta tie 内看人工优先级
  -> 选择一个动作，或显式 abstain
```

selector 不修复一个不安全的候选，也不从 action family 猜 motion slot。priority 只能在成功率近似
并列时表达战术偏好，不能复活硬失败、低支持、OOD 或低于最低 LCB 的动作。训练 Reward 不能直接
跨动作比较：动作的 imitation 尺度、课程难度和 reference 都不同。用户所说“谁回出的球质更好”
必须先变成同一部署口径的 held-out quality/utility（例如落点误差、出球速度/旋转、恢复成本），
再进入 selector。

## 2. Catalog 与身份

catalog 必须是非空的 dense slot `0..N-1` 有序表。每行包含：

- `action_id`：经规范化、无控制/格式字符的人类可读身份；
- `action_uid`：由 `action_id + family + content_sha256` 派生的稳定正整数；
- `slot`：本 catalog 的临时数组索引；
- `family` 和 exact motion/content SHA-256。

`action_uid=0` 永远保留给 abstain。跨 JSON/double/C++ 传输的 UID 上限为 `2^53-1`。catalog
增删或重排可以改变 slot，却不能改变同一内容动作的 stable UID。

## 3. Capability artifact

一份能力工件只对下列 exact tuple 有效：

```text
catalog SHA
policy/checkpoint SHA
task definition SHA
effective Reward SHA
held-out paper SHA
capability model SHA
calibration SHA
ordered action UID list
```

任一项漂移都拒绝选择，不做“最接近版本”回退。逐动作训练课程给出的 train success 不能直接填入
工件；工件来自冻结、ball-conditioned 的留出卷。

建议首轮每动作 `512` 题：

- `64` 个中心重复；
- position、标量速度、拍面、base 四个单轴各 `96`；
- `64` 个联合边界点。

每题使用 all-attempt 分母。成功同时要求 exact strike、位置误差 `<=7.5 cm`、速度向量或标量误差
`<=0.5 m/s`、拍面误差 `<=15 deg`、base 误差 `<=10 cm`，并在 recovery 结束前没有 table hit 或
physical fall。工件要逐动作/逐轴报告支持数、校准成功率、Wilson 或预注册校准 LCB、p50/p90
误差和 unsafe 率。

## 4. Candidate evidence

对同一 query，每个 catalog action 必须恰有一行 candidate。每行至少绑定：

- query SHA；
- 完整 adapter/physics candidate SHA；
- capability artifact SHA；
- stable action UID；
- `hard_ok` 与具名 hard reason；
- support count、OOD score、成功率 LCB；
- 覆盖上述字段的 evidence SHA。

硬失败先于所有数值证据处理。硬通过但证据缺失、NaN/Inf、范围错误或 receipt 被篡改时，只让该
candidate 失效；不得让 NaN 比较穿透，也不得让一行坏数据把其他动作改成安全。

## 5. 固定选择顺序

1. **Hard safety/admissibility：** 桌网/身体碰撞、动作证书、时序、可达性、场地和任务约束任一失败，
   该动作永久不进入本次排序。
2. **Support/OOD：** `support_count < min_support` 或 `ood_score > max_ood_score` 时拒绝。
3. **Calibrated LCB floor：** `lcb_success < min_lcb_success` 时拒绝，priority 不得复活。
4. **Best-success set：** 取最高 LCB；只把与最高值 exact binary64 差值
   `<= delta_tie` 的 eligible 动作放入并列集合。
5. **Quality/priority：** v1 pure core 只使用独立 profile 的整数 priority；它可以是人工战术偏好，
   也可以由另外一份已校准、同尺度的 held-out quality model 预先产出，但不得直接填训练 Reward。
   并列集合中较小整数 priority 优先；priority 相同时先取较高 LCB，再取较小
   stable action UID 作最终确定性决胜。dense slot 不参与最终 tie-break，避免 catalog reorder
   无故翻转同一组动作的选择。
6. **Abstain：** 没有 eligible 动作时返回唯一空身份
   `(action_uid=0, action_id="", slot=-1)`，并给出 all-hard-reject、low-support、OOD、
   below-min-LCB 或 mixed-no-eligible 原因。

selector profile 的阈值与逐 UID priority 必须覆盖 catalog 全集并内容哈希；运行时从独立 activation
manifest 钉住 expected profile SHA，不能让 caller 同时自报 profile 和“期望 profile”。

## 6. Decision receipt

每次决策输出全部 slot 的 assessment，而不只输出胜者。canonical decision receipt 绑定：

- artifact/profile/query SHA；
- 每个 candidate/evidence SHA；
- 每行 hard/support/OOD/LCB/priority/reason；
- selected stable UID、action ID 和本地 slot，或唯一 abstain identity。

selected 行必须恰有一行且确实 eligible；abstain 时不得残留 eligible/selected 行。直接构造、
JSON reload 和 receipt tamper 都必须重新做语义校验。

## 7. 生产接入尚缺什么

当前生产链仍以正/反手 sign 选两个 clip：

- planner node 不是逐 catalog action 的 trusted candidate producer；
- schema 4 exact22 没有 stable action UID；
- C++ runner、ONNX metadata 和 reference clock 仍硬绑定二动作 sign table；
- candidate SHA 目前由 caller 提供，独立 profile authority 尚未接入 activation。
- 当前 capability artifact 只冻结 model/calibration identity，尚没有生产的 query-conditioned
  quality model；因此 v1 priority 仍是静态 profile，不能声称已经按实际球质实时排序。

任意 N 动作生产接入需另做一个 wire/runner 合同。候选设计是 schema 5 exact23，在 schema 4 的
22 个 double 后以 `[22]` 传 stable `action_uid`，同一 task revision 内冻结 UID；C++ 再用
catalog UID 查本地 slot。该设计尚未 adopted，也未通过 ROS/Jazzy、vendor first tick 或 Gate3，
不得把 Python core test 写成 planner 已完成。
