# EXP-P1-FACE-PLANT-SCALEOUT — 拍面语义 × plant 广度矩阵

- 状态：`completed/rejected`（全部 16 条 fresh 训练臂已保留证据并按负责人运营决定停止）
- 阶段/轴：阶段 1 固定点；拍面语义 × 关节摩擦执行配方 × seed
- 集成小目标：先找出能跨 seed、跨 milestone 保持正反手单拍方向的配方，再进入可信 q50 与最终
  厂商 MuJoCo `Gate3/Gate3B`
- 人类负责人：franco
- 执行者：Codex
- 复核/决策负责人：franco
- 最高证据等级：[E4](../../DEFINITIONS.md#证据和文档术语)（q10/q50 均为 Python BankExam
  解析诊断，不是 physical return 或 Gate3）
- 创建日期：2026-07-11
- 最后复核日期：2026-07-13

共享缩写按 [术语与人话对照](../../DEFINITIONS.md) 解释。`SZ` 是共用拍面语义加零关节摩擦的
正式目标格；`SP/LZ/LP` 是拍面或历史非零摩擦对照，只作诊断。

## 问题与假设

24 臂 Phase-1 池中的 16 条 fresh 实验臂用四个 seed 覆盖 `SZ/SP/LZ/LP`。问题是：哪一格能在
同一方向卷上同时保住正手与反手，并值得消耗后续完整决策卷与长训算力？

本记录同时保存 2026-07-13 的一次**算力运营决定**：人类负责人明确授权把已经显示持续塌陷的
实验臂停止并换入更高优先级工作。这个决定不是实验发射前冻结的统计停止规则，不能回写成
“q10 正式判死”，也不能改变既有 q50 合同中的 `whole_arm_stop_allowed=false`。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练 checkout | clean `6d93bcb16c422a2f42748c2dc99432559653480b`；有任一 trainer 存活时不修改 |
| eval checkout | clean `46a0ce24524fdb843e55fe82ba4c045f2adc090f`；live worker/judge/Kit 时不修改 |
| 动作/action 集 | schema-2 `v4rg_runtime_order_v3` 正反手；31 维关节目标 |
| 观测合同 | schema-3、179 维；fresh lineage 必须为 `1` |
| 训练规模 | 每臂 4096 env；原计划 17000 次迭代；同卡实测 4 臂广度并发 |
| q10 | clean K20，正反手各 10 题；只作方向 screen，不得晋级 |
| q50 | clean K100，正反手各 50 题；独立预注册 runner；原合同不授权整臂停止或晋级 |
| 正式格 hard contract | `3a3b3d956e19d47f7e6f0a157159dc96c8f09d8345c436a776c8c7e99c0b9972` |
| `SP` hard contract | `d10099c2693e4ec04ea0d5546d7318bfe7d1d0e830a5199e2a1cb029662a67b9` |
| `LZ` hard contract | `0f65930c27724b837348982ee2739b6df921241445fcfbfa020e6063c400bb06` |
| `LP` hard contract | `b9feb4d511b10942e2042ed50882c7a958a5268b9504b06718c7b92e567d123e` |

## 决策边界

- 原始科学协议不变：q10 只说明方向，不能晋级；model-2000/model-4000 q50 的既有合同也写明
  `whole_arm_stop_allowed=false`。因此下面的停止动作不是“按预注册阈值 reject setting”。
- 2026-07-13 的运营裁剪依据是：负责人看过已有 q50、多个连续 q10 milestone 和每侧分解后，明确
  决定不再为明显持续塌陷的运行购买剩余迭代。它只改变之后的算力分配。
- 停止前必须保留日志和最新 checkpoint，并验证文件名迭代号等于 checkpoint 内嵌迭代号、全部
  `1,762,715` 个浮点元素 finite、schema=3、fresh lineage=1、相邻 hard-contract SHA 完全一致。
- 已停止的 checkpoint 仍保留，可被以后独立、内容绑定的同卷复判；不得因停止而删题、改分母、
  隐去 seed，或把 q10 方向分写成正式质量结论。
- 安全、部署、真机均未授权；本次只管理明确登记的 trainer
  [PGID（进程组编号）](../../DEFINITIONS.md#当前训练与判卷术语)。

## 截至裁剪时的方向曲线

每个数字是同一类 clean q10 中的 20 题解析上台率；它不是 physical return。`—` 表示该格没有
对应 milestone 结果，不按相邻点插值。

| 配方（人话） | Seed | 2k | 4k | 6k | 8k | 10k | 12k | 已知 q50 | 2026-07-13 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 正式共用拍面+零摩擦 `SZ` | 1 | .90 | .50 | .65 | .25 | .00 | .00 | 2k `83/100`；4k `50/100` | 停止 |
| 正式共用拍面+零摩擦 `SZ` | 2 | 1.00 | .90 | .50 | .00 | .00 | .00 | 2k `100/100` | 停止 |
| 正式共用拍面+零摩擦 `SZ` | 3 | 1.00 | 1.00 | .50 | .00 | — | — | 2k `100/100`；4k `98/100`，但正手 signed composite `0/50` | 第二波停止 |
| 正式共用拍面+零摩擦 `SZ` | 4 | .25 | .00 | .00 | .00 | — | — | 2k `20/100` | 停止 |
| 共用拍面+历史非零摩擦诊断 `SP` | 1 | .00 | .00 | .00 | .00 | — | — | 未测 | 停止 |
| 共用拍面+历史非零摩擦诊断 `SP` | 2 | 1.00 | .60 | .55 | .90 | — | — | 未测 | 第二波停止 |
| 共用拍面+历史非零摩擦诊断 `SP` | 3 | 1.00 | 1.00 | .75 | 1.00 | — | — | 未测 | 第二波停止 |
| 共用拍面+历史非零摩擦诊断 `SP` | 4 | .00 | .00 | .00 | .00 | — | — | 未测 | 停止 |
| 旧异号拍面+零摩擦诊断 `LZ` | 1 | .50 | .50 | .50 | .50 | — | — | 未测；正手始终 0、反手始终 1 | 停止 |
| 旧异号拍面+零摩擦诊断 `LZ` | 2 | 1.00 | 1.00 | .50 | .60 | — | — | 未测 | 第二波停止 |
| 旧异号拍面+零摩擦诊断 `LZ` | 3 | 1.00 | .80 | .75 | .55 | — | — | 未测 | 第二波停止 |
| 旧异号拍面+零摩擦诊断 `LZ` | 4 | .90 | .95 | .65 | .95 | — | — | 未测 | 第二波停止 |
| 旧异号拍面+历史非零摩擦诊断 `LP` | 1 | .00 | .00 | .00 | .00 | — | — | 未测 | 停止 |
| 旧异号拍面+历史非零摩擦诊断 `LP` | 2 | .50 | .50 | .50 | .50 | — | — | 未测；正手始终 0、反手始终 1 | 停止 |
| 旧异号拍面+历史非零摩擦诊断 `LP` | 3 | .60 | 1.00 | .65 | 1.00 | — | — | 未测 | 第二波停止 |
| 旧异号拍面+历史非零摩擦诊断 `LP` | 4 | 1.00 | .70 | .95 | 1.00 | — | — | 未测 | 第二波停止 |

seed1/seed2 的 0k/1k formal q10 另有 `0/.50` 与 `0/.50`，这里只从所有 fresh 格都可比较的 2k
列起表。曲线显示强烈的 seed × 拍面/plant 交互与 milestone 回落；在 signed-face 诚实门通过前，
这些数字不能用于选部署 policy。

## 2026-07-13 精确停止记录

| Pod / PGID | 运行（人话 + `run_name`） | 最后保留 checkpoint | checkpoint SHA-256 | 合同/谱系验证 | 信号结果 |
| --- | --- | --- | --- | --- | --- |
| Pod1 / `1311754` | 正式 `SZ` seed1（`phase1_fresh_v3_S1_seed1`） | `model_12100.pt` | `a251219ebb9a34d232ab301db6f2892954bba5e90f726c554ed128524d74311d` | checkpoint↔formal contract SHA exact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod1 / `1347205` | `SP` seed1（`phase1_fresh_v3_SP_seed1`） | `model_9500.pt` | `c2ce10c88a8c70379fa6eb56a9aa721b4b45f5601826171b96ab8f728006abec` | checkpoint↔`SP` contract SHA exact；evaluation inexact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod1 / `1350527` | `LZ` seed1（`phase1_fresh_v3_LZ_seed1`） | `model_9400.pt` | `8ebe4c5ec041e498e55062639952fee77d26d8c10be00cfea469407ff8f3c345` | checkpoint↔`LZ` contract SHA exact；evaluation inexact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod1 / `1355281` | `LP` seed1（`phase1_fresh_v3_LP_seed1`） | `model_9300.pt` | `a560d302ed85023d7f735e13c1e1650f8ebfd0a7037a8a38f336d10803ad9422` | checkpoint↔`LP` contract SHA exact；evaluation inexact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod2 / `162836` | 正式 `SZ` seed2（`phase1_fresh_v3_S1_seed2`） | `model_13000.pt` | `4e080c46379bccda71ffbf709eac7485a4e3ff72fd6fc8658b8bd4ad17837e6f` | checkpoint↔formal contract SHA exact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod2 / `178741` | `SP` seed4（`phase1_fresh_v3_SP_seed4`） | `model_12900.pt` | `c39b6f09ac9013c4744bc72ec907dec02005f275fe644d4b16c33ac4f4c9636d` | checkpoint↔`SP` contract SHA exact；evaluation inexact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod2 / `179322` | 正式 `SZ` seed4（`phase1_fresh_v3_SZ_seed4`） | `model_12900.pt` | `46edec164d2ef058c0255759c6f8580d023c98a8caef183f249468f55f781bb6` | checkpoint↔formal contract SHA exact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |
| Pod2 / `182929` | `LP` seed2（`phase1_fresh_v3_LP_seed2`） | `model_9500.pt` | `f02dd6a6eac83a3846b8faede9cef2877f7d5f96e4509b70bf09e9c38105a237` | checkpoint↔`LP` contract SHA exact；evaluation inexact；schema3；lineage1；finite | TERM 未退出；精确 PGID KILL 后消失 |

对每臂均先核对文件名迭代号等于内嵌迭代号、`1,762,715` 个浮点元素且 `nonfinite=0`、相邻
`params/training_contract.json` SHA 完全相等。TERM 之后再次确认没有 live child 或 Kit-lock holder，
才向该臂已登记 PGID 发送 KILL；没有使用 `pkill/killall` 或模式匹配信号。之后复核被停 PGID 已
消失，未向剩余接受臂、worker、judge 或任何真机进程发信号。

第一波停止后曾仍运行 8 臂：Pod1 的 `SZ seed3 / SP seed3 / LP seed3 / LZ seed3`，
以及 Pod2 的 `SP seed2 / LZ seed2 / LZ seed4 / LP seed4`。其中 Pod1 `LZ seed3` 只承认 malloc
失败后成功重试的 PGID `1354525`；旧失败尝试不参与结果。下节记录它们的第二波
证据收口；现在 16 臂均已停止。

## 2026-07-13 第二波：符号失真后停止剩余 8 臂

model-4000 K100 完成后，对剩余臂最近三个已有 K20 milestone 做了不运行新 judge 的
横向取证。共 24/24 个格子的正手 raw-A signed composite 都为 `0`，法向误差为
`164.4°–175.2°`；但旧 parsed return 可达 `1.0`。同期反手法向通常为 `2.9°–12.3°`，
composite 为 `.7–1.0`。`SZ/SP` 的 shared face 与 `LZ/LP` 的 legacy face 都同样反号，
因此继续长训不能分离 face/plant 效应；必须先修复训练目标/Reward 与 scorer 的共同符号链。

| Pod / PGID | 运行 | 最后 checkpoint | checkpoint SHA-256 | 验证 |
| --- | --- | --- | --- | --- |
| Pod1 / `1348951` | `SZ seed3` | `model_13800.pt` | `478efa8d163ec53dbade328c5de18947f6c068df78cbadff8e46a29844bdc9e6` | formal contract exact; schema3; lineage1; finite |
| Pod1 / `1349699` | `SP seed3` | `model_13800.pt` | `611fecc2087a52cb2b7602d6932d10fb3becf9cd0389bfcf6507ccea8d08d5fb` | `SP` contract exact; schema3; lineage1; finite |
| Pod1 / `1353018` | `LP seed3` | `model_13800.pt` | `cbb157bce2aba4df816e58ed4126b6ee744fc776039ea2466a8849f38fb48bcd` | `LP` contract exact; schema3; lineage1; finite |
| Pod1 / `1354525` | `LZ seed3` accepted retry | `model_13600.pt` | `fe5b06cf70c1f2ee2923d75dcd7da06df841ba487b9247e0798a68b7121fec53` | `LZ` contract exact; schema3; lineage1; finite |
| Pod2 / `177630` | `SP seed2` | `model_10400.pt` | `6d205edbf4cab838ccbdae5cbd353102c148fd1c3fda54eb561f81e1917e2f06` | `SP` contract exact; schema3; lineage1; finite |
| Pod2 / `179908` | `LZ seed2` | `model_10400.pt` | `2852ac7ed3394871cddbf5044e0bf861e0c3ed504de6737efe5858410c2c0dad` | `LZ` contract exact; schema3; lineage1; finite |
| Pod2 / `181685` | `LZ seed4` | `model_14300.pt` | `bc872a4e356015520ed8c126d4785dafe9f8843b44ae4e7d13c0eac6a33a79d2` | `LZ` contract exact; schema3; lineage1; finite |
| Pod2 / `182286` | `LP seed4` | `model_14400.pt` | `ee9539f1a8a711ac557d08cdcbac67ce067a4315637515ef74feee2978c8b2be` | `LP` contract exact; schema3; lineage1; finite |

两份 no-clobber 停臂前审计的 file SHA 为 `aca8e4f4...f5a3` / `773940ac...7b86`。每臂都验证
文件名迭代=内嵌迭代、76 个 tensor、`1,762,715` 个浮点值且 nonfinite=0、schema3、
fresh lineage1、内嵌合同 SHA=相邻合同 SHA，并保留 launch/log SHA。完整日志的
NaN/Traceback/OOM/malloc/bad_alloc/Killed/segfault 均为 0。TERM 后 trainer 仍存活；确认
进程组只有 trainer 与它的 git helper，且 Kit lock 无 holder 后，只对上表精确 PGID 发 KILL。
最终八组均消失，两 Pod 三卡均回到 0 MiB/0% GPU，train/eval 仍 clean exact
`6d93bcb` / `46a0ce2`。没有 broad signal、新 judge、checkout 修改或真机命令。

与已停 trainer 对应、且无 child/Kit holder 的四个等待型 fresh curve worker 也随后按精确
PGID `1432280/1432304` 与 `200706/200730` TERM 退出；它们没有产生新分或信号传播。

## 决定

- 科学决定：`rejected_as_baseline_selector`。formal `SZ` 在 2k/4k 都不具 seed 稳定性；
  24/24 最近诊断格的正手 signed composite 为 0，所以这个旧 scorer 下的 face×plant 矩阵
  不能选 baseline。这不是用 q10 阈值正式 reject 单臂，而是 q50 失败+符号尺失真后的
  家族级结论。
- Runtime 决定：两波合计停止全部 16 臂并保留 checkpoint/log/contract/审计；释放槽位只分配给
  [NOW 唯一队列](../../NOW.md#统一工作队列唯一优先级账本)中已经通过离线门的工作。
- 是否已纳入当前 setting：`no`。现役 setting 没有改变；只是部分 run 不再继续购买迭代。
- 下一个 gate：K100 已完成；现在先修正训练目标/Reward/scorer 的 signed-face 符号链，
  跑 `n/-n` 负控与同卷复判，再启动新训练。不把两波运营停止反推成旧 q10 预注册结论。

## 复现与证据

- 原始矩阵：[phase1_scaleout_matrix_20260711.json](../../../configs/phase1_scaleout_matrix_20260711.json)
- q10 manifest 与结果 state 保存在两 Pod 的
  `/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/`，由 pinned eval checkout
  `46a0ce24524fdb843e55fe82ba4c045f2adc090f` 生成。
- 本次 stop 的原始 launch/log/checkpoint 仍保存在 Pod；仓库当前没有单独的 no-clobber stop
  transaction sidecar。所以上表是经现场复核的运营台账，不冒充一份已内容寻址的正式停止结果。
- formal model-2000 q50：
  `configs/phase1_fresh_SZ_model2000_seed_stability_q50_pod{1,2}_result_20260711.json`。
- model-4000 后续卷：[稳定性实验](EXP-P1-FRESH-SZ-STABILITY.md)与
  [操作文档](../../operations/run_phase1_fresh_sz_model4000_seed_stability_q50.md)。
- 精确信号与 checkpoint 验证流程：[共享 RunPod 操作](../../operations/run_on_runpod.md#已登记-phase-1-实验臂的算力释放)。
- [Pod1 第二波停臂前审计](../../../configs/phase1_remaining_fresh_arms_pre_stop_audit_pod1_20260713.json)
  file SHA `aca8e4f445026d7f7a36619233df48e01c346f8d22bc75d2aa092e6b67e0f5a3`。
- [Pod2 第二波停臂前审计](../../../configs/phase1_remaining_fresh_arms_pre_stop_audit_pod2_20260713.json)
  file SHA `773940accc8bfb8f9d28a6bbac9e61dc2583db649bffc845e1130ce15efc7b86`。

本记录没有执行真实机器人命令，也不授权任何真机动作。
