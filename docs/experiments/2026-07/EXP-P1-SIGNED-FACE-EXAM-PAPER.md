# EXP-P1-SIGNED-FACE-EXAM-PAPER — 新 signed-face 题库能否冻结成不混旧卷的 K100？

- 状态：`paper_materialized_not_started`
- 阶段/轴：阶段 1 / signed-face 独立判卷纸
- 集成小目标：从 exact rebound exam bank 生成一份每侧 50 题、全出手计分且不能复用旧题序的不可变考卷
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：[`E2`](../../DEFINITIONS.md)（source/static/攻击负测 + exact private-bank runtime consume；
  尚无 checkpoint/judge 行为）
- 创建日期/最后复核日期：2026-07-14

本文的 [`K100`](../../DEFINITIONS.md#q50-and-k100) 是正手和反手各 50 次的同一份有限考卷；
[`raw A / physical B`](../../DEFINITIONS.md) 分别表示球拍安装坐标的 `+Y` 法向和真正朝向对手的选中胶面。

## 问题与假设

E2 rebind 已发布 exact exam bank SHA
`60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca`，但 bank SHA 已改变每道
`bank-exam-question-v1` 原子 question ID；所以旧 K100 即使 bank row 数值相同，也不是同一份 exact
纸。假设是：复用现有 `bank_exam_schedule.py`，能从新 bank bytes 重新导出确定性、每侧 50、无放回、
不 wrap 的纸，并在确认 raw-A 目标按 `[+1,-1]` 映射到 opponent-facing physical-B 后，最后发布一份
只表示“paper materialized”的 activation。

任一情况证伪 source gate：bank path/bytes/SHA/physics/family/`183+188` 题不符；loader 使用 legacy；
任一目标法向非 finite/非单位或 physical-B `x<=1e-6`；question ID 重复；任一侧少于 50；schedule 不是
seed `0`、hold `[0,100]`、严格 round-robin、100 个唯一无放回题；任一旧 schedule receipt 被接受；
partial 输出能被覆盖/续写；activation 不是最后写入；或 activation 开启 trainer、judge、L2、第二 seed、
晋级、部署或真机权限。

## 冻结合同

| 项 | 值 |
| --- | --- |
| exact bank | `63,643` bytes；SHA `60e1a7ad...d1ca`；schema 3 `exam`；正/反手 `183/188` |
| physics / source family | `09dfe899...afb95` / `9603a178...a9db` |
| 拍面身份 | bank target=`mount_plusY_A`；external=`physical_striking_face_B`；clip order `forehand,backhand`；sign `[+1,-1]`；unsigned/oriented-plane fallback 禁止 |
| 原子题号 | 现有 `bank-exam-question-v1`，包含 exact bank SHA、clip、bank row 和完整来球/答案向量 |
| 抽题 | 现有 schema-v3 deterministic hash order；schedule seed `0`；每侧无放回 `50`；严格正反手轮转 |
| hold | `[0,100]`，`stand-policy-actions-then-raw-frame0-v1` |
| 分母 | aggregate `100`，正手 `50`，反手 `50`；missing/invalid/reset 都算失败，不能从分母删除 |
| 输出 | `.../papers/signed_face_exam_k100_v1/`；root 必须不存在；schedule no-replace；activation 最后写 |
| manifest | [`phase1_signed_face_exam_k100_activation_prereg_20260714.json`](../../../configs/phase1_signed_face_exam_k100_activation_prereg_20260714.json)，SHA `e401305d4564def80677e6d881ef4afabde01d96ea7ea6aa08224d86835de556` |
| consumer | [`materialize_phase1_signed_face_exam_k100.py`](../../../scripts/materialize_phase1_signed_face_exam_k100.py)，SHA `4e094bbebe525fb9cd756c3fa6eebe7436c72f94aba2a12ecd136f612761ac6e` |

旧 paper 的 file/semantic/question-order SHA
`66e89986...71cb3` / `7dc6af82...dff3e` / `b87e81a3...21f91` 被登记为明确禁用 receipt；consumer
不接受一个外部 schedule 作为输入，而是从 exact rebound bank 重建所有 question ID 后调用现有 schedule
module。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | 输入 | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| source/static 与攻击负测（`host_signed_exam_k100_source_v1`） | completed | tracked manifest、consumer、现有 loader/schedule/scorer | focused `14 passed`；latest-main root `747 passed, 10 skipped`；`py_compile`；`static-validate` rc0 | 无 runtime 产物 | E1；覆盖 manifest mutation、旧 bank schedule、unsigned activation、重复题、单侧不足和 partial no-reuse |
| exact private bank 消费（`signed_exam_k100_consume_v1`） | completed | ignored rebound bank `60e1a7ad...d1ca`；clean detached source `748b6d5` | `static-validate` 与单次 `consume` rc0；[runtime receipt](../../../configs/phase1_signed_face_exam_k100_runtime_receipt_20260714.json) | schedule file/semantic/order SHA `f2777dcd...1ca` / `3ca4bdba...3365` / `09f778f2...bd0`；activation file/content SHA `e0125b0e...bb4` / `533beb03...3d8` | E2 paper materialization；100 unique、50/侧、activation-last；全部执行/判卷/晋级权限仍 false |

## Publication 与授权边界

`consume` 先独占创建 output root，再以 `O_EXCL` 写 canonical schedule；它随即从落盘 bytes 重新加载，
逐题对 exact bank question ID、每侧 50、唯一性、hold/seed/no-wrap 和三个旧 receipt。只有全部通过后才最后
以 `O_EXCL` 写 activation。若中间失败，保留 partial root，activation 不存在；同一路径重跑必拒绝，修复
必须发布 v2。

activation 只绑定 manifest/consumer/input bank/source closure、signed-face audit、schedule file/semantic/order
SHA 和全出手分母。它固定 `auto_start=false`、trainer/judge/L2/第二 seed/stop/promote/formal score/Gate3/
deployment/hardware 全 false。即使未来 runtime consume 成功，也必须由后续独立、reviewed execution
合同显式授权 checkpoint/judge；本 source gate 不会自动开卡。

## 决定

- 决定：`adopt_materialized_paper_execution_blocked`
- 理由：Pod1 的 clean detached `748b6d5` source 已消费 exact bank，物化出 100 个唯一题、正反手各 50
  的不可变 schedule，并在落盘复核后最后写出 paper-only activation。新 receipt 已绑定 schedule 的
  file/semantic/question-order SHA 与 activation 的 file/content SHA；activation 明确保持 trainer、judge、
  L2、第二 seed、停止/晋级、formal score、Gate3、部署和真机全 false。
- 是否纳入当前 setting：否。训练、判卷和采用配置均未改变；`NOW.md` 只同步 paper 已物化与仍阻断的
  当前运行态，不把它写成采用配置或行为成绩。
- 下一步：另立 L2/judge execution contract，逐项绑定 exact checkpoint↔相邻 hard contract/lineage、
  evaluator source/runtime/MJCF/plant 和本 schedule/activation；该合同 review、进 main 并独立激活前，不得
  启动 judge 或 L2，也不得购买第二 seed。
