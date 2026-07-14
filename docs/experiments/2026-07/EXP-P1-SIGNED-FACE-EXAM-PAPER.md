# EXP-P1-SIGNED-FACE-EXAM-PAPER — 新 signed-face 题库能否冻结成不混旧卷的 K100？

- 状态：`paper_materialized_not_started`；checkpoint attestor 只有 source/static gate，尚无 runtime claim
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
| checkpoint 判卷前一次性取证源码门（`host_signed_k100_checkpoint_attestor_v1`，即为一份明确 checkpoint 冻结全执行输入） | source/static completed；runtime not run | generic exact-request schema、materialized paper receipt、evaluator/export closure、MJCF/plant contract | focused `21 passed`；rebase 后仓内 `tests/` 为 `956 passed, 9 skipped`；`py_compile`；`static-validate` rc0 | 新 source manifest + versioned receipt correction；没有 checkpoint evidence/claim | E1 source gate；路径通配/穿越、symlink ancestry、runtime replacement、request TOCTOU、dangling/partial namespace 均 fail closed；不读 Pod、不启动 judge；任一真实候选仍需独立 exact request 和单次 runtime attest |

## Publication 与授权边界

`consume` 先独占创建 output root，再以 `O_EXCL` 写 canonical schedule；它随即从落盘 bytes 重新加载，
逐题对 exact bank question ID、每侧 50、唯一性、hold/seed/no-wrap 和三个旧 receipt。只有全部通过后才最后
以 `O_EXCL` 写 activation。若中间失败，保留 partial root，activation 不存在；同一路径重跑必拒绝，修复
必须发布 v2。

activation 只绑定 manifest/consumer/input bank/source closure、signed-face audit、schedule file/semantic/order
SHA 和全出手分母。它固定 `auto_start=false`、trainer/judge/L2/第二 seed/stop/promote/formal score/Gate3/
deployment/hardware 全 false。即使未来 runtime consume 成功，也必须由后续独立、reviewed execution
合同显式授权 checkpoint/judge；本 source gate 不会自动开卡。

### Runtime receipt 的数值类型更正

原 runtime receipt 文件保持 SHA `c0eca638...2048` 不变；其中摘要字段
`signed_face_contract.mount_normal_sign_per_clip` 误写为 JSON integers `[1,-1]`。actual source manifest、
materializer 和已物化 activation 都要求 exact floats `[1.0,-1.0]`。新的
[`versioned correction pointer`](../../../configs/phase1_signed_face_exam_k100_runtime_receipt_correction_20260714.json)
保留旧值与原 receipt SHA，并把 actual activation file/content SHA 定为字段权威。后续 consumer 不接受
“数值相等”替代类型相等，也不静默改旧收据；必须直接读 actual activation 并重算 content SHA。

### Generic checkpoint attestor source gate

新的 [`checkpoint attestor`](../../DEFINITIONS.md) 不是一个自动挑 checkpoint 的 judge wrapper。每个候选
必须用独立 reviewed request 显式列出 checkpoint path/bytes/SHA、filename/embed iteration、相邻 hard
contract SHA、fresh lineage integer `1` 和 producer-claim file/canonical SHA；禁止 glob、latest 或目录猜测。
同一 request 还绑定 clean source commit/tree、evaluator source closure、两个 Python runtime fingerprint、
MJCF bytes/SHA 与 hard-contract plant semantic SHA。

consumer 对 actual schedule file/semantic/order 和 actual activation file/content SHA 直接复核；activation
里的 raw-A/physical-B sign 必须是 exact floats `[1.0,-1.0]`。所有 no-write 检查通过后，它才在
`.../executions/signed_face_k100_v1/<checkpoint_sha256>/` 独占写 evidence，并最后写 claim；同 checkpoint
不能通过换 request id 或 output 路径重判。claim 仍为 `attested_not_executed_no_decision`，没有 judge、
formal score、stop/promote、第二 seed、Gate3、部署或真机授权。复现见
[操作文档](../../operations/run_phase1_signed_face_k100_checkpoint_attestor.md)。本分支未连接 Pod，未创建真实
request，也未运行 runtime attest。

攻击回归还冻结了使用中 byte identity：checkpoint、hard contract、producer claim、MJCF、schedule、
activation、request 与 manifest 在解析/子进程检查后都必须保持同一 inode/metadata/SHA；中途替换不允许靠
换回原路径蒙混。输出 root 用 checkpoint SHA 唯一派生，dangling symlink 和只写出 evidence 的 partial 都视为
已消费且永久拒绝复用。

## 决定

- 决定：`adopt_materialized_paper_execution_blocked`
- 理由：Pod1 的 clean detached `748b6d5` source 已消费 exact bank，物化出 100 个唯一题、正反手各 50
  的不可变 schedule，并在落盘复核后最后写出 paper-only activation。新 receipt 已绑定 schedule 的
  file/semantic/question-order SHA 与 activation 的 file/content SHA；activation 明确保持 trainer、judge、
  L2、第二 seed、停止/晋级、formal score、Gate3、部署和真机全 false。
- 是否纳入当前 setting：否。训练、判卷和采用配置均未改变；`NOW.md` 只同步 paper 已物化与仍阻断的
  当前运行态，不把它写成采用配置或行为成绩。
- 下一步：checkpoint attestor source gate 合入 main 后，为一个真实候选写独立 exact request 并单次
  runtime attest；随后仍需另一份 reviewed judge runner 消费该 claim。attestor 本身不启动 judge 或 L2，
  不得据此购买第二 seed、停止或晋级。
