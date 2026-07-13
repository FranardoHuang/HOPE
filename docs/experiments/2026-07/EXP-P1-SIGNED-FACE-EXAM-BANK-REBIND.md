# EXP-P1-SIGNED-FACE-EXAM-BANK-REBIND — 旧考试题能否严格迁移到当前物理合同？

- 状态：`complete_exact_e2_data_gate_schedule_blocked`
- 阶段/轴：阶段 1 / signed-face 判卷数据合同
- 集成小目标：让训练题与考试题同时绑定当前源码 family，才允许后续冻结同题 schedule
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：[`E2`](../../DEFINITIONS.md)（真实 371 题 runtime replay 与 report-last 发布）
- 创建日期/最后复核日期：2026-07-13

本文的 [`schema-3 bank`](../../DEFINITIONS.md) 是训练题与考试题分离、并绑定内容 SHA 的第三版题库；
[`no-clobber`](../../DEFINITIONS.md) 表示目标目录一旦存在就拒绝覆盖。

## 问题与假设

旧考试题库与已经迁移成功的训练题库来自同一 source family，但都绑定旧
`virtual_ball.py`。当前 commit `882fea4` 相对旧 commit 只增加一个未被题目生成与物理回放路径调用的
`signed_face_hemisphere` helper。问题是：能否用与 train-v2 完全相同的严格证据，把旧 exam bank
只改四个 metadata leaf，保持全部题目数组和旧/新物理行为逐字节相同，并得到与新 train bank 相同的
目标 physics/source-family SHA？

假设被下列任一结果证伪：Git/AST 不是唯一加法 helper；loader/generator 改过；输入 exam 的 path、
bytes、SHA、split、题数或旧 family 不匹配；任一非 metadata 数组变化或非 finite；任一题 old/new
contact/flight 输出不同；landing/net 门失败；目标 runtime 不能以 `allow_legacy=false` 加载；输出目录
已存在；或 completion report 不能最后独占写入。

## 冻结合同

| 字段 | 冻结值 |
| --- | --- |
| base → target commit | `6d93bcb16c422a2f42748c2dc99432559653480b` → `882fea4285f0cf9a97ba79d79ae8af31d26ea1ed` |
| 旧 exam path | `/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/s1_v4rg_runtime_order_schema3_exam.npz` |
| 旧 exam receipt | `63,968` bytes；SHA `d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096` |
| 旧 split / 题数 | `exam`；正手 `183`，反手 `188` |
| 旧 physics / family | `70242d798f5b97e1405df7dedfd22a5f81421c9c03127e71c254982236cfad35` / `b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5` |
| 目标 physics / family | `09dfe8999c54e36b258fe54b5ec3da5d9816ff3be3675963b919371d7f4afb95` / `9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db` |
| 动作合同 | 正手 motion SHA `f2cb2d9f...141687`、141 帧、strike phase `0.471`；反手 `17225533...7534`、134 帧、`0.338` |
| exam manifest | [`phase1_signed_face_exam_bank_rebind_prereg_20260713.json`](../../../configs/phase1_signed_face_exam_bank_rebind_prereg_20260713.json)，SHA `2153553abe105ace0ae8a90c174198e57b141379d9b3bfc76bdee8d52af7616a` |
| consumer | [`rebind_stage1_question_bank_physics_contract.py`](../../../scripts/rebind_stage1_question_bank_physics_contract.py)，SHA `cf8f6353a6b2a8d90aa7cbb960d5bdb9e681fb174458e088a9341ccda5b8e968` |
| 独立输出 | `.../assets/schema3_exam_bank_rebind_v1/`；bank `s1_v4rg_runtime_order_schema3_exam_882fea4_rebound.npz`；report `rebind_report.json` |

原 train-v2 manifest 保持逐字节不变，SHA 仍为
`5b22a6dd3c41ba1abd44e631e408ed73ada2ac66fc7ff86dc62d48f69ff2ad29`。generalized consumer 通过
manifest ID 选择两个封闭 profile：历史 train-v2 仍只接受原字段和原输出合同；exam-v1 另外强制输入
bytes 与旧 family。不能把任意 bank、split、计数或输出路径塞进同一个 consumer。

## 实验差异与决定规则

- 对照：已经完成 E2 runtime 迁移的 train-v2，题目数组不重算。
- 改变的变量：只把输入角色从 train 换成独立 exam，并把 split/题数/输入 receipt/输出目录冻结成 exam 值。
- 其余固定项：base/target Git、七文件物理合同、唯一 helper、loader/generator、动作、四个 metadata leaf、
  target runtime 与行为 replay 规则完全相同。
- 通过规则：`validate` no-write 全绿；随后独立 `run` 发布 bank 与 report-last；report 必须绑定 manifest、
  consumer、source proof、371 题两侧 replay、输出 bank SHA 与目标 family。
- 停止/无效规则：任何输入、源码、数组、行为、runtime 或 publication 不符即 fail closed；不得开 legacy、
  不得删除旧目录原名重跑、不得用旧 schedule 或启动 judge。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | 输入 | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 双 profile consumer 单元/攻击测试（`host_exam_rebind_e1`） | completed | tracked synthetic banks + real Git pair | `18 passed` | 无 runtime artifact | 证明 profile 分流、train-v2 SHA 不变、exam mutation/bytes/no-clobber；只是 E1 |
| exam no-write 前置（`exam_bank_rebind_v1_validate`） | completed | 冻结旧 exam + clean exact target repo | `validated_no_writes`，24 个非 metadata 数组 finite | 无写入 | Pod1 目标 runtime 通过，随后才运行发布 |
| exam 独立发布（`exam_bank_rebind_v1_run`） | completed | validate 同一输入 | 371/371 old/new output bytes equal；landing/net 全过 | bank `60e1a7ad...d1ca`；report `dd4332ed...ad0` | E2 数据门通过；不含 schedule/judge/行为成绩 |

## 与判卷 schedule 的边界

metadata 变化会让新 bank 文件 SHA 改变，而 `atomic_bank_question_id` 把 bank SHA 纳入每道题 ID；所以
旧 immutable schedule 即使 bank row 相同也不能复用。exam rebind E2 已成功；新的
[signed-face K100 source gate](EXP-P1-SIGNED-FACE-EXAM-PAPER.md) 已冻结 no-clobber materializer、raw-A/
physical-B signed contract、每侧 50 与 all-attempt 分母，但真实 private bank consume 尚未运行。当前仍没有
materialized schedule/activation，也没有 judge、训练、MuJoCo/Isaac 或 Gate3 结果。

## 决定

- 决定：`adopt_exact_rebound_exam_bank_as_data_input_only`
- 理由：Pod1 target runtime 以 `allow_legacy=false` 发布 63,643-byte bank SHA
  `60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca`；18,795-byte report SHA
  `dd4332edb47f1fb1f4d51ca00ceed612dbcadf9e395eb536c9b73bef9de69ad0`、content SHA
  `7bdf4d6c...a19d4`。24 个非 metadata 数组未变，正/反手 `183/188` 题 old/new replay bytes 相同，
  landing/net 全过。机器账本见
  [`phase1_signed_face_exam_bank_rebind_results_20260714.json`](../../../configs/phase1_signed_face_exam_bank_rebind_results_20260714.json)。
- 是否已纳入当前 setting：`data input adopted; training/paper setting unchanged`
- 局限/下一个 gate：按新 [K100 运行手册](../../operations/run_phase1_signed_face_exam_k100.md)消费 exact
  private bank 并归档 schedule/activation receipt；source/static 通过不能替代这次 runtime consume。此前
  signed-face L2/judge/formal score 仍阻断，G06 保持 `Partial`。
