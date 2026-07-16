# 从第 0 帧静止准备态生成短击球路径

本操作只生成 host-only [schema-2 motion](../DEFINITIONS.md) 候选。它不启动训练、simulator、部署或真机，
也不把 quintic 路径冒充 [`TOPP`](../DEFINITIONS.md) / 动力学证书。完整实验边界见
[EXP-MOTION-READY-TO-STRIKE-0P5](../experiments/2026-07/EXP-MOTION-READY-TO-STRIKE-0P5.md)。

## 单候选生成

`--ready-source` 是提供共同准备姿态的 schema-2 动作；工具只取其第 0 帧姿态并把速度显式置零。
`--contact-frame` 和 `--join-frame` 都是源动作行号；`--blend-intervals` 是从最后一个静止准备行到
源 join 行的 50 Hz 区间数。下面命令只是形状示例，真实路径和帧号必须来自预注册动作记录：

```bash
python3 scripts/build_ready_to_strike_motion.py \
  --source /ABS/SOURCE.schema2.npz \
  --ready-source /ABS/SHARED_READY.schema2.npz \
  --ready-frame 0 \
  --contact-frame 66 \
  --join-frame 60 \
  --hold-frames 4 \
  --blend-intervals 16 \
  --output-npz /ABS/EMPTY/candidate.npz \
  --output-contract /ABS/EMPTY/candidate.contract.json
```

对于 50 Hz、`hold_frames=4` 的 0.5 秒候选，必须满足：

```text
blend_intervals + (contact_frame - join_frame) = 22
output_contact_frame = 25
```

join 还必须至少早于受保护的触球前 `0.1 s` 一行。不要看到结果后随意调一个 join；应先冻结一组
`(contact_frame - join_frame, blend_intervals)` 配对，再全量报告。每个输出 JSON 必须确认：

- `frame0_shared_ready_pose_bitwise_equal=true`；
- `initial_zero_velocity_frames>=3`；
- `protected_window_bitwise_equal=true`；
- `ready_source_velocity_channels_ignored=true`；
- `training_authorized=false`；
- NPZ SHA 与 JSON 内绑定相同。

生成器使用同目录双 hard-link 的 no-clobber 发布，但不是断电/crash 原子事务。若只出现 NPZ 或 JSON
之一，或进程被强杀，保留目录作为失败证据；不得删除后用同一个 attempt 自动重放。输出目录只能是
可信私有 real directory，不能含 symlink。

输入字段集必须精确。原生 schema-2 可以只有六个时序通道、`fps` 和四项 active kinematics metadata；
若存在历史迁移溯源，则以下三项必须同时存在且由 canonical v2 writer 产生：

```text
kinematics_migration_source_sha256 = lowercase 64-hex unicode scalar
kinematics_migration_source_point  = link_origin | center_of_mass unicode scalar
kinematics_migration_tool          = migrate_motion_kinematics.py/v2 unicode scalar
```

不要删除正式资产的三元组来绕过检查。生成器把击球 source 三项逐位复制到输出，并只在 JSON 中另行记录
ready-source；它不会重读旧 legacy ancestor bytes，所以这不是 ancestor SHA 的重新认证。三项残缺、数组而非
scalar、bytes/object/integer、未知额外字段或非法值都会在发布前拒绝。

2026-07-17 的首次 Pod2 namespace `attempt_2137b82b` 使用旧 v1 source gate，正反手均因错误拒绝上述
完整三元组而停止；没有候选、TOPP 或 GPU 行为。该目录只作证据，不得删除或在同 namespace 重发。

第二次 namespace `attempt_2_66f93559` 已证明生成合同可消费真实资产，但 production-FK TOPP 只找到
正手 `0.64 s`、反手 `0.94 s` 可行上界。后续不得手改 join 猜点；使用版本化
[`configs/ready_to_strike_join_ladder_20260717.yaml`](../../configs/ready_to_strike_join_ladder_20260717.yaml)
派生 `join_frame=contact_frame-delta`、`blend_intervals=22-delta`。先跑 `delta=6/17` 的 ready×side
端点因子阵，再按冻结规则跑中点 `12` 与按需细化 `9/14`；已跑的 shared-ready `delta=6` 不重放。
新格逐个报告，不因某一格失败自动 retry。

## Stage-1 历史结果的只读认证

端点 Stage-1 的原始 runner 没有把自身源码写进 summary，所以原始数值不能直接解锁下一层。不要重跑
六格；用 tracked historical attestor 对既有树做一次只读重建。默认命令只 dry-run，不写 receipt：

```bash
python3 scripts/attest_ready_to_strike_ladder_stage1.py \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage1_8d74025e \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml
```

dry-run 成功后，才允许在同一组未变输入上发布唯一的 no-clobber receipt：

```bash
python3 scripts/attest_ready_to_strike_ladder_stage1.py \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage1_8d74025e \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --execute \
  --confirm ATTEST_READY_TO_STRIKE_STAGE1_ONCE
```

执行源码、queue 和 Stage-1 树必须位于同一台 host 的绝对路径；如果 Pod 上没有该 tracked source，先以
O_EXCL 写入 Stage-1 根目录并核 source SHA，再从该副本运行。receipt 固定为根目录下
`stage1_historical_attestation.json`，存在即拒绝再次执行。认证会重验候选 schema-2、generator contract、
TOPP input/output、生产 FK body order、直接工具依赖、预算、触球行、拍速、拍面、首帧零速及可行时间上界。
queue 中 `checkout_commit` 与 `generator_source_commit` 是两个独立 source root：前者必须提供绑定的
TOPP/MJCF/URDF/body-order，后者由 Stage-1 根目录中实际执行的 immutable generator copy 对 SHA；不要
错误要求旧训练 checkout 包含后置生成器，也不要拿当前 main 文件替换历史副本。
schema-2 source/candidate 的 `joint_vel` 必须按 generator 的 float32 输入梯度逐位重算；TOPP FK output
必须按其 float64 工作区梯度再转 float32。两条 producer 合同不同，审计时不可混用或放宽为任选其一。
本 Stage-1 历史族的 TOPP budget envelope scale 固定为冻结 v3 工具默认 `1.5`；receipt 必须精确核该值，
不得误写 `1.0`，也不得只检查为正数。
它只把旧结果升级为 screening evidence；因为历史证书没有完整 argv、transitive source 和 MJCF closure，
`physics_replay_exact/source_closure_exact/mjcf_closure_exact` 仍必须是 `false`，不能冒充动力学重放或部署通过。

## Stage-2 四个中点的一次性执行（当前 v5）

Stage-1 receipt `7cf1c7c9…c377f` 已成功发布后，四个 `delta=12` 中点只能由 tracked runner 消费一次。
activation 精确绑定 runner SHA、Stage-1 receipt SHA 和唯一结果目录；换目录重复执行会在创建任何 namespace
前 fail closed。先从包含 runner 的 clean main source 做 dry-run：

```bash
python3 scripts/run_ready_to_strike_join_ladder_stage2.py \
  --activation configs/ready_to_strike_join_ladder_stage2_activation_v5_20260717.json \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage2_d12_v5_exact_log_bytes
```

dry-run 必须报告四格且不创建结果目录。确认 receipt、runner、queue、旧 runtime 与两份动作资产 SHA 全部
一致后，才可用同一组输入执行：

```bash
python3 scripts/run_ready_to_strike_join_ladder_stage2.py \
  --activation configs/ready_to_strike_join_ladder_stage2_activation_v5_20260717.json \
  --queue configs/ready_to_strike_join_ladder_20260717.yaml \
  --root /workspace/codexschema/ready_to_strike_0p5_20260717/join_ladder_stage2_d12_v5_exact_log_bytes \
  --execute \
  --confirm RUN_READY_TO_STRIKE_STAGE2_ONCE
```

v5 不再重跑 generator：它先完整消费 v2 的 terminal summary、runner、activation、V1 prior、四份
candidate/contract 和四份 missing-mesh log，逐字节复验 lineage 后把同一 candidate 复制进新 namespace。
因此本轮唯一改变是补齐 MJCF 外部资产闭包。runner 解析原 XML，拒绝 DTD/entity/include、路径逃逸、
重复引用和非 regular/symlink 文件；然后从冻结 commit 的 Git object 读取 `1 XML + 74 mesh`，核
model-tree OID、每个 blob OID、SHA 和大小，以原相对目录 O_EXCL 快照。闭包固定为 `75 files / 14,127,373
bytes / e0381752…b962de`，worktree fallback 不构成授权。TOPP/URDF/body-order 与两份预算动作也只读冻结。
四个 TOPP 可并行，但每个 CPU child 的 reviewed timeout 为 3600 秒；任一格
失败都发布 terminal summary、全批不重试。`runup_s` 必须同时等于 output contact-frame/fps 与 timing bound，
budget scale 必须为冻结默认 `1.5`。成功只说明 Stage-2 screening 执行完整；只有 `<=0.5 s` 且 hard gate
全过的格才能进入 L0/L1，仍无训练、部署或真机权限。

旧 v1 namespace `join_ladder_stage2_d12_8d74025e` 已永久消费且不得再运行：四格 generator 均 rc0，
但 validator 把 generator 的 float32 producer-gradient 错按 TOPP 的 float64 workspace-gradient 重算，故
全部在 TOPP 前 fail closed。summary SHA=`f92e6b8b…63c0e`。v2 不改变动作、join、预算或 acceptance；只把
candidate/TOPP 两条已冻结的 producer 合同分开验证，并在新 activation 中精确绑定旧 failure summary、
旧 runner/activation SHA 和 `automatic_retry=false`。旧 summary 缺失或改变时，v2 在创建新 namespace 前拒绝。
v2 namespace `join_ladder_stage2_d12_v2_float32_producer` 也已永久消费：四份 candidate 与 v1 逐字节一致，
但四个 TOPP 都因隔离 MJCF 旁缺少其引用的 STL 而 rc1；summary SHA=`6910db28…f1476`，四份 log 已逐一
绑定。v3 dry-run 在创建结果 namespace 前又正确拒绝：consumer 错把 v1 contract SHA 配给 v2 candidate；
stderr SHA=`c58baf2d…2e16`，execute/TOPP 均未启动，固定 v3 结果目录仍不存在且不得用新源码复用该 activation。
v4 使用新 source、activation 和 namespace，绑定 v2 summary 中实际四份 contract SHA；除此之外只修
source-closure，不改 candidate、join、hold、预算、acceptance 或 TOPP 算法。
v4 dry-run 随后又在结果 root 前暴露重复量尺：四份 log 已按完整 SHA 绑定，consumer 仍额外猜测日志必须
同时含 `.stl` 与英文 `no such file or directory`；真实不可变 log 格式不同，execute/TOPP 同样未启动。
v5 使用新 activation/namespace，删除这条脆弱文本解释，只保留四份 exact log SHA；其他输入与配方不变。

## 下一步不是直接训练

每条候选依次执行：

1. 使用 exact vendor MJCF、URDF 和 runtime body order 做 production FK 重建；
2. `topp_mintime.py --objective runup --body-mode fk`，要求 run-up `<=0.5 s`，并保持触球行、拍速和拍面；
3. schema-2 L0、vendor L1 自碰/自打、整轨桌网余隙 `>=5 mm`；
4. CoP、摩擦锥、力矩和连续平衡动力学；
5. 用该候选 motion/certificate SHA 新物化 0.5 秒 K100；Isaac 只作 inexact 诊断，最终跑 vendor MuJoCo。

任一上游门失败都停止该候选。不能用 Reward 抵消自碰、桌网、非 finite 或动力学失败。
