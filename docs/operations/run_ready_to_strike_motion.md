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
它只把旧结果升级为 screening evidence；因为历史证书没有完整 argv、transitive source 和 MJCF closure，
`physics_replay_exact/source_closure_exact/mjcf_closure_exact` 仍必须是 `false`，不能冒充动力学重放或部署通过。

## 下一步不是直接训练

每条候选依次执行：

1. 使用 exact vendor MJCF、URDF 和 runtime body order 做 production FK 重建；
2. `topp_mintime.py --objective runup --body-mode fk`，要求 run-up `<=0.5 s`，并保持触球行、拍速和拍面；
3. schema-2 L0、vendor L1 自碰/自打、整轨桌网余隙 `>=5 mm`；
4. CoP、摩擦锥、力矩和连续平衡动力学；
5. 用该候选 motion/certificate SHA 新物化 0.5 秒 K100；Isaac 只作 inexact 诊断，最终跑 vendor MuJoCo。

任一上游门失败都停止该候选。不能用 Reward 抵消自碰、桌网、非 finite 或动力学失败。
