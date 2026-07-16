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

## 下一步不是直接训练

每条候选依次执行：

1. 使用 exact vendor MJCF、URDF 和 runtime body order 做 production FK 重建；
2. `topp_mintime.py --objective runup --body-mode fk`，要求 run-up `<=0.5 s`，并保持触球行、拍速和拍面；
3. schema-2 L0、vendor L1 自碰/自打、整轨桌网余隙 `>=5 mm`；
4. CoP、摩擦锥、力矩和连续平衡动力学；
5. 用该候选 motion/certificate SHA 新物化 0.5 秒 K100；Isaac 只作 inexact 诊断，最终跑 vendor MuJoCo。

任一上游门失败都停止该候选。不能用 Reward 抵消自碰、桌网、非 finite 或动力学失败。
