# 运行反手拉 B 的 vendor L1 整轨安全审计

本操作只处理 [`vendor L1 safety audit`](../DEFINITIONS.md#motion-vendor-l1-safety)：机器人自碰与
球拍/拍柄打机器人。它不运行训练、不占 GPU、不检查桌网/动力学，也绝不授权真机。实验真源见
[EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-VENDOR-L1.md)。

## Source gate（当前唯一已执行步骤）

```bash
cd /path/to/clean/nohope
PLAN=configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json
PLAN_SHA=f8530d834392545105cc4dd89d6a177d4f34ce970cc1ba5d7bb3fdb4d04af699

python3 scripts/audit_motion_schema2_vendor_l1_safety.py \
  --prereg "$PLAN" \
  --expected-prereg-sha256 "$PLAN_SHA" \
  static
```

成功行必须含 `source_exact=true runtime_audit=false no_write=true continuous_time_claim=false`。

## Runtime 前置（尚未执行）

只在 code review 后使用 exact CPU venv。必须先只读确认 L0 certificate 与 B NPZ 的 SHA 分别为
`60c08185...afc6`、`e2eb99e6...d28cc`，checkout 中 validator/dependencies/MJCF closure 与 plan 全部
exact，C 没有任何读取或 consume。输出 certificate 必须不存在且父目录必须是人工建立的真实目录；
已有文件或 symlink 一律保全并停止，禁止删除重跑。

环境固定为：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
```

先运行一次只读 full audit：

```bash
/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_vendor_l1_safety.py \
  --prereg configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json \
  --expected-prereg-sha256 f8530d834392545105cc4dd89d6a177d4f34ce970cc1ba5d7bb3fdb4d04af699 \
  dry-run
```

当前任务**没有运行这条命令**。只有 dry-run 通过、逐帧 safety evidence 保存且另有显式发布授权后，
才把最后一个参数改为 `audit` 执行唯一一次 `O_EXCL` 发布。通过只令 `vendor_l1_complete=true` 并解锁
下一张桌网整轨门；dynamics/training/formal motion/hardware 仍 false。

## 失败处理

- lineage/runtime/MJCF/closure 漂移：停止，不改 plan 或阈值；
- 任一自碰穿透或 `<5 mm` 球拍/拍柄余隙：保全 frame/pair/距离，B 在本门失败，不用 reward 补偿；
- warning（`5–20 mm`）：保全并人工检查，但不改 hard threshold；
- 输出存在：禁止覆盖、删除或自动重试；
- 有限 400 Hz sweep 通过也不得写成连续时间、桌网、动力学、训练或真机通过。
