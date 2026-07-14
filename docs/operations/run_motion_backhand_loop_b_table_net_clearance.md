# 运行反手拉 B 的整轨桌网余隙审计

本操作只检查 Franco 反手拉 B 与固定桌板、网和两根网柱的整轨几何余隙。它不占 GPU、不调用
`mj_step`、不训练、不部署，也不授权真机。实验真源见
[EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)。

## Source gate

```bash
cd /path/to/clean/nohope
PLAN=configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json
PLAN_SHA=c1899ffff4564986ced934413d581ffbacc5328a90663421526589f3630804b9

python3 scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg "$PLAN" \
  --expected-prereg-sha256 "$PLAN_SHA" \
  static
```

成功行必须含 `source_exact=true runtime_audit=false no_write=true continuous_time_claim=false`。本地专项：

```bash
python3 -m pytest -q tests/test_motion_backhand_loop_b_table_net_clearance.py
# 16 passed
```

source gate 只证明预注册、坐标系、输入 lineage、5 mm 边界和 no-clobber 反例闭环，不是 B 的桌网通过。

## Runtime 前置

只在 code review 后使用 vendor L1 同一 exact CPU runtime：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
```

必须先只读确认：

1. checkout clean，plan/validator 仍与上面的 SHA 一致；
2. B NPZ SHA 为 `e2eb99e6...d28cc`；
3. vendor L1 certificate SHA 为 `6840df34...db60`，且 certificate 内
   `vendor_l1_complete=true`、`table_net_authorized=true`；
4. L1 plan/validator、vendor MJCF/75-file closure 和 compiled collision SHA 全部 exact；
5. 输出父目录
   `/workspace/codexschema/motion_video_intake_20260711/table_net_primary_v1` 已由操作者建立、是真实目录，
   certificate target 不存在且不是 dangling symlink。已有 target 必须保全并停止，禁止删除重跑。

## 只读 full dry-run

```bash
/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json \
  --expected-prereg-sha256 c1899ffff4564986ced934413d581ffbacc5328a90663421526589f3630804b9 \
  dry-run
```

它在内存中把四个 world-fixed box 追加到 canonical MJCF 末尾并重编译，要求原 37 个 robot geom ID、
qpos0、拓扑和 compiled collision SHA 不变；随后以 1201 个有限 400 Hz 样本检查每样本 148 个
robot-obstacle pair。成功行必须含 `runtime_audit=true certificate_written=false`。任何失败都不得自动重跑或
改阈值。

## 唯一 no-clobber audit

只有 full dry-run 通过且结果经人工审阅后，才把末尾命令改为 `audit`：

```bash
/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json \
  --expected-prereg-sha256 c1899ffff4564986ced934413d581ffbacc5328a90663421526589f3630804b9 \
  audit
```

写入使用 `O_EXCL`/`O_NOFOLLOW`，目标为
`franco_backhand_loop_b_98e7b883b29d.table_net_clearance_certificate.json`。certificate 通过只令
`table_net_complete=true` 并授权下一道 vendor 动力学/平衡门；simulator/RL/formal motion/hardware 仍 false。

## 失败处理

- L1 certificate 未授权、NPZ/MJCF/closure/source 漂移：停止并保全，不放宽绑定；
- frame/桌位漂移：停止；不可把 capture 视频背景或另一套 table pose 猜成现役 tracking 桌位；
- 任一 37×4 pair `<5 mm`：B 在本门 hard fail，保全 dense frame/source time/geom pair，不用 reward 补偿；
- `5–20 mm` warning：保全并人工查看，但不改 hard threshold；
- in-memory augmentation 改了 robot geom ID、qpos0、拓扑或 collision SHA：视为 harness failure，不判 B；
- 输出存在或 symlink：禁止覆盖、删除或重发；
- 有限 400 Hz 通过也不得写成连续时间、动力学、击球、训练、Gate3 或真机通过。
