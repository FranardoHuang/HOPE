# 运行反手拉 B 的整轨桌网余隙审计

本操作只检查 Franco 反手拉 B 与固定桌板、网和两根网柱的整轨几何余隙。它不占 GPU、不调用
`mj_step`、不训练、不部署，也不授权真机。实验真源见
[EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-TABLE-NET.md)。

## Source gate

```bash
cd /path/to/clean/nohope
PLAN=configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json
PLAN_SHA=1c73faf9034c1ed5136641ff4594917d5d5f66a5c93e92b35d300107ae9ec6b4

python3 scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg "$PLAN" \
  --expected-prereg-sha256 "$PLAN_SHA" \
  static
```

成功行必须含 `source_exact=true runtime_audit=false no_write=true continuous_time_claim=false`。本地专项：

```bash
python3 -m pytest -q tests/test_motion_backhand_loop_b_table_net_clearance.py
# 36 passed
```

source gate 只证明预注册、坐标系、输入 lineage、5 mm 边界和 no-clobber 反例闭环，不是 B 的桌网通过。
旧 schema-v1 plan SHA `9d7126bc...eb1e6` 已在首次 Pod2 dry-run 证明会把 MuJoCo 的确定性 `+4`
world-geom 编号插入误判为 robot 重排；它永久只作根因证据，不得运行或用于发布 certificate。

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
  --expected-prereg-sha256 1c73faf9034c1ed5136641ff4594917d5d5f66a5c93e92b35d300107ae9ec6b4 \
  dry-run
```

它在内存中把四个 world-fixed box 追加到 canonical MJCF worldbody 并重编译。MuJoCo 会先编号
worldbody 直属 geom，因此唯一允许的全局编号变化是 floor=`0`、四个 obstacle=`1..4`、全部 child-body
robot geom 精确 `+4`；validator 仍逐项核对 robot 相对顺序/名字、37 个 enabled collision geom、
root/joint topology、qpos0、collision row/mesh 和归一化后的 frozen compiled collision SHA。随后才以
1201 个有限 400 Hz 样本检查每样本 148 个 robot-obstacle pair。成功行必须含
`runtime_audit=true certificate_written=false`。任何失败都不得自动重跑或改阈值。

## 唯一 no-clobber audit

只有 full dry-run 通过且结果经人工审阅后，才把末尾命令改为 `audit`：

```bash
/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_table_net_clearance.py \
  --prereg configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json \
  --expected-prereg-sha256 1c73faf9034c1ed5136641ff4594917d5d5f66a5c93e92b35d300107ae9ec6b4 \
  audit
```

输入消费不是“先验 SHA、再按 path 打开”：certificate/NPZ/XML/74 mesh 都从 `O_NOFOLLOW` fd 的单次
bytes snapshot 做 hash 与 parse/load，MJCF closure 由 pinned model-root dirfd 读取。写入绑定输出 parent
device/inode，使用 `openat(O_EXCL|O_NOFOLLOW)`、file+directory `fsync`，并从同一 dirfd 复核新建
inode/bytes；任何输入 path swap 或输出 parent swap 都 fail closed。目标为
`franco_backhand_loop_b_98e7b883b29d.table_net_clearance_certificate.json`。certificate 通过只令
`table_net_complete=true` 并授权下一道 vendor 动力学/平衡门；simulator/RL/formal motion/hardware 仍 false。

运行不得通过 `sys.path`/`sys.modules` 导入完整 phase/self-collision project module。四个 phase kernel 必须
在 source gate 与绑定 upstream AST 等价；距离 kernel 必须通过 upstream parity。结果中的
`minimum_clearance_midpoint_estimate_m` 只是二分中点，只有
`minimum_clearance_certified_lower_bound_m` 是可称作下界的 lower bracket；它还会扣除 saturation
predicate 的 `1e-12 m` 数值裕量。

## 失败处理

- L1 certificate 未授权、NPZ/MJCF/closure/source 漂移：停止并保全，不放宽绑定；
- frame/桌位漂移：停止；不可把 capture 视频背景或另一套 table pose 猜成现役 tracking 桌位；
- 任一 37×4 pair `<5 mm`：B 在本门 hard fail，保全 dense frame/source time/geom pair，不用 reward 补偿；
- `5–20 mm` warning：保全并人工查看，但不改 hard threshold；
- worldbody augmentation 不是 floor=`0`、obstacle=`1..4`、robot 精确 `+4`，或 robot 相对顺序/名字、
  qpos0、拓扑、collision row/mesh、归一化 SHA 任一变化：视为 harness failure，不判 B；
- 输出存在或 symlink：禁止覆盖、删除或重发；
- 有限 400 Hz 通过也不得写成连续时间、动力学、击球、训练、Gate3 或真机通过。
