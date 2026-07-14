# 运行反手拉 B 的 CPU-only L0 静态审计

这里的 [`L0`](../DEFINITIONS.md#motion-l0-static) 是“纯 CPU、零 dynamics step 的静态动作可行性审计”；
source/static pass 不等于真实资产已有 runtime certificate。权威实验边界见
[EXP-MOTION-BACKHAND-LOOP-B-L0](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)。

## 当前授权

当前只有 source/static gate。下面的 runtime 命令必须等分支经 review 合入 main、用该 exact source
建立 clean detached checkout，并由操作人重新确认四份冻结输入存在且证书路径不存在后才能运行。
本命令不运行 simulator、trainer、judge、部署或真机。

## Source/static gate

```bash
cd /path/to/clean/nohope
PLAN=configs/motion_backhand_loop_b_l0_static_prereg_20260714.json
PLAN_SHA=7118b9cda1d2ec4affb7906d1a330f6c04a85b1d624e894d369b7badefe595a6

python3 scripts/audit_motion_schema2_l0_static.py \
  --prereg "$PLAN" \
  --expected-prereg-sha256 "$PLAN_SHA" \
  static
```

预期唯一成功行包含：

```text
[motion-l0] PASS static asset=franco_backhand_loop_b source_exact=true runtime_audit=false no_write=true
```

## Runtime 前置只读核对

在 exact CPU venv 中先逐份复算 SHA；任一不符就停止，不创建输出父目录：

```bash
NPZ=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/franco_backhand_loop_b_98e7b883b29d/franco_backhand_loop_b.98e7b883b29d.schema2_fk.npz
REPORT=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/franco_backhand_loop_b_98e7b883b29d/schema2_fk_report.json
CLAIM=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/.bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.claim.json
SUCCESS=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/.bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.success.json
CERT=/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v1/franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json

test "$(sha256sum "$NPZ" | awk '{print $1}')" = e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc
test "$(sha256sum "$REPORT" | awk '{print $1}')" = 4f5245937956290b3f623acbb588d99b346e5a1d874e55ee9caf010f2d75bc38
test "$(sha256sum "$CLAIM" | awk '{print $1}')" = 76e7ff88fea39c13b45096edaad504b2570b3ce079acc96366b820a9c1295fb0
test "$(sha256sum "$SUCCESS" | awk '{print $1}')" = c0a25f2cba0e61bf0df7f63e6493948e16c5a3d3074f65091430f29e417f4f8b
test ! -e "$CERT" && test ! -L "$CERT"
```

不要删除、覆盖或“清理后重跑”已有证书。若路径存在，先保全并审计；no-clobber 是结果合同的一部分。

## Runtime audit（合入并复核后才运行）

预注册要求 certificate parent 已存在且为真实目录。只在上述只读检查全过后创建一次：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

mkdir -p /workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v1

/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_l0_static.py \
  --prereg configs/motion_backhand_loop_b_l0_static_prereg_20260714.json \
  --expected-prereg-sha256 7118b9cda1d2ec4affb7906d1a330f6c04a85b1d624e894d369b7badefe595a6 \
  audit
```

成功只代表 exact B NPZ 的离散静态 L0 证书。随后先复算 certificate SHA、检查 JSON 与源码/输入绑定，
再单独预注册并运行 vendor L1 自碰/球拍自打；不得直接启动桌网、动力学或 RL。

## 失败处理

- 2026-07-14 首次调用已在上游 claim 校验阶段、任何运动学检查和 certificate 写入前停止；只创建了
  certificate 父目录。不要把它记成 L0 行为失败，也不要在 portable source 修复合入前重跑。修复只把
  历史 consume checkout 的绝对路径降为 provenance，仍严格要求 activation bytes/SHA、canonical path、
  inspected source commit、attempt ID、receipt/runner、NPZ/report、claim/success 全部一致。
- 输入、谱系、runtime、MJCF closure 或 compiled collision SHA 不符：保留原件，停止，不改阈值。
- 关节范围、FK byte equality、支撑脚最低 body 或地面余隙失败：记录 exact frame/value，停止该 B
  动作；这是 internal gate failure，不能自动切换 B 的 frozen fallback。
- 输出路径已存在或含 symlink：不删除，不覆盖；先人工审计已有对象。
- 本操作绝不授权真机。
