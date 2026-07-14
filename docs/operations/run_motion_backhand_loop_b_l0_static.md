# 运行反手拉 B 的 CPU-only L0 静态审计

这里的 [`L0`](../DEFINITIONS.md#motion-l0-static) 是“纯 CPU、零 dynamics step 的静态动作可行性审计”；
source/static pass 不等于真实资产已有 runtime certificate。权威实验边界见
[EXP-MOTION-BACKHAND-LOOP-B-L0](../experiments/2026-07/EXP-MOTION-BACKHAND-LOOP-B-L0.md)。

## 当前授权

V1 full `dry-run` 已在 Pod2 fail closed，原因是 schema-2 未保存 producer 原始 free-joint qpos，却要求
把 post-FK float32 root body pose 再注入后逐字节相等；V1 原字节和失败账冻结，禁止改阈值或重跑。
V2 目前只完成本地 source/static gate 与 dependency-light 测试，本任务没有连接 Pod。只有分支 review
并合入 main 后，才允许操作人用 exact detached-clean source 在 Pod2 执行**一次** V2 full `dry-run`。
它不写 certificate，也不运行 dynamics step、trainer、judge、部署或真机。C 的一次性 consume 继续禁止。

## Source/static gate

```bash
cd /path/to/clean/nohope
PLAN=configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json
PLAN_SHA=185612a99d5dd1e0aba0d04d50467103ea9b3967b917c58371bd409d10fc6ccb

python3 scripts/audit_motion_schema2_l0_static_v2.py \
  --prereg "$PLAN" \
  --expected-prereg-sha256 "$PLAN_SHA" \
  static
```

预期唯一成功行包含：

```text
[motion-l0-v2] PASS static asset=franco_backhand_loop_b source_exact=true runtime_audit=false no_write=true v1_unchanged=true
```

## Runtime 前置只读核对

在 exact CPU venv 中先逐份复算 SHA；任一不符就停止，不创建输出父目录：

```bash
NPZ=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/franco_backhand_loop_b_98e7b883b29d/franco_backhand_loop_b.98e7b883b29d.schema2_fk.npz
REPORT=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/franco_backhand_loop_b_98e7b883b29d/schema2_fk_report.json
CLAIM=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/.bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.claim.json
SUCCESS=/workspace/codexschema/motion_video_intake_20260711/schema2_fk_primary_v1/.bc_schema2_fk_consume_control_v2/franco_backhand_loop_b.success.json
V1_CERT=/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v1/franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json
V2_CERT=/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v2/franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json

test "$(sha256sum "$NPZ" | awk '{print $1}')" = e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc
test "$(sha256sum "$REPORT" | awk '{print $1}')" = 4f5245937956290b3f623acbb588d99b346e5a1d874e55ee9caf010f2d75bc38
test "$(sha256sum "$CLAIM" | awk '{print $1}')" = 76e7ff88fea39c13b45096edaad504b2570b3ce079acc96366b820a9c1295fb0
test "$(sha256sum "$SUCCESS" | awk '{print $1}')" = c0a25f2cba0e61bf0df7f63e6493948e16c5a3d3074f65091430f29e417f4f8b
test ! -e "$V1_CERT" && test ! -L "$V1_CERT"
test ! -e "$V2_CERT" && test ! -L "$V2_CERT"
```

不要删除、覆盖或“清理后重跑”已有证书。若路径存在，先保全并审计；no-clobber 是结果合同的一部分。

## Pod2 V2 full dry-run（仅在合 main 后的远端下一步）

`dry-run` 不创建 certificate 或其父目录；若 Pod2 上父目录不存在也保持不存在：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

/workspace/hope_mjeval_venv/bin/python \
  scripts/audit_motion_schema2_l0_static_v2.py \
  --prereg configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json \
  --expected-prereg-sha256 185612a99d5dd1e0aba0d04d50467103ea9b3967b917c58371bd409d10fc6ccb \
  dry-run
```

预期成功行必须同时包含 `runtime_audit=true certificate_written=false l0_static_complete=false`。成功只表示
exact B NPZ 的完整 L0 只读路径通过；仍没有 certificate，也不授权 vendor L1、桌网、动力学或 RL。
复核 dry-run 后才可另行授权同计划的 `audit` 发布动作证书。V2 runtime 未跑前，不得根据 V1 已知差值
声称 V2 必然通过。

## 失败处理

- 2026-07-14 首次调用已在上游 claim 校验阶段、任何运动学检查和 certificate 写入前停止；只创建了
  certificate 父目录。不要把它记成 L0 行为失败，也不要在 portable source 修复合入前重跑。修复只把
  历史 consume checkout 的绝对路径降为 provenance，仍严格要求 claim 绑定的 activation bytes/SHA 与
  source tuple、当前 detached-clean source commit/runner/source-validator/body-order、attempt ID、receipt、
  NPZ/report、claim/success 全部一致；没有历史路径 fallback。
- portable 修复后的 V1 dry-run 已进入运动学门并因 float32 round-trip byte equality 停止：position
  `537 / 1.1920929e-7`、quaternion `917 / 5.9604645e-8`、COM velocity
  `1261 / 2.9802322e-6`、angular velocity `2320 / 5.9679151e-6`（component count / max abs）；
  没有 certificate。保留该负结果，禁止删除 V1 输出目录后重跑。
- 输入、谱系、runtime、MJCF closure 或 compiled collision SHA 不符：保留原件，停止，不改阈值。
- V2 的 field-specific 数值包络、关节范围、支撑脚最低 body 或地面余隙失败：记录 exact frame/value，停止该 B
  动作；这是 internal gate failure，不能自动切换 B 的 frozen fallback。
- 输出路径已存在或含 symlink：不删除，不覆盖；先人工审计已有对象。
- 本操作绝不授权真机。
