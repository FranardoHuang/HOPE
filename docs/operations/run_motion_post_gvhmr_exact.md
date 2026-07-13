# Consume exact S0/M0 post-GVHMR evidence

本操作把已经完成的 [`S0/M0`](../DEFINITIONS.md) GVHMR 结构结果收成一个不可覆盖的 lineage handoff。
它只授权下一步另行预注册的 canonical-beta materialization，不运行 GMR、schema-2、仿真、训练或真机。

> 2026-07-13：两批 `consume` 已完成。S0 handoff 是 4,970 bytes / `d57a93e0...a1054`，M0 是
> 9,242 bytes / `60c55150...088ef`。不要删除 output root 或重跑 `consume`；当前下一步入口是
> [`run_motion_handoff_canonical_betas.md`](run_motion_handoff_canonical_betas.md)。下方 consume 命令只保留为
> 首次执行的可复现记录。

详细证据与边界见
[S0/M0 post-GVHMR 实验记录](../experiments/motion_post_gvhmr_s0_m0_handoff_20260713.md)。

## 1. Host static dry-run

不需要私有资产即可检查 committed contract、上游 prereg、donor result 和工具 SHA：

```bash
for PLAN in \
  configs/motion_post_gvhmr_s0_prereg_20260713.json \
  configs/motion_post_gvhmr_m0_prereg_20260713.json
do
  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  python3 scripts/consume_motion_post_gvhmr_exact.py \
    --prereg "$PLAN" --expected-prereg-sha256 "$PLAN_SHA" static
done

python3 -m pytest -q tests/test_consume_motion_post_gvhmr_exact.py
```

预期为两次 `PASS static` 与 `8 passed`。这不证明 Pod evidence 仍在，也不创建 handoff。

## 2. 恢复 ignored/private 依赖

runtime inspection 必须在保留原 exact evidence 的机器上运行。不要复制 basename 相同的 PT 代替。每份
prereg 已绑定：

- S0/M0 各自的 execution record、state root 和最终 queue state；
- 每条 `bindings/<asset>.json`、`audits/<asset>.json` 与实际 `hmr4d_results.pt`；
- 旧 canonical-beta donor 的 `canonical_betas.json` 和 `materialization_manifest.json`。

若任何绝对路径缺失，应按
[`setup_local_sync.md`](setup_local_sync.md#v12staticlateral-private-motion-video-intake-2026-07-13)
恢复原制品或在原证据机运行；不能改 prereg 指向一份未登记 copy。GMR private runtime 不属于本操作，
也不阻塞 `inspect`。

## 3. 只读 inspect

先对每批只读复算完整链：

```bash
S0=configs/motion_post_gvhmr_s0_prereg_20260713.json
S0_SHA=$(sha256sum "$S0" | awk '{print $1}')
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$S0" --expected-prereg-sha256 "$S0_SHA" inspect

M0=configs/motion_post_gvhmr_m0_prereg_20260713.json
M0_SHA=$(sha256sum "$M0" | awk '{print $1}')
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$M0" --expected-prereg-sha256 "$M0_SHA" inspect
```

`inspect` 必须在 output root 不存在时和存在时都保持只读。任何 bytes/SHA、queue status、asset set、
binding/audit 或 donor 不一致都会返回非零；不要通过编辑 JSON、删旧 binding 或覆盖 PT“修复”。

## 4. 一次性 consume

只有 `inspect` PASS 后才分别执行：

```bash
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$S0" --expected-prereg-sha256 "$S0_SHA" consume
python3 scripts/consume_motion_post_gvhmr_exact.py \
  --prereg "$M0" --expected-prereg-sha256 "$M0_SHA" consume
```

consumer 先完成所有验证，再以新目录 claim exact output root，以 `O_CREAT|O_EXCL` 写 `handoff.json` 并
fsync。root 已存在会 fail closed；不得删除后重跑。中断后即使只留下 output root，也视为保留的失败证据，
需新版本 prereg 和新 root 才能重试。

## 5. handoff 后仍然 blocked 的项目

- S0 没有真实触球，不能使用拉球题或声称高点拍压有效；
- M0 没有 robot foot stance 结果；GMR 后必须预注册 foot-site mapping、二维初始/终态脚间向量和数值容差，
  更窄或双脚并拢必须失败；
- 新五条 canonical-beta PT 仍未物化；
- GMR、schema-2、L0、vendor L1/dynamics、RL、Gate3 和硬件均未授权。
