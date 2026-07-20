# Materialize S0/M0 canonical betas from exact handoffs

本操作把 [`S0/M0`](../DEFINITIONS.md) exact post-GVHMR handoff 中的五条 PT，逐条写入同一份已验 donor
body shape。它只使用 CPU，不占训练 GPU；不运行 GMR、仿真、训练或真机。证据与边界见
[实验记录](../experiments/motion_canonical_beta_s0_m0_20260713.md)。

## 1. Host 静态检查

```bash
python3 scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg configs/motion_canonical_betas_s0_prereg_20260713.json \
  --expected-prereg-sha256 \
  236cace8aeae6c80f333194f8f73f9a718720057e8badc62d2769c1a08d94f19 \
  static
python3 scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg configs/motion_canonical_betas_m0_prereg_20260713.json \
  --expected-prereg-sha256 \
  c70d1fdbe75b3f22d5ca55193cb15c199882ac7df5976dc31fe19dc1fc9fcb69 \
  static

python3 -m pytest -q \
  tests/test_materialize_motion_handoff_canonical_betas.py \
  tests/test_materialize_canonical_gvhmr_betas.py
```

预期两次 `PASS static`。它只验证 committed plan/tool closure，不读取私有 PT，也不创建目录。

## 2. exact runtime 与资产

必须使用 prereg 冻结的 CPU runtime：

```bash
export CUDA_VISIBLE_DEVICES=
PY=/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10
"$PY" --version
```

要求 Python `3.10.20` 且规范化 `pip freeze --all` SHA 为
`56b0f8af9677b279bbb4925b6f49113f484dcb9ded1ed8d9bc56af71f304c694`。consumer 会自己复核；不要改计划
来迁就另一套环境。

需要同时恢复：

- S0/M0 两个 exact `handoff.json` 及 handoff 中绑定的五条 GVHMR PT；
- 旧 donor `canonical_betas.json` 与它的 completion manifest；
- prereg 所列绝对路径。不能用 basename 相同的 copy 代替。

恢复路径见
[`setup_local_sync.md`](setup_local_sync.md#v12staticlateral-private-motion-video-intake-2026-07-13)。

## 3. 先 inspect，完全不写

```bash
S0=configs/motion_canonical_betas_s0_prereg_20260713.json
M0=configs/motion_canonical_betas_m0_prereg_20260713.json
S0_SHA=236cace8aeae6c80f333194f8f73f9a718720057e8badc62d2769c1a08d94f19
M0_SHA=c70d1fdbe75b3f22d5ca55193cb15c199882ac7df5976dc31fe19dc1fc9fcb69

"$PY" scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg "$S0" --expected-prereg-sha256 "$S0_SHA" inspect
"$PY" scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg "$M0" --expected-prereg-sha256 "$M0_SHA" inspect
```

`inspect` 会加载每个 PT、验证 beta shape/dtype、复算 non-beta digest，并确认 donor float32 bytes 正好复现
冻结 vector SHA，但不会写文件。S0 与 M0 独立；一批失败时保留错误和原制品，另一批仍可单独检查。

## 4. 一次性 consume

只有对应 `inspect` PASS 后才执行：

```bash
"$PY" scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg "$S0" --expected-prereg-sha256 "$S0_SHA" consume
"$PY" scripts/materialize_motion_handoff_canonical_betas.py \
  --prereg "$M0" --expected-prereg-sha256 "$M0_SHA" consume
```

输出分别是：

- `/workspace/codexschema/motion_video_intake_20260713_s0/canonical_betas_v1/`；
- `/workspace/codexschema/motion_video_intake_20260713_m0/canonical_betas_v1/`。

root 必须原先不存在。consumer 先在 private staging 完成全部 save/reload 验证，再 no-clobber hard-link 普通
制品并 fsync 目录，最后才 link completion manifest 并再次 fsync。中途失败留下的 partial root 是失败证据；
不得删除覆盖，重试必须另建版本化 prereg/root。

## 5. consume 后怎么检查

每个 completion manifest 必须满足：

- status 为 `complete_exact_donor_beta_materialization`；
- 结果数分别为 1 和 4，顺序与 handoff 相同；
- 每条 `non_beta_bit_exact=true`，source/output non-beta digest 与 leaf count 相等；
- output canonical vector SHA 为 `a03f1642...d9cc6`；
- `canonical_betas.json` SHA 与 donor 原件相同，并明确标成 donor copy，不是新 cohort aggregation；
- `formal_eligible/training_authorized/hardware_authorized` 全为 false；
- M0 foot mapping、初末 `d_xy`、容差、`stance_passed` 全为 null。

本层完成时只允许新建并审查 exact GMR prereg。旧 GMR queue/consumer 不接受这个新 result
status/suffix；不得手工把新 PT 塞进旧队列。后来 exact-GMR v2 诊断已完成，但仍无
schema-2、安全、动作效果或训练资格；当前结论见
[exact-GMR 卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。

2026-07-13 的第一次正式运行已按上述顺序完成。S0/M0 completion manifest SHA 分别为
`964a7333...f1be3` / `5cef05f7...71a65`，结果数为 `1/4`；五条 non-beta 内容全 bit-exact，donor copy
SHA 均为 `f405ba45...4cbf2`。这些值只用于识别这次已完成输出；后续不得删除 output root 或用同一
prereg 覆盖重跑。
