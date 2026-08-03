# A211 frame0 exact Pod nominal-hold receipt

本工序只为 tracked `Take_061_unit04_BH` frame0 exact candidate 生成
[`A211 frame0 exact hold receipt`](../DEFINITIONS.md#a211-frame0-exact-hold-receipt)：在 exact clean Pod
checkout 中复用 live Isaac nominal-hold probe，跑足 `200` 个 policy tick（`4.0 s / 800` 个
physics substep），然后把未经删减的 live safety evidence 嵌入 canonical no-clobber receipt。
它不启动 PPO，也不授权训练、晋级、导出、部署或真机。

## 冻结输入

| 输入 | exact binding |
| --- | --- |
| frame0 artifact | `configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json`; file SHA `ad17d984559776e90c70182ac4c0361c01de95094859e725162fb958defdbc54`; content SHA `41d7da29faab531dc72130495d5d3f760028c5bb460ff99c7dde07bc5124e6f5`; first-containing commit `5ed998f1e1526fa84dfc2198b064f9f8e6ab6068` |
| plant template | `configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json`; file SHA `ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069` |
| measured motion | `assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz`; file SHA `aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e` |

wrapper 只从 template 复用 31 关节顺序和 exact runtime plant。临时 probe input 的 physical root/q
逐值等于 frame0 artifact，root/joint velocity 全零，`hold_qdes=frame0 q`，normalized action 必须由
`qdes=default+scale*action` 对每关节反解并落在 exact qdes soft envelope 内。旧 template 的
physical-ready、LP hold、teacher/physical split 或历史 PASS 均不会继承。

## exact Pod 命令模板

先将本实现提交为 clean commit；`SOURCE_COMMIT` 必须是含 wrapper、consumer 和既有 live probe 的
exact Pod `HEAD`。工作目录必须在 checkout 外且尚不存在，最终 receipt 路径也必须尚不存在。

```bash
SOURCE_ROOT=/workspace/franco/<clean-exact-checkout>
SOURCE_COMMIT=<40-char-exact-HEAD>
ISAACLAB_ROOT=/workspace/IsaacLab
WBT_SOURCE="$SOURCE_ROOT/hope_training/whole_body_tracking/source/whole_body_tracking"
ISAAC_SOURCE="$ISAACLAB_ROOT/source"
WORK_DIR=/workspace/franco/a211_frame0_hold_<fresh-id>

env -u CUDA_VISIBLE_DEVICES \
  HOPE_URDF_IMPORTER_NO_UI=1 \
  HOPE_AGIBOT_A3_USD_PATH=/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/model.usd \
  PYTHONPATH="$WBT_SOURCE:/opt/drone_venv/lib/python3.11/site-packages:$ISAAC_SOURCE/isaaclab:$ISAAC_SOURCE/isaaclab_tasks:$ISAAC_SOURCE/isaaclab_assets:$ISAAC_SOURCE/isaaclab_rl" \
  LD_LIBRARY_PATH=/workspace/franco/runtime_assets/libopengl_noble_1_7_0/usr/lib/x86_64-linux-gnu:/workspace/franco/runtime_assets/libglu_af791d1e \
  /workspace/hope_isaac_venv/bin/python \
  "$SOURCE_ROOT/hope_training/whole_body_tracking/scripts/run_action_ball_a211_frame0_nominal_hold.py" \
  --repo-root "$SOURCE_ROOT" \
  --probe-source-commit "$SOURCE_COMMIT" \
  --artifact-source-commit 5ed998f1e1526fa84dfc2198b064f9f8e6ab6068 \
  --frame0-artifact-path configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json \
  --expected-frame0-artifact-sha256 ad17d984559776e90c70182ac4c0361c01de95094859e725162fb958defdbc54 \
  --plant-template-path configs/action_ball_n1_measured_20260803/evidence_holdpass_robust20n_20260803/take061.measured_teacher.yaw_aligned_full_seed.robust20n.dynamic_ready.v2.json \
  --expected-plant-template-sha256 ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069 \
  --motion-path assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz \
  --expected-motion-sha256 aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e \
  --device cuda:<FREE_PHYSICAL_GPU> \
  --python /workspace/hope_isaac_venv/bin/python \
  --work-dir "$WORK_DIR" \
  --output configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.nominal_hold.receipt.v1.json
```

不要在两次 Kit boot 之间复用 `$WORK_DIR`，不要设置逻辑 `CUDA_VISIBLE_DEVICES` 后再把 `cuda:0`
误写成物理卡号。wrapper 在启动前和 live child 自然退出后各检查一次 exact clean checkout；任何
其他 tracked/untracked 文件都会阻断。失败会保留 checkout 外的 probe input、raw receipt/截图目录，
但不会铸造最终 PASS。

## 验收与后续

最终 receipt 必须同时满足：artifact file/content/first commit、probe source commit、plant template、
临时 probe input file/content 和 raw live receipt file/content 全绑定；raw evidence 本体嵌入 receipt；
五类安全 termination 均 active；无 terminal/truncation/reason；current/substep actual hard-edge 均为零；
最终 hard gap 为正；root/foot telemetry 有限；`raw reset / exact frame0 write / step1 / step10 / final`
五帧截图各有 SHA。

人工审查 `$WORK_DIR` 与最终 JSON 后，只提交最终 receipt；随后用
`materialize_action_ball_a211_lineage.py` 生成 commit-required lineage。receipt、lineage 和后续 launcher
仍保持 `diagnostic_unauthorized=true`，oracle32 未过前不得进入 4096。
