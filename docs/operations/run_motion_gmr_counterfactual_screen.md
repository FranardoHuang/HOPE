# Run the GMR → HOPE counterfactual frame and phase screen

日期：2026-07-11
范围：CPU-only offline audit；禁止训练、GPU、真机和覆盖已有结果

## 私有输入恢复

原视频仍在 `${HOME}/Downloads/{Franco,v6_dang,v7_dang}`，必须先与
`configs/motion_video_intake_20260711.json` 的 bytes/SHA 对齐。grounded GMR 从 Pod1 精确恢复到
ignored root（不要用 basename 拼接不同版本）：

```text
vendor_assets/motion_video_intake_20260711/gmr_canonical_betas_grounded_v2/
```

逐文件以 `configs/motion_video_canonical_gmr_ground_results_20260711.json` 的 `output` 绑定验收。
视频、GMR、witness crop 和 full result 都不进 git。

## 1. 验 frame prereg

```bash
CUDA_VISIBLE_DEVICES= python3 scripts/audit_motion_gmr_frame_contract.py \
  --manifest configs/motion_video_gmr_frame_contract_prereg_20260711.json \
  --expected-manifest-sha256 f625e0a0a403b8908a2e5c575917cea93e9e2a2c88e2db4db385d2b14a07e97e \
  validate
```

`witnesses` 必须写一个全新 ignored 路径；输出 crop SHA 后人工逐条看正常方向中文标签，再与
`configs/motion_video_mirror_witness_review_20260711.json` 对表。禁止提交 crop 像素。

最终 accepted frame result SHA 为
`e70492becf5a2fae5ee74724d22f9ca9d2e874d535231e3ee6649f01669048f0`；十矩阵只由
frame-0 pelvis + 已审地面生成，不读问题结果。

## 2. v5 external runtime

Pod1 control/output：

```text
/workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v5/
/workspace/codexschema/motion_video_intake_20260711/phase_safety_v5/
```

已接受的 manifest SHA 是
`fee1b1f9a68fcc0323c1be5832db1b29bdc5f49421712c6f44506d16dae45529`，screen tool SHA 是
`d3924b1a045efa08b77aee92da81633ed68a0fd05422da0a1543692452357b0f`。先运行 `validate`：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/hope_isaac_venv/bin/python \
  /workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v5/screen_motion_gmr_phase_safety.py \
  --manifest /workspace/codexschema/motion_video_intake_20260711/phase_safety_control_v5/motion_video_gmr_phase_counterfactual_prereg_20260711.json \
  --expected-manifest-sha256 fee1b1f9a68fcc0323c1be5832db1b29bdc5f49421712c6f44506d16dae45529 \
  validate
```

v5 已经完成，PID/PGID `1471093`；**不要重跑/覆盖**。full result：792,241 bytes，SHA
`c299b7a04417e855005ad315b40203204bb0cc192398d83980179b212e6bef53`。小账本是
`configs/motion_video_gmr_phase_counterfactual_results_20260711.json`。

**2026-07-13 诚实门边界：**上述 v5 绑定旧 `41fe2a07...64d11` scorer；它的
`orient_normal` 把 `n/-n` 当成同一平面，所以所有 virtual-return/phase/library 列只保留为
历史 unsigned-plane 诊断，不得晋级动作。安全、ground、frame witness 子树仍按原 SHA 有效。
不要重写旧结果或用当前源码冒充复现；未来回球筛卷必须新 prereg，明确传入 raw-A achieved、
raw-A target、clip id 和 `[+1,-1]` physical-B 映射，并在 plane orientation 前做 signed-face 门。

## 3. 结果验收

必须同时检查：

- result JSON 可严格解析、无 NaN/Inf，log 无 Traceback/OOM/error；
- 64 题 semantic SHA `4dfa0548...` 且 `consumed_for_returnability=true`；
- frame evidence SHA `e70492be...`，`capture_table_pose_observed=false`；
- contact truth、real-capture returnability 都是 `null`；
- v5 十条 safety subtree 与 accepted v4 逐资产相等；
- 进程退出后无同 control/output 命令行残留；
- frozen training checkout 仍 clean HEAD `6d93bcb16c422a2f42748c2dc99432559653480b`。

测试：

```bash
pytest -q \
  tests/test_motion_video_gmr_frame_contract.py \
  tests/test_screen_motion_gmr_phase_safety.py \
  tests/test_motion_video_gmr_phase_counterfactual_prereg_manifest.py \
  tests/test_motion_video_gmr_phase_counterfactual_result_manifest.py
```

## 4. 决策纪律

exact `0/64` 不是“动作无效”，只是 fixed-position zero-retarget 无共同支持。当前只保留 Franco
反手拉 B/C 为显式 spatial-retarget 候选；2-vs-4 继续暂停。TOPP 必须等空间重定向、schema-2、
L0/L1、桌网与动力学门通过；最终动作/动作库选择由智元 vendor MuJoCo Gate3/Gate3B 主判，
连续动作评测不允许 reset。

下一步的全十动作×immutable question 原子 SE(2) proposal 屏已单独预注册；它不修改
本轮 frame matrix，也不反推录制现场桌外参。恢复 full v5 大结果、运行 proposal 屏以及
schema-2/L0/L1/桌网证书边界见
`docs/operations/run_motion_spatial_retarget_screen.md`。
