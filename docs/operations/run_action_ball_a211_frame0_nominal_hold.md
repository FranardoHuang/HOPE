# SUPERSEDED / COUNTEREXAMPLE — A211 exact-frame0 Pod nominal hold

> **2026-08-04：本页已退役，不是当前 A211/C211 发车工序。**同一双足/地面/支撑门下的73条
> direct measured-frame0 physical-birth 扫描结果为 `0/73`。旧 wrapper 的 exact Pod raw FAIL
> 只保留为 counterexample/provenance；不得用本页命令生成或补签 current lineage。

当前合同把 physical reset 与 teacher authority 分开：physical reset 消费 tracked split-ready artifact
`ab6b7e41…d38069`，其关节速度逐字节为零；`60 policy tick / 240 physics substep / 1.2 s` hold
receipt `c8b92a28…b19b` 覆盖 hidden WAIT 的最大25个 control tick。WAIT 中 plant 与 teacher 都停在
split-ready；task reveal 同 tick 把 teacher 切到 measured frame 0，并公开原始
`time_to_teacher_start≈.712376 s`，由 dense mimic 学 safe-ready→frame0 bridge。4 s 被动 hold 的后续
termination 只记行为反例，不恢复本页 `62/248` exact-frame0 birth 门或 `200/800` 前置。

以下内容是历史工序的可复现记录。它当时只为 tracked `Take_061_unit04_BH` frame0 exact candidate 生成
[`A211 frame0 exact hold receipt`](../DEFINITIONS.md#a211-frame0-exact-hold-receipt)：在 exact clean Pod
checkout 中复用 live Isaac nominal-hold probe。时长从同一 sealed timing receipt 和 plant 推导：
`1 reset-readback + max_wait_ticks + ceil(pre_swing_wait/policy_dt)`；当前为
`1+25+36=62` 个 policy tick，plant decimation=`4`，因此是 `248` 个 physics substep。
然后把未经删减的 live safety evidence 嵌入 canonical no-clobber receipt。
它不启动 PPO，也不授权训练、晋级、导出、部署或真机。

## 历史冻结输入（非 current lineage）

| 输入 | exact binding |
| --- | --- |
| frame0 artifact | `configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json`; file SHA `ad17d984559776e90c70182ac4c0361c01de95094859e725162fb958defdbc54`; content SHA `41d7da29faab531dc72130495d5d3f760028c5bb460ff99c7dde07bc5124e6f5`; first-containing commit `5ed998f1e1526fa84dfc2198b064f9f8e6ab6068` |
| plant template | 当时分支物化并提交的 threshold-first candidate；必须 `exact_measured_frame0_selected=true`，路径与 file SHA 由该历史 clean commit 固定；current A/C 不消费 |
| measured motion | `assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz`; file SHA `aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e` |

wrapper 只从 template 复用 31 关节顺序和 exact runtime plant。临时 probe input 的 physical root/q
逐值等于 frame0 artifact，root/joint velocity 全零，`hold_qdes=frame0 q`，normalized action 必须由
`qdes=default+scale*action` 对每关节反解并落在 exact qdes soft envelope 内。旧 template 的
physical-ready、LP hold、teacher/physical split 或历史 PASS 均不会继承。

## 历史 exact Pod 命令模板（只用于复核旧 counterexample）

如只读复核旧 counterexample，`SOURCE_COMMIT` 必须是含 wrapper、consumer 和既有 live probe 的
历史 exact Pod `HEAD`。不得把输出接入 current A/C launcher、lineage 或 PRE-LONG。工作目录必须在
checkout 外且尚不存在，最终 receipt 路径也必须尚不存在。

```bash
SOURCE_ROOT=/workspace/franco/<clean-exact-checkout>
SOURCE_COMMIT=<40-char-exact-HEAD>
ISAACLAB_ROOT=/workspace/IsaacLab
WBT_SOURCE="$SOURCE_ROOT/hope_training/whole_body_tracking/source/whole_body_tracking"
ISAAC_SOURCE="$ISAACLAB_ROOT/source"
WORK_DIR=/workspace/franco/a211_frame0_hold_<fresh-id>
PLANT_TEMPLATE_REL=<repo-relative-threshold-first-candidate.json>
PLANT_TEMPLATE_SHA=<64-char-file-sha256>

env -u CUDA_VISIBLE_DEVICES \
  HOPE_URDF_IMPORTER_NO_UI=1 \
  HOPE_AGIBOT_A3_USD_PATH=/workspace/franco/runtime_assets/a3_preconverted_usd_1b3fecd7/model.usd \
  PYTHONPATH="$WBT_SOURCE:$ISAAC_SOURCE/isaaclab:$ISAAC_SOURCE/isaaclab_tasks:$ISAAC_SOURCE/isaaclab_assets:$ISAAC_SOURCE/isaaclab_rl" \
  LD_LIBRARY_PATH=/workspace/franco/runtime_assets/libopengl_noble_1_7_0/usr/lib/x86_64-linux-gnu:/workspace/franco/runtime_assets/libglu_af791d1e \
  /workspace/hope_isaac_venv/bin/python \
  "$SOURCE_ROOT/hope_training/whole_body_tracking/scripts/run_action_ball_a211_frame0_nominal_hold.py" \
  --repo-root "$SOURCE_ROOT" \
  --probe-source-commit "$SOURCE_COMMIT" \
  --artifact-source-commit 5ed998f1e1526fa84dfc2198b064f9f8e6ab6068 \
  --frame0-artifact-path configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260803/take_061_unit04_bh.frame0_exact.v1.json \
  --expected-frame0-artifact-sha256 ad17d984559776e90c70182ac4c0361c01de95094859e725162fb958defdbc54 \
  --plant-template-path "$PLANT_TEMPLATE_REL" \
  --expected-plant-template-sha256 "$PLANT_TEMPLATE_SHA" \
  --motion-path assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz \
  --expected-motion-sha256 aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e \
  --device cuda:<FREE_PHYSICAL_GPU> \
  --python /workspace/hope_isaac_venv/bin/python \
  --work-dir "$WORK_DIR" \
  --raw-nominal-output configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260804/take_061_unit04_bh.frame0_exact.raw_nominal_hold.v1.json \
  --output configs/action_ball_n1_measured_20260803/a211_frame0_exact_20260804/take_061_unit04_bh.frame0_exact.nominal_hold.receipt.v1.json
```

不要在两次 Kit boot 之间复用 `$WORK_DIR`，不要设置逻辑 `CUDA_VISIBLE_DEVICES` 后再把 `cuda:0`
误写成物理卡号。wrapper 在启动前和 live child 自然退出后各检查一次 exact clean checkout；任何
其他 tracked/untracked 文件都会阻断。raw live receipt 由 producer 直接 no-clobber 发布到上面的
repo-relative `--raw-nominal-output`；checkout 外只保留 probe input 与截图。失败会保留 raw FAIL
evidence，但不会铸造最终 PASS。

## 历史验收结果与证据边界

最终 receipt 必须同时满足：artifact file/content/first commit、probe source commit、plant template、
临时 probe input file/content 和 raw live receipt file/content 全绑定；raw evidence 本体嵌入 receipt；
五类安全 termination 均 active；无 terminal/truncation/reason；current/substep actual hard-edge 均为零；
最终 hard gap 为正；root/foot telemetry 有限；`raw reset / exact frame0 write / step1 / step10 / final`
五帧截图各有 SHA。

历史 exact Pod commit `ea8c7e1d` 在 policy step 9（`.18 s / 36` physics substep）触发
`robot_hit_table`，没有 current/substep actual-hard edge，只发布 raw FAIL，没有 PASS receipt。该结果
与后续 `0/73` 扫描共同否定 direct exact-frame0 physical birth；不再调用
`materialize_action_ball_a211_lineage.py` 消费本页输出。保留文件必须继续标
`diagnostic_unauthorized=true`。

## 当前结论

不要为寻找第74条 direct-frame0 candidate 重跑本工序，也不要关 table/fall/too-low、安全 termination，
改腰 qdes 或加入隐藏 connector。current launch 必须闭合 split-ready artifact/hold、5--25 tick WAIT、
measured-frame0 reveal 与 learned bridge 的同一 lineage；缺一就阻断。`200/800` 仍只是一份独立耐久
诊断，不写入 birth seal，也不阻断 first learnability long。
