# ChingMu 73 件源资产胶囊

这一步只把 Pod2 上的 73 件动作、逐件转换元数据和逐件球轨迹整理成一个可迁移、
逐字节绑定的源资产包。它不编译动作、不签发 motion admission，也不允许训练或真机。

> **2026-08-03 边界：**本页命令只能复现下面绑定的 legacy schema-2/FK
> source capsule，不能消费 corrected `chingmu73_measured_v4_20260803` 或新的 measured-channel
> 73 动作 manifest。严格重载已暴露旧 prototype 缺 `velocity_contract`；在工具升级为
> 无损携带 schema-v4 measured channels 和 schema-v2 prototype 前，禁止用本页的旧 PASS
> 代签 formal measured-racket authority。v4 当前也只有运动学准入，机械审计已发现
> 37/73 超速和 58/73 近限位反例。

## 已钉住的输入

- action-ball manifest：
  `action_ball_chingmu73_nomove_f10_20260728.json`
  (`fd7c087957ae4619d27d9510f97d7f5d52611af82829a54668b9fbb9b8806493`)
- build report：
  `action_ball_chingmu73_nomove_f10_20260728.buildreport.json`
  (`1db19a4a3a10ca0d72e5844abdc88c9db9410bfeb9d97f5ac4dfcc1c7d85601c`)
- ChingMu batch v1：
  `chingmu_manifest_v1.json`
  (`6a0f3e1c816ba2673953351cbbba54b28839e57c566c1c6ad97bbf271d065710`)
- 选择规则：batch v1 的 74 件按原顺序去掉
  `Take_085_unit00_FH`，得到固定的 73 件。

## Pod2 复现命令

输出目录必须事先不存在；工具用原子 no-clobber 发布，绝不覆盖旧包。

```bash
mkdir -p /workspace/codexschema/chingmu73_source_capsules
cd /workspace/codexschema/ballfirst_wip_20260728

python3 /path/to/nohope/hope_training/whole_body_tracking/scripts/materialize_chingmu73_source_capsule.py \
  --action-manifest configs/action_ball_chingmu73_nomove_f10_20260728.json \
  --expected-action-manifest-sha256 fd7c087957ae4619d27d9510f97d7f5d52611af82829a54668b9fbb9b8806493 \
  --build-report configs/action_ball_chingmu73_nomove_f10_20260728.buildreport.json \
  --expected-build-report-sha256 1db19a4a3a10ca0d72e5844abdc88c9db9410bfeb9d97f5ac4dfcc1c7d85601c \
  --batch-manifest /workspace/codexschema/chingmu_batch_20260728/chingmu_manifest_v1.json \
  --expected-batch-manifest-sha256 6a0f3e1c816ba2673953351cbbba54b28839e57c566c1c6ad97bbf271d065710 \
  --profile-root /workspace/codexschema/ballfirst_wip_20260728 \
  --batch-root /workspace/codexschema/chingmu_batch_20260728 \
  --motion-root /workspace/codexschema/ballfirst_wip_20260728/motions/chingmu73_20260728 \
  --ball-root /workspace/yikang/chingmu_retarget/chingmu_a3_units_v2/ball_ext \
  --output /workspace/codexschema/chingmu73_source_capsules/chingmu73_f10_exact_v1
```

`SOURCE_CAPSULE_RECEIPT.json` 必须报告
`PASS_SOURCE_INVENTORY_ONLY`，并明确保持：

- `motion_admission_present=false`
- `training_authorized=false`
- `deployment_authorized=false`
- `hardware_authorized=false`

## 工具核对的内容

逐动作检查：

1. action order、action id、family 与 batch v1 顺序一致；
2. WIP motion、batch motion 和两份 manifest 的 SHA-256 完全一致；
3. schema-2 key/order、50 Hz、31 joints、32 bodies、长度和 finite；
4. `hit_frame_50`、`strike_phase`、`t_hit`、`t_cycle` 指向同一击球帧；
5. `base_spawn_center = station_xy_hope + [0.5, 0.7625]`，且 no-move
   的全部 base travel 项为零；
6. `.meta.json` 的第一击帧、站位、球位置、来球和回球速度与 batch 一致；
7. `.ball.npz` 的 120 Hz、unit offset、第一击帧、击球点和 coverage 与 batch 一致；
8. 为此前未绑定的 73 份 metadata 和 73 份 ball sidecar 计算 SHA-256，
   并写入可迁移 receipt 路径。

batch 中的 retarget source PKL 仍由原有 SHA-256 记录约束，但不包含在这个
motion+ball 胶囊内；如需从动捕原料重放转换，还要单独迁移并重开这些 PKL。

## 尚未完成的 admission 链

这个胶囊仍只是 raw full-body source inventory。正式训练前还必须另行完成：

- 按[独立任意 N 动作 canonical bank 工序](run_arbitrary_motion_bank.md)重建
  content-bound recipe，并生成每个动作独立的 `upper` 和 `full` 输出；
- shared-ready、恢复段、速度端点、MuJoCo FK/动力学、grounded/table safety
  与逐动作 post-retime fitted-ball 物理回台验收；
- 完整的 73×2 `generic_v2` bank-gate report；
- registry、alignment、evidence、adoption 与 opaque motion-admission certificate。

在这些材料齐全之前，Pod2 只能做 CPU inventory 或独立小规模 canary，不能把
N=73 写成 N=93，也不能接入真机。
