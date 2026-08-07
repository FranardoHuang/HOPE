# EXP-A3P-P1-0807-DUAL-ENGINE-MODEL-SET-20260807

Status: built and self-verified on host；**两个引擎都未在本机编译/导入过**

Evidence: host 结构与 ABI 校验 + 收据可复现 + 变异测试；无 Isaac Lab、无 `mujoco`、无 `mujoco_warp`

## Question

2026-08-07 的 A3P-P1 重发交付，能否落成一套 Isaac 与 MuJoCo/`mujoco_warp` 都能直接导入的模型，
且不覆盖任何现役资产、不伪造几何、不把口头确认写成书面证据？

## 交付回应了什么

| 项目提出的问题 | 0807 结果 |
| --- | --- |
| `right_elbow_joint` x=0.001 破坏左右镜像 | ✅ 改回 `0.01`。与项目 v2 补丁的拍心只差 **5 µm** |
| 5 个 fixed 关节上的非法 `<axis>` | ✅ 全部删除 |
| 两个 `ankle_pitch` 的 ±1.5 mm 侧向不对称 | ✅ 归零（厂商主动修的，我们没提过） |
| 20 个夹爪碰撞网格 | ✅ 口头确认"碰撞=视觉" |
| 躯干 −1.181 kg / 双肘双肩质量 / 夹爪耦合 / changelog | ❌ 仍开放 |
| 重复 imu link / NaN visual / 82 处网格大小写 | ❌ 未修 |

**项目的 v2 镜像对称补丁就此退役**——它的值被源头确认。v2 保留在仓库里作为历史记录，不删。

## 交付形态：两块拼不上的拼图

0807 URDF 不含任何网格（63 KB）；`OmniPicker3-T1-0324-T1.5-close-ROS2` 只含 20 个夹爪视觉网格。
124 个引用里 **104 个只能回到 0803 包里取**（其中 82 个大小写不符），20 个夹爪碰撞仍不存在。
OP3 的 20 个网格与 0803 **20/20 逐字节相同**——不新增任何几何。

因此 [`intake_a3p_p1_0807_bundle.py`](../../../scripts/intake_a3p_p1_0807_bundle.py) 拼出的是
**project-assembled bundle**，不是 vendor closure，收据 `assembled_not_a_vendor_closure=true`
自陈这一点，并逐文件记录来源包与 SHA-256。网格落成 URDF 要求的精确大小写，Linux 上零失配。

- bundle：140 文件，closure `db90230a…7af6`
- 收据：[`configs/a3p_p1_0807_raw_intake_v1.json`](../../../configs/a3p_p1_0807_raw_intake_v1.json)

## Adopt / Reject

- **Adopt:** 夹爪碰撞按字节复制自视觉网格，复制后逐个 SHA 校验相等。授权来源是厂商口头确认，
  收据以 `written_evidence_on_file=false` / `channel=relayed_by_project_owner_from_vendor` 自陈。
  这取代 0803 的 collision-disabled 合同。
- **Adopt:** 九个夹爪坐标继续锁在 `q=0`。Isaac 侧改 `fixed` 让 importer 合并；MuJoCo 侧**不新增
  任何 body 或 joint**，质量并入 `left_wrist_yaw_Link`（`0.280678 → 0.846940 kg`，22 link 合并），
  20 个网格作为该 body 自身的 geom 挂在 q=0 位姿（0.102–0.2198 m，全在安装点外侧）。
  32-body / 31-actuator ABI 与 keyframe 宽度 38 全部不变。
- **Adopt:** 补上 `imu_in_pelvis_link` 的 20 g。现役 MJCF 把 `imu_in_torso` 并进躯干却漏了骨盆这个，
  而 Isaac 的 `merge_fixed_joints` 会并——这是先于 0807 就存在的 20 g 跨引擎不一致，一并修掉并声明。
- **Reject:** 原地改现役 `a3_pingpong.xml`（被 4 处 SHA 钉死）或现役 `assets/agibot_a3/`。
  派生前先校验现役 MJCF 的 SHA 未漂，输出路径命中任一现役目录直接拒绝。
- **Reject:** 把"结构校验通过"说成"能跑"。

## 结果

MJCF 相对现役的改动**只有几何**，统一 diff 123 行（+76 / −23）：

| 类别 | 数量 | 内容 |
| --- | --- | --- |
| `<body pos>` 真几何改动 | 4 | 两个 ankle_pitch 1.5 mm、right_hip_roll 1.1 mm、right_elbow 0.5 mm |
| `<body pos>` 坐标取整 | 4 | ≤0.0054 mm |
| `<inertial>` 重写 | 7 | torso −1180.936 g、left_wrist_yaw +566.262 g、双肘 −42 g、pelvis +20 g、双肩滚转 +10 g |
| 逐字保留 | — | armature / damping / frictionloss / 31 actuator / 33 凸包 / 球拍面代理 / 6 site / 155 sensor / 27 exclude / keyframe / 场景 |

7 个惯量的特征分解残差 `8.3e-17 ~ 4.3e-19`。

拍心 `q=0` 相对现役 0409：**`0.502087 mm`**（orientation 相同）。仍是 `racket_fk_ref.py:177`
门槛 `1e-4 m` 的 5 倍，**动作库仍必须在 successor 上重跑 audit**。

## 变异测试

| 变异 | 结果 |
| --- | --- |
| intake 收据里抹掉厂商确认 | BLOCKED |
| 现役 MJCF 字节漂移 | BLOCKED |
| bundle 与其 intake 收据不符 | BLOCKED |
| 输出路径指向现役 Isaac 资产 / 现役 MJCF 目录 | BLOCKED |
| 重复运行 prepare | BLOCKED（no-clobber） |

## 复现

```bash
python3 scripts/intake_a3p_p1_0807_bundle.py --check
python3 scripts/prepare_a3p_p1_0807_model_set.py --check
python3 -m pytest tests/test_a3p_p1_0807_model_set.py tests/test_prepare_a3_p1_0803_31d_asset.py -q
```

Observed 2026-08-07：intake `PASS`；model set `PASS`；`31 passed`（host py3.8）。

## 未验证边界

本机既无 Isaac Lab 也无 `mujoco`/`mujoco_warp`。收据里
`isaac_import_verified` / `mujoco_compile_verified` / `mujoco_warp_load_verified` /
`mujoco_identity_v3_minted` / `cross_engine_parity_verified` / `motion_bank_revalidated` /
`training_authorized` / `deployment_authorized` / `hardware_authorized` **全部为 `false`**。

上 pod 后的最小验证：

1. `mujoco.MjModel.from_xml_path` 编译新 MJCF，确认 `nq=38 / nv=37 / nu=31 / nbody=33`（含 world）；
2. `A3_PINGPONG_XML=<新 MJCF>` 走 `mjlab_lane`（`a3_plant_env.default_xml()` 不做 hash 校验，
   所以 GPU lane 不需要 identity v3）；
3. Isaac `convert_urdf.py --merge-fixed-joints`，确认 31 关节 / 32 body 且顺序与
   `configs/a3_runtime_body_order.txt` 一致；
4. 跨引擎 parity；
5. 动作库在新 plant 上重跑 full-phase audit。

identity v3 只有在要重新进入 CPU 侧四个 fail-closed 证据门
（`table_termination` / `mujoco_teacher_motion_fitted_ball_gate` /
`mujoco_action_ball_policy_fitted_gate` / `solve_chingmu_canonical_racket_full_phase`）时才必须，
而仓库目前**没有铸造 identity manifest 的入口**——只有测试里的 `_manifest_from_current` 辅助函数，
且 `compiled_mjb_sha256` / `compiler_toolchain_sha256` 只能在有 MuJoCo 的机器上现编译产生。
