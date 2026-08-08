# EXP-A3P-P1-0807-DUAL-ENGINE-MODEL-SET-20260807

Status: built and self-verified on host；**两个引擎都未在本机编译/导入过**
（2026-08-08 补记：Isaac 侧已在 Pod1 实跑，见文末"补记"——在那之前，Isaac 通路一直加载的是旧机器人）

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

---

# 补记 2026-08-08：Isaac 那条通路其实一直在跑旧机器人

## 人话

`82ee3ae8` 把 `AGIBOT_A3_ASSET_ROOT` 指向 0807，但 **Isaac 的 ActionBall 通路根本读不到这个常量**。
只要环境变量 `HOPE_AGIBOT_A3_USD_PATH` 被设上，`robots/agibot_a3.py` 就直接返回一份
预转换好的 USD，永远走不到 URDF 那一支；而所有 A211/C211 发射器都**强制**设这个变量
（不设直接拒收，不会退回 URDF）。那份预转换 USD 是 2026-07-17 从**退役的 0409 机器人**转出来的。

发射器有一枚 `A3_RUNTIME_USD_BUNDLE_SHA256`，六个文件的 sha256 全部命中——**门是活的，
但它证明的是"这份缓存没被人改过"，不是"这份缓存是当前机器人的缓存"**。任何机器人的 USD
都同样能通过。指纹在，语义没查。

## 三层查证：到底加载的是哪个机器人

**第一层，机制码**：`robots/agibot_a3.py:182` 一旦 `HOPE_AGIBOT_A3_USD_PATH` 有值就 `UsdFileCfg`
早返回。**第二层，缓存自陈**：那份 bundle 自己的 `config.yaml` 写着
`asset_path = .../assets/agibot_a3/urdf/model.urdf`，`Generated by UrdfConverter on 2026-07-17`。
**第三层，从字节读回来**——这一层才是决定性的，因为前两层都还是在读字符串。

用 `pxr` 直接打开两份 USD 缓存，和两副 URDF 逐项对：

| 读回来的量 | 0409 缓存 `1b3fecd7` | 0807 缓存 `13e5ecfe` | 能不能区分 |
| --- | --- | --- | --- |
| 关节数 | 31 | 31 | **不能** |
| 关节名集合 | 相同 | 相同 | **不能** |
| 关节限位（1e-4 rad 容差） | 0 处不符 | 0 处不符 | **不能** |
| 整机质量 | `58.277233 kg` | `57.600011 kg` | **能，差 0.677 kg** |

每份缓存与**自己那副** URDF 的整机质量差是 `1e-6 kg`，与**另一副**差 `0.677 kg`。
逐 body 看，32 个 body 里只有 6 个动了：`torso_Link −1.180906`、`left_wrist_yaw_Link +0.566262`、
双肘各 `−0.042`、双肩滚转各 `+0.010`——正好是厂商说的躯干重称加左夹爪折进腕部。

**要点**：Franco 建议的"dump 关节数/限位"这条路**抓不到这个 bug**——0807 的 ABI 是刻意
逐位兼容的。真正能分辨两副 plant 的是**质量**。以后再要问"加载的是哪个机器人"，问质量。

## 改了什么：从"字节完整性 pin"升成"身份 pin"

`launch_n1_reward_screen_diagnostic.py`（A211 与 C211 共用这一个基础模块）：

1. 六个字节摘要重切到 0807 缓存 `a3p0807_preconverted_usd_13e5ecfe`。**原来的强度一点没减。**
2. 新增 `_validate_a3_plant_identity`，七项比对，收据里逐项自陈比了什么、比出来是什么：

| 比对项 | 左边 | 右边 |
| --- | --- | --- |
| `live_spawner_asset_root_vs_pin` | 从 `robots/agibot_a3.py` 源码里读出的 `AGIBOT_A3_ASSET_ROOT` | pin 声明的资产包名 |
| `plant_receipt_manifest_type` | `configs/a3p_p1_0807_model_set_v1.json` 的 `manifest_type` | 评审过的那一种 |
| `plant_receipt_asset_root_vs_live_spawner` | 收据说的资产包 | 活代码 spawn 的资产包 |
| `plant_receipt_urdf_sha256_vs_pin` | 收据声明的 URDF 摘要 | 切 pin 时那份 URDF 的摘要 |
| `worktree_urdf_sha256_vs_plant_receipt` | **现场重新哈希** work tree 里的 URDF | 收据自己声明的摘要 |
| `bundle_config_asset_path_vs_plant_receipt` | 转换器自己记下的它读了哪个文件 | 收据指的那个文件 |
| `bundle_isaaclab_asset_hash_vs_rederived_from_worktree_urdf` | bundle 里的 `.asset_hash` | **用现场 URDF 字节重算出来的同一个值** |

最后一项是唯一一项**不是在比名字**的。IsaacLab 的
`asset_converter_base.py::_config_to_hash` 把 `.asset_hash` 定义成
`md5( json.dumps(转换配置去掉三个路径键) + 源文件字节 )`；我们离线照抄了这个算法
（不需要起 Kit，只要 `yaml`+`json`+`md5`）。它成立**当且仅当**这份 USD 缓存确实是用这份
URDF 的字节、在这套转换配置下转出来的。**这是派生证明，不是名字匹配**——就算有人把
`config.yaml` 里的路径字符串改成当前 plant，这一项照样过不去。

先在真实数据上验过这个算法对不对，再拿它当门：0807 缓存 × 0807 URDF 重算 = `676efde5…`
= 文件里存的值，**逐位相同**；换成 0409 URDF 立刻变 `6e10869e…`。

## 变异测试：粗一档就过不了

代码层（每次只削弱一条腿，跑 `test_launch_n1_reward_screen_diagnostic.py` 69 项）：

| 削弱掉的那条腿 | 结果 | 是哪一条测试抓住的 |
| --- | --- | --- |
| 去掉 `.asset_hash` 派生证明 | CAUGHT | `..._retired_plant_is_refused_even_when_restamped` |
| 去掉 config.yaml 路径核对 | CAUGHT | 同上 |
| 不再读活代码里的 plant 指针 | CAUGHT | `..._plant_pointer_moved_without_recutting...` |
| 相信收据声明的 URDF 摘要、不现场重算 | CAUGHT | `..._urdf_that_drifts_from_its_own_receipt...` |
| 不再核对收据与 pin 的切点一致 | CAUGHT | `..._receipt_urdf_digest_must_match_the_pin...` |
| **去掉原来的六个字节摘要** | CAUGHT（2 项） | 含 `..._byte_tamper_is_still_refused...` |
| 接受任意 plant 收据种类 | CAUGHT | `..._not_the_reviewed_model_set...` |

七条腿全部 load-bearing，**原来的字节完整性强度也还在**。

真实资产层（在 pod 上对真 bundle 跑，不是夹具）：

| 场景 | 结果 |
| --- | --- |
| 0807 bundle + 0807 work tree | **通过**，收据自陈七项全对 |
| 0409 bundle，六个摘要**全部重新盖章**（今天骗过我们的那个情形） | **REFUSED**：`bundle was converted from a different robot: config.yaml asset_path=…/agibot_a3/urdf/model.urdf` |
| 0409 bundle，摘要不动 | REFUSED（字节 pin 先开火） |

## 实跑

Pod1 GPU1，解释器 `/workspace/hope_isaac_venv/bin/python`（3.10.18），
worktree `/workspace/franco/plantid_20260808`，USD bundle
`/workspace/franco/runtime_assets/a3p0807_preconverted_usd_13e5ecfe`（08-07 22:11 从 0807 URDF 转出）：

| 格 | 阶段 | 结果 |
| --- | --- | --- |
| C0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off | materialize | `EXIT=0`，48 s |
| 同上 | recipe | `EXIT=0`，53 s |
| 同上 | **oracle32**（第一个真正走 env step 的阶段） | **`EXIT=2`，见下一节** |
| A0-base-safety-standard-init-sigma1p0-proprio-obs-noise-off | materialize | `EXIT=0`，42 s |
| 同上 | recipe | `EXIT=0`，52 s |

**这是 ActionBall 第一次真的在 0807 plant 上开机**——Kit 起来了、场景建出来了、发射门过了。
但要注意 `materialize` 与 `recipe` 的 `ppo_update_count=0`：它们**不走 env step**，
所以只证明"装配成功"，不证明"能跑"。

## 换完 plant 立刻撞上的第二枚同款 pin（本轮没修，因为它会改物理）

`oracle32` 是第一个真的 step 的阶段，它当场炸了：

```
hope_actions.py:4784 process_actions
  -> :3589 prepare_robot_table_pose_guard
  -> terminations.py:2037 _verify_loaded_runtime_usd_bundle
  -> terminations.py:170
RuntimeError: robot_hit_table live USD bundle file differs from pin: .asset_hash
```

`terminations.py:27-60` 里 `_A3_COLLISION_PROXY_RUNTIME_USD_TREE_SHA256` 和
`_A3_COLLISION_PROXY_RUNTIME_USD_FILES` 是**退役 0409 那份 bundle 的六个摘要的第三份拷贝**
（第一份在发射器，本轮已重切；第二份在 `scripts/materialize_a3_table_collision_proxy.py:54/57`）。
它在**运行时**、在桌面守卫构造时重新哈希活体 USD。ActionBall 一定走
`full_table_assembly=True`（`train.py:2164` 强制），所以这条路一定会踩到。

**换句话说：本轮把"悄悄跑错机器人"变成了"到第一步就大声拒收"。方向是对的，
但在下面这件事做完之前，ActionBall 的 step 类阶段（oracle32 / scale4096 / long）跑不了。**

为什么不顺手一起改：因为它不是接线，是物理。把
`materialize_a3_table_collision_proxy.py` 分别对两副 URDF 跑一遍（纯 CPU），
桌面碰撞代理的分量从 **43 变成 62**：

| | 数量 | 说明 |
| --- | --- | --- |
| 两副共有 | 42 | 局部 OBB 中心与半轴最大差 `1.1e-12 m`，mesh 字节**完全相同** |
| 只有 0409 有 | 1 | `left_wrist_yaw_Link <- left_hand_link`（旧的占位手） |
| 只有 0807 有 | 20 | `left_wrist_yaw_Link <- left_base_link / left_link1..left_link18`，即左手 OmniPicker3 夹爪 |

也就是说：**几何没重塑，是左夹爪的碰撞体积第一次进入桌面守卫的视野**，
同时旧占位手退场。`robot_hit_table` 的判定会因此变——这是要 Franco 过目的科学改动，
不是能夹在换 plant 里悄悄带过的。另外 `component_count == 43` 在
`mujoco_native/table_termination.py` 里是硬编码的，两个引擎都要跟着动。

顺便点清：退役 0409 那六个摘要在仓里一共**六份拷贝**——
`launch_n1_reward_screen_diagnostic.py`（本轮已重切）、
`scripts/materialize_a3_table_collision_proxy.py`、
`tasks/tracking/mdp/terminations.py`、
`configs/a3_table_collision_proxy_20260731/a3_table_collision_components.v1.json`（产物自陈）、
`configs/phase1_balance_action_slew_launch_manifest_20260720.json` 与
`configs/phase1_lower_body_stability_launch_manifest_20260720.json`（两条老队列的作业清单）。
前四份要连着动，后两份是历史作业记录、不动。

## 回归数字

同一棵 worktree（`/workspace/franco/plantid_base`，`aba8d96f`）先后各跑一次全量，
解释器 `/workspace/hope_isaac_venv/bin/python`（Python 3.10.18），
命令 `pytest hope_training/whole_body_tracking/tests -q -rs -p no:randomly -p no:cacheprovider --tb=no -rf`：

| | failed | passed | skipped | errors | 用时 |
| --- | --- | --- | --- | --- | --- |
| 改之前 | 122 | 7693 | 53 | 19 | 35:27 |
| 改之后 | 122 | **7701** | 53 | 19 | 34:38 |

`+8` 全是本轮新增的测试，**失败集合逐条相同**（两个方向的 `comm` 差都是空的），
skip 数与 error 数不变。那 122 个失败与 19 个 error 是 `aba8d96f` 上就有的，与本轮无关。

只跑发射器相关的十个测试文件时：干净基线 `8 failed, 644 passed`，改后
`8 failed, 652 passed`，**同样的 8 个**（都在
`test_launch_a3_vendor_identity_smoke.py` 与 `test_launch_n1_measured_vendor_v2_diagnostic.py`，
是 vendor identity bootstrap repin 的既有失败，不是本轮引入的）。

一处必须同批改的测试：`test_launch_n1_vendor_baseline_diagnostic.py` 里
把 `_B._validate_runtime_asset_environment` 换成零参 lambda，而基础模块的
`build_plan` 现在会带 `checkout=` 调它，所以改成 `lambda **_kw:`。

## 还没做 / 边界

- **动作库还是旧的**。拍心动了 `0.502 mm`，是 racket-FK 门槛的 5 倍，而
  `commands.py` 是把 `body_pos_w` 原样读进去而不是重新推的。本轮**只换 plant 不换库**，
  所以现在跑出来的 C0/A0 是"新机器人 + 旧目标"。要出训练结论必须先把库在 0807 上重解
  （`17f4bae7` 已经把 73 clip 的 measured 库在 0807 上重跑过，但接线是另一件事）。
- **`configs/a3p_p1_0807_model_set_v1.json` 里 `candidate_role` 仍写着
  `future_primary_successor_candidate_not_current_runtime`**。这句现在与事实不符了，
  但改它会动收据的 sha，而 `configs/a3p_p1_0807_pod_verification_v1.json` 等四处引用它，
  故意留给下一轮连同引用方一起改。
- 门只覆盖走 `launch_n1_reward_screen_diagnostic` 基础模块的通路
  （N1 reward-screen / A211 / C211 / measured-vendor-v2 / vendor-identity-smoke /
  materialize-n1-vendor-dynamic-ready）。A211 与 C211 调用时不传 `checkout=`，
  于是用的是"正在运行的这份代码所在的 checkout"；由于 `robots/agibot_a3.py` 与
  plant 收据都是 git 跟踪文件、且 `_verify_clean_source` 已经锁死 commit 与干净度，
  两棵树在同一 commit 上这两份必然逐位相同，**残余口子只剩 git-ignored 的 URDF**。
  要彻底封死就是在 A211/C211 的四个调用点补 `checkout=checkout`——本轮没做，
  因为这两个文件当时有别的 session 在改，不能夹带。
- 这道门现在要求**发射用的 checkout 里必须有 0807 URDF**（59 KB，不需要 mesh）。
  这是有意的：拿不出 plant 就不能声称在跑这个 plant。老 checkout 需要同步一次资产。
- 顺带核实的一件事，不是本轮造成的：基础模块自己那条 N1 reward-screen 通路**早就发不出去了**。
  `LEGACY_ROBOT_SOURCE_SHA256` 钉的是 `1fd2bf2d…`，而 `robots/agibot_a3.py` 现在是
  `a7a5da60…`（`82ee3ae8` 改过它），`_validate_runtime_sources` 会先拒。
  那枚 pin 的注释写明这是**有意的**——那个实验就是钉死在换 plant 之前的机器人上。
  A211/C211 不调 `_validate_runtime_sources`，所以不受影响。

## 同一形状的其他地方：钉了派生物，没钉它从哪派生

按"被钉的东西是不是派生物、门里有没有任何一项把它连回源"扫了全仓约 200 处 64-hex 常量
（`.claude/worktrees/` 除外）。**这类东西按源的关键词是搜不到的**——搜"urdf"只会漏掉，
判据必须是"pin 的对象是派生物"。三处成立：

| 处 | 钉的是什么派生物 | 门里做了什么 / 缺了什么 | 影响 |
| --- | --- | --- | --- |
| `mujoco_native/table_termination.py:96` `EXPECTED_COLLISION_PROXY_ARTIFACT_SHA256` | 43 个 OBB 桌面碰撞代理 `configs/a3_table_collision_proxy_20260731/…v1.json`（由 `materialize_a3_table_collision_proxy.py` 产出） | 查了文件 sha、schema、artifact_type、body_order、自封 `content_sha256`、43 个分量、有限性、覆盖率——**唯独没读它自己文档里就写着的 `source_urdf` 块**。同一份产物的 Isaac 侧门（`mdp/terminations.py:1004-1010`）**是查的**，这个不对称本身就是证据 | 决定 `robot_hit_table` 何时终止，**会改行为** |
| `scripts/materialize_a3_table_collision_proxy.py:54/57` `PINNED_RUNTIME_USD_*` | 预转换 USD bundle 的六个文件 | 查了内部一致性、软链、路径集、逐文件大小与 sha——**没和同一个脚本 200 行后才去哈希的 `DEFAULT_SOURCE_URDF` 比过**。URDF→USD 这一步是"声称"的，然后被写进一份两个引擎都信的**已跟踪**产物里 | 同一形状，但它是**写方**；而且这六个摘要就是**退役 0409 那份** |
| `tasks/tracking/stage1_natural_clip_contract.py:29` 三条 lane 的 `motion_sha256` | 重定向后的 `.npz` 动作片段 | 按摘要查表→取 `strike_phase`。**没开旁边的 `SOURCE_MANIFEST.json` / `BANK_IMPORT_RECEIPT.json`**，也没把 lane 声明的帧数/击球帧和 npz 里的实际内容对一遍 | `strike_phase` 进评分，会改行为；但上游 pkl 在仓外，只能查两份旁证收据 |

判为**不成立**（确实有源链）的例子，留在这里是为了说明扫描是真扫过的：
`mdp/terminations.py:26` 有 `_A3_COLLISION_PROXY_SOURCE_URDF_SHA256` 并在 `:1005` 真比；
`table_termination.py:79` 的 identity manifest 走 `verify_exact_mujoco_identity` 重新编译 MJCF；
`selected_rubber_classifier.py:333` 用 `np.allclose(proxy_outer, urdf_outer)` 从几何上验代理；
`action_ball_runtime.py:412` 的 `ARM_CATALOG_SHA256` 在 import 时现算；
`prepare_a3_p1_0803_31d_asset.py:1134` 不满足 URDF 与 closure 双条件就整份收据不发。
