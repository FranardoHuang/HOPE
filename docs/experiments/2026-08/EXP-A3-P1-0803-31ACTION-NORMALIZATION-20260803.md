# EXP-A3-P1-0803-31ACTION-NORMALIZATION-20260803

Status: partial；future-primary candidate 已 materialize，现役 runtime pointer 未切换

Evidence: E2 exact Pod IsaacLab import + 20-step finite diagnostic；successor safe-ready、全动作拍心
FK/重定向与 MuJoCo parity 仍未测

## Question

能否不覆盖现役 `agibot_a3/`，把 2026-08-03 A3-P1 交付确定性地投影成与现有
31-action policy ABI 兼容的独立 successor，同时不伪造左夹爪资产，并确保右拍心合同没有被
归一化过程移动？

## 2026-08-04 decision

- **Adopt:** 用户裁决 raw URDF 是该交付的几何 ground truth；workbook 中冲突的
  `left_OP3_joint` mount 不再拥有否决权。左夹爪不进入 policy action，因此项目明确拥有一个
  versioned training projection：九个 gripper movable coordinates 全锁在 raw-URDF `q=0`。
  这里的 `q=0` 是**项目锁定位**，不是 vendor neutral、硬件 home 或耦合模型声明。
- **Adopt:** 保留 21 个夹爪 link 的全部 delivered inertial、质量与 joint origin；夹爪子树质量
  `0.76626209416 kg`，normalized 63 unique-link 总质量 `57.60001015416 kg`。九个零位均在 raw
  limit 内。
- **Adopt:** 交付缺 20 个 gripper collision mesh。不得用 visual mesh 或自造 primitive 冒充；
  归一化只删除这 20 个确实无法解析的左夹爪 collision element，并把它登记成 project-owned
  collision-disabled contract。它允许右手训练 candidate bring-up，但不证明左手碰撞或真机 parity。
- **Adopt:** 删除 fixed joint 上无运动学意义、会阻塞 Isaac importer 的五个非法 axis；对四个
  含 `-` 的 mesh basename 产生原字节 `_` alias；删除唯一无 geometry 且颜色为 NaN 的 visual。
- **Defer:** 现役 pointer 切换、successor safe-ready、桌/自碰、同动作全相位拍心 FK/重定向、
  dynamics/fixed-tape parity 与 MuJoCo identity v3。它们通过前，`training_authorized=false`。
- **Reject:** 剪掉整只夹爪及其质量；伪造缺失 collision；把 project lock 写成 vendor neutral；
  因为右拍 local 合同相同就声称 successor world-FK 与现役相同；在 receipt 闭合前覆盖
  `agibot_a3/`。

## Raw and generated authority

- Raw source: `112 files / 57,803,270 bytes`，closure SHA-256
  `b1da6430fb20901ffd4fedbf60ee1cda452b12d25bd02f3816f359c24a47818f`；主 URDF
  SHA-256 `7dc98e48602036a93d1e7492f7632d30d71477743e76559730637d9f95151704`。
- Tracked producer: [`prepare_a3_p1_0803_31d_asset.py`](../../../scripts/prepare_a3_p1_0803_31d_asset.py)，
  SHA-256 `058f7d50…a0eb0`。
- Tracked exact diff/closure receipt:
  [`a3_p1_0803_31d_v1.json`](../../../configs/a3_p1_0803_31d_v1.json)，schema 2。
- Ignored generated output:
  `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3_p1_0803_31d_v1/`。
- Output: 63 links / 62 joints / 31 movable joints / 104 mesh references / 100 unique meshes；
  URDF SHA-256 `2f15df8a…2535`；101-file closure SHA-256 `73a47e85…8f08`。

`scripts/prepare_a3_p1_0803_31d_asset.py` 默认 no-clobber：版本化 output 已存在就拒绝；输出路径若是
现役 `agibot_a3/` 或其子目录也在写入前拒绝。`--check`（只核验，不写文件）同时重算 raw closure、
normalization diff、31-joint order、右拍合同、output closure 与 manifest。

## Racket-centre result

拍心的 local contract 没有漂移：normalized/现役的 `pingpang_red_joint` 是

```text
parent = right_hand_pingpang_Link
xyz    = [0.21021, 0.032078, 0.032036] m
rpy    = [0, 0, 0]
```

raw 中同一 parent/child 只采用交付的小写 `_link` 拼法；归一化的名称映射不改变 transform。
这里的“拍心”沿用现役 `official_racket_site` 工程控制点，不冒充红面面积几何质心；二者的
面内距离约 `1.264 mm`，详细定义仍只认
[`racket_contact_geometry.md`](../../interfaces/racket_contact_geometry.md)。

rigid hand、red、black 与 centre-marker 四个 mesh 的 SHA 也逐一相同。Normalized 对 raw 的共同
`q=0` 拍心 world transform 是 position/orientation exact `0/0`；更强的依据是归一化保留了整个
右臂 chain 的 joint origin/axis/limit 与 link inertial，所以对所有共同 31-joint `q`，normalized
就是 raw successor 的拍心 FK。

但 successor **不是**旧 plant 的同一个 world-FK：0803 raw 将 `right_elbow_joint` origin 从现役
约 `[0.010, 0, -0.133]` 改成 `[0.001, 0, -0.1325]`。在共同 `q=0`，successor 与现役拍心位置差
`9.013878 mm`，orientation 相同。因此现有 measured-v4 动作只证明旧 URDF；切 successor 前必须
在新 URDF 上重跑 full-clip measured-site→FK/retarget audit。不能用 local site exact 代签该 gate。

## Pod evidence boundary

项目 lock/collision 裁决没有改变 generated output bytes；closure 仍是此前 Pod1 实测的
`73a47e85…8f08`，所以原 exact importer receipt 仍只证明同一字节：

- IsaacLab converter 正常退出；31 articulation joints 与 runtime order exact；
- merge-fixed 后 32 个 required runtime bodies，missing/extra=`0/0`；
- 20 physics steps 全 finite，初始 q/qdes 在 imported hard limit 内；
- 现役 `agibot_a3/` pointer 未改。

它不证明 safe-ready、table/self collision、拍心 world-FK/Jacobian、训练可学性或真机。尤其缺失
夹爪 collision 的 project contract 是明确的模型简化，不是物理完整性证书。

## Reproducible host checks

```bash
python3 scripts/prepare_a3_p1_0803_31d_asset.py --check
python3 -m pytest -q tests/test_prepare_a3_p1_0803_31d_asset.py
python3 -m py_compile scripts/prepare_a3_p1_0803_31d_asset.py
```

Observed 2026-08-04: `PASS`；`6 passed`；producer compiles。

## Promotion boundary

这项改动不阻塞当前右手训练：现役训练仍消费原来的 `agibot_a3/`，没有被热切。若要把 0803
candidate 升为 future primary/pre-long plant，必须在 exact closure 上补：

1. 当前 threshold-first birth/safe-ready 与后续 durability receipt；
2. 为 successor exact URDF/USD 重生目前硬钉旧 SHA/43-OBB geometry 的 table-collision proxy，
   并明确 left-gripper termination coverage；同时确认 collisionless gripper simplification 风险；
3. 新 URDF 上 Take061 及候选动作的 full-phase official-site position/orientation/velocity/FK audit；
4. same-state old/new dynamics 差异解释与 fixed-tape RNG/reason/counter/safety parity；
5. MuJoCo identity v3 与 cross-engine parity。

全部闭合前，manifest 保持 `canonical_runtime=false / training_authorized=false /
deployment_authorized=false / hardware_authorized=false`。

## 2026-08-06 v2 decision: the 9 mm is a delivery defect, not a design change

上一版把 `right_elbow_joint` 的 `9.013878 mm` 记成 vendor 的真实几何改动。**这个判定是错的**，
本节推翻它并落成 v2。

**人话**：这次交付表面上是"左手换成 OP3 夹爪"，但它其实是整条基线换代（robot name 从
`0000014503_A3T2.5-URDF-std-pingpang-0409` 变成 `A3-P1-URDF-std-0717`），顺带把右肘的一个
坐标写错了一个零。

### 四条独立证据

1. **零件没改**（最硬的一条）：交付与现役共有的 44 个 mesh **全部 SHA-256 逐字节相同**，含
   `right_shoulder_yaw_Link` —— 它就是定义这个 origin 的父件、带着肘部安装面。安装面挪 9 mm
   而父件 CAD 一个字节没变，物理上讲不通。
2. **交付自己的惯量说这两个肘是同一个件的镜像**：`left_elbow_link` 与 `right_elbow_link`
   质量完全相等（Δ = `0.000 g`），质心在 y-镜像下残差 x 方向 `0.0745 mm`、总体 `0.19 mm`。
   零件级镜像残差是 0.1 mm 量级，而两者 mount origin 的 x 差 `9 mm` —— **90 倍以上**，
   单看 x 分量是 **120 倍**。
   （注：两件的三角剖分不同，顶点数不等，所以这条依据的是交付的惯量张量，不是逐点比网格。）
3. **对称**：同一份交付里 `left_elbow_joint` 仍是 `x=0.01`，只有右肘是 `0.001`；现役 plant
   两边都是 `0.01`。新版是**造出**不对称，不是修正不对称。
4. **装配（旁证）**：把 `right_elbow_Link` 按候选 origin 放到 `right_shoulder_yaw_Link` 上量
   AABB 重叠 x，左臂基准 `76.221 mm`；右臂 `x=0.01` 得 `75.353 mm`（差 0.87 mm），右臂交付值
   `x=0.001` 得 `84.353 mm`（差 8.13 mm）。四对上肢零件的 AABB 左右镜像容差实测 `1.26 mm`。

同一方法确认 `z` 的改动是**真修复**：右臂 `z=-0.1325` 的重叠 `75.979` 比 `z=-0.133` 的 `75.479`
更贴近左臂的 `76.129`。旧 plant 的右 `z=-0.133` / 左 `z=-0.1325` 才是错的。

### Adopt

- **Adopt:** 铸 `v2`（`assets/agibot_a3_p1_0803_31d_v2/` + `configs/a3_p1_0803_31d_v2.json`），
  = v1 + 一条声明式 mirror-symmetry 覆盖：`right_elbow_joint` `x` 恢复 `0.01`，保留交付的
  `z=-0.1325`。v2 的 URDF 与 v1 逐行 diff **恰好两行**（robot name + 那一个 origin），
  100 个 mesh 逐字节相同。
- **Adopt:** 覆盖是 `provisional_pending_vendor_confirmation`。交付自带的 joint workbook
  **也写 `0.001`**，说明缺陷在上游 CAD 而非 URDF 导出，因此项目本地打补丁**不免除上报义务**。
  manifest 以 `reproduces_delivered_joint_origins_exactly=false` 与
  `mirror_symmetry_correction_vendor_confirmed=false` 自陈。
- **Adopt:** 其余交付差异**全部原样接受**，不动：`right_hip_roll_joint` 的 `x -0.0011 → 0`
  （真对称化修复）、五个 link 的质量/质心/惯量（torso `-1.181 kg`、两个 elbow `→0.670`、
  两个 shoulder_roll `→0.901`）。
- **Adopt:** v1 目录与其同字节 Pod import receipt **冻结**。`require_isolated_successor_root`
  现在硬拒写入 v1，`pod_import_verified` 对 v2 求值为 `false`，v2 的 `pod_import_receipt` 为
  `null`、`status=host_static_candidate_pod_import_pending`。v1 的证据不继承。

### Reject

- **Reject:** 直接改 v1、或让 v2 复用 v1 的 Pod receipt。
- **Reject:** 把五个 link 的质量改动或 `right_hip_roll` 也"修"掉 —— 没有证据支持，只有 elbow x
  有三条独立证据。
- **Reject:** 因为打了补丁就不向厂商提问。

### 后果

拍心 `q=0` 偏移（orientation 三者全同）：

| 对比 | position delta |
| --- | --- |
| raw 交付 vs 现役 | `9.013878 mm` |
| v2 vs raw 交付 | `9.000000 mm`（= 声明的覆盖量） |
| **v2 vs 现役** | **`0.500000 mm`** |

`0.5 mm` 仍是 `racket_fk_ref.py:177` PASS 门槛 `1e-4 m` 的 5 倍，也是重定向自身 full-phase p95
`6.403e-05 m` 的 7.8 倍。**动作库仍必须在 successor 上重跑 audit**；变化的是量级——占拍面半宽
从 `14%` 降到 `0.77%`，所以"确定要重新重定向"降级为"要先量、大概率不用重做"。

### 护栏与变异测试

`apply_mirror_symmetry_corrections` 在改任何字节之前逐条断言前提：被改关节的交付 `xyz`、
交付 `rpy`、镜像基准关节的 `xyz`、只动哪一个分量、幅度是多少。实测变异：

| 变异 | 结果 |
| --- | --- |
| 厂商重出 `x=0.002` | BLOCKED（premise drifted） |
| 厂商自己修好 `x=0.01` | BLOCKED（premise drifted，需人工撤销覆盖） |
| 厂商改 `z=-0.133` | BLOCKED |
| 镜像基准 `left_elbow` 漂到 `0.02` | BLOCKED |
| `right_elbow` rpy 变 | BLOCKED |
| 合同偷偷多改 `z` 却仍声明只改 `x` | BLOCKED |
| 合同谎报 `correction_m=0` | BLOCKED |
| 交付里偷改 `right_knee` origin | BLOCKED（`verify_raw_closure` 层，`total_bytes` 不符） |
| 写进 v1 / 现役 `agibot_a3` | BLOCKED |

注意第二行：**厂商把它修好了，脚本也会拒绝**。这是有意的——覆盖必须被人工撤销，不能静默
变成 no-op。

### 复现

```bash
python3 scripts/prepare_a3_p1_0803_31d_asset.py
python3 scripts/prepare_a3_p1_0803_31d_asset.py --check
python3 -m pytest tests/test_prepare_a3_p1_0803_31d_asset.py -q
```

Observed 2026-08-06：`PREPARED` closure `0cb41604…b221`、URDF `4dc4ee9d…76d6`；`--check` `PASS`；
`17 passed`（host py3.8）。

### 仍未闭合（v2 不改变这些）

MuJoCo 侧仍完全未动：`a3_pingpong.xml` 仍钉在旧 plant，需手改 `right_elbow_Link` 的 `z`、
`right_hip_roll_Link` 的 `x`、五个 `<inertial>`，再铸 identity v3 并跑 cross-engine parity。
`racket_fk_ref.py:67` 与 `pp_racket_fk.hpp:91` 两处手抄 elbow 常量仍是旧值，且
`pp_parity_test.cpp` 的 golden 由 Python 侧同一常量生成，两边一起漂仍会 PASS —— 改常量时必须
连这个失明的 parity 测试一起修。厂商待答问题见
[0803 交付核实清单](../../interfaces/a3_p1_0803_vendor_questions.md)。
