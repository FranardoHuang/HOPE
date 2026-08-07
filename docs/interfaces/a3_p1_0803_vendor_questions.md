# A3-P1-32dof-0803-BerkeleyPingpang-90deg 交付核实清单

Status: **partially resolved by the 2026-08-07 re-delivery** —— 问题 1 与 3a 已闭合，问题 2、3b 仍开放

> **2026-08-07 更新**：厂商交付了
> `A3P-P1-32dof-0807-OP3+pingpang_20260807_083135.urdf` + `OmniPicker3-T1-0324-T1.5-close-ROS2`，
> 并口头确认"夹爪碰撞几何等同视觉几何"。逐条状态见下方 §0。

Delivery (0803): `vendor_assets/agibot/A3-P1-32dof-0803-BerkeleyPingpang-90deg/`（112 files /
57,803,270 bytes，closure SHA-256 `b1da6430…7818f`，主 URDF SHA-256 `7dc98e48…51704`；
git-ignored，intake 收据在 [`configs/a3_p1_0803_raw_intake_v1.json`](../../configs/a3_p1_0803_raw_intake_v1.json)）

**这份清单只列"必须厂商回答"的问题。**其余交付缺陷项目已自行处理并留了收据，见
[0803 31-action 归一化记录](../experiments/2026-08/EXP-A3-P1-0803-31ACTION-NORMALIZATION-20260803.md)。

---

## §0 逐条状态（2026-08-07）

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | `right_elbow_joint` x=0.001 破坏左右镜像 | ✅ **已修**：0807 交付为 `0.01`，与项目 v2 补丁只差 5 µm（坐标取整）。项目补丁随之退役 |
| — | 5 个 fixed 关节上的非法 `<axis>` | ✅ **已修**：0807 为 0 个 |
| — | 两个 `ankle_pitch` 的 ±1.5 mm 侧向不对称 | ✅ **已修**（厂商主动修的，我们没提）：0807 两侧均为 0 |
| 2 | 躯干 −1.181 kg 而外壳网格逐字节未变 | ✅ **已答（2026-08-07，口头）**：之前**材质设错**，后来**重新称重修正**。见下方 §2 |
| 3 | 双肘 →0.670、双肩滚转 →0.901 | ⚠️ **推定同源**：与躯干同属那次重新称重修正，但厂商未逐条确认这四个 link |
| 4 | 夹爪耦合模型 / 中立位 / 真实限位 | ❌ **仍开放**：0807 的 `<mimic>` 仍为 0，8 个手指关节仍是 `±2 / effort=1 / vel=1` 占位值 |
| 5 | 20 个夹爪碰撞网格未交付 | ✅ **已答**：口头确认"碰撞=视觉"，项目据此按字节复制生成并留收据。**仍缺书面确认** |
| 6 | changelog / armature / SKU 确认 / "90deg" 含义 | ❌ **仍开放**：0807 同样没有 README 或变更记录 |

**0807 新引入的问题**：交付被拆成两块——URDF 不含任何网格，OmniPicker3 包只含 20 个夹爪视觉网格。
124 个网格引用里 104 个只能回到 0803 包里找（且 82 个大小写不符），20 个夹爪碰撞仍不存在。
**请下次把一个自洽、可直接导入的完整包一次发全**：URDF + 其引用的全部网格，文件名大小写与 URDF 内的
引用逐字符一致。

---

## 前提：这次交付不是"只换左手"

对比现役 plant，两份 URDF 的 `robot name` 是：

| | 现役 | 本次交付 |
| --- | --- | --- |
| robot name | `0000014503_A3T2.5-URDF-std-pingpang-0409` | `A3-P1-URDF-std-0717` |

产品线 A3T2.5 → A3-P1，基线日期 0409 → 0717。左夹爪只是其中最显眼的一项，交付同时带来了
躯干与双臂的质量改动。下面三个问题都源于此。

---

## §2 躯干质量的答复（2026-08-07，已闭合）

**厂商答复**：之前**材质设错了**，后面**重新称重做了修正**。

**人话**：`17.44497063 → 16.26406424 kg` 不是拆件，是 CAD 里材质密度填错导致的估值偏高，实测称重后
改回来了。这解释了为什么外壳 `torso_Link.stl` 逐字节没变——**几何本来就没动，动的是材料属性**。

对我们的影响：新值是**实测值，优于旧的 CAD 估值**，直接采信。MuJoCo 侧 `torso_Link` 的
`<inertial>` 已按此重算（`17.465 → 16.284064`，含并入的 `imu_in_torso` 0.02 kg）。
按体重归一化的力矩阈值随之变化约 1.1%，切换 plant 时要一并重算。

> ⚠️ 这是**口头答复**，无书面证据。同一次重新称重**很可能**也覆盖了双肘（`0.71204442/0.71187773
> → 0.670`）与双肩滚转（`0.89048605/0.89016898 → 0.901`）—— 它们同样是"网格未变、质量变整数"的
> 形态。但厂商没有逐条确认这四个 link，所以 §3 仍标为推定。

---

## 问题 1 — 躯干少了 1.181 kg，但外壳网格逐字节没变（**已由 §2 闭合**）

`torso_link` 质量 `17.44497063 kg → 16.26406424 kg`（`-1.180906 kg`，`-6.8%`），质心移动
`3.848 mm`，惯量最大分量变化 `6.892e-03`。

但 `torso_Link.stl` 与现役**逐字节相同**（SHA-256 一致）。外壳一点没动而质量掉了 6.8%，只能
是内部件变了。

**请说明：0409 → 0717 之间躯干内部拆掉/替换了什么件（电池？线束？配重？）。**

这直接影响 MuJoCo `a3_pingpong.xml` 的 32 个硬编码 `<inertial>`、整机重心与所有按体重归一化的
力矩阈值，我们需要知道它是设计变更还是数据修正。

---

## 问题 2 — 双肘、双肩滚转的质量被改成整数，且强制左右相等

| link | 现役 | 本次交付 |
| --- | --- | --- |
| `left_elbow_link` | `0.71204442` | `0.67000000` |
| `right_elbow_link` | `0.71187773` | `0.67000000` |
| `left_shoulder_roll_link` | `0.89048605` | `0.90100000` |
| `right_shoulder_roll_link` | `0.89016898` | `0.90100000` |

原本是两个不同的 8 位小数（CAD 估值的正常形态），现在变成同一个 3 位整数值。但质心与惯量
仍是 8 位小数且左右不同。四个 link 的 mesh 同样**逐字节未变**。

**请说明：这四个质量是实测称重值、设计目标值，还是换了材料/密度重算的？**

如果是实测值我们直接采信；如果是设计目标值，我们需要知道公差。

---

## 问题 3 — 左夹爪：缺 20 个碰撞网格，且没有任何耦合权威

### 3a. 20 个碰撞网格未随包交付

URDF 引用但交付里不存在（与文件名大小写问题无关，这 20 个在任何大小写下都不存在）：

```
base_link_collision.stl
Link1_collision.stl   Link2_collision.stl   Link3_collision.stl
Link4_collision.stl   Link4-1_collision.stl Link6_collision.stl
Link7_collision.stl   Link7-1_collision.stl Link8_collision.stl
Link9_collision.stl   Link10_collision.stl  Link11_collision.stl
Link11-1_collision.stl Link13_collision.stl Link14_collision.stl
Link14-1_collision.stl Link15_collision.stl Link17_collision.stl
Link18_collision.stl
```

对应的**视觉网格 20 个一个不缺**，所以形状是有的。项目当前的处理是显式禁用这 20 个 collision
element 并登记为 collision-disabled 合同 —— 它允许右手训练 bring-up，但不证明左手碰撞或真机
parity。**请补发这 20 个碰撞网格。**

### 3b. 九个夹爪自由度没有任何耦合/中立位信息

交付里 `<mimic>` 数量为 0，`<transmission>` 为 0，平行四连杆已被剪断
（`Link4-1`/`Link7-1`/`Link11-1`/`Link14-1` 成为终端固定桩）。八个手指转动关节的限位是
清一色的占位值：

```
lower=-2  upper=2  effort=1  velocity=1     （八个全部相同）
left_joint1 (prismatic)  lower=0  upper=0.0175  effort=1  velocity=1
```

原样导入仿真，手指会自由甩动。

**请提供：夹爪的耦合模型（哪些关节被哪个主动自由度驱动、传动比）、中立位/home 位、以及八个
手指关节的真实限位。**

另外，文件夹名写 `32dof`，但 URDF 里是 40 个可动关节（31 主体 + 9 夹爪）。若 `32 = 31 + 1`
成立，则需要上述耦合模型才能自洽。**请确认 32 这个数字的定义。**

---

## 附：项目已自行处理、仅作知会的交付缺陷

以下不需要厂商为本次交付返工，但**建议在导出工具里修掉，避免下一版重复**：

| 缺陷 | 位置 | 项目处理 |
| --- | --- | --- |
| `right_elbow_joint` `x=0.001` 破坏左右镜像（左肘同版仍是 `0.01`），且 joint workbook 同样写 `0.001`，说明错在上游 CAD | URDF L584 | 已在 v2 声明式覆盖回 `0.01`，**provisional，等本清单回复** |
| `imu_in_pelvis_link` 声明两次（两份不同：一份碰撞复用视觉网格，一份用专用碰撞网格） | URDF L793 / L821 | 保留带专用碰撞网格的那份 |
| `left_base_footprint` 的 `<visual>` 无 `<geometry>`，颜色为 `nan nan nan nan` | URDF L1256 | 删除该 visual |
| 78 个 mesh 引用大小写与磁盘文件不符（URDF 写 `_link.stl`，磁盘是 `_Link.stl`）；Linux 上解析不了 | 全文 | 按交付磁盘文件名重写引用 |
| 5 个 `fixed` 关节带非法 `<axis>`（非三向量），会挡住 Isaac importer | `torso_shell_joint` / `imu_in_torso_joint` / `imu_in_pelvis_joint` / `left_knee_shell_joint` / `right_knee_shell_joint` | 删除该 axis |
| 4 个带连字符的 mesh 名（`Link4-1` 等）USD 不安全 | meshes/ | 生成下划线别名，字节不变 |
| 44 个 `*_collision.stl` 与对应视觉网格 SHA-256 完全相同（零简化，全身 306,611 个碰撞三角形） | meshes/ | 仿真侧用项目自有凸包 |
| link 表与 PDF 的总质量 `57.62001 kg` 多算了重复的 imu link `0.02 kg`，真值 `57.600010 kg` | xlsx / pdf | 按去重后计算 |

## 附：顺便想要的（非阻塞）

1. **changelog**：0409 → 0717 的变更说明。交付里 112 个文件没有任何 README 或变更记录。
2. **各关节 armature / 转子惯量 / 减速比**。这是电机+减速箱属性，URDF 装不下，交付也没给。
   仓库现有的 9 个值来路是智元 parkour 表，且存在两套精度不一致的副本
   （`0.06646569891` vs `0.066472`，后者不是前者的四舍五入），需要权威值来收口。
   joint workbook 里的 `rpm` 列是关节侧不是电机侧，推不出减速比。
3. **确认执行器 SKU 未变**：31 个主体关节的 `effort`/`velocity` 与现役 31/31 精确相等，强烈
   暗示未换型，但那是重述不是重测。
4. **文件夹名里的 `90deg` 指什么**。四个球拍 mesh 与四个球拍关节 origin 跟现役逐字节相同，
   球拍没有旋转。交付里唯一的 90° 是 `left_OP3_joint` 的 pitch `-1.5707963 rad`，我们推断指
   夹爪安装法兰，但无从证实。
