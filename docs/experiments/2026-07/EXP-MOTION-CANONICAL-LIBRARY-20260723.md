# EXP-MOTION-CANONICAL-LIBRARY-20260723

Status: `in progress / probe2 rejected and rebuild pending / 0 of 10 training-authorized / no training or hardware launch`

Owner: Franco（拍板与复核）

Executor: Codex（07-23/24 白天）→ Fable 接力（07-24 晚续建）；使用 2026-07-23 CloudCode
产物作为被审计输入，不继承其结论

本实验从第一性原理重做 Franco 五动作库：先得到从共同等待状态直接出发、经过完整击球机会、再回到
共同等待状态的可播放候选，然后逐门判断它们是否可执行、能否回球、是否改善平衡。

术语见 [`canonical ready`](../../DEFINITIONS.md#canonical-ready)、
[上肢/全身 scope](../../DEFINITIONS.md#motion-body-scope)、
[`adv2c3`](../../DEFINITIONS.md#adv2c3)和
[动作归档（NPZ）、MuJoCo 模型描述（MJCF）、机器人描述（URDF）、正向运动学（FK）和
TOPP](../../DEFINITIONS.md#动作库术语)。现行接口以
[动作预处理合同](../../interfaces/motion_preprocessing_contract.md)为准；诊断/候选区别见
[`motion artifact class`](../../DEFINITIONS.md#motion-artifact-class)，整桥与窗口内容绑定分别见
[`canonical-ready contact bridge receipt`](../../DEFINITIONS.md#canonical-ready-bridge-receipt)和
[`protected-window digest`](../../DEFINITIONS.md#protected-window-digest)。定义表中 `adv2c3`
已统一为比较项口径。

## 1. 本轮纠正的五个问题

1. **窗口不是静止区。** 击球动作是完整连续动作，路径可以一路加速到击球机会末端，再在随挥段
   刹车。窗口标记点不锁姿态、速度或加速度。
2. **ready 不是前置小动作。** 禁止先从 ready 走到旧 frame 0，再播放源动作。所有动作都要重新
   搜索 `shared ready → selected core/window → shared ready` 的整条路径。
3. **`adv2c3` 未被证明最优。** 它只是固定 2/3 前奏切片，没有入口/出口搜索、ready 融合或完整
   重定时；现在只保留作比较项。
4. **正手挡不是单腕 180 度。** 新求解共同改变右肩、肘和腕七关节，保留拍点路径并翻转带符号拍面
   法向；upper/full 必须分别求解。
5. **五动作不是全台移动库。** 它们是以骨盆局部站位为参照的原地护台动作。locomotion（移动规划）
   负责把机器人运到局部站位；组合后仍须重新验证。
6. **face core 不是候选。** scope-specific 换面窗口只属于 diagnostic；必须重新编译
   canonical-ready→contact→ready 全桥，并由独立 verifier 绑定 source/output 窗口，才可能获得
   `compiler_candidate`。
7. **当前 ready v1 不是拍面中立位。** 它只是已登记的 `bh_loop_c` frame-0 donor baseline；
   exact FK 显示它明显更靠近反手拍面。下一候选要保留站姿/拍心设计目标、重解右臂成 face-neutral，
   以新 identity 重跑十件，不能覆盖 v1。

这里没有“维持已冻结规范”的模糊选项。继续保持不变的只有可核对事实：URDF/MJCF 模型、具名关节/
body 顺序、source/ready 哈希、安全 Gate、失败封闭和 Franco 的训练/硬件授权权力。旧的固定切片、
窗口静止、单轴翻面和 ready→frame0 串行都不是保留项。

## 2. 目标输出

五个高层动作是：

```text
fh_loop
bh_loop_c
fh_block_syn
bh_block
s0_highpress
```

每个动作目标各生成 upper 和 full 一份 exact schema-2 output，共十份；只有独立 verifier 通过后才
接受为 candidate。compiler/recipe manifest 显式要求：

```text
artifact_class=canonical_compiler_output
publication_class=compiler_candidate
training_authorized=false
hardware_authorized=false
```

十份候选进入 bank registry 时，registry row 的 `deployment_authorized` 也必须为 false。
其中 `artifact_class` 说明“这是什么工件”，`publication_class` 说明“它在单向发布链哪一级”；
diagnostic face core 没有后者，两个字段不能靠改名互换。

候选 recipe 为 `configs/canonical_motion_library_v2_20260724.json`。它钉住 source、ready、模型、
body order、窗口 seed、scope 规则和 post-build gates。`bh_loop_c` 仍是五动作槽中的反手拉；
`bh_loop_b` 只是宽窗口挑战者，`bh_loop_a` 只是离线挑战者。

这份当前 recipe 钉住的是下方 v1 donor baseline，所以只能复现当前 probe，不能在
face-neutral audit 失败后继续作为最终十件的发布配方。新 ready 通过独立预检后，必须形成钉住其新
path/hash/identity 的新版 recipe，并从零重编十件；不能原地偷换 ready bytes 而保留旧 recipe 身份。

当前 ready provenance 的固定事实是：

```text
ready NPZ SHA =
cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0
donor = bh_loop_c source frame 0
donor source NPZ SHA =
d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba
```

这些值只登记 baseline，不等于 final ready。exact FK 的
[`face-neutral ready audit`](../../DEFINITIONS.md#face-neutral-ready-audit)给出 upper
`ready→bh f44 / ready→flipped fh = 33.4/146.6 deg`，full 为 `32.9/147.1 deg`；v1 明显偏反手，
face-neutrality 失败。下一候选必须保留站姿/root/拍心设计约束、重解右臂，写新 sidecar/hash 并重跑
十件；旧 v1 bytes/hash 只读保留，不能覆盖或追认。

ready 是 9-key 单状态旁车，不是 schema-2 motion。runtime-float32 pose digest、exact-zero velocity
digest、ready-FK SHA 和 bank 级
[`registry shared-ready digest`](../../DEFINITIONS.md#registry-shared-ready-digest)仍须由十件 output
与独立 verifier 共同给出；现在没有被接受的值，因此不能把上述文件 SHA 单独当成 common-ready Gate。

## 3. 已实现的管线

“已实现”只表示本功能分支已有相应代码入口，不表示它满足本页全部合同或已经发布资产；probe2 的
strike-first、50 Hz no-brake、sidecar 和 dynamics 失败见 §7.1。

| 模块 | 当前已实现能力 | 明确不声称 |
| --- | --- | --- |
| `canonical_motion_recipe.py` | 内容绑定 recipe、source、ready、模型与五乘二输出矩阵 | 下游 Gate |
| `canonical_body_scope.py` | upper 的完整骨盆旋转折腰；full 的原子 root 对齐 | 平衡、接触 |
| `canonical_face_manifold.py` | 七关节、全窗口、多候选图回溯的拍面翻转 | 碰撞、回球 |
| `canonical_motion_geometry.py` | 枚举入口/出口并建 direct ready-to-ready 二阶连续路径 | 可学性 |
| `canonical_path_topp.py` | 位置/速度/显式加速度约束，并用 [no-brake time law](../../DEFINITIONS.md#no-brake-time-law) 硬卡窗口末前标量路径不减速的运动学初始解 | 拍速单调、执行器恒扭矩、数学 C3 |
| `canonical_torque_path_topp.py` | 固定基座直驱力矩核、窗口末前不提前刹车 | 站地自由根认证 |
| `canonical_root_pose_codec.py` | full root 姿态/速度的连续坐标编解码 | 支撑力 |
| `canonical_schema2_builder.py` | 31 关节、32 body、三组 runtime 速度在 start/end 的六类 zero checks 与 shared-ready endpoint | 动态执行 |
| `canonical_motion_compiler.py` | 五动作 × 两 scope 全枚举、全有或全无原子发布 | 训练资格 |
| `canonical_motion_bank_gate.py` | 独立重读 bank 并失败封闭检查；producer manifest 不自证 | 尚无被接受的十件 verifier report |
| `mujoco_motion_player.py` | Agibot MuJoCo 逐帧 FK 回放 | 控制器 replay |
| `canonical_mujoco_dynamics_gate.py` | 关节/碰撞/地面/root/质心与力矩解释边界 | grounded torque 完整解 |
| `canonical_motion_registry.py` | 五行 identity、哈希、ready、状态机和 runtime table | 自动晋级 |
| `MotionCommand.canonical_ready_mode` | registry-bound ready hold/true reset 和 clip 边界门 | 当前候选训练授权 |

`canonical_motion_compiler.py` 的搜索明确记录：

```text
entry_exit = enumerate_all_then_gate_and_rank
adv2c3 = comparator_only_not_default
contact opportunity = marker_only
acceleration_allowed_through_window_end = true
nonnegative_scalar_acceleration_through_window_end = true
```

这些 producer 字段不能自证。现行合同还要求 `t_hit<=0.5 s`、先排 ready→window/anchor 再排 cycle、
显式惩罚可避免的 pre-window zero-acceleration plateau，以及跨 exact `window_end` 的 50 Hz segment
也不减速；probe2 尚未满足。

## 4. 正手挡七关节实测

scope-specific receipt 位于忽略资产目录：

```text
vendor_assets/motion_finalize_20260724/evidence/face_scopes/face_scopes_summary.json
```

这些文件的类别是 `diagnostic_face_core`。它们没有 canonical-ready→contact→ready bridge，也没有
source/output 成对 protected-window digest 或独立 compiler-candidate verifier report；因此下面的
数值不能通过改名、复制或补一个 `publication_class` 晋级。

窗口为 source f34..f48，共 15 帧；每帧保留 64 个候选，再对完整窗口回溯。active joint 顺序是
右肩 pitch/roll/yaw、右肘、右腕 roll/pitch/yaw。

| 指标 | upper | full |
| --- | ---: | ---: |
| 最大拍点位置残差 | `1.49738e-6 m` | `1.34709e-6 m` |
| 最大拍面法向残差 | `2.61879e-6 rad` | `1.26397e-6 rad` |
| 最大拍点速度残差 | `6.69581e-5 m/s` | `6.73550e-5 m/s` |
| 最大相邻步长 | `0.279628 rad` | `0.243387 rad` |
| 窗口最大 URDF 速度利用率 | `0.890084` | `0.936698` |
| ready→窗口任一点最坏乐观下界 | `0.193068 s` | `0.189138 s` |
| ready→f34 乐观下界 | `0.190745 s` | `0.176346 s` |
| f48→ready 乐观下界 | `0.192385 s` | `0.188336 s` |

f44 七关节角（度）：

| upper | full |
| --- | --- |
| `[-79.078, -47.461, 82.962, 14.212, -121.882, 5.660, 27.171]` | `[-87.045, -45.485, 80.071, 10.901, -110.965, -18.322, 29.371]` |

结论只到这里：这证明“不是单腕 ±90/180，而是两套 scope-specific 七关节运动学流形”且窗口拍点/
法向/速度能保持。表中的时间是对角动力学乐观下界，不是最终动作时间；receipt 没有碰撞、桌网、
grounded inverse dynamics、平衡、接触结果、ready bridge、cross-splice 防护或训练证书。

## 5. upper/full 处理审计

upper 版本已经改为：

- root、腿、头取 canonical ready；
- 源骨盆相对旋转的完整三维旋转折到腰部 Z-X-Y；
- 记录并移除源骨盆平移；
- 在该 scope 上重新做正手挡流形和整条路径搜索。

full 版本已经改为：

- 用一次原子平面刚体变换把源 frame 0 的 XY/yaw 对齐 ready；
- 保留之后的 root 局部运动和 31 关节运动；
- 把 root 六坐标与关节共同重定时；
- S0 若需要抬高，必须显式形成新资产并重扫。

旧 full S0 在厂商模型中最大地穿 `7.3716 cm`，不能训练。旧 full 合成正手挡的线性锥回件也已
废弃。旧七件“文件存在”不能转化为新十件的任何 Gate。

## 6. `adv2c3` 审计结论

旧实现只比较 `contact/3` 和 `2*contact/3`，并直接取 suffix：

- 没有共同 ready；
- 没有合法入口/出口全搜索；
- 没有 whole-path retiming；
- 没有 scope-specific 正手挡流形；
- 旧切片首帧仍有约 `31–89 deg/s` 的源速度。

因此本轮没有确认 `adv2c3` 是最终优化；相反，已经确认它不够资格预先获胜。新编译器把它作为
historical comparator 记录。候选先必须满足窗口末前标量路径不减速，再比较整条时长和缩放后的
路径总变化；不允许 `adv2c3` 获得默认或 tie-break 优先。

## 7. 时间与动力学的真实状态

运动学 TOPP 已能对关节和 full root 做连续路径的速度/加速度约束，并允许窗口末前继续加速。
力矩层已能在 fixed-base、每自由度一个直驱执行器的模型上处理 actuator gear/force/control/joint
force 交集、摩擦损失和窗口末前不提前刹车。

本实验把前者统一称为
[`motion timing envelope`](../../DEFINITIONS.md#motion-timing-envelope)与
[`no-brake time law`](../../DEFINITIONS.md#no-brake-time-law)。任何 bang-bang 切换只描述某个解析
profile，不是数学 `C3` 光滑度，也不是恒执行器扭矩证书。

但是 A3 的正式动作站在地面且根节点自由。当前没有接触力分配的线性规划，也没有沿整条路径的
连续全局力矩证明。因此：

- `canonical_path_topp` 只作为运动学初始解；
- `canonical_torque_path_topp` 对站地自由根返回
  `INCOMPLETE_FAIL_CLOSED`；
- `canonical_mujoco_dynamics_gate` 的 joint effort 只能叫 proxy，不能叫已认证执行器扭矩；
- 最终动作时间、窗口宽度、恢复时间和 cycle time 暂不填表。

### 7.1 probe2 当前失败账（不是终表）

2026-07-24 probe2 的 exact 状态是：

```text
vendor_assets/motion_finalize_20260724/probes/
  candidates_v2_noearly_probe2/BUILD_MANIFEST.json
```

- 十件 NPZ 的 Agibot MuJoCo headless FK 为 `10/10 PASS`；这只证明 schema/FK 可播放；
- 顶层 CLI 因 sidecar 白名单 bug 以 `rc=2` 结束，没有 run receipt，所以不是完整成功 run；
- 按 `fh_loop upper/full, bh_loop_c upper/full, fh_block_syn upper/full, bh_block upper/full,
  s0_highpress upper/full` 的顺序，
  [`t_hit`](../../DEFINITIONS.md#canonical-t-hit)约为
  `[0.834,0.708,0.532,0.657,1.499,1.817,0.372,0.580,0.394,0.575] s`，只有 `2/10`
  满足 `<=0.5 s` compiler screening gate；大量窗口前零加速度平台进一步说明当前 ranking 仍被
  cycle/全局 time-scale 牵着走，不是击球优先；
- `fh_loop full`、`bh_loop_c full`、`bh_block full`、`s0_highpress full` 和
  `s0_highpress upper` 的 50 Hz segment 在 exact `window_end` 跨窗处出现负加速度；
  `BUILD_MANIFEST.json` 的
  `retiming.scalar_no_early_brake_proxy.no_negative_scalar_acceleration_inside_window=false`，连续
  prefix 检查不能掩盖该离散违反；
- dynamics gate 因 producer/verifier 对 body velocity 的语义冲突在入口拒绝 `10/10`；修复尚在进行，
  没有任何 dynamics PASS。

因此 probe2 是“FK 可读，但发布、击球时序、50 Hz no-brake 和 dynamics 都未闭合”的失败证据。
当前仍为 `0/10 training-authorized`，上述 `t_hit` 只作编译器定位，不进入最终动作时间/窗口/适用域表。

### 7.1.1 Franco 设计裁决（2026-07-25 深夜，随 probe6 终表落账）

- **block 压缩路线**：若时间律压不进参考带，处置是**缩短 block 的向前行程**（更早更近迎球、
  小行程完成），而不是剪掉动作前半——block 的本分就是位置容差（在哪都能挡）。该"短行程
  block 变体"列为源平滑之外的第二条压缩路线。
- 名义空挥锚点确认仅作血统与排序参考；fh_loop 实测其锚点落在随挥回收段（拍面向后），而
  重定时行为窗内 29/35 帧为向前+向上击球——对齐工具的"+X 主导"判据对拉球类过窄，记工具债。
- 全身版偏慢主因：probe 的 root 六坐标限值是占位保守值（平移 1/1/0.5 m/s 等），root 限值
  按真实平衡能力标定后需重出全身数字。
- 翻面 block 偏慢主因：七关节腕系重构的关节空间行程 + 真实更远的几何位置 + 翻面核步长更大。

### 7.2 2026-07-24 晚续建状态（probe3 前置）

- 中断的 marker-authority v2 迁移已续完：compiler / bank gate / neutral-ready CLI 全部改从
  `configs/canonical_motion_marker_semantics_v2_20260724.json` 取窗口 seed（legacy ge80）与排名
  anchor（nominal event；合成正手挡取 construction annotation，lineage-only）。三个数据事实
  被迁移坐实：fh_loop/bh_loop_c/s0 的名义空挥事件在各自 ge80 种子窗之外（61∉[48,55]、
  84 在 [88,94] 之前、58∉[53,57]），事件到窗口的混叠继续被禁止；eligible 候选新增
  “retained core 必含 anchor”约束，合成件候选并被过滤到 construction solve span 内。
- 击球优先排序补上第 2 优先级：窗前 50 Hz 零加速度平台票数（|段均加速度|<=1e-9，段起点在
  window_end 前），排在 anchor 之后、恢复/cycle 之前；每格 ready→anchor 运动学
  `t_hit<=0.5 s` 现为编译硬筛门（超过即格失败，manifest `search_contract` 记录门值）。
- 合同 §5.1 受保护窗口摘要已实现为 `canonical_protected_window.py`（六通道域分离摘要 +
  transformation receipt + 独立重算 verifier；本地 6 项专项过）；compiler/writer 与 bank gate
  的逐件接线仍在队列上，接线完成前十件仍不可能获得 verifier 接受。
- ready v2 路线按独立审计修正：v1 静态失败的真因是 donor ready 在精确模型中双脚零接触；
  正确组合是「中立右臂挑战者（角度极小极大数学已独立复核，0.65° 全局最优）+ G1 平足
  12 腿关节重解（root/非腿关节逐位不动）」，再全门重审并以新 sidecar 身份发布。审计另立两条
  设计问题待 Franco 裁：①中立度主目标应否从拍面角改成五动作可达时间对称（现只有 bh_block
  16 行进入求解，时间只作 tie-break）；②`t_hit<=0.5 s` 单一常数门对拉球类是否过紧
  （预算本应来自球流/locomotion）。
- **ready v2 组合候选已在 pod2 产出并发布（diagnostic，三类授权全 false）**：G1 对照
  （donor 臂）与 G1+中立臂两条均 `PASS_STATIC_GROUNDED_READY_CANDIDATE` 且静态地面接触力
  LP PASS；发布件
  `pod2:/workspace/codexschema/canonical_v2_20260724/ready_v2_g1_attempt3/published_g1_neutral_arm/`
  （`grounded_ready_candidate_v1.npz` + RECEIPT，receipt SHA
  `0f4f0f88d002849280485de908e18e4d514f71f7a06bb544d5a6be4d88d52ee4`；发布路径内部做了
  exact 构造重建复审）。同 run 附五动作可达时间代理：组合 v2 对五个 ge80 种子起点高度均匀
  （0.094–0.114 s 速度限对角下界），donor v1 在原始关节空间更贴反手族种子——但该表的
  fh_block_syn 行用的是未翻面源帧，不能反推 v1 的正手可达性（v1 的正手灾难见 probe2
  t_hit 1.50/1.82 s）；FH/BH 真对称证据仍以挑战者 16 行连接矩阵（0.133/0.134 s）为准。
  下一步（须 Franco 审后执行）：为组合态铸 ready v2 sidecar 新身份（9-key、新 SHA、独立
  face/站立复核报告），出钉住该身份的 recipe v3，再重编十件。model/identity 度量收据在
  attempt2/attempt3 目录（MEASURED_GROUNDED_IDENTITY 等，均 MEASUREMENT_ONLY 待钉）。

### 7.3 probe7：全身 root 限值放宽与"最短时间"语义修正（2026-07-25 深夜二）

人话：把全身版躯干的占位限速放宽 3 倍重编后，正手/发球的全身版都反超了上肢版；反手拉的
全身版仍更慢，而且这是合法的——因为我们的编译器是"跟踪器"（全身版必须跟完源动作的躯干
轨迹，是多出来的任务，不是多出来的自由度）。正手上肢版 1.33s 的旧数字被证明是拼接处假
曲率钉出来的伪瓶颈（同样的帧，全身版 0.77s 就能进窗）。

- Franco 裁决落账：`t_hit<=0.5 s` **从编译硬筛门降为参考值**（`_STRIKE_TIME_REFERENCE_S`
  仅记录、不筛掉候选）；表格汇报"实际最短需要多久"。§7.2 中"硬筛门"表述自此过时。
- probe7a（pod1，root 限值 ×3：平移速度 [3,3,1.5] m/s、加速度 [30,30,15] m/s² 等，仍是
  probe 占位、待按真实平衡能力标定）对比 probe6a（×1）：
  - fh_loop full：win_start 1.214→**0.769 s**（t_hit 2.864→1.495 s），反超 upper 1.327 s。
  - s0_highpress full：0.434→**0.317 s**，反超 upper 0.350 s。
  - bh_loop_c full：1.955→**0.714 s**（t_hit 0.520 s），仍慢于 upper 0.447 s（源躯干行程
    大，属"跟踪任务更重"的合法差距）。upper 三行与 probe6a 逐值复现（确定性对照通过）。
- probe7c（bh_loop_c root ×10 诊断）反而更慢（win_start 1.105 s、入口从 84 漂到 77）：
  排序元组确实 window_start 时间优先，但 (a) 8×8 探针带按"可行集合内最晚入口/最早出口"
  切片，集合随限值漂移;(b) exact-caps 的 4× 控制率后验守卫会在更松限值的更快侧写下毙掉
  节点间超限的候选。**因此 probe 表数字的语义是"带内最优上界"，不是全局最短**；限值继续
  放宽不保证数字变好。正式穷举+区间认证重建仍是终审路径。
- 速度/扭矩余量成因（Franco 问"是不是太保守"）——**07-26 凌晨勘误**：此前误报"加速度上限
  来自源动作包络"；查收据后确认 `results/source_diagonal_acceleration_envelope.json` 的
  method = `(URDF effort − 静态 bias − 干摩擦)/Mjj`，**本来就是机器人 100% 扭矩的能力换算**
  （"source"只指在源姿态集合上取最差）。对照计算
  `results/robot_torque80_acceleration_envelope.json`（pod2）：80% 扭矩版逐关节 ≈ 旧值×0.8，
  证实同源。所以"打八折上机器人极限"没有增量——我们已在 100%，且 gate 逆动力学代理显示
  bh_block full 的 wrist_yaw 峰值已 113%（对角近似忽略耦合导致的过许可）。真实的剩余保守
  在三处：①**最差姿态全局盒**——整条路径被最弱姿态的能力封顶，姿态相关扭矩限的 retimer
  升级是正式解锁路径（排队）；②拼接假曲率（已证明的主导杠杆，fh 全身 0.77 vs 上肢 1.33
  同帧）；③示范的温和体现在"形状"（行程长、引拍大），不在加速度上限——这正是 RL
  exploration 可以越过参考的空间（Franco 点 3 依然成立）。速度上限一直是 URDF 100%。
- 源平滑 probe 旋钮（拼接假曲率的根治，正手学到的手法对全部动作通用）：HEAD 里已提交的版本
  停止规则是"整体一超容差即放弃"，真实挥拍一遍即超 → `passes_applied=0` **旋钮空转**（probe8a
  收据坐实，数字与不平滑逐值相同）。已改为**管内钳制平滑**（每遍二项式平滑后把偏差 clip 回
  ±容差管、端点钉死、管壁内缩 1e-12 防浮点越界；算法串
  `tube_clamped_iterative_binomial_...`）；单元 6 绿 + bank gate 16 绿（probe 件照旧拒收）。
- **正手平滑扫掠（probe8b/e/f，pod1，root×3+exact-caps）**：上肢窗开
  1.327（raw）→1.205（5 mrad）→0.937（20 mrad）→**0.690 s（50 mrad）**；全身
  0.769（5 mrad 最优）→**0.687 s（50 mrad）**。50 mrad（2.9°）处上肢≈全身收敛——
  拼接噪声磨净后两个 scope 的时间律合流，佐证"上肢 1.33 s 是伪瓶颈"的判定。t_hit
  同步 2.455→1.153 s。**待 Franco 裁**：50 mrad 全程无区分容差会把窗内挥拍形状也磨掉
  ≤2.9°/关节（拍面朝向多关节累计可达数度）；细化方向是分区管壁（窗内紧、连接段松），
  正式版必须走分区+独立复核。20 mrad 全身反弹（1.237）再次示范"改旋钮=换几何"的非单调，
  逐件最优仍是汇报口径。
- probe7b（两件 block，root ×3，pod2）：bh_block full 1.044→**0.573 s**，反超 upper 0.610 s；
  fh_block_syn full 0.841→1.009 s **反而变差**（同一入口/出口 37/48）。机制已核实
  （compiler L468-478）：全身加权弧长把 root 限速直接当坐标权重（scale=1/velocity），改限值
  = 改度量 = 换路径几何（采样密度、拼接样条、尖刺位置全变），所以时间对 ROOT_SCALE **按构造
  非单调**。工程债入队：把度量权重与硬限值解耦（权重钉住、只放宽 cap），解耦前全身表取
  各 run 逐件最优：fh_loop 0.769（×3）、bh_loop_c 0.714（×3）、s0 0.317（×3）、
  bh_block 0.573（×3）、fh_block_syn 0.841（×1）。upper 五行跨 run 逐值复现（确定性通过）。
- 冒烟训练（fh_loop+bh_loop_c upper 新 clip，800 iter，pod2）已跑完：管线级 PASS（跑满、
  存 checkpoint、无崩）；学习级待定——iter800 时 pre_strike_fall_rate 0.28→1.0、episode
  22 步，需与 v4rg 同 iter 基线对照才能区分"adaptive σ 收紧的正常早期形态"与"新 clip 绑定
  问题"。

## 8. Agibot MuJoCo 与 runtime 消费

`canonical_schema2_builder.py` 以具名 31 关节和 32 body 重建 exact schema-2，并要求共同 ready
首末姿态，以及 `joint_vel/body_lin_vel_w/body_ang_vel_w` 三组速度在 start/end 的六类 exact-zero
checks。`mujoco_motion_player.py` 已提供 headless FK 和可选 viewer，可逐帧在厂商
Agibot MuJoCo 中播放候选。

runtime 不再接受五套散装配置。`canonical_motion_registry.py` 原子绑定五个文件、family、phase、
face sign、窗口、ready、source/build manifest 和所有哈希；每件还必须带 source/output 两个
protected-window digest、transformation receipt 和 ready bridge。bank 从 ready donor、runtime pose/
zero-velocity、ready-FK 与五个有序 endpoint 导出一个 registry shared-ready digest，不能只靠五行
重复同一个 ready path/SHA。授权状态机固定为：

```text
compiler_candidate -> training_adopted -> deployment_adopted -> hardware_adopted
```

`publication_class` 只允许从左到右逐级新建 immutable manifest/registry record。diagnostic face core
在链外，不能改名进入第一格；链内也不能跳级、原地手改 authorization 或倒退覆盖旧 record。证据撤销
另记 quarantine/revoke，不改写历史 class。

`MotionCommand.canonical_ready_mode` 只有在 registry SHA、alignment SHA、ready SHA 与五个
training-adopted 文件全部精确匹配时才能打开；它拒绝随机状态初始化、post-swing replay、reset
noise、yaw 扰动、wrap teleport 和中途换动作。当前 registry/consumer 代码完成不等于候选已经
training-adopted；没有独立接受的十件 bridge/window/common-ready report 时保持 `0/10`。

## 9. Planner 与 locomotion 边界

动作选择器未来接收的不是“左/右手”一个比特，而应至少包括：

```text
motion_id
time-to-contact / contact opportunity
pelvis-local strike target
face / spin intent
root or locomotion plan
```

候选规则仍是假设，待行为卷验证：

- 正常距离且时间充足：同侧 loop；
- 近身、来球快或 loop 几何不可行：同侧 block；
- 高球进入专门范围：S0；
- 无动作满足硬门：fail closed。

locomotion 在引拍期运输 pelvis，在击球机会附近稳定 root，在随挥后回到“平移后的 shared ready”。
一旦组合移动，原地动作的窗口、碰撞、地面、动力学和回球证书全部要重跑。

原地 motion 的 frame ID 是 `a3_robot_origin_ground_z0`，不是 HOPE 球台/ROS 的 `world`。固定
counterfactual station 的双向平移以
[frames_and_coordinates](../../interfaces/frames_and_coordinates.md#canonical-motion-hope-world-bridge)
为真源；locomotion 后必须绑定新的 station/root transform，不能继承固定桥的桌网或行为证书。

## 10. 全身学习的配对消融

“全面升级包”同时改变太多机制，只能留到二阶段做兼容性压力测试，不能回答全身参考是否改善平衡。
第一个因果问题必须只看五套 pelvis-local 原地守台动作：允许动作自身的重心转移，不教有意平移或
迈步；需要位移时仍由 locomotion 负责。

### 10.1 当前实现边界与首轮三格

当前新增的 `lower_body_pose_imitation` 只对 12 个腿关节的位置误差计算有界 pose kernel。它不模仿
root/pelvis、足底接触、下肢 body position/orientation、腿速度或 root 速度。因此本轮准确名称是
“full-scope 参考 + 12 腿 pose mimic”，不是完整的 root/contact/velocity full-body mimic；后者若要
回答，必须先另做实现和预注册。

| 格 | 资产/参考 | 显式下肢 reward | 只回答的问题 |
| --- | --- | --- | --- |
| `U0` | upper 五动作库 | off | 纯上半身参考基线 |
| `F0` | full 五动作库 | off | full 资产及其参考观察本身的影响 |
| `F1` | 与 `F0` 逐字节相同的 full 五动作库 | on | 12 腿 position-pose reward 的因果增量 |

冻结的对比是 `F0-U0`（full 资产/参考观察效应）、`F1-F0`（12 腿 pose reward 主效应）和
`F1-U0`（当前可执行 full 包的总采用价值）。`U1`（upper 资产 + static-ready 腿教师）延后：它回答
“任意静态腿约束是否有用”，不是第一轮的 full-reference 问题；只有首轮有信号时才用
`(U1-U0)` 对照 `(F1-F0)`。

### 10.1.1 probe 级消融首波队列（2026-07-26 晨，Option-B 谱系）

人话：这一波在**冒烟同款 motion_file 通道**上跑 probe 级 clip（Franco 效率裁定:对 RL 无关的
工序从简），回答 U/F 与 reward scale 的方向性问题;§10.3 的正式发射门**只管正式采用**，不管
这一波。绑定=4 条 clip（fh/bh × upper/full，形状安全档:fh 取 5 mrad,bh upper 取 raw,
bh full 取 5 mrad），BINDINGS.json 见 pod2:`/workspace/codexschema/newmotion_matrix_20260726/`。

**发射前置（全表共用）**:① 5000 iter 延长冒烟的 pre_strike_fall_rate 出现基线式翻转下降
（fresh_c 基线 i136=1.0→i816≈0.08;若钉死 1.0 先修绑定再发全表）;② 绑定 FK+≤5° 对齐门全过;
③ 每臂发射前 1 env×2 iter 热身。单 seed 广度,胜者才补 seed（精简治理）。

| 槽 | 臂 | 配置 | 依赖 |
| --- | --- | --- | --- |
| pod2 GPU1 | `U0` | fh_upper+bh_upper,腿 pose reward off | 前置①② |
| pod2 GPU2 | `F0` | fh_full+bh_full,腿 pose reward off | 前置①② |
| pod1 GPU0 | `F1` | 同 F0 clip,`lower_body_pose_imitation_weight=0.5`（§10.2 冻结参数） | 前置①② |
| pod1 GPU1 | `R-penlight` | U0 + 软限制减负档（对应"软罚压模仿"担忧） | U0 发射后空槽即发 |
| 队列尾 | `R-imit-scale` | U0 + 模仿层权重 scale 档 | 首波有信号后 |
| 队列尾 | `E-push`/`E-speedmix` | NOW 升级后仍要紧的 env 差异各一臂 | 首波有信号后 |

GPU0(pod2) 被延长冒烟占用至其收官;收官后该卡回收进空槽轮转。每臂逐动作出数,禁止只报汇总。

### 10.2 冻结变量

分别在[W、V 两个旧模型 parent](../../DEFINITIONS.md#当前训练与判卷术语)内做 matched 对比，禁止
跨 parent 比绝对值。每个 parent 内必须冻结：

- 五动作采样权重、每动作 pelvis-local 题库、来球流、strike phase/窗口、共同 ready 和 consumer
  metadata；`F0/F1` 必须绑定同一 full bank 内容摘要。
- locomotion/footwork teacher、随机速度推、随机力推、凹凸地形、额外摩擦改写、`action_acc`、
  qbar 和其余集成升级全部关闭；不得用“全面升级包”替某格兜底。
- scorer、初始化、optimizer、训练预算、milestone 和 seed 完全一致；每动作单独出数，禁止只报
  五动作汇总后掩盖坏动作。
- `lower_body_stability_bundle_weight=0.0` 在三格相同。pose 通道参数固定为
  `std=0.35`、`support_pre_s=10.0`、`support_post_s=10.0`、
  `lower_body_imitation_scale_in_window=0.25`；10 秒前后窗覆盖 active swing，现有 hold gate
  仍排除等待段。唯一 reward treatment 是 `U0/F0` 的
  `lower_body_pose_imitation_weight=0.0` 对 `F1` 的 `0.5`。
- 权重为零的格也必须显式传相同 pose/bundle 参数并保留 activation ledger，避免“没接线”伪装
  成有效 control。旧 `weight=2.0, pre/post=10/10, scale=1.0` 只能作为后续剂量挑战，不得混入
  首轮或用其失败否定整个 full-reference 假设。

### 10.3 发射门、seed 纪律与裁决

发射前必须同时满足：

1. upper/full 十件资产逐门闭合，由 Franco 明示从 `compiler_candidate` 提升为
   `training_adopted`；候选状态、相邻动作证书或 pod 本地产物都不够。
2. strict upper/full registry、consumer、source/ready/bank 内容摘要一致；`F0/F1` 的 full bank
   必须逐字节相同。
3. 机器 render gate 已真正读取全局 launch authorization；checked-in 旧队列保持 false，所有
   probe/science 命令生成均失败封闭。Franco 本人明示解除后也必须新建预注册/队列，不能就地复活
   已被覆盖的旧 runner。
4. probe 的 observed/eligible/active 分母非零且守恒，full active swing 覆盖与三格参数一致；
   缺日志、缺分母或窗口漂移整格 `invalid`。

先在每个 parent 做 U0/F0/F1 各 1 个 blocking seed；只有通过门的 treatment 和 matched control
才补第 2 seed。只有准备提出采用结论时，才另开 3–4 个 fresh exact seed。W/V 旧 checkpoint
续训只能给分层诊断信号，不能单独把 full-body 设置升为正式配方。

逐动作、逐 parent 记录：

- physical fall、ready 后 2 秒稳定和 episode completion；
- legal return、接触时机、早/中/晚窗口恢复；
- root/质心、足底接触/滑移、倾角、12 腿 qdot 和 stance width；
- grounded torque/contact 是否完整，而不是只报 effort proxy；
- pose absolute error 与 activation ledger，但 pose error 下降本身不是稳定性通过。

在 V 脆弱层，treatment 相对 matched control 的 fall 至少下降 `25%` 且绝对下降 `5` 个百分点，
同时 narrow/crossover 或 leg-qdot-tail 至少相对下降 `20%` 且绝对下降 `0.01`，另一项不得恶化
超过 `max(0.02, 0.10*control)`。在 W 名义层，fall 不高于
`max(0.5%, control+0.2 个百分点)`，两项稳定代理也不得恶化超过同一容差。两层 completion
下降均不得超过 `2` 个百分点，legal return 下降均不得超过 `3` 个百分点；physical fall 硬失败
不能由 pose error、reward 或 composite 改善抵消。

只有十件资产完成正式采用、上述三格通过且 Franco 明示后，才能重写
[集成升级波](EXP-P1-INTEGRATED-UPGRADE-WAVE.md)去回答“已过主效应的机制能否共存”。

## 11. 十件候选当前账

用户要求“修完再看表格”，所以这里不提前填最终时长或能力数字。每格当前都停在
`probe2 rejected / new-ready rebuild and independent gate pending`，明确为
`0/10 training-authorized`。已有 face receipt 只是
diagnostic；ready pose/zero-velocity/common-ready digest、成对 protected-window digest、整桥收据和
独立 verifier report 未齐时，任何一格都不能改写为 compiler-candidate accepted：

| 动作 | upper | full | 已知专属债 |
| --- | --- | --- | --- |
| `fh_loop` | probe2 rejected；新 ready 重建 pending | probe2 rejected；新 ready 重建 pending | 新窗口行为重扫 |
| `bh_loop_c` | probe2 rejected；新 ready 重建 pending | probe2 rejected；新 ready 重建 pending | 新窗口行为重扫 |
| `fh_block_syn` | probe2 rejected；新 ready 重建 pending | probe2 rejected；新 ready 重建 pending | 七关节构形后的碰撞/接触/行为 |
| `bh_block` | probe2 rejected；新 ready 重建 pending | probe2 rejected；新 ready 重建 pending | 新窗口行为重扫 |
| `s0_highpress` | probe2 rejected；新 ready 重建 pending | probe2 rejected；新 ready 重建 pending | full 地面修复；方向专卷 |

每格必须交付 NPZ、build manifest、Agibot MuJoCo FK、位置/速度、grounded torque/contact、
自碰/拍体/地面/桌网、scope-specific 窗口、原地行为/恢复和 registry/consumer 证书。

## 12. 执行顺序与停止条件

1. 先生成新的 face-neutral ready candidate：保留站姿/root/拍心设计目标，联合重解右肩/肘/腕七关节，
   以新 sidecar/hash/identity 保存；独立复核正反手拍面角、拍心、关节限位、碰撞、站立和 exact-zero
   速度。不得覆盖 v1；求解器、容差或报告未齐就停止。
2. 用钉住该新 ready identity 的新版 recipe 编译五动作 × upper/full 十件 direct-ready 候选；每件
   生成 ready bridge、source/output protected-window digests 和 transformation receipt。旧 probe2
   不继承任何通过项。
3. 独立 verifier 从 exact bytes 重算十件整桥、窗口、endpoint 和 registry shared-ready digest；任一
   diagnostic/cross-splice/缺字段都拒绝。
4. 逐件做 Agibot MuJoCo headless FK 与人工播放。
5. 完成 grounded contact/torque 层；在此之前保持失败封闭。
6. 重跑自碰、拍体、地面、桌网和 scope-specific contact opportunity。
7. 对早/中/晚触球分别验证恢复到 shared ready。
8. 生成 strict upper/full registry，验证 consumer 和 exporter，但保持 authorization=false。
9. 十格证据齐全后再生成最终动作时间/窗口/适用域表，交 Franco 审核。
10. Franco 明示资产进入 `training_adopted` 后，按 §10 做 W/V × U0/F0/F1 首轮；通过后才重写
   集成升级波。

任一格出现坏 provenance、模型漂移、地穿、自碰、窗口行为失败、grounded torque/contact 不完整或
endpoint 不是共同零速 ready，整格立即失败；不得用相邻动作、另一个 scope 或运动学下界补证。
