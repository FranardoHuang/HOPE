# Canonical 五动作预处理与消费接口

本页是五动作从人体重建参考进入 Agibot A3 候选库的接口真源。它定义几何、时间、文件、注册表和
运行时消费边界，不宣布任何候选已经可训练或可上真机。实验状态见
[五动作库正式化实验](../experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)，
运行方法见[五动作编译与换面检查](../operations/run_motion_face_shift.md)。

首次出现的项目术语：

- [`canonical ready`](../DEFINITIONS.md#canonical-ready) 是五个动作共用的零速等待状态；
- [`face-neutral ready audit`](../DEFINITIONS.md#face-neutral-ready-audit)检查等待位到正反手拍面的
  几何是否相称；当前 v1 donor baseline 已失败，不是最终中立位；
- [上肢/全身 scope](../DEFINITIONS.md#motion-body-scope) 是同一动作语义的两套身体参考；
- [`adv2c3`](../DEFINITIONS.md#adv2c3) 只保留作历史切片比较项，与定义表口径一致；
- [`motion artifact class`](../DEFINITIONS.md#motion-artifact-class)先区分诊断核心与完整编译输出，
  [`canonical-ready contact bridge receipt`](../DEFINITIONS.md#canonical-ready-bridge-receipt)
  证明候选确实从共同等待状态走完整座桥；
- [`protected-window digest`](../DEFINITIONS.md#protected-window-digest)分别内容绑定 source/output
  的击球窗，不能用“看起来一样”的切片互相拼接；
- [动作构建清单](../DEFINITIONS.md#motion-build-manifest)回答“哪些字节怎样生成”，
  [逐门能力证书](../DEFINITIONS.md#motion-capability-certificate)回答“这份资产通过了哪一门”；
- 所有入口都必须[失败封闭](../DEFINITIONS.md#fail-closed-motion-preprocessing)。

## 1. 动作集合和空间边界

当前目标是五个 pelvis-local（以骨盆附近的局部站位为参照）的原地护台动作，各编译一份 upper 和
一份 full 候选，共十份：

| `motion_id` | 原地职责 | upper 候选 | full 候选 |
| --- | --- | --- | --- |
| `fh_loop` | 有准备时间时主动正手拉 | 必须生成 | 必须生成 |
| `bh_loop_c` | 有准备时间时主动反手拉 | 必须生成 | 必须生成 |
| `fh_block_syn` | 近身快球的正手挡候选 | 必须生成 | 必须生成 |
| `bh_block` | 近身快球的反手挡候选 | 必须生成 | 必须生成 |
| `s0_highpress` | 高点向前下方拍压 | 必须生成 | 必须生成 |

目标十份完整输出经独立 verifier 验收后，初始 `publication_class` 也只能是
`compiler_candidate`，不是“最终主件”。当前 probe2 尚未获得这一验收，仍是 `0/10
training-authorized`。`bh_loop_b` 仍可做宽窗口挑战者，`bh_loop_a` 仍可做离线挑战者，但都不改变
上述五个运行时槽位。

工件类别与发布等级不能混写：

- `diagnostic_face_core` 只允许保存 scope-specific 七关节换面窗口和诊断 receipt；它不是完整 motion，
  不得写 `publication_class`，不得进入 bank registry；
- `canonical_compiler_output` 必须包含 direct ready→contact→ready 全路径、完整时间律和 exact
  schema-2，且 build manifest 与独立 verifier 都确认整桥和 source/output 窗口绑定后，才有资格使用
  第一级 `publication_class=compiler_candidate`；
- 复制 diagnostic bytes、改名、补字段或把它嵌进另一个目录，都不能获得候选身份。

这五个动作只回答“机器人已经站到一个局部护台位置后，怎样准备、击球和恢复”。覆盖整张球台要由
locomotion（移动规划）把 pelvis/root 运输到合适站位，再在击球机会前稳定；移动轨迹、站位和击球
轨迹组合后必须重新过全身安全、接触、动力学和平衡门。不得从这十份原地候选外推“已覆盖整台”。

recipe/source/output 的 motion frame ID 是 `a3_robot_origin_ground_z0`；ROS/球台 frame ID 是 `world`。
两者同轴但不同原点，固定 counterfactual station 的双向平移和 locomotion 后重新认证规则以
[frames_and_coordinates](frames_and_coordinates.md#canonical-motion-hope-world-bridge)为唯一真源。
任何 manifest、桌网报告或 planner adapter 都必须写清变换方向，不能用裸 `world` 混指两者。

## 2. 唯一允许的几何拓扑

每个候选必须直接构成：

```text
同一个零速 canonical ready
  -> 选中的源动作 core 与击球机会
  -> 同一个零速 canonical ready
```

这里的 direct（直接）有四个硬含义：

1. 不先走 `canonical ready → 旧 source frame 0`，再播放旧动作；
2. 编译器枚举合法的 core 入口和出口，比较整条 ready-to-ready 路径，不把某个固定切片入口当答案；
3. 首帧和末帧的关节、root 和 32 个 body 姿态必须是同一 ready；关节、root、body 的速度全部为
   字面零；
4. 等待、出发、击球和恢复是一条连续参考，不把 ready 到切片入口留给运行时临时补洞。

因此“face manifold 解出一段正确窗口”和“完整候选可发布”是两件事。每份候选 manifest 必须包含
[`canonical-ready contact bridge receipt`](../DEFINITIONS.md#canonical-ready-bridge-receipt)，至少绑定：

- canonical-ready/source 整文件 SHA、scope、选中的 source entry/exit；
- direct ready→entry、retained core/contact opportunity、exit→ready 三段几何摘要；
- source marker 到 output fractional frame/time 的一一映射和
  [`no-brake time law`](../DEFINITIONS.md#no-brake-time-law)摘要；
- output 首末共同 ready 姿态摘要、三组 runtime 速度在 start/end 的六类 exact-zero checks 和整文件
  SHA；
- compiler manifest SHA，以及由不同入口重读 exact bytes 的独立 verifier report SHA。

缺任一项时，face core 即使残差全绿也只能保留 diagnostic 身份；不得把它改名为
`compiler_candidate`。

`adv2c3` 是 `int(contact_frame * 2 / 3)` 一类固定切片的历史比较项。它没有搜索入口/出口，没有把
ready 融入路径，也保留了源切片首帧的非零速度。因此它可以放进比较报告，不能决定正式候选的
入口、时间或身份，也不能靠改名成为训练资产。

## 3. 击球机会不是静止片段

recipe 中的 protected source span 只是旧行为扫描留下的 contact-opportunity seed，不是新资产已经
认证的窗口。编译后，窗口起点、名义触球点和窗口终点先作为路径上的具名 marker（标记点）：

- 击球前可以持续加速；
- 窗口内可以继续加速；
- 优先允许路径加速度一直保持到窗口末端，再在随挥段制动；
- 不要求窗口内姿态、速度或加速度恒定，也不要求每个源帧落到输出的同一个整数帧；
- 重定时后必须在新时间轴上重新扫描早/中/晚触球，不能继承旧窗口的回球证书。

只有 scope-specific 重扫通过后，这段 marker 才能登记为真实 contact opportunity。
“窗口末前不提前刹车”约束的是一维路径参数的加速度符号，不等于每个关节都正加速，更不等于各
执行器扭矩恒定。若关节速度、位置、力矩、碰撞或接触硬门要求更早制动，候选应延长随挥或更换
几何路径；不得隐藏违反硬门的结果。

当前 compiler producer 在连续路径网格上把 `window_end` 前一维路径参数不减速声明为
[`no-brake time law`](../DEFINITIONS.md#no-brake-time-law)约束：先求普通 rest-to-rest 速度包络，
再取窗口末前最大非递减下包络，并做连续极值/marker 时长检查。probe2 证明该实现尚未覆盖跨 exact
`window_end` 的 50 Hz 离散 segment，所以 producer claim 不能自证通过；下方 §3.1 的离散检查也是
硬门。这仍只是 conservative kinematic model 下的 `sddot>=0`，允许零加速度平台，也不保证拍速单调
或恒执行器扭矩；grounded floating-root 求解完成前，不能声称动力学目标已满足。

这条时间律和下文的
[`motion timing envelope`](../DEFINITIONS.md#motion-timing-envelope)都不是数学 `C3` 光滑度声明。某个解析下界可使用
bang-bang 切换结构，但文档、manifest 和 registry 中统一只写“motion timing envelope”或
“no-brake time law”，不能用 `C3` 冒充连续阶数或证据等级。

### 3.1 候选选择必须击球优先

[`t_hit`](../DEFINITIONS.md#canonical-t-hit)严格定义为共同零速 ready 到重定时后 nominal
source-anchor marker 的时间。现行 compiler screening gate 是每格 `t_hit<=0.5 s`；超过即该格失败，
不能因为完整 cycle 更短、随挥更好或另一个 scope 通过而放行。通过所有硬门的 entry/exit/time-law
候选必须按以下击球优先 tuple 排名，而不是先最小化总 cycle：

1. 最早到达 contact-opportunity start，再最早到达 nominal anchor；
2. 窗口末前 50 Hz 离散段的零加速度平台总时长/段数更少；
3. 完整恢复时间和总 cycle；
4. scaled path variation、entry/exit 等确定性 tie-break。

零加速度平台不违反 `sddot>=0`，但会浪费准备时间，所以必须显式计数并排在 cycle 之前；只有具名
硬速度/加速度/接触约束真正迫使平台时才可保留，并在 receipt 中写出限制项。不得用一次全局
time-scale 让窗口后的限制拖慢 ready→hit 段，却仍称为击球优先。

连续路径证明之外还必须有 output 50 Hz 离散硬门：对每个左端时间
`t_i < t_window_end` 的 sample interval（包括跨过 exact `window_end` 的那一段），重建的 scalar
path-speed 不得下降，finite-difference scalar acceleration 必须 `>= -tolerance`。跨窗段不能因为
右端已在随挥区就逃过检查；tolerance 必须在 manifest 中固定，不能按输出调大。

### 3.2 当前 probe2 是失败证据

2026-07-24 probe2 只证明十件 NPZ 均可通过 headless FK `10/10`。它不是被接受的 candidate run：
CLI 因 sidecar 白名单 bug 以 `rc=2` 结束且没有 run receipt；十件按
`fh_loop upper, fh_loop full, bh_loop_c upper, bh_loop_c full, fh_block_syn upper,
fh_block_syn full, bh_block upper, bh_block full, s0_highpress upper, s0_highpress full` 顺序的
`t_hit` 约为
`[0.834,0.708,0.532,0.657,1.499,1.817,0.372,0.580,0.394,0.575] s`，只有 `2/10` 通过
`0.5 s`；窗口前还存在大量零加速度平台。`fh_loop full`、`bh_loop_c full`、`bh_block full`、
`s0_highpress full` 和 `s0_highpress upper` 的 50 Hz 跨窗段出现负加速度，manifest 字段
`retiming.scalar_no_early_brake_proxy.no_negative_scalar_acceleration_inside_window=false`，违反本节
离散硬门。dynamics gate 又因 producer/verifier 对 body velocity 语义冲突在
入口拒绝 `10/10`，正在修复，不能算 dynamics 运行通过。故当前仍是 `0/10 training-authorized`，
这些数值只用于定位编译器失败，不进入最终动作时间/能力表。

## 4. upper 和 full 的身体处理

`canonical_body_scope.py` 是两种 scope 的唯一预处理入口：

- upper：root 固定在 canonical ready；腿和头固定在 ready；源骨盆相对旋转的完整三维旋转折入
  腰部 Z-X-Y 三轴，而不是只拷一个 yaw；源骨盆平移被删除并在报告中记账。
- full：对整条源 root 只做一次原子的平面刚体对齐
  （[`SE(2)` 术语见定义表](../DEFINITIONS.md#动作库术语)），把源 frame 0 的 XY/yaw 对齐到
  canonical ready；之后保留源局部 root、腰腿和关节运动。需要抬高避地时，只允许显式、常量、
  有上限的 Z 修正，并形成新资产身份。

upper 与 full 共享高层 `motion_id`，不共享碰撞、地面、力矩、平衡、窗口或行为证书。full 的 root
平移与旋转必须和关节共同重定时，不能先压缩上肢再把旧 root 时间轴贴回去。

## 5. 正手挡是七关节拍面流形

合成正手挡不再定义成“某个腕关节减 180 度”，也不假设一个关节从左 90 度转到右 90 度。
`canonical_face_manifold.py` 在准确的厂商
[MuJoCo 模型描述（MJCF）和机器人描述（URDF）术语表](../DEFINITIONS.md#动作库术语)上，
逐帧共同求解右臂七个关节：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

窗口内的目标是：

1. 保留源反手挡每帧的球拍 site 世界位置，因此也保留该离散路径的世界速度；
2. 翻转球拍带符号的 raw `+Y` 法向；
3. 在关节范围、逐帧连续性和 URDF 速度约束内，从每帧最多 64 个候选组成的全窗口图中回溯一条
   连续分支；
4. 分别在 upper 和 full 构形后求解，因为 root/腰部不同会改变逆运动学答案。

名义 source f44 只是报告注释点，不是数值锚。当前 scope 实测中，f44 七关节角分别为：

| scope | f44 七关节角（度，按上方顺序） |
| --- | --- |
| upper | `[-79.078, -47.461, 82.962, 14.212, -121.882, 5.660, 27.171]` |
| full | `[-87.045, -45.485, 80.071, 10.901, -110.965, -18.322, 29.371]` |

这组数值直接否定“只改腕 roll”与“固定正负 90 度”两种简化。它们只证明拍点和法向的运动学解
存在；碰撞、桌网、接触结果、力矩、平衡和学习性仍是下游独立门。

### 5.1 source/output 击球窗不能跨件拼接

只有两类 NPZ 属于正式 motion 字节：内容注册的 source motion，以及 compiler 生成的 output motion。
canonical-ready 单状态旁车、scope 临时材料化文件和 `diagnostic_face_core` 都不是正式
source/output，不能被 motion loader 或独立 verifier 当作其中之一。

每份正式 source 和 output 必须分别生成
[`protected-window digest`](../DEFINITIONS.md#protected-window-digest)。摘要固定以下六个时序通道顺序：

| 顺序 | 通道 | exact on-disk dtype / shape |
| ---: | --- | --- |
| 0 | `joint_pos` | little-endian `<f4`, `(T,31)` |
| 1 | `joint_vel` | little-endian `<f4`, `(T,31)` |
| 2 | `body_pos_w` | little-endian `<f4`, `(T,32,3)` |
| 3 | `body_quat_w` | little-endian `<f4`, `(T,32,4)` |
| 4 | `body_lin_vel_w` | little-endian `<f4`, `(T,32,3)` |
| 5 | `body_ang_vel_w` | little-endian `<f4`, `(T,32,3)` |

摘要域为 `canonical-protected-window-v1`。其 canonical header 使用 UTF-8、key 排序、无空白且禁止
NaN/Inf 的 JSON，明确写入 `role=source|output`、motion 整文件 SHA、`motion_id/scope`、六通道名/
dtype/完整 motion shape/切片 shape、C-order、frame-index dtype=`<i8` 和完整 index vector。SHA-256
输入流严格为：
ASCII 域名 + `NUL` + header 字节长度的 little-endian `uint64` + header bytes；随后按上表顺序，对每个
切片追加 payload 字节长度的 little-endian `uint64` + little-endian C-order raw bytes。不得只 hash
`joint_pos`、只写 `[first,last]` 或静默转换别的 dtype。

- source index 是 recipe 中 inclusive protected span 的每一个原始 frame；
- output marker 可以落在 fractional frame。transformation receipt 必须保存 start/end 的 exact
  binary64 hex 值，以及闭区间内完整整数 index vector
  `ceil(start_fractional_frame)..floor(end_fractional_frame)`；不得把 marker 四舍五入到最近帧；
- output digest 不应等于 source digest：scope 处理、七关节换面、ready bridge、重定时和 exact FK
  都可能合法改变 bytes。二者必须由同一 transformation receipt 同时绑定。

transformation receipt 还必须绑定 source/output 整文件 SHA、两个 window digest、recipe/compiler/
MJCF/URDF/body-order SHA、scope/face 报告、entry/exit、source marker→output time 映射和允许的变换清单。
独立 verifier 必须从整文件重新切片重算两个摘要，并拒绝 source/output、upper/full、不同
`motion_id` 或不同 build 之间的 cross-splice。只有 face residual、窗口截图或单个整文件 SHA 都不能
替代这对摘要。

## 6. 路径、时间和力矩分层

现行编译器按以下顺序工作：

```text
内容绑定 recipe
  -> upper/full 身体处理
  -> 正手挡七关节 face manifold；其余动作 pass-through
  -> 枚举 core 入口/出口并建 direct ready-to-ready 几何
  -> 关节/root 运动学重定时初始解（warm start）
  -> exact schema-2 重建
  -> 十件 canonical_compiler_output producer bytes 原子落盘
  -> 独立 verifier 通过后才接受为 compiler_candidate
```

[`TOPP`](../DEFINITIONS.md#动作库术语)表示“沿固定几何路径重新分配时间”。当前分层如下：

- `canonical_motion_geometry.py` 建 direct 二阶连续几何并枚举入口/出口；
- `canonical_path_topp.py` 只按位置、速度和显式加速度上限生成运动学初始解；
- `canonical_torque_path_topp.py` 已实现固定基座、每自由度一个直驱执行器的力矩约束核，并可检查
  “窗口末前不提前制动”；
- 对站在地面但根节点自由的 A3，接触力分配的线性规划尚未实现，所以力矩重定时器和
  `canonical_mujoco_dynamics_gate.py` 都必须返回 `INCOMPLETE_FAIL_CLOSED`，不能把去掉 root 行的
  逆动力学数值叫作执行器力矩。

因此运动学时间只是搜索初始解。最终选择还必须有站地接触、完整逆动力学、足底摩擦、
质心/支撑、碰撞和回球结果证据。任何文档都不得把“匀关节加速度”改写成“恒执行器扭矩”。

## 7. schema-2 和 Agibot MuJoCo 播放

“正式 motion 输入/输出”只指 source motion 和 compiled output motion；两者都必须是动作归档
（NPZ；[术语表](../DEFINITIONS.md#动作库术语)）并接受 exact schema-2 的 11 或 14 字段：

```text
fps
joint_pos joint_vel
body_pos_w body_quat_w body_lin_vel_w body_ang_vel_w
kinematics_schema_version body_pos_point body_lin_vel_point body_names
[kinematics_migration_source_sha256
 kinematics_migration_source_point
 kinematics_migration_tool]
```

[`canonical-ready sidecar`](../DEFINITIONS.md#canonical-ready-sidecar)是 compiler/registry 的单状态
辅助输入，不是第三种 motion，也不接受上面的 schema-2 loader。当前
`vendor_assets/motion_finalize_20260724/ready/canonical_ready_v1.npz` 的 exact 9-key schema 是：

| key | dtype / shape | 语义 |
| --- | --- | --- |
| `joint_pos` | `<f8 (31,)` | donor 姿态，registry 消费前转成 runtime `<f4` |
| `joint_vel` | `<f8 (31,)` | 必须字面全零 |
| `root_pos_w` | `<f8 (3,)` | `a3_robot_origin_ground_z0` 中的 floating-root 位置 |
| `root_quat_w` | `<f8 (4,)` | `wxyz`，finite，norm 误差不超过 `1e-6`；符号属于身份 |
| `source_segment` / `source_npz` / `note` | 非空 scalar string | 注释；其中 path 不能代替 source SHA |
| `source_frame` | integer scalar | 当前 v1 必须为 `0` |
| `striking_joint_ids` | integer `(7,)` | 七个唯一 runtime joint index |

当前 v1 身份必须同时绑定，而不能只绑定旁车文件名：

```text
canonical_ready_v1 NPZ SHA =
cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0
donor source = bh_loop_c / source frame 0
donor source NPZ SHA =
d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba
```

这组绑定只把 v1 登记为 donor baseline，不证明“正反手等距的中立位”。exact FK 的
[`face-neutral ready audit`](../DEFINITIONS.md#face-neutral-ready-audit)已显示：upper 从当前 ready
拍面 normal 到 scoped 反手 f44 目标约 `33.4 deg`，到翻面正手目标约 `146.6 deg`；full 分别约
`32.9/147.1 deg`。它明显偏向反手，否定“从该 ready 向正反手距离基本一样”的假设。

因此下一 ready candidate 必须在不静默改变站姿、root、拍心安全目标的前提下，重新求解右肩/肘/腕
使拍面真正 face-neutral；容差与拍心/碰撞/站立门须预注册。它必须写新 sidecar 路径、文件 SHA、
pose/zero-velocity/FK digest 和 registry identity，绝不能覆盖或改写上面的 v1 bytes/hash。ready 身份
改变后，五动作 × upper/full 十件 bridge、window、时间律、FK、动力学和行为证书全部重跑，旧 probe2
不能继承。

ready 的 runtime pose digest 使用域 `canonical-ready-pose-v1`，按
`joint_pos/root_pos_w/root_quat_w/body_pos_w/body_quat_w` 顺序，逐项写 ASCII name、`<f4` dtype、
shape 和 little-endian C-order bytes；每段都以前缀 little-endian `uint64` 长度消歧。zero-velocity
digest 使用域 `canonical-ready-zero-velocity-v1`，以同样 framing 按
`joint_vel/body_lin_vel_w/body_ang_vel_w` 顺序写 exact-zero `<f4` 数组。两种摘要都禁止 tolerance、
NaN/Inf、四元数换号或读取后再猜 body order。

也就是说，runtime pose digest 在 `<f4` 上域分离绑定 `joint_pos/root_pos_w/root_quat_w` 和独立
ready-FK 旁车的 `body_pos_w/body_quat_w`；zero-velocity digest 绑定 runtime archive 的
`joint_vel/body_lin_vel_w/body_ang_vel_w` 三组 exact-zero `<f4` 数组。每件 start/end 各检查这三组，
合称六类 endpoint zero checks。旁车本身没有 body velocity 字段，不能用字段缺席冒充零；
compiler 的 root link-origin linear/angular input 也必须在 bridge receipt 中另证为零。上述摘要必须由
output endpoint 和独立 verifier 共同重算。pose digest、zero-velocity digest 和
[`registry shared-ready digest`](../DEFINITIONS.md#registry-shared-ready-digest)尚未有被接受的十件
verifier report，因此本页不填一个猜测值，也不把当前 ready 写成 training-ready。

硬约束：

- 31 个关节和 32 个 body 都按具名 runtime 顺序绑定，禁止按列号猜；
- `body_pos_w` 是 link origin，`body_lin_vel_w` 是刚体质心速度；
- 四元数为 `wxyz`，归一且相邻同半球；
- 最终重定时后才以同一 MJCF 全轨重建关节速度、root 速度、正向运动学
  （[`FK`](../DEFINITIONS.md#动作库术语)）和 body 速度；
- `canonical_schema2_builder.py` 在写入前量化成 runtime float32，再做 exact FK 和 endpoint 检查；
- `mujoco_motion_player.py` 必须能在 Agibot 厂商 MuJoCo 中逐帧 `mj_forward` 回放并对齐 32 个 body。

播放器只证明“文件可按同一模型做运动学播放”，不推进控制器动力学，也不证明机器人能执行。

## 8. 构建、注册和授权状态机

`canonical_face_manifold.py` 的 output/receipt 属于 `artifact_class=diagnostic_face_core`，位于
publication state machine 之外；它不得出现一个可被 registry 接受的 `publication_class`。
`canonical_motion_compiler.py` 只能请求第一级 `publication_class=compiler_candidate`，且完整
compiler/recipe manifest 必须显式固定：

```text
training_authorized=false
hardware_authorized=false
```

该字符串不是自我授权。独立 verifier 必须区分 `diagnostic_face_core` 与
`canonical_compiler_output`，并从 exact source/output/ready bytes 重算 ready bridge、成对
protected-window digests、endpoint digests 和 manifest binding；缺其中任一项时，即使 compiler
写出 `PASS_COMPILER_CANDIDATE_ONLY`，也不得进入 registry。

当 compiler candidate 写入 bank registry 时，registry row 还必须显式
`deployment_authorized=false`。

每个 upper/full bank 的 `canonical_motion_registry.py` 必须精确绑定：

- canonical-ready 文件/SHA、donor source SHA/frame、runtime pose digest、exact-zero velocity
  digest，以及由同一 MJCF 计算并内容寻址的 32-body canonical-ready FK 真值文件/SHA；
- bank 级
  [`registry shared-ready digest`](../DEFINITIONS.md#registry-shared-ready-digest)，并逐件重算五个
  output 的首末 endpoint；不能只检查五行都抄了同一个 ready 文件 SHA；
- 五个有序 `motion_id`、scope、NPZ 路径/哈希和帧数；
- 每件的 family、击球 marker、contact opportunity、拍面 sign；
- 每件 source/output 的 protected-window digest 和 transformation receipt；
- source/build、applicability、evidence 和 adoption manifest 的路径/哈希；
- schema-3 question bank、training config、ONNX model binding 和 ONNX metadata 的路径、哈希与受支持
  registry binding schema version；未到对应发布层时这些三元组必须整体为 null；
- publication class 与三类 authorization。

`registry shared-ready digest` 的 canonical stream 固定为 ASCII
`registry-shared-ready-v1` + `NUL` + canonical JSON：JSON 必须含 bank `scope`、ready NPZ/donor
source/ready-FK/joint-order/body-order SHA、donor frame、ready pose/zero-velocity digest，以及按
`fh_loop,bh_loop_c,fh_block_syn,bh_block,s0_highpress` 排序的 start/end endpoint pose+zero digest。
JSON 使用 UTF-8、key 排序、无空白、禁止 NaN/Inf。任一行不能自报该值；registry verifier 从 exact
bytes 重算一次 bank 值，并要求五行、alignment 与 consumer export 全部引用同一个摘要。

发布规则采用
[`motion publication state`](../DEFINITIONS.md#motion-publication-state)。registry 值统一叫
`publication_class`；状态机只允许表中从上到下逐级新建 immutable record，不能跳级、倒退、原地改名
或手改 authorization。名称、三类布尔值、最低证据等级和必需工件必须同时精确匹配：

| `publication_class` | training | deployment | hardware | 最低证据 | 额外必需工件 |
| --- | ---: | ---: | ---: | --- | --- |
| `compiler_candidate` | false | false | false | E0 | 无 adopted 工件；base provenance 仍必填 |
| `training_adopted` | true | false | false | E2 | question bank v3、training config v1、adoption manifest |
| `deployment_adopted` | true | true | false | E4 | 上项 + ONNX model binding v1、ONNX metadata v2 |
| `hardware_adopted` | true | true | true | E5 | deployment 全套 + E5 真机证据 |

从 diagnostic face core 进入首行也不是“晋级”：必须由 canonical compiler 重建完整 ready-to-ready
motion，再由独立 verifier 承认它是 `compiler_candidate`。后续每次真正晋级都产生新的
manifest/registry bytes 并保留前态；证据撤销时另写 revoke/quarantine disposition，不能把旧行改写成
低一级 class。

E-level 不是 registry row 自报字段。每个非 E0 声明必须有
[`motion evidence certificate chain`](../DEFINITIONS.md#motion-evidence-certificate-chain)：从 E1 到
所声明等级逐级恰有一份内容寻址 certificate，manifest 与 certificate 都精确绑定同一
`<motion_id, scope, variant, NPZ SHA>`。证书缺级、跨件复用或只改 `evidence_level` 都必须拒绝。

路径、SHA 和 schema version 也不等于工件内容有效。消费端必须使用
[`strict motion artifact parser`](../DEFINITIONS.md#strict-motion-artifact-parser)核对 question bank、
training config、ONNX metadata 和 ONNX model 内部合同。尤其 deployment/hardware 需要官方 ONNX
parser 加 full checker；环境没有该 parser 时必须失败封闭，不能退化成“文件非空且哈希正确”。

临时 JSON、任意 slug、手改布尔值或只改文件名都不能晋级。正式非 audit 适配必须执行
[`canonical runtime four-pin`](../DEFINITIONS.md#canonical-runtime-four-pin)：由调用配置同时精确钉住
registry JSON SHA、五行
[`motion alignment digest`](../DEFINITIONS.md#motion-alignment-digest)、canonical-ready SHA 和
canonical-ready FK 真值 SHA。alignment 必须包含有序动作语义以及
source/build/model/applicability/evidence/adoption provenance、成对 protected-window digests 和
registry shared-ready digest；随后重新读取每个 NPZ、manifest、工件和 endpoint ready。

[`identity-only registry audit`](../DEFINITIONS.md#identity-only-registry-audit)只能核对身份、内容和
上述 digest，不能授予任何 authorization，也不能导出可直接交给动作 loader 的 runtime table。
audit 读出的 digest 可供人工审查和配置生成，但 audit 结果本身不是训练入口。

`MotionCommand.canonical_ready_mode` 默认关闭；打开后必须：

- 从已 training-adopted 的 strict registry 原子装入五个文件、family、phase 和 face sign；
- hold、body reference、anchor reference 和 true reset 全部取 clip 自己的 shared-ready 帧；
- true reset 相邻写入 root 和 31 关节，所有速度为零；
- 禁止随机状态初始化、post-swing replay、reset noise、yaw 扰动、wrap teleport 和中途换 clip；
- 只允许在
  [`shared zero-speed ready boundary`](../DEFINITIONS.md#shared-zero-speed-boundary)
  重新选择动作。

由于当前十件候选均未通过下游门，registry 和 consumer 的代码存在不代表任何 bank 已获训练授权。

## 9. Provenance 和逐门证书

每次构建必须绑定 recipe/source/ready、工具、运行时、MJCF、编译模型、URDF、关节/body order、
所有限制、搜索参数、随机种子、输出和 manifest 的 path/bytes/SHA。运行表和 alignment digest 还要
逐行携带 source/build/model/applicability/evidence/adoption 的 path/SHA，不能只靠 registry 总哈希
让消费者事后猜 provenance。manifest 还必须记录：

- `artifact_class=canonical_compiler_output` 与
  `publication_class=compiler_candidate` 的不同层级；不得把 face diagnostic 的 class 改名代用；
- canonical-ready donor source SHA/frame、ready pose/zero-velocity digest、ready-FK SHA 和
  registry shared-ready digest；
- source/output 两个 protected-window digest、完整 index 描述与 transformation receipt；
- canonical-ready contact bridge receipt 及其独立 verifier report；
- 选中的 source 入口/出口及全部被拒候选原因；
- source marker 到 output time 的映射；
- 窗口策略明确为 marker-only、允许持续加速；
- upper/full body scope 报告；
- 正手挡全窗口 face-manifold 报告；
- 运动学/力矩层分别做了什么和没有做什么；
- endpoint shared-ready digest 与三组速度在 start/end 的六类 exact-zero checks；
- 所有未完成 Gate 和 authorization=false。

compiler 自身写出的 manifest 只是 producer claim。独立 verifier 必须是不同入口，从 no-follow
snapshot 重读 source/output/ready/model/manifest，重新计算上述 digest 和桥；它的 report 也必须
no-clobber、内容寻址。当前尚无被接受的十件独立 verifier report，所以仍是 `0/10
training-authorized`，也不生成最终能力表。

逐件证书顺序为：

```text
schema/source
  -> Agibot MuJoCo exact FK 播放
  -> 关节位置/速度与 grounded torque/contact
  -> 自碰、拍体、地面、桌网
  -> scope-specific 击球机会重扫
  -> 原地行为和早/中/晚触球恢复
  -> registry/consumer/export
  -> training adoption
  -> deployment/hardware
```

任何证书都不能跨动作、scope、路径、时间律或文件哈希继承。

## 10. 当前明确不作的结论

- 不用解析下界或运动学时长宣布“已经能在该时间内击球”；
- 不用拍点/法向残差宣布正手挡已经能回球；
- 不用 MuJoCo FK 播放宣布动力学、平衡或训练可行；
- 不用五个原地动作宣布已覆盖整张球台；
- 不用 upper 的通过替 full 背书，反之亦然；
- 不在十件新候选和逐门报告完成前填写最终动作时间、窗口宽度、球速或方向覆盖表。
