# Canonical 五动作库正式化预注册草案

Status: `draft / candidate generation and gates pending / no launch`

本页只保留正式化前必须锁住的实验设计，避免与运行队列或结果表竞争。详细接口见
[动作预处理合同](../interfaces/motion_preprocessing_contract.md)，实现与证据账见
[EXP-MOTION-CANONICAL-LIBRARY-20260723](../experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)，
操作步骤见[五动作编译与换面检查](../operations/run_motion_face_shift.md)。

术语见 [`canonical ready`](../DEFINITIONS.md#canonical-ready)、
[上肢/全身 scope](../DEFINITIONS.md#motion-body-scope)、
[`adv2c3`](../DEFINITIONS.md#adv2c3)与
[动作归档（NPZ）、MuJoCo 模型描述（MJCF）、机器人描述（URDF）、正向运动学（FK）和
TOPP](../DEFINITIONS.md#动作库术语)。本草案把 `adv2c3` 预注册为历史比较项，而非主件。

## 1. 研究问题

1. 把共同等待状态直接融入完整击球路径后，五个原地动作能否在 Agibot A3 的几何、速度、
   站地接触/力矩、碰撞和平衡约束下执行？
2. 正手挡在各 scope 上的七关节拍面流形能否保留拍点/法向/速度，并在独立来球卷中获得
   合法回球？
3. 在动作、来球、预算和父模型相同的情况下，full dynamic reference 和下肢模仿是否独立降低
   fall、足滑和 root/质心误差？

## 2. 预注册资产矩阵

固定五个 `motion_id`：

```text
fh_loop
bh_loop_c
fh_block_syn
bh_block
s0_highpress
```

每个动作只生成 upper/full 两个 compiler candidate，共十格。几何拓扑固定为：

```text
shared zero-velocity canonical ready
  -> selected source core and contact opportunity
  -> shared zero-velocity canonical ready
```

禁止：

- ready→旧 source frame 0 的串行 bridge；
- 把 `adv2c3` 当默认入口或 tie-break；
- 把 contact opportunity 变成姿态/速度/加速度锁定段；
- 用单腕固定角偏移生成正手挡；
- 用 upper 证书替 full 背书，或反之；
- 在十格未闭合时先把其中一格改成训练资产。

## 3. 编译前固定合同

- recipe：`configs/canonical_motion_library_v2_20260724.json`；
- ready：`canonical_ready_v1`，姿态 donor 为 `bh_loop_c` source frame 0，所有 endpoint velocity
  必须为字面零；
- scope：upper 把完整 pelvis 相对旋转折到腰部并固定 root/腿/头；full 只做一次 root 平面对齐并
  保留局部 root/关节运动；
- 正手挡：右肩三轴、右肘、右腕三轴共同求解；逐帧保留球拍 site 世界位置并翻转 signed raw
  `+Y` normal；
- marker 真源：`configs/canonical_motion_marker_semantics_v2_20260724.json`；重定时窗口 seed
  取 legacy ge80 行为种子（禁止把名义空挥事件混叠成窗口），排名 anchor 取 nominal event
  （合成正手挡取 construction annotation，lineage-only）；
- window：marker-only，并硬约束窗口末前路径参数加速度不为负（含跨 exact `window_end` 的
  50 Hz 离散段）；这不冒充每关节正加速度、拍速单调或恒执行器扭矩；
- selection：击球优先——所有硬门可行后，先比最早到达 opportunity start、再比 anchor 到达、
  再比窗前零加速度平台票数，然后才是恢复/整条时长与缩放路径总变化 tie-break；每格
  ready→anchor 的运动学 t_hit 必须 ≤0.5 s（筛选门，非行为承诺）；
- publication：所有编译输出均为 `compiler_candidate`，training/deployment/hardware 三类授权全
  为 false。

## 4. 十格逐门验收

每一格必须独立通过：

1. recipe/source/ready/model/tool/runtime 的 path/bytes/SHA 闭环；
2. exact schema-2、具名 31 关节/32 body 和共同零速 endpoint；
3. Agibot MuJoCo 逐帧 FK 播放；
4. 关节位置、速度和 full-root 路径约束；
5. 站地接触力分配、执行器力矩、足底摩擦和支撑；
6. 自碰、拍体、地面、桌网连续密采样；
7. scope-specific contact opportunity 重扫和早/中/晚触球恢复；
8. 原地来球行为：接触、合法回球、适用速度/方向/高度；
9. strict registry、consumer、exporter 和 shared-ready 边界换动作；
10. Franco 审查后才可把 publication class 晋级为 `training_adopted`。

运动学 TOPP、对角加速度下界、fixed-base 力矩核、headless FK 或文件存在都不能跳过第 5–8 门。

## 5. Planner/locomotion 组合假设

十格只代表 pelvis-local 原地护台。planner 先用预测触点、time-to-contact、当前 pelvis/root、
拍面/旋转意图和动作适用域筛选，再让 locomotion 把 pelvis 运到局部站位。击球机会附近 root 应
稳定，随挥后回到平移后的 shared ready。

这是待测组合假设，不预注册“已经覆盖整张球台”。组合轨迹必须重新走第 4 节全部门。

## 6. 全身学习配对消融

在十格获得 `training_adopted` 且 Franco 明示启动后，第一轮只用三格（与
[EXP §10](../experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md#10-全身学习的配对消融)
一致；`U1` 静态腿教师延后，不混进第一个因果问题）：

| 名称 | 资产 | 下肢显式模仿 |
| --- | --- | --- |
| `U0` | upper 五动作库 | off |
| `F0` | full 五动作库 | off |
| `F1` | 与 `F0` 逐字节相同的 full 五动作库 | on（12 腿关节 position-pose kernel） |

当前实现只是 12 个腿关节的 position-pose kernel，不模仿 root/接触/速度；结果不得表述成完整
full-body mimic 结论。分别在 W、V 两个 parent 内做 matched 对比；先每格 1 个 blocking seed，
过门后才补第 2 个，采用结论另开 3–4 个 fresh seed（种子纪律见 EXP §10.3）。固定来球、动作
bytes、初始化、预算、reward 其余项和判卷；关闭随机推力、地形、额外 locomotion 与
"全面升级包"其他机制。

预注册主要结果：

- fall 与 ready 后 2 秒稳定；
- episode completion 与 legal return；
- 早/中/晚触球后的恢复；
- root/质心、足底接触/滑移、倾角和关节速度；
- grounded torque/contact 完整性。

“全面升级包”只在该主效应实验后做二阶段兼容性压力测试，不用于归因全身参考。

## 7. 停止和发布规则

- 站地自由根的力矩/接触仍为 `INCOMPLETE_FAIL_CLOSED`：停止在候选；
- 地穿、自碰、拍体、桌网硬碰、坏 provenance 或 shared-ready endpoint 不一致：拒绝该格；
- 正手挡 face manifold 只有运动学残差、没有行为证据：不填适用率；
- 十格未全部完成：不发布最终时间、窗口或覆盖表；
- registry 的 slug 或布尔值被手改：不晋级；
- 未收到 Franco 明示：不发射训练、不改 deployment/hardware 授权、不做真机。
