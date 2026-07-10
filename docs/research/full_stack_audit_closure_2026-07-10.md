# 全栈审计闭环与残余边界（2026-07-10）

状态：代码修复收口中；真机门仍为 `Partial`。Owner：Codex。

## 结论先行

总体方向成立，但需要把“模仿专业人”改写为：**专业路径、触球几何与发力顺序是软先验，A3
自行重解绝对时间、行程和关节分工**。V5 值得抢救到底，因为它能回答专业动作里是否存在可迁移
信息；不能把人类逐关节轨迹当机器人最优解，更不能因肘/肩两个口头力矩数直接改安全限值。

本轮没有整体合并 `origin/hitter@5c346ea`。它包含大量 RallyFinal/消融配方及相互耦合的 runner
行为；我们只逐项核对等价修复。无独立数据、合同或测试支撑的实验性改动不进入 `main`。

## 已闭环的正确性问题

| 链路 | 修复后的不变量 |
| --- | --- |
| BankExam 判帧 | 动作推进物理后，以同步推进后的 strike clock 判 exact frame；不再晚一个 20 ms 控制步。 |
| BankExam 分母 | 每一道已出题 attempt 在 hold 前开账；触球前摔、guard reset、timeout 都留在全分母，截断样本不可 eligible。 |
| 同题比较 | exam bank 预物化不可变 schedule；默认全卷一次，也可固定分层 K；所有模型/噪声列使用相同 question ID 顺序、hold 与标准高斯流，禁止 wrap。 |
| 题目身份 | ID 绑定 bank SHA、clip、row、incoming velocity/spin、demanded velocity/normal；schedule 自带 SHA 与 per-attempt seed。 |
| 训练/考卷隔离 | schema-v3 bank 强校验 train/exam split、动作文件 SHA、clip 顺序、锚帧、物理实现与 source-family SHA。 |
| 落点几何 | 球状态统一为球心；桌面接触判在 `surface + ball_radius`，而不是裸桌面。 |
| 训练配置 | VirtualBall 显式钉 `racket_velocity/racket_normal=0.5/0.5`、`foot_orientation=0`，不再继承 10/5 和 -0.3。 |
| A1 指令 | 位置、速度、拍面与 swing sign 走同一延迟/掉包原子消息；真 reset 清空 held/drop/bias 状态。 |
| 奖励取证 | raw 与 gated 遥测真正分开；base decel 在 frozen hold 关闭，不再与 hold-ready 对打。 |
| planner | 来球速度间断清估计窗；网柱带使用 `TableParams.y_max`；落点使用球心接触面。 |
| 导出 | 不再克隆 donor 的旧 normalization 标签；从 checkpoint 合同重建并核对 metadata，图数据流与标签矛盾即失败。 |
| 训练血统 | schema-3 合同从已实例化 env 记录 joint/action order、decoder、actuator integration、PD、armature、PhysX friction 语义、effort/velocity limits、soft q-des limits、dt、obs/body/motion order；legacy 或 override resume 不能续训洗白为 exact。 |
| C++ 安全 | 模式授权 generation、命令发送和 zero-gain barrier 线性化；独立 deadline supervisor；异常/NaN/Inf fail-closed；soft 训练限位与外层硬限位分层；requested/measured effort 越界停机。 |
| C++ 进入/定位 | MOTION/SHADOW 重入清挥拍相位；station policy 无 fresh localization 禁止发布；yaw capture pending 为显式 zero-gain 门。 |
| CLI/runbook | 未知、重复、缺值参数直接失败；现役命令移除无测试且不安全的 `--arm-hold-nominal`；`--hold-recover` 保留。 |

## 球拍控制点与速度审查

官方 A3 ping-pong URDF 的统一控制点是 `pingpang_red_Link` 原点；它不是腕原点。腕坐标固定偏移为
`(0.210210, 0.032078, 0.032036) m`，与 URDF `pingpang_red_joint`、MuJoCo
`right_racket` site、Python 和 C++ FK 逐位一致；旧 Python 值来自球网格 joint，只差
`1.49 um`但已移除。

这次复核发现旧 Isaac 拍速**并不对**:Isaac Lab 2.1 `body_pos_w` 是 link 原点，
`body_lin_vel_w` 却是 COM 点速度；原实现在 COM 速度上又加 `omega x r_site`，混了两个点。
V5/hopex 触球姿态上误差约正手 `0.401 m/s`、反手 `0.598 m/s`。修后强制使用
`body_link_lin_vel_w + omega x r_site`，缺 link 速度属性当场报错，不再回退 COM。Phase
`TableTennisEnv` 也同步修正，并为被 PhysX 合并的零质量球拍 link 加入腕+official offset FK。
MuJoCo 用 site 的 `mj_objectVelocity`，原本就在正确的同一点。

STL 审查显示该点是红胶外表面的设计接触中心，而不是拍体积中心：红面几何中心距它约
1.3 mm；官方 20 mm 球网格与红面的切点仅约 0.07 mm。**因此动作本身的拍点/拍速计算是对同一
URDF 控制点的。**

但现役 planner/venue/bank 还有一个更上层的物理近似：把球心 `p_ball` 直接当控制点目标，未做
`p_face = p_ball - R_ball n`；反手只翻 normal，没有把红面 site 移到黑面外表面。红面球心到
site 约 `20.040 mm`，黑面约 `33.232 mm`。接触模型还把 site 速度当实际面中心速度，
黑面尚差 `omega x r_face`。旧考卷内部自洽，但不是精确接触几何；在完成一次带版本的
planner/wire/bank/训练/判卷合同迁移前，不可偷偷改默认而让旧 checkpoint 与题库错配。
该项已加入 V5/Phase 消融。

目标拍速另有口径边界:`clean_reference_strike_velocity` 是 50 Hz 下 `+-2` 帧、跨
`80 ms` 的同点差分，不是瞬时接触真值。V5hLs 正手 `2.488` vs `+-1` 帧 `2.315 m/s`，
反手 `3.404` vs `3.533 m/s`；两个触球帧仍标 `unverified`。因此触球帧×差分窗口已作为
消融轴，不把 `+-2` 值写成物理真值。完整合同见
[racket_contact_geometry.md](../interfaces/racket_contact_geometry.md)。

## 不能在本仓用户态假装解决的边界

1. **控制器级确认**：本仓可保证本进程内四 topic 调用的互斥、重试与 zero-gain 末帧，不能证明
   下游控制器已原子接收、应用或 ACK。真正的 command sequence、controller timeout 与物理
   E-stop 仍是硬件 Gate blocker。
2. **179/181 部署合同**：现役 flat wire 没有 demanded normal/rho；181 的 station/normal term
   顺序也尚未冻结，spawn-world station anchor 没有 wire 来源。C++ 继续拒绝 179/181，比只扩
   shape 白名单安全。需一次 contract day 同步升级 publisher、wire、builder、metadata 和测试。
3. **external-base 朝向**：当前候选设计混用 mocap 世界位置与 engage-relative IMU yaw。必须先由
   团队冻结“桌轴世界朝向由 mocap quaternion 还是 IMU 对齐拥有”，不能盲补一个旋转。
4. **力矩包络**：肘 24 Nm、肩 60 Nm 尚缺连续/峰值、输出轴/电机轴、速度温度降额与固件限幅
   语义。代码只使用模型内可追溯 effort envelope；拿到厂家曲线后再换成真实连续安全模型。
5. **真实安装误差**：URDF 内部一致不等于实物支架完全一致。上真机前仍须量 `T_mount`，至少
   验证控制点位置、拍面法向与腕姿的静态标定残差。
6. **摩擦合同**：现有 A3 训练把 MuJoCo `frictionloss` 的 Nm 数值直接填进
   Isaac/PhysX 无量纲、载荷相关的 joint-friction coefficient。所有现有 checkpoint 都继承
   这组未标定系数。Formal BankExam 对非零值 fail-closed；直接数值 proxy 强制
   `evaluation_contract_exact=false`。不能在缺实测时猜一组“正确数”。
7. **exact 的语义**：现阶段仅指判卷执行协议与已列 plant 字段被绑定，不是
   PhysX↔MuJoCo 完全动力学等价。刚体 mass/inertia/COM、asset SHA、contact/solver 与 DR
   分布仍待进合同。

## 验证纪律

- 本地纯 CPU：固定题表/判卷合同、schema-3、planner、动作时间律与行程工具。
- RunPod：只做 Linux/ROS2/C++ build+smoke 与最小 Isaac 环境/导出合同验证；不查看、启动、停止
  或调度训练。
- 真机：本轮不自动执行。必须等 G07 的 controller ACK/timeout、mount calibration 与 operator
  safe-halt 检查通过。

具体命令和结果随本次提交写入 G05/G06/G07、`PROGRESS.md` 与操作文档；聊天记录不作为状态源。
