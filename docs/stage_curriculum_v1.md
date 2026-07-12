# 四阶段课程与开关矩阵 v1（2026-07-05 历史方案）

> **现行性警告：本文件已被 2026-07-08 的课程重排取代。** 现行课程是“阶段 1 固定点 →
> 阶段 2 虚拟球变到达状态 → 阶段 3 物理球进场”，详见 [NOW](NOW.md)。
> 本文件的“2a 不挪步、2b 放开站位”仍可作为阶段 2 内部消融；“阶段 3 旋转”已改为物理球阶段的
> 后段能力，是否以后另列阶段 4 尚未决定。禁止再用本文件定义现行阶段编号。

北极星 = 回球率(o4)。课程拆四段,**每段只放开一个难度轴**;奖励先点亮再扩散。
本文件只保留 2026-07-05 旧拆法及其 anti-cheat/开关设计依据，不承担当前阶段、升段或排程真源。
当前阶段、小目标、组合能力和采用/拒绝状态统一见 [NOW.md](NOW.md)；具体裁决见实验记录和 Gate。

## 阶段表

| 段 | 学什么 | 变什么 | 钉死什么 | 考卷 | 升段线 |
|---|---|---|---|---|---|
| **1 固定点养成** | 用合适的角度+速度把不同速度的球打回固定落点 | 来球速度(先场馆分布档再外扩,见 phase-scan 结论) | 击球点(登记表触球帧 blade 位)、站位(base_mode=pinned,bank 激活即钉 ready 锚点)、无旋、固定落点 | eval-B 固定点无旋变体,**只用 exam split 记分** | 回球率 ≥80%、拍面误差 **p90 <15°(不用中位数)**、落点 <0.3m、0 摔 |
| **2a 拍点放开** | 臂展内应变(不挪步够球) | p_racket:固定 → 小框 → 真球分布(臂展内截断) | 站位(base_mode=pinned 延续,anti-cheat 门见下)、无旋 | 拍点阶梯考卷(框逐级放大),exam split | **分格门**:S2a 框打网格,**每格**回球率过 S1 线,报告以**最差格**为准;且 anti-cheat 门全过 |
| **2b 站位放开** | 球驱动站位 + 步法;减速入位并入配套(R12/base_decel 在此归位) | p_base:pinned → ball_driven;远球进场 | 无旋 | 远球考卷(同步新建),exam split | 远球回球率过线、0 摔、入位减速包络达标(**暂行代理:现有 base_speed_xy_prestrike 指标**,直至减速包络 v2 落地) |
| **3 旋转进场** | 随旋调整拍面/切向速度;消旋奖励从此才有真效用 | 旋转:无 → 低 → 场馆档(≤15 rev/s) | — | 旋转阶梯考卷,exam split | 各旋转档回球率过线 + 消旋指标 |

**S2a anti-cheat 门(具体定义,防偷步)**:整个 pre-strike 窗口内(不是只在击球帧)
root-XY 距 ready 锚点最大偏移 **< 0.15 m**、**步数 = 0**(双脚触地帧无交替离地)、
root-yaw 最大偏移 **< 20°**;训练侧指标与考卷**用同一代码路径**计算(不许各写一份)。
base 钉锚已落地:`hope_commands._qb_base_anchor_off_xy`(bank 激活时 base 需求一次性
由固定击球点求出,永不逐题耦合)。

**记分通则(所有段)**:考卷只在 exam split 上记分(gen_stage1_questions `--split`,
按来球速度哈希 ~80/20,任意 seed 下与训练题不相交);每个门槛数字必须同时打印
kept/N 分母与 in-cone(挥速锥内可答)分母——没有分母的百分比不作数。

## 开关矩阵(全部默认关;关闭时现状逐字节不变,每个开关单独 mech-verify on/off)

| 开关 | cfg 键(已落地 ✓ / 预留名) | 生效段 |
|---|---|---|
| 题目模式 | ✓ `racket.question_bank`(npz 路径;空=关) | S1+ |
| 拍面指令+奖励重锚 | ✓ `racket.face_command`(无 bank 时启动即报错——normal 全零会静默杀死拍面奖励) | S1+ |
| 观测契约 +4 维 | ✓ `racket.face_command_obs`:**[normal(3), rho 占位(1,填零)],175→179**(契约日定案;rho 为 S3 自旋道预留,免二次改契约/重训阶梯) | S1+ |
| 回放变速 | ✗ `motion.speed_scale_range`:**bank 开启时不兼容,train.py 硬报错**(bank 答案是绝对物理量,变速无法缩放答案) | — |
| HER achieved replay | ✓ 自动:**S1–S2a 强制关**(bank 激活即置 0 并打日志;回放目标会被 bank 覆写);S2b+ 仅可作 solver-verified 变体重新评估 | S2b+ |
| 击球点模式 | 预留 `racket.question_point_mode: fixed\|box\|ball_dist` | S2a |
| torch 在线求解器 | 预留 `racket.online_solver`(连续拍点分布下离线 bank 覆盖不了;需 StrikeSpec 的批量 torch 移植 + 与 numpy 版的 parity 测试,复用球物理归一的测试模式) | S2a 第 3 级+ |
| 站位模式 | 预留 `racket.base_mode: pinned\|ball_driven`:**pinned 自 S2a 起生效**(bank 激活即 base 钉死 ready 锚点,已落地于 hope_commands base pin);**S2b 仅把值翻成 ball_driven** | S2a(pinned)/ S2b(ball_driven) |
| 旋转档 | 预留 `racket.spin_tier: none\|low\|venue`(bank 增 spin 列;StrikeSpec 本就吃 omega;消旋奖励 vb_spin_mode 联动;obs 侧 rho 占位已在 179 维契约中留好) | S3 |
| 物理球+球桌真值仪 | ✓ `physical_ball`(**任务顶层键**,默认关;S1 训练环境加真 PhysX 球桌 + 每 env 真球:题库来球按场馆模型**反向积分发球**——击球帧恰好以题目速度到达题目触点、场馆气动力逐 substep 施加、**代码驱动拟合桌面反弹**(venue contact.table)、穿透机器人(球碰撞体关);**纯 metrics**(pb_serve_err_m / pb_serve_vel_err + 计数),奖励/观测零耦合;物理基线 = isaac_ball_inloop_check 17 mm systematic) | S1+(truth instrument;**Phase B = 引擎内拟合拍面冲量,待做**) |

规则:预留键**到实现段才添加**(不放死键);每段考卷和训练同源升级(同一个
bank/求解器产题);升段以考卷过线为准,结果写入对应实验记录，采用状态再同步到 NOW.md。

## 已知风险锚点

- S1:hopex 挥速锥内可答率 0%/10%(phase-scan)——奖励不可锚死模仿;速度档从场馆分布起步。
- S1 信号跑前 TODO(**未做**):A1 传感真实化开关(延迟/噪声/丢帧)只覆盖 pos/vel/swing_sign
  通道,**不覆盖 normal 道**——target_normal_cmd 不进延迟环缓冲。延迟缓冲扩展是 S1 信号跑
  之前的欠账,不是已完成项;在此之前 face-command 通道等效"零延迟完美传感"。
- S2a→S2b 边界:考卷必须含上文 anti-cheat 门(root-XY/步数/yaw 三条同过),否则 2a 会偷学
  步法、2b 失去对照。
- S3:球拍旋转传递未实测验证(fit report F 系列)——S3 的 sim 结论转真机要打折,补采
  高 u_n 低旋数据仍是欠账。
