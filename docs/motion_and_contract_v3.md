# Motion & Contract v3 — 连续动作库 + 下一代观测契约 (2026-07-04)

来源:franco × GPT-Pro 的长期动作方案讨论(cost-based selector + 连续强度 latent,
原文要点收录于本文),与 eval-B 模式的结构级发现(拍面法线无指令通道)合并成
下一代(v3)的统一设计。原则:**少量离散选择(正/反手)+ 连续参数(强度/速度/幅度)
+ cost-based selector,不上 if-else 动作库,不先上学习 classifier**——与 HITTER 的
架构哲学一致(它也只有 fh/bh 两个参考 + heuristic 选择)。

## 1. 契约 v3:planner 输出 ←→ policy 输入(2026-07-04 已逐条对照代码核实)

**franco 的"目标纵向/横向速度" = StrikeSpec 的 v_n / v_t 分解**:v_n(法向,纵向穿透,
决定落点深浅,敏感度 0.85 m/(m/s) 是控制预算之王)、v_t(切向,横向摩擦,决定出球旋转)。
planner 空间里解这两个 + 拍面法线 n;给 policy 时合成完整速度矢量
`v_r = v_n·n + v_t_x·b1 + v_t_y·b2`,分解信息由 n 通道隐式携带,不必单独进 obs。

### 1a. 现役 175 维:planner 给什么、policy 收什么(代码核实,file:line 见下)

| planner/部署侧(世界系) | 线协议 `/racket/command` | 变换(训练=部署同式) | policy obs 通道 |
| --- | --- | --- | --- |
| p_intercept 击球点(Stage-2 轨迹预测) | `position` ✓ | `yaw(base)⁻¹·(p_w − p_racket_FK_w)` | `racket_target_pos_b` [167:170] |
| v_racket 目标拍速 | `velocity` ✓ | **原样直通,无任何变换** | `racket_target_vel_w` [170:173] |
| n_racket 拍面法线 | `normal` ✓ **线上已在传** | **被丢弃——175 维没有坑位** | (无) |
| t_strike 击球时刻 | `strike_time`/`time_to_strike` ✓ | 直通(秒) | `time_to_strike` [173:174] |
| k 正/反手 | (部署由 heuristic 定 clip) | ±1 标量 | `swing_type` [174:175] |

本次核实的修正与发现:

1. **速度通道名为 `racket_target_vel_w`,不是 `_b`**(本文档上一版写错)。位置做自我
   中心化(`hope_commands.py:1887`,`pp_obs_builder.hpp:184-188` 同式),**速度是世界系
   裸直通**(`hope_commands.py:1896-1898`,`pp_obs_builder.hpp:191`)——现役契约本身就
   不对称。真机模式 `use_base_yaw_for_targets=false` 时 yaw≡单位阵(目标本来就表达在
   机器人标称 +x 朝向系),两者在真机退化一致;训练里 base 转头时语义不同。
2. **法线在部署线协议里已经在传**(`RacketCommand.normal`,`node.py:167-172`)——
   planner→runner 这一段没有缺口;缺口只在观测契约(`pp_obs_builder` 无坑位)与训练
   (`racket_normal` 奖励跟 clip 而非指令)。
3. **critic 已有特权 `racket_target_normal_w`(3 维,`hope_env_cfg.py:96-109`)**——
   v3 补 actor 通道时,训练侧只需把 racket_normal 奖励的参照从 clip 换成指令法线,
   机制全部现成。
4. p_hit 与 t_strike 来自 Stage-2 轨迹预测(`StrikeTarget`),不是 StrikeSpec 的解;
   StrikeSpec 的独有产出 = (n、tilt 角、v_n/v_t、v_r、预测落点 + 敏感度)。且当前
   StrikeSpec 是**诊断旁路**(`publish_strike_spec` 默认 False,≤1 Hz,不在命令链上;
   `/racket/command` 仍由 legacy 镜面律 planner 填)——v3 落地时 StrikeSpec 转正为
   命令生成者。
5. 部署 EKF 仍是影子模式,命令链 spin-blind(`omega_ball=None`)——拍面补偿要吃旋转,
   转正 StrikeSpec 时一并接 EKF 旋转估计。

### 1b. v3 增量:175 → 179,一次做完(法线 3 + ρ 1)

- `racket_target_normal_b`(3 维):**随位置通道的惯例走 `yaw(base)⁻¹·n_w`**——方向量
  是机器人要用身体姿态实现的,应自我中心;真机 identity-yaw 下与世界系数值一致,无部署
  分裂。eval-B 反事实证明这 3 维 = 25/25 vs 0/25 合法回球。
- `stroke_intensity ρ`(1 维,[0,1] 直通)。
- **同场决定**:是否把 `racket_target_vel_w` 一并改成 `yaw⁻¹` 旋转,修掉 1a-1 的不对称。
  代价:热启动时该 3 维语义微变(base 朝前时数值几乎相同,宽容热启可用);收益:契约全
  自我中心,base 转头时速度语义正确。倾向:**改**(反正是一次性的契约日)。
- 级联清单:actor 契约文档 → `hope_env_cfg` actor 增项 → `realsensor_obs_reference`
  LAYOUT → 导出器 → `pp_obs_builder.cpp`(把线上已有的 normal 字段接进坑位)→ eval
  双模 → ckpt_compat(179 载 175:新通道列零初始化)。这是 P3 开山工程,全队排期。

## 2. 连续动作库:q_ref^k(φ, ρ)

- k ∈ {forehand, backhand}(离散,今后可扩);φ = 相位(现役 motion phase);
  **ρ ∈ [0,1] 连续强度**——不做慢/中/快 if-else。
- 实现:每侧 3 个 anchor clip(compact/normal/fast),**相位对齐后线性混合**
  `q_ref(φ,ρ) = blend(q_compact, q_normal, q_fast; ρ)`。MotionLoader 扩展:per-side
  anchor 组 + 混合权重;击球帧对齐为混合的相位锚点(anchor 间用击球帧对齐,再均匀重定相)。
- ρ 的来源(部署):v1 手写平滑函数
  `ρ = smoothsat((|v̂_racket| − v_min^k)/(v_max^k − v_min^k))`,输入以后扩到
  (v̂_racket, t_strike, p_hit, p_base, 上一拍);v2 换小网络。
- **相位重定时(bounded phase controller,franco 的"拉伸")**:
  `φ̇_req = (φ_hit − φ)/(t_strike − t)`,经平滑 + [γ_min, γ_max] 限幅——防过度慢放
  与暴力压缩;超界时不 if-else,由 selector 换更 compact 的 k/ρ。
  注:这推广了现役 time_step_for 的匀速时钟;训练侧需同分布(hold + 变速时钟增广)。
- **训练侧增广已落地(2026-07-04 晚,R14)**:`motion.speed_scale_range`——每挥一次采样
  播放速度 s,时钟 ×s、参考速度 ×s、tts ÷s、目标拍速 ×s,(帧,tts,速度) 配对全程一致。
  这就是"变速改幅度"的 v0:**变速改的是速度幅值,空间幅度的杠杆是裁剪窗口(R6)**,
  两臂合看 = 无新数据的连续强度雏形;等 6 套 anchor 落地后 ρ 混合替代。
  附:部署 runner 已有 `swing_speed` 旋钮(pp_policy.hpp)但**不缩放参考速度与目标速度**
  ——policy 若训练过 R14,部署侧启用 swing_speed 时必须同步补上这两个缩放,否则正好落进
  R14 消除的 OOD 配对。

## 3. Cost-based stroke selector(不上 classifier)

对每个候选 k:`C_k = w_reach·C_reach + w_time·[T_min^k − t_strike]₊² + w_vel·d(v̂, V_k)²
+ w_base·‖p̂_base^k − p_base‖² + w_switch·1[k≠k_prev] + w_recover·C_recover`,
k* = argmin,部署加滞回(C_new < C_old − δ 才换)防边界抖动。
输入至少 (p_hit^base, v_in, v̂_racket, t_strike, p_base, q, q̇, k_prev),将来 + ω_ball。
**只按球速分类是不够的**(同速不同位/高/远需完全不同动作)。
学习升级顺序:规则 cost → 成功数据训 gating 网络 → RL/bandit 直接优化回球质量——
第一阶段坚决不上学习。

## 4. 采集方案(下一次专门拍摄;今天的 3 条先行版不冲突,是 P2.0/P2.4 的急救包)

**6 套**:{正手, 反手} × {小幅挡 compact, 标准攻 normal, 快速攻 fast}。
(原方案的"+1 ready 静止条"取消——franco 改判 2026-07-04:v5 clip 首尾已贴 ready
(起始帧互差 0.15 rad),ready 锚从 clip 首帧提取,swing 间填充交 RL;见 G08 P2.0。)
每套 **10-20 次重复**(选 3-5 条最干净或做 prototype);每条完整包含
ready → 引拍 → 加速 → 击球 → 随挥 → 恢复(首尾都在 ready);**自然速度,禁止慢动作
表演**(否则速度剖面失真)。标注击球帧、拍速、随挥。
**现在不收** 搓/弧圈/削(spin-aware strokes 等 flat rally 稳定 + 旋转感知之后)。

## 5. PACE 减速:论文机制、v1 实现、v2 设计(回应 franco 质疑 2026-07-04)

**PACE 论文原文机制**(`papers/2509.21690` §III-B.1 + 其开源代码核实):对 base 生成
**伪速度指令** `v_cmd = 4 × base 站位误差`(clamp ±7 m/s),奖励 `‖v_base − v_cmd‖`
向量失配,配平顶容差核(pos <0.05 m / vel <0.1 m/s 内奖励封顶、零梯度),接触/飞行终段
硬置零(不与击球争梯度)。它是**奖励目标不是底层命令**;P 律 ⇒ 指数衰减逼近 ⇒ 论文
所称"平滑减速"。**论文本身就不是拟合的加减速剖面——franco 的质疑指向的正是 PACE 没做的事。**

**v1 已实现**(`base_decel_tracking`,`hope_rewards.py:161-189`,默认 `weight=0`):
`r = exp(−(‖v_base_xy‖ − v_des)²/0.4²)·pre_strike`,
`v_des = clamp(2.0·planar_dist(拍→目标), 0, 1.6 m/s)`。与 PACE 的刻意差异:用拍-目标
平面距离而非 base 站位误差(不引入世界系 base 位置,175 契约与 base-free 奖励结构不动);
pre_strike 门控(击球帧后即死,防"追旧目标加速"病理)。

**v1 的三个结构弱点(franco 质疑成立,且不止一处):**

1. **P 律不是可实现剖面**:远处 v_des 恒饱和,近处指数蠕行、永不利落到站。真实逼近是
   加速段(a_acc 限)→ 巡航(v_max)→ 减速段(a_dec 限)的梯形/S 曲线。
2. **只比模长不比方向**:朝错误方向以"正确速率"移动也拿满奖励(PACE 原文是向量失配)。
3. **无时间预算**:v_des 只看距离不看 t_strike——球快该更早更急,球慢不必赶;且到位
   时刻 ≠ 击球时刻,要给引拍留提前量 T_backswing(k,ρ)。

**v2 设计(拟合剖面 + 动作幅度耦合,等 6 套采集落地):**

- **减速包络**:`v_des(d) = min(v_max(k,ρ), √(2·a_dec(k,ρ)·d))`——恒减速到达包络。
  a_dec/v_max 两处拟合取小:①6 套人类采集逼近段的 v(d) 剖面;②真机/成功 rollout 实测
  的**有效**加减速上限(执行器+稳定性,不是标称值)。
- **加速侧限幅**:v_des 上升沿限 a_acc,消除目标重采样瞬间的阶跃速度需求(暴力起步的根)。
- **方向项**:改 `‖v_base_xy − v_des·û‖`(û = 平面单位方向),向量失配。
- **时间预算**:约束到位时刻 ≤ t_strike − T_backswing(k,ρ);不可行时不硬拉速度,
  由 selector 换更 compact 的 (k,ρ)(§3)。
- **幅度耦合**:compact anchor 天然低 v_max、短 T_backswing——ρ 同时缩放剖面参数与
  引拍提前量。**同一批 6 套采集喂三张嘴:动作库 anchor、ρ 标定、减速剖面。**
- **平顶核**(PACE 的好东西,v1 没抄到):|v−v_des| 进容差后奖励封顶,不过度约束。

v1 保留为消融臂(NOW.md R12):它回答"减速塑形这个方向有没有信号";v2 回答"塑形对不对"。

## 6. 分阶段落地

| 阶段 | 内容 | 依赖 |
| --- | --- | --- |
| 现在 | v1 消融:base_decel(已实现,R12)、clip_switch(已实现,R11)、固定法线 StrikeSpec(未做,CPU 活,eval-B 复测) | 无 |
| 视频后 | 6 套处理→anchor 组;ρ 混合 MotionLoader;相位重定时训练增广 | franco 拍摄 |
| 契约日 | 175→179 一次迁移(法线+ρ);normal 奖励改跟指令;pp_obs_builder 同步 | 全队排期 |
| 之后 | cost selector 部署;gating 网络(有成功数据后) | 上两行 |
