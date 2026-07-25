# v2.2 冻结权重表(v4rg 谱系)— prereg 草稿,2026-07-26

**状态**:Franco 2026-07-26 口头授权首臂("这个动作可以先发一个训,用来对照换动作")。

## 0. 发射收据(对照臂)

- smoke:`v2smoke_v4rg_ctrl_fresh_seed3_20260726` @ 7bd7c392 —— r1 被守卫误拦(action_rate 基线键,修为剥离+记账),**r2 通过**(迭代 0/2、1/2 零报错,剥离标记落账 ×1)。
- science:**`v2sci_v4rg_ctrl_fresh_seed3_20260726`** @ pod1/GPU0,commit `7bd7c392`,fresh-from-random,20000 迭代 save100,4096 env,v2.2 冻结表全默认(质量 393.4/295.1/229.5、landing 1648.8 legal_base、clamp 9.0/36.0)。角色:**换动作对照臂**——canonical 动作臂将以同 seed/同题库/同权重发射,唯一变量=动作库。
- 后续臂仍需:各自 smoke 一格;canonical 谱系先重 probe 出自己的冻结表。
- **r2 发射收据(07-26)**:smoke `v2smoke_..._r2` @ de888cb4 通过(2 迭代收据完整,stand_start=1.0 生效:ready 台账全采样、站位偏差均值 7.6cm、swing 完成率 84%);science **`v2sci_v4rg_ctrl_fresh_seed3_20260726_r2`** 发射——配方在冻结表之上加:death_penalty −1800、landing 延付 0.24s、stand_start_prob=1.0(废挥拍中段 RSI 空降)。首臂(farmed)目录留证。

## 0.5 首臂事故与修订(2026-07-26,3k 迭代止损)

首个对照臂在 iter~3k 被抽查抓到**重生刷分**:RSI 每次重生出生在参考挥拍中段 → 借参考动量
几步内 capture+legal → 领 landing 大奖(触球步立发)→ 摔死 → 重生再领。实测:回合长
18→7.3 步、摔倒终止 ×10、模仿/站正收入 −66%,而 exact 误差 3.8mm/legal 62%(打得极准,
纯为刷奖)。**死亡成了结算加速器**——one-shot 大奖 × 死亡重置 × RSI 的组合漏洞(PACE 用
−1000 终止罚防的就是这个;我们不用全局死亡罚以免复活 fresh 早期自杀区间)。

**修订(prize 延付制)**:landing 大奖改为【触球后 0.24 s(12 步)内同 attempt 存活】才
发放,死亡/重置/换题没收(settle_delay_s=0.24 进 v2 包;same_attempt 时钟复用 settle 系)。
语义 = 上台且站得住才算数;不误伤学站阶段。工程约束:延付窗必须小于随挥 wrap 窗
(planner 重定时下 ~21 步),relaunch 后盯 landing 实付率验证不被 wrap 误没收。
事故臂 run 目录保留取证。

**延付降级为消融 flag(07-26 Franco 二次复核)**:"延付可以先不需要,或者变成 flag 今天消融"——death_penalty −1800 + stand_start 已双重关死刷分回路,延付边际价值改由单变量消融回答。落地:v2 包默认 settle_delay_s=0(立发),新 CLI 键 `rewards.virtual_landing_settle_delay_s` 臂级显式开。**消融设计**:r2 臂(延付 0.24,GPU0,在跑)= treatment;新发 `v2sci_v4rg_ctrl_fresh_defer0`(延付 0,GPU1,其余与 r2 逐字同含同 seed)= control。判读轴:回合长/摔倒率轨迹、landing 实付率、20k 终点 legal 率与落点质量。

**Franco 复核后加固(07-26)**:延付只把刷分周期拉长 12 步没关死——(奖26−0)/28步 vs 26/46步,
摔死重生仍在"跳过等下一球"上套利。两刀补全:①**死亡罚 death_penalty −1800**(=每次死
−36 实际 > 满分上台券 33,"比上台抽奖大一些";只计真终止不计 timeout;刷分账变
(26−36)/28=−0.36/步严格负;PACE −1000×dt=−20 同量级先例);②**relaunch 起步改
stand_start_prob=1.0**(去掉挥拍中段 RSI 空降——BeyondMimic 有 RSI 但 PACE 无、HITTER 连续
挥拍不靠空降,内证:yikang r2 取证单 clip 随机相位仅 38.7% 到击球帧、canonical_ready_mode
本就设计 frame-0 起步)。延付保留为纵深(领奖前 0.24s 摔=直接没收)。

## 1. 冻结表(scripts/v2_weight_calibration.py 输出,逐字)

| 键 | 冻结值 |
|---|---|
| racket_position / velocity / normal | **393.4 / 295.1 / 229.5**(内部比例 60:45:35 保持) |
| virtual_landing(legal_base,base_frac 0.6,σ 1.0) | **1648.8** |
| virtual_pass_net / strike_success / capture_bonus / spin | 0(v2.1/v2.2 删除项,冻结防回流) |
| 模仿六项 / upright_exp / hit_unstable_support / qbar / 各安全平滑项 | v2.2 默认包不变 |
| action_rate / action_acc 值 clamp | 9.0 / 36.0(**机制未实现——fresh 臂发射前置,见 §4**) |

**阶梯核对**(每步等效收入,weight-units):模仿 **2.462** : 质量 **7.385** : 上台 **18.461** = 1 : 3 : 7.5(Franco 07-26 终裁比例,锚在实测模仿收入);单拍上台奖 ~1425;罚项预算 0.27/步(f=0.15×早期地板 1.8)。

## 2. 依据(probe 收据)

- run:`v2probe_a_resume_seed3_20260725_r3` @ pod1,checkout `fb4c5baa`(分支 Franco_codex/v2-reward-20260725),resume model_6700(hs W 谱系),迭代 6700→6899,4096 env;三次尝试事由见 namespace 的 ATTEMPTS.md(r1 admission 空信任集/r2 并行 boot 死锁,皆基础设施性,r3 干净跑完)。
- 实测(last-50 均值):T_c=46.3 步;ρ_I=0.547;exact-strike 误差 位置 9mm / 速度 0.15m/s / **拍面 1.93°**(σ 三通道全部收至 floor 0.075/0.886/0.262——sigma-normal 活体验证);p_capture≈1.0;p_legal≈0.6(net 0.794×inbounds 0.615);落点距台心 0.643m → legal_base 门内值 0.864;`strike_window_hit_rate`=0.216=13/46.3 诚实窗(**C1 修复活体验证**;旧 bug 会读 0.5+)。
- **probe 关键发现(定权口径修正)**:质量核为"触点尖峰"非窗内平铺(swing-through 基准与拍速目标在窗前后段天然远)——实测每拍有效满值步数 k_eff = pos 0.73 / vel 0.057 / normal 0.165;计算器据此从 duty×ρ_Q 口径改为 k_eff 口径。
- measured JSON(逐字节):`{"motion_lineage":"v4rg_runtime_order_v3","I_weight_sum":4.5,"rho_I":0.547,"k_eff_pos":0.73,"k_eff_vel":0.057,"k_eff_normal":0.165,"T_c_steps":46.3,"window_steps":13,"p_legal_target":0.6,"E_land_value_per_legal":0.864,"action_rate_sq_p95":282.0,"action_acc_sq_p95":40.0}`

## 3. 谱系约束

本表**只对 v4rg_runtime_order_v3 谱系有效**。换 canonical(Franco)动作库:①走 canonical 正门 admission 并收回 v4rg 两条 legacy 信任集条目;②按新谱系重跑 200-iter probe → 重出冻结表(k_eff/T_c/p/E 全是动作依赖量;公式与 1:3:7.5 比例共享)。

## 4. 发射前置与已知残项

1. 任意臂:一格 2-iter smoke(渲染层默认即 v2.2+冻结值)。
2. **fresh-from-random 臂**:action_rate/acc 值 clamp 机制必须先实现(冻结表已给档位 9.0/36.0;无 clamp 的 fresh 臂存在早期自杀区间)。resume 谱系臂不受此限。
3. probe 用旧混合(capture 850/success 30/landing climb 30)运行——所有测量为权重无关仪器量,不受影响;但该 run 的 reward 总量曲线不代表冻结表行为,勿用于对照。
4. ‖Δa‖² 实测均值 ~94 偏高(200-iter 换奖励混合后的 KL 扰动期),p95 用 3×均值保守替代;首个 science 臂读实测分布后可回调 clamp。
