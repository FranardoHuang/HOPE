# 外部尽调:BeyondMimic / mjlab / unitree_rl_lab 的 DR 与 reward 设置对照(2026-07-31)

**触发**:有同事建议"训练时给 policy 加的基础泛化不够(比如随机 kp/kd)"。本文对照三个外部库
(BeyondMimic 上游 `HybridRobotics/whole_body_tracking`、`mujocolab/mjlab`、`unitreerobotics/unitree_rl_lab`,
均 2026-07-31 浅克隆最新 main)逐项核实:他们随机化什么、罚什么、用什么数值;我们已有什么、缺什么、
值得借什么。

**方法与可信度**:5 个抽取 agent(我方 reward / 我方 DR / 三个外部库各一)+ 5 个对抗核查 agent
(逐条 file:line 复核、反向搜漏项)+ 主线人工抽查(kp/kd yaml 链、joint_acc 数值、push 区间、v2 展开
机制)。核查共揪出 24 处修正与 37 处漏项,下文均为**修正后**数值。逐库原始 JSON 存
scratchpad(会话级,不入库);本文是归档结论。

---

## 一、结论先行(人话)

1. **"我们没随机 kp/kd"这个前提是错的——而且四家里只有我们有。**
   我们的 `randomize_pd_gains`(每次 reset 重抽、log_uniform、scale、31 关节全覆盖、kp/kd 独立抽样)
   在**所有现役臂上以 ±15% 开启**(含 N1 在用的 ActionBall,经 `defaults:` 链继承 Hitter 的
   `pd_gain_range: [0.85, 1.15]`,[HOPEPingPongHitter.yaml:527](../../hope_training/whole_body_tracking/cfg/task/HOPEPingPongHitter.yaml));
   只有对齐 HITTER 论文的 `HOPEPingPong.yaml` 臂显式设 null(论文固定 PD)。
   BeyondMimic 上游、unitree_rl_lab:**完全没有**这个功能;mjlab:写了 `dr.pd_gains` 原语但
   **没接进任何任务**。建议者说的方向没错,但我们早已领先,不是短板。
   (07-31 补:**厂商 instinct_mj A3 parkour 也随机 PD**——startup 级,Kp ±20%、
   **Kd (0.7,1.3) 不对称更宽**,见 §11;对齐动作在 §11.3。)
2. **提问里点名的两个惩罚我们也都有,而且不弱于三家:**
   - last-action 差值罚(`action_rate_l2`)四家全部 **-0.1** 同值(unitree 步行臂更松,-0.05);
     我们在 v2 包里还换成了带值钳的 `action_rate_clamped -0.2`(clamp 9.0),并且是**四家唯一**
     把二阶差分(jerk,`action_acc_l2 -0.05`,clamp 36.0)真开起来的(mjlab 定义了但权重 0,只做指标)。
   - 关节超限罚:外部三家都是 isaaclab 标准 `joint_pos_limits`,-10(mjlab 步行臂 -1,unitree 人形步行 -5)。
     我们 DeployParity 系同为 -10;ActionBall 系是**四家最狠且唯一有预判性的**:实际 q 屏障 -40 +
     pre-clamp q_des 屏障 -40 + 投影罚 -5(clamp+barrier 设计裁定,不上 tanh)。**无可借。**
3. **真正的差距不在上面,而在三条现役配方全关/全零的轴:**
   - **外部推撞**:三家在模仿任务上全都开(BeyondMimic/mjlab/unitree-mimic 同配方:每 1–3 s、
     线速 ±0.5/±0.5/±0.2 m/s + 角速 ±0.52/±0.52/±0.78 rad/s)。我们这边**不是没推过,
     是推的裁定断在了传导上**(修正说明见 §3.1,时间线:Wave-P 6 臂 07-20/21 真推过但无判读;
     07-26 v2.3 模板裁定 push 默认带上;N1 发射器 argv 实测无任何 `task.push.*` 键)。
   - **执行器/观测延迟**:四家其实都没开(全是死代码或默认 0),但我们是真机部署方,这轴对我们
     比对他们更重要;我们的 `DelayedImplicitActuator` 同样是死代码,A1 球目标延迟/噪声旋钮全零。
   - **实际关节加速度罚 `joint_acc_l2`**:unitree 在**全部任务族**(四足、人形步行、人形模仿)用
     同一个 -2.5e-7;我们完全没有(我们罚的是动作空间差分,不是实现出来的 q̈)。
4. **优先级提醒**:现行裁定是"速度泛化优先,其他泛化轴降级"(2026-07 决定)。下面的可借清单
   是证据台账,不是自动排产;除"零风险探针"外,任何一条都应作为独立臂 A/B,不动现役配方。

---

## 二、我们已覆盖/领先的(不必动)

| 项 | 我们(现役值) | 外部对照 |
|---|---|---|
| kp/kd 随机化 | ±15%,reset 级,log_uniform,全关节(唯 HOPEPingPong 臂 null 对齐论文) | 三家全无(mjlab 有原语未接线) |
| 连杆质量 | 全身 scale ±15%,recompute_inertia | BeyondMimic 无;mjlab 无;unitree 仅躯干 additive (−1,+3) kg |
| 摩擦/恢复系数 | 全身 static (0.3,1.6) / dynamic (0.3,1.2) / restitution (0,0.5) | 与 BeyondMimic 同;宽于 mjlab(仅足底、切向)与 unitree(人形 restitution 钉 0) |
| 关节零位标定误差 | ±0.01 rad,**同时写进 default_joint_pos 和动作偏置**(train==deploy 一致) | BeyondMimic 同;mjlab 只偏 actor 观测;unitree 仅 mimic 有 |
| 躯干 CoM 抖动 | x±0.025 / y±0.05 / z±0.05 m | 四家几乎同值 |
| 一阶动作差分罚 | -0.1(v2:clamped -0.2) | 四家同 -0.1 量级 |
| 二阶动作差分罚(jerk) | **唯一真开**:-0.05,clamp 36,自带 prev_prev 缓冲 + reset 有效位掩码 | mjlab 定义未用(只做指标);另两家无 |
| 关节超限罚 | ActionBall:实际 q -40 + q_des pre-clamp -40 + 投影 -5;DeployParity:-10 | 三家全是被动 soft-limit 尾巴,-1 ~ -10 |
| 关节速度/力矩罚 | joint_vel -1e-4;joint_torques -3e-5(yaml) | BeyondMimic/mjlab 两项全无;unitree 部分任务有 |
| 非对称 actor/critic 噪声 | actor 逐通道 Unoise,critic 干净+特权 | 四家同构,数值基本一致 |
| RSI reset 噪声 | 位姿/速度/关节噪声与三家逐字节一致,另有 stand/post_swing 起点混合(独有) | — |
| 推撞**机制**(非配置) | 三套已写好:Wave-P 速度踢、F-axis 恒力推(带逐步清扫)、Bernoulli 互斥合并 | 比三家的实现都精细;Wave-P 实测挂载过,现役 N1 未启用(§3.1) |

## 三、可借鉴清单(按优先级)

### 3.1 高:把已裁定的推撞真正传导到现役谱系(修正:不是新提议,是断链修复)

**初版报告此处写"我们一次都不撞",经 Franco 质询后核实为误——准确时间线:**

1. **Wave-P(EXP-P1-PUSH-ROBUSTNESS-20260721)真发射过 6 条推撞臂**:`p1push_{w,v}_{p02,p035,p05}`
   (07-20/21,W/V 父本 model_6700 续训,interval 5–15 s 速度推 ±0.2/±0.35/±0.5 m/s;
   probe 确认 event 真挂载,6 条 science rc=0)。**但至今没有判读记录**——16700 终档 K100
   裁决没有发生,07-26 v2 冻结换代后这波悬空;实验文档底部"运行表/决定"仍停在
   preregistered 未更新,与头部运行态自相矛盾。
2. **07-26 v2 冻结审计(§0.10,Franco 触发,原话"随机推没开,是不是还漏了什么")已把 push
   裁进 v2.3 模板**:`++task.push.enable=true ++task.push.vel_xy_mps=0.35 interval [10,30]s`
   + `++task.force_push.enable=true force_n=68 duration 0.3s`(双事件近似合并,合成期望
   ≈每 ~10 s 一次),并注明"pod2 新动作臂发射前必须采纳"。
3. **N1 发射器没有继承 v2.3 的 push 键**:`launch_n1_reward_screen_diagnostic.py` 组装的
   argv 实测无任何 `task.push.*`/`task.force_push.*`。若 N1 作为 reward 筛查波刻意保持与
   在跑 v2 臂单变量可比(§0.10 原文"全队一致,不影响臂间比较"),这是合理的;但"下一代
   基线必须带 push"的裁定目前没有任何机械保证会被执行。

- **外部证据的作用**:三家外部库独立佐证 v2.3 裁定的方向——模仿任务全部开推
  (BeyondMimic `tracking_env_cfg.py:190-195`、mjlab `tasks/tracking/tracking_env_cfg.py:165-171`、
  unitree `tasks/mimic/.../tracking_env_cfg.py:210-215`,同配方 interval (1,3) s,线速
  x/y ±0.5、z ±0.2 m/s,角速 ±0.52/±0.52/±0.78 rad/s;unitree 步行臂温和版:5 s 一次、仅 xy ±0.5)。
  外部 1–3 s 节奏**不要抄**(会砸进击球窗);v2.3 的 [10,30] s 与 Wave-P 的 [5,15] s 都比
  外部保守,方向正确。
- **落点(两件事,都不是"要不要推"而是"把已裁定的推落实")**:
  (a) 给 Wave-P 6 臂补收口:要么判读要么正式作废,把 EXP-P1 文档底部的矛盾状态改一致;
  (b) 把 v2.3 push 键写进下一代基线的发射器/模板本体(而非只留在 v2 冻结文档第 87-88 行),
  否则每次新写发射器都会像 N1 这样静默丢掉。
- **风险**:撞进击球窗=直接税击中收入层(红线);合并抽签 CLI 未接线是已知 TODO,v2.3 用双
  独立事件近似时两种推可能同帧叠加(概率低但非零);大踢恢复可能要求真机给不出的加速度
  (包络仅 ~3.5% 余量)。

### 中:`joint_acc_l2`(实现出来的关节加速度罚)
- **人话**:unitree 从四足到人形模仿全用同一个 -2.5e-7,跨任务不改一位,说明它安全;
  我们只罚动作差分,从不罚真实 q̈。
- **证据**:unitree 五个任务文件同值 -2.5e-7(mimic `tracking_env_cfg.py:223` 等,已逐一复核)。
- **落点**:DeployParity/HitterPure 的 rewards cfg 各加一条,权重照抄 -2.5e-7,配 CLI 键可消融;
  可用 swing_only 的 torch.where 模式在击球窗内免罚。
- **风险**:它罚的正是击球最需要的触拍峰值加速度,离"软罚压制击球"的旧坑最近。-2.5e-7 比
  joint_vel_l2 小四个量级,理论上是舍入误差,但**只准 A/B,不准默认翻转,不准加码**。

### 中:失败加权自适应 RSI 起点采样
- **人话**:BeyondMimic 和 mjlab 都会统计"policy 最近在片段哪一段挂",下次开局多从挂的地方练;
  mjlab 甚至把它设成默认。我们的 CurriculumCfg 是空 `pass`。
  (**§9.1 修正**:同常数的失败加权采样器其实已在我们库内、checkpoint 安全,只是所有已注册
  task 都结构性跑不到它;落地方式与 cheat 规避见 §9.5 R3/R6,本条按彼处执行。)
- **证据**:同一套常数两家复现:EMA α=0.001、uniform 兜底 0.1、λ=0.8,约每秒动作一个 bin
  (BeyondMimic `mdp/commands.py:207-241,363-371`;注意 `commands.py:220` 自注释是**非因果**核,
  混的是未来邻 bin)。
- **落点**:扩展我们 MotionCommand 的 resample 路径,bin 按 `strike_annotations.yaml` 的挥拍相位切
  (预备/引拍/击球窗/恢复),只作用于 RSI 分支,不动 stand/post_swing 概率。
- **风险**:v2 默认 `stand_start_prob=1.0`(防 RSI 作弊)时该机制是 no-op,须先重审那个默认;
  且我们的终止以模仿包络断为主,失败加权可能反向多采"模仿差"而非"击球差"的相位——考虑按
  击球失败而非终止来计权。

### 中:执行器延迟(设计借 mjlab,数值自标定)
- **人话**:四家都没真开延迟,但我们要上真机,这是最值钱的没人做的轴。mjlab 的参数化最好:
  除了 lag 区间,还有 `delay_hold_prob`(丢包保持上一条命令)——那才像真丢包,不是干净恒延。
- **证据**:mjlab `actuator/actuator.py:67-112,151-224`(delay_min/max_lag、hold_prob、update_period、
  per_env_phase,全默认 0);我们 `robots/actuator.py` 的 DelayedImplicitActuator 是死代码,
  A3 五组执行器全是裸 ImplicitActuatorCfg;A1 球目标延迟/噪声旋钮七个 yaml 全零
  (标定过的场馆噪声 σ_white=0.0019 m、AR(1) σ=0.0052 m 就躺在注释里没启用)。
- **落点**:AGIBOT_A3_CFG 五组换 DelayedImplicitActuatorCfg,挂 `task.plant.actuator_delay_steps: [min,max]`
  (null=逐字节不变);先给我们的 Delayed 类补 hold_prob;**区间从真机部署环路实测来**
  (sim dt 5 ms,实测 10–20 ms 往返即 [2,4] 步),不抄外部(外部全是 0,没数可抄)。
- **风险**:延迟是最容易砸挥拍的轴(触拍毫秒级、厘米级);必须排在推撞之后单独上,
  且注意与 ±15% kp/kd 叠加可能超出真机实际散布。

### 低(记台账,不排产)
- **armature/关节阻尼随机化**(mjlab 有原语没人用):我们 armature 是手抄 MJCF 精确常数;
  若做,startup scale (0.9,1.1) 起步,且须同步更新 MuJoCo 交叉验证协议(否则单点对照失效)。
- **力矩上限随机化**:方向对(只准向下 0.9–1.0,模拟热/老化电机),但加速度包络仅 3.5% 余量,
  0.9 直接不可行,(0.97,1.0) 又窄得没意义——**记为已知未覆盖轴,不排**。
- **`action_acc` 零权重探针**:借 mjlab"不定价但常测"的做法,用我们 v2 屏障探针同款模式把
  jerk 指标在 DeployParity/HitterPure 系点亮(现在权重 0 ⇒ RewardManager 直接跳过 ⇒ 完全失明)。
  零风险,顺手做。
- **腿部 hip_roll/hip_yaw 偏离锚**(unitree -1.0):我们唯一没有类似物的正则族,但与
  racket_progress(+10)的步法收入正对撞;只有 in_hold 门控形态值得试,起步 -0.1。
- **球拍 restitution 钉死**:我们全身 restitution (0,0.5) 随机化把球拍也随机了。现役臂用解析
  虚拟球记分,暂无实害;哪天物理球接触参与记分,这就是暗雷。下次动 event cfg 顺手拆成
  "全身随机 + 球拍钉标定值"。
- **观测历史堆叠**(unitree G1 5 帧,H1/Go2 注释掉了):证据最弱、破坏 110-D 部署观测契约,
  只在做延迟臂时一并考虑(堆叠是让 policy 自己估延迟的标准配方),否则不动。

## 四、全量对照表

(修正后数值;NONE=该库确实没有;"原语未用"=代码存在但零配置接线)

| 维度 | 我们 (HOPE/A3) | BeyondMimic (G1 上游) | mjlab (G1/Go1) | unitree_rl_lab (G1/H1/Go2) |
|---|---|---|---|---|
| kp/kd 随机化 | **±15% 全臂开**(reset 级 log_uniform scale;HOPEPingPong 臂 null;python/base 默认 ±20%) | NONE | 原语未用 | NONE |
| 力矩上限/motor strength | NONE(静态常数) | NONE | 原语未用 | NONE(29dof 实值 88/139/25/5 Nm) |
| armature/关节摩擦 | NONE(手抄 MJCF 常数) | NONE | 原语未用 | 电机摩擦模型是死代码(仅 Go2HV 实例化且 Fs=Fd=0) |
| 摩擦/restitution | 全身 (0.3,1.6)/(0.3,1.2)/(0,0.5) | 同我们 | 仅足底切向 abs (0.3,1.2);无 restitution 原语 | 人形 (0.3,1.0)² + restitution 钉 0;mimic 同 BeyondMimic |
| 连杆质量 | 全身 scale ±15% | NONE | 原语未用 | 仅躯干 add (−1,+3) kg;mimic 无 |
| CoM 偏移 | 躯干 x±0.025/y±0.05/z±0.05 | 同 | 同(velocity 臂更紧) | 仅 mimic 有,同值 |
| 关节零位误差 | ±0.01 rad 双写(obs+动作偏置) | 同 | encoder_bias 仅偏 actor 观测(±0.01/±0.015) | 仅 mimic,双写 |
| 外部推撞 | **现役 N1 无**;Wave-P 6 臂 07-20/21 推过(±0.2/±0.35/±0.5,5–15 s,无判读);v2.3 模板已裁定默认带 push,N1 发射器未继承(§3.1) | **开**:(1,3) s 6-DOF ±0.5/±0.2/±0.52/±0.78 | **开**:同左(play 时摘除) | 步行:5 s 仅 xy ±0.5;**mimic 开**:同 BeyondMimic |
| 执行器延迟 | 死代码 | 死代码(同款) | 字段全默认 0 | 仅 Go2 有字段且未设;H1/G1 无字段 |
| 观测延迟/噪声(球目标) | 旋钮最全(delay/jitter/白噪/AR1/dropout/bias)**全零**;场馆标定值躺注释 | NONE | 字段默认 0(pipeline 含 delay→history) | NONE |
| actor 观测噪声 | 逐通道 Unoise(anchor ±0.25 m…racket_target ±0.02 m) | 基本同值 | 基本同值(velocity 臂 joint_vel ±1.5) | 同构;G1 独有 history_length=5 |
| 一阶动作差分 | -0.1;v2 clamped -0.2 | -0.1 | -0.1 | 步行 -0.05;mimic -0.1 |
| 二阶动作差分 (jerk) | **-0.05 clamp 36(唯一真开)** | NONE | 定义未用,但常测指标 | NONE |
| 关节位置超限 | -40(实际 q)+ -40(q_des pre-clamp)+ -5 投影;DeployParity -10;soft factor 0.9 | -10 | 步行 -1 / 模仿 -10 | 人形步行 -5;mimic -10 |
| 关节速度罚 | -1e-4 | NONE | 定义未用 | G1/Go2 -0.001;H1/mimic 无 |
| 关节力矩罚 | -3e-5(yaml) | NONE | 定义未用 | Go2 -2e-4;mimic -1e-5 |
| 关节加速度罚 | **NONE** | NONE | 定义未用 | **-2.5e-7 全任务族同值** |
| 能耗 | NONE(拆成 vel²+torque² 单卖) | NONE | 定义未用 | G1/Go2 −2e-5 Σ\|q̇·τ\| |
| 课程/自适应 | CurriculumCfg 空 pass;adaptive_sigma 全关;步法课程=手工三段重发 | **失败加权自适应 RSI**(α .001/兜底 .1/λ .8;非因果核) | 同左且为**默认**;velocity 另有地形+指令课程 | 步行地形/速度课程;mimic 无课程 |
| PPO/控制 | 50 Hz,[512,256,128] ELU,lr 1e-3 自适应 KL,entropy 0.01,4096–8000 envs,10–16 s | 同骨架,entropy 0.005,30k iters | 同骨架 | 同骨架,50k iters(mimic 30k) |

## 五、尽调途中的侧发现(非本题,但要记)

1. **[PLAUSIBLE,待 dry-run 定谳] 旧臂可能已无法默认发车**:2026-07-25 裁定 `reward_pack` 缺省=v2 后,
   v2 DIRECT 表的非零项(如 `virtual_landing 1648.8`、`hit_unstable_support -10`)在
   HitterPure/Hitter/DeployParity/HOPEPingPong/Rally(V3) 的 rewards cfg 类里**不存在**,
   `_expand_reward_pack` 的 `_require` 会在配置合成期 fail-loud——即这些臂现在必须显式
   `reward_pack=v1` 才能起。静态分析高置信(已核对 `train.py:9041-9054` 跳过逻辑仅豁免零权重项),
   未实跑;测试套只有 DeployParity 形状的 mock,盖不到。若属 07-25 裁定预期,补一行文档即可;
   若非预期,是发射工序雷。
2. 陈旧注释两处:`hope_env_cfg.py:991` 注释写 ±10% 实为 ±15%;Hitter yaml 里 IdealPD 时代的
   rationale 文案已过期(代码侧已注明)。
3. `arm_torque_saturation` yaml 写 -0.5 但组装期强制清零(A3 是 ImplicitActuator,无 pre-clip 需求)——
   行为正确,但 yaml 读者会误以为它活着。

## 六、和现行决定的关系

- 本文不改变"速度泛化优先"的排序;推撞/延迟/joint_acc 都排在速度混合臂之后,除非重新裁定。
- 所有"落点"均为**新臂 yaml + CLI 键、默认逐字节不变**的形态,符合精简治理(真护栏保留,
  不加仪式);一格 smoke 通过即可入队列,不给 GPU 分角色。

---

## 七、误差→奖励核形状专项(07-31 补,应"误差大时梯度太低"之问)

**方法**:4 抽取(我方核清单 + 三外部库)+ 4 对抗核查(逐条重推公式与饱和半径,共 15 处修正)
+ 1 综合。数学基准:我们所有核都是 `exp(-e²/σ²)`(e 为真欧氏/测地距离,3 维求和在 exp 内是
度量本身,不是聚合选择);reward 跌破峰值 1% 在 **e=2.146σ**,梯度跌破自身峰值 1% 在
**e≈2.53σ**(梯度峰在 0.707σ)——"reward 小"和"梯度死"不是同一半径。

### 7.1 判定(人话):**成立,但只在 3 个窗内 task 核上;机制和直觉略有不同**

**病灶确凿的 3 项(全是 strike-window 门控的 task 核):**

| 项 | 现役 σ | 梯度峰 | reward<1% | 现实运行误差 | 病情 |
|---|---|---|---|---|---|
| `racket_normal`(w 0.5) | 0.262 rad | 10.6° | **32.2°** | 冷启动实测 33°/53°/116°(repo 自己的病历,[hope_rewards.py:3471-3481](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py)) | **最重**:所有记录在案的冷启动全在死区,116° 时 reward≈1e-26,拍面连"往哪转"的信号都没有 |
| `racket_position`(w 4.0) | 0.075 m | 5.3 cm | **16.1 cm** | bh_loop_c 触点离中立站姿拍位 ~0.70 m(机制表述经 §9.3 修正:此距离适用于**学习早期、策略尚不会挥拍时的窗口开启时刻**,非回合起点——起点在核峰值) | e/σ=9.3→exp(-87)≈1.5e-38:数值上可表示,但加进 1–10 量级的每步收入后差 30 个数量级,在 float32 分辨率下是**精确 no-op**——"差 20 cm 的近失"与"差 70 cm 的挥空"advantage 完全相同 |
| `racket_velocity`(w 0.5) | 0.5 m/s | 0.35 m/s | 1.07 m/s | 参考拍速 1.81 m/s(manifest) | 最轻:窗开时近静止拍 vs 1.8 m/s 目标 ≈2e-6 峰值,病在但量级尚可 |

**关键 nuance(比"误差大梯度低"更精确的病理)**:接近阶段本来就不靠这些核——它们的窗口
mask 在 pre_strike 时是关的,接近梯度由 `racket_progress`(w 10.0,±0.15 m/步 clamp 的距离
差分,**任意误差下常梯度**,且 pre_strike 与 strike_window 在 tts∈(0,0.1] 重叠,无交接断层)
供给。真正的伤害是:**窗内的失败之间没有排序信号**——策略需要"这次比上次近了"的信号来
爬完最后 15 cm,而现在近失和挥空同零。唯一裸奔段是 follow-through 半窗(tts∈[-0.1,0]),
0.1 s 内只有饱和核在场。

**不成立/已缓解的项**:全套 imitation 核(σ=0.3 m/0.4 rad/1.0/3.14,饱和半径 0.64 m/49°/…,
对正常跟踪误差绰绰有余;聚合是 exp-of-mean,与三家外部逐字一致);`racket_progress`(构造性
免疫);`virtual_landing`(σ=1.0 m + 60% 合法保底,饱和不是它的问题——它的问题是硬 AND 门:
三个 N1 profile 的 `strike_opportunity_count` 全零,稀疏性压在饱和之上把它整个掩着);
`upright_exp`(误差上界 1.0 < 死区 0.96,物理上进不了死区);两个 barrier(单调有界 cost ramp,
不同族);`base_position`(σ=0.20 m 跟随 spawn 尺度,最远角 0.50 m 恰在梯度 1% 半径上,轻度);
`motion_global_anchor_ori`(σ=0.4 rad 饱和半径 0.858 rad,而终止阈值 0.8 rad **先杀**——
终止耦合在这项上偶然地成立,racket 系无一有此保护)。

**σ 的设计哲学错位(病根)**:racket 系的 σ 是**验收容差**(docstring 明说"step-14 acceptance
tolerances,阈值处 reward≈exp(-1)"),base_position 的 σ 跟随 **spawn 尺度**——后者才是
运行误差尺度的正确选法。外部三家的窗内核之所以不出事,不是核宽,而是 **RSI 把每集出生点撒在
参考轨迹 ±0.05 m/±0.2 rad 内**(运行误差/饱和半径 ≈0.08),我们 stand_start=1.0 后这个比值
是 **4.3×(位置)/1–3.6×(拍面)**。这正是 §9(reset 专项,另行补充)与本节的交叉点。

### 7.2 两个结构性放大器(先记着,别踩)

1. **N1 wave 只消融权重、不消融 σ**(current_low 4/0.5/0.5 vs task_strong_x4 16/2/2):
   死核乘 4 还是 0——**这个 wave 无法区分"饱和无害"和"饱和是卡点"**。
2. **v2 校准权重表(393.4/295.1/229.5)是死的**:yaml 显式权重按"用户覆盖优先"规则赢
   ([train.py:9028-9034](../../hope_training/whole_body_tracking/scripts/train.py))。**不要在修 σ 之前"修"这个覆盖**——那等于给死核乘 100,
   0.70 m 处依旧为零,却在策略首次进入有效带的瞬间制造 100 倍收入悬崖。

### 7.3 当前真正 binding 的还不是核(先做零成本诊断)

N1 实况(`n1_live_wave_4ff48b21.v1.json`):`strike_opportunity_count=0` 三 profile 全零,
episode 在稳定到窗之前就被砍——bh_block 是关节 fault 主导(30730+75),但旗舰
bh_loop_c/current_low 在 update 35 是 **ee_body_pos=2246(模仿跟踪终止)> 关节 fault 合计
1448**,"关节 fault 主导"的笼统说法对旗舰不成立。**结论:光修核不会立刻出击球;先修
终止/安全压力,或先证明核死是下一层卡点。**

零成本第一步:把已有的 `racket_target_distance` 指标([hope_commands.py:16323](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py))**按窗口开启
时刻条件化**记录——分布若在 ~0.7 m,核死实锤;若已 <0.2 m,饱和故事不成立,卡点纯在终止层。

### 7.4 方案菜单(按序;P0 前先跑 7.3 的诊断)

| # | 方案 | 外部先例 | 落点 | 风险 | 裁量 |
|---|---|---|---|---|---|
| P0 | **σ 课程阶梯**(静态、跨臂收紧;face 先行:0.60→0.40→0.262 rad;pos 0.30→0.15→0.075 m) | 无一家做训练时 σ 调度(三家靠 RSI 回避);mjlab manipulation 的粗细双 σ 是部分类比 | **零代码**:`racket_*_std` 键已全程接线(train.py:8590/8705/10215);新臂 yaml `defaults: ActionBall` 只改 std | 低-中:拓宽 σ 给平庸状态发钱,但 pos 项 w4.0×±0.1 s 窗,增量收入对 imitation 每步 4×1.0 是小数 | **首选**,符合"静态权重+σ课程"钦定工具 |
| P1 | **粗细双核求和**(`w_f·exp(-e²/0.075²)+w_c·exp(-e²/0.30²)`,粗核从 ~0.6 m 起供梯度,细核保验收语义) | **mjlab 唯一实弹先例**:lift_cube 同误差挂 bring_std 0.3 + 0.05 双核 | 新 RewTerm `racket_position_coarse` 复用 `_pos_kernel_raw`,默认 0.0 逐字节不变 | 中:新增"大概到位"收入层(位于击中层之下,哲学上正位);粗核收入须 < 击中层 | 结构上最干净的**永久解** |
| P2 | **线性 face guidance 重启**(`min(angle,θ_max)` 常梯度罚,repo 已内置且 docstring 写明就是为这个病造的;θ_max=π 小剂量) | 无(三家皆无同误差 L1+exp 并联) | 已接线 w=0.0:[hope_rewards.py:3466-3494](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py) + train.py:11302(强制非正) | **中-高:负项,直接踩"软罚压制击球"红线**,排 P0/P1 之后单臂 A/B | face 通道专用杠杆 |
| P3 | **pos_gate 开启**(sigmoid 够得着才发拍面/拍速钱,已实现默认 None) | mjlab staged reach×bring 同角色(乘法 exp 形态,更陡) | train.py:11278 已穿参;设 `rewards.face_gate_by_pos` | 低:只减坏状态收入,不加罚;别单跑(会掩盖 P0 效果) | 伴随项 |
| P4 | **自适应失败 bin RSI**(三家全默认开、是它们窄核可行的根本机制) | **最强外部信号** | 但被 [ActionBall.yaml:49-66](../../hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBall.yaml) 的 canonical-ready 裁定显式禁用(stand_start=1.0) | **治理题不是代码题**:动它 = 重定义 action-ball 谱系的"true reset";须 Franco 点名 RSI 作弊为何不回来 | 上呈裁量,不擅自开臂 |
| P5 | mean-of-exp 聚合改造 | 无一家用 mean-of-exp;racket 核上 mean-over-dims ≡ σ×√3(改σ即可) | — | imitation 栈没病还要动共享基座,高爆低益 | **否决**(记档防再提) |
| P6 | racket 误差终止耦合 | 模仿误差有先例(且三家实现都漏:z-only/yaw-blind),task-goal 误差无先例 | — | death_penalty −72/次砍在"没够到球"上,教科书级压制击球 | **否决**;可考虑反向(诊断臂放宽 ee_body_pos 0.25 m 验证 7.3) |

### 7.5 四库核设计对照(修正后)

| 维度 | 我们(ActionBall+v2) | BeyondMimic | mjlab | unitree_rl_lab |
|---|---|---|---|---|
| task 核形式 | `exp(-e²/σ²)`,racket 三通道,窗门控 | **无 task 通道**(纯模仿) | 速度指令核 + manipulation reach/bring 同形 | 速度指令核同形 |
| imitation 聚合 | exp-of-**mean**(上身 swing_only) | exp-of-mean,14/17 body | 同 | 同 |
| 现役 task σ | pos **0.075 m**/vel 0.5/normal **0.262 rad**/base 0.20/landing 1.0+0.6 保底 | n/a | lin_vel 0.5;manip 粗 0.3+细 **0.05 双核** | lin_vel 0.5;foot_clearance 0.05(该库唯一 σ 不平方项) |
| σ 选自 | **验收容差**(racket)/spawn 尺度(base) | 运行误差尺度 | 运行误差尺度(manip 显式粗细对) | 运行误差尺度 |
| 运行误差÷饱和半径 | **4.3×(pos)/1–3.6×(face)→死区** | ~0.08(RSI 保证) | ~0.08 | ~0.08 |
| 终止 vs 饱和半径 | anchor_ori 0.8<0.858 ✅(偶然);racket 误差**零终止覆盖** | anchor_pos 终止只查 **z 分量** vs reward 全 3D ❌;SMPL 的 ee_body_pos 是死 no-op | 同 z-only 漏洞;body_ori/速度核零覆盖 | z-only 0.25 m 在 ~50% 峰值处触发 ✅(三家最紧);ori 78° vs 49° ❌ |
| RSI | **裁定禁用**(stand_start 1.0) | 默认开,每 reset | 默认开 | 默认开 |
| 自适应失败 bin | 无 | 开(α .001+10% 兜底) | 开(默认) | 开 |
| 距离差分 progress 项 | **有,w 10.0(四家唯一)** | 无 | 无 | 无 |
| 同误差 L1 并联 | 已建未开(guidance 双子) | 无 | 无 | 无 |

**一句话总结**:外部三家从不修核——他们用 RSI+终止把 policy 关在核的有效带里;我们把 RSI
关了(有裁定理由)、终止又没盖 racket 误差,于是唯一能在死区供梯度的就剩 `racket_progress`
和(待启用的)P0-P3。方案排序:**先 7.3 诊断,再 P0(σ 阶梯)+P3(gate)同臂,P1 做永久解,
P2 单独 A/B,P4 上呈治理,P5/P6 否决留档。**

---

## 八、其余训练轴 sweep(07-31 补:探索/归一化/视野/动作管线/配平/对称性等 15 轴)

**方法**:4 抽取 + 4 对抗核查(20 修正/14 漏项)+ 1 综合;下文为修正后结论。
先把四个头条问题的裁决说清,再列雷区与可借项。

### 8.1 四个头条问题的裁决

1. **`init_noise_std=0.02`:正当,但理由和"warm-start"无关,且证明只盖第 0 步。**
   N1/ActionBall **全部 fresh 起跑**(launcher 从不发 checkpoint_path;bootstrap 在
   checkpoint_path≠None 时拒绝应用)。0.02 的真正理由:actor 末层被 bootstrap 成
   weight=zeros / bias=ready 姿态,[train.py:5004(**§21 核查修正:零初始化末层实际在 scripts/train.py:7010/7062/7065,调用点 16237;5004 是 init_noise_std 的强制校验点**)](../../hope_training/whole_body_tracking/scripts/train.py) 强制 0.02 精确等值,
   [training_contract.py:4132](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/training_contract.py) 逐关节证明 4σ×action_scale+startup 偏移带严格卡在
   硬限位内 2% 包络里;σ=1.0 会在第一步就打穿限位。外部三家全是 σ=1.0——因为没人做
   末层清零的安全 bootstrap,**外部共识在这个轴上不适用**。
2. **obs 归一化两谱系分裂:现役 ActionBall 走的是"活"路径(判定),但 fresh 跑零运行时证据(雷)。**
   preflight/ckpt_compat 的 2×2 真值表、exporter 的 `actor(normalizer(x))` 烘焙、
   make_std_sidecar 的真实 mean/var 抽取都指向 virtual-ball 谱系的活归一化;但这些护栏
   **只盖 resume 和导出**,fresh 长跑期间没有任何断言证明 normalizer 不是 Identity,
   而生死由 pod 上的 rsl_rl 版本决定(repo 无版本断言)。见雷区 #1。
3. **timeout 自举:完好。** 自定义 runner 只包 telemetry,rollout 走 `super().learn()`;
   IsaacLab wrapper 每步写 `extras['time_outs']`,rsl_rl 照常对截断加 γ·V(s_T)。四家一致。
4. **γ=0.99 信用饥饿质疑:量化证伪,正式关掉这条线。** `virtual_landing` 在**触球步即发**
   (解析弹道 `vb_landing_xy`,不等真球飞;ActionBall `settle_delay_s=0.0`)——因果零延迟。
   最长真实因果窗:起手→触球 0.76 s(保留 68.3% 信用),最慢 teacher_rate 0.6 → 1.267 s
   (52.9%)。半衰期 1.379 s 吃得下。**下轮审计别再捡起来。**

### 8.2 雷区(按严重度)

| # | 雷 | 严重度 | 人话 + 处置 |
|---|---|---|---|
| 1 | **fresh 跑的 obs 归一化无运行时证据** | 高 | `empirical_normalization: true` 是 runner 级老键,rsl_rl 3.1.2 线上会被 IsaacLab 塞的 `actor_obs_normalization: {}` 静默吃掉(我们 ppo.yaml 注释自己写的),失败模式无报错、只是学得更难。**处置**:(a) runner 构造后加 fail-loud 断言 normalizer 非 Identity(fresh/resume 都跑);(b) rsl_rl 版本写进训练 receipt;(c) 先去现役 N1 checkpoint torch.load 看 `obs_norm_state_dict` 在不在——一条命令定谳 |
| 2 | **σ=0.02 的安全证明只在 t=0 成立,之后裸奔** | 高 | scalar 参数化下 std 是可负的裸 nn.Parameter,无地板/上限/告警,单次迭代即可穿零→NaN(静默)。**处置**:先拉现役 N1 的 `Policy/mean_noise_std` 前 200 迭代——爬离 0.02 = 探索没死,只补护栏;趴着/下行 = 探索被掐,再议 entropy。护栏 = 新臂 `noise_std_type: log`(恒正,我们 pin 的 rsl_rl 原生支持,默认 scalar 字节等价) |
| 3 | **adaptive-KL × σ=0.02 的未标注交互** | 中 | 高斯 KL 主项 (Δμ/σ)²:σ 从 1.0→0.02 放大 **2500 倍**,desired_kl=0.01 早期疯狂触发 lr/1.5,可能把 LR 按到库内地板 1e-5——恰在 weight=zeros 的 actor 最需要动的时候。外部无人踩过此组合(全在 σ=1.0)。**处置**:与 #2 同一次数据拉取看前 500 迭代 LR 轨迹;实锤则新臂单变量 A/B(entropy 0.02 或 desired_kl 0.02 二选一) |
| 4 | **ppo.yaml `max_iterations: 3000 亿`** | 中 | 绕过 launcher 手起 = 无声无限跑。**处置**:默认改 25000(团队 2 万-2.5 万上沿);launcher 路径本就显式传 20001,字节等价 |
| 5 | γ 变频无补偿机制 | 低 | 现役被 ActionBall 的 decimation==4 断言挡住;beyondmimic 有 γ^(1/freq_scale) 公式。**处置**:进"新臂检查单"文档,不改代码 |
| 6 | 关节摩擦系数 UNCALIBRATED → formal MuJoCo 评估 fail-closed | 低(训练)/高(交付门) | 已知且自我记录;现役 N1 checkpoint **无一具备正式 sim-to-sim 评估资格**,需要时须另立零摩擦契约臂。排交付计划时明写 |
| 7 | tracking_env_cfg.py:106-110 clamp 注释过期 | 低 | 实际默认 clamp=True(07-06 裁定),注释还写 False。顺手改 |

### 8.3 可借项

1. **`noise_std_type: 'log'`**(高):对雷 #2 最便宜对症;mjlab 所在的 rsl-rl 5.4.0 另有
   std_range 夹紧同证方向。落点:ppo.yaml 加键(默认 scalar)+ ppo_cfg.py 透传,新臂启用。
2. **归一化运行时断言 + 版本入 receipt**(高):mjlab 把归一化下沉到 model 层逐 task 显式,
   架构性免疫我们的 #1;我们用现成 `is_empirical_normalizer` 三行实现同效。
3. **per-ObsTerm 手工 scale 兜底**(中,条件触发):unitree 全家 `empirical_normalization=False`
   但每个 ObsTerm 手写量纲 scale(joint_vel 0.05、ang_vel 0.2)——若 #1 查出归一化是死的,
   这是正确的替代;反例警示:beyondmimic Humanoid 臂 False 且无 scale,裸吃生 obs。
4. **max_iterations 天花板**(中):三家全有限值(30000/50000/…),我们 3e11。
5. **entropy 方向判断记档**(中):外部共识"模仿族 0.005 < 探索族 0.01"**不适用于我们**——
   他们在 σ=1.0 上,我们在 σ=0.02 上,entropy 是唯一把 std 推离零界的力。**防止下轮审计
   照抄外部往下调**;若 #3 实锤,方向是往上 A/B。
6. **γ 变频公式**(低):beyondmimic 独有,进检查单即可。

### 8.4 十五轴四库对照表(修正后)

(见附表;要点:PPO 骨架四家逐字相同——lr 1e-3/adaptive/KL 0.01、clip 0.2、5×4、
γ 0.99/λ 0.95、24 步、50 Hz——真正分化的只有:init_noise_std(唯我们 0.02)、
obs 归一化策略(四家四样:老键/True/model 层显式/False+手工 scale)、RSI(唯我们关)、
q_des clamp(唯我们有,deploy 对齐裁定)、entropy 族内分档(唯我们不分)、
episode 10 s vs unitree mimic 30 s。)

| 轴 | ours (N1) | beyondmimic | mjlab | unitree_rl_lab |
|---|---|---|---|---|
| init_noise_std | **0.02**(CLI 强制精确等值) | 1.0 | 1.0 | 1.0 |
| noise_std_type | scalar(无地板) | scalar | scalar(库层 std_range 夹紧) | scalar |
| actor 末层 init | **zeros+ready bias**(fresh-only) | PyTorch 默认 | PyTorch 默认 | PyTorch 默认 |
| entropy_coef | 0.01 全局 | 0.005 | 0.01 velo / 0.005 track | 0.01 loco / 0.005 mimic |
| obs 归一化 | true(runner 老键,fresh 无验证) | True(G1)/ **False(Humanoid,裸奔)** | True(model 层显式,actor+critic 独立) | False + **手工 scale** |
| timeout bootstrap | 完好 | 完好 | 完好 | 完好 |
| γ/λ/半衰期 | 0.99/0.95/1.379 s | 同(低频变体重取幂) | 同 | 同 |
| 结果奖延迟 | **0 s**(触球步即发) | 无结果奖 | 无结果奖 | 无结果奖 |
| episode | 10 s | 10 s | 20/10 s | 20 s loco / **30 s mimic** |
| action 语义 | delta+per-joint scale(0.25·τ/kp)+ **q_des 硬 clamp** | delta+同公式 scale,无 clamp | 同,无 clamp | loco 扁平 0.25;mimic 同公式;无 clamp |
| 多 clip 配平 | **平衡轮询采样器**(计数差≤1,可断点续采;N1 单 clip 下退化) | 单 clip/run | 单 clip/run | 单 clip 连续回放 |
| 对称性增强 | 无(symmetry_cfg: null) | 无 | 无 | 无 |
| obs history | 0 | 0 | 0 | G1-loco 5;其余 0 |
| reward/advantage 归一化 | 无/全 batch | 同 | 同 | 同 |

---

## 九、Reset 设计专项(07-31 补;含对 §3/§7 三处结论的修正)

**方法**:2 抽取(我方深挖 + 三外部合并)+ 2 对抗核查(4 修正/10 漏项)+ 1 综合。
本节修正了前两轮的三个结论,先列修正再讲诊断。

### 9.1 三处上游修正(以本节为准)

1. **"v2 默认 stand_start_prob→1.0"对所有已注册 task 都是 no-op**:六个 yaml 全部显式设了
   这两个键("user override wins",[train.py:9104-9112](../../hope_training/whole_body_tracking/scripts/train.py))。实况:DeployParity/Hitter 的 RSI
   活在 **50%** 真 reset 上(stand 0.25/post 0.25),HitterPure/Rally 是 **75%**(stand 0.25)。
   100% 站姿起步**只有 ActionBall/N1**,且是 `canonical_ready_mode` 验证器强制的
   ([commands.py:1487-1532](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py)),不是 07-26 v2 裁定的直接产物。
2. **N1 真 reset 是零噪声,不是"与外部同噪声"**:§2 表里"RSI reset 噪声与三家逐字节一致"
   只描述 legacy 路径;canonical_ready 验证器硬性要求 pose/velocity/joint/yaw 四组区间
   **精确 (0,0)**,root/关节速度硬清零——N1 的起点分布是**每 action 一个 delta 函数**
   (下身 12 关节与默认站姿逐位相同,上身 19 关节为该 action 的引拍姿态)。
3. **"我们没有失败加权自适应 RSI"要改为"机制在库内、结构性死亡"**:单 clip 分支
   ([commands.py:4633-4684](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py))实现了与 BeyondMimic **同常数**(α=0.001、
   兜底 0.1、λ=0.8)的 bin-EMA 采样器,带 checkpoint 续采;但 multiseg 谱系永远取 clip 第 0 帧、
   action-ball 直接短路给 birth broker——**所有已注册 task 都跑不到它**。§3 的可借项据此降级:
   不是"抄外部",是"给自己已有的采样器接线"。

### 9.2 诊断:N1 的起点分布塌缩与自我收窄循环

- **外部三家**:每次 reset 传送到 clip 内**随机相位**(失败加权),复制该帧参考速度,再加
  噪声球——起点分布在"clip 相位 × 参考流形邻域"上**全支撑**,失败相位加权。
- **我们(N1)**:有限 delta 函数集。**引拍之后的所有相位(加速、触球、随挥、恢复)永远
  不是回合起点**,只能靠策略自己从 ready 姿态滚过去——这正好倒置了 RSI 的发明初衷
  (给失败相位加质量,我们给它们零质量),也和已知缺口"随挥后无恢复专项训练"同源。
- **wrap 是唯一的多样性来源,而且会退化**:回合内第 2..N 拍经 wrap 从策略自己留下的状态
  开始(on-policy、不可控)。当前 #1 吞吐问题 `joint_actual_forbidden` 每 update ~4700-5100 次
  终止 → 回合更短 → **wrap 更少 → 更大比例的拍从同一个 ready 姿态开始 → 中后相位经验更少
  → 策略更差 → 更多终止**。失败模式在收窄自己的训练分布。
- **前缀税的准确形态**:不是"走路到位税"(下身本来就是站姿),是**固定相同前缀**:每回合
  重放同一 ready → 同样零 hold → 同一第一拍。

### 9.3 与 §7 的真实耦合(修正 §7.1 的机制表述)

§7.1 说"stand_start=1.0 使每集从 0.70 m 全距离出发(核死区)"——**这把回合起点和窗口开启
混为一谈了**。准确图景:canonical-ready 起点的模仿跟踪误差为**零**(核峰值、最大梯度区);
外部 RSI 也落在参考上再加噪——**没有人从饱和区出发**。饱和是**尾部**问题:rollout 漂移超过
~2-3σ 后核变平。外部有两个机制把 env 关回有效带:(i) 跟踪保真终止砍掉饱和尾;(ii) RSI 把
env 重新注入高梯度参考态。**N1 两个都关了**:RSI 被验证器禁;保真终止被发射器钉成
`+task.racket.reference_guard_mode=metrics_only`([launch_n1_reward_screen_diagnostic.py:1511](../../hope_training/whole_body_tracking/scripts/launch_n1_reward_screen_diagnostic.py),
cfg 默认本是 phase_gated)——**关它的理由是吞吐(reset 风暴),不是学习论**。于是漂移的 env
既不被砍也不被再注入,在平坦区把 10 s 烧完。§7.1 的"0.70 m"数字仍适用于**学习早期策略还
不会挥拍时的窗口开启时刻**,结论(racket 核死区真实存在)不变,机制表述以本节为准。

**全篇最高杠杆的因果链**:真 reset 的 ~7 ms/env Python 成本(~~canonical_sha256 八处调用点
无缓存重算~~【§15 修正:07-30 已缓存,残余成本为】逐 env 证明转录 ~9 次 `.item()`、O(N²) 区间扫描、receipt 链逐 env Python;实测 25-116 s/iter,健康带
20k-40k steps/s)让"reset 频率"变成被征税资源,保真终止因此被关,分布病理因此产生——
**修好 reset 性能,保真终止就重新买得起,§7 的一半问题跟着解**。

### 9.4 反 RSI 裁定:对我们比对外部**更**成立,不重开

Cheat 的精确命名:**mid-swing airdrop**——传送到挥拍中段第 k 帧会连带复制参考速度,拍子
白拿教师积累的动量。在 BeyondMimic 里这只是自限性捷径(唯一收入是 tracking,接不下去就
不挣钱);**在我们这里是记分漏洞**:`RacketTargetCommand` 在同一次 resample 发新球任务
([hope_commands.py:14168-14182](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py)),airdrop 的 env 能为一次"整个进场都是白给的"
击球**领 hit 收入**。07-26 裁定封的就是这个,不重开。但裁定封的是**一种机制**,不是
"起点多样性"这个目标——下面的方案全部绕开该 cheat。

### 9.5 方案(按依赖排序;P0 是后面一切的解锁前提)

| # | 方案 | 要点 | Cheat 暴露 |
|---|---|---|---|
| R0 | **给真 reset 降本**(~~memoize canonical_sha256 八调用点~~ **§15 修正:07-30 `7e98c3d6` 已加 `@_WeakIdentityCachedCanonicalSha256` 缓存,此项已完成**;剩余:证明转录懒化/GPU 化、O(N²) 扫描、逐 env receipt 链批量化;assert_contract 原样保留) | 外部先例:legged_gym reset O(kernel) 不 O(env);**这是 R1/R3/R4/R5 的 blocker** | 无 |
| R1 | **恢复 N1 保真终止 verdict**(metrics_only→phase_gated;先读现成 raw 预测器的触发率定价 reset 增量,再翻) | 三家全常开无开关;砍饱和尾=§7 外部机制的免费一半;不可能压击球收入(它只砍已经不在挣钱区的 env) | 无 |
| R2 | **拆 canonical_ready_mode 双职能**:契约绑定(hash/字节等价/禁 wrap 传送)与 reset 分布裁定分开,新键 `canonical_ready_strict_reset: true` 默认字节等价 | mjlab 的 sampling_mode Literal 是 API 先例;R3-R5 的 enabler,本身零行为变化 | 无(只是把门从焊死改为带锁) |
| R3 | **失败加权选 action**(起点集合完全不变,只把 round-robin 换成失败加权抽签;复用库内 bin-EMA) | 买到自适应 RSI 的课程一半,零 airdrop 风险;**多样化家族里性价比最高** | 无 |
| R4 | **ready 姿态噪声球**(分级:先关节 ±0.02,再速度,最后 root 位姿/yaw) | 外部三家逐字节同款噪声;摆脱测度零起点集 | 低(root 噪声若不重解题目几何会压击球——所以 root 排最后) |
| R5 | **post-swing buffer 对 N1 开启**(DeployParity/Hitter 已活跑 25% 无事故;给中后相位放质量的最便宜方式) | 自家先例;两件实活:birth broker 消费 post-swing 状态、ledger 对齐 | 中(状态是策略自己产的,无教师动量;但须审 hit 记账) |
| R6 | **恢复期专用 RSI**(`rsi_phase_window` 仅允许触球之后的相位;失败加权采样器本来就在库内) | 唯一结构上买不到击球的参考 RSI;直指"无恢复专项"缺口 | 本表最高,须 R2 白名单 + 触球前硬禁 |
| R7 | phase-gated 推撞(≡ §3.1,战术上排最后) | 见 §3.1 | 中 |

### 9.6 五列对照表(live / 已建未开 / 三外部)

| 维度 | 我们-live(N1) | 我们-已建未开 | BeyondMimic | mjlab | unitree |
|---|---|---|---|---|---|
| 真 reset 起点 | 该 action 的 canonical-ready 帧字面量(root XYZ 由 birth broker 改写) | RSI(DeployParity/Hitter 50%、HitterPure/Rally 75% 活跑);post-swing 回放(25% 活跑) | clip 随机帧(自适应 bin) | 随机帧(adaptive/uniform/start 三模) | 随机帧(BeyondMimic 逐字 fork) |
| 起点的 clip 相位支撑 | 每 action 一个点(delta) | multiseg=永远第 0 帧;单 clip 路径=全 clip 连续 | 全 clip 连续 | 连续或第 0 帧(play) | 全 clip 连续 |
| 失败加权采样 | 无(短路给 birth broker) | **库内已实现、checkpoint 安全、结构性死亡**(同常数 α .001/λ .8/兜底 .1) | 常开无开关 | 训练默认 | 常开无开关 |
| root 位姿/速度噪声 | **全零(验证器强制)**,速度硬清零 | ±0.05/±0.01/±0.1/±0.2 + 参考帧速度+噪声(legacy 路径) | ±0.05/…/±0.2 + 速度 ±0.5/±0.78 | 同 | 同(locomotion yaw ±π) |
| 关节噪声 | (0,0) | ±0.1(soft-limit 截断) | ±0.1 | ±0.1 | ±0.1 |
| clip/action 抽取 | 平衡轮询(seed 0,确定性) | uniform randint | bin 内 uniform | uniform/adaptive | uniform |
| 保真终止 | **metrics_only(launcher 钉死)** | phase_gated(cfg 默认) | 常开(anchor 0.25/0.8, ee 0.25) | 常开 | 常开 |
| reset 级 DR | 仅 pd_gains(log_uniform ±15%) | 推撞三套 | 速度踢+interval push | 同 | 同(5 s 固定) |
| 每 reset 成本 | **~7 ms/env Python**(25-116 s/iter 实测) | — | 一次 multinomial+张量写 | 同量级 | 同量级 |

---

## 十、循环解剖学普查:update 里都在做什么(07-31;硬件口径 = RTX 5090)

**定位**:update/循环的时间账主体在 [design_audit_and_speedup_20260729.md](design_audit_and_speedup_20260729.md)
§3/§5/§9(含与 yikang r2fqs 的功能 diff)。本节补它没做的一块:**与 BeyondMimic/mjlab/
unitree 三家的逐维循环普查**(2 普查 + 2 对抗核查 + 1 综合),并按两处实况修正重算:
(a) 硬件是 **RTX 5090**(audit 的"V100 外推 20k-40k 带"作废;自家实测锚点:旧代 4096 env
独占 **2.0-2.2 s/iter**、physical_ball 代际 **4.6 s**,runbook"RTX 5090 实测算力手册");
(b) **升级已落地**:现役 N1 反手挡长跑 update 250 处 **23.48 s/update**,actual-hard
1420→116、qdes forbidden=0——audit 工单 #1(joint_actual 风暴)已兑现,残差不再是终止。

### 10.1 HEAD 重数:已修好的与还在的

**已修好(实证,audit 旧账作废):** 每步同步大簇已批量化(`_batched_host_scalar_values`
一次 `.cpu().tolist()` 替掉 10+8N 簇;strike-timing 重复调用已去重)——每步任务级同步从
60-120 降到 **~7-8**;"32 个 filtered sensor"实为 **1 个 ContactSensor×32 body×5 collider 列**
(口径修正,问题仍在但形态不同);logging 一直是每迭代不是每步。

**结构差(四库普查,修正后):**

| 维度 | 我们(N1 HEAD) | BeyondMimic | mjlab | unitree |
|---|---|---|---|---|
| live reward 项 | **29**(65 注册/36 死项也在构造) | 9 | 9 | 11 |
| 终止项 | **9** | 4 | 4 | 4 |
| actor obs 项 | **17(194-D)** | 8 | 8 | 6 |
| CommandTerm | **2**(racket_target 每步 FK+扫掠接触+EMA) | 1 | 1(+wrap 时 sim.forward) | 1(BeyondMimic 逐字拷贝) |
| 每步任务级 host 同步 | ~7-8 | ~3(含自适应采样 torch.any) | ~3 | ~3 |
| 每 reset host 读 | **~24 次/env**(.item()/H2D 往返)+ sha256×8 + 22 键档案转录 | **17 次/批,与 env 数无关** | 同 | 同 |
| reset 级 EventTerm | 1(pd_gains) | **0**(reset 全在 command 内) | 0 | 0 |
| env-reset 频率 | **33.0 次/policy step**(episode 124) | 8.2(episode ~500) | 8.2 | **2.7**(episode 30 s) |

关键外部硬数:**四家参考栈的 reset 成本都是每批固定 17 次 host 读、零逐 env Python**——
这一维的差距不是 2-4×,是 **~3 个数量级**(我们 17-22 ms Python/env vs 他们 8-20 ms/整个 update)。

### 10.2 23.48 s 的分解(5090 口径,算术自洽)

| 成分 | 估计 | 依据 |
|---|---|---|
| solver/场景底 | **5-8 s**(21-34%) | physical_ball 锚点 4.6 s 为下界;N1 多 6 collider/env + 5 列 filtered 接触矩阵 |
| 每步 host 税 | **0.7-2.0 s**(3-8%) | 7 vs 3 同步差×24 步 + 29 项 reward 循环 + 17 项 obs + 9 终止 + 2 command |
| **reset 仪式** | **13.5-17.8 s(60-75%)** | 残差;÷792 env-resets/update = 17-22 ms Python/env,与 ~24 次往返+sha256×8+档案转录的 CPython 成本吻合;参考栈同 792 个 reset 总共 ~8-20 **ms** |

**6-8 s 目标线在 5090 上的裁决:校准得当但不保守。** 两条独立算路都落带内
(A:reward 项比 3.22× × 旧代锚 2.0-2.2 s = 6.4-7.1 s;B:场景底 4.6 + manager 增量 ≈
6.0-7.5 s);普查预测全工单完成后落点 5.8-10.3 s。**分级门**:gate A = 9-11 s(仅 reset
O(term) 化)、gate B = 6-8 s(**必须**加 contact 视图合并——它不是配菜,是 6-8 s 在算术上
成立的前提)、stretch <5 s(需"审计移 checkpoint 边界"的结构裁定)。校准点:yikang 同硬件
6.383 s(4 终止、单一无过滤接触视图、零逐 env 仪式)。

### 10.3 对 audit 工单的增删改(标记制)

**ENDORSES(外部证据加固):**
- reset 从 O(env) 变 O(term)(audit §9.1 #1-3):4/4 参考栈实证可达,硬数 17 读/批;
- **episode 长度是 reset 税基的免费乘数**(audit §9.4"宽核续命"定量化):我们 reset 频率是
  BeyondMimic 的 4×、unitree 的 12×;episode 124→300 步即砍 2.4× 税基,零代码——这也把
  §7/§9 的学习侧方案(保真终止要配核宽/RSI)与吞吐侧连成了一件事;
- contact 视图合并(audit §9.4):3/3 外部先例(单一全身视图+几何归因),不只 yikang 一家。

**ADJUSTS(排序/口径修正):**
- 每步同步批量化从工单 #2 降级为收尾清理(剩余价值 ≤2 s;两个大头已修);
- racket_target 的 EMA/metrics 散点不逐个向量化,**结构性搬家**:指标半区挪到 logging 节拍,
  policy 节拍只留 reward/obs 真消费的状态;
- 6-8 s 单点目标改为分级门 A/B/stretch(见 10.2),并给 CI 挂每步 `.item()` 计数回归检查,
  防 60-120→7 的战果无声回退。

**NEW(audit 未覆盖):**
- **vendored rsl_rl 的 runner 循环每步 2 次无条件 `.cpu().numpy().tolist()`**(rewbuffer/
  lenbuffer,只要 log_dir 非空就跑)——四家全付这笔,但它意味着"每步同步降到 0"不可能,
  校准所有同步预算时要扣掉这个地板;
- `debug_vis=True` 在 command 与 contact-sensor cfg 里默认开——20-update 对照即可定价,headless 应默认关;
- **36 个死 reward 项别再注册**(构造 65 个跑 29 个;另 IsaacLab `RewardManager.compute` 每项
  每步做一次 `list.index()` 线性扫——65 元素表×29 次/步,量小但白付);
- 保护清单:**两处我们本来就比参考栈精简的地方别在提速时"修坏"**——非对称 actor/critic 保留;
  追吞吐期间不要顺手加 push/interval 事件(§3.1/§9.5 R7 的推撞臂应排在吞吐修复落地后,
  或独立排臂,不与提速改动混线)。

---

## 十一、厂商基准:智元 instinct_mj A3 parkour 对齐(07-31,Franco 提供摘要;**对齐优先级最高**)

**来源与性质**:智元 A3 运动组的 `Instinct-Parkour-Target-Amp-A3-v0`(MuJoCo 栈,AMP 系,
`a3_ultra` 29dof)DR/PD 配置摘要,由 Franco 提供;**无法克隆核验,以摘要为准**,涉及基准值
分歧处标"待厂商定谳"。它与我们是**同底盘**(A3;我们是 31 自由度含头的乒乓变体),因此其
数值的迁移效力高于 BeyondMimic/mjlab/unitree 三家——Franco 裁定:**优先与它对齐**。

### 11.1 Kp/Kd/effort/armature 逐关节 diff(我方 [agibot_a3.py:222-364] vs 厂商表)

**结论:绝大部分已经对齐**(两边同源自 deploy 常数;armature 前 4-5 位有效数字一致,交叉
确认我们的 MJCF 转录基本正确)。真实差异只有三处:

| 关节 | 厂商 parkour | 我们 | 判定 |
|---|---|---|---|
| 髋(pitch/yaw 80/3,roll 120/4)、膝 250/8、踝 50/2、肩 40/3、肩yaw/肘 30/2、腕roll 30/2 | — | 同值 | ✅ 全对齐(effort 220/320/118.2/54.75/60/24 亦同) |
| **waist_yaw** | Kp **80**/Kd 3(与髋同组) | Kp **85**/Kd 3 | **已定谳(07-31):我方正确**。厂商自家 deploy 头文件 `a3_policy_parameters.hpp:98` 明写 `a3_kps[0]=85.0 waist_yaw_joint`;instinct_mj 的 80 应是把 waist_yaw 并进髋组 regex 的简化 |
| **waist_pitch effort** | **115** | **118** | **已定谳:我方正确**。厂商 URDF `effort="118"` + MJCF `actuatorfrcrange="-118 118"` |
| **wrist_pitch/yaw 整组** | Kp **30**/effort **24**/armature **0.004968** | Kp **20**/effort **6**/armature **0.00081** | **已定谳并经厂商口头确认(08-01,Franco 与智元直接对质)**:wrist_pitch/yaw 峰值扭矩就是 **6 N·m**,parkour cfg 是**合组写错**(把 wrist_roll 常数抹到整组)——与本节四原件推断完全一致。我方配置零改动。证据链:乒乓版与标准版两份 URDF `effort="6"`;MJCF `actuatorfrcrange="-6 6"`、armature 0.0008100893338;deploy 头文件 "wrist_roll=30, wrist_pitch/yaw=20"。**连带后果:账本 EXP-ACTION-BALL-PHASED-READINESS §0.2 的"ADOPT 智元新表 30/2/24/0.004968 并重算 action scale"裁定在腕组上作废**(action scale 若按错值重算会差 2.7 倍并进 deploy 合同);另按 Franco 08-01 口径,Kp/Kd 系任务自调无标准,waist_yaw 80↔85 不再是冲突项(deploy parity 建议随 deploy 头文件 85);waist_pitch effort 115↔118 仍待厂商顺口确认(URDF/MJCF 均 118,疑同型表错) |
| head_yaw/pitch | (无头,29dof) | 40/2, effort 6 | 不适用 |

**定谳的证据链(全在本仓)**:`agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf` 与
`agi/URDF/a3_t2d5/urdf/model.urdf`(两版 URDF 同值,支持"电机无变体差异")、
`agi/A3_MuJoCo_Sim/.../a3_pingpong.xml:160-171`、
`agi/a3_deploy_example/.../a3_policy_parameters.hpp:94-100`。
**推论**:"对齐厂商"应指对齐厂商 **deploy 常数**(我们已经对齐),而非盲抄其 parkour 训练 cfg
——那份 cfg 自身在腕/腰上存在 regex 分组简化误差。此发现可回传智元运动组。

### 11.2 DR 全面对照(厂商 vs 我们现役)

| 轴 | 厂商 instinct_mj A3 | 我们(N1) | 判定 |
|---|---|---|---|
| PD 随机化 | **startup**,Kp scale log_u **(0.8,1.2)**,**Kd (0.7,1.3) 不对称更宽**;play 关 | **reset 级**(每集重抽),Kp/Kd 同 (0.85,1.15) | 方向一致;他们认为**阻尼不确定度大于刚度**;我们 reset 级重抽比 startup 更强 |
| **执行器延迟** | **delay [0,2] 控制步,每集抽一次、集内固定** | 死代码,全 0 | **§3 延迟项的最大更新**:厂商同底盘实跑值,直接替代"从真机环路标定"的猜测 |
| 摩擦 | static **(0.2,1.8)** / dynamic **(0.2,1.5)**(MuJoCo 无 per-geom restitution) | (0.3,1.6)/(0.3,1.2)/restitution (0,0.5) | 他们两端各宽 0.1-0.3 |
| 质量 | scale **±20%** 但**只随机 torso/踝/腕** + pseudo_inertia;可开关 | 全身 ±15% + recompute_inertia | 哲学不同:选择性末端质量。**对我们的启示:腕+拍的质量不确定性值得单独 ±20%**(拍子是真实的质量未知源) |
| CoM | **全 body** xyz 各 ±0.02 | 仅 torso ±0.025/0.05/0.05 | 覆盖面互补 |
| 关节零点 | ±0.01 rad 双写 action offset | 同 | ✅ 一致 |
| reset 状态噪声 | 位姿 x/y/yaw ±0.1;速度六维 **±0.2**;关节 **±0.15**、关节速度 0 | N1 全零(§9);legacy ±0.05/±0.2 位姿、±0.5-0.78 速度、±0.1 关节 | 厂商速度噪声比 BeyondMimic 温和 2.5×、关节更宽 1.5×;**§9.5 R4 的分级目标改用这组 A3 值** |
| **推撞** | interval **(1,3) s**,vx/vy **±0.25**、vz ±0.1、r/p ±0.26、yaw ±0.39 | 全关 | **恰好是 BeyondMimic 半幅**——与 §3.1"半幅起步"的建议重合,且现在有同底盘背书;推撞臂直接用这组数(节奏仍建议 5-15 s 或相位门控,理由见 §3.1:他们连续运动、我们有击球窗) |
| obs 噪声/处理 | ang_vel ±0.2(**scale 0.25**)、gravity ±0.05、joint_pos ±0.01、joint_vel ±0.5(**scale 0.05**),**全组 history=8** | 同噪声区间;无 scale;无 history | 噪声区间一致✅;**手工 scale + 8 帧堆叠是厂商同底盘实跑**——§8 可借#3 与 §3 history 项从低优先升格,但仍破坏 110/177/194-D 部署契约,须独立臂 |
| 深度/视觉 DR | 相机外参 6-DoF ±0.01m/±2°/±4°、深度 artifact 三档、帧延迟 0-1 | 无相机 obs | 现阶段不适用;**将来球感知走真实传感器时,这是现成的噪声管线模板**(对应我们 A1 旋钮全零的病,§2 表) |
| 地形/课程 | Perlin heightfield + terrain_levels 课程 | 平地(rough patch 已建未开) | 任务差异,不对齐 |
| Play 口径 | DR 全关,**obs 噪声保留** | eval_deterministic noise_scales=0.0 | 口径差异记录:他们评测带噪声,我们不带 |
| G1 对照 | **A3 有全套 DR,G1 没有** | — | 厂商自己认为 **A3 这台底盘需要 DR 才能 sim2real**——整份报告 DR 方向的最强背书 |

### 11.3 对齐动作清单(按优先级;全部走新臂键/独立臂,默认字节等价)

1. **[已闭环 07-31]** 三处基准值分歧全部定谳为**我方正确**(证据链见 §11.1;A3/A3 Ultra
   电机相同,分歧源于 instinct_mj cfg 的 regex 分组简化)。我方配置**零改动**;
   可选动作:把腕组 4× 力矩上限误差回传智元运动组。
2. **[高]** 执行器延迟臂:DelayedImplicitActuatorCfg 接线,**[0,2] 控制步、每集抽一次**照抄
   厂商;取代 §3 延迟项的标定猜测。仍排在推撞之后单独 A/B(§3 风险分析不变)。
3. **[高]** Kd 随机化拓宽:新臂 `pd_gain_range` 拆成 kp/kd 两键,kd 用 **(0.7,1.3)**;
   kp 保持现值或对齐 (0.8,1.2)。
4. **[高]** 推撞臂的数值直接采用厂商组(±0.25/±0.1/±0.26/±0.39),节奏/相位门控按 §3.1;
   这条与 §9.5 R7 合并执行。
5. **[中]** §9.5 R4(ready 噪声球)的分级目标改用厂商 A3 值:关节 ±0.15 为天花板、速度 ±0.2、
   位姿 ±0.1。
6. **[中]** 腕+拍质量选择性 ±20% 随机化(厂商"末端质量"哲学 × 我们的拍子不确定性)。
7. **[中]** 摩擦区间外扩至 (0.2,1.8)/(0.2,1.5) 做一格消融。
8. **[低]** obs scale+history=8 臂:与延迟臂捆绑(§3 原判:堆叠是策略自估延迟的标准配方),
   部署契约同批改。
9. **[记录]** 评测口径(play 带不带 obs 噪声)在 judge 文档里明示两种口径,防跨栈比较踩坑。

### 11.4 厂商 reward 对照(regularization + safety;07-31 second batch)

**先立一个换算警告**:instinct_mj 是 **AMP 系**——平滑/自然度主要由 discriminator style 收入
承担,显式正则只是补丁,所以其权重(`action_rate_l2` **-1e-3**,比我们/BeyondMimic 的 -0.1
轻 100 倍)在**另一套收入经济**里定价。**可迁移的是项的形状与相对结构,不是绝对权重。**

| 厂商项 | 权重 | 我们的对应物 | 判定 |
|---|---|---|---|
| `action_rate_l2` | -1e-3 | -0.1(v2: clamped -0.2) | **勿抄**——AMP 经济;我们无 discriminator,-0.1 是四家模仿系共识 |
| `dof_pos_limits` | **-2.0** | -40(ActionBall 双 barrier)/-10(DeployParity) | 厂商最温和;但他们无固定目标 qdes 治理问题,**不构成给 -40 降权的证据** |
| `torque_limits`(>90% limit 罚) | -0.01 | `arm_torque_saturation` 已建但被强制清零(ImplicitActuator 无 pre-clip 需求)、`joint_torques_l2` -3e-5 | **形状值得借**:力矩饱和尾巴与"93% 加速度预算"直接相关;落地须先解决 PhysX 隐式 PD 下力矩需求的可观测性(MuJoCo 侧他们可直接读) |
| `angular_momentum`(全身角动量,root_angmom) | -1e-4 | **无对应物**(我们只有 base_ang_vel_xy -0.05、strike_ang_vel -0.5) | **新形状**:全身角动量是挥拍后失衡的更早预警量;PhysX 侧可由 body 质量×速度合成;低价值观测探针先行(§8 探针模式) |
| `self_collision`(**子步整数计数 [0,4]**,force 阈 10) | -0.1 | `undesired_contacts` -0.1(力阈 1.0) | **鲁棒性技巧值得借**:计数代替力幅,防大力自碰爆炸罚;可作为我们 undesired_contacts 的变体键 |
| `joint_deviation_hip`(hip_yaw/roll,平方) | **-0.01** | 无(§3 曾提议 in_hold 门控 -0.1) | 厂商常开但比 unitree(-1.0)轻 100 倍——**支持 §3 的轻量版**;起步值改用 -0.01 |
| `joint_deviation_torso`(waist L1) | -0.05 | `prestrike_waist_twist` -1.0(pre_strike 门控) | 结构不同任务合理(我们要引拍转腰,门控版正确) |
| `freeze_upper_body`(肩/肘/腕 L1) | -0.004 | 无 | **不可迁移**——parkour 要手臂安静,我们手臂就是任务 |
| `flat_orientation_l2` torso -0.6 + `pelvis_orientation_l2` **-3.0** | 双 body 拆分 | `upright` -1.0(单 base) | **拆分思路可借**:骨盆直立权重 5× 于躯干——直立的"根"在骨盆;我们单 base 项可拆 pelvis/torso 两键消融 |
| `feet_air_time` +0.5 / `feet_slide` -0.1 / `feet_flat_ori` -0.4 / `feet_at_plane` -0.1 / `soft_landing` -1e-5 / `feet_close_xy` +0.2 | 步态组 | foot 组七项(§2 表) | 大体已覆盖;**`feet_close_xy` 形状有趣**(exp(−clamp(th−dy)/σ²)−1,过近变负)——横向站距 shaping 对我们 ready 稳定性可作低优先探针 |
| `volume_points_penetration` -8.0 | 穿障 | 无 | parkour 专属,不适用 |

**并入 §11.3 清单的补充动作**:
10. **[中]** `torque_limits` 形状借入:先解决隐式 PD 力矩可观测性(或用 τ=kp·(q_des−q)−kd·q̇
    的解析近似),再以 -0.01/ratio 0.9 起步做单臂 A/B;与加速度包络 3.5% 余量的告警联动。
11. **[低]** `angular_momentum` 观测探针(零权重)先行,一个 wave 后看它对摔倒的预警力再定价。
12. **[低]** `self_collision` 子步计数变体键;`pelvis/torso` 直立拆分消融;`feet_close_xy`
    站距探针;`joint_deviation_hip` 若做则 -0.01 起步(改 §3 原议的 -0.1)。


---

## 十二、外部参考补遗(07-31;为报告中"无先例/NONE"条目补文献与 repo 锚点)

**方法与可信度**:7 个主题检索(web + 浅克隆 repo)→ 逐条对抗核查(每个 locator 实际打开重读,
共 12 处修正、若干整条否决)→ 综合。期间 WebFetch/WebSearch 两度挂死,A-F 由缓存装配、
D 改纯 git 取证、G 全程本地(PDF pdftotext 全文比对 + 本地 review 交叉)。B 段 6 条来自无配套
核查文件的批次,已标〔二次核查未覆盖〕——引作承重论据前须补一手核对。诚实空档集中列于文末。

**对 §7.4 的三处修正(先例状态更新)**:
1. **P0(σ 课程)"无一家做训练时 σ 调度"不再成立**:KungfuBot/PBHC(NeurIPS 2025)把
   σ←min(σ, EMA(误差)) 的单调收紧做成了**论文主贡献且出厂默认开启**(main.yaml L60-64)。
   注意:这恰恰是我们 07-26 退役的 adaptive_sigma 同族机制(误差自适应,非固定迭代表),
   而"纯迭代计划表式 σ 课程"至今仍无先例——**外部最强先例支持的是我们退役的那个方向**,
   钦定的静态阶梯反而没有先例;这个张力值得 Franco 知情再裁。
2. **P1(粗细双核)最强先例升级**:不再是 mjlab lift_cube——**IsaacLab 官方 Reach 任务出厂默认**
   就在同一末端误差上叠"裸 L2 线性粗项(-0.2)+ 1−tanh(d/0.1) 细项(+0.1)",代码字面叫
   `fine_grained`,Franka/UR10 全系继承,就在我们用的框架里。
3. **P2(同误差 L1+exp 并联)"无先例"不再成立**:上述 Reach 线性项即是;另有 TCN(ICRA 2018,
   真机)明写"平方项给远处强梯度、Huber 项管近处精度"的同构论证〔二次核查未覆盖〕。

### A. 碰撞/接触解算加速(§10 提速工单的文献面)

- **Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning**(paper, arXiv:2108.10470 / ar5iv 全文)— 选 TGS(Temporal Gauss-Seidel)求解器就是为了 GPU 吞吐:每个 substep 只做一次 Gauss-Seidel 迭代,而不是每步多次迭代;4096 环境下 Humanoid(21 DOF)不到 4 分钟到 reward 5000、约 200K env-steps/s,ANYmal 平地 <2 分钟、崎岖地形 sim2real <20 分钟(RTX A6000)。— 直接先例;§10 工单里"求解器类型/迭代次数"这一格的原始出处。
- **MuJoCo XLA(MJX)官方文档**(official-doc, https://mujoco.readthedocs.io/en/stable/mjx.html)— GPU 上推荐 NEWTON,"常常一次迭代就收敛";几何有硬上限(凸-凸约 32 顶点、mesh-primitive 约 200 顶点、不支持 SDF);官方明说只标必要的碰撞对"对性能有戏剧性影响";人形批量吞吐 650K(M3 Max)/1.8M(3995WX)/950K(A100)/2.7M(TPU v5)steps/s。— 直接先例;§10 工单"碰撞对裁剪 + 低迭代求解器"两格。
- **Genesis-Embodied-AI/genesis-world**(repo, https://github.com/Genesis-Embodied-AI/genesis-world)— 宣传口径:比 Isaac Gym/Sim/Lab 与 MJX 快 10–80 倍,头条数字是单张 4090 上 Franka 场景 43M FPS。核查修正:这两个数字**已不在当前 README/最新文档里**,只在 2024-12 首发 README(commit 771ee4f)和 readthedocs v0.3.3 里能查到,属历史宣传口径。— 仅理论(营销基线);§10 工单里"换引擎能省多少"的对照锚。
- **How fast is the new hyped Genesis simulator?**(blog-by-authors, https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is,作者是 ManiSkill 维护者 Stone Tao)— 复现那条 43M FPS,发现它用了 1 个 substep、90% 时间动作为零、关掉自碰撞、静止物体休眠;把设置调回真实(开自碰撞、2–4 substep、连续动作),同一张 4090 上掉到 0.29M FPS(约 150 倍),此时比 ManiSkill 慢 3–10 倍,还记录了抓着的方块会掉出夹爪的精度 bug。— 直接先例;§10 工单最该先贴的一条:**自碰撞开关和 substep 数,而不是求解器内核,才是速度数字的主要杠杆**,评估任何提速承诺(包括 PhysX/Newton 自家的)都要按这个标准复核。
- **zhouxian/genesis-speed-benchmark(ANYmal-C 脚本)**(repo, https://github.com/zhouxian/genesis-speed-benchmark/blob/aa79858a/anymal_c/test_genesis.py)— Genesis 自家腿足 benchmark 默认就用 `constraint_solver = Newton`,并有 `--mjx-solver-setting` 一键切到 MuJoCo Menagerie 的 anymal_c_mjx.xml 口径:tolerance=1e-8、iterations=1、ls_iterations=4。— 直接先例;§10 工单"低迭代 Newton 是跨引擎的腿足默认调法",这是落在代码里的证据,不是说法。
- **leggedrobotics/legged_gym(ANYmal-C 资产 + 配置)**(repo, commit 8fa29acc,legged_robot_config.py L106-195 + resources/robots/anymal_c/urdf/anymal_c.urdf)— 配置默认 collapse_fixed_joints=True、self_collisions=0、thickness=0.01、contact_offset=0.01、rest_offset=0.0;直接解析 URDF:45 个 mesh **全是视觉用**,13 个 collision 块只有 9 个 cylinder + 4 个 sphere,**零 mesh 碰撞体**。— 直接先例;§10 工单"碰撞几何一律降成基本体"。
- **isaac-sim/IsaacGymEnvs(AnymalTerrain)**(repo, commit aeed2986,anymal_terrain.py L220/228/282 + AnymalTerrain.yaml L141)— collapse_fixed_joints=True、thickness=0.01、contact_offset=0.02;`create_actor(..., i, 0, 0)` 用环境序号 `i` 当 PhysX 碰撞组,跨环境的形状对在 broadphase 阶段就被整体跳过。— 直接先例;§10 工单"跨环境 broadphase 不该花钱"。
- **Isaac Lab 文档:PhysX Simulation Performance and Tuning**(official-doc, IsaacLab@99e7bc1d docs/source/how-to/simulation_performance.rst)— 原话:ANYmal-C 上"保留膝和脚的碰撞几何,去掉腿上其他部位的碰撞几何以优化性能";基本体比 mesh 便宜、SDF 最贵;高长宽比的凸包 mesh 会**静默回落到 CPU**,只打一条 warning(`ConvexMeshCookingTask: failed to cook GPU-compatible mesh, collision detection will fall back to CPU`)而不让任务失败。— 直接先例;§10 工单"保膝脚、剥腿"那格的权威出处,外加一个值得单列的正确性陷阱。
- **Isaac Lab:PhysxCfg 求解器与 GPU 缓冲区旋钮**(official-doc, IsaacLab@99e7bc1d source/isaaclab_physx/.../physx_manager_cfg.py L19-227)— solver_type 默认 1=TGS;position 迭代范围 [1,255]、velocity [0,255];bounce_threshold_velocity=0.5 m/s;friction_offset_threshold=0.04;friction_correlation_distance=0.025;GPU 缓冲区默认 gpu_max_rigid_contact_count=2^23、gpu_max_rigid_patch_count=5·2^15、gpu_found_lost_pairs_capacity=2^21、gpu_found_lost_aggregate_pairs_capacity=2^25、gpu_collision_stack_size=2^26、gpu_max_num_partitions=8(2 的幂,最大 32);文档明确警告 GPU 缓冲区**不能动态扩容**,开小了直接硬失败。— 直接先例;§10 工单每个旋钮的默认值来源。
- **Isaac Lab 文档:Performance Benchmarks**(official-doc, IsaacLab@99e7bc1d docs/source/overview/reinforcement-learning/performance_benchmarks.rst)— 同一张 RTX 4090、同一套测法:Cartpole-Direct(4096 envs,几乎无接触)= 1,100,000 env-step FPS;Velocity-Rough-G1(人形 + 崎岖地形,接触密集)= 94,000 FPS,差 11.7 倍。— 直接先例;§10 工单的"接触负担值多少钱"官方基准锚。
- **isaaclab#1995(地面静态碰撞体的 GPU contact filter 不支持)**(repo, https://github.com/isaac-sim/IsaacLab/issues/1995)— NVIDIA 维护者 kellyguo11 确认根因是**取接触的 API 缺口**,不是物理解不出来:"contacts on static colliders that are not rigid bodies are not yet supported…However, the physics simulation should have no issues in solving for contacts on static colliders";绕法是把静态几何做成 kinematic RigidBody。— 直接先例;§10 工单的正确性条目(接触被正确求解但被静默漏报)。
- **isaaclab#4108(mesh 资产触发同一条警告)**(repo, https://github.com/isaac-sim/IsaacLab/issues/4108)— 同一族问题的另一个面:GPU 接触过滤要求参与体是 box/sphere/capsule/凸包,非凸三角网格会把 GPU 过滤路径关掉,退回 CPU("significantly slower and not always recommended")。— 直接先例;§10 工单同上,补第二种触发条件。
- **isaaclab#5018(Newton SensorContact 规模化)**(repo, https://github.com/isaac-sim/IsaacLab/issues/5018)— NVIDIA/Newton 的 eric-heiden 亲自 profile:Kuka-Allegro、4 个接触传感器、512 环境、L40 上,完整 env.step=16.70ms,去掉接触消费者=10.09ms(接触相关 6.61ms),而 `solver.update_contacts` 本身只要 0.053ms、4 次 `SensorContact.update` 合计 0.181ms;根因是 `ContactSensor.data` 每次访问都触发 `_update_outdated_buffers()`,reward/observation 各自独立取一次(实测每步 28 次 `.data`,每传感器 7 次),每步只缓存一次读取就省回 3.37ms;8192 环境的合成任务上重复读 2.09ms vs 缓存 0.43ms(4.9 倍),而孤立求解调用只要 0.052ms。核查提醒:原文"接触求解 <1.5%"这个比例是拿 16.70ms 全步长做分母算的(0.234/16.70=1.4%),若按 6.61ms 的接触增量做分母则约 3.5%,写进报告时要把分母说清楚。— 直接先例,也是 §10 最该先做的一格:**贵的是消费端重复读过滤矩阵,不是 PhysX/Newton 的接触求解;每步缓存一次即可**。
- **Isaac Lab 文档:Contact Sensor 核心概念**(official-doc, https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html)— `net_forces_w`(每个 body 的净接触力,不过滤)和 `force_matrix_w`(过滤后的归因矩阵)报的是同一份力;过滤只支持"多对一",多传感器体对多目标时 `force_matrix_w` **静默返回 None**。— 直接先例;§10 工单"用净力 + 几何归因替代加宽过滤矩阵"这条替代方案的 API 依据。
- **Newton Physics Integration(experimental)— Isaac Lab 文档**(official-doc, https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)— Isaac Lab 3.0 的 Newton 后端(Warp 上的 MuJoCo-Warp 求解器)已用双向策略迁移做过验证,原句是"we have also successfully deployed a Newton-trained locomotion policy to a G1 robot";仓库树里 `isaaclab_physx` / `isaaclab_newton` / `isaaclab_ovphysx` 并存,共用 `PhysicsCfg` 基类。— 类比/前瞻;§10 工单"要不要换接触后端"的背景。
- **PACE: Physics Augmentation for Coordinated End-to-end RL toward Versatile Humanoid Table Tennis**(paper, arXiv:2509.21690 / html v3)— 训练栈是"built on LeggedLab, an RL benchmark for humanoid locomotion developed on the IsaacLab platform",PPO + 4096 并行环境,"workstations equipped with NVIDIA RTX 4090 GPUs";Booster T1(23 DOF)上 sim2real,仿真里 hit rate ≥96%、success ≥92%。— 直接先例;§10 的现实标尺:和我们几乎同构的栈(manager-based、4096 envs、单张消费级卡)能跑成什么样。
- **SMASH: Mastering Scalable Whole-Body Skills for Humanoid Ping-Pong with Egocentric Vision**(paper, arXiv:2604.01158)— Unitree G1 上的整身乒乓系统(Motion-VAE 击球库 + 动作跟踪 RL),首次户外人形乒乓、首次人形整身扣杀;**仿真器/并行环境数/GPU 一概没披露**。— 类比;§10 只能当话题背景引,不能当吞吐数字来源。
- **IsaacIPC: Coupling High-Fidelity Simulation and Realistic Rendering for Contact-Rich Robotic Systems**(paper, arXiv:2605.24339)— 把 GPU 版 IPC 接到 IsaacSim/Lab 当替代接触求解器,加了 mortar 接触势做触觉;演示对象是四足、灵巧手、UMI 夹爪,**没有人形,也没有任何吞吐/加速数字**。— 仅理论(弱先例);§10 "替代求解器方向"备注一句即可。

### B. 误差核形状(§7 P0-P2 的外部先例补遗)

- **KungfuBot / PBHC(NeurIPS 2025)**(paper, arXiv:2506.12851)— 跟踪奖励还是 DeepMimic 那套 r(x)=exp(-x/σ),但把 σ 做成**不固定**:σ ← min(σ, x̂),x̂ 是瞬时跟踪误差的 EMA,σ 单调不增、从一个较大的 σ_init 开始收紧;作者自己给的理由是"selecting a single, fixed value for the tracking factor that works for all motion scenarios is impractical"。— 直接先例;P0/P1(σ 退火/宽核起步)最强的一条,而且是论文主贡献不是边角消融。
- **PBHC 官方实现 general_tracking.py**(repo, https://github.com/TeleHuman/PBHC — humanoidverse/envs/motion_tracking/general_tracking.py L958-993,`_init_adaptive_sigma`/`_update_adaptive_sigma`)— 每步按 `error_ema = ema*(1-alpha) + error.mean()*alpha` 更新,再单调收紧 σ。**核查修正**:默认分支 `type: origin` 走的是**朴素 `min(ema, 当前σ)`,没有 scale 乘子**;`ema*scale` 只属于 `adptype == "scale"` 分支。因为出厂 scale=1.0,数值效果相同,但写报告时别把公式挂错分支。— 直接先例;P0/P1 的"这是真跑着的代码,不是设想"。
- **PBHC 默认 reward 配置 main.yaml**(repo, humanoidverse/config/rewards/motion_tracking/main.yaml L60-64)— `adaptive_tracking_sigma: enable: True / alpha: 1e-3 / type: origin / scale: 1.0` 是**出厂默认**,不是留着没开的开关。— 直接先例;P0/P1 的"默认即开启"证据。
- **IsaacLab Reach 任务 RewardsCfg**(repo, IsaacLab@99e7bc1d source/isaaclab_tasks/.../manipulation/reach/reach_env_cfg.py L180-188)— 官方默认在**同一个末端位置误差**上叠两项:`end_effector_position_tracking`(`position_command_error`,裸 L2 距离,线性罚,weight −0.2)+ `end_effector_position_tracking_fine_grained`(`position_command_error_tanh`,1−tanh(d/std),weight +0.1,std=0.1);Franka、UR10 等所有继承该任务的机器人配置都保留这两项。— 直接先例;P1(粗核+细核并存)与 P2(线性项与饱和核相加)最强的一条,而且就在我们自己用的框架里,代码里字面写着 `fine_grained`。
- **IsaacGymEnvs FrankaCabinet 的 compute_franka_reward**(repo, isaacgymenvs/tasks/franka_cabinet.py L489-546)— 默认奖励 `dist_reward = 1/(1+d²)`(大误差处梯度不消失的粗核),再平方,再 `where(d<=0.02, dist*2, dist)`(2cm 内精度加倍),外加开门角 0.01/0.2/0.39 rad 三级递增奖金。— 直接先例;P1 的第二个"粗核 + 精度奖金 + 分级奖金"出厂配方。
- **IsaacGymEnvs FrankaCubeStack 的 compute_franka_reward**(repo, isaacgymenvs/tasks/franka_cube_stack.py L697-747)— `1−tanh(10·(d+d_lf+d_rf)/3)` 的粗饱和项,和第二个 tanh 核 `align_reward=(1−tanh(10·d_ab))·cubeA_lifted` 取 max,再加二值 lift 与成功时覆盖一切的稀疏 stack 项。— 直接先例;P1 在灵巧操作场景的第三个出厂多级核栈。
- **Time-Contrastive Networks(ICRA 2018)**(paper, arXiv:1704.06888)— 真机 Sawyer 倒水任务的模仿奖励写成 `R = -α‖w−v‖² − β√(γ+‖w−v‖²)`,即**同一个误差上,平方项与 pseudo-Huber 项直接相加**;作者原话:平方项"gives us stronger gradients when the embeddings are further apart, which leads to larger policy updates at the beginning of learning",Huber 项"starts prevailing when the embedding vectors are getting very close ensuring high precision"。— 直接先例;P2 最贴字面的一条,连"大误差要梯度、小误差要精度"的论证结构都和我们一致。〔二次核查未覆盖〕
- **Policy Invariance Under Reward Transformations(Ng, Harada & Russell, ICML 1999, pp.278-287)**(paper, https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)— Theorem 1:塑形项保最优策略的**充要**形式是 F(s,a,s′)=γΦ(s′)−Φ(s);论文自己的 5×5 网格例子取 Φ(s) = −曼哈顿距离/每步平均进展,就是"距离势能"的模板。— 仅理论(但是要引的那条理论);P2 里把 racket_progress 这类 Δ距离项论证成势能塑形。
- **Benchmarking Potential Based Rewards for Learning Humanoid Locomotion**(paper, arXiv:2307.10142)— 明确按 Ng 的式子 P(s_k,s_{k+1})=γΦ(s_{k+1})−Φ(s_k) 加在基础奖励上,在 IsaacGym 的 MIT Humanoid(18 DOF)跑步任务上试了三个势能(姿态、高度、关节正则);结论是收敛速度只小赚,但**训练方差明显更低(0.946 vs 1.905)**,且在 0.1×–10× 权重扫描里都有效,而未塑形/直接塑形的可调窗口很窄。— 直接先例(同栈同形态);P2 里"势能项的收益主要是稳,不是快"这一判断的数字来源。
- **Boosting RL in Continuous Robotic Reaching Tasks using Adaptive Potential Functions**(paper, arXiv:2402.04581)— 同样显式引 Ng 的 F(s,s′),在 Baxter 手臂到达任务上用**学出来的**势能;APF-DDPG 显著优于 DDPG(t=−19.1,p=5.8e-78),真机 Baxter 上 5 步到位。— 类比;P2 的第二个真机应用,但要注意它的势能是学出来的、不是裸距离。
- **Reward Engineering for Object Pick and Place Training**(paper, https://arxiv.org/pdf/2001.03792)— 明确引 Ng 的保最优性结论,并把距离缩减塑形用在 FetchPickAndPlace-v1(7-DOF Fetch,MuJoCo)+ HER 上。— 类比(比较非正式);P2 的"这套做法在机械臂上确实有人用"。〔二次核查未覆盖〕
- **HPRS: Hierarchical potential-based reward shaping from task specifications**(paper, Frontiers in Robotics and AI 2024)— 多处引 Ng 1999 作为形式基础,把分层势能用在含人形移动与自动驾驶的任务上,真机跑在 F1TENTH 上;但势能是分层加权构造,不是欧氏距离。— 类比;P2 的辅证,离"距离势能"这一具体形式较远。〔二次核查未覆盖〕
- **Learning Dexterous In-Hand Manipulation(OpenAI)**(paper, arXiv:1808.00177)— 稠密奖励就是 r_t = d_t − d_{t+1}(到目标朝向的角距离每步减少量),外加 +5 成功、−20 掉落;论文自己**没有**引 Ng、也没用"potential-based shaping"这个词。— 直接先例(形式上);P2 里 racket_progress 的高引用真机背书,但要老实说明作者没有做这个理论连接。
- **PHC 的 compute_point_goal_reward**(repo, https://github.com/ZhengyiLuo/PHC — phc/env/tasks/humanoid_im.py L1557-1562;论文 arXiv:2305.06456)— `clamp(prev_dist − curr_dist, max=1/3) * 9`,一个**带上限的 Δ距离进展奖励**,与同一任务里的乘性 exp 模仿核(`exp(-k_pos·dist)` 等,L1523-1554)一起用;两者在同一个 `_compute_reward` 里合并(进展项全环境计,模仿项在 0.25 距离内以 0.5 权重加入)。— 直接先例;和我们 racket_progress + 跟踪核的结构最像的一条,也给出了"进展项要 clamp"的现成做法。
- **DeepMimic(SIGGRAPH 2018)**(paper, arXiv:1804.02717)— 核宽是**写死的**:r^p=exp[−2·Σ‖姿态误差‖²]、r^v=exp[−0.1·…]、r^e=exp[−40·…],权重 w_p=0.65/w_v=0.1/w_e=0.15/w_c=0.1,全文没退火过;它处理"离参考很远"的手段是 RSI + 早停(第 6 节把初始状态分布与终止条件明确当成设计变量,原话说 RSI 是"an additional channel through which the agent can access information from the reference")。— 直接先例(反向);P0-P2 的定位依据:**DeepMimic 一脉的传统答案是改初始化/终止,不是改核形状**,所以"这一脉没有核形状先例"的说法成立,而 PBHC 是 2025 年才分出去的另一支。
- **AMP: Adversarial Motion Priors(SIGGRAPH 2021)**(paper, arXiv:2104.02180)— 干脆把手调的姿态跟踪核整个换掉,用判别器从无结构动作集里学 style reward,不再逐 clip 调核。— 类比;P0-P2 的第三条路(不调核、换机制),供报告写"我们没走这条"时引。〔二次核查未覆盖〕
- **Perpetual Humanoid Control(ICCV 2023)**(paper, arXiv:2305.06456)— 同样用固定系数的 DeepMimic 式指数核(100/10/0.1/0.1,不退火);它对付大跟踪误差的机制是 Hard Negative Mining(重采当前策略跟不上的 clip)+ 专门的 Fail-state Recovery 原语,用 Progressive Multiplicative Control Policy 拼进来。— 直接先例(反向);和上面 DeepMimic 一起支撑 P0-P2 的定位。〔二次核查未覆盖〕
- **NEAR: Noise-conditioned Energy-based Annealed Rewards**(paper, arXiv:2501.14856;代码 https://github.com/anishhdiwan/near)— 用能量模型在逐级加噪的专家分布上学模仿奖励,训练中按噪声级别退火(受 Annealed Langevin Dynamics 启发),在 IsaacGym 人形移动/武术任务上与 AMP 可比;**具体噪声表、退火级数、超参数值没能确认**。— 类比;P0 的"先容忍大偏差、再收紧"同思路但换了实现,比 KungfuBot 弱。〔二次核查未覆盖〕

### C. 小初始噪声下的探索(§8 雷 2/3 的文献面)

- **Residual Policy Learning**(paper, arXiv:1812.06298,Silver et al. 2018)— π_θ(s)=π(s)+f_θ(s),**把残差网络最后一层初始化为零**,开局残差策略就等于基策略;探索靠固定的加性高斯噪声 scale 0.2(hook 任务 0.1),噪声是加在(初始为零的)残差输出之上的。— 直接先例(技术种类,不是同算法:它是 DDPG 系离线策略);§8 雷 2 的"零初始化最后一层 + 顶上固定小噪声"源头,后来几乎所有残差 RL 都引它。
- **Residual Reinforcement Learning for Robot Control**(paper, arXiv:1812.03201,ICRA 2019)— Algorithm 1 是 u_t=π_θ(s_t)+N_t、u′_t=u_t+π_H(s_t),骨干是 TD3(rlkit),就是标准离线策略探索噪声;**全文确认没有**零初始化最后一层的做法(是核实过的缺失,不是没搜到)。— 类比(反向对照);§8 雷 2 里说明"残差 RL ≠ 都做了近零初始化"。
- **What Makes Value Learning Efficient in Residual RL?(DAWN)**(paper, arXiv:2602.10539)— 原话:"the residual policy's final layer is initialized with near-zero weights, producing near-zero actions with minimal variance at the start of training";由此 |log π| 极大,SAC 自动熵调节下即便 α 很小,熵项也会"dwarfs the task reward by an order of magnitude",在任何显式 warmup 阶段引发 Q 值发散/崩塌,作者把它当成**近确定性残差初始化的一般病理**;探索策略消融里 Base Policy Only(零探索噪声)不劣于甚至好过其他方案,Gaussian Noise(σ=0.1)让 PegInsertionSide"collapse entirely"。**核查修正**:Fig.15 比的是 Base Policy Only / Full Action / Gaussian Noise / Epsilon-Greedy 四项,Trajectory-Consistent Noise 是另一张图(Fig.16)的独立负结果,别并成一句。— 类比(必须标注算法不同:SAC 的 α·log π,不是 PPO 的 adaptive-KL);§8 雷 3 最接近的一条,同为 1/σ 型放大;同时也是雷 2 的反证:在近零初始化的头上加探索噪声未必有好处,甚至会崩。
- **Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model**(paper, arXiv:2412.13630)— 两个真跑的抑制手段:(1)bounded residual action,把残差乘一个任务相关的 α,使其只能小幅扰动基动作;**核查修正**:Table 8 的实际范围是 **0.03–0.8**(Diffusion Policy 那几行是 0.05–0.8),原表述的"0.03–0.3"漏掉了 DP 各行;(2)progressive exploration schedule,ε=min(t/H,1) 线性把"用残差动作"的概率从 0 拉到 1,出厂 H 在 100K–8M 步之间。— 类比(SAC/操作任务,不是 PPO/移动);§8 雷 2 的数值锚:"早期把残差影响压多小、压多久"。
- **ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning**(paper, arXiv:2510.05070,Sec. III-C2)— 原话:"we initialize the final layer of the PPO actor using Xavier uniform initialization with a small gain factor … so that the initial outputs are close to zero";这是 PPO、人形、整身、挂在冻结 GMT 基策略上的残差头,架构上最像我们 BeyondMimic-fork + 任务头;但**没给探索噪声 std 的数值,也没讨论 std 塌缩或钳位**(均已确认缺失)。— 直接先例(近零初始化那一半);§8 雷 2;另一半(噪声 std)要老实说它没答。
- **RL-augmented MPC Framework for Agile and Robust Bipedal Footstep Locomotion**(paper, arXiv:2407.17683)— 双足、PPO、挂在 MPC 步态规划器上的残差策略,原话"we initialize the last layer of the neural network to be zero, similar to [29]",[29] 经查确为 Silver 等的 Residual Policy Learning;探索是 a_k ~ N(μ_θ, σ(r)),σ(r) 被描述成"a scheduled parameter",**但全文找不到这个表的具体数字**。— 直接先例(PPO + 腿足 + 零初始化);§8 雷 2;数值那半是空档。
- **rsl_rl v5.4.2(GaussianDistribution + adaptive-KL 学习率)**(repo, https://github.com/leggedrobotics/rsl_rl @ c281c32e,pyproject 5.4.2;**核查修正的行号**:GaussianDistribution 构造与 std_range 逻辑在 distribution.py **L144-175**,不是 L30-60;ppo.py L50-51 与 L234-259)— 出厂默认 `std_range=(1e-6, 1e6)`,下界再取 `max(std_range[0], 1e-6)`,注释写明"Avoid zero std for numerical stability";`std_type ∈ {scalar, log}`,log 模式先在对数空间钳位再取指数,构造上不可能为零或负。PPO 构造函数默认就是 `schedule='adaptive'`、`desired_kl=0.01`;自适应学习率算 `torch.distributions.kl_divergence(Normal, Normal)` 后:kl > 2·desired 则 lr=max(1e-5, lr/1.5),kl < desired/2 则 lr=min(1e-2, lr·1.5)。— 直接先例(就是我们自己在用的库);§8 雷 2(std 下限机制)与雷 3(自适应 KL 学习率公式)的第一手依据。
- **rsl_rl issue #33 / IsaacLab issue #673(actor std 变 NaN)**(repo, 两个 issue 全线程)— 真实生产事故,2024–2026 间几十位独立报告者(有人在 3600 并行环境上),报错 `RuntimeError: normal expects all elements of std >= 0.0`,多人追到值函数 loss 发散成 inf/NaN 后传染到 std 参数(scalar 模式下 std_param 被优化器推成负);维护者 ClemensSchwarke 原话:"We now added logarithmic noise standard deviation, which prevents negative values."— 直接先例;§8 雷 2 要引的"这不是理论风险,是同库里反复发生过的事故"。
- **rsl_rl PR #67 / #190 / #201**(repo, 三个 PR 的完整评审线程)— #67 里的设计论证:先有"using exp is not recommended as it may cause the training to crash"的顾虑,被 vogeldylan 纠正为"Using exp with state-independent noise (e.g., as a parameter and not as an output of the network) is actually fine, and can be set as the default… If we ever add state-dependent noise, outputted by the network itself, then appropriate clipping with softplus is recommended";#190 标题直译就是"加 log_std 边界以防 std 下溢崩溃",被 #201 取代,后者才是 v5.4.2 里那个 std_range 钳位(首发于 v5.2.0)。— 直接先例;§8 雷 2 的"状态无关 std 用 exp 安全、状态相关 std 要额外钳"的维护者级论断。
- **PyTorch torch.distributions.kl 的 register_kl(Normal, Normal)**(repo, torch/distributions/kl.py)— 源码就是 `var_ratio=(p.scale/q.scale)²; t1=((p.loc−q.loc)/q.scale)²; return 0.5*(var_ratio + t1 − 1 − var_ratio.log())`,其中 t1 正是 (Δμ/σ_q)²,σ_q→0 时对任何非零均值差都无界发散;rsl_rl 的自适应 KL 调 lr 走的就是这个函数。— 直接先例(字面数学);§8 雷 3 的机理闭环。
- **Learning to Walk in Minutes Using Massively Parallel Deep RL**(paper, arXiv:2109.11978,Appendix A.4/Algorithm 1/Table 3)— desired KL = 0.01,更新规则 kl>2kl\* 时 α←max(1e-5, α/1.5)、kl<0.5kl\* 时 α←min(1e-2, 1.5α),和今天 rsl_rl 出厂完全一致;论文自己也说是沿用更早的做法(Heess et al.)。— 直接先例;§8 雷 3 里说明这是全领域通行默认、不是某个实现的怪癖。
- **CleanRL ppo_continuous_action.py**(repo)— `actor_logstd = nn.Parameter(torch.zeros(...))`、`action_std = exp(action_logstd)`,默认 log_std=0 即 std=1.0,**全文件没有任何钳位**,正性完全由 exp 保证。— 类比(基线参照);§8 雷 2 的对照:状态无关 std 的最简实现根本不需要下限。
- **Stable-Baselines3 common/distributions.py**(repo)— DiagGaussianDistribution 的 docstring 自己写着"standard deviation (log std in fact to allow negative values)",默认 log_std_init=0.0;而 gSDE 的 StateDependentNoiseDistribution 默认 log_std_init=−2.0(std≈0.135),并用 expln() 取代 exp,文档写明是"to ensure std is positive and prevent it from growing too fast"。— 类比;§8 雷 2 的第二个大库佐证,而且和 rsl_rl 维护者独立收敛到同一个"状态无关 vs 状态相关"的分界。

### D. DR 轴(§3/§11 延迟·力矩上限·armature 的先例;纯 git 取证,零 web)

**三条"无先例"轴的共同正主找到了:IsaacGymEnvs 的 Dextreme/ADR 谱系**(OpenAI Dactyl 后继;
配置逐行核实于本地克隆):

- **力矩上限随机化**:`AllegroHandDextremeADR.yaml:134-137` `dof_properties.effort` scale
  uniform,ADR 初始 [0.9,1.1]、上限可自动放宽到 [0.4,10.0](:278-281)——直接先例;
  另证 IsaacLab 框架**没有**内置力矩上限事件(全仓无 `effort_limit_distribution_params`),
  我们若做须自写薄封装(§3 原判成立)。
- **armature 随机化**:同文件 142-145,scale,ADR 初始 [0.8,1.2]——直接先例;IsaacLab 框架
  自带 `randomize_joint_parameters`(armature/friction,events.py:1303)但 unitree 官方配置不用。
- **执行器延迟**:`allegro_hand_dextreme.py` 完整动作队列机制(`action_latency` randint 进滚动
  buffer + 概率性额外一步 `action_delay_prob` + 观测侧独立延迟/跳帧),ADR 可从 0 放宽到 60 步;
  非 ADR 变体按固定 schedule 退火(`actionLatencyScheduledSteps=2e6`)。**vanilla ShadowHand
  (Dactyl 复现)没有任何延迟建模**——这是 Dextreme 世代的新增,不是 Dactyl 原版。
- **反向确认**:legged_gym / unitree_rl_gym / humanoid-gym 三家 legged 系**全部没有**这三条轴
  (armature 在 legged_gym 只是静态常数 config.py:118)。
- **定位结论**:这三条轴不是"没人做过",是"**legged 社区没做、灵巧手/ADR 社区标配**"。
  我们的手臂击球任务对动力学误差的敏感度更像灵巧手;加上厂商 instinct_mj 的同底盘延迟
  [0,2] 步(§11.2),延迟臂的先例链已完整。
- 论文锚(**本次未能核对原文——网络挂死,出自训练知识,待补spot-check**):Peng et al.
  1710.06537(动力学随机化奠基)、Dactyl 1808.00177(噪声轴)、Tan et al. 1804.10332
  (显式延迟建模,最贴)、Hwangbo 1901.08652(执行器网络=随机化的替代路线)、
  ADR 1910.07113(自适应放宽机制本尊)。

### E. clip 课程与恢复训练(§9.5 R3/R6 的先例)

- **PHC 的 Auto-PMCP(困难样本挖掘)**(paper+repo, arXiv:2305.06456;phc/utils/motion_lib_base.py L351-387、phc/learning/im_amp.py L104-132/L240-345)— 全数据集评测跑完后收集 failed_keys:hard 模式把采样概率全部清零、只在失败 clip 上均匀采;soft 模式累计每个 clip 的 `_termination_history`,采样概率正比于观测到的失败率。每个评测周期都真的重排训练批次构成。— 直接先例;R3"按失败率加权选 clip"最接近的参考实现,代码级别的。
- **PHC 的 fall-init / recovery-episode 机制**(repo, phc/env/tasks/humanoid_amp_getup.py L41-160;phc/data/cfg/env/env_im_getup_mcp.yaml,以及 env_im_g1_phc.yaml / env_im_h1_phc.yaml L33-35)— `_generate_fall_states()` 用随机朝向把人摔下去、静置 150 步攒一批"塌掉的姿态";每次 reset 对已终止环境掷 Bernoulli(recoveryEpisodeProb=0.5),中了就**不硬 reset**、只给 recoverySteps=90 步自己爬起来,否则再掷 Bernoulli(fallInitProb=0.3) 直接从姿态库里挑一个摔倒姿态起始。**核查修正**:这三个参数虽然确实写在 G1/H1 的 yaml 里,但那两个配置的 `task: HumanoidIm` 继承自 HumanoidAMPTask、**根本不读这三个键**(grep 确认 humanoid_im.py 零引用),README 的 G1/H1 训练命令也没换 task;真正接线的是 SMPL 角色的 env_im_getup_mcp.yaml(task: HumanoidImMCPGetup)。所以"真机人形配置出厂启用"的说法不成立,G1/H1 里那几行是复制过去的死键。— 直接先例(机制本身),但先例强度要降级:**只在角色任务里真跑,不是人形机器人配置的默认**;对应 R6 的"恢复 RSI 混进正常训练"。
- **MoCapAct(NeurIPS 2022 D&B)**(paper+repo, arXiv:2208.07363 Table 2;mocapact/distillation/dataset.py L18-22/L265-387;cfg/multi_clip/{bc,rwr,awr,cwr}.txt)— 明确定义并基准了 4 种 clip/数据加权:BC(w=1,均匀)、CWR(w=exp(clip 回报/λ),λ=0.2)、AWR(w=exp(优势/λ),λ=8)、RWR(w=exp(Q/λ),λ=4);归一化回合奖励 BC=0.654±0.005、CWR=0.671±0.003、AWR=0.661±0.003、**RWR=0.688±0.002(最好,比 BC 高约 5%)**,回合长度 BC=0.855 / RWR=0.868;代码另有按长度上采样(snippet_weight = max_clip_len/clip_len)且默认 train.txt 就开着。— 直接先例;R3"均匀 vs 按长度 vs 按表现加权"唯一带干净消融表的一条。
- **ProtoMotions(NVlabs)**(repo, protomotions/agents/evaluators/mimic_evaluator.py L98-122;motion_manager.py L321-450;g1-bones-deploy/experiment_config.py L415-416)— `_update_motion_sampling_weights()` 按指数衰减更新:成功 clip 的权重按 success_discount^eval_interval 衰减,失败 clip 除以 failure_discount^eval_interval(failure_discount=0 时直接重置回 1.0),把 PHC 的二值 PMCP 推广成连续 EMA 难度跟踪;出厂的 Unitree G1 部署配置就是 success_discount=0.999、failure_discount=0。— 直接先例;R3 的"这套思路到 2025-2026 仍是 NVIDIA 现役人形跟踪栈的默认",不是 2023 年的孤例。
- **BeyondMimic / whole_body_tracking(我们 fork 的上游)**(paper+repo, arXiv:2508.08241;source/whole_body_tracking/.../tracking/mdp/commands.py L80-90/L207-241/L296-299/L355-371)— `_adaptive_sampling()` 是**clip 内的时间段课程**:按 clip 长度/dt 切 bin,失败时给当前 bin 记一笔,bin_failed_count 每次重采样按 EMA 更新(adaptive_alpha=0.001),采样概率 = bin_failed_count + adaptive_uniform_ratio(0.1)/bin_count,再用因果填充的指数衰减核平滑(adaptive_lambda=0.8、kernel_size=1),归一化后 multinomial 抽下一回合的起始 bin;还会记录熵/top-1 概率监控课程集中度。— 直接先例(而且就是 fork 基座);R3 可以直接复用或对照的现成公式。
- **Learning to Get Up(SIGGRAPH 2022)**(paper, arXiv:2205.00307)— 三阶段课程,从任意摔倒姿态学起身、**不用动捕**:阶段一用强化/理想化角色先找到解法模式,阶段二逐步换成更弱的角色,阶段三把弱角色的动作放慢重放;产出的策略有静态稳定性(中途可暂停),还能推广到受限情形(比如一条腿打石膏)。— 类比(独立阶段式,而非混入式 RSI);R3/R6 的对照设计点。
- **HumanUp: Learning Getting-Up Policies for Real-World Humanoid Robots(RSS 2025)**(paper, arXiv:2502.12152;repo github.com/RunpeiDong/HumanUP)— 两阶段:先在最少约束下发现起身轨迹,再精炼成慢、平滑、可部署且对地形/构型鲁棒的动作;Unitree G1 零样本上真机,成功率 **78.3% vs 出厂 getup 控制器 41.7%**,还能解决出厂控制器解不了的可变形/湿滑/坡面。— 直接先例;R6 的"独立阶段式恢复课程也能成"的硬数字。
- **HoST: Learning Humanoid Standing-up Control across Diverse Postures(RSS 2025)**(paper, arXiv:2502.08378)— 多 critic 架构 + 对**辅助力**的课程:训练早期用一个竖直拉力扶着,随训练退火掉。**核查修正**:阶段门限是绝对高度 Hstage1=0.45 m、Hstage2=0.65 m,论文**从未**表述成"机器人身高的 ~35%/~70%";而且那个限制力矩/速度的 action rescaler β **不由这两个高度门限触发**,它是按训练进度单独退火的另一条课程(与竖直力退火同类但不同项),原表述把两个机制混成了一个。— 类比;R6"把任务约束随训练放宽"的对照模式(区别于按失败率重采样)。
- **HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery**(paper, arXiv:2502.20061)— 分阶段逐步纳入更复杂、更高维的恢复任务,真机验证,报告了跨多种摔倒构型的高成功率与快速恢复;阶段边界的定义从摘要里抽不出来。— 类比;R6 的第三条 2025 年佐证:"摔倒恢复自成一套阶段课程"是当下活跃方向。
- **Catching Spinning Table Tennis Balls with End-to-End Curriculum RL(2025)**(paper, arXiv:2503.01251)— 三阶段**任务**课程:阶段一奖励只管击中,阶段二加"打回去",阶段三加落点精度误差 e_lt;实现方式是一个按(轨迹状态,阶段)索引的奖励系数矩阵,只有当前阶段的项非零;阶段间**直接继承上一阶段权重不重置**以避免灾难性遗忘;PPO + D2RL,Isaac Gym。— 直接先例;R6 的"击中→回球→精度"分级课程,注意它是子任务复杂度分级,不是字面上的"目标框放大"。
- **HITTER: A HumanoId Table TEnnis Robot**(paper, arXiv:2508.21043)— 明确"following the approach of BeyondMimic"做动作参考跟踪,球拍目标奖励 r_g 是稀疏且时间门控的(只在击球时刻附近的短窗口激活),base 位置奖励同样门控;击球类型与目标在固定、互不重叠的正/反手区域内**均匀采样**——通读训练章节**没有**任何对目标区大小或来球速度的课程(全文连 "curriculum" 这个词都没出现)。真机 26 球:96.2% 击中率、92.3% 回球率,与人对拉 106 拍。— 直接先例(反向);R6 的关键反证:一个奖励结构几乎和我们一样的 SOTA 系统压根没用目标放大课程,说明这块文献空白是真的,不是我们没搜到。
- **Humanoid Whole-Body Badminton via Multi-Stage RL**(paper, arXiv:2511.11218)— 摘要原话的三阶段:"footwork acquisition, precision-guided swing generation, and task-focused refinement",专门设计成腿和臂共同服务于击球目标;仿真里两台人形对拉 21 拍、击球精度 0.10 m 位置 / 0.2 rad 朝向,真机零样本迁移,羽球速度到 19.1 m/s。— 类比(跨项目);R6 的第二个球拍类运动多阶段课程佐证。
- **Achieving Human Level Competitive Robot Table Tennis(DeepMind;arXiv:2408.03906,后发表于 Nature 2026)**(paper)— 把来球手工分成 7 类(Fast、Normal、Slow、Topspin、No-spin、Underspin、Lob),每个训练回合**先按"与该类当前回球率成反比"的概率抽类别,再在类内均匀抽初始球态**——就是把 PHC 的 clip 级困难挖掘搬到任务/发球条件级;这套任务分布是通过 7 轮对拉 + 2 轮发球的真实世界迭代、3 个月、50+ 位人类对手(共 14.2k 对拉 + 3.4k 发球初始球态)攒出来的,第 6 轮专门补了覆盖不足的慢球/高球。真机 45% 胜率(13/29),对初学者 100%、对中级 55%;**核查修正**:所谓"对高级选手 0%"是误读,论文原文是**根本没和高级选手打过比赛**,不是打了全输。— 直接先例;R6 核心想法(按失败率加权选任务条件,而不只是选动作 clip)最好的一条,机制、数字、真实迭代闭环都完整。
- **Robotic Table Tennis: A Case Study into a High Speed Learning System**(paper, arXiv:2309.03315)— 没有自动的逐回合课程,而是系统级的"由简到难"原则(原话大意:先做简单任务如击中球,再放大到复杂任务如与人对打);其中一个精度落点任务的加难方式被明确写成"a wider ball distribution"(更宽的来球分布)。— 类比;R6 里"目标框/来球分布放宽"最贴字面的一条,但它是人工设定的实验条件,不是自动课程表。

### F. reward 形状(§11.4 各项的文献锚)

**F1 全身角动量正则**

- **Centroidal dynamics of a humanoid robot**(paper, Orin, Goswami, Lee, Autonomous Robots 35:161-176 (2013), DOI 10.1007/s10514-013-9341-4;PDF 经 lava.kaist.ac.kr 镜像)— 定义质心动量矩阵(CMM),把广义速度投影到质心处的 6D 空间动量;摘要明说基于 CMM 的动量平衡控制器"can significantly reduce unnecessary trunk bending during balance maintenance against external disturbance",即把角动量当调控目标而不只是分析量。核查说明:该 PDF 直接抽文本失败,内容由多个独立二手来源交叉确认,不是第一手通读。— 仅理论;§11.4 F1 的概念出处。
- **Improved Computation of the Humanoid Centroidal Dynamics and Application for Whole-Body Control**(paper, Wensing & Orin, IJHR 13(1):1550039 (2016), https://www.cs.cmu.edu/~cga/z/Wensing_IJHR_2016.pdf)— CMM 及其导数可直接从关节空间质量矩阵与科氏项里取出,不需要专门算法;摘要原话是质心动量控制"has recently emerged as an important component of whole-body humanoid control, resulting in emergent upper-body motions and increased robustness to pushes",并演示用质心角动量(CAM)控制调节脚下净偏航力矩。— 仅理论;§11.4 F1 的主力理论锚(唯一第一手通读的那篇)。
- **Resolved Momentum Control(Kajita et al., IROS 2003, pp.1644-1650)**(paper, IEEE Xplore doc 1248880)— 经典 RMC:解出实现指定总线动量/角动量的全身关节速度,用于人形平衡与运动生成。**注意**:书目信息经多个二手列表交叉确认,但**原文全文未读**(IEEE/ResearchGate 均取不到)。— 仅理论;§11.4 F1 引用时要标明只确认了存在性与出处。
- **Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent RL**(paper, arXiv:2507.04140,MIT)— 手臂 agent 观测当前与目标 CAM,并用两项奖励训练:竖直 CAM 跟踪 r_CAM = exp(−((k̂_z−k_z)/(1+|k̂_z|))²/σ),水平 CAM 阻尼 r_dCAM = −min(0, Σ_{x,y} k^i·k̇^i);论文显式引 Orin/Wensing/Kajita,真机验证(行走/崎岖地形/上下楼),是主方法不是弃用消融。— 直接先例;§11.4 F1 最强的 RL 先例(角动量既做罚项又做观测)。
- **Achieving Stable High-Speed Locomotion for Humanoid Robots with DRL(KSLC)**(paper, arXiv:2409.16611)— 总角动量按各连杆质心贡献求和 L_total=Σ(c_i×m_i·ċ_i + I_i·ω_i),奖励取 r_t^a = clip(−exp(‖A_t^a‖₂), c1, c2),用于让摆臂帮助高速稳定(仿真里跟到 3.5 m/s)。— 直接先例;§11.4 F1 的第二个 RL 先例,函数形式与 MIT 那篇不同,可用来说明这是一"族"形状而非唯一写法。

**F2 力矩上限接近罚**

- **leggedrobotics/legged_gym 基础奖励集**(repo, legged_gym/envs/base/legged_robot.py L353-370 与 L868-870;legged_robot_config.py L152)— 出厂就有解析 PD 代理 `torques = p_gains*(q_des−q) − d_gains*qdot`(再钳到 ±torque_limits),以及尾部罚 `_reward_torque_limits = sum(clip(|τ| − soft_torque_limit*τ_lim, min=0))`,由 `soft_torque_limit` 这个"限值百分比"旋钮参数化,**默认 1.0 即等于关闭**,但机制是所有下游配置继承的基类奖励。— 直接先例;§11.4 F2 的 τ 代理公式与">x% 限值"罚形状的源头。
- **LeCAR-Lab/FALCON**(repo, humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml L162/L79;`_reward_limits_torque` 在 legged_robot_base_ma.py L679)— 出厂配置 `soft_torque_limit: 0.95`(只罚最后 5% 余量),上半身该项权重 −0.1,公式沿用 legged_gym 一脉的 clip(|τ|−soft·limit, min=0)。另有一个 `_reward_limits_upper_body_torque` 把 0.9 直接写死,源码里作者自己留了 `# TODO: Hardcode the 0.9`;**核查修正**:这个函数**不在** legged_robot_base_ma.py,而在 humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py。— 直接先例;§11.4 F2 的具体数值锚(95%),外加一句编辑性判断:连原作者都把这个硬编码分数当成毛边。
- **BoosterRobotics/booster_gym**(repo, envs/t1.py 与 envs/T1.yaml)— 每 substep 算解析 PD 力矩(`k_p*(target−q) − k_d*qdot`,补摩擦后钳到 ±τ_lim)再下发,并以此驱动两个饱和邻近罚:`_reward_torque_limits = sum((|τ| − τ_lim*soft_torque_limit).clip(min=0))` 与 `_reward_torque_tiredness = sum((τ/τ_lim)².clip(max=1))`。**核查修正**:T1.yaml 里 `soft_torque_limit` 实为 **1.0(不是 0.9)**,历史上也从未是别的值;而且该项的 reward scale 是 `torque_limits: -0.`,**默认根本不生效**——真正默认在跑的是 `torque_tiredness`(scale −1e-2)。— 直接先例(仅"tiredness"那一形状是出厂生效的);§11.4 F2 引用时必须把这个默认关闭状态写清楚,不要拿它当"业界默认罚力矩尾部"的证据。
- **Isaac Lab 的力矩相关奖励与执行器**(repo, source/isaaclab/isaaclab/actuators/actuator_pd.py L117-142 与 envs/mdp/rewards.py)— actuator_pd.py 的注释直言"computes the approximate torques for the actuated joint since PhysX does not compute this quantity explicitly",实现是 `computed_effort = stiffness*error_pos + damping*error_vel + joint_efforts` 再 `applied_effort = clip(computed_effort, ±effort_limit)`;`applied_torque_limits()` = Σ|applied − computed|,但 docstring 明确警告"only works for explicit actuators…For implicit actuators, we currently cannot retrieve the applied torques from the physics engine",而 `joint_torques_l2` 两种都能用。交叉检查:unitree_rl_lab 的 G1/H1(用 ImplicitActuatorCfg)**干脆一个力矩类奖励都不配**。— 直接先例;§11.4 F2 最该写进报告的诚实注脚:官方那条力矩邻近罚在隐式 PD 下不被支持,而同框架的人形 repo 是靠"不用"来绕过这个问题的。

**F3 站宽/脚距成形**

- **roboterax/humanoid-gym(XBot-L,arXiv:2404.05695)**(repo, humanoid/envs/custom/humanoid_env.py L282-306;humanoid_config.py L176-197)— 出厂奖励是两侧指数带:`d_min=clip(dist−min_dist, −0.5, 0)`、`d_max=clip(dist−max_dist, 0, 0.5)`,reward=(exp(−100|d_min|)+exp(−100|d_max|))/2,配置 min_dist=0.2、max_dist=0.5,脚和膝各一份,权重都是 0.2 且默认非零。— 直接先例;§11.4 F3(这是有同行评审、零样本 sim2real 的仓库,不是玩具)。
- **BoosterRobotics/booster_gym 的 _reward_feet_distance**(repo, envs/t1.py L719-725;T1.yaml `feet_distance_ref: 0.2`)— 在 yaw 系里投影出横向脚间距,奖励 `clip(feet_distance_ref − feet_distance, 0, 0.1)`,是单边的"别靠得比参考更近",与 humanoid-gym 的双边带不同。— 直接先例;§11.4 F3 的第二种出厂写法。
- **LeCAR-Lab/ASAP 与 FALCON 的 close_feet_xy / close_knees_xy**(repo, ASAP humanoidverse/envs/locomotion/locomotion.py;FALCON reward_dec_loco_stand_height_ma_diff_force.yaml)— `norm(left_foot_xy − right_foot_xy) < 阈值` 时直接返回 1.0 的二值罚(膝同理),FALCON 出厂 yaml 里 `penalty_close_feet_xy: -10.`(阈值 0.17)生效、`penalty_close_knees_xy: -2.`(阈值 0.16)被注释掉;该 yaml 就是 FALCON README 默认训练命令调用的那份。— 直接先例;§11.4 F3 里与报告 `feet_close_xy` **同名**的第三个独立实现,用的是二值阈值而非平滑带。

**F4 自碰撞罚的写法**

- **leggedrobotics/legged_gym 的 _reward_collision**(repo, legged_robot.py L849-851)— `sum(1.0 * (norm(contact_forces[penalised_indices]) > 0.1))`,纯粹是"接触力范数超过 0.1 N 的身体数",计数而非力值求和。— 直接先例;§11.4 F4 的祖型,下面几乎所有派生仓库都继承它。
- **roboterax/humanoid-gym 的 _reward_collision**(repo, humanoid_env.py L523-529;config `collision: -1.`)— 同为计数式,触发阈值低到 0.1 N(基本等于"有接触就记一次"),权重 −1.0 默认生效。— 直接先例;§11.4 F4 的第二个出厂实现,那个极低阈值本身就值得在报告里点一句。
- **Isaac Lab 的 undesired_contacts 与 contact_forces**(repo, source/isaaclab/isaaclab/envs/mdp/rewards.py)— `undesired_contacts(threshold, sensor_cfg)` = Σ(历史最大力范数 > 阈值),是**计数**;`contact_forces(threshold, sensor_cfg)` = Σ clip(最大力范数 − 阈值, min=0),是**力值超额和**;两者作为同一信号的两种写法并存于官方奖励模块。— 直接先例;§11.4 F4 的"计数 vs 力值"二分法,就在我们自己的框架里。
- **unitreerobotics/unitree_rl_lab 的 H1/G1 配置**(repo, .../tasks/locomotion/robots/h1|g1/velocity_env_cfg.py)— 两个官方配置都只挂 `undesired_contacts`,threshold=1、weight=−1、body_names=`['(?!.*ankle.*).*']`(所有非踝身体),这是它们唯一的自/非期望接触项。— 直接先例;§11.4 F4 的"官方人形配置就用计数式"证据。
- **BoosterRobotics/booster_gym 的 _reward_collision**(repo, envs/t1.py L627-629)— `sum(norm(contact_forces[penalized_indices]) > 1.0)`,同样计数式,阈值 1.0 N。— 直接先例;§11.4 F4 的第四个独立仓库,口径一致:**能查到的默认配置里,没有一个是对自碰撞做原始力值求和的**。
- **PPL: Point Cloud Supervised Proprioceptive Locomotion RL for Legged Robots in Crawl Spaces**(paper, arXiv:2508.09950 Sec. III-C)— 原话:"the collision states penalties are sparse, and twelve joint actions are not easy to be searched simultaneously along the proper directions in the training. Therefore, we add force related components to densify collision penalties";其罚项(Eq.9)是二值碰撞项 + 指数力值项 `Σ λ_i(1 − exp(−μ_i‖f_i^xy‖)) + c_head + c_base + Σc_hip`,接触力"obtained from Isaac Lab"。**重要限定**:这是**环境**碰撞(爬行空间的顶/壁,四足),不是机器人自碰撞。— 类比;§11.4 F4 里"稀疏计数罚学不出修正方向,所以要用力值稠密化"这个机理的唯一实读出处,但要写明它不是自碰撞。
- **Isaac Gym Simulation Tuning 文档**(official-doc, https://docs.robotsfan.com/isaacgym/programming/tuning.html)— 原话大意:不稳定的一个来源是初始构型不佳与存在自碰撞,"可以通过可视化系统里的接触力并尽可能关掉自碰撞来诊断";并说明深穿透会注入巨大的求解器修正力。**核查修正**:该修正力的调节参数 `max_depenetration_velocity` 的**文档默认值是 100 m/s,不是 5 m/s**——5 m/s 只是底层 PhysX API 的原生默认,Isaac Gym 明确把它上调("a larger value, e.g. 100m/s, can help")。— 仅理论(仿真器层面,不是奖励设计层面的消融);§11.4 F4 里"为什么原始力值自碰撞罚会噪、会不稳"的机制依据,引用时不要把它说成 RL 训练发散的实证。

**F5 骨盆 vs 躯干直立项拆分**

- **LeCAR-Lab/FALCON 的 orientation 双项**(repo, humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml)— 出厂默认同时启用 `penalty_orientation: -1.5`(来自 `self.projected_gravity`,即本体/根系倾斜)与 `penalty_torso_orientation: -1.0`(显式索引 torso_link 刚体做 `quat_rotate_inverse(rigid_body_rot[:, torso_index], gravity_vec)`,是另一次刚体位姿查询,不是根状态)。**核查修正**:`_reward_penalty_torso_orientation` 的实现在 humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py,**不在** ASAP 的 locomotion.py(ASAP 根本没有 torso_orientation 这一项)。— 直接先例;§11.4 F5 最贴的一条:出厂配置里真有"骨盆/根 + 躯干"两个独立朝向罚,权重还不同。
- **LeCAR-Lab/ASAP 的 G1 移动配置**(repo, humanoidverse/config/rewards/loco/reward_g1_locomotion.yaml L15/L36;`_reward_penalty_orientation` 在 legged_robot_base.py L696 = Σ(projected_gravity_xy)²;**核查修正**:`_reward_penalty_ang_vel_xy_torso` 在 humanoidverse/envs/locomotion/locomotion.py L161,不在 legged_robot_base.py;simulator/isaacgym/isaacgym.py L346-348 显示 torso_index 找不到 'torso_link' 时会退回 'pelvis')— G1 出厂同时开 `penalty_orientation: -1.0`(骨盆/根系)与 `penalty_ang_vel_xy_torso: -1.0`(torso_link 角速度),等权;H1_2 的配置里躯干那项是注释掉的。— 直接先例(但形状是"骨盆朝向 + 躯干角速度",不是两个朝向项);§11.4 F5 的第二个佐证,同时提醒:机器人若没有独立躯干体,这个拆分会退化成同一个身体。
- **Isaac Lab 官方 G1 rough_env_cfg.py**(repo, source/isaaclab_tasks/.../locomotion/velocity/config/g1/rough_env_cfg.py)— 只有一项 `flat_orientation_l2`(weight −1.0)作用在 articulation base 上,源码注释写着 `# G1 uses "torso_link" as base body`;torso_link 同时是外力事件体与 base-contact 终止判定体,**没有**单独的骨盆项。**核查修正**:质量随机化(`add_base_mass` / `base_com`)对 G1 是直接设为 None(整个关掉),不是改挂到 torso_link。— 直接先例(反向对照);§11.4 F5 的校准点:在我们同一个框架家族里,**单体惯例才是多数派**,FALCON/ASAP 的拆分属于少数派,所以厂商这一项"新颖度"要按此定级。
- **BeyondMimic / whole_body_tracking 的奖励模块**(repo, source/whole_body_tracking/.../tasks/tracking/mdp/rewards.py,全文件通读)— 只有动作跟踪的指数项(anchor 位置/朝向、相对身体位置/朝向、身体线/角速度误差)加一个 `feet_contact_time`;**没有**角动量项、**没有**力矩邻近项、**没有**脚距项、**没有**自碰撞项,朝向锚点只有一个 'torso_link'、没有骨盆拆分。— 直接先例(负向);§11.4 五项全部的定性依据:**这五个形状都不是 BeyondMimic 自带的,是厂商在 fork 之上从 legged_gym/ASAP/Booster-Gym 那套移动奖励库里搬来的**。

### G. 源头论文 HITTER / SMASH / PACE 三分账(全程本地:PDF 全文 pdftotext 比对 + 本地 review 交叉,17/17 核查)

**G.1 HITTER(arXiv:2508.21043v2;本地 PDF 即最新版;ICRA 2026 最佳论文入围-Planning and Control)**
- **已取(核实成立)**:PD 固定("set heuristically following BeyondMimic")✅;reward 权重/
  核宽未发表(Eq.7 仅符号式)✅;asymmetric critic、50 Hz、MLP [512,256,128] 等(参考 setup 文档)。
- **勘误(已当日修复)**:①repo 注释"HITTER's prose DR = mass/friction/restitution + 感知噪声/
  延迟"系**把 PACE §IV-A-1 张冠李戴**——HITTER 全文零 DR 内容(无 randomiz/delay/push 字样);
  hope_env_cfg.py 975-977 注释已改正。**推论:"HITTER 对齐:无推撞"从来没有文献依据**(HITTER
  不是说不推,是什么都没说),且 SMASH/PACE 都推——推撞裁定(§3.1/v2.3)的执行必要性再加强。
  ②HOPE_WBC 参考文档 episode 长度"推测 ~3.4 s"→论文 §V-B-1 明写 **10 s**(=500 步@50Hz,
  每拍后重抽 swing type+目标,击球平面固定 0.4 m,正反手目标区不重叠),已修。
- **尚未开采**:Fig.3 规划预测误差双阈值曲线(0.5 s 前跨 7.5 cm 拍半径阈、0.3 s 前跨 20 ms 阈);
  Fig.4 WBC 敏捷性(1000 rollouts 943 有效=94.3%,base 偏移<0.75 m 时收敛<0.8 s);
  真机 26 抛 24 回、106 拍人机对拉、0.42 s 扣杀回合。

**G.2 SMASH(arXiv:2604.01158v1,仅 v1、暂无 venue、无公开代码)**
- **已取(核实成立)**:门控 exp 核 r_j=exp(−e_j/σ_j)·I[τ∈W_j],位置窗 0.02 s 紧 vs 姿态/速度窗
  0.1 s 宽(Eq.12);adaptive tracking sigma(在线按运行误差收紧,粗到精)与 adaptive region
  sampling(向低成功区重分配)两机制分立——我们退役的 adaptive_sigma 与 σ 课程钦定即源于此;
  本地 review 的 SR 数字全部对上原文。
- **尚未开采**:①Motion-VAE 生成库完整训练目标(KL 周期退火 Eq.4、相位循环重建 Eq.5-6、
  自回归边界平滑 Eq.7、足穿地罚 z_g=0.035 m)——对 canonical 动作库战役直接相关;
  ②**Table IV:SMASH 第三方复现"HITTER 式上身-only 模仿"基线,任务 SR 86.63% vs 全身 86.38%
  打平,但动作质量差 33%(E_mpjpe 100.05 vs 75.01)、力矩更高(5.79 vs 5.58)**——
  "上身 only vs 全身模仿"之争目前唯一受控证据,直接关联我们全身模仿准绳;
  ③§VI-C-d 642 连发 50 min 硬件测试:总触球 93.7%,正手回球 66.7% vs 反手 38.9% 不对称
  (反手 miss 9.0% + 52.1% 触而不回);Table VI 感知消融(去 AEKF 弹跳处理 3.5→12.7 cm,
  本地只引过后者)。
- **空档**:无代码,DR 幅度/reward 权重(Eq.13 的 w_pos/w_ori/w_vel)/PPO 超参不可验证
  (论文只有定性"push+摩擦随机化"——注意:**SMASH 有推撞**)。

**G.3 PACE(arXiv:2509.21690;本地 PDF v3 已落后 arXiv v4 一版——待替换)**
- **已取(核实成立,引用卫生升级)**:"push vxy ±0.2 每 5-15 s"确认为 **shipped default**
  (TTRL-ICRA2026 @aad2fc7,T1TableTennisEnvCfg→TTEnvCfg.domain_rand 继承链)——但**论文正文
  没有这个数**,引用一律指向代码 commit;`hit_unstable_support -10`(触球瞬间失稳罚)同为
  code-only,我方表述精确成立;PACE 的 DR 定性句即被误标为 HITTER 的那句(见 G.1)。
- **尚未开采(核查刮削后仅剩一项)**:PACE actor 观测**世界系绝对 base 位置**(mocap 来源)——
  与 HITTER 相对指令方案相反的设计选择,本地零讨论;其余"未开采"候选五项
  (residual 动作式/predictor MLP[64,64]/22 轨迹阻力拟合/Table II/31-29-19)
  ball_physics_2026-07-03/pace.md 已有等精度记录(核查否决了检索 agent 的"未开采"断言)。
- **行动项**:papers/ 换 v4 PDF;PACE 数字引用指向代码;两处勘误已修(见 G.1)。

### 诚实空档

- **MJWarp 的一手 benchmark 数字拿不到**:到处传的"RTX 4090 上 70x/152x/313x"只出现在二手博客/聚合站;MJWarp 的 GitHub README 与 benchmarks 目录只有跑分脚本没有内联数字,官方 mujoco.readthedocs.io/en/latest/mjwarp/ 页面在本次会话里加载失败,因此这些数字一律未采信。
- **Genesis vs Isaac Gym vs MJX 的腿足机器人官方对比表拿不到**:唯一可能存在该表的是一份看起来私有的 Notion 文档,取不到内容;只有 Franka 单臂 43M FPS 这一个数字(及其被拆穿的过程)是可读来源确认的。
- **2025–2026 里针对桌面/箱体接触人形 loco-manipulation 的 GPU 接触求解吞吐数字**(env-steps/s、墙钟)未确认:找到了几篇题材相关的 2026 论文(MPC-RL π^n 批量求解器推箱、HMC 接触密集 loco-manipulation、IsaacIPC),但仅凭可访问的摘要级文本无法确认仿真器与吞吐细节,故不引未证实数字。
- **没有官方文档把"接触传感器报告开销"量化成帧时间百分比**:唯一有量化的来源是维护者在 isaaclab#5018 里的 profiling,不是官方 benchmark 页面。
- **没有找到真正 Huber 式的单函数 reward kernel 先例**(近零处二次/指数、远处平滑过渡到线性,并被作者如此描述):最接近的 IsaacLab Reach 是**两个分开的加权项**(裸 L2 粗项 + tanh 细项),构造上与融合式 Huber 核实质不同,不能拔高成"Huber 先例";而 TCN 是 L2 + pseudo-Huber **相加**,也不是单函数。
- **没有一篇论文同时做到**(a)显式引 Ng 1999 势能塑形、且(b)在机器人操作/移动任务里用**裸距离**势能并做有对照的塑形/不塑形消融:引 Ng 的两篇(2402.04581、2307.10142)用的是学出来或从奖励项导出的势能,用裸 Δ距离的(OpenAI 1808.00177、PHC)又不引 Ng。形式理论与工程实践只能分别佐证,拿不到一份两全的来源。
- **'Self-Imitation Learning of Locomotion Movements through Termination Curriculum'(arXiv:1907.11842)未采信**:看起来是早期"终止阈值退火"的好候选(二手摘要说初始阈值 0.75 放宽到 0.5),但 PDF 两次抽取都是乱码/二进制,配套仓库是老 C++/ODE 代码来不及查,按"没读过不引"原则剔除。
- **灵巧多指操作 repo 里没找到出厂默认的粗核+细核多级栈**:查过 IsaacGymEnvs FrankaCubeStack(只有单 tanh 项),ShadowHand/Bidex/DexPBT 的奖励代码没来得及查;这一模式最强的已确认先例反而来自 IsaacLab 非灵巧的单臂 Reach 任务。
- **没有按迭代数固定表退火 σ 的先例**(像地形难度/指令范围那种挂在训练进度上的课程):已找到的两条(KungfuBot、NEAR)都是误差自适应/数据驱动的退火,"纯迭代计划表式 σ 课程"仍未证实。〔另注:原先"KungfuBot 的 adaptive sigma 是否在出厂配置里默认开启无法确认"这条空档**已被填补**——PBHC 的 main.yaml L60-64 已核实为默认开启。〕
- **没有任何人形/腿足 residual-RL 论文公布过具体的 PPO 探索噪声 std 或 log_std 初值**:唯一"PPO + 腿足 + 零初始化"的那篇(arXiv:2407.17683)明说 σ 是"a scheduled parameter",但全文从未给出这张表的数字。
- **没有任何论文、博客或 issue 把 adaptive-KL 学习率的 (Δμ/σ)² 爆炸单独当成"warm-start 近零 σ"的失效模式来分析**:最近的类比是 DAWN(arXiv:2602.10539)记录的 |log π| 爆炸,但那是 SAC 的自动熵项 α·log π,不是 PPO 的 adaptive-KL-LR 项;同为 1/σ 型放大,算法族不同,也找不到把两者显式打通的来源。
- **`RuntimeError: normal expects all elements of std >= 0.0` / NaN-std 崩溃没有任何同行评审论文诊断过**:只有 GitHub issue(rsl_rl #33、IsaacLab #673)和 PR 讨论(#67/#190/#201),按"工程论坛级记录、非学术分析"定级。
- **Johannink et al. 2019(arXiv:1812.03201)确认没有零/近零初始化最后一层**:是专门找过之后确认的缺失,不是片段没提到。
- **HumanPlus(CoRL 2024)与 H2O/Human2Humanoid(IROS 2024)不能当摔倒恢复先例引**:两篇都不把摔倒恢复当核心贡献(HumanPlus 是全栈影随模仿,H2O 是 sim-to-data 的特权模仿器筛选流水线),这一族里恢复最多出现在相关工作/未来工作,查不到可确认的机制。
- **2023–2026 的乒乓/网球/羽毛球机器人论文里,没有字面意义上的"目标框逐步放大"或"球速阶梯"自动课程**:HITTER、PACE、DeepMind 的 Nature/arXiv 系统、羽毛球多阶段、旋球接球课程全查过。最接近的两类替代是(a)子任务复杂度分级(击中→回球→精度,或步法→挥拍→精修)与(b)DeepMind 的类别级失败率加权来球采样——都在上面引了,但都不是字面上的目标框放大或速度阶梯。
- **全尺寸(非桌上)网球人形的课程机制未确认**:找到一篇很新的 'Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data'(arXiv:2603.12686,Unitree G1),但摘要没说是否/如何用课程学习,可访问内容里确认不出机制,宁可不列也不凑。
- **HiFAR(arXiv:2502.20061)的阶段边界定义与"有无课程"的定量消融拿不到**:全文 PDF 抽不出阶段切换判据,上面的引用严格限定在摘要级内容。
- **Isaac-Lab 系 / legged_gym 系仓库里没有任何 CAM(角动量)奖励或观测的出厂默认**:查过 isaac-sim/IsaacLab 的 mdp/rewards.py、HybridRobotics/whole_body_tracking、unitreerobotics/unitree_rl_lab、LeCAR-Lab 的 HumanoidVerse+ASAP+FALCON、BoosterRobotics/booster_gym。唯一的具体 RL 先例是 2025 年那篇 MIT 论文(未找到公开代码),所以"角动量正则"目前只能定性为**论文技术,而非这一框架家族里的既成惯例**。
- **没有任何仓库出厂用力值幅度做自碰撞(机器人对自身连杆)罚**:查过的 7 个仓库(Isaac Lab、legged_gym、humanoid-gym、unitree_rl_lab、Booster Gym、HumanoidVerse/ASAP/FALCON)里所有自碰撞相关项都是计数/二值阈值式。也**找不到**一份专门把训练不稳定归因于"力值幅度自碰撞罚"的对照消融——唯一读到的"稀疏罚难学、故用力项稠密化"论证来自 PPL(爬行空间四足的**环境**碰撞),只能当类比。
- **'Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostic Motions'(arXiv:2502.01465)未引用**。核查修正:原先记为"页面和 PDF 都取不到"的说法不成立——重试时摘要页正常返回。真实空档是:**从中没有提取出任何可确认的、可引用的结论**,故不列入参考。
- **Kajita et al. 2003(IROS,Resolved Momentum Control)原文全文未读**:IEEE Xplore 与 ResearchGate 均拒绝抓取,只交叉确认了书目信息;Orin et al. 2013 的内容也是靠多个独立二手来源交叉确认(PDF 直接抽文本失败),不是第一手通读。真正第一手读完的理论锚是 Wensing & Orin 2016。
- **没有找到"骨盆直立 + 躯干直立"两个字面同型的朝向项同时出厂的配置**——注:这条空档**已被部分填补**,FALCON 出厂 yaml 里 `penalty_orientation: -1.5` 与 `penalty_torso_orientation: -1.0` 确实同时生效(见 F5)。仍然成立的部分是:ASAP 那一份是"骨盆朝向 + 躯干**角速度**",不是两个朝向项。
- **隐式驱动下估计实际力矩,没有任何非 PD-proxy 的替代技术先例**:找不到学习式力矩估计器,也找不到直接查询 PhysX 内部量的做法;整个领域看起来一致地依赖同一个 kp(q_des−q)+kd(q̇_des−q̇) 近似。

---

补充说明(不进正文,供你决定要不要落到报告里):

- **来源与去重**:Topic B 与 Topic F 各有两份 hunt 文件,已按 locator 合并;Topic B/C 的另两份是同名同内容副本(逐字节相同),已忽略。HITTER/SMASH/PACE origin-papers 那份按指示未纳入(但其中 PACE/SMASH 在 Topic A 里独立出现,已保留)。
- **共应用了 12 处核查修正**,最要紧的三处是:booster_gym 的 `soft_torque_limit` 实为 1.0 且该项默认权重为 0(不是 0.9、也不生效)、Isaac Gym 的 `max_depenetration_velocity` 文档默认是 100 m/s(5 是 PhysX 原生值)、PHC 的 G1/H1 配置里那三个恢复参数是死键(真正接线的是 SMPL 角色任务)。另有 Policy Decorator 的 α 全表范围 0.03–0.8、PBHC 默认 origin 分支是朴素 min 无 scale 乘子、DeepMind 对高级选手是"没打过比赛"而非 0 胜率、HoST 的阶段门限是绝对 0.45/0.65 m 且 action rescaler 不由它触发。
- **B 段里 6 条标了〔二次核查未覆盖〕**(TCN、2001.03792、HPRS、AMP、PHC 论文、NEAR):它们来自那份没有配套 verify 文件的 hunt 结果,单看仍可引,但若要当承重论据,建议先补一次一手核对——尤其 TCN,因为它是 P2 最贴字面的一条。


---

## 十三、Curriculum 升级机制专项(08-01;应 Franco"升级设置合不合适"之问)

**方法**:2 抽取(我方机制 + 7 家本地克隆升降级规则普查)+ 2 对抗核查(8 修正/12 漏项)
+ 1 裁决(已核实的 §12.E 论文先例一并入证)。全程本地,零 web 依赖。

### 13.1 我方机制全貌(修正后;两处前提纠错)

- **前提纠错①**:`curriculum_rollout_end_advancement_disabled` 不是代码开关——全库 grep 零命中,
  它只是 N1 wave 合同 json 里的人话标签;真实开关是 `task.racket.action_ball_diagnostic_unauthorized=true`
  (N1 发射器恒传,发射器 claim 里明写 `curriculum_promotion_prohibited: True`,Franco 07-28 批准的
  bypass)。**前提纠错②**:`on_rollout_end` 在代码里明写 REPORT-ONLY("live PPO rollouts never
  advance the curriculum")——升级从来不在训练 rollout 里发生。
- **难度轴**:32 条 arm = 16 物理轴×双侧(触球时刻抖动、触点 x/y/z、来球速度、旋转幅度、
  base 出生/移动 x/y、落点瞄准 x/y、来球方向锥 u/v、旋向 u/v),每条 arm 独立 5 档
  L∈{0,.25,.5,.75,1},宽度 = initial + L·(max−initial) 再受硬物理帽;`no_move` 模式自动剔除
  base_travel 组(活跃 26-28 条)。此外还有一个**全局 joint-rho** 标量(同 5 档)在 marginal
  之后统一缩放所有已认证 arm。状态按 (action_uid, profile_sha256, mobility) 全局共享,非逐 env。
- **升级判决**:冻结策略 checkpoint 上跑**canary(≥256/目标 320)→ heldout(≥768/目标 960)双窗口**,
  由独立评测 sidecar 执行、文件收件箱回传;指标 = Wilson 95% CI 下失败率对准 10%±2.5% 目标带,
  外加 5 个全域质量 blocker(admit/install/start/close ≥95% + 零容忍撞台/qdes/实关节/归因异常);
  marginal 阶段只用**新带子样本(≥154 行)**判被探的那条 arm;节拍 = 每 100 PPO updates 至多一轮,
  认证后需全局 drain+reset。
- **降级**:单臂**零降级、一次定终身**(失败即 `lock_marginal` 永久锁档);全局 rho 一个坏窗口
  立即退一档(无连败缓冲);center 连败 8 次 → 终态 `stalled` 等人工。
- **现状**:N1 全线 diagnostic_unauthorized=true → 域永久钉在 manifest initial,**整套晋级机器
  从未在真跑中端到端行使过**;正式 N5(launch_action_ball_curriculum.py 设 false)将是首次通电。
- **checkpoint**:curriculum state_dict + sha256 交叉校验,resume 失败 fail-loud——全场最好。

### 13.2 裁决(原文入档)

骨架是对的,而且有三处明显强于所有外部先例;但三个结构性设定叠在一起,会让它在真跑里"早早锁死",实际扩张远低于设计意图。而且当前活着的 N1 全线晋级是关掉的,所以现在跑的本来就是固定难度。

强于先例的地方(应该保留):
1) 指标选对了。把失败率钉在 10%±2.5% 的带内,等于"把难度维持在能力边缘",比 legged_gym / unitree 的 "tracking reward > 0.8×权重" 这种启发式更有原则,和 Dextreme-ADR 的 5/20 成功数死区、DeepMind 乒乓按各发球类回球率反比配额是同一族思路。
2) 冻结策略 + canary→heldout 双窗口:没有任何一个外部这么做。legged_gym / IsaacLab / mjlab / PBHC / beyondmimic 全都是在活策略上边跑边判,统计上是脏的;我们是唯一把判决和策略漂移解耦的。
3) 状态 checkpoint 是全场最好的:state_dict + curriculum_state_sha256 交叉校验 + resume 失败即 fail-loud。外部里只有 ProtoMotions 的 motion_weights 确认存盘、ADR 有个 adr_load_from_checkpoint 开关、mjlab 只存了 step counter;legged_gym / IsaacLab / unitree / beyondmimic 的课程进度 --resume 后直接归零。

对比先例最不寻常的四点(都是风险):
1) 32 条轴独立晋级。所有外部要么 1 轴(地形)要么 2 轴(地形+指令);唯一多轴的是 ADR 的 ~10 参数×2 边界 ≈ 20 个边界——但 ADR 是并行异步推进的(clear_other_queues=False),我们是一次只探一条、串行走 100-update 的节拍。
2) 单臂完全没有降级,而且是"一次定终身"。legged_gym / IsaacLab / mjlab 地形是双向的,ADR 是双向且永不永久锁定(队列清空后同一边界无限重测),PBHC 罚项/软限是双向的。我们所有外部先例里唯一没有任何回头路的设计;更反常的是滞回方向反了——单臂永不下降,而全局 rho 一个坏窗口立刻退一档、连败缓冲都没有,连我们自己已退役的 task_first_curriculum.py 都有 enter_dwell/exit_dwell 连续窗口滞回 + rollback。
3) 判决样本量与判据不匹配。marginal 用的是新带 154 行:在 z=1.96、带 [0.075,0.125] 下,"too_easy 继续扩"实际要求 F≤5/154(3.25%),而 F 落在 6–27(3.9%–17.5%)一律"认证并永久锁死"。也就是说 too_easy 门测的不是"低于 10% 目标",是"低于 3.25%"。真实失败率 5% 的臂有 79% 概率第一轮就被永久锁在当前档,真实 7% 的是 96%。叠加"永久",净效果是系统性欠扩张。
4) 现在 N1 全线 diagnostic_unauthorized=true → 域永久钉在 manifest 的 initial,整套晋级机器在真跑中从未被端到端行使过一次;正式 N5 长跑会是第一次。HITTER 完全没有课程也拿到 92.3% 回球率,所以固定难度本身不丢人——但这意味着真正的设计决策是 manifest 的 initial 宽度,而现在的审查精力全花在了没通电的晋级机器上。

预算上能塞得下但没有余量:一次请求同时开 canary+heldout(hope_commands.py:9203-9207),所以一次判决至少 100 updates。no_move + counter-rally 掩码下活跃臂 26 条,按上面的功效分析绝大多数臂第一轮就锁 → 整个 marginal 阶段约 28 轮 ≈ 3000 iteration 就走完,剩下 17k–22k iteration 域再也不动;理论满扩张要 ~109 轮 ≈ 11k iteration,但几乎不会发生。更糟的是这些永久决定是在训练早期(~3k iter)用当时最弱的策略做出的,而所有外部先例都持续重判、早期悲观可自愈,我们不能。

结论:metric / window / frozen-policy / checkpoint 这几项设计得好甚至超前;threshold+样本量、可逆性、blocker 作用域这三项配错了,叠起来会把一个精心设计的课程变成"前 3000 步微调一下、之后等于没有课程",而且带着完整的仪式成本。建议按下面 R1–R3 修,全部可以做成默认 byte-identical 的新 key。

### 13.3 问题清单(按严重度)

| 严重度 | 问题 |
|---|---|
| critical | 单臂晋级不可逆 + 判决样本只有 154 行 → 系统性欠扩张,而且是在训练最早期用最弱策略做出的永久决定 —— action_ball_curriculum.py:4683-4694:marginal 分支 quality_bad or too_hard → statuses[index]='decided'(bound_marginal),否则 certify;只有 too_easy 且未到 1.0 才继续探,其余一律 'lock_marginal' status='decided',代码中无任何路径把 'decided' 改回可探。判决样本 n=NB(floor HELDOUT_NEW_BAND_MIN=154, :79)。实测 Wilson(z=1.96, 带 0.075/0.125):n=154 时 expand 需 F≤5(3.25%),F∈[6,27](3.9%–17.5%)全部永久锁死,F≥28 才判 too_hard。二项功效:真实失败率 5% → P(继续扩)=0.213 / P(永久锁)=0.787;7% → 0.038 / 0.962;3% → 0.683 / 0.317。对比:n=768 时 expand 阈值才升到 5.6%,n=4000 才到 6.7%。所有外部先例(legged_gym legged_robot.py:421-452 每次 reset 重判、ADR adr_vec_task.py:760-917 队列清空后同一边界无限重测、PBHC legged |
| critical | '样本不够'被当成'太难':NB<154 直接置 quality_bad,在 marginal 分支等价于 too_hard → 永久锁死该臂;而代码底线窗口在算术上恰好没有余量 —— action_ball_curriculum.py:4640-4643 `if evidence.ledger.NB < heldout_min_new_band: blockers += ('new_band_safe_closed_below_gate',); quality_bad=True`,随后 :4683 `if quality_bad or too_hard: statuses[index]='decided'` → bound_marginal 永久锁。而新带比例 = frontier_slots/(center+interior+frontier) = 1/5 = 0.2(action_ball_sampling.py:880-884),768×0.2 = 153.6 < 154 = HELDOUT_NEW_BAND_MIN。即在代码底线窗口(safe_closed=768)下期望新带行数就低于门限;运营目标 960 行只有在新带 safe-close 率不低于全域均值 80% 时才有余量——而新带恰恰是最容易不闭合的那 20%。对比 ADR:deque 未满 256 时不做任何决定,只等(adr_vec_task.py:760-917)。 |
| high | 整窗零容忍安全 blocker 门控的是单臂新带判决:一次来自 center/interior 行(占 80%)的撞台或关节越限,会永久锁死一条完全无关的轴;且随域变宽概率单调上升 → 晋级越往后越不可能 —— action_ball_curriculum.py:4498-4507 blockers 由 ledger.U_table / U_joint_qdes / U_joint_actual / X 的全窗计数决定,:4630-4636 的注释明确写 'Admission and every safety blocker above intentionally remain whole-domain gates',而 :4683 让 quality_bad 与 too_hard 走同一分支(永久锁)。窗口 1280 行(canary 320 + heldout 960)下,单回合零容忍事件率 1e-4 → P(至少一次)=12.0%;5e-4 → 47.3%;1e-3 → 72.2%。即只要残余不安全率高于约 1e-5,晋级流水线就会被无关事件随机掐断。对比:HoST 用绝对高度门与难度课程分离,ADR 的 limits 是外层安全夹、与 objective 死区是两套独立机制。 |
| high | 当前 N1 全线晋级被硬关闭,整套机器从未在真跑中端到端行使过;首次通电会发生在 20k–25k 的正式长跑里 —— scripts/launch_n1_reward_screen_diagnostic.py 恒传 task.racket.action_ball_diagnostic_unauthorized=true 且 claim 里写 'curriculum_promotion_prohibited': True(:1838);hope_commands.py:4630-4733 该标志下不绑定 evaluator_authority / drain_reset_authority,:9060-9071 frozen_evaluation_boundary 直接短路返回 {'diagnostic_unauthorized': True},canary/heldout 永不请求。configs/n1_contact_20260729/n1_upper_reward_wave_20260729.v1.json:21 的 'diagnostic_fixed_domain': true 与之一致。正式路径 launch_action_ball_curriculum.py:3873 才置 false。docs/ 下 grep 'expand_marginal|lock_marginal|center_pass' 零命中,即没有任何真跑产生过晋级判决记录(单元测试有,hope_training/whol |
| medium | 滞回方向反了:单臂零降级、零重试,而全局 rho 一个坏窗口立即退一档、无连败缓冲;我们自己已退役的 task_first 反而有连续窗口滞回 —— action_ball_curriculum.py:4712-4759:'joint' 首次失败 joint_rho_index=max(0,candidate-1) 并转 steady(bound_joint);'steady' 失败 joint_rho_index -= 1(retreat_joint),条件里没有任何连败计数;rho=0 再失败 → phase='stalled' 终态。对照 task_first_curriculum.py:574 `_rollback_axis_index` 与 :661-665 `indices[rollback_index] -= 1`,由 exit_dwell 连续窗口门控。外部对照:ADR 死区 5/20 两侧对称且夹回 init_range 而非终态;PBHC 40/42 双阈值死区双向;legged_gym 下降夹到 0。 |
| medium | 串行单臂探索 × 100-update 节拍,与 26–28 条活跃臂和 20k–25k 目标之间没有余量;且每次认证要全局 drain+reset,request_due 被 needs_reset 阻塞,有效节拍还要更长 —— hope_commands.py:9203-9207 一次请求同时开 ('frozen_canary','frozen_heldout'),:9103-9124 request_due 要求 no_inflight and not needs_reset and step-last>=interval,:18674 interval 默认 100。action_ball_curriculum.py 一次只有一个 selected_arm_key(_reselect_arm 轮转)。活跃臂:no_move 排除 4 条 base_travel + counter-rally 掩码 2 条 landing_aim_y → 26 条。最少 ~28 轮(每臂 1 轮即锁)≈3k iteration 走完 marginal;满扩张 26×4+center+joint ≈109 轮 ≈11k iteration。对比 ADR:~20 个边界并行异步推进(clear_other_queues=False);legged_gym/IsaacLab:数千 env 每次 reset 全并行判决。 |
| medium | center 阶段连败 8 次 → 'stalled' 终态需人工介入,与单队列排程和长跑不兼容;所有外部先例都只夹住不卡死 —— action_ball_curriculum.py:531-546 max_center_failures: int = 8;:4650-4661 center 分支 center_failures>=8 → progress.phase='stalled'(kind='stalled_at_center'),后续调度 fail-loud。外部对照:legged_gym torch.clip(level,0) 夹到 0;IsaacLab 同;ADR clamp 回 init_range;PBHC clip 到 [min,max];没有任何一个有终态卡死。 |
| low | 域宽度只朝'更宽'单调推进,若 manifest 的 maximum 设得过激,唯一的保护就是永久锁死(即用 issue #1 的缺陷去挡 issue #8) —— action_ball_sampling.py:494 width = initial + L*(maximum-initial),L 只经由 frontier_index 单调上升;唯一的收窄路径是全局 rho(action_ball_sampling.py:3803-3856 target_width=max(initial_width, rho*current_certified_width)),且 rho 只在所有臂 decided 之后才启用。样本 manifest configs/n1_contact_20260730_stable_v2/bh_loop_c.manifest.v3.775f74183e58.json 里 incoming_speed lower-std 0.15→2.0429 m/s(中心 3.40 m/s),spin_magnitude upper 5→40 rad/s,即 maximum 端相当激进。 |

### 13.4 建议 R1-R7(全部新 key、默认 byte-identical)

**R1(1):让 marginal 的锁定可逆:'lock/bound' 改成'休眠 + 到期重开',新增 manifest/CLI key `arm_reopen_after_epochs`(默认 0 = 关闭,现行行为 byte-identical)。到期时把 status 'decided' 恢复为 'probing' 并保留已认证的 frontier_index(重开只可能向上,不会缩窄已认证宽度,因此不会压制击球收入)。**
- 先例:Dextreme-ADR(adr_vec_task.py:760-917):每次调整后清空该 (param,bound) 队列并让它重新累积 256 样本,同一边界可被无限次重测,判错会自愈;legged_gym legged_robot.py:421-452 与 IsaacLab terrain_levels_vel 每次 reset 重判;PBHC legged_robot_base.py:866-891 每次 reset 重判。没有任何外部先例把某条轴永久钉死。DeepMind 乒乓按当前回球率反比配额,同样是持续重估。
- 落点:action_ball_curriculum.py `_apply_formal_evidence` marginal 分支(:4683-4700)与 `_reselect_arm`;_Progress 增加 arm_decided_epoch 字段(已在 state_dict/load_state_dict 序列化框架内);BallCurriculumConfig 增加 arm_reopen_after_epochs 并进 as_dict(仅非默认时写出,沿用 objective_inactive_arms 的写法以保 legacy 配置字节不变)。
- 风险:重开会消耗额外的 100-update 轮次,可能挤占 joint 阶段预算;缓解:只在 phase 已进入 steady 且当轮无待决臂时重开(空槽拉最前就绪项,符合单队列排程)。另需在 state_dict 版本号上做兼容处理,否则旧 checkpoint resume 会 fail-loud。

**R2(2):把判决样本花在刀刃上:让冻结评估窗口用一套独立于训练的采样混合(eval-only frontier 配额),把新带行数从 ~154 提到 ~400+;同时把 heldout_min_new_band 从'恰好等于 768×0.2'改成带余量的独立配额(如 target 384 / floor 256)。这不增加任何回合数,只是把已经在跑的 768–960 行重新分配。**
- 先例:Dextreme-ADR 用 worker_adr_boundary_fraction 专门把 60% 的 env 分配给边界采样(adr_vec_task.py:747-753,注意 yaml 注释与代码相反),即'评估分布 ≠ 训练分布'正是这个思路;DeepMind 乒乓按类反比配额同理。统计侧:n=154 时 expand 阈值 3.25%,n=384 约 4.6%,n=768 才 5.6% —— 单靠加样本不能补齐到 10%,必须与 R1 或 R7 同时上。
- 落点:action_ball_sampling.py SamplingMixture(:880-884)新增 eval_frontier_slots / eval_interior_slots(默认等于现行 1/3/1,byte-identical);action_ball_curriculum.py:79 HELDOUT_NEW_BAND_MIN 保留为 floor,新增 heldout_target_new_band。
- 风险:冻结窗口的域分布不再等于训练域分布,全域 blocker(admit/install/start/close/unsafe)的统计含义随之改变,必须在 decision 记录里显式标注'在评估混合下测得',否则后续复盘会误读;另外 canary 与 heldout 的混合必须完全一致,否则 :4602-4614 的字段比对会 fail-loud。

**R3(3):NB 不足判'作废重测',不判'太难':把 'new_band_safe_closed_below_gate' 从 quality_bad 里拆出来,走独立 kind='insufficient_new_band',不改 arm status、不消耗 frontier 决定,只重排一轮;配一个连续 N 次不足即 fail-loud 的计数(默认 3)以保留真护栏。**
- 先例:Dextreme-ADR:队列未满 256 时不做任何决定,只等(adr_vec_task.py:760-917);PHC im_amp.py:136-241 与 ProtoMotions mimic_evaluator.py:98-122 都是跑完整 sweep 才更新权重。没有任何先例把'样本不足'解释成'难度过高'。
- 落点:action_ball_curriculum.py:4640-4643 与 :4683;新增 _Progress.insufficient_new_band_streak,超阈值时 fail-loud(与 max_center_failures 同风格)。
- 风险:若新带长期填不满会变成静默空转、白烧节拍;连败 fail-loud 计数是必须的配套,不能省。此项与 R2 强相关:R2 落地后 NB 不足的概率大幅下降,但 R3 仍是正确的语义修复。

**R4(4):安全 blocker 分层:zero-tolerance 事件(撞台 / qdes / 实关节越限 / attribution 异常)继续全域一票否决,但作用改为'全局 hold + 报警,不推进任何臂';只有落在新带内的安全事件才参与'是否锁死当前被探臂'的判决。**
- 先例:HoST(arXiv:2502.08378)用绝对高度门 0.45/0.65 m 作为安全/阶段判据,与被退火的助力和 action rescaler 是两套独立机制;Dextreme-ADR 的 limits(外层安全夹)与 adr_objective_threshold 5/20(难度死区)同样是两套独立机制,安全夹从不永久停用某个参数。
- 落点:action_ball_curriculum.py `_metrics`(:4461-4520)返回值拆成 whole_domain_blockers 与 new_band_blockers;marginal 分支(:4683)只用后者决定 status,前者改为置 kind='domain_hold' 并保持 status 不变。
- 风险:会弱化'任何安全事件都必须让域停下'的直觉,必须保证 hold 语义真的会阻塞下一轮请求(复用 request_due 的 needs_reset 阻塞路径),否则会退化成静默继续、把真护栏砍掉——这与 lean governance 的边界要看清:砍的是误伤,不是护栏。

**R5(5):给全局 rho 退档加连败滞回(默认仍为 1,新 key rho_retreat_dwell),并把 center 的 'stalled' 从终态改为'退回上一 certified 并 hold + 报警'(新 key center_stall_action,默认 'stalled' 保持现状)。zero-tolerance 事件必须绕过滞回立即退档。**
- 先例:我们自己已退役的 task_first_curriculum.py:574,661-682 的 exit_dwell 连续窗口 rollback(EXP-TASK-FIRST-N-ACTION-20260727.md 判为 superseded/ablation-only,但滞回设计本身比现行成熟);PBHC 40/42 双阈值死区;legged_gym 夹到 0、ADR 夹回 init_range —— 没有任何外部先例存在需要人工解锁的终态。
- 落点:action_ball_curriculum.py:4712-4759 的 joint/steady 分支与 :4650-4661 的 center 分支;BallCurriculumConfig 新增两个默认保持现状的字段。
- 风险:滞回意味着一个真的坏域会多跑一个窗口(约 1280 回合)的不安全采样,所以 zero-tolerance 旁路是硬性配套;center_stall_action 若默认改掉会让'需要人工看一眼'的信号消失,建议默认不变、只在长跑 profile 里开。

**R6(6):承认并写死'N1 谱系 = 固定难度跑'的契约,把审查重心搬到 manifest 的 initial 宽度;晋级机器的首次端到端行使放在一个专门的短 N5 跑里验收(至少完成 1 次真实 marginal 决策 + 1 次 drain/reset + 1 次 resume 校验),而不是直接押在 20k–25k 的长跑上。**
- 先例:HITTER 完全没有课程、目标分布从第一天就均匀,拿到 92.3% 回球率 —— 固定难度是有先例背书的合法选择;mjlab 的 commands_vel 干脆不看性能、纯 step schedule(velocity_env_cfg.py:396-411),同样说明'不自适应'不等于差。反面:legged_gym / IsaacLab / unitree / beyondmimic 的课程状态 resume 后归零却无人发现,正是因为从没被端到端验收过。
- 落点:scripts/launch_action_ball_curriculum.py 的 stage-budget validator(:2597-2744)已有 canary 阶段合约,加一条'canary 阶段必须产出至少 1 条 kind∈{center_pass, expand_marginal, lock_marginal} 的决策记录'作为验收项;docs/operations/run_action_ball_curriculum_no_clobber.md 补一条人话行说明 N1 是固定域。
- 风险:多花一次短跑的机器时间;但比在 20k 长跑第三天才发现 drain/reset 死锁或 NB 门限误锁便宜得多。另需注意该短跑会真的改变域,不能复用到 N1 的对照序列里。

**R7(7):(仅在 R1 落地之后)放宽 too_easy 判据:从 UCB<0.075 改为'点估计 < target 且 LCB < target',把继续扩张的实际阈值从 3.25% 抬到接近 10% 的设计意图。**
- 先例:DeepMind 乒乓(arXiv:2408.03906)直接按当前各类回球率反比配额,是点估计驱动、不做置信区间保守化;PBHC 的 sigma←min(sigma,EMA) 同样是点估计驱动的单向收紧。保守化在这些先例里之所以不需要,是因为它们都可逆。
- 落点:action_ball_curriculum.py `_metrics`(:4525-4527 too_easy/too_hard 定义),新增 expand_criterion 枚举,默认 'ucb'(现行)。
- 风险:高:不可逆 + 乐观判据 = 有可能一步扩得过宽然后永久锁在过宽档,直接压制击球收入(违反'不得压制击球收入'的硬约束)。因此绝不可单独上线,必须在 R1(可逆)与 R5(滞回退档)之后;若 R1 被否决,则本条应一并否决,宁可欠扩张。


### 13.5 八家升降级设计对照

| 维度 | 我们 (action_ball) | legged_gym | IsaacLab + unitree_rl_lab | mjlab | Dextreme-ADR | PBHC | ProtoMotions / PHC | DeepMind-TT |
|---|---|---|---|---|---|---|---|---|
| **指标 metric** | 失败率 Wilson CI(marginal 阶段只用新带 NB_F/NB)+ 5 项全域质量 blocker + 4 项零容忍计数 | 地形:本回合走的距离;指令:刚 reset 的 env 的 tracking_lin_vel 平均回报 | 同 legged_gym(IsaacLab 逐字继承);unitree 指令轴同形状 | 地形:同上;指令:**不看任何性能**,只看 common_step_counter | successes 连续成功**计数**(非比率),按 (参数, 上/下界) 分别入队 | average_episode_length(EMA,窗口 10000 次 reset);sigma 轴用逐项误差 EMA | 每个 clip 的整库 eval 通过/失败 | 每个发球类别的当前回球率 |
| **阈值 threshold** | 目标失败率 0.10 ± 0.025,z=1.96;实测 n=154 时"继续扩"实际要求 F≤3.25%,3.9%–17.5% 一律永久锁 | 升:距离 > env_length/2;降:距离 < ‖cmd_xy‖·T·0.5;指令:回报 > 0.8×权重 | 与 legged_gym 同形(IsaacGymEnvs 的 anymal 版降级系数是 0.25) | 地形同上;指令为 3 个硬编码步数界(0 / 120000 / 240000) | 死区 [5, 20]:>20 扩(变难),<5 缩(变易),中间不动 | 死区 [40, 42] 步;degree 1e-5(罚项)/ 2.5e-7(软限三件套) | 成功 ×0.999^200≈0.819,失败 ÷0.819(≈×1.22);PHC hard 模式把未失败 clip 权重直接清零 | 无阈值,按回球率反比配额 |
| **判决窗口 window** | canary 256/256(仅可否决)+ 不相交 heldout 768/768,新带 floor 154;两窗必须同一 checkpoint | 单次 reset 的一批 env(无窗口) | 单次 reset 的一批 env | 单次 reset(地形)/ 无(指令) | 每个 (参数,界) 独立 deque,maxlen=256 | 10000 次 reset 的 EMA(罚项)/ alpha=1e-3 逐步 EMA(sigma) | 整库 eval sweep(每个 clip 跑到终止或 600 步) | 持续在线统计 |
| **节拍 cadence** | 每 ≥100 个 PPO update 一次(且 no-inflight、无待决 reset);由独立冻结策略 sidecar 异步执行 | 每次 reset(指令轴额外每 max_episode_length 步一次) | 每次 reset(数千 env → 稳态下几乎每步) | 每次 reset;指令实际只在 3 个步数界变 | 每个 episode 结束事件累积入队,队满即判 | 每次 reset(罚项)/ 每个 env step(sigma) | 每 200 epoch 一次 eval | 持续 |
| **作用域 per-env vs global** | 全局,按 (action_uid, profile_sha, mobility) 分键;所有并行 env 共享,且只在全局 drain+reset 后生效 | 地形:**per-env**;指令:全局 | 地形:per-env;unitree 指令范围:全局 | 地形:per-env;指令:全局 | 范围全局,但 per-env 的 worker_types 决定谁采边界(P(边界)=0.6) | 全局标量 | 全局的 per-clip 权重向量 | 全局的 per-class 配额 |
| **降级 demotion** | **32 条臂一律无降级、无重试,一次定终身**;唯一下降路径是全局 rho,单个坏窗口即退一档、无连败缓冲,rho=0 再失败进 stalled 终态 | 双向,夹到 0;指令轴只扩不缩 | 双向(与 legged_gym 同形);unitree 指令轴只扩不缩 | 地形双向;指令纯单调前进 | 双向且对称,夹回 init_range;**永不永久锁定**,队列重填后同一边界再测 | 双向(罚项与软限三件套方向相同、算术符号相反) | 连续再分配,失败 clip 权重上升;PHC hard 模式为全有全无 | 隐式:回球率上升 → 该类采样自然减少 |
| **状态 checkpoint** | **全场最好**:完整 state_dict/load_state_dict + curriculum_state_sha256 交叉校验 + resume 漂移即 fail-loud | 无(vanilla rsl_rl save/load 无 env 钩子),--resume 后 per-env 等级随机重置 | 无(同上) | 部分:自定义 runner 存 common_step_counter("以保留课程状态"),但不存 terrain_levels | 有一等公民配置开关 adr_load_from_checkpoint | 未确认(未追到存盘路径) | ProtoMotions:motion_weights 确认存盘;PHC:_sampling_prob / _termination_history 未见存盘;beyondmimic bin_failed_count 未见存盘 | n/a |
| **轴数 axes count** | **32**(16 个物理轴 × 上下/正负两侧;no_move + counter-rally 掩码后活跃 26),**一次只探 1 条,串行** | 2(地形 + lin_vel_x) | 2(地形 + lin_vel_x/y;ang_vel_cmd_levels 在所有出厂配置里都是死代码) | 2(地形 + 指令,后者非性能门控) | ~10 参数 × 2 界 ≈ 20 个边界,**并行异步推进** | 4+(sigma 约 15 个奖励项 + 罚项 scale + 软限三件套) | 1 个分类轴(clip 采样权重) | 1 个分类轴(发球类别) |
| **判决可逆性(关键)** | 单臂**不可逆**(status 'decided' 无任何恢复路径);rho 可逆但无滞回 | 每次 reset 重判,天然可逆 | 每次 reset 重判,天然可逆 | 地形可逆;指令不可逆但也不依赖性能 | 完全可逆,无限重测 | 完全可逆 | 完全可逆(每次 eval 重算) | 完全可逆 |

### 13.6 追加(08-01,Franco 两问):失败加权采样 R8 与多臂并行扩张 R9

**目的重述(Franco 定调)**:curriculum 的真目标是后期机器人直接对打,**有效护台面积**要涨。
据此新增一个显式 KPI:**certified 护台面积 = 目标失败带下已认证的 contact_offset_x 宽 ×
contact_offset_y 宽(× 来球方向锥立体角因子)**,每次认证后记账——把 32 臂的抽象进度折成
一个可读单标量,也直接指导 R9 的族配额。注意 `no_move` 模式硬禁 base_travel 组;真对打要开
mobility,该组回归的预算须提前计入。

**R8:已认证域内失败加权出题(训练侧,可独立先行)**
- **定位**:训练分布旋钮,**不碰认证纪律**——训练采样(活策略,允许失败加权)与认证窗口
  (冻结、固定申报混合,R2 已立此界)是两条互不污染的通道。
- **先例**(全部已核实):DeepMind 乒乓按发球类别以回球率**反比**配额(任务条件级,与此处
  完全同构);PHC PMCP soft(采样概率∝失败率);ProtoMotions success/failure_discount EMA;
  beyondmimic 的 10% uniform 兜底常数。**与 mid-swing airdrop cheat 无关**——重加权的是
  任务条件(出题),不是初始状态,不触 07-26 反 RSI 裁定。
- **机制**:已认证域离散成 cell(自然选择:复用各 arm 的档位带 × 关键轴粗格);per-cell 失败
  EMA 由**已有的 on_rollout_end report-only 台账**喂(零新遥测);birth sampler 在
  SamplingMixture 内按失败率加权重分配 slot。
- **护栏**:①uniform 兜底 ≥10%;②center stratum 保底配额不许饿死(中心题是击中收入现金牛,
  收入分层红线);③指标口径:难度混合非平稳后 raw hit rate 不可跨 update 直读,台账同时记
  per-cell 与 mix-standardized 两套,判平台期用后者;④balanced round-robin 动作采样器不动
  (动作级平衡 × 动作内题目加权,两层正交)。
- **落点**:action_ball_sampling.py SamplingMixture 增 `failure_weighted_slots`(默认 0=现状
  byte-identical);EMA 状态进 curriculum state_dict(序列化框架已有)。

**R9:多臂并行扩边界(治"太慢/方向单一")**
- **审计数字支持担忧**:26-28 臂串行 marginal ≈28 轮;并行 2-3 臂/窗 → ~10 轮;
  配 R1 可逆后早期误锁可自愈。
- **先例**:Dextreme ADR 本来就是多边界并行——每 (param,bound) 独立队列,60% env 分给边界
  采样,各边界独立攒满 256 样本独立出判,互不阻塞(clear_other_queues=False)。
- **形态 1(小改,先做)**:一个 heldout 窗口的 960 行分给 2-3 条候选臂(每臂 256-384 行,
  R2 的 eval-mix 重分配已把行数腾出);唯一 schema 改动 = `in_new_band: bool`(:2509,单布尔
  天然一窗一臂)升级为 `probed_arm` 行标签;归因靠"每 env 恰探一臂"的构造(ADR 同款),
  Wilson 判据每臂独立算。
- **形态 2(ADR 全款,视形态 1 瓶颈再定)**:每臂证据队列跨窗口连续累积,攒满各自 floor
  独立出判——判决节拍与窗口节拍解耦,慢臂不堵快臂;改动大(队列状态入 checkpoint)。
- **方向单一的对治**:arm 选择从"单候选"改**族配额轮转 + 失败信息加权**——按轴族
  (护台面积族:contact_offset x/y、方向锥、base_travel;速度族;旋转族;落点族)每轮保底
  轮转,护台面积族权重最高,防 incoming_speed 一族饿死方向锥/落点。
- **风险与顺序**:并行探臂 × 不可逆锁定 = 误锁速度同样 ×2-3,**R1(可逆)是 R9 的硬前置**;
  多臂同窗使全域 blocker 误伤面变大,R4(blocker 分层)应先行;多臂同时认证时 drain/reset
  是全局的,须确认一次 reset 可释放多张认证单(needs_reset 语义)。
- R8 与 R9 正交:R8 治"已认证域内练什么",R9 治"边界往哪扩、扩多快";两者共享 per-cell/
  per-band 失败统计基础设施,建议同一张 schema 设计(行标签 = cell id + probed_arm)。


---

## 十四、Isaac→MuJoCo 训练迁移尽调(08-01;应 Franco 换引擎之问)

**方法**:5 抽取(我方耦合面 / mjlab 能力 / Newton+厂商替代路线 / 生态现成支持 / 账本适配)
+ 5 对抗核查(31 修正/27 漏项)+ 1 终裁。全程本地(克隆+仓内文档),零 web。

### 14.1 终裁(原文入档)

值得迁，但不是现在整体搬——而是「现在开一条 CPU-only 勘探支线，在 N=1 判读到正式 N=5 之间裁引擎」。

人话结论：我们今天的处境是「训练在 PhysX、验收在 MuJoCo、部署在厂商 MuJoCo」，中间那道翻译层已经被我们自己的文档判定为形式上不合法。迁到 MuJoCo 能一次性拆掉这道翻译层；但它买不到吞吐，因为我们最大的时间开销是自己写的 Python reset 仪式，与引擎无关，搬家会原封不动跟着走。

支持迁移的三条硬证据：
(1) 关节摩擦是死结，不是可修的 bug。robots/agibot_a3.py 的 2026-07-10 审计注释写明 ImplicitActuatorCfg.friction 的数值是「从 MuJoCo frictionloss（恒定 N·m 库仑力矩）照抄进 PhysX 无量纲负载相关系数」的未标定遗留选择。training_contract.py:279 起把 JOINT_FRICTION_BACKEND='physx' 硬钉进 schema-3 fail-closed；G06 明说「非零 PhysX joint friction 没有 exact MuJoCo frictionloss 等价」；mujoco_eval_onnx.py 因此对任何非零摩擦系数必须 --allow-inexact-contract 才肯跑。这意味着路线 D（留 Isaac + 加强 MuJoCo 交叉验证）存在一个天花板：当前整条谱系的正式 MuJoCo 评估在形式上已经 fail-closed，「加强交叉验证」加强不过这堵墙，除非把关节摩擦清零（那是改 plant）或者换引擎。迁到 MuJoCo 后同一批数字变回原生正确单位，整套 fail-closed 机制（schema-3 摩擦精确性标志、zero_joint_friction override、judge.sh 的 31 个零系数检查）从「待修的 bug」变成「待退役的死代码」。
(2) 厂商对齐是 Franco 的最高优先级裁定，而厂商 instinct_mj 是 MuJoCo 栈、同 A3 底盘族（a3_ultra）。部署侧唯一终审 Gate3/Gate3B 跑的是厂商 C++ runner + 真 a3_pingpong.xml，它从来不依赖 Isaac。我们现在是整条链上唯一还在 PhysX 上的环节。
(3) 判官链已经在 MuJoCo 侧。scripts/mujoco_eval_onnx.py（8140 行）零 isaaclab import；judge.sh 只在 ONNX 导出那一段激活 Isaac venv（等 Kit boot lock、跑 play.py），评分段全是 mjeval venv。迁移后砍掉的是那段 shell 胶水，不是物理移植。

迁移明确不修的事（必须写在最前面）：
- 吞吐。design_audit_and_speedup_20260729.md §9 的双栏 diff 是决定性的：同硬件、同 Isaac，队友 yikang r2fqs 谱系 formal 6.383 s/update、warm 3.78 s，我们最好 25.4 s；每步 .item() 他 3 个、我们 35 个；终止项他 4 个、我们 9 个。§9.3 排名前四全在 reset 仪式（broker 典礼 / receipt resolve 链 / resample item 循环 / 正式档案），可回收诊断臂 4.5–11 s、正式臂 8–17 s，§9.5 落点 6–8 s/update。同一个引擎上别人快 4 倍，就证明这不是引擎的锅。迁到 mjlab，这 60–75% 的 Python 仪式一行不改地跟着搬过去。
- 唯一的例外、也是唯一能顺手买回的吞吐：32 个 filtered ContactSensor。§9.4 写明这 32 个 sensor 的存在理由就是「为保住一个在钉死的 Isaac Lab 上本来就坏的 filtered 语义」（force matrix 维度错，IsaacLab #1995/#4108），是每步最大单项嫌疑；yikang 直接删机制，改单一全身 contact_forces + 桌面 AABB。迁移让这 32 倍的理由自然消失——这是迁移能带来的真实但有限的加速。
- 不修课程/reward 设计问题、不修题库、不修单 clip 臂二等公民。这些都在那 65,771 行引擎无关代码里，搬过去还是原样。
- 不修 G06 的「未完成」项：正式逐 checkpoint 验收、W/Y 谱系补救、100 行 vendor adapter，换哪个引擎训练都还欠着。

耦合的真实规模（我独立复算，比 LOC 口径乐观）：整包 121,206 行，其中 38 个文件、55,435 行 import isaaclab（132 条 import 行），65,771 行零耦合。但按 LOC 算严重高估：真正的物理 API 调用点（root_physx_view / write_*_to_sim / .data.body_*_w / force_matrix_w / soft_joint_pos_limits / default_root_state 等）全包只有 96 处，且集中在 commands.py（19 处）、terminations.py（9 处）、isaac_lateral_perturbation.py（5 处）、hope_push_events.py（4 处）。hope_commands.py 19,147 行只有 5 条 isaac import 且全是只读 buffer 读取（无 write_*_to_sim、无 default_root_state）。所以「46% 代码要重写」是错的口径，实际是「几十个调用点 + 约 10 个必须继承基类的 cfg/action/scene 文件」。

一个必须点名的口径修正：ledger 把 physics/solver profile pins（physics aa5c9085…、solver f89587db…/146c4d6a…）算作 Isaac 沉没成本是错的。我打开 configs/n1_contact_20260729/action_ball_profile_pins.v1.b6489cea.json 确认这些 hash 的是球飞行虚拟物理常数（k_d、k_m、table_e_eff、drag/magnus）和纯 Python 求解器代码（continuous_questions.py、virtual_ball.py、racket_contact_geometry.py），零 Isaac/PhysX API；而且 hope_training/whole_body_tracking/scripts/audit_action_ball_cross_engine_physics.py 已经在拿同一份 profile 对 isaac_consumer 和 mujoco_consumer 做逐字节等价核对，profile 自己的 contact_geometry 已把 canonical frame 命名为 official_pingpang_red_Link_origin_MJCF_right_racket_site、把 Isaac 数字降级为 legacy_isaac_site_offset_wrist_m。MuJoCo 消费者已经存在。真正 Isaac 绑死的沉没成本只有 Hctrl 机械应力 receipt 链、identity→authority→bundle 的 USD 关节序权威、和 OpenGL/USD closure 那一层——而 ledger 自己已经把它们标成 PAUSED-ENGINE / DEFER-ENGINE 可弃。

### 14.2 五条路线对照

**(A) 全量 port 到 mjlab（推荐路线，条件是勘探支线先答完三个未知）**
- 成本:约 15–30 人周（4–7 人月，单个熟练工程师量级）。分项：① reward/curriculum/contract 内核（65,771 行引擎无关）——只重接调用点，1–2 人周；② commands.py + hope_commands.py 共 26,058 行——3–6 人周（大但机械：hope_commands 只读 buffer、无 reset 写，只有 commands.py 有 19 处 write_*_to_sim/default_root_state 需要重导出；mjlab EntityData 的 body_link_pos_w/quat_w/lin_vel_w/ang_vel_w/joint_pos/joint_vel 命名几乎逐字对应 ArticulationData）；③ hope_actions.py ClampedJointPositionAction + substep 安全账本（4,124 行）——1–2 人周（clamp 本身在 mjlab 已是 JointPositionActionCfg 的 clip 字段，是配置不是子类；真正要重推的是 decimation==4 假设下的 substep 记账，硬断言分布在 hope_env_cfg.py:782 一处和 hope_actions.py 约 866–885 / 958–979 两处）；④ 场景/资产：a3_pingpong.xml 作为 Entity attach + 从 table_frame.py/geometry.py 既有法定球台常数写 MJCF 球/台/网——2–4 人周，这是单项最大（mjlab 的 g1_constants.py 296 行 + g1.xml 308 行是现成模板/清单）；⑤ 接触与桌碰检测重表达为 regex primary/secondary（顺手采纳 yikang 单 sensor 设计）——1–2 人周；⑥ 球接触策略裁定（原生 solver vs 移植 physical_ball.py 的 code-driven override）+ 恢复系数改写到 solref/solimp——2–4 人周含标定；⑦ rsl_rl 2.3.1 → 5.4.0 + my_on_policy_runner.py（实测 7,015 行，非二手资料说的 2,100 行）的 exact-resume 内部手术——3–6 人周，最高风险项，且与 Warp determinism 阻塞耦合；⑧ DR 移植（mjlab dr.* 是我们现有 DR 原语的超集，含 encoder_bias/pseudo_inertia/5 分量 pair friction）——0.5–1 人周；⑨ judge/gate 重构（judge.sh 砍导出腿、G06 改锻）——0.5–1 人周；⑩ bundle/合同重钉——1–2 人周。
- 风险:HIGH，但风险是可枚举的、且集中在三处：(1) **Warp determinism 是当前唯一 blocking 项**——mjlab 自己的 docs/source/faq.rst:247-256 写明 MuJoCo Warp 尚不保证确定性，「即使设了 seed，训练也不会完全可复现」，上游 mujoco_warp#562 未解；我另在代码层坐实：src/mjlab/utils/random.py 的 seed_rng() 只调 random.seed/np.random.seed/torch.manual_seed，根本碰不到 Warp/GPU kernel RNG，且 mjlab 全树没有任何 CPU/non-Warp 回退（pyproject 硬钉 mujoco-warp>=3.10.0.3）。这直接打我们的 exact-resume verifier 和 no-clobber 课程续跑。(2) rsl_rl 2.3.1→5.4.0 跨度对一个 7,015 行的自定义 runner 是非机械迁移。(3) mjlab v1.5.3 虽标 Production/Stable、923 行 changelog、全树只有 2 处 TODO/FIXME，但仍是年轻库，多月项目钉住一个移动靶有 churn 风险。次级风险：MotionCommandCfg.motion_file 是单字符串（一个 command 实例一条 clip，无 batch 内多 clip 混合），我们正反手双 clip 需要自己写包装层——正好撞上仓内既有的「单 clip 臂是二等公民」老毛病。
- 对齐:最好。三条实证：(a) mjlab 的 tracking 任务 header 自述是 BeyondMimic 的 re-implementation，Based on HybridRobotics/whole_body_tracking commit f8e20c880d9c8ec7172a13d3a88a65e3a5a88448——那正是我们自己的上游谱系，等于官方给了我们一份带版本号的 port 范例；reward 权重/std（0.5/0.3、0.5/0.4、1.0/0.3、1.0/0.4、1.0/1.0、1.0/3.14、action_rate -1e-1、joint_limit -10.0）和 PPO 超参（clip 0.2、entropy 0.005、epochs 5、minibatches 4、lr 1e-3、gamma 0.99、lam 0.95、hidden [512,256,128]）逐字节一致搬过去了，我复核过。可量化的 port 足迹：同一个任务目录 beyondmimic 16–17 文件/1,363 行 → mjlab 15 文件/1,855 行（1.36 倍），7 个文件 1:1 重写（commands.py 377→608、rewards.py 82→135、observations.py 83→69、terminations.py 58→86、env_cfg 322→313），events.py 93 行整个被吸收进共享 dr/ 包。(b) manager API 几乎是同名的（RewardTermCfg/ObservationTermCfg/EventTermCfg/TerminationTermCfg/CommandTerm/ActionTerm/SceneEntityCfg 在 IsaacLab 侧本来就叫这些名字，RewTerm/ObsTerm 只是下游 import 别名——所以比二手资料说的「重命名」还要更机械），唯一结构变化是 manager 配置用 plain dict 而非嵌套 @configclass。(c) 场景用纯 mujoco.MjSpec.from_file + attach，EntityCfg.spec_fn 是任意 Callable[[], MjSpec]，没有 USD/Omniverse 管道挡路——直接 attach 厂商 a3_pingpong.xml。
- 裁定:推荐，但先跑 2–3 人周有界勘探支线，只答三个问题：① Warp determinism 对我们的 exact-resume 到底是硬阻塞还是可用「每 N update 落盘 + 接受 bitwise 漂移」绕过；② 我们的场景（机器人+台+网+球+球拍接触、4096 env）在 5090 上真实 steps/s；③ 球接触走原生 solver 还是移植 code-driven。三题答完再决定是否投 4–7 人月。

**(B) Isaac Lab Newton 后端切换（MJWarpSolverCfg）**
- 成本:表面 2–5 人周（换 solver_cfg + 逐机器人 shape-margin/solver 迭代调参，manager/cfg 层完全不动，USD 球台/网场景零改动）。**但有一个二手资料没点破的隐藏成本：我们钉的是 Isaac Lab 2.1**（training_contract.py:279 明写「Isaac Lab 2.1 把 friction 传给 PhysX 作为无量纲…」，rsl-rl-lib 2.3.1），**而 isaaclab_newton 只存在于 release/3.0.0-beta2**。所以路线 B 实际是「IsaacLab 2.1 → 3.0-beta2 大版本迁移 + 后端切换」，真实量级 6–12 人周，且 2.1→3.0 那段本身没有被任何现有尽调量化过。
- 风险:MEDIUM-HIGH。IsaacLab 3.0 整条线自标 Beta 2；isaaclab_newton 子包仍是 0.x（0.13.6、53 个 changelog 版本、3 个标 Breaking）；vendored newton[sim] 钉 ==1.2.1（6 天前还是 1.2.1rc2）。全包 51 处 NotImplementedError / 10 个文件，主要在 fixed/spatial tendon（articulation.py 约 25 处 + articulation_data.py 10 个裸 stub）和 gravity_compensation_forces（上游 newton#2497/#2529/#2625，测试里有 strict xfail）——我核对过我们仓内零 tendon / 零 gravity_compensation 用法，所以目前不挡路。真正要点名的两条：(a) 常被引用的「旗舰 G1 已在双后端验证」**证据不足**——源码只证明 newton_mjwarp preset 存在且带 G1 专调常数（flat_env_cfg.py:19-29，njmax=95/nconmax=10），没有任何 CI/benchmark/changelog 记录真跑过 G1+Newton 训练或平价结果（isaaclab_newton CHANGELOG 零个 G1 命中）；框架自己的 kamino-solver.rst 明说「任务必须已兼容 Newton 后端，若 physics=newton_mjwarp 构建失败请先修资产或任务配置」「每个 sensor 与 renderer 组合仍需各自验证」。(b) contact_sensor_data.py:253 有 force_matrix_w_history = None # TODO——filtered 力矩阵历史在 Newton 上永久未实现（只有 net_forces_w 有历史缓冲）；我们今天恰好没撞上（tracking_env_cfg.py:71-73 的 contact_forces 用 history+air_time 无 filter，hope_env_cfg.py:620-637 的桌面对 sensor 用 filter 无 history），但这是未来合并 sensor 时的雷。另外我们的 TABLE_HIT_MARGIN_M / TABLE_HIT_FORCE_THRESHOLD_N 注释明确是按 PhysX 接触解算余量推的，换 solver 要重新标定。
- 对齐:中偏低。它确实是真 MuJoCo 物理——MJWarpSolverCfg 的 solver_type='mujoco_warp'，字段直接来自 MuJoCo（njmax/nconmax/cone='pyramidal'/impratio/ccd_iterations/tolerance），且 use_mujoco_cpu 能退到纯 MuJoCo CPU（我另 clone 了 newton-physics/newton 确认 SolverMuJoCo.__init__:3315 真声明并广泛使用该字段，因为 isaaclab_newton 自己只做泛型 kwargs 过滤、按名字读不到它）。所以 frictionloss 语义鸿沟会消解，USD 球台/网场景零成本继承（这是相对 A 最大的省钱点，A 要从零写 MJCF 球台）。**而且它独家解一个 A 解不了的问题：use_mujoco_cpu 给了一条确定性逃生通道，mjlab 树里根本没有这种东西。** 但它不把我们带上厂商的栈，不给 mjlab 生态，继续绑 NVIDIA Kit/USD 工具链——而厂商不用这套。对 Franco「厂商对齐最高优先」的裁定，B 是打折的答案。
- 裁定:不作为主线，但作为 A 的对照与保险。具体动作：在勘探支线里花 2–3 天单独测一件事——use_mujoco_cpu 路径下 exact-resume 是否真的 bit-exact。如果测出「mjlab Warp 不确定性确实卡死我们的课程续跑，而 Newton CPU 路径可以」，B 的排名立刻上升到与 A 并列。

**(C) 采用厂商 instinct_mj（access 未确认）**
- 成本:未知。整条路线的成本无法估——我们手上零源码。dr_reward_external_diligence_20260731.md 自己写着「无法克隆核验，以摘要为准」，全仓 + scratchpad 搜 instinct 零命中。任务提示里点名的那几个面包屑（src/instinct_mj/tasks/…、envs/mdp/events/randomization.py、BuiltinPdActuatorCfg、NoisyGroupedRayCasterCameraCfg）我全仓 grep 过，**在本仓任何 .py/.md/.json/.yaml/.txt 里零命中**——这几个路径需要向提供方回溯来源，不能当已知事实用。
- 风险:XL，且是 access 风险不是技术风险。整条路线阻塞在一个既未确认也尚未正式提出的厂商配合上。我们真正知道的只有二手摘要：Instinct-Parkour-Target-Amp-A3-v0，MuJoCo 栈、AMP 判别器风格奖励、同 A3 族 a3_ultra（29dof 无头）；DR 轴为启动期（非每 reset）PD 随机化、Kp/Kd 非对称 (0.8,1.2)/(0.7,1.3)、每 episode 执行器延迟 [0,2] 控制步、摩擦 (0.2,1.8)/(0.2,1.5)、末端（躯干/踝/腕）质量 ±20% + pseudo-inertia、全身 CoM ±0.02、push (vx/vy ±0.25, vz ±0.1, r/p ±0.26, yaw ±0.39)、obs noise 逐通道手调 + history=8；reward 含轻量 action_rate_l2 (-1e-3)、dof_pos_limits -2.0、torque_limits(>90%) -0.01、全身 angular_momentum -1e-4、按 substep 计数的 self_collision、pelvis(-3.0)/torso(-0.6) 分裂姿态罚、以及 parkour 专用不可迁移的 freeze_upper_body。
- 对齐:名义上最好（就是厂商自己的栈、同底盘族），实际不可验证。注意这条路线与 A/B 不互斥：即使拿不到代码，摘要里的 DR/reward 数值对齐已经在做，那部分价值已经吃到了。
- 裁定:不作为工程路线排产，作为一次**信息请求**排产。要问智元的确切问题，按重要性排序：(1) instinct_mj 代码库是否可共享？什么条款/NDA？(2) 物理后端到底是原生 MuJoCo（CPU 参考）还是 MuJoCo-Warp / MJX（GPU 批量）？摘要只描述了 DR/reward 语义，从没说执行基底。(3) 他们训练实际达到的并行 env 数与 steps/s 是多少？在什么卡上？(4) 框架是否镜像 Isaac Lab 的 TermCfg/Manager 组合模式？（这决定我们 19k 行 RacketTargetCommand 是小改还是重写。）(5) a3_ultra 的资产/URDF/MJCF 与执行器配置授权条款，我们能否合法复用？(6) 他们在 MuJoCo 里怎么处理球/台/网接触与恢复系数——有没有已标定的 solref/solimp？(7) 他们是否遇到并解决过 Warp 确定性/exact-resume 问题？(8) 启动期（非每 reset）PD 随机化是有意的设计裁定还是实现约束？

**(D) 留在 Isaac + 加强 MuJoCo 交叉验证（现状）**
- 成本:增量 ≈ 0，但有持续税。
- 风险:看着最低，实际有一个硬天花板和一个未解正确性 bug。天花板：非零 PhysX joint friction 让整条当前谱系的**正式** MuJoCo 评估 fail-closed（BankExam 拒收，mujoco_eval_onnx.py 要 --allow-inexact-contract），所以「加强交叉验证」在形式证据上加强不过这道墙。未解 bug：IsaacLab GPU contact-filter 对静态碰撞体的 filtered pair 静默返回 0/NaN（#1995/#4108），4+ 个桌面 collider × 4096 env 在警告名单上，robot_hit_table 目前恒读 0.0000 **尚未被证明是真零而不是坏 sensor**；并且我们为保住这个坏语义把 sensor 乘了 32 倍，成为每步最大吞吐嫌疑。此外 G06 停在 Partial，且这是在推翻一个 2026-07-12 已经有因作出的裁定（Isaac 指标高而 held-out MuJoCo 击球/回球退化，所以停止优先扩 Isaac-only sweep）。
- 对齐:最差。直接违反 Franco 的厂商对齐最高优先裁定。
- 裁定:作为**过渡期**默认而非终局：N=1 诊断长训继续在 Isaac 上跑完（理由见 phase_recommendation），但**冻结 Isaac 专属开发队列**（ledger 已经这么写了：不再为即将迁移的 Isaac/PhysX receipt 链新增 feature）。生态校准很能说明 D 的定位：unitree_rl_gym 的 deploy_mujoco.py（130 行）、booster_gym 的 play_mujoco.py（130 行）、humanoid-gym 的 sim2sim.py（直接复用 IsaacGym 训练 cfg 类 XBotLCfg 构建 obs）、PBHC 的 deploy/mujoco.py（592 行，单 env + GLFW 键盘遥控，按键其实是 K/L/;/'/,/./ 一族不是 WASD）——**清一色单 env、实时、只做验证**，全都在 IsaacGym/IsaacLab 训练、在 MuJoCo 验收。这就是路线 D，是行业常态且很便宜。更强的反证：HumanoidVerse 家族（ASAP/FALCON/PBHC 三个仓，注意 base HumanoidVerse 其实没有）都带一份 config/simulator/mujoco.yaml 指向 humanoidverse.simulator.mujoco.mujoco.MuJoCo，**但四个 clone 里都不存在这个实现文件**——「用 simulator 抽象做向量化 MuJoCo 训练」在那个家族里是纸面脚手架。所以 D 技术上完全站得住，反对它的理由不是可行性，而是厂商对齐裁定 + 那道 fail-closed 天花板。

**(E) mujoco_playground —— 第五条路线，建议直接否决**
- 成本:若强行走，> 路线 A（估 25–40 人周）且交付更少。
- 风险:高且无补偿。
- 对齐:物理层对齐（MuJoCo/MJX），API 层严重不对齐。
- 裁定:**否决，理由是 API 距离，有证据**：它用直接 env 类而非 manager 组合——G1Env(mjx_env.MjxEnv) 在 locomotion/g1/base.py，Joystick(G1Env) 在 joystick.py（831 行）把 reward 项写成硬编码私有方法（_reward_tracking_lin_vel / _reward_alive / _reward_feet_air_time / _reward_feet_phase）在 step() 里手工相加，**没有 RewardTermCfg/ObservationTermCfg 注册表**；DR（locomotion/g1/randomize.py）是单个 jax.vmap 函数配硬编码魔数下标（TORSO_BODY_ID=16、dof_armature[6:]），而不是 mjlab 那种按名字/regex 的可复用 dr 库。后果很具体：我们整个 manager 形状的 cfg 层（RewTerm/ObsTerm/EventTerm/DoneTerm/CommandTerm/SceneEntityCfg + 19k 行 RacketTargetCommand）在那边**没有落点**，等于要先自己造一层 manager 框架再谈移植。而且 G1 只有 joystick.py（速度指令跟踪），**没有 BeyondMimic 式全身动作跟踪任务**，连范例都没有。唯一亮点是 RL 集成确实双通（learning/train_jax_ppo.py 走 brax，learning/train_rsl_rl.py 通过 wrapper_torch 的 JAX↔torch 桥 + warp 接 rsl_rl.runners.OnPolicyRunner），但那救不了 API 距离。在所有 MuJoCo 路线里排最后。

### 14.3 阶段插入点(账本适配)

插入点：选项 (b) —— **N=1 判读之后、正式 N=5 之前裁引擎**；同时**现在**就并行开一条 CPU-only 勘探支线。不要按 ledger 里 N1-TONIGHT-3LANE 那句「短迁移则三条 20000-iteration 直接发 MuJoCo」执行，因为迁移不短。

具体三步，且与单一队列和在飞的 N=1 诊断相容：

第一步（今天，队列外）：开 mjlab 勘探支线，CPU-only，2–3 人周有界，只答三题。这条支线**结构上不可能抢队列**：它不碰 GPU 生命周期锁（/tmp/hope_lean_queue_gpu{0,1}.lock），不走 nvidia-smi UUID/占用准入，所以不与 lane C 争那个即将空出的槽位。这也正好对上 G06 2026-07-12 的自有裁定「先做原生 CPU MuJoCo、测完 A3 负载再选加速后端」，以及现有 codex/mujoco-training-preflight 分支的实际范围（single-env 核心正确性，尚有 4 个红队 P1 缺口保持 NO-MERGE，与排程无关）。三题是：① Warp determinism 对我们 exact-resume 是硬阻塞还是可绕（顺带测 Newton use_mujoco_cpu 作为对照）；② 我们真实场景在 5090 上的 steps/s；③ 球接触走原生 solver 还是移植 code-driven physical_ball.py。**一旦需要 GPU 槽（加速后端或真 N-env 探针），它必须回到同一条就绪队列排，不给侧门预留。**

第二步（今天，队列内）：**解冻 N1-TONIGHT-3LANE，三条 lane 照发 Isaac，不等迁移裁定。** 三条理由：
(a) 时长完全错配。N=1 长训是 20000 iteration：按当前 23.5 s/update 每 lane 约 5.4 天，两个槽跑三条约 11 天（修完 §9.1+§9.2 到 8 s/update 则每 lane 1.9 天）。而路线 A 的 port 是 4–7 人月。等迁移完再发 N=1，等于把 N=1 推迟一个季度换一个引擎标签。
(b) ledger 自己已经把这三条 lane 的科学配方标为**引擎无关**（A=bh_loop_c static，B=bh_block static，C=bh_loop_c monotonic adaptive-sigma，从 0.20/1.0/0.52 锁步收到 0.075/0.5/0.262，共同 coarse position 0.30，禁恢复 action_one_hot）。N=1 要看的东西——std/LR 走向、Reward hacking、课程锁死、strike/return、策略塌缩——全部跨引擎可迁移。**在哪个引擎上先看到这些病理，比在哪个引擎上看到它们更重要。**
(c) 无论如何都要重物化。Franco 关于智元新 A3 setting 的裁定（waist-yaw Kp=80、waist-pitch effort=115、全部 wrist roll/pitch/yaw Kp/Kd/effort/armature=30/2/24/0.004968、按 0.25×effort/Kp 重算 action scale）已经作废当前 C1 plant，整条 identity→authority→candidate/hold→bundle→A/B/C 钉都要重跑一遍——**这是厂商 nominal bytes 变了驱动的，不是引擎驱动的**。既然这次重建反正要做，它就是给 plant/runtime-identity 层重新定值的便宜窗口：那份「正确数字是多少」的推导做一次就好，不会因为后面换引擎再做第二次。但这个折扣**不延伸到** physics/solver/球接触层（N5-PHYSICS 的 2026-07-30 OptiTrack 重钉是另一个独立事件，且恢复系数无论何时排产都要逐引擎标定）。
硬约束：这一轮 Isaac 发车必须 feature-frozen——只收口 launcher 的 Hydra append 错（要 `+task.table_contact_attribution_diagnostic=true`，不是 `task.…`）和智元 nominal/action-scale 两个确定性修正，不再给即将迁移的 Isaac/PhysX receipt 链加 feature（ledger 的 N1-DIAG-PROBE 已经这么写了，PAUSED-ENGINE）。

第三步（N=1 判读时）：在勘探支线三题的答案 + N=1 学习结论一起摆上桌时裁引擎，然后**正式 N=5 只在选定引擎上物化一次信任集**。

为什么是 (b) 不是别的：
- (a)「现在就整体切」不成立：port 是季度级，会把 N=1 推迟一个季度，且勘探支线的三个未知（尤其 Warp determinism）还没答，属于未定标就发车。但 (a) 的**勘探支线部分**成立且已被采纳为第一步——ledger 自己的 N1-TONIGHT-3LANE 触发条件就写着「CC 直接 MuJoCo 训练尽调回来即裁发车引擎」，本来就是并行设计。
- (b) 最便宜的结构性理由：**正式 N=5 的信任集今天是空的**——motion promotion 证书、frozen evaluator V4 receipt、GPU1 sidecar code/launch receipt、drain/reset runtime receipt 全部不存在。在 N=5 之前切引擎，**零份正式 receipt 被搁浅**。这是整个排程判断里最硬的一条。
- (c)「N=5 之后」是账本机制上最贵的点：那时换引擎要把整套正式信任集在新引擎上重新物化一遍（motion promotion 证书、带不可选择性停止的 100/320/960 样本窗口的 frozen evaluator V4 receipt、双 GPU 双锁 supervisor、exact-resume verifier、签名 prelaunch safety attestation，外加绑定的 runtime_inventory / ground_plant_contract_sha256 / effective_reward_recipe_sha256 / ppo_recipe_sha256 配方）——一次完整的第二轮正式发射周期。直接违反精简治理（不重复重仪式）。
- (d)「永不迁、只加强交叉验证」是在推翻一个 2026-07-12 有因作出的裁定，而且现在还多撞上 Franco 的厂商对齐最高优先裁定和 frictionloss fail-closed 天花板。

对 N1-TONIGHT-3LANE / N1-LONG-GATE 两行状态的建议：把 `WAIT-MUJOCO-DECISION` 改成 `READY`（发 Isaac），另开一行 `MUJOCO-SPIKE`（CPU-only、队列外、三题验收），并把引擎裁定挂到 N=1 判读→正式 N=5 的边界上，而不是挂在今晚发车上。

### 14.4 前置清单

1. 1. 【勘探，队列外，先于一切承诺】Warp determinism / exact-resume 裁决。读 mjlab docs/source/faq.rst:247-256 与 src/mjlab/utils/random.py 的 seed_rng（只播 host RNG，碰不到 Warp kernel），实测：同 seed 两次跑是否 bit-exact；若否，我们的 no-clobber 课程续跑与 exact-resume verifier 能否降级为「每 N update 落盘 + 显式接受 bitwise 漂移」而不破坏 §8 的 normalizer roundtrip 要求。同一支线里对照测 isaaclab_newton 的 use_mujoco_cpu 路径（mjlab 全树无 CPU 回退，Newton 有——这是唯一的路线级 tiebreaker）。**这一项不通过则 A 不可承诺。**
2. 2. 【勘探，队列外】吞吐实测，不接受自报数据。mjlab faq 只说「与 Isaac Lab 持平或更快」，本地无任何 FPS/steps-per-second 基准表。必须用我们自己的场景（A3 + 台 + 网 + 球 + 球拍接触）在 5090 上测 4096 env 的 steps/s，并**分开报告 solver 时间与我们 Python reset 仪式时间**——否则会把 §9 已证明的引擎无关开销误记到引擎账上。基准线是同硬件同 Isaac 上 yikang 的 formal 6.383 s/update。
3. 3. 【勘探，队列外】球接触策略裁定。physical_ball.py（2,482 行）今天是「真 PhysX 刚体但 PhysX 从不解算它的接触」——台面弹跳和球拍冲量都是代码驱动 override（predict_table_contact / virtual_ball.predict_paddle_contact 经 write_root_velocity_to_sim）。要裁定：MuJoCo 原生 solver 直接解算球-台-拍接触（可删约 2,500 行 override，但把「代码决定弹跳」变成「solver 决定弹跳」，改变 reward shaping 的确定性，需要全新验证），还是原样移植 code-driven。注意 virtual_ball.py（解析 Tier-1 模型）才是真正给 LIVE strike/landing reward 打分的东西，且它零 isaaclab import、原样可搬——这个裁定只影响 physical_ball 这个「真相仪器」。
4. 4. 【资产】训练级 MJCF 场景。厂商 a3_pingpong.xml（490 行）有完整 A3 运动树 + 明确球拍几何（right_racket_collision mesh、right_racket_handle_collision capsule、right_racket site 位于 0.21021/0.032078/0.032036，与 agibot_a3.py 的 A3_MOUNT_OFFSET 逐字相等）+ 干净的 <contact><exclude> 自碰撞抑制表，**但全文零 ball、零 table、零 net**（我方 grep 与 G06 原文一致）。要做：从 table_frame.py / table_tennis/geometry.py 既有法定球台常数（纯解析、零 isaaclab）写出台/网/球 geom，attach 成 mjlab Entity。附带红利：MJCF 那套干净凸包碰撞设置正是 agibot_a3.py 注释里说「合并腕部碰撞网格会在 sim 启动时腐蚀 PhysX、所以 enabled_self_collisions=False」的解药，迁移后自碰撞可以重新打开。同时 scripts/prepare_a3_isaac_asset.py（把 vendor URDF 的 package:// 改写成 Isaac 可加载相对路径）整个作废可删。
5. 5. 【执行器语义 parity】一次性把 31 关节 scale→action term→lag0/2→qdes 联合 golden 在新后端立起来（PORTABLE-GOLDEN-GATE 已列为唯一剩余缺口 a）。要点：mjlab 的 BuiltinPdActuator 走原生 <position>+<velocity> 配对（隐式积分器下刚性增益数值稳定），IdealPdActuator 走 Python 显式 PD 接 <motor>（最接近 PhysX 显式驱动）——**必须显式裁定选哪个**，因为 G06 记录过显式-vs-隐式 PD 这一项就造成过击球速度误差 0.61→0.31 m/s、速度达成率 0.35→0.88 的 sim-to-sim 背离。delay_min_lag/max_lag/hold_prob/update_period/per_env_phase 在 mjlab 是共享 ActuatorCfg 基类字段（比我们现有需求还富），[0,2] episode-fixed delay 直接可表达。同时确认 action_acc_weight=0.0 / action_rate_clamped=-0.2 的 vendor profile 显式钉在两引擎一致（ledger 08-01 要求）。
6. 6. 【接触/桌碰 parity】把 table hit / undesired contact 重表达为 mjlab ContactSensor 的 regex primary/secondary 模式。注意三点：(a) reduce 模式要选对——mjlab 出厂的三个 self_collision_cfg（tracking/config/g1/env_cfgs.py:24-31 与 velocity 下的 g1/go1）全用 reduce="none"，netforce 是另一个 sensor（feet_ground_cfg）在用，别抄错；(b) secondary 可展开多元素（secondary_policy 'first'/'any'/'error'），球拍/球/台三方接触需要 secondary_policy='any' 加多个 sensor 实例叠层；(c) **借这次把 32 个 filtered ContactSensor 收敛掉**，改 yikang 的单一全身 contact_forces + 桌面 AABB 判别——这是 §9.4 点名的每步最大单项吞吐嫌疑，且它存在的唯一理由（保住 IsaacLab 那个坏掉的 filtered 语义）迁移后消失。同时**必须重新验证 robot_hit_table 那个恒定 0.0000 到底是真零还是坏 sensor**——contact-filter bug 的根因消失了，但那个未回答的问题跟着迁过来。
7. 7. 【课程 exact-resume on new RNG】在新后端上跑通「真实 runner save → 第二 runner load，194/318 normalizer tensor 全等且 count 不回退」（PORTABLE-GOLDEN-GATE 缺口 c，且 08-01 的 Pod smoke 已证实 rsl_rl checkpoint 确实含 obs_norm_state_dict / privileged_obs_norm_state_dict 及各自 _mean/_std/_var/count）。这一步与第 1 项耦合：如果 Warp 不确定性成立，先在这里定义降级后的 resume 语义，再谈课程 sampler resume 与 adaptive-normal resume fixture。另外要正视 rsl-rl-lib 2.3.1 → 5.4.0 的跨度落在我们 7,015 行的 my_on_policy_runner.py 上（RNN 在 2.3.1 已有，只是以 RslRlPpoActorCriticRecurrentCfg 子类形式；真正新增的是 cnn_cfg 与 distribution_cfg）。
8. 8. 【判官/门重构 G06→?】把 G06 从「Isaac 产出 → MuJoCo 核对」改锻成「训练 MuJoCo profile → vendor_gate3_v1 平价门」。要点：(a) 训练腿改后，judge.sh 砍掉 Isaac venv 激活 + Kit boot lock 等待 + play.py 导出腿，评分腿（mjeval venv + mujoco_eval_onnx.py）原样不动；(b) G06 自己已定义的两个**非等价** profile 必须继续区分——isaac_bank_parity_v1（复现当前 schema-3 BankExam/Isaac profile）与 vendor_gate3_v1（1 ms vendor plant、逐步显式 PD、硬关节限、neck override、冻结 runtime flags），即使 Isaac 退出也不能合并；(c) 新后端**不得**从 evaluator import 共享的 observation/action/reward 实现（G06 明文：共享错误会造成 common-mode false green）；(d) Gate3/Gate3B（厂商 C++ runner + 真 a3_pingpong.xml）不动，仍是唯一晋级权威。
9. 9. 【bundle 重钉】按新引擎重物化合同层，但**只重钉真正引擎绑定的部分**。要退役/改写：schema-3 的 JOINT_FRICTION_BACKEND='physx' / JOINT_FRICTION_SEMANTICS='load_dependent_spatial_force_coefficient' / JOINT_FRICTION_UNITS='dimensionless' 三元组（training_contract.py:279-338，且 configs/a3_dynamic_ready_20260730/bh_block.dynamic_ready.v1.json:2035 确实带 joint_friction_backend='physx'、training_contract.py:4819-4820 确实强制它必须是 physx）、zero_joint_friction override、judge.sh 的 31 零系数检查。**不需要**重钉：physics/solver profile pins——它们 hash 的是引擎无关的球飞行常数与纯 Python 求解器代码，且 audit_action_ball_cross_engine_physics.py 已在做 isaac_consumer / mujoco_consumer 逐字节等价核对。另注意 training_contract.py:1523 有一条真实的运行时 `from isaaclab.envs.mdp import generated_commands`（在 _canonical_actor_leg_ref_mask_callables 里做 callable 身份比对，line 1535/1982），**与该文件自己 line 3 的「本模块刻意不含 Isaac/Torch/Hydra/ONNX import」docstring 矛盾**——迁移时必须重定向或重实现这个身份检查，别被 docstring 骗过去。
10. 10. 【多 clip 包装层】mjlab 的 MotionCommandCfg.motion_file 是单字符串，一个 command 实例一条 clip，batch 内无原生多 clip 混合。我们正反手双 clip 假设需要自建包装层——正好一并处理仓内既有的「单 clip 臂是二等公民」问题（三处单臂无法自述/被硬编码成正手）。
11. 11. 【恢复系数重标定】MuJoCo 没有 PhysX 式标量恢复系数（CoR），弹跳完全由 solref/solimp 约束柔度决定，且 mjlab 的 dr.* 里**没有** solref/solimp/restitution 随机化函数。2026-07-30 OptiTrack 拟合出的 table restitution 0.9215 / racket 0.646（连同 k_d=0.1253、k_m=0.00404）必须在 solref/solimp 空间重新推导，不能直接搬数值；若训练期要随机化弹跳刚度，得按 dr.* 现有模式自己写一个。ledger 明确这一层**不享受**今天智元重钉的折扣。

### 14.5 溶解的问题 vs 新增的风险

【消解的问题】

1. **frictionloss 阻塞（最大收益）**。同一批数字从「照抄进 PhysX 的未标定无量纲系数」变回原生正确单位的 MuJoCo frictionloss。连带退役：schema-3 的三个摩擦语义标志、zero_joint_friction override、judge.sh 的 31 零系数合同检查、mujoco_eval_onnx.py 的 --allow-inexact-contract 逃生口。这些从「待修的 bug」变成「待删的死代码」——注意区别：不是修好了，是问题的前提没了。

2. **contact-filter 正确性 bug（IsaacLab #1995/#4108）**。GPU broadphase 对静态碰撞体的 filtered pair 静默返回 0/NaN，这个根因随 IsaacLab 一起走。**但要点名一个跟着迁过来的未答问题**：robot_hit_table 目前恒读 0.0000，从未被证明是真零而不是坏 sensor——迁移消灭了 bug，没有回答那个读数。必须在新引擎上重新验证。附带真收益：那 32 个 filtered ContactSensor 的存在理由（保住坏语义）一并消失，可以收敛成 yikang 式单 sensor + AABB，这是 §9.4 点名的每步最大单项吞吐嫌疑。注意 mjlab 的 ContactSensor（671 行）是原生 regex 匹配 primary/secondary + reduce 模式（none/mindist/maxforce/netforce），架构上根本不同于 IsaacLab 的 GPU broadphase filtered-pair 设计——所以是「这类 bug 换了形状」而不是「所有接触 bug 都没了」。

3. **G06 gap**。「Isaac 产出 → MuJoCo 核对」这条腿整个消失。但**不塌成零**：替代它的是「训练 MuJoCo profile → vendor_gate3_v1」的跨配置平价门（仍是真门，只是在一个引擎内部而非跨引擎），加上从来不依赖 Isaac 的终审 Gate3/Gate3B。Isaac 降级为可选诊断（过渡期对现有 checkpoint 做交叉核对还有用）。判官链本体（mujoco_eval_onnx.py 8140 行、bank_exam_schedule.py 的 K100 不可变卷、schema-v3 plant 合同的 float32 网格精确比对）零改动继续用。**G06 当前那些「未完成」项（正式逐 checkpoint 验收、W/Y 谱系补救、100 行 vendor adapter）不因换引擎而消失**——所以这是改锻不是免单。

4. **厂商资产 parity**。a3_pingpong.xml 从「部署侧参照物」变成「训练资产本体」，机器人+球拍几何与真机逐字一致（right_racket site 与 A3_MOUNT_OFFSET 相等）。scripts/prepare_a3_isaac_asset.py 整个作废可删。MJCF 自带的干净凸包碰撞 + <contact><exclude> 表正是 agibot_a3.py 注释里说 PhysX 会被腐蚀所以关掉 enabled_self_collisions 的解药——自碰撞可以重新打开。

5. **可能消解**：physical_ball.py 那约 2,500 行代码驱动接触 override（存在的唯一理由是「PhysX 从不解算球的接触」，还额外绕了 Isaac Lab 2.1 的 body-frame-only 外力约定和 reward_manager 先于 command_manager.compute 的 manager 执行序 quirk）。若原生 solver 能直接解算球-台-拍接触，这些全部消失。**但这是设计裁定不是移植**，见新风险第 1 条。

6. **口径澄清**：physics/solver profile pins 并不是 Isaac 沉没成本（它们 hash 的是引擎无关的球飞行常数与纯 Python 求解器代码，且 audit_action_ball_cross_engine_physics.py 已有 mujoco_consumer 在做等价核对）；ballfit 的双 YAML（configs/ball_physics_*.yaml + agi/A3_MuJoCo_Sim 镜像副本）也**不合并**——它俩服务两个永久独立的消费者（快速解析 torch/Triton reward 模型 vs 实时 C++ vendor sim），双份是刻意设计，与引擎选择正交。

【新出现的风险】

1. **恢复系数/弹跳建模（确定会痛）**。MuJoCo 没有标量 CoR，弹跳全由 solref/solimp 约束柔度决定，且 mjlab 的 dr.* 里没有任何 solref/solimp/restitution 随机化函数。table 0.9215 / racket 0.646 必须在 solref/solimp 空间重推，不能搬数值。更深一层：如果同时把球接触从「代码决定弹跳」改成「solver 决定弹跳」，reward shaping 的确定性性质就变了，需要全新验证——这不是移植工作量，是重新做一次物理标定 + 重新建立信任。ledger 明确这一层不享受今天智元重钉的折扣。

2. **Warp 确定性 vs exact-resume（当前唯一 blocking 项）**。mjlab faq.rst:247-256 自述 MuJoCo Warp 尚不保证确定性、「即使设 seed 训练也不会完全可复现」，上游 mujoco_warp#562 未解。代码层坐实：seed_rng() 只播 host RNG，够不着 Warp kernel。**且 mjlab 全树没有 CPU/non-Warp 回退**（pyproject 硬钉 mujoco-warp>=3.10.0.3）——这与 Newton 有 use_mujoco_cpu 逃生口形成不对称，是路线 A vs B 的真实 tiebreaker。直接威胁：no-clobber 课程续跑、exact-resume verifier、以及「第二 runner load 后 normalizer tensor 全等且 count 不回退」这条已排产的门。

3. **mjlab API churn**。v1.5.3、Apache-2.0、自标 Production/Stable，33–49 页 rst 文档、923 行 changelog、全树只 2 处 TODO/FIXME、有 CI/nightly/release workflow——成熟度信号是好的。但本地 clone 是 depth-1 单 commit，无法评估 commit 节奏；把一个多月项目钉在一个年轻库的移动版本上是真实风险。缓解：钉版本 + 只用其 manager/dr/sensor 层，不深度依赖其 tasks/ 内容。

4. **5090 上性能未知**。mjlab 只自报「与 Isaac Lab 持平或更快」，本地文档无任何基准表。我们的场景（机器人+台+网+球+球拍接触，4096 env）从没被任何人跑过。**而且必须防一个记账错误**：迁移后如果 s/update 没降，很容易误判成「MuJoCo 慢」——实际上 §9 已经用同硬件同 Isaac 上 yikang 6.383 s/update vs 我们 25.4 s 证明了大头是我们自己的 Python reset 仪式。测吞吐时必须把 solver 时间与仪式时间分开报。

5. **Isaac 侧 receipt 可比性丧失**。整条 Hctrl 机械应力 receipt（v7a PASS、canonical/file/log SHA 79e14853…/cb0fcfdc…/087028bf…）、identity→authority→bundle 链、USD 交织关节序权威、RUNTIME-ASSET-LOADER-V2 的 OpenGL/GLU/USD closure 钉，都是 Isaac Kit bytes，跨引擎不可比。ledger 已把它们标为 PAUSED-ENGINE / DEFER-ENGINE 可弃，且**正式 N=5 信任集今天是空的**（motion promotion 证书、frozen evaluator V4 receipt、GPU1 sidecar receipt、drain/reset receipt 全无），所以正式层面零损失。真实损失只在诊断层：换引擎后 N=1 的数值不能与迁移前的 Isaac 跑逐点对比，只能对比现象级结论（std/LR 走向、Reward hacking、课程锁死、strike/return）。

6. **新引擎特有的能力缺口**：MotionCommandCfg 单 clip（正反手需自建包装层）；接触必须重表达为 regex 模式而非显式 pair 矩阵（三方球拍/球/台需要 secondary_policy='any' + 多 sensor 叠层）；mjlab 要求 write 后显式 sim.forward() 才能读到刷新后的派生量（对照：IsaacLab 是 timestamp 脏标记 + 读时惰性重算，所以从不需要显式 forward——这是 mjlab 移植 BeyondMimic 时唯一带注释标出的载荷性差异，在 mdp/commands.py:407-415，漏了会静默读到 teleport 前的旧位姿一整步）。

7. **组织性风险**：过渡期两套栈并存（Isaac 跑着 N=1 长训、MuJoCo 在勘探），必须严格执行「不为等迁移报告继续增长 Isaac 专属开发队列」，否则会两头投入两头不到位。

### 14.6 组件×路线成本表

| 组件（人话 / 代号） | 现状规模与耦合实况 | A. port 到 mjlab | B. Newton 后端 | C. 厂商 instinct_mj | 原样不动 |
| --- | --- | --- | --- | --- | --- |
| 奖励/课程/合同内核（hope_rewards 4558、action_ball_curriculum 6144、training_contract 5090、action_ball_runtime 10284 等，合计约 65,771 行零 isaaclab import） | 引擎无关；唯一暗礁是 training_contract.py:1523 一条真实运行时 `from isaaclab.envs.mdp import generated_commands` 做 callable 身份比对，与该文件 line 3 的「刻意无 Isaac import」docstring 矛盾 | **S**（1–2 人周）只重接调用点 + 修那条身份比对 | **无**（cfg 层完全不动） | 未知（取决于其 manager 形状） | ✅ 科学配方本身不动 |
| 解析球模型 virtual_ball.py（921 行，含 fused-Triton 快路径 + bitwise-parity 回退 + kill switch） | 零 isaaclab；**这才是给 LIVE strike/landing reward 打分的东西** | **无** | 无 | 无 | ✅ 原样 |
| 真相仪器 physical_ball.py（2,482 行，代码驱动台弹与拍冲量，因为 PhysX 从不解算球接触） | 绕了 PhysX 三个限制：不解球接触、body-frame-only 外力约定、manager 执行序 | **L–XL**（2–4 人周 + 标定）——是设计裁定不是移植：原生 solver 解算可删约 2500 行，但「solver 决定弹跳」改变 reward 确定性 | **L**（同样问题，Newton 也是 MuJoCo 接触） | 未知 | ❌ |
| 两个巨型 command（commands.py 6,911 + hope_commands.py 19,147） | 合计 26,058 行但只有 5–7 条 isaac import；**commands.py 有 19 处物理 API 点（含全部 write_*_to_sim / default_root_state），hope_commands.py 只读、零 reset 写** | **M–L**（3–6 人周，大而机械；mjlab EntityData 命名逐字对应 ArticulationData） | **S**（cfg/manager API 完全共享，只需验证 buffer 语义） | 未知（取决于 TermCfg 平价度） | ❌ |
| 动作项 ClampedJointPositionAction（hope_actions.py 4,124 行） | 直接继承 isaaclab JointPositionAction；substep 安全账本假设 Isaac 的 decimation/apply_actions 节奏 | **S–M**（1–2 人周）clamp 本身降级为 mjlab 的 clip 配置字段；只需重推 substep 记账 | **无–S** | 未知 | ❌ |
| 执行器语义（agibot_a3.py 5 组 ImplicitActuatorCfg + friction） | friction 是「MuJoCo frictionloss 数值照抄进 PhysX 无量纲系数」的未标定遗留 | **M**（需裁定 BuiltinPdActuator 隐式 vs IdealPdActuator 显式——G06 记录该项曾造成 0.61→0.31 m/s 背离） | **M**（同样裁定；但 Newton 额外送 joint_computed_f pre-clamp 力矩缓冲，解掉 §11.4 那个阻塞 torque_limits reward 的观测缺口） | 名义最好（同 A3 族一手配置） | ❌ |
| 场景资产（ShadowTable/PhysicalTable USD + table_frame.py 常数） | 常数是纯解析零耦合；USD collider spawn 代码是 PhysX/USD 绑死；厂商 MJCF **零 ball/table/net** | **L–XL**（2–4 人周）从既有法定常数写 MJCF 台/网/球；attach a3_pingpong.xml；顺带可重开自碰撞 | **无**（USD 场景零成本继承——**这是 B 相对 A 最大的省钱点**） | 未知 | ⚠️ 常数不动，spawn 代码不保 |
| 资产转换脚本 prepare_a3_isaac_asset.py | 把 vendor URDF 的 package:// 改写成 Isaac 路径 | **删**（厂商 MJCF 直接可载） | 保留 | 删 | ❌ |
| 接触/桌碰检测（terminations.py 960 行 + 32 个 filtered ContactSensor） | 32 倍 sensor 的存在理由是「保住 IsaacLab 那个坏的 filtered 语义」，§9.4 点名每步最大吞吐嫌疑 | **S–M**（1–2 人周）重表达为 regex primary/secondary；顺带收敛成 yikang 单 sensor + AABB | **S**（force_matrix_w 支持，但 force_matrix_w_history 是永久 TODO=None） | 未知 | ❌ |
| DR / 域随机化（events.py、hope_push_events.py、isaac_lateral_perturbation.py） | randomize_actuator_gains 其实是 isaaclab 内置库代码（wildcard 再导出），不是我们写的 | **S**（0.5–1 人周）mjlab dr.* 是超集（含 encoder_bias、pseudo_inertia、5 分量 pair friction、逐字段 mj_setConst 等价性安全表） | **S**（IsaacLab events.py 已有 _Physx/_Newton 双后端分支） | 名义最好（一手 DR 轴表已有） | ⚠️ 纯 torch 的 lateral_perturbation.py 不动 |
| PPO runner（my_on_policy_runner.py **实测 7,015 行**，非二手说的 2,100） | 深度内部状态手术做 exact-resume / 自定义存取 | **L**（3–6 人周）rsl-rl-lib 2.3.1→5.4.0，最高风险项，与 Warp determinism 耦合 | **M**（IsaacLab 3.0 也换了 rsl_rl 版本，同样非机械） | 未知 | ❌ |
| ONNX 导出（exporter.py 938 行 + isaaclab_rl glue） | actor 本体是普通 nn.Module | **S**（约 30 行手写 torch→ONNX 替掉 isaaclab_rl 导出器） | **无** | 未知 | ❌ |
| 判官 mujoco_eval_onnx.py（8,140 行，零 isaaclab） | 已经是管线的 MuJoCo 原生一半 | **无** | 无 | 无 | ✅ 原样 |
| judge.sh（1,325 行） | 只在 ONNX 导出腿激活 Isaac venv + 等 Kit boot lock | **S**（0.5 人周，砍导出腿，shell 编辑） | 保留 Isaac 腿 | S | ⚠️ 评分腿不动 |
| G06 门 | 标题就是「Isaac-To-MuJoCo Parity」 | **L（文档/组织成本非代码）** 改锻成「训练 profile ↔ vendor_gate3_v1」平价门 | 部分改锻 | 部分改锻 | ⚠️ Gate3/Gate3B 与其未完成项一律不动 |
| Gate3 / Gate3B（厂商 C++ runner + 真 a3_pingpong.xml） | 从来不依赖 Isaac，唯一晋级权威 | **无** | 无 | 无 | ✅ 唯一终审不变 |
| 合同/bundle 重钉（schema-3 摩擦三元组、identity/authority/USD 关节序） | 摩擦三元组 fail-closed 强制 physx | **M**（1–2 人周）退役摩擦 fail-closed；重钉 identity/authority | **M**（同样退役摩擦；但 identity/USD 层可留） | 未知 | ⚠️ physics/solver profile pins 不必重钉（已有 mujoco_consumer + 跨引擎审计脚本） |
| ballfit 双 YAML（configs/ball_physics_*.yaml + agi 镜像） | 服务两个永久独立消费者，刻意双份 | **无** | 无 | 无 | ✅ 与引擎选择正交 |
| 恢复系数 / 弹跳（table 0.9215、racket 0.646、k_d 0.1253、k_m 0.00404） | PhysX 标量 CoR | **M（新工作量）** 必须在 solref/solimp 空间重推；mjlab dr.* 无对应随机化函数 | **M**（同样） | 名义已解（若他们已标定） | ❌ 这层不享受今天智元重钉的折扣 |
| **reset 仪式（§9.1 七项 + §9.2 向量化，占 23.5 s/update 的 60–75%）** | **纯我们自己的 Python，引擎无关**；同硬件同 Isaac 上 yikang 6.383 s vs 我们 25.4 s | **原样搬过去** | **原样** | **原样** | ✅✅ **换哪条路线都不修——只有自己动手才修** |
| 平台版本前提 | 我们钉 Isaac Lab 2.1 / rsl-rl-lib 2.3.1 | n/a | **隐藏成本：isaaclab_newton 只在 release/3.0.0-beta2，等于 2.1→3.0-beta2 大版本迁移，此段尚未被任何尽调量化** | n/a | ❌ |
| **合计量级** | | **约 15–30 人周（4–7 人月）** | **约 6–12 人周（含 2.1→3.0）** | **不可估（access 未确认）** | 约 65,771 行 + 判官链 + Gate3 保持不动 |


---

## 十五、工程结构第一性重设计终审(08-01;与引擎迁移同轨)

**方法**:2 抽取(结构债台账 17 修正吸收 / 外部架构模式普查)+ 2 对抗核查 + 1 终审
(终审自带 36 处 file:line 独立复核,并纠正了本 doc 前文三处口径:canonical_sha256 已于
07-30 `7e98c3d6` 缓存——§9.5 R0/§9.3 已打修正标;RacketTargetCommand 为 179 方法非 220;
在线 FK 一直在热路径)。以下终审原文入档。

验证性检查已跑完（36 处 file:line 独立复核，含 arch-debt 的 13 条更正在内）。以下是终审。

---

# 目标架构终审：HOPE 训练器第一性重设计（与 Isaac→MuJoCo 迁移同轨执行）

**总裁定**：重设计成立，但**它不是迁移的一部分，也不该等迁移**。真正的分界线是——凡是"引擎换了这些行反正要重写"的，绑在 port 上做；凡是"纯 Python、和引擎无关、还顺手买吞吐"的，**现在就在 Isaac 上做，而且必须做在 port 之前**（否则 port 的吞吐测量在会计上不可解读，这正是 mig-judge 前置清单第 2 条的硬要求）。

先纠三个会误导排产的口径（来自 arch-debt 自身的核查更正，下文一律按更正后的数字）：

- **`canonical_sha256` 已经缓存了**，不是待修项。8 个定义点（`action_ball_runtime.py:684/792/1608/1685/2266/2492/2792/4466`）全部挂着 `@_WeakIdentityCachedCanonicalSha256`（:180）。这条在 07-30 的 `7e98c3d6` 就修了，design_audit 的行号整体偏 −65。**别再把它写进工单。**
- RacketTargetCommand 是 **179 个方法**不是 220；`_update_metrics` 在 **17877**（582 行）不是 15625；`actor_*` 五个降级访问器在 **18490–18509** 且是普通方法不是 property。
- "无在线 FK / 已修正"那条引用是**编造的**。真实位置：`_racket_fk:14363`、`_racket_angular_velocity_w:14440`、`_compute_racket_state:14583`。在线 FK 一直都在，是热路径的一部分。

---

## 1. 第一性需求清单（10 条，每条配可测验收）

从"系统实际在做什么"倒推，不是从"框架该长什么样"正推。

| # | 人话需求 | 为什么是第一性 | 验收（可测） |
|---|---|---|---|
| **N1** | **多动作 clip 服务**：N 条 clip 平衡取样、每条独立身份、能自述族属 | 系统本体就是"多 clip 模仿 + 击球"。今天 `_clip_family_is_forehand`（`hope_commands.py:12313`）硬编码判族，单 clip 臂是二等公民；mjlab 的 `MotionCommandCfg.motion_file` 是单字符串（mig-judge 风险 6），迁移必然要自建包装层 | N=1 与 N=73 走**同一条**代码路径，零 `if n==1` 分支 |
| **N2** | **球拍运动学核**：每 policy step 的 FK / 角速度 / 接触几何 / 击球时序 | 这是纯运动学，与引擎无关，却今天长在引擎绑定的 CommandTerm 里（14363/14440/14583/15114/15147 + `racket_contact_geometry.py` 1471 行） | 给一组 golden 关节角，在 CPU-only host（py3.8）复现 racket site 位姿，零 isaaclab |
| **N3** | **出题服务**：birth→pool→receipt→commit，且 exact-resume | 题库是真卡点。今天 `action_ball_runtime.py` 10,284 行/42 类是服务端，客户端 6,908 行/57 方法卡在 god object 里 | 题目序列 save→load 后逐字节续接（已有能力，只是位置错） |
| **N4** | **课程即服务**：冻结策略上跑 canary 320 → heldout 960，判决与训练漂移解耦 | 这是我们**强于所有 8 家外部先例**的设计（§13.2 第 2 条）。协议本体已经是进程外的——`action_ball_evaluation_inbox.py:1-17` 明写"trainer 与 frozen evaluator 刻意不共享可变 Python 对象" | trainer 侧不 import evaluator 实现，只写 request / 读 evidence |
| **N5** | **收据与合同是产品，不是仪式**：fail-loud / no-clobber / exact-resume 的**保证**一条不减，**成本**全部搬到边界 | 治理是核心需求；但 §10.1 硬数：参考栈每 reset 是**每批固定 17 次 host 读**，我们是 **~24 次/env**，差 3 个数量级 | 保证侧：所有 roundtrip/purity/archive 证明仍逐条执行且失败即炸；成本侧：每 env 每 reset 的 Python 开销归零 |
| **N6** | **部署合同 parity 神圣**：训练 / ONNX 导出 / `mujoco_eval_onnx.py`(8140 行) / 厂商 Gate3 四个消费者看同一份 194-D | 唯一终审 Gate3/Gate3B 从不依赖 Isaac；换引擎会新增第五个生产者 | 合同带显式 version + content hash，四个消费者各自独立断言同一 hash |
| **N7** | **引擎可换且成本可数** | 必须能回答"要动几个调用点"，而不是"要重写 46% 代码"。实测：`source/` 下物理 API 调用点约 **96 处**（含 scripts 共 125 处），而 LOC 口径是 55,435 行——差 500 倍 | 一条 grep 枚举全部物理调用点，且它们全在**一个目录**下 |
| **N8** | **吞吐：reset 成本 O(term)/O(batch) 不是 O(env)** | 23.48 s/update 里 60-75% 是**我们自己的 Python 仪式**；同硬件同 Isaac 上 yikang 是 6.383 s。**换引擎一行不改地跟着搬** | 分级门 A=9-11 s / B=6-8 s；CI 挂每步 `.item()` 计数回归 |
| **N9** | **配置单一真源**：一次发射的实际生效配方从**一个**工件推得，且能在 CPU-only host 上对全部 9 个 task 枚举验证 | 今天 4 层（dataclass 默认 / yaml / reward_pack 表 / CLI），**7/9 task 今天直接开不起来**且测试套完全看不见 | 9 task × 全部 preset 的解析矩阵是一个 ~20 行 host 测试 |
| **N10** | **确定性边界显式化** | 今天 exact-resume verifier 隐含要求物理逐字节，MuJoCo-Warp 给不了（mjlab `faq.rst:247-256`，`seed_rng()` 碰不到 Warp kernel） | 两次同 seed：**课程判决序列必须逐条相同**；物理轨迹只报散度、不做断言 |

---

## 2. 目标分层架构：8 层，每层写明"搬什么进来 / 抄谁 / 退役哪条债"

命名用目录名，因为边界要能被 `grep` 和 CI 强制，不能只活在文档里。

### L0 `plant/` — 物理适配层（唯一允许碰引擎的地方）

- **搬进来**：`commands.py` 的 17 处 `write_*_to_sim`/`default_root_state`、`isaac_lateral_perturbation.py`（1742 行，14 处）、`physical_ball.py` 的 13 处、`terminations.py` 的 10 处、`hope_rewards.py` 6 处、`shadow_ball.py` 5 处、`hope_actions.py` 4 处、`events.py` 2 处、`robots/agibot_a3.py` + `robots/actuator.py`。
- **抄谁**：IsaacLab 3.0 的 `PhysicsManager(ABC)` + `PhysicsCfg.class_type` 分派（`physics_manager.py` 414 行，三后端各 17–27k 行，核心 94,066 行引擎无关）——**但只抄边界形状，不抄多后端**（见反面清单 #1）。更贴身的先例是**我们自己**：`lateral_perturbation.py`(1902，纯 torch) vs `isaac_lateral_perturbation.py`(1742，PhysX 适配) 已经是这个模式，`*_torch.py` 命名约定也已存在（`counter_rally_torch.py`/`strike_spec_torch.py`/`stroke_adapt_torch.py`）。
- **退役债**：COUPLING（132 import / 38 of 99 文件）；DelayedImplicitActuator 死代码（~90 行，全仓 3 处引用全是自指）。
- **红线**：IsaacLab 自己在 `sim/spawners/from_files/from_files.py:15` 有一条**无条件**的 `from isaaclab_physx...` 顶层 import——抽象层会漏。我们的对策是 CI grep：`plant/` 之外任何文件出现引擎 import 即 fail。

### L1 `state/` — 实体数据门面（只读张量结构）

- **搬进来**：`hope_commands.py` 的 5 条只读 isaac import（全是 buffer 读，**零 `write_*_to_sim`、零 reset 写**——这是 19,147 行文件能低成本迁移的根本原因）。
- **抄谁**：mjlab `EntityData` 的 `body_link_pos_w/quat_w/lin_vel_w/ang_vel_w/joint_pos/joint_vel` 与 IsaacLab `ArticulationData` **几乎逐字对应**（mig-judge 路线 A ②）。
- **注意**：mjlab 要求 write 后显式 `sim.forward()` 才能读到刷新值（IsaacLab 是脏标记惰性重算）——这是 mjlab 移植 BeyondMimic 时**唯一带注释标出的载荷性差异**（`mdp/commands.py:407-415`）。这条语义差必须由 L1 门面吞掉，不能泄漏给 L2。

### L2 `brain/` — 任务内核（纯 torch，65,771 行，**不动**）

- **已经在这**：`hope_rewards.py` 4558、`action_ball_runtime.py` 10284、`action_ball_sampling.py` 7963、`action_ball_curriculum.py` 6144、`action_ball_evaluation.py` 5113、`training_contract.py` 5090、`action_ball_evaluation_inbox.py` 4526、`virtual_ball.py` 921、`continuous_questions.py` 1058、`counter_rally.py` 1129、`racket_contact_geometry.py` 1471、`stage1_question_bank.py` 716…
- **唯一要修的一处**：`training_contract.py:1523` 有一条**真实运行时** `from isaaclab.envs.mdp import generated_commands`（在 `_canonical_actor_leg_ref_mask_callables` 里做 callable 身份比对），与该文件 line 3 的"本模块刻意不含 Isaac import"docstring 直接矛盾。改成按 qualified name 比对或注入式注册表，约 20 行。
- **抄谁**：PHC 的 `motion_lib_base.py`（566 行，grep `self.env`/`gymapi`/`self.sim` 零命中）——数据面类彻底无环境耦合是可达的。
- **退役债**：GOD MODULE `action_ball_runtime.py`（10,284 行 / **42** 类）——不是删，是按域切成 3 个文件：`LazyActionTaskPool`(~3,300)、`ActionBirthBroker`(~1,600)、`ActionBallTaskReceipt`(~1,300)。这是纯机械切分、零行为变化，随时可做。

### L3 `terms/` — 管理器薄壳（目标：从 26,058 行降到 ~3,000）

- **搬走什么**：见第 3 节的 RacketTargetCommand 拆解。
- **抄谁**：mjlab 整个 tracking 任务 **1,855 行 / 15 文件**，其中 command 是 **608 行**。我们的一个 command 是 16,319 行——**26.8 倍**。mjlab 的 `sampling_mode: Literal["adaptive","uniform","start"]`（`commands.py:598`，分支在 320-325）是我们 5 个 `_sample_targets_*` 并列实现（8235/13299/13456/13539/13584）+ 12 个布尔焊死开关的正解。
- **退役债**：GOD OBJECT RacketTargetCommand；task_first 谱系（类内 **524 行/11 方法** + 模块 1,823 行 = **2,347 行**，0/9 task yaml 引用，import 是函数内 lazy 的 3665/3670，删除无 import 面阻力）；65 注册 term 里 26-36 个零权重死项。

### L4 `ledger/` — 治理层（**在 checkpoint / rollout 边界，不在热路径**）

这是本次重设计**收益最大**的一层。

- **搬进来**（保证不变、位置变）：
  - `hope_commands.py:5400 / 6385 / 6444 / 6498 / 16868` 的 `from_dict(to_dict()) != receipt` 不变性重证明——移到构造一次 + 边界一次。
  - **`hope_actions.py` 的 `ClampedJointPositionAction` 是 3,457 行 / 70 方法，其中 1,760 行（51%）是 `_joint_safety_*` / `_table_contact_*` 证据机器**（fingerprint / archive / snapshot / prepare_view / export_clone）。热路径只该做一件事：往预分配的 GPU 环形缓冲追加张量。抽取、指纹、归档全部由 L4 在边界抽干。
  - broker/provider state_dict purity 前像与检查；terminal-archive 物化；5×7 每步 clone 记录表；4096 元素 receipt 字符串元组 + None 扫描；`assert_contract` 三方冗余（`action_ball_runtime.py` 14 处 + 两个 command 8 处 → 单一 owner）。
- **抄谁**：mjlab 的 **`RecorderManager`（265 行）+ `MetricsManager`（225 行）**——外部把逐步记账放进有显式节拍的专用 manager（`per_substep` / `reduce: mean|last|max`），而不是塞进 term 内部。ProtoMotions `base_evaluator.py`（721 行）从**外部**驱动 `env.reset`/`env.step`（:174/:183），env 本体 1,619 行零 MotionMetrics 逻辑。
- **顺手转正一个 hack**：`my_on_policy_runner.py:3941/3943` 用 monkeypatch（`self.env.step = ...` / `self.alg.update = ...`，`finally` 里复原）挂 reward-activation ledger 和 rollout 边界钩子——因为要插进上游 `learn()` 循环体内部。L4 把 **rollout 边界**变成一等公民 API，钩子注册而非替换绑定方法。rsl_rl 5.4 已经把 `Logger` 抽出去了（2.3.1 的 529 行 runner → 5.4 的 250 行，−53%），我们的 runner 是 **7,015 行**，是 2.3.1 形状的放大版。
- **退役债**：HOT-PATH GOVERNANCE（§9.1 七类，估 13.5–17.8 s / 23.48 s）；runner monkeypatch 耦合。
- **注意**：外部**九家框架零 receipt/hash-chain 机制**（grep `hash_chain|receipt|provenance|merkle` 全空）——它们的可复现性是"run 边界的 {config 快照, seed, checkpoint} 三件套"。所以**这一层没有先例可抄，是我们的自有资产**；能抄的只有"节拍与位置"（RecorderManager），不是"要不要有"。

### L5 `profile/` — 配置单一真源（见第 4 节详述）

- **搬进来**：`scripts/train.py` 的 `_apply_task_overrides`（**9818**，2,466 行）、`_expand_reward_pack`（**8975**）、四张 v2 展开表（`_REWARD_PACK_V2_KEYED/DIRECT/OPTIONAL/CALIBRATED`）、`_calibrated_override_marker`（~70 行）、23 个 `_validate_action_ball_*`/`_finalize_*_training_cfg`/`*_reward_contract` 辅助函数。
- **抄谁**：mujoco_playground 的 `default_config()` → 单个 `ConfigDict`（`locomotion_params.py:26,171`），零展开层；mjlab 的 `dict[str, RewardTermCfg]`（`reward_manager.py:54-56`）取代嵌套 configclass 树。**注意：reward-pack 式中层展开在可克隆的外部框架里零先例，这条债是我们独有的。**
- **退役债**：GOD OBJECT train.py（13,379 行 / 131 顶层 def）；4 层 racket_position_weight；DIRECT 层无逃生口；**7/9 崩溃**；测试拓扑（唯一合成 fake 恰好长成能跑的那条血统）；`canonical_ready_mode` 焊死双职能（`commands.py:580-583` 同时管契约绑定 1262-1710 与 reset 分布 633+）；RealSensor 死别名；12 个焊死布尔（`commands.py:6726/6733/6757/6776/6811-6816/6829/6839/6868/6888/6898`）。

### L6 `examiner/` — 课程评测服务（**已经 80% 到位，只差把客户端搬出来**）

- **已经在这**：`action_ball_evaluation_inbox.py` 4,526 行——append-only、内容寻址、durable temp + no-clobber 安装、重复 key/非有限数/部分写/序列缺口/重放身份/重叠分配/未钉 sidecar 代码**全部 fail closed**，且自述"dependency-light 以便在 CPU-only host 上审计与测试"。这是全仓最好的一块架构。
- **要搬进来**：RacketTargetCommand 里的 **6,908 行 / 57 个 `_action_ball_*` 方法**（birth claim/provide/refill/commit、drain-reset、frozen canary/heldout 执行、exact-resume state_dict）。
- **抄谁**：ProtoMotions `base_evaluator.py` 从外部驱动 env；Dextreme-ADR 的 per-(param,bound) 独立队列（R9 形态 2 的先例）。
- **退役债**：god object 最大的一块（6,908/16,306 = 42%）。

### L7 `contract/` — 观测合同注册表（部署 parity 的物理位置）

- **已经在这**：`actor_observation_contract.py` 735 行，**6 个**固定形状合同（FULL 180 / DEPLOY_PARITY 175 / DEPLOY_PARITY_FACE179 / DEPLOY_PARITY_STATION181 / HITTER_FOOTWORK 177 / HITTER_PURE 110）+ 5 个 N 参数化构造器（:137/:172/:195/:287/:388）。
- **债**：N 参数化家族**没有版本字段**，靠字符串前缀 + 正则分派（`resolve_actor_observation_contract`:602）；`HOPEPingPongActionBall.yaml:36` 是 `actor_obs_contract: null`，要求每个 launcher 在 CLI 手拼 `action_ball_table_pose_twist_heading_task_n<N>`。该模块自己的 docstring 就承认："旧的 194-D checkpoint 形状相同但速度/法向语义不同"——**形状不足以区分语义**。
- **修法**：`(family, schema_version:int, action_count:int, layout_sha256)` 四元组取代字符串约定；`total_dim` 降级为派生量而非身份。N 从字符串后缀变成字段。四个消费者（训练 / exporter / mujoco_eval_onnx / vendor gate）各自独立断言 `layout_sha256`。
- **成本 0.5 人周，但必须现在做**：port 会产生第五个合同生产者，在字符串约定上再叠一层是自找。

---

## 3. 拆解 RacketTargetCommand：16,319 行 → 9 个模块 + 1 个 ~900 行薄壳

我按方法边界做了**行加权**统计（不是方法计数），这是拆解的实际重量分布：

| 责任簇 | 行数 | 方法数 | 去处 | 依据（file:line） |
|---|---|---|---|---|
| **birth / pool / frozen-eval 客户端** | **6,908** | 57 | → L6 `examiner/`，拆 3 个模块 | `_action_ball_refill_pool_many:6900`(803)、`action_ball_frozen_evaluator_execute_v1:10363`(584)、`_action_ball_frozen_eval_solve:9300`(490)、`_action_ball_frozen_eval_install:9790`(455)、`_action_ball_load_exact_resume_state_dict:11383`(650) |
| **构造/接线**（`__init__` 1288 + `_initialize_action_ball_runtime` 1135） | 2,423 | 2 | → L5 `profile/` 消费 + 一个 builder；`action_ball_runtime_bootstrap.py`(989) 已存在，是落点 | :2265 / :3931 |
| **虚拟球评估 `_vb_evaluate`** | 728 | 2 | → L2（`virtual_ball.py` 921 行已纯 torch，零 isaaclab） | :16470 |
| **EMA 指标 + 自适应 σ** | 640 | 2 | → L4 `ledger/`，logging 节拍不是 policy 节拍（§10.3 ADJUSTS 明确点名） | `_update_metrics:17877`(582)、`_update_adaptive_sigma:17819` |
| **目标采样 5 个并列实现** | 583 | 5 | → L3，收成一个 `Literal` 分派的策略表 | :8235/:13299/:13456/:13539/:13584 |
| **task_first 谱系** | **524** | 11 | → **删**（连同模块 1,823 行，共 2,347 行） | `_initialize_task_first_runtime:3662`(253) |
| **连续题库 `_cq_*`** | 512 | 12 | → L2，并入 `continuous_questions.py`(1058) | — |
| **击球时序状态机** | 248 | 5 | → L2 纯 torch 模块 | `_compute_strike_timing:15147`、`_refresh_strike_timing_for_policy_step:15114` |
| **回合/步法记账** | 165 | 3 | → L4 | `_count_swing_starts:16165` |
| **FK / 球拍状态** | 112 | 3 | → L2 纯运动学核（**这是热路径上真正必要的计算**） | `_racket_fk:14363`、`_racket_angular_velocity_w:14440`、`_compute_racket_state:14583` |
| **hold 恢复 / A1 降级观测 / planner 修订** | 84 | 7 | A1 → L3（延迟抖动数学本身可移植）；planner → `planner_revision.py`(828) 已存在 | `actor_racket_target_pos_w:18490`–`actor_time_to_strike:18509` |
| **其余散件**（题库安装、事件时序绑定、精确行为计数、稀疏奖励资格、CommandTerm API） | ~3,400 | ~50 | 约 1/3 进 L4（记账），1/3 进 L2（题库/事件），**1/3 留在薄壳** | `_resample_command:14144`(219)、`_update_command:15292`(102)、`install_external_exam_questions:13856`(143)、`_ensure_exact_behavior_decision_counters:17328`(143) |

**留在薄壳的 CommandTerm（目标 ≤900 行）只做四件事**：(1) `command`/`_resample_command`/`_update_command`/`_debug_vis` 的 API 面；(2) 从 L1 门面读张量；(3) 向上面 9 个模块分派；(4) 把结果写回 command buffer。参照系：mjlab tracking command **608 行**。

同文件的 4 个前置适配类（`_ActionBallPoolSolverAdapter:200`、`_ActionBallDomainAuthorityAdapter:267`、`_ActionBallBirthProviderAdapter:302`、`_ActionBallDrainResetRuntimeSource:1273`，共 2,053 行）随 L6 一起搬走——它们本来就是端口适配器，只是放错了文件。

---

## 4. 配置单一真源：杀掉 4 层覆写链

### 今天的四层（外加一条无逃生口的第五规则）

1. dataclass 默认（`hope_env_cfg.py:877` weight=4.0）
2. task yaml（8/9 个文件显式声明；RealSensor 靠 Hydra `defaults` 继承，所以"9/9"是错的）
3. reward_pack v2 展开表（`train.py:8838/8860`，冻结值 393.4）——**对所有 9 条已注册臂是死码**，因为 layer 2 总是先写
4. CLI（Hydra 在 train.py 读到之前就并进 layer 2，事后不可区分）
5. **DIRECT 层：15 个 term 无条件覆写、零 CLI 键**（`train.py:9034-9053`，注释自认"今天没有 CLI 键的项"）

07-27 那次事故的形态很说明问题：三份内部文档把在跑的 4.0 写成 393.4（**98 倍**），修法是加了 ~70 行 `_calibrated_override_marker` 让漂移**事后可检测**，而不是让四层结构不存在。

### 提案：一个 run 一个 `RunProfile`，解析一次，冻结

```
resolve_run_profile(task_name, overrides) -> RunProfile   # 纯函数，零 isaaclab，零 torch
```

产物是**值**不是**变异**。今天 `_apply_task_overrides` 是 2,466 行的原地 mutate；新解析器返回一个 frozen dataclass，之后任何 writer 抛异常。

**reward_pack 语义如何存活** —— 关键洞察：v2 是**三件不同的事**共用一个名字，拆开后各自有正确归宿：

| v2 的哪一部分 | 真实语义 | 新归宿 |
|---|---|---|
| KEYED 12 项（`full_body_mimic`、`foot_slip_sq_weight` 等） | "推荐默认值，用户可覆盖" | **preset 文件**，与 task yaml **同层**参与 Hydra defaults 组合 → 覆盖规则从 4 条变 1 条（Hydra 自己的） |
| CALIBRATED 3 项（393.4 / 295.1 / 229.5） | "标定冻结数，不该被静默压过" | profile 的 `frozen:` 段。覆盖它必须显式写 `unfreeze: [racket_position_weight]` 并进 receipt，否则 **fail-loud**。等于把今天 `reward_pack_strict=true` 的行为变成**默认且结构性** |
| DIRECT 15 项（零 CLI 键、无条件写） | 这**根本不是 preset**，是"任务血统缺项"的补丁 | 消失。task **声明自己的 term 名单**（纯数据列表），preset 只能给已声明的 term 赋值 |

**7/9 崩溃如何变成结构性不可能**：

今天崩溃的机制是——DIRECT 循环对每个非零权重项做 `_require(hasattr(R, name))`（`train.py:9040-9043`），而 `HOPEHitterPureRewardsCfg`(:2369) 直接继承裸 `RewardsCfg`、`HOPERewardsCfg`(:873) 是 DeployParity 的**父类**（legacy `HOPEPingPong.yaml` 用的就是它，首个崩点是 `hit_unstable_support` −10.0 而非 `virtual_landing`）。零权重缺项现在有"retired-zero 跳过"分支，但非零项仍炸。

新结构下，解析器拿到的是**task 声明的 term 名单（纯 python 列表）**，不是某个继承 isaaclab `RewTerm` 的 dataclass 的属性表。名单不匹配在纯 python 层就炸，且打印 `(task_name, preset_name, term_name)` 三元组。

**这直接解开测试拓扑死结**：今天 `tests/test_reward_flags_overrides.py` 的 `_make_env_cfg()`（:89-338）是**唯一**的合成 fake，而它恰好长成能跑的那条血统（含 `virtual_landing`/`hit_unstable_support`/`upright_exp`…）。真实的 `HOPEHitterPureRewardsCfg` 在 py3.8 host 上**根本 import 不了**（41/204 测试文件提到 isaaclab，0 个能真跑）。解析器纯 python 之后，那个会捕获这个 bug 的测试是：

```python
for task in ALL_TASKS:            # 9
    for preset in ALL_PRESETS:    # v1, v2
        resolve_run_profile(task, {"reward_pack": preset})   # 不抛即通过
```

**~20 行，18 个组合，host 上跑，0.1 秒。**

**顺手退役的两条**：
- 零权重项不进名单 → 65 注册降到 29 live，26-36 个死 RewTerm 对象不再构造，IsaacLab `RewardManager.compute` 的每项每步 `list.index()` 线性扫（65 元素表 × 29 次/步）跟着消失。
- `canonical_ready_mode` 拆成 `canonical_ready_contract: bool` + `reset_policy: Literal["canonical_ready_strict","rsi","post_swing_mix"]` —— 正好是 §9.5 R2 要的那把锁，先例是 mjlab 的 `sampling_mode` Literal。**默认字节等价，本身零行为变化**，但它是 R1/R3/R4/R5 的 enabler。同理，12 个焊死布尔收敛成 2-3 个 Literal。

---

## 5. 确定性边界：决策级逐字节，物理级允许漂移

这是迁移最大的单点阻塞（mig-judge 新风险 #2），也是最容易被含糊过去的一条。**必须写成合同条款，不是原则。**

### 必须逐字节（Tier-1 Exact）

| 对象 | 为什么它能做到逐字节 | 落点 |
|---|---|---|
| **课程判决序列**（arm status、frontier_index、rho、center_failures） | 判决的输入是 evidence ledger 的**行**（计数、Wilson CI、blocker 布尔），不是浮点物理量 | `action_ball_curriculum.py` state_dict + `curriculum_state_sha256` 交叉校验（**全场最好**，8 家里唯一） |
| **收据 / 内容寻址** | JSON 规范化 + sha256，纯 CPU | `action_ball_runtime.py:684…4466` 8 个 `canonical_sha256`（**已缓存**） |
| **续跑身份** | normalizer 张量（194/318）、sampler 状态、policy 权重、题目序列 | PORTABLE-GOLDEN-GATE 缺口 (c) |
| **观测合同布局 + action scale + q_des clamp** | 纯配置与整数索引 | L7 `layout_sha256` |
| **出题序列**（给定 seed 与课程状态） | birth broker 是纯 Python FIFO | `LazyActionTaskPool` |

### 允许漂移（Tier-2 Statistical）

MuJoCo-Warp 的接触解算、逐 env 轨迹、reward **数值**、击球/回球率（只在置信区间内比较）。

### 机制（三件，都可实施）

**M1 — 判决级确定性 + 可重放事件**。冻结评测的裁决是 `f(frozen_policy_sha, question_list_sha, evidence_rows)` 的纯函数。把裁决记成签名事件，**重放 = 对记录下来的 evidence 行重跑判决函数，不重跑仿真**。这不是新发明——`action_ball_evaluation_inbox.py` 的 `request → evidence → acknowledgement` 三段内容寻址协议已经是这个形状，只是没被声明为"可复现性的定义"。把它升格为定义，`bit-exact reproducibility` 这根治理支柱就从"物理逐字节"迁移到"判决逐字节"，而后者**在 Warp 上可达**。

**M2 — 量化闸门（新增的硬规则）**：任何喂给治理判决的浮点量，必须先过一个**显式量化器**（声明容差），且**被 hash 的是量化后的值**。例：撞台判定 `force > TABLE_HIT_FORCE_THRESHOLD_N` 已经是量化的（布尔）；落点距离要变成 `round(d / 5mm)`。这条让判决确定性**对物理漂移免疫**，而不是祈祷物理逐字节。今天没有这条规则，所以任何一处直接 hash 浮点的地方都会在 Warp 上随机失败。

**M3 — 确定性探针作为门，不作为假设**：同 seed 跑两次，比 (a) 判决序列 —— **必须逐条相同，不同即真 bug**（说明某个判决读了未量化的浮点）；(b) qpos 轨迹 —— 只记录最大散度与增长率，进 receipt，不断言。这条探针在 Isaac 上今天就该建（现在也没有），迁移后原样复用。参照物：Newton 的 `use_mujoco_cpu` 路径可作为 golden 参考实现（mjlab 全树无 CPU 回退，`pyproject` 硬钉 `mujoco-warp>=3.10.0.3`）——这是路线 A vs B 唯一真正的 tiebreaker。

**合同修订**：exact-resume verifier 今天隐含断言 Tier-2，必须改成显式两档，否则迁移后它会在一件 MuJoCo-Warp 结构上给不了的事情上永久红灯。**这不是软化 fail-loud——是把断言指向一个真实存在的保证。**

---

## 6. 迁移途中 vs 迁移后：排产

### 判据

**必须搭 port 的**：port 反正要重写这些行，分两次做等于付两次。
**必须在 port 之前的**：纯 Python、引擎无关、且**会显著缩小 port 本身**，或者是 port 测量的前提。
**必须在 port 之后的**：科学改动（混进 port 会让任何回归不可解读）。

### 排产表

| 阶段 | 内容 | 人周 | 与引擎的关系 | 门（验收） |
|---|---|---|---|---|
| **P0**（现在，队列外） | mjlab CPU 勘探三题（mig-judge 已裁）**+ L7 合同版本化** | 2–3 + **0.5** | 无关 | 三题有答案；`layout_sha256` 四消费者断言绿 |
| **P1**（现在，Isaac 上，与 N=1 长训并行） | **L4 热路径治理搬家** + **L5 配置单一真源** | **2–4** + **2–3** | **完全无关**——§9 已证同硬件同 Isaac yikang 6.383 s vs 我们 25.4 s | gate A ≤11 s/update；9×2 解析矩阵 host 绿；在跑 N1 臂的 effective recipe sha **字节不变** |
| **P2**（Isaac 上） | **L6 课程服务客户端抽出**（6,908 行）+ **死代码删除**（task_first 2,347、DelayedImplicitActuator 90、RealSensor 别名） | **2–3** | 无关 | RacketTargetCommand ≤8,000 行；`grep task_first` 零命中 |
| **P3**（port 本体） | L0 适配层 + L1 门面 + L3 薄壳重写 + MJCF 场景 + 接触重表达 + 执行器裁定 + 恢复系数标定 + rsl_rl 2.3.1→5.4 | **12–20** | 就是 port | 194-D golden parity；Tier-1 exact-resume；gate B 6–8 s |
| **P4**（port 之后） | 课程 R1/R3/R4/R5 可逆性与样本量、R8 失败加权、R9 多臂并行 | 另计 | 无关（纯科学） | 各自 A/B |

**总计 P0–P3 ≈ 19–33.5 人周**，对比 mig-judge 单算 port 的 15–30 人周。

**关键论点：P1+P2 的 6–10 人周不是净增成本，是从 P3 里挪出来的。** 三条理由：

1. mig-judge 给 `commands.py + hope_commands.py`（26,058 行）的 port 报价是 **3–6 人周**，这是路线 A 的第二大单项。P2 之后 RacketTargetCommand 从 16,319 降到 ~8,000（删 2,347 + 搬走 6,908 的一半以上落在 L2/L6，两者都是**零 isaaclab 的纯 Python，port 不碰**），这一项直接缩水约 40%。
2. port 一个 2,466 行的原地 mutate 覆写分派器，比 port 一个纯函数解析器贵得多——而且解析器**在 port 之后完全不用改**。
3. **P1 是 P3 吞吐测量的前提**。mig-judge 前置清单第 2 条原话要求"分开报告 solver 时间与我们 Python reset 仪式时间——否则会把 §9 已证明的引擎无关开销误记到引擎账上"。仪式还在的情况下测 mjlab，得到的数字**在会计上不可解读**，会直接导致对 MuJoCo 的错误裁决。

**回答"热路径治理搬家能不能先在 Isaac 上做"：不但能，而且必须。** 它是 P1 的核心，是唯一能在 N=1 长训还在跑的这 11 天里兑现的吞吐收益（23.48 → 9–11 s 意味着每 lane 从 5.4 天降到 ~2.3 天），而且它同时是 §9.5 的 **R0**——被文档点名为 R1/R3/R4/R5 的 blocker，即"修好 reset 性能，保真终止就重新买得起，§7 的一半问题跟着解"。

**硬约束（继承看板纪律）**：P1/P2 期间对在跑的 N=1 臂必须**默认字节等价**——每一步的门是 `effective_reward_recipe_sha256` 不变，不是"测试通过"。不为等迁移报告继续增长 Isaac 专属 receipt 链（ledger 已标 PAUSED-ENGINE）。

---

## 7. 反面清单：5 件明确不做

**① 不要为 2 个引擎建通用多引擎抽象层。**
证据：IsaacLab 为支持 3 后端付出 94,066 行引擎无关核心 + 3×(17k–27k) 后端包，**并且仍然漏**——`sim/spawners/from_files/from_files.py:15` 有一条无条件顶层 `from isaaclab_physx...`（Newton-only 安装会 ImportError）。HumanoidVerse 家族的第 4 后端更彻底：ASAP/FALCON/PBHC 三个仓都有 `config/simulator/mujoco.yaml` 指向 `humanoidverse.simulator.mujoco.mujoco.MuJoCo`，**四个 clone 里都不存在这个实现文件**。我们是**换**引擎不是**支持**两个引擎：做一条单向适配（~96 调用点），钉死目标引擎，让 Isaac 死。反目标：一个有两个实现的 `PhysicsBackend` ABC。

**② 不要重写那 65,771 行大脑。**
它是资产不是负债：`hope_rewards.py`/`virtual_ball.py`/`action_ball_curriculum.py`/`action_ball_runtime.py`/`training_contract.py` 全是纯 torch/python。唯一缺陷是 `training_contract.py:1523` 那一条运行时 isaaclab import（~20 行修掉）。特别地：`virtual_ball.py`（921 行，含 fused-Triton 快路径 + bitwise-parity 回退）才是给 LIVE strike/landing reward 打分的东西，**零改动可搬**；`action_ball_runtime.py` 的切分是按域切文件，不是重写。

**③ 不要软化 fail-loud / no-clobber / exact-resume。**
重设计的口号是**"移成本，不移保证"**。具体禁止：不许对 purity 检查做抽样；不许给 `assert_contract` 加"性能模式"开关；不许把 no-clobber 降级成覆盖警告。L4 的正确表述是"用**同样的输入**在边界检查**一次**"，不是"少检查"。唯一允许的语义变更是第 5 节的 Tier-1/Tier-2 分档——那是**把断言指向一个真实存在的保证**，不是放松。同理，§13.4 的 R4（blocker 分层）要盯紧："砍的是误伤，不是护栏"——`hold` 语义必须真的阻塞下一轮请求（复用 `request_due` 的 `needs_reset` 路径），否则就退化成静默继续。

**④ 不要在 port 的同一批次里改课程科学（R1–R9）或 reward 权重。**
port 本身已经背了一次物理重标定：MuJoCo 没有标量 CoR，table 0.9215 / racket 0.646 必须在 solref/solimp 空间重推，且 mjlab 的 `dr.*` 里**没有**任何 solref/solimp/restitution 随机化函数。再叠一个 reward/课程改动，任何回归都不可归因。§10.3 的保护清单同向：追吞吐期间不要顺手加 push/interval 事件，非对称 actor/critic 要保住。**唯一例外**是 R6（把"N1 = 固定难度跑"写死进合同）——那是零成本文档动作，现在就做。

**⑤ 不要把配置重构做成"再加一层"。**
最可能的失败模式是引入一个 `RunProfile` 之后，它自己又长出覆盖钩子，变成第 5 层。规则写死：**解析后冻结，任何 writer 抛异常**；`grep` 到 `profile.*=` 赋值即 CI fail。也不要建通用配置 DSL——目标形状是 mujoco_playground 的一个 `default_config()` 返回一个 `ConfigDict`（`locomotion_params.py:26,171`），不是 Hydra 之上再造一个 Hydra。

**附加一条口径纪律（不算"不做"，但同等重要）**：**不要用 LOC 口径估迁移。** 55,435 行 import isaaclab，但真实物理 API 调用点在 `source/` 下只有约 96 处（含 scripts 125 处），集中在 `commands.py`(17)、`isaac_lateral_perturbation.py`(14)、`physical_ball.py`(13)、`terminations.py`(10)。按 LOC 估要么导致瘫痪（"46% 要重写"），要么导致 port 被严重低估范围。**每一次范围估算都必须报调用点数，不报行数。**


---

## 十六、Reward 绝对量级(scale)与 PPO 耦合专项(08-01;应 Franco"不只是比例,还有整个 scale"之问)

**覆盖边界交代**:本 doc 此前只做了**形状与比例**(§7 核宽、§11.4 厂商形状迁移),
**没有**系统做过绝对量级 × PPO 机制耦合。本节补齐。方法:2 抽取(我方生效量级与 PPO
尺度敏感性 / 三库+厂商量级)+ 2 对抗核查(4 修正/7 漏项)+ 1 裁决。**两个决定性事实经源码
逐行坐实**:IsaacLab `reward_manager.py:150` 的 `func × weight × dt`(所以全部权重要乘 0.02
才是真实每步收入);rsl_rl 2.3.1 是**一个 ActorCritic 模块、一个 Adam、一次 backward、
一次全局 `clip_grad_norm_`**(actor/critic 共用裁剪——但见下文对"强版本"的否定)。

### 16.1 裁决(原文入档)

## 一句话裁决

我们的 **dense（每步）层量级是对的**（+0.03~+0.09/步，和三库的 +0.09~+0.10/步同一档），病灶全部在**两个事件尖峰**（死亡 -72.0、落点 +32.98）和**罚项层剂量**（barrier 单关节地板 -0.20/步）上——它们相对 dense 锚高出 2~3 个数量级，把回报分布变成双峰，于是 batch advantage 归一化之后，除"别死"以外的一切梯度被压到约 0.3%~1%。同时 §7 的 σ=0.075 m 死核让"击中"这一层的实际收入恒等于 0，三层分层（模仿<击中<质量）在现役配方里已经塌成两层（模仿 + 不死）。

## 用我们自己的数字说话

- 死亡一次 = -72.0（-3600 × dt 0.02，一次性）。以 dense 0.06/步计：
  - = **1200 步 = 24 秒**完美 dense 收入才还得清；
  - 而 update250 的**平均 episode 只有 124.11 步 = 2.48 秒**；
  - = **约 10 倍整段 episode 的全部正收入**；
  - = **12 倍折扣视野收入**（1/(1-γ)=100 步 × 0.06 = 6.0）。
- 三库最大单步罚项 joint_limit -10.0 → post-dt 约 -0.2，只有各自视野收入（≈9.5）的 **2%**。我们是 **1200%**。差约 **600 倍**。
- 落点 +32.98 = 5.5 倍视野收入；但 §13 已核实 strike_opportunity_count=0，**这一项在现役窗口里一次都没发放过**——现在的重尾是**单边负尾**。
- 项间比例的真问题：死亡:落点 = 72:33 = **2.2:1**，意味着挥拍的盈亏平衡命中率 = 72/(72+33) = **69%**。一个 from-scratch 策略永远够不到 69%，所以**"不挥拍"在结构上就是当前 reward 下的最优解**——这与 update250 的 exact 命中 0.23%、window 5.01%、capture/return=0/0 完全自洽。

## scale 通过哪些通道真正影响学习（逐条判定，含"不成立"）

**A. 真正尺度不变（uniform rescale 下确认不变）**

1. **clipped surrogate loss 与 actor 从 surrogate 拿到的梯度** —— 成立。rsl_rl 的 advantage 在**整个 rollout buffer 上归一化一次**（rollout_storage.py:167，98,304 个样本 = 24 步 × 4096 env），`normalize_advantage_per_mini_batch` 默认 False 且我们没覆盖。reward 全体乘 k，归一化后 advantage 不变。
2. **adaptive-KL 学习率调度** —— 成立。KL 只读 actor 自己的 mu/sigma（ppo.py:269-303），公式里没有 reward/return/advantage。
3. **熵项的值和它自己的梯度** —— 成立。entropy = Normal(mu,std).entropy()，纯 σ 的函数，与 reward 无关。

**B. 明确不成立的说法（别再按这个推理）**

4. **"reward scale 直接放大 actor 的梯度"—— 不成立。** actor 参数的梯度只来自 surrogate 和 entropy，两者都尺度不变；critic 参数与 actor 参数不相交。
5. **"entropy_coef 需要随 reward scale 一起调"—— 不成立。** 熵梯度对 reward 尺度严格不变；把它当 scale 补偿旋钮是错配（详见建议 R8）。
6. **"共用裁剪 ⇒ critic 把 actor 饿死"这个强版本 —— 不成立（弱版本成立，见 C4）。** `clip_grad_norm_` 是给**整个参数向量乘同一个标量**，不改变 actor 与 critic 分量之间的相对方向或比例；而 Adam 的 per-parameter 归一化（m̂/√v̂）会**抵消一个恒定的公共因子**。所以"critic 梯度大 ⇒ actor 步长被恒定压小"不成立。

**C. 真正不变不了的（按危害排序）**

1. **重尾在归一化之后仍然压制其他项 —— 这是首要通道。** 全 batch 归一化统一的是**整体尺度**，不改变**项间比例**。γ=0.99 的视野是 100 步、episode 只有 124 步，所以**一个以死亡结尾的 episode 里，几乎每一个状态的回报都是 -72×0.99^k ∈ [-72, -21]**，dense 收入 7.4 只是零头。GAE 的有效窗口 1/(1-γλ)=17 步，死亡 delta ≈ -51 而 dense 步间 delta 差 ≈ 0.01~0.03。推导（非实测）：σ_A ≈ 10~20，于是 dense 项的归一化 advantage ≈ 0.007，死亡方向 ≈ 3，**相差约 500 倍**。PPO 的 clip=0.2 和 KL=0.01 这两个"预算"几乎全被死亡方向花光。这一条不需要任何 rsl_rl 版本假设即成立。
2. **value_loss ∝ scale² —— 成立，且无任何缓冲。** value_loss_coef=1.0，且**活跃路径上没有任何 reward/return/value 归一化**（RolloutStorage 无 normalizer 字段、ActorCritic.evaluate 是裸 MLP、OnPolicyRunner 只给 obs/privileged_obs 包 EmpiricalNormalization）。*修正*：库里确实存在一个 `EmpiricalDiscountedVariationNormalization`，但只服务 RND 内在奖励，我们从不传 `rnd_cfg`，所以确认失活——"整个栈没有归一化"要收窄成"我们实际走的这条路径没有"。死亡附近单样本平方误差 ~72²=5184，dense 步 ~0.01，差 5~6 个数量级。
3. **critic 初始化不看 scale。** ActorCritic 用 PyTorch 默认 nn.Linear 初始化；正交缩放初始化的 helper 存在但注释写明 "not used at the moment"。初值≈0 面对 O(-70) 的真实回报，初期 explained_variance 必然接近 0；而 GAE 的 delta 直接吃 critic 的裸输出（rollout_storage.py:156），所以**早期 advantage 本身就是坏的**——尺度问题先污染 critic，再经 GAE 污染 actor。
4. **单次全局 grad-norm 裁剪确实共用（弱版本成立）。** 已在 pinned 源码逐行核实：一个 ActorCritic nn.Module、一个 Adam（ppo.py:96）、一次 backward（372-373）、一次 `clip_grad_norm_(self.policy.parameters(), 1.0)`（385）。残留的真实伤害有两条，都不是"恒定压小"：(i) 裁剪因子 c=1/‖g‖ **随 batch 里死亡事件数波动**（快照里终止占比 3.1%/15.7%/32.0%，波动一个数量级），给 actor 注入乘性噪声，而 Adam 的二阶矩记忆（β2=0.999，此处 5 epoch×4 minibatch=20 步/update ⇒ ≈50 个 update）跟不上这种批间跳变；(ii) **LR 是 actor/critic 共享的**，由 adaptive-KL 控制，而 KL 几乎全被死亡方向消耗——想给 dense 层加速就必然同时给 critic 加速。危害等级：中，不是致命。
   > **版本警告**：这一条完全依赖 rsl_rl == 2.3.1。本仓库内**找不到任何 rsl-rl 版本 pin**（grep 全空），diligence doc 自己也把"把 rsl_rl 版本写进 training receipt"列为未完成项。若 pod 上实际是 5.x，actor 与 critic 是**分开各自裁剪**，C4 直接降级为不成立。这是必须先关掉的一个洞。
5. **熵 vs 任务信号的相对权威随重尾变化（真实的间接耦合）。** init_noise_std=0.02、31 个关节：每维熵 = 0.5·ln(2πe·0.02²) = **-2.494 nat**，合计 **-77.3 nat**；entropy_coef=0.01 ⇒ 对 loss 贡献 +0.773，对每个 log σ 的梯度是**恒定 -0.01**（往上推 σ）。而 surrogate 对 log σ 的梯度 ≈ 归一化 advantage × O(1)：dense 侧 ≈ 0.007（往上比不过熵），死亡侧 ≈ 3（往下狠压 σ）。结论是一句很难听但精确的话：**现在 σ 是被"死亡"和"熵"共同决定的，和击球质量无关**；熵项对 σ 的推力已经和"非死亡任务信号"同量级甚至更大（0.01 vs 0.007）。

**D. 与 §7 的合流：为什么 strike_opportunity_count=0 不是"还早"，而是被三件事共同锁死**

init_noise_std=0.02 的探索 + racket_position σ=0.075 m 的核（0.3 m 误差处 exp(-16)=1.1e-7，梯度精确为零）+ 唯一的远场梯度 racket_progress（clamp 后上限 +0.03/步 post-dt）。也就是说：**没有任何机制能把拍子送进核的有效带**，同时"进不去"这件事本身还在被 -72 的尾巴反复教育成"别乱动"。§13 的 NO_OPPORTUNITY_CONTINUE 契约在流程上正确（不许用这个窗口下结论），但它不能豁免这个结构诊断——诊断依据是核函数的解析形状，不是 outcome 统计。

**E. 顺带核实到的两处漂移（不改结论，但要修）**

- -3600 这个值**不在** hope_env_cfg.py:2108-2120（那里的类默认是 0.0），真正生效点是 scripts/train.py:8896 的 v2 DIRECT 表和 train.py:11171-11184 消费的 yaml 键 `death_penalty_weight`。两处都写 -3600，结论不变。
- train.py:~11172 的中文注释写"包 direct 写 -1800 在前"，与 DIRECT 表实际的 -3600 不符，属仓库内注释漂移。
- 权重跨度的分母要用**活值**：joint_torques 的 -1e-5 只是 dataclass 默认，ActionBall 经 `defaults: [HOPEPingPongHitter@_here_]` 继承到 **-3e-5**（HOPEPingPongHitter.yaml:235）。真实跨度 1648.8/3e-5 = **5.5e7 ≈ 7.7 个数量级**，不是 8.2；这让我们和 unitree 的 7.6 基本持平，而不是明显更宽。

### 16.2 问题清单(按严重度)

- **[critical] 死亡尖峰 -72.0 支配整个回报分布：一次死亡 = 10 倍整段 episode 正收入 = 12 倍折扣视野收入**
  - 机制:γ=0.99 的折扣视野是 100 步，而平均 episode 只有 124.11 步，所以死亡的影响覆盖几乎整段轨迹：以死亡结尾的 episode 里每个状态的回报都是 -72×0.99^k ∈ [-72,-21]，dense 收入 7.4 只是零头。回报分布因此是双峰的。全 batch advantage 归一化统一的是整体尺度、不改变项间比例，于是 σ_A 被死亡尾巴单独决定（推导值 10~20），dense 项的归一化 advantage 被压到 ≈0.007，死亡方向 ≈3，相差约 500 倍。PPO 的 clip=0.2 与 desired_kl=0.01 这两个'策略移动预算'几乎全花在死亡方向上。
  - 证据:weight -3600 × dt 0.02 = -72.0/次（生效点 train.py:8896 与 train.py:11171-11184，非 hope_env_cfg.py:2108-2120）；mean_episode_length=124.11、dense 0.03~0.09/步（docs/PROGRESS.md 2026-07-30）；快照终止占比 3058/31505/15393 / 98304 = 3.1%/32.0%/15.7%（configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json）；三库最大单步罚项 post-dt ≈ -0.2，仅为其视野收入 9.5 的 2%
- **[critical] §7 死核：racket_position σ=0.075 m 在真实误差带梯度精确为零，导致'击中'这一层的实际收入恒等于 0，三层分层塌成两层**
  - 机制:exp(-(e/σ)²) 在 e=0.30 m、σ=0.075 时 = exp(-16) = 1.1e-7；e=0.15 m 时 = exp(-4) = 0.018。核只在 e≲0.10 m 内有活梯度。而唯一的远场梯度是 racket_progress（clamp ±0.15 m/步 × 权重 10 × dt = 上限 +0.03/步），加上 init_noise_std=0.02 的极小探索，策略没有任何机制把拍子送进核内。于是'模仿 < 击中 < 质量'的中间层收入恒为 0，只剩模仿收入和不死。
  - 证据:HOPEPingPongActionBall.yaml:114 racket_position_std=0.075，比父配方 HOPEPingPongHitter.yaml:200 已经'再收紧过'的 0.15 还小一半，而父配方权重是 14.0 我们只有 4.0；仓库自身反复记录过同一病：hold_ready 'std 0.5 → 核死 0.002，Episode_Reward ~1/100 of expected'、racket_velocity_std 1.8→1.2→0.8→0.5 的加宽-再课程收紧、base_position 'Do NOT tighten below ~0.18: at std 0.15 the kernel is dead (8e-4)'；strike_opportunity_count=0 × 3 个 profile
- **[high] value_loss ∝ scale² 且活跃路径无任何 return/value 归一化，critic 初始化也不看 scale**
  - 机制:value_loss = mean((V-R)²) × value_loss_coef 1.0，returns 是 (weight×dt) 原始回报的折扣和，全程无归一化：RolloutStorage 无 normalizer 字段、ActorCritic.evaluate 是裸 MLP、OnPolicyRunner 只给 obs/privileged_obs 包 EmpiricalNormalization。critic 用 PyTorch 默认 nn.Linear 初始化（正交缩放 init 的 helper 存在但源码注释写明 not used at the moment），初值≈0 面对 O(-70) 的真实回报。GAE 的 delta 直接吃 critic 裸输出，所以尺度问题先污染 critic、再经 advantage 污染 actor。
  - 证据:死亡附近单样本平方误差 ~72²=5184 vs dense 步 ~0.01，差 5~6 个数量级；value_loss_coef=1.0（hope_training/whole_body_tracking/cfg/algo/ppo.yaml:62）；库内唯一的 reward normalizer（EmpiricalDiscountedVariationNormalization）只服务 RND，我们从不传 rnd_cfg，确认失活
- **[high] 罚项层可以单步压过全部正收入——'软罚压制击球'事故的结构性复发条件**
  - 机制:qdes/actual barrier 权重 -40，带 0.25 非零地板：单个越界关节、单通道、单步 = -40×0.25×0.02 = -0.20，最深 -0.80。而整步 dense 正收入只有 +0.03~+0.09。也就是说一个关节轻微蹭限，一步就吃掉当步全部收入的 2~7 倍；理论最坏 31 关节×2 通道 = -49.6/步。策略最省事的解法是把动作幅度整体收回来，正好压掉挥拍。
  - 证据:HOPEPingPongActionBall.yaml:141,144 及其自带算式注释（-0.20 地板 / -0.80 满深 / -49.6 单步上限）；三库 joint_limit 一律 -10.0 且是纯越限尾部（无非零地板）；MEMORY 记录的历史事故'软罚压制击球'
- **[high] actor/critic 共用一次全局 grad-norm 裁剪（弱版本成立），且 LR 由共享的 adaptive-KL 控制器决定**
  - 机制:一个 ActorCritic nn.Module、一个 Adam、一次 backward、一次 clip_grad_norm_(policy.parameters(), 1.0)。注意：裁剪是对整个梯度向量乘同一标量，不改变 actor/critic 分量的相对方向；Adam 的 per-parameter 归一化还会抵消一个恒定公共因子——所以'critic 恒定饿死 actor'不成立。真实残留伤害是两条：(i) 裁剪因子随 batch 里死亡事件数波动（终止占比在快照间从 3.1% 跳到 32.0%），给 actor 注入批间乘性噪声，Adam 的二阶矩记忆（≈50 个 update）跟不上；(ii) LR 是共享的，KL 预算几乎全被死亡方向占满，想给 dense 层提速就必然同时给 critic 提速。
  - 证据:rsl_rl 2.3.1: ppo.py:96 单 Adam、324 合并 loss、372-373 单 backward、385 单裁剪、386 单 step；actor_critic.py:15 单 nn.Module。⚠ 全仓库 grep 不到任何 rsl-rl 版本 pin，diligence doc 自己把'rsl_rl 版本写进 receipt'列为未完成 TODO；若实际是 5.x（分开裁剪）本条不成立
- **[high] 熵项对 σ 的推力已经和'非死亡任务信号'同量级，σ 实际由死亡和熵共同决定，与击球质量无关**
  - 机制:init_noise_std=0.02、31 关节：每维熵 = 0.5·ln(2πe·0.0004) = -2.494 nat，合计 -77.3 nat；entropy_coef=0.01 对每个 log σ 是恒定 -0.01 的上推梯度。surrogate 对 log σ 的梯度 ≈ 归一化 advantage × O(1)：dense 侧 ≈0.007（推不过熵），死亡侧 ≈3（狠压 σ）。所以探索幅度的实际控制权在熵常数和死亡尾巴手上，任务质量完全插不上话。
  - 证据:cfg/algo/ppo.yaml:50 entropy_coef=0.01（已是三库 0.005 的 2 倍）；init_noise_std=0.02 由 train.py:5004(**§21 核查修正:零初始化末层实际在 scripts/train.py:7010/7062/7065,调用点 16237;5004 是 init_noise_std 的强制校验点**) 与 training_contract.py:4041-4051 硬性锁死（!=0.02 直接报错）；C1 推导出的归一化 advantage 量级 0.007 vs 3
- **[medium] 权重跨度 7.7 个数量级，但真正的病不是跨度而是'最大项是尖峰不是 dense 锚'**
  - 机制:跨度本身不是病：unitree_rl_lab 跨度 7.6 个数量级也照样能训，因为它的极端来自把 joint_acc 刻意做成 2.5e-7 的'已知安全常数'（全任务族复用），最大项仍是 -10.0 的常规罚项。我们的极端方向相反：最大项 1648.8 是一个只在事件上发放的巨型尖峰，dense 锚反而是最小的一档。BeyondMimic/mjlab 都刻意把跨度压在 2 个数量级内。
  - 证据:我们 1648.8 / 3e-5 = 5.5e7 ≈ 7.7 数量级（注意分母必须用活值 -3e-5，来自 HOPEPingPongHitter.yaml:235 经 defaults 继承，不是 dataclass 默认的 -1e-5）；BeyondMimic 10.0/0.1 = 100x；mjlab 10.0/0.1 = 100x；unitree 10.0/2.5e-7 = 4e7
- **[medium] v2 校准表 393.4/295.1/229.5 不只是'时机未到'，而是与 O(0.1)/步的经济数值上永久不兼容**
  - 机制:393.4 × dt 0.02 = 7.87/步，单项就是三库整层 dense（0.095/步）的 83 倍。§7 已警告在修 σ 前激活会给死核乘 ~100 造成收入悬崖；但即使 σ 修好，这张表在核值 0.5 时单项就付 3.9/步，仍然会把整个经济掀翻。
  - 证据:scripts/train.py ~8847-8880 的 _REWARD_PACK_V2_CALIBRATED 表及其 silent no-op 说明；HOPEPingPongActionBall.yaml:96-99 明写'deliberately let these explicit low tracking weights win'；三库 windowed/dense 层权重和一律 ≈5.0 raw
- **[low] reward 无 NaN/Inf 兜底；train.py 注释与 DIRECT 表漂移**
  - 机制:IsaacLab RewardManager.compute() 直接把 func×weight×dt 累加，没有任何 nan_to_num；mjlab 在同一行之后加了 torch.nan_to_num(nan=0, posinf=0, neginf=0) 并注明 'to avoid policy crash'。以我们 -72/-49.6 这种量级，一次 NaN 会直接毁掉 critic 与整个 batch 的 advantage 归一化（σ 变 NaN）。另 train.py:~11172 注释称 DIRECT 表写 -1800，实际是 -3600。
  - 证据:isaaclab reward_manager.py:150 无兜底；mjlab reward_manager.py:127,129 有；train.py:8896 vs 其上方注释

### 16.3 建议(排序;全部走新臂键/A-B,默认字节等价)

**S1(P0（在任何权重改动之前）):【先记账，别先动刀】把判断量级所需的六个数写进每次 run 的 receipt/日志：①每个 reward term 的 Episode_Reward 占比（IsaacLab RewardManager 本来就在累计 per-term episode sums，取出来即可）②raw return 的 mean/std ③explained_variance ④PPO clip fraction ⑤裁剪前的 grad norm ⑥pod 上 rsl_rl 的实际版本号。判据直接用仓库自己发明过的那条：某项 Episode_Reward 只有预期的 ~1/100 即判定为死核。**
- 先例:仓库自身：HOPEPingPongHitter.yaml hold_ready 注释用 'Episode_Reward ~0.013/s, ~1/100 of expected' 判定核死并据此改 σ；mjlab 是三库中唯一有 reward scale 一等文档（docs/source/rewards.rst 'Reward scaling by dt'）的
- 落点:/Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py（日志）+ training receipt schema
- 风险:无。纯观测，不改激励

**S2(P0):【答 (b) 的前半：先修 σ，再动尖峰，顺序不能反】racket_position_std 0.075 → 0.20 起步，按 pos_err 中位数做课程收紧 0.20→0.15→0.10→0.075；或更稳的做法是加一层'外壳核'（coarse σ=0.30 m 权重 1.0 + 保留 fine σ=0.075 m 权重 4.0），保住精度层的同时在 0.2~0.5 m 处给出非零梯度。racket_velocity_std=0.5 同理复核（父配方的历史是 1.8→1.2→0.8→0.5 的课程，不是一步到位）。这是把'击中'这一层从恒等于 0 变成真实收入的唯一办法。**
- 先例:仓库自身三例：racket_velocity_std 'WIDEN to BRACKET the ~2 m/s error, then curriculum-tighten 1.8→1.2→0.8→0.5'、hold_ready std 0.5→1.5（死核 0.002）、base_position 'Do NOT tighten below ~0.18: at std 0.15 the kernel is dead (8e-4)'。外部：BeyondMimic/mjlab/unitree 的 body_pos σ 一律 0.30 m（比我们宽 4 倍），全部按'你实际会看到的误差尺度'定 σ
- 落点:/Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBall.yaml:113-118（racket_position_std / racket_velocity_std / racket_normal_std）
- 风险:加宽会重新打开仓库 2026-07-03 记录过的 'box-center 懒解'（打盒子中心也能拿 0.63 的钱，pos_err 卡在 0.14-0.19）。所以必须是课程而非永久加宽，且质量层要由 virtual_landing 兜底；建议把收紧触发条件写死成 pos_err 中位数阈值，不靠人盯

**S3(P0):【答 (a)：目标量级】把整个经济按'折扣视野收入'H = 100 步 × dense 锚来定价，写进 doc 当硬约束：①dense 锚（非事件步总和）+0.05~+0.10/步 post-dt（raw 2.5~5.0）——我们现在 0.03~0.09，与三库同档，不动；②任何单步罚项的常态总和 ≤ 0.3× dense 锚（≈-0.03/步），峰值 ≤ -0.3/步；③任何事件尖峰 ≤ 1×H ≈ +6~10 post-dt；④三层目标（post-dt）：模仿 ≈5/episode < 击中（窗内 task 核）≈2~4/swing < 质量（落点）≈6/swing。理由：三库三家的 dense 锚一致是 raw 5.0 / post-dt 0.1，最大单步事件只占 H 的 2%；我们现在是 1200%，差约 600 倍，而 8 个数量级的权重跨度本身不是病（unitree 也 7.6 个数量级），病在最大项是尖峰而非锚。**
- 先例:BeyondMimic / mjlab / unitree 三家 dense 层权重和都恰好是 0.5+0.5+1+1+1+1=5.0（raw），post-dt 0.09~0.10；mjlab 文档明写 dt 缩放的目的是让 episode 总回报对仿真频率不变
- 落点:新增/更新 docs 里的 reward 量级准绳条目；数值落到 /Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBall.yaml
- 风险:约束写死后，后续任何'加个罚项试试'都要过这道算术关，会拖慢临时实验；这正是要的效果

**S4(P0（必须与 R2 同批，且 R2 在前）):【答 (b) 的后半：尖峰降一个数量级，并把比例从 2.2:1 改成 1:1】death_penalty_weight -3600 → -300（post-dt -6.0 = 1.0×H），virtual_landing_weight 1648.8 → 300（post-dt +6.0 = 1.0×H）。量级降 10 倍是主菜；比例也要改：72:33 = 2.2:1 意味着挥拍的盈亏平衡命中率 = 69%，from-scratch 策略永远够不到，等于把'不挥拍'定义成最优解——改成 1:1 后门槛降到 50%。落点仍是一次性事件，不改成逐步记账（逐步记账会奖励'处在好状态'而非'打成'，是漏点）。**
- 先例:三库全部没有 death penalty 这一项——终止的代价就是失去未来收入（隐含 ≈1×H）。我们保留显式一项但把它压回 1×H，等于'比三库严一点点、但不再是 12×H'
- 落点:HOPEPingPongActionBall.yaml:119(virtual_landing_weight) 与 :125(death_penalty_weight)；同时改 /Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/scripts/train.py:8896 的 _REWARD_PACK_V2_DIRECT，否则包默认与 yaml 会分叉
- 风险:死亡变便宜 → 摔倒/上桌率回升。缓解：barrier 层保留（它才是密集安全信号）；必须作为 A/B 臂发，不做全局改；建议同批再挂一个 death=0 的'纯截断'臂对照三库惯例。另注意 R2 未做时单独做本条会更糟（尖峰小了但击中层仍是 0 收入，等于只剩模仿）

**S5(P1):【最便宜的方差修复】把落点奖励摊到 settle 窗口发放（virtual_landing_settle_delay_s 目前 0.0，base_frac 0.6 已经把它拆成 base+quality 两块）：同样的 episode 积分，单步尖峰除以 N。这是唯一'完全不改激励、只改回报方差'的动作，对 value_loss ∝ scale²、对 batch σ_A、对 grad-norm 波动三条通道同时减压。**
- 先例:理论：回报方差与单步尖峰幅度平方成正比；工程：mjlab 在 reward manager 里对每个 term 做 nan_to_num 说明这一层的数值卫生是被同行当回事的
- 落点:HOPEPingPongActionBall.yaml:120-121（virtual_landing_base_frac / settle_delay_s）+ 对应的 hope_rewards 发放逻辑
- 风险:需要确认发放窗口内提前终止不会吞掉尾款——若会，必须在终止那一步一次性补齐，否则等于偷偷降权

**S6(P1):【罚项剂量对齐三库】qdes_limit_barrier_weight / joint_limit_weight 由 -40 → -10，并把 -10 作为低剂量臂加进已有的 -20/-40/-80 预登记消融。-10 时单关节单通道地板 = -0.05/步、满深 -0.20/步，相对 dense 锚 0.06 才是'能感觉到但压不死'的量。注意我们的 barrier 带 0.25 非零地板，同权重下本来就严于三库的纯越限尾部实现，所以 -10 并不等于放水。**
- 先例:BeyondMimic / mjlab / unitree 三家的 joint_limit 全部是 -10.0，一个不差；仓库自身也有先例：pre_strike_foot_slip -0.4→-0.2 的理由正是'双计的 -0.9/(m/s) 在惩罚一个还在学走路的策略'
- 落点:HOPEPingPongActionBall.yaml:141(qdes_limit_barrier_weight) 与 :144(joint_limit_weight)
- 风险:qdes 越界可能回升（现在是 0）。但 update250 的证据显示硬复位已从 1420 掉到 116 且从腰迁到踝，安全侧有余量。仍必须是 A/B 臂

**S7(P1):【明令封杀 393.4/295.1/229.5，不是缓期执行】§7 的警告是'修 σ 前别开'，我建议升级为'任何 σ 下都不要原样开'：393.4×0.02 = 7.87/步，单项就是三库整层 dense 的 83 倍；即便 σ 修好、核值只有 0.5，单项也付 3.9/步。要提高 task 层权重，上限按 'windowed task 层权重和 ≤ 15（post-dt 总额 ≤ 0.3/步 ≈ 3× dense 锚）' 推，即从现在的 4.0+0.5+0.5=5.0 最多提到 ~15，不是 ~918。**
- 先例:§7 自身的收入悬崖警告；三库 dense+windowed 层权重和一律 5.0；scripts/train.py 自己的 _calibrated_override_marker 注释已把这张表标为 silent no-op
- 落点:/Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/scripts/train.py:8847-8880（_REWARD_PACK_V2_CALIBRATED）——建议直接删表或改成会报错的哨兵，别留着当陷阱
- 风险:无（这是'别做'）。唯一成本是要同步改依赖这张表的文档叙述

**S8(P1):【把 rsl_rl 版本钉进 receipt】本仓库 grep 不到任何 rsl-rl 版本 pin（*.py/*.toml/*.txt/Dockerfile 全空），而'actor/critic 共用一次 grad-norm 裁剪'这条结论完全依赖 2.3.1。IsaacLab 自己的 pin 在版本间是会变的（IsaacLab 2.1.0 → rsl-rl-lib==2.3.1；本地另一份 IsaacLab VERSION=3.0.0 → rsl-rl-lib==5.0.1），5.x 是 actor 与 critic 各自独立裁剪。先在 pod 上跑一次 `python -c 'import rsl_rl; print(rsl_rl.__version__)'` 并写进 receipt。**
- 先例:diligence doc 自己把'rsl_rl 版本写进 training receipt'列为未完成 TODO；外部对照里 unitree_rl_lab 的裁剪拓扑正是因为 README 标 IsaacLab 2.3.0（无本地对应 checkout）而无法判定，只能标 UNRESOLVED
- 落点:training receipt schema + /Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/training_contract.py（那里目前只钉了 Isaac Lab 2.1，完全没提 rsl_rl）
- 风险:无。若查出是 5.x，则 scale_issues 里的'共用裁剪'一条直接降级——这本身就是有价值的结论

**S9(P2（但结论要现在写进 doc，防止有人把 entropy 当 scale 旋钮）):【答 (c)：entropy_coef 不动，init_noise_std 也不动】保持 entropy_coef=0.01（已是三库 0.005 的 2 倍），init_noise_std 保持 0.02。理由要说清楚：熵项的值和梯度对 reward 尺度严格不变（纯 σ 的函数），所以'用 entropy 补偿 scale 问题'是错配的旋钮。真实耦合是间接的——熵对每个 log σ 是恒定 +0.01 的上推，而 surrogate 那侧对 σ 的梯度被死亡尾巴主导（dense 侧归一化 advantage ≈0.007，死亡侧 ≈3）。换句话说现在探索幅度是被'熵常数'和'死亡'合议决定的，任务质量插不上话，而且熵的推力已经压过非死亡任务信号（0.01 vs 0.007）——此时再调大 entropy_coef 就是纯注噪，在 31 维、Adam 逐参数归一化下还可能让 log σ 每个 update 涨 ~2%（约 35 个 update 翻倍），直接破坏 shared-ready 的安全起步性质。正确顺序：先做 R2+R4 修尾，然后看 log std / mean action std 的轨迹；只有在修尾之后 σ 仍单调下滑到 <0.02 时，才做 0.01 vs 0.02 的同批 A/B，且按 ppo.yaml:50 自己的规矩用 task 级 algo: override 钉，绝不改全局默认。**
- 先例:三库一致用 entropy_coef=0.005（我们已经翻倍）；cfg/algo/ppo.yaml:50 的历史注释本身就记载过'全局改 0.015 被审计撤回、同日回退 0.01'的教训并规定了 task 级 override + 同批 A/B 的做法
- 落点:/Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/cfg/algo/ppo.yaml:50（保持不动）；init_noise_std=0.02 由 train.py:5004(**§21 核查修正:零初始化末层实际在 scripts/train.py:7010/7062/7065,调用点 16237;5004 是 init_noise_std 的强制校验点**) 与 training_contract.py:4041-4051 硬锁（!=0.02 直接报错），要动就得先拆这道 shared-ready bootstrap 护栏，不建议
- 风险:不动的风险：若重尾修完仍探索不足，会多花一个 A/B 周期。动的风险大得多：σ 膨胀会同时破坏 deploy-parity 的 clamp 假设和安全起步

**S10(P2):【数值卫生 + 注释除锈】①给 reward 加 nan_to_num 兜底（我们这边 IsaacLab 的 RewardManager 没有）；②修 train.py:~11172 那句写着 '包 direct 写 -1800' 的注释，实际 DIRECT 表是 -3600；③把 -3600 的真实生效点写清楚（train.py:8896 / train.py:11171-11184），hope_env_cfg.py:2108-2120 那里的类默认是 0.0，别再当引用出处。**
- 先例:mjlab reward_manager.py:129 的 torch.nan_to_num(nan=0, posinf=0, neginf=0)，注释写明 'to avoid policy crash'；IsaacLab 的同一函数没有
- 落点:reward 计算侧的包装层；/Users/Franco/Dropbox/乒乓/nohope/hope_training/whole_body_tracking/scripts/train.py:~11172 与 :8896
- 风险:nan_to_num 会掩盖真实 NaN 源，需同时打一条 WARN 进 summary（符合仓库既有的『WARN 必进摘要』规矩）

### 16.4 量级对照表

| 维度 | 我们（ActionBall + reward_pack v2） | BeyondMimic | mjlab | unitree_rl_lab（mimic dance_102） | 厂商 instinct_mj |
|---|---|---|---|---|---|
| dt 约定 | IsaacLab `raw×weight×dt`，dt=0.02 s（50 Hz），每项无豁免 | 同（IsaacLab reward_manager.py:148），dt=0.02 s | 同 + `nan_to_num` 兜底（reward_manager.py:127,129），dt=0.02 s；唯一有 dt 缩放一等文档的库 | 同（IsaacLab），dt=0.02 s | **未知**（自研 MuJoCo，非 mjlab 非 IsaacLab，无法克隆核实；不能假设有 ×dt） |
| 最大单项权重 | **1648.8**（virtual_landing）；次高 -3600（death，事件项） | 10.0（joint_limit） | 10.0（joint_limit / self_collisions 并列） | 10.0（joint_limit） | -8.0（volume_points_penetration，parkour 专用）/ 常规最大 -3.0（pelvis_orientation_l2）；**判别器与任务奖励权重未知** |
| 最小活权重 | **-3e-5**（joint_torques，经 defaults 继承自 Hitter；dataclass 默认 -1e-5 是死值） | 0.1 | 0.1 | 2.5e-7（joint_acc，全任务族复用的"已知安全常数"） | 1e-5（soft_landing） |
| 权重跨度（数量级） | **≈7.7**（5.5e7）——极端来自"把一项做大" | **2**（100×） | **2**（100×） | ≈7.6（4e7）——极端来自"把一项做到极小的安全常数"，方向与我们相反 | ≥5.5（3e5，仅罚项之间；含判别器后无法计算） |
| 典型每步总量（post-dt，无事件） | **+0.03 ~ +0.09** | +0.09 ~ +0.10 | +0.09 ~ +0.10 | +0.09 ~ +0.10 | 未知 |
| 最大单步尖峰（post-dt） | **-72.0**（death，一次性）/ **+32.98**（landing，§13 现役未发放过）= 12×/5.5× 折扣视野收入 | ≈-0.2（joint_limit 触发）= 2% 视野收入；**无 death penalty** | ≈-0.2（joint_limit / self_collisions）；**无 death penalty** | ≈-0.2；**无 death penalty** | 未知 |
| entropy_coef | **0.01**（三库的 2 倍） | 0.005 | 0.005 | 0.005 | 未知 |
| value_loss_coef | 1.0 | 1.0 | 1.0 | 1.0 | 未知 |
| grad-norm 是否 actor/critic 共用一次裁剪 | **共用一次**（rsl_rl 2.3.1：单 Adam ppo.py:96、单 backward 372-373、单 clip 385）。⚠ 本仓库无版本 pin，若实际是 5.x 则为分开裁剪 | **共用一次**（IsaacLab 2.1.0 pin rsl-rl-lib==2.3.1，与 README 徽章一致，已核实） | **分开裁剪**（pin rsl-rl-lib==5.4.0，actor/critic 各 clip 一次，各自封顶 1.0） | **未定**（README 标 IsaacLab 2.3.0，本地无对应 checkout；IsaacLab 的 rsl-rl pin 在 2.1.0→3.0.0 之间从 2.3.1 跳到 5.0.1） | 未知 |
| advantage 归一化 | 全 batch 一次（24×4096 = 98,304 样本，rollout_storage.py:167） | 全 batch 一次 | 全 batch 一次 | 全 batch 一次 | 未知 |
| reward / return 归一化 | **无**（活跃路径确认无：RolloutStorage 无 normalizer、critic 裸 MLP、只有 obs 被 EmpiricalNormalization 包；库内 RND 专用 normalizer 因从不传 rnd_cfg 而失活） | 无 | 无 | 无 | 未知（AMP 判别器收入自带尺度约束，但无源码可查） |

---

## 十七、人 vs A3 关节能力差异专项(08-01;应 Franco"哪些关节差最多/是不是靠小臂发力"之问)

**方法**:三路证据交叉——(a) A3 侧从 URDF/MJCF 第一性硬算(逐关节远端惯量+杠杆臂+α=τ/I,
经 5 条对抗核查);(b) 人体乒乓生物力学文献(WebFetch 三次挂死后改用 curl + Europe PMC/PubMed
JSON API,13 条发现全部有 PMID/DOI,注明全文读到还是仅摘要);(c) **73 条真实打球动捕**
(`chingmu73_20260728`,已 retarget 到 A3 31 关节空间)的逐关节实测。
**刻意排除 bh_loop_c 族作人类基线**(Franco 指出它是按肩优先人工改造的),只作对照——
这个决定被数据完全证实(见 17.4)。

### 17.1 裁决:"人靠小臂/手肘发力"不成立,文献与实测双否

**文献侧**(Bańkosz & Winiarski 2018, JSSM, n=10 精英女子, PMC5950751, 全文):
- 正手峰值角速度:**肩带/躯干 513–579°/s > 骨盆 312–392(§19.1 核查修正,原写 321–393 系转写笔误) > 肩屈 440–473 > 肩内外旋 248–462
  > 肘屈 291–324 > 前臂旋前 209–274 > 腕屈 224–228**——躯干最快,肘腕垫底;
- 反手排序不同:**持拍臂肩内外旋 779–955°/s 为全场最高**,前臂旋前 336–371、腕屈 269–387,
  躯干只有 115–145;
- **同篇回归:预测拍速的是肩(正手肩内旋+内收;反手肩外展+肩带旋转),肘/腕虽快但非显著预测量**;
- Iino & Kojima 2016(PMID 27111711):反手前挥中 racket arm 机械能的 **65%(对上旋)/
  77%(对下旋)由躯干传入**,不是手臂自产;
- 技能差(Iino 2009/2011):高水平选手**更多**用躯干轴向旋转、肩内旋力矩显著更大;提速路径是
  **提高躯干→上臂能量传递率**,不是加大肘/前臂动作;
- 肘腕的真实职能(Iino 2008;UCM 分析 2017):**方向分解与触球瞬间拍面角度/时机控制**
  (旋转与落点),反手中肘伸展对**向上**拍速甚至是**负**贡献;
- 人类拍速参考:正手 VRmax 13.1–14.7 m/s；反手旧写 11.6–14.1 m/s 混用了 VRcont 与 VRmax，
  不能与正手带直接比较。已核实 BH1 的 VRmax 是 12.55 m/s，统一口径的完整反手带须按原表重列。

**73 动作实测侧**(拍速峰值帧的 |ω_j|×r_j 贡献分解):

| 关节 | Pooled | FH | BH |
|---|---|---|---|
| right_shoulder_pitch | 21.3% | 8.0% | **24.5%** |
| right_elbow | 18.6% | 12.4% | 20.1% |
| right_shoulder_yaw | 18.4% | 15.7% | 19.0% |
| right_wrist_roll | 9.3% | 8.2% | 9.6% |
| right_wrist_pitch | 7.6% | 10.1% | 7.0% |
| waist_yaw | 7.5% | **14.3%** | 5.8% |
| right_shoulder_roll | 7.3% | **18.2%** | 4.8% |
| 其余(waist_pitch/roll、wrist_yaw) | ~10% | ~13% | ~9% |

**结论:自然打法是分布式动力链——肩 yaw+肩 pitch+肘 合计 60–71%,没有单一"甩鞭"关节;
肘约 19% 确实重要但不主导,腕三轴合计约 20% 主要管拍面控制。**

### 17.2 A3 的两个瓶颈,分处两端(且都不是"小臂不够")

| 维度 | 最紧的关节 | 数字 |
|---|---|---|
| **速度**(末端) | right_**wrist_pitch** | 峰值 531°/s = 上限 **73%**;其次 shoulder_pitch 68%、wrist_roll/elbow/shoulder_yaw 59–61% |
| **力矩**(近端) | **waist_roll** | tau_proxy p90 = 预算 **86%**,3/73 片段超限(1.01–1.08×);因为它要拖动整个上身+摆臂 **I≈2.9 kg·m²** 却只有 **46 N·m**(waist_yaw 有 220、waist_pitch 118) |
| 加速度/甩鞭担当 | elbow、wrist_roll、shoulder_yaw、shoulder_pitch、wrist_pitch | 两种平滑法交叉确认(Spearman 0.92) |

A3 侧硬算的补充事实:**质量不是向末端堆积**(前臂+腕+拍 2.05 kg vs 上臂 2.91 kg,比 0.70,
形状类人),所以"A3 因远端太重只能靠大臂"不成立;`α=τ/I_distal` 是肩 114–127、肘 210、
腕 pitch 280、腕 yaw 513、腰 roll 仅 **18** rad/s²——**扭矩最大的肩因为要甩整条下游链,
角加速度反而最低**;肩到拍心杠杆 0.52–0.68 m vs 腕 0.21–0.45 m,所以**肩赢在线速度/角速度、
肘腕赢在加速度预算,肘处在甜点**。

### 17.3 最重要的发现:反手弱是**示范数据自带的天花板**

- 73 库拍速:**FH 中位 5.50 m/s、BH 中位 2.39 m/s(仅 43%)**;
- BH 少用躯干:waist_yaw 5.8%(FH 的 2/5)、shoulder_roll 4.8%(FH 的 1/4),更靠肩 pitch+肘的
  "关起来"动作——**与文献的反手模式一致**(躯干贡献小、肩内外旋与前臂/腕更活跃);
- 库本身 **FH 14 条 vs BH 59 条**,严重不均衡(汇总统计偏 BH)。

**含义**:我们观察到的反手弱(以及 SMASH 硬件测试 66.7% vs 38.9%)**部分是示范数据决定的,
不能指望 policy 凭空学出示范里没有的躯干参与度**。要提反手,要么补"多用腰"的反手示范,
要么 reward 在反手相位主动奖励 waist_yaw/shoulder_roll 参与——这是可直接落到 §16 收入分层里
的一条设计结论。

### 17.4 bh_loop_c 对照:肩优先改造把负载挤进了最弱的执行器

| 关节 | bh_loop_c(改造) | 73 库中位 | 倍数 |
|---|---|---|---|
| shoulder_roll | 21.6% | 5.1% | **4.3×** |
| **wrist_yaw** | **14.9%** | **2.7%** | **5.5×** |
| shoulder_yaw | 3.5% | 21.2% | **1/6** |
| shoulder_pitch | 22.7% | 20.7% | ~1.1× |

改造版把 shoulder_yaw 峰值速度压到 **54°/s(自然 BH 中位 465°/s 的 1/9)**,却靠
**wrist_yaw 344°/s(自然 1.5×)** 硬凑出 5.38 m/s 拍速。

**推论(新,值得单独验证)**:`wrist_pitch/yaw` 是全机**力矩预算最小的执行器(6 N·m)**,
而肩优先改造恰恰把工作量挤了进去——这与 **V12 r8 腕关节贴限需豁免**、以及腕 pitch 速度
占比 73% 为全机最高,构成一条自洽的因果链:**不是人类打法需要腕发力,是我们的改造动作
逼腕去补被压掉的肩旋转。** 现役 N1 就跑在这条 clip 上。

### 17.5 行动项

| 方向 | 结论与动作 |
|---|---|
| 动作库 | ①**别再往"小臂驱动"方向改**(文献+实测双否);②肩优先改造的代价已量化,建议重估 bh_loop_c 的关节配比,或至少把"腕 yaw 5.5 倍超载"作为已知副作用记档;③反手要补"多用腰"的示范 |
| 自动可行性检出 | 73 条里 **3 条**(全部 BH、全部 waist_roll 1.01–1.08×)在 savgol 口径下力矩超限;raw 口径 40 条是二次差分噪声假阳性(平滑越强收敛到 3)。建议把 **savgol + tau_proxy/effort** 做成动作准入的自动门 |
| 硬件反馈给智元 | **waist_roll 的 46 N·m 是全机相对负载最欠配的执行器**(拖 2.9 kg·m² 却只有 waist_yaw 的 1/5 预算,p90 已用 86%);其次是 wrist_pitch 的速度上限。这两条是"每 N·m 买最多击球质量"的答案 |
| 提速预警 | 这批动捕是**演示强度**(FH 中位 5.5 m/s vs 职业 13–15 m/s);若要打到职业级(~2.4×),按线性外推 **wrist_pitch(73%)与 shoulder_pitch(68%)会直接超速度上限**——"现在余量够"不能外推到目标速度 |
| reward/curriculum | 反手相位鼓励躯干参与(waist_yaw/shoulder_roll);§16 的收入分层可据此加一条"反手躯干参与"通道,但须按 §16 量级准绳定价,不得压制击球 |

**方法学警示(踩坑记录)**:npz 的 31 列关节顺序**不是** `agibot_a3.py` 的 `AGIBOT_A3_JOINT_NAMES`
(那是 GMR/CSV 源顺序),而是 `configs/a3_runtime_articulation_joint_order.txt` 的 Isaac 运行顺序;
按错顺序读会把 waist_yaw 与 left_hip_pitch 完全搞反。本次已用体旋转一致性数值验证锁定
(2.089°=2.089° 对齐,错序则读到 −48°)。仓内已有 `a3_joint_order_bijection_v1.json` +
契约测试防漂移——**任何新的离线动作分析都必须先过这道门**。


---

## 十八、关节负载普查(多维)与提速物理学(08-02;应 Franco"最大/平均负载各占多少"与"职业速度不是线性外推"之问)

**方法**:复用 §17 的 73 动作管线(关节顺序契约、动态 I_distal、savgol 口径),新增五个负载维度
(功率、力矩-转速角点、RMS/热占空、jerk、ROM),并把力矩分母从 URDF 峰值盒子换成**扣掉重力+
干摩擦的净预算**(仓内 `source_diagonal_acceleration_envelope.json`)。2 抽取 + 2 对抗核查
(8 修正)+ 1 裁决。**注意:τ 仍是纯惯性代理,厂商实测到手后须重算。**

### 18.1 裁决

一句话:73 库整体只用掉 A3 大约一半的能力——链上关节峰值速度中位 12–60%、峰值力矩中位 10–53%、峰值机械功率从来不超过额定盒子角点的 14.1%,唯一真正贴限的是 waist_roll 的力矩;**但这个"安全"是盒子模型给的假象**。

盒子模型 vs 真实力矩-转速包络的差别(我重算过,不是推断):
- 盒子模型(effort_limit 与 velocity_limit 各自独立、都用 URDF 原始值):73 个 clip 里只有 waist_roll 在 3 个 clip 超 100%(最高 107.9%),其余 9 个链关节全部"安全",right_shoulder_pitch 最高 97.7% 从不越线。
- 换成**线性力矩-转速包络**(ω/ω_max + τ/τ_max ≤ 1,即真实 PMSM 的反电动势降额线)**并且把力矩分母换成扣掉重力+干摩擦后的净预算**(用本仓已有的 `vendor_assets/motion_finalize_20260724/evidence/acceleration/source_diagonal_acceleration_envelope.json`,同一份 URDF,waist_roll 46.0→31.87 N·m、shoulder_pitch 60.0→46.95 N·m、wrist_pitch 6.0→4.82 N·m):
  - **waist_roll 从 3/73 变成 15/73 个 clip 超限**(20.5%,最高角点 1.590),而且 FH/BH 都有。
  - **right_shoulder_pitch 从"从不超限"变成 1/73 超限**(clip `hope_Take_060_unit07_BH.npz`,角点 1.201、净力矩 1.030、同帧速度 65.6%)——这就是"盒子模型下安全、线性包络下不安全"的那一个。
  - waist_pitch 0.662→0.848、right_elbow 0.795→0.840、shoulder_roll 0.740→0.811、shoulder_yaw 0.762→0.801、wrist_pitch 0.749→0.771,全部抬进 0.77–0.85 的报警带,只剩 waist_yaw(0.27)和 wrist_yaw(0.60)真有余量。

所以结论要改写成:**在演示强度(FH 中位 5.5 m/s)下,我们已经把 waist_roll 用满了,shoulder_pitch 在最坏 clip 上刚好擦线,其余链关节余量只有 15–25% 而不是看起来的 35–40%。**"功率从不超 14.1%"这一条是真的、也是最稳的——峰值力矩和峰值速度在时间上不同帧发生(shoulder_pitch 单独看是 67.7% 速度 + 80.6% 力矩,朴素相加 148%,同帧最大只有 97.7%),说明我们从来没到真正的电机角点,瓶颈永远是单轴的。

重要前提(必须一起说):τ 用的是纯惯性代理 (armature + I_distal)·α,不含重力/科氏/摩擦;我把分母换成"净预算"正是为了让分子分母口径一致(惯性需求 vs 惯性可用预算)。这仍是估计,不是厂商实测——所以下面的 vendor_questions 是硬需求,不是 nice-to-have。

### 18.2 逐关节多维负载表(Franco 要的那张)

## 表 1:腰 + 右臂(执拍臂)10 个关节的多维负载

全部来自 73 个真实 clip(FH 14 / BH 59,50 fps)。**中位 = 典型一拍;最大 = 73 个 clip 里最坏的那一拍。**百分比 = 占该关节额定能力的比例。

| 关节 | 峰值速度%<br>中位/最大 | 平均速度%<br>全程(击球±5帧) | 峰值力矩%<br>中位/最大(盒子) | 峰值力矩%<br>最大(重力净) | RMS力矩%<br>中位/最大 | 峰值功率%<br>最大 | 平均功率%<br>中位 | 角点距离<br>盒子/净(最大) | ROM%<br>中位/最大 | 峰值jerk rad/s³<br>中位/最大 | **最紧的那一维** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| waist_yaw | 12.1 / 20.0 | 4.1 (8.3) | 10.3 / 18.8 | 19.2 | 4.1 / 7.0 | 1.7 | 0.13 | 0.27 / 0.27 | 5.7 / 12.0 | 167 / 613 | 速度 20.0%(**全链最松,是可用的躯干杠杆**) |
| **waist_roll** | 4.4 / 8.5 | 1.3 (2.3) | **53.0 / 107.9** | **155.8** | 20.6 / 50.3 | 5.2 | 0.23 | **1.111 / 1.590** | 28.4 / 50.7 | 106 / 269 | **力矩 — 全库唯一超 100%,盒子 3/73、净包络 15/73** |
| waist_pitch | 12.2 / 26.9 | 4.6 (5.1) | 26.6 / 60.5 | 79.1 | 12.0 / 20.1 | 8.7 | 0.41 | 0.66 / 0.85 | 21.8 / **69.2** | 169 / 490 | 典型=力矩;最坏=**ROM 69.2%**(有 clip 到硬限位的 90.2%) |
| right_shoulder_pitch | **59.5 / 67.7** | 18.9 (**38.9**) | 28.8 / 80.6 | **103.0** | 11.4 / 18.7 | **14.1** | **1.70** | 0.977 / **1.201** | 27.4 / 37.5 | 520 / 1796 | 典型=速度;**最坏=力矩,净包络下唯一新增的超限关节** |
| right_shoulder_roll | 21.1 / 48.8 | 5.1 (9.6) | 20.1 / 65.7 | 73.7 | 6.1 / 15.4 | 9.3 | 0.25 | 0.74 / 0.81 | 13.8 / 46.0 | 347 / 1210 | 最坏=力矩 65.7% |
| right_shoulder_yaw | 50.9 / 58.8 | 8.8 (20.0) | 22.6 / 36.6 | 43.5 | 7.1 / 12.2 | 12.4 | 0.58 | 0.76 / 0.80 | 15.4 / 25.7 | 738 / 2482 | 速度 58.8% |
| right_elbow | 57.3 / 59.5 | 14.1 (36.7) | 27.9 / 62.6 | 74.6 | 10.4 / 16.6 | 13.0 | 1.31 | 0.80 / 0.84 | **54.0 / 64.0** | 835 / 2339 | **唯一"速度+ROM 双高"关节**(中位就 57%/54%);最坏=ROM |
| right_wrist_roll | 57.9 / 61.4 | 12.2 (23.6) | 5.1 / 25.8 | 26.4 | 1.6 / 4.9 | 4.8 | 0.16 | 0.67 / 0.67 | 16.1 / 34.0 | 895 / **6110** | 速度;**全机 jerk 冠军(平滑性需求最高)** |
| **right_wrist_pitch** | **59.1 / 72.9** | 9.0 (23.6) | 8.5 / 19.8 | 24.7 | 2.8 / 6.0 | 6.0 | 0.24 | 0.75 / 0.77 | 14.8 / 22.2 | 548 / 3390 | **速度 72.9% — 全机速度冠军** |
| right_wrist_yaw | 30.0 / 57.2 | 8.7 (14.3) | 6.6 / 15.1 | 16.7 | 2.6 / 4.6 | 3.0 | 0.17 | 0.60 / 0.60 | 25.8 / 40.6 | 377 / 1705 | 速度(方差大,峰值常在引拍不在触球) |

## 表 2:按部位汇总(直接回答"每个部位最大/平均占能力百分之多少")

| 部位 | 关节数 | 速度% 平均/最坏 | 力矩% 平均/最坏 | 功率% 平均/最坏 | ROM% 平均/最坏 | 人话 |
|---|---|---|---|---|---|---|
| 腰(3) | 3 | 9.5 / 26.9 | **30.0 / 107.9** | 1.2 / 8.7 | 18.7 / 69.2 | **全身唯一超额定的部位**,靠 waist_roll 一个关节拖垮 |
| 右肩(3) | 3 | 43.9 / 67.7 | 23.9 / 80.6 | 5.3 / 14.1 | 18.8 / 46.0 | 速度和力矩都到 2/3 以上,双轴都紧 |
| 右肘(1) | 1 | 57.3 / 59.5 | 27.9 / 62.6 | 9.1 / 13.0 | **54.0 / 64.0** | 平均就用掉一半速度+一半行程,**每一拍都这么累,不是个别 clip** |
| 右腕(3) | 3 | 49.0 / 72.9 | 6.7 / 25.8 | 1.7 / 6.0 | 18.9 / 40.6 | 纯速度受限,力矩/功率几乎不构成约束 |
| 左肩/肘/腕 | 7 | ≤6.5 / ≤20.7 | ≤4.4 / ≤18.3 | ≤0.2 / ≤1.7 | ≤5.5 / ≤40.2 | 非执拍臂机械上基本闲置 |
| 腿(12) | 12 | 8.6 / **51.9**(右踝pitch) | 未计算 | 未计算 | 11.0 / 66.1 | **口径缺口:腿不在 TAU_SCOPE,力矩/功率没算过,需要补** |
| 头(2) | 2 | 0.0 / 0.0 | — | — | 0.0 / 0.0 | **数据缺陷,不是负载结论**:head_yaw/head_pitch 在全部 73 clip 每一帧恒为 0.000,像是 retarget 把头 pin 住了,要找 Jiayi 核 GMR 流程 |

## 读表的三个要点

1. **平均 vs 峰值差 3–4 倍,但击球窗口才是真正的持续负载区**。全程平均速度看着很低(肩pitch 18.9%),但只看触球前后 ±5 帧(±0.1 s),平均就翻到 38.9%——肩pitch 是整条挥拍持续吃力,wrist_pitch 是触球那一下的尖峰(全程峰值 72.9% 与击球窗峰值 72.9% 完全相同,73 个 clip 里有 65 个的全程峰值就落在触球窗内)。
2. **功率永远不是约束**。最高只有 shoulder_pitch 的 14.1%(115 W 对 816 W 角点)。峰值力矩和峰值速度从不同帧发生,所以我们从来没逼近真正的电机角点,瓶颈永远是单轴的(速度 **或** 力矩)。
3. **jerk 没有硬件限,只能排序**:腕 > 肘/肩 >> 腰,腕比腰高 10–23 倍。这是"哪个关节最需要指令平滑和控制带宽"的排序,不是超限检查。数值用双重 savgol 求的,偏保守(实际更高)。

### 18.3 提速物理学:职业速度不是线性外推

## 先直接回答 Franco 的两个问题

**Q:"×2.4 会不会真的让 wrist_pitch/shoulder_pitch 超限?"**
会,但**它们不是第一个撞墙的,而且撞墙远早于 ×2.4**。分三个口径讲清楚(这也是"击球帧速度 vs 全程峰值速度"的区别):

| 口径 | ×2.388(13.13 m/s)超速度限的关节 | ×2.672(14.69 m/s) |
|---|---|---|
| **A. 触球那一帧的瞬时角速度**(中位 FH clip) | **0 个**,最紧的 shoulder_roll 只到 78.3% | 0 个 |
| **B. 各关节自己全程峰值**(中位 FH clip) | **1 个**:shoulder_yaw 115%;wrist_pitch 97.5% 擦线,elbow 88% | **2 个**:shoulder_yaw 129%、wrist_pitch 109% |
| **C. 各关节自己全程峰值,取 14 个 FH clip 中最坏的**(异步最坏界) | **6 个**:wrist_pitch 168%、sh_pitch 159%、elbow 142%、wrist_roll 139%、sh_yaw 138%、sh_roll 117% | 7 个(wrist_yaw 加入,103%) |

这三个数字都对,但含义完全不同,**之前流传的"6/10 超限"是口径 C(每个关节各取自己最坏的那一拍、且互不同步),而 part_b_swing.py 自己默认打印的是口径 A(0/10)**。诚实的说法是口径 B:典型一拍线性提速 ×2.4,**第一个超速度限的是 right_shoulder_yaw,不是 wrist_pitch/shoulder_pitch**;wrist_pitch 正好卡在 97.5%;shoulder_pitch 只有 56%(它是最坏 clip 才爆,不是典型 clip)。所以 §17 里"速度瓶颈在 wrist_pitch/shoulder_pitch"在**全 73 库最坏界**上成立(它俩全程峰值分数最高:72.9% / 67.7%),但在**典型 FH 一拍**上第一个撞速度墙的是 shoulder_yaw。

**但速度墙根本不是先撞上的墙。** 刚性时间缩放下角加速度按 k² 涨、力矩按 k² 涨,所以力矩先爆。逐关节算"还能提速多少倍才第一次超限"(典型 FH clip,力矩用重力/摩擦净预算):

| 关节 | 速度维 k_max | 力矩维 k_max(净) | **先撞哪个** |
|---|---|---|---|
| **waist_roll** | 29.9 | **1.19**(盒子 1.43) | **力矩 — 全身最先,只能再快 19%** |
| waist_pitch | 6.7 | 1.55 | 力矩 |
| **right_shoulder_yaw** | **2.08** | 2.03 | 力矩/速度几乎同时 |
| right_shoulder_pitch | 4.25 | 2.13 | 力矩 |
| right_elbow | 2.71 | 2.19 | 力矩 |
| right_wrist_pitch | **2.45** | 3.22 | 速度 |
| right_shoulder_roll | 3.29 | 2.72 | 力矩 |
| waist_yaw | 6.47 | 2.84 | 力矩(**余量最大**) |
| wrist_roll / wrist_yaw | 4.38 / 4.66 | 8.07 / 4.44 | 速度 / 力矩 |

**全身第一个墙 = waist_roll 力矩,k = 1.19(净)/ 1.43(盒子)。** 也就是说,**保持现有动作形状、只把时间轴压缩,我们连 ×1.5 都做不到,更别说 ×2.4。**

---

**Q:"职业速度是不是线性外推?"答:不是,而且我们最大的一块提速空间根本不用碰硬件。**

拍速 = Σ|ω_j|·r_j(向量和)。提速有 4 条互相独立的路,代价差一个数量级:

**H0 — 同步(时序对齐):最便宜,而且我们几乎没用**
我实测了 14 个 FH clip 的"同步效率":触球帧各链关节的 Σ|ω_j|·r_j 中位 6.73 m/s,而如果这些关节把**它们在同一次前摆里各自已经达到过的**角速度对齐到触球帧,Σ 会是 9.6 m/s;把向量对齐系数(中位 0.818)带回去,拍速会从 **5.50 m/s → 7.85 m/s(+44%)**,**任何关节的峰值角速度、峰值力矩都不增加一分**。如果再乐观一点(允许对齐到各关节全程峰值,上界),是 10.68 m/s(×1.88)。
→ 也就是说,**职业和我们的差距里,大约一半是时序(鞭打时序把峰值错开了),不是硬件**。同步做完之后,到 13.13–14.69 m/s 还差 ×1.67–1.87,而不是 ×2.39–2.67。

**H1 — 纯线性时间缩放(同路径、缩短时间):最贵,ω∝k、α∝k²、τ∝k²**
上表已算:waist_roll 在 k=1.19 就爆。**判定:单独用这条路走不通,连 ×1.5 都到不了。**

**H2 — 加长引拍(Franco 说的"更长的加速距离"):方向对,但回报是平方根律**
ω_peak² = 2·α·Δθ。**引拍路径加长 50%,同样峰值 α 只换来 √1.5 = 1.225 倍拍速**——只覆盖了所需 2.39–2.67 倍的 46–51%。反过来,若靠 1.5 倍路径达到 ×2.39,所需 α 比是 k²/1.5 = 3.84 倍(比 5.70 好,但仍然):waist_roll 3.83×限、waist_pitch 2.20×、elbow 1.31×、sh_yaw 1.30×、sh_pitch 1.23× 全部超限。若想完全不加力矩(α 比 = 1),需要引拍路径加长 **k² = 5.7 倍**——ROM 上不可能(right_elbow 中位已用掉 54% 行程,waist_pitch 最坏 clip 到硬限位的 90.2%)。**判定:必要不充分,必须和 H0/H3 叠加。**

**H3 — 提高躯干占比:有效,但只有 waist_yaw 是对的躯干**
当前 FH 中位 waist_yaw + shoulder_roll 合计贡献 35.4%。翻倍到 70.9% 并按现比例分配:**shoulder_roll 会到自身速度限的 118–132%(超限)**,因为它力臂只有 0.339 m;而 **waist_yaw 只到 44–49%**,力臂 0.736 m、220 N·m 力矩预算、净预算余量最大(k=2.84/6.47)。→ **文献说"职业用更多躯干"要翻译成"用 waist_yaw,不是用 shoulder_roll"。** 但把整份加倍的躯干贡献全压给 waist_yaw,天花板是 8.84 m/s(waist_yaw 自身速度限),只覆盖目标的 60.2–67.3%,剩下 33–40% 仍得手臂出。**注意别顺手压给 waist_roll——它是全身唯一已超限的关节。**

---

## 综合:到职业速度的可行配方(粗算,不是承诺)

5.50 →(H0 同步,+44%,零硬件代价)→ 7.85 →(H3 把躯干贡献移到 waist_yaw,+20–30%)→ 9.4–10.2 →(H2 引拍加长 50%,×1.225)→ **11.5–12.5 m/s**,此时才需要 ×1.05–1.28 的真实时间压缩。这条路线里 waist_roll 的力矩需求增长约 (1.05–1.28)²/1.5 ≈ 0.74–1.09 倍——**刚好卡在临界**,取决于同步改造是否让 waist_roll 反而更省力。
朴素 ×2.4 需要 waist_roll 力矩 ×5.7(净预算 4.8 倍超限)。**两条路差 5 倍以上。所以答案很明确:职业速度不是线性外推,是"时序 + 杠杆 + 路径"三项先做完,时间缩放只做最后 5–28%。**

## 附:速度上限到底是什么物理原理(Franco 直接问的)
真实 PMSM+减速器:峰值力矩由**电流限**(功率器件/退磁)决定,与转速基本无关直到拐角;转速上升 → 反电动势 Ke·ω 线性上升 → 吃掉固定的母线电压裕度 → 同电流下可用力矩从堵转力矩线性掉到零(空载转速)。所以真实可用区是"电流限直线 ∩ 连续力矩(I²R 热限)直线"围出的**三角形**,不是矩形。我们的 velocity_limit_sim 是一个**独立的硬速度墙**(在 PhysX 里是 `root_view.set_dof_max_velocities()` 真刹车),它到底代表电压/反电动势极限、控制器软限、还是机械极限——**仓里没有任何数据能回答,必须问智元**(见 vendor_questions 第 8 条)。而 MuJoCo 没有等价机制,`mujoco_eval_onnx.py` 里的速度限只是 mj_step 之后的 np.clip 代理,自己标了 `proxy_is_post_integration_nonexact`、要 `--allow-inexact-contract` 且不可入账——**训练真守速度盒子,正式 MuJoCo 评测守不住,这是个已知但没解决的口径裂缝。**

### 18.4 该向智元索取的电机数据(每条注明用途)

1. 每个关节的力矩-转速包络曲线(峰值线与 S1 连续线两条,含拐角转速与母线电压条件)。用途:把我们的矩形盒子模型换成真实三角形包络,重算 waist_roll(现 15/73 clip 超限)和 right_shoulder_pitch(现 1/73 超限)到底是不是真的超——这两个判定现在完全建立在我们自己的线性包络假设上。
2. 峰值力矩额定 + 允许持续时间(过载时间常数 / S2-S3 短时额定曲线),以及连续(S1)力矩额定。用途:现有 46/60/24/6 N·m 我们只知道是 URDF 里的峰值盒子,不知道能扛多久。击球是 0.1–0.3 s 的脉冲,如果峰值能扛 1 s,waist_roll 那 15 个 clip 可能全部合法;如果连续额定只有峰值的 40%,我们的 RMS/峰值 50% 就已经越线了。
3. 热模型:绕组与驱动器的热阻/热容、允许温升、以及降额曲线(vs 环境温度、vs 占空比)。用途:我们全链 duty ratio 都在 0.30–0.50、waist_roll 最坏 clip RMS 达额定 50.3%,但完全无法判断连续对打(而不是单拍)会不会热保护。runtime 只有 motor_temperature.csv 两个数,代码里自己写明'不是连续热模型'。
4. 母线电压、额定/最大转速、力矩常数 Kt、反电动势常数 Ke、相电阻 R、相电感。用途:mjlab 已经有 BuiltinDcMotorActuator(tau = Kt(V - Ke·ω)/R)可以直接吃这四个数,建出比 IsaacLab DCMotorCfg 更真的电机模型。工具已经有了,缺的只是这四个数字——这不是能力缺口,是数据缺口。
5. 减速器型号、减速比、类型(谐波/行星/RV)、效率、回差,以及减速器自身的峰值/平均/瞬时允许力矩与 L10 寿命曲线。用途:很多机器人的真瓶颈是减速器不是电机,尤其 waist_roll(46 N·m 拖 2.9 kg·m² 负载惯量)。如果减速器的瞬时允许力矩低于电机峰值,我们的 100% 线画错了位置。
6. 相电流限值、驱动器电流限值、过流保护阈值与触发/恢复逻辑。用途:真实峰值力矩上限由此决定;同时我们要知道触发保护后关节的行为(掉力矩?锁死?),这直接关系到 sim2real 上机时的安全预案。
7. 每关节实测的零速力矩偏置(重力项)、库仑摩擦、粘滞摩擦系数。用途:我们现在用 source_diagonal_acceleration_envelope.json 的估计值,据此扣掉了 waist_roll 30.7%、waist_pitch 23.6%、shoulder_pitch 21.7%、wrist_pitch 19.7% 的力矩预算——这几个扣减直接决定了'盒子安全但包络超限'的判定,必须用厂商实测替换,否则整套 net 结论是估计叠估计。
8. velocity_limit_sim(如 shoulder_pitch 13.6 rad/s、wrist_pitch 12.7 rad/s)的物理来源到底是什么:反电动势/母线电压极限、控制器软限、编码器采样带宽、还是机械/轴承极限?用途:Franco 直接问的问题。如果是控制器软限,提高母线电压或改参数就能谈;如果是反电动势物理墙,那 ×2.4 提速在 wrist_pitch/shoulder_yaw 上就是死路,必须全部靠时序+躯干杠杆解决。这一条的答案会改变整个提速路线图。
9. 编码器分辨率与类型、电流环/速度环/位置环的控制频率、指令到力矩的延迟与闭环带宽、执行器内部是否有低通滤波及其截止频率。用途:我们的 wrist_roll 峰值 jerk 到 6110 rad/s³、wrist_pitch 3390 rad/s³(全机最高),需要知道 200 Hz 的动作指令会不会被执行器内部滤波削平——这既影响动作库能不能真的复现,也影响 sim2real 的 gap 归因。
10. 每关节的硬限位与软限位角度、以及厂商在接近限位时的减速/缓冲策略(从多少度开始限速、减速斜率)。用途:waist_pitch 最坏 clip 已经到硬限位的 90.2%、right_elbow 中位就用掉 54% 行程,我们需要知道厂商的软限位从哪开始介入,否则动作库里合法的 clip 上机会被静默减速,变成学到的动作在真机上打不出来。
11. 关节的反向可驱动性(back-drivability)、机械回差,以及碰撞/冲击后的保护策略。用途:我们算出球碰撞对拍速的影响只有 0.10–1.8%(球 2.7 g 对等效质量 0.28–4.96 kg),确实可忽略;但落台、碰桌、拍框撞击是真实冲击,需要知道保护会不会误触发。
12. 每关节的转子惯量与反射到关节侧的总惯量(J_motor·i²),以及厂商推荐的负载惯量比上限。用途:核对我们从 URDF 拿的 armature 值,以及验证 waist_roll 那个 2.9 kg·m² 负载惯量是否已经超出厂商推荐的惯量比——如果是,waist_roll 的问题就不是'动作太猛'而是'选型不匹配',结论和对策完全不同。

### 18.5 行动项

**[TOPP / 重定时策略(裁定)]** 【裁定:'TOPP 到最快'不是首要目标,Franco 的判断成立】理由是量化的:拍+臂等效质量 0.28–4.96 kg 对球 2.7 g,触球时拍速只掉 0.10–1.8%,所以'把这一拍打到最快'没有物理收益。真正的目标是**触球边界值问题**:给定预测的球到达时刻与位置,求一条在关节限内的轨迹,使触球瞬间的拍速矢量 + 拍面法向命中目标。把 topp_mintime.py 从'轨迹生成目标'降级为'余量诊断器'(这条路径在硬件限内还能快多少)。注意:B3 的碰撞公式隐含球静止(自抛/发球),回击场景要按接近速度 V_拍 + |V_来球| 重算,损失会更大——先补这个计算再把'几乎不掉'写进结论。

**[动作库 / 提速路线(最高优先级)]** 【同步性改造优先于任何提速】实测:14 个 FH clip 的触球帧同步效率中位只有 53.4%。把各链关节在**同一次前摆里已经达到过的**角速度对齐到触球帧,拍速 5.50 → 7.85 m/s(+44%),任何关节的峰值速度/力矩都不增加。这是全部提速手段里唯一零硬件代价的一档,也是我们和职业差距里最大的一块。做完之后到 13.13–14.69 m/s 只差 ×1.67–1.87,而不是 ×2.39–2.67。建议先做一个诊断脚本把 73 个 clip 的同步效率排出来,低效率的 clip 优先重定时。

**[动作库 / 重定时 clamp]** 【任何时间缩放脚本必须加 waist_roll 净力矩闸门】典型 FH clip 的全身第一道墙是 waist_roll 力矩,k_max = 1.19(重力净)/ 1.43(盒子),而不是 wrist_pitch/shoulder_pitch 的速度墙(k = 2.45 / 4.25)。现在的重定时如果只查 URDF effort_limit,会放过 20.5% 的 clip。改法:分母换成 source_diagonal_acceleration_envelope.json 的 minimum_effort_margin_nm(waist_roll 31.87 N·m 而非 46.0),并改用线性包络判据 ω/ω_max + τ/τ_max ≤ 1 而不是两个独立的 ≤1。

**[动作库 / waist_roll 专项]** 【给 15 个超限 clip 挂标记】盒子模型下 3/73 超限,线性包络+重力净下 15/73(20.5%)超限,最坏 hope_Take_064_unit05_BH.npz 角点 1.590。这 15 个 clip 要么重定时降 α、要么标为不可用于提速基线。同时:bh_loop_c 的'肩优先'改造(已知把负载挤进 6 N·m 的腕)要重新核一遍是不是同时把力矩推回了 waist_roll——这是全身唯一超额定的关节,不能再加。

**[训练 clamp / 仿真口径]** 【补上 PhysX↔MuJoCo 的速度约束裂缝】velocity_limit_sim 在 PhysX 里是真刹车(root_view.set_dof_max_velocities),训练策略是在有硬速度墙的世界里学的;MuJoCo 没有等价机制,mujoco_eval_onnx.py 的代理是 mj_step 之后的 np.clip、自标 proxy_is_post_integration_nonexact、不可入账。建议:在 reward/惩罚层加一条与 PhysX 一致的速度 barrier(不改 actuator 模型),让策略在两个引擎里表现一致,而不是依赖 solver。**不要**改回 explicit/IdealPD——2026-07-02 已经试过,200 Hz 离散化过冲把反手打坏了,那个坑不用再踩。

**[reward 设计]** 【加一项触球帧同步奖励,不加提速奖励】依据上面的同步性发现:奖励项 = 触球帧各链关节 |ω_j| 相对其自身前摆峰值的比值(越接近 1 越好),而不是奖励拍速绝对值。这与既定的'收入分层:模仿 < 击中 < 质量'一致——同步属于质量层。另外按 §17 和本次数据,躯干贡献应引导到 waist_yaw(力臂 0.736 m、220 N·m、余量 k=2.84)而不是 shoulder_roll(力臂 0.339 m,躯干贡献翻倍会到自身速度限 118–132%),更不能引导到 waist_roll。

**[训练 clamp / 力矩包络]** 【把线性包络做成惩罚,不做成硬 clamp】按既定准绳(超限用 penalty 不上 tanh),在 reward 里加 max(0, ω/ω_max + τ_net/τ_max - 1) 的 barrier,而不是改 actuator 模型。这样既覆盖了盒子模型漏掉的角点区(shoulder_pitch 盒子 0.977 安全 / 净包络 1.201 超限),又不改变 action scale 的局部斜率。

**[数据质量 / 上游核查]** 【两个必须找 Jiayi 核的数据缺陷】(1) head_yaw_joint 和 head_pitch_joint 在全部 73 clip 的每一帧位置恒为 0.000 —— 这几乎肯定是 GMR retarget 把头部 DOF pin 住了,不是'头部真的零负载',不要把它当结论用。(2) 触球帧用'拍速峰值帧'代理,有 6/73 clip 存在第二个 >90% 高度、距全局峰值 >10 帧的次峰(如 hope_Take_061_unit09_BH.npz 帧 107 vs 帧 37),这几个 clip 的触球帧可能选到了引拍而不是前摆,所有击球窗口统计对它们不可信。

**[分析口径 / 补洞]** 【三个口径缺口要补】(1) 腿部完全没算力矩/功率(不在 TAU_SCOPE),但右踝 pitch 速度已到 51.9%、右膝 48.5%,需要补一次腿部力矩普查再下'腿部不吃力'的结论。(2) τ 代理是纯惯性 (armature + I_distal)·α,没有重力/科氏/摩擦项;我用净预算做分母是让口径一致的权宜,厂商实测摩擦/重力数据到手后要正式重算。(3) 把'各关节自身峰值、73 clip 取最坏'这个统计写进 part_b_swing.py,和现有的'触球帧中位'并列打印——现在脚本自己打印 0/10、记忆里写 6/10,谁读都会以为有 bug。

**[既有结论修订]** 【修 §17 的一处贡献排序错误】'肩yaw+肩pitch+肘 ≈ 60–71%' 需要复核:按 14 个 FH clip 触球帧实测,单关节贡献中位排序是 right_shoulder_roll 1.259 m/s > right_shoulder_yaw 1.098 > waist_yaw 0.955 > right_elbow 0.872 —— 肘排第四不是第一,shoulder_roll 才是最大单项贡献者。这会改变'该往哪个关节加权重'的判断,尤其因为 shoulder_roll 的力臂短、加负载最容易超自身速度限。



---

## 十九、拍速/球速、旋转-速度耦合、反解缺陷与动捕采集设计(08-02;Franco 五问)

**方法**:3 抽取(外部执行器包络 / 我方反解代码审计 / 技能与旋转文献)+ 3 对抗核查(9 修正)
+ 1 综合裁决。新增一次**独立测量**:直接从动捕球 sidecar(`chingmu_a3_units_v2/ball_ext/*.ball.npz`,
实测球 120 Hz)算 74 unit 的来球/出球速度——**不经 GMR 重定向、不经 A3 模型**。

### 19.1 Q1 拍速 vs 球速

**一句话:13.1–14.7 m/s 是拍速(racket velocity),不是球速;我们的 5.50 m/s 也是拍速,两者同口径可比。但真正的差距在球那一侧,是 4.0 倍,不是 2.4 倍。**

**(1) 文献那个数字的身份,已到全文级核实**
Bańkosz & Winiarski 2018(J Sports Sci Med 17(2):330-338, PMC5950751 / PMID 29769835)标题本身就写着 "Velocity of Table Tennis **Racket**"。PMC 的 XML 被出版商挡了,但 JSSM 是全开放期刊,PDF 可直接取(https://www.jssm.org/volume17/iss2/cap/jssm-17-330.pdf,HTTP 200)。Table 1 的 VRmax:FH1/FH2/FH3 = **13.13 / 14.63 / 14.69 m/s**——与我们仓里记的 13.1–14.7 m/s 逐位吻合。
两处口径要修:
- 反手那个 11.6–14.1 m/s 是**混口径**:下端 11.63 来自 BH1 的 VRcont(触球瞬间速度),而 BH1 的 VRmax 其实是 12.55。正手用 VRmax、反手用 VRcont 的最小值,两条带不能直接对比。应统一写 VRmax。
- `docs/research/dr_reward_external_diligence_20260731.md:1626` 的"骨盆 312–392(§19.1 核查修正,原写 321–393 系转写笔误)°/s"是转写错误,Table 2 实际是 320.63/312.21/392.45 → **312–392°/s**(同段其余 10 条关节区间全部与原表逐位一致,是孤立笔误)。

球速参照:Kidokoro 等 2025(PMID 36537568)测的是**球**,23 m/s(在过底线后 0.15 s 处测),旋转 117±29 rps。**注意:不要再把 23 ÷ 14 = 1.6 当作"拍速到球速的转换系数"**——见 q3,那 1.6 是"法向 ~8.75 m/s + 切向 ~10–12 m/s"的几何投影,不是恢复系数,而且两个数来自不同样本、不同研究,没有任何一篇同时测同一拍的拍速和球速(这是文献真空,诚实标注)。

**(2) 我们自己这一侧:新做的一次独立测量(不经过机器人模型)**
我直接从动捕的球 sidecar(`vendor_assets/chingmu73_20260728/chingmu_a3_units_v2/ball_ext/*.ball.npz`,实测球、120 Hz、HOPE 正典系)算了 74 个 unit 的来球/出球速度:
- **来球 1.16–4.90 m/s,中位 3.88 m/s**;
- **出球 2.89–7.96 m/s,中位 5.79 m/s**(FH 5.08、BH 5.82);
- 触球高度 0.11–0.35 m,中位 **0.24 m**(台面以上)。

这条很关键:它**完全不经过 GMR 重定向、不经过 A3 模型**,所以"我们的动作库很软"不是重定向伪影。职业球速 23 m/s ÷ 我们 5.79 m/s = **4.0 倍**,比拍速侧的 2.39–2.67 倍**还大**。差额说明这些示范把拍速转成球速的效率很低:用 venue 拟合的 e(u_n)(`configs/ball_physics_optitrack_20260730.yaml`,e = 0.8636·exp(−0.07033·u_n))反解,出球 5.79 / 来球 3.88 只需要 **法向拍速 2.2–2.8 m/s**,而触球帧 Σ|ω_j|·r_j 是 5.50 m/s——**只有 40–50% 的拍速落在法向上**。这是除 53.4% 同步效率之外的**第二层损失(方向/刷球角)**,以前没量过,而且两者可能是同一件事的两个投影,必须分开测。

**(3) 2.4 倍到底是"技术"还是"准备时间"?——文献能把两者分开,答案是:在我们这批数据里不是准备时间**
- **技能不等于拍速,这是最强的一条**:Iino & Kojima 2009(PMID 19746298)原文摘要逐字:"The racket speed at impact was **not significantly different** between the two player groups"(高水平 n=9 vs 中等 n=8)。显著不同的是躯干轴向旋转对拍速的贡献、s_max(拍速–时间曲线最大斜率)、以及更短的加速时间——**这三样正好就是"同步/时序"**,与我们实测 53.4% 同步效率是同一个物理量的两种测法。
- **准备时间确实是真因子,但走的是躯干这条路**:Iino & Kojima 2016(PMID 26208598,n=8)把喂球频率从 35 提到 75 球/分,触球拍速**显著下降**,机理是骨盆与上躯干轴向旋转受限。Bańkosz & Winiarski 2025(JSSM 24:311-325,n=8 国家队)在连续三板+步法条件下,关节角速度仍然很高,但 ROM 比单板文献低几十度。**结论:时间压力先吃掉躯干旋转幅度(= 我们的 waist_yaw),再吃拍速。**
- **但我们这批采集不受时间压力**:来球中位仅 3.88 m/s,是慢速多球喂球,准备时间充裕——**却依然只打出 5.79 m/s 的球**。所以我们和职业的差距不能归因于准备时间,更像 Sato 等 2025/26(DOI 10.1177/17479541251380169,10 人 × 60–100% 五档发力)里的 60% 发力档。
- **拆账(粗算)**:2.39–2.67 倍拍速差 ≈ 1.44 倍(同步 53.4%→100%,§18 已有)× 剩下 1.67–1.85 倍(发力档 + 躯干占比 + 刷球角/瞄准效率)。而球速侧 4.0 倍差里,多出来的那一截正是上面的"法向利用率只有 40–50%"。

**(4) 因此不要把"追到 13–15 m/s 拍速"当目标。** 文献自己说拍速不区分高水平与中等水平;区分他们的是 Iino/Yoshioka/Fukashiro 2017(PMID 29112886,同一批被试,UCM 分析)——高水平选手**触球瞬间垂直拍面角的关节组合方差显著更小**,更会用冗余去稳住那个角度。这与 §18 已定的"加触球帧同步奖励、不加提速奖励"完全同向,应把"稳拍面角"作为同一奖励族的第二项。

### 19.2 Q2 外部库的力矩-转速包络(三角形限制)

**一句话:有,而且是行业标准做法,但只在"显式执行器"上;我们用的 ImplicitActuator 结构上就吃不到它。缺的是 3 个厂商数字,不是缺代码。**

**(A) 我们现在是什么(已核实)**
`hope_training/.../robots/g1.py:63/101/109/117/125` 五组全是 `ImplicitActuatorCfg`(legs/feet/waist/waist_yaw/arms),只有 `effort_limit_sim` + `velocity_limit_sim`;`robots/actuator.py:11/72` 的 `DelayedImplicitActuator` 只加了一个指令延迟环形缓冲,**没有覆写 `_clip_effort`**,所以继承的还是静态盒子。IsaacLab 自己的 docstring 写死:对 implicit 执行器 `velocity_limit` **不参与力矩计算**。唯一生效的是 `torch.clip(effort, ±effort_limit)`——一个矩形,不是三角形。

**(B) 各栈的公式与"是否真在用"**

| 栈 | 公式 | 真在用? |
|---|---|---|
| IsaacLab **ImplicitActuator**(我们) | `τ = clip(τ_cmd, ±τ_lim)` 静态盒子 | 用,但**无包络** |
| IsaacLab **DCMotor** | `τ_max(ω) = clip(τ_sat·(1 − ω/ω₀), max=τ_lim)`;`τ_min(ω) = clip(τ_sat·(−1 − ω/ω₀), min=−τ_lim)`;`τ = clip(τ_cmd, τ_min, τ_max)`。**梯形**(τ_sat=τ_lim 时退化成纯三角形) | **真在用**:Unitree A1(sat=33.5=τ_lim, ω₀=21.0)、Go2(23.5/23.5/30.0)喂给出厂的 a1/go2 locomotion 任务;G1_29DOF 腿 sat=180、τ_lim=88(**膝是 139,不是统一 88**)、ω₀={32,20},脚 80/50/37,喂给出厂的 pick_place locomanipulation 任务 |
| IsaacLab **ANYmal B/C/D** | 出厂装的是 `ANYDRIVE_3_LSTM_ACTUATOR_CFG`(学习网络),**不是** `ANYDRIVE_3_SIMPLE_ACTUATOR_CFG`(DCMotorCfg,后者声明了但从未被赋值 = 死代码)。因为 `ActuatorNetLSTM(DCMotor)` 没覆写 `_clip_effort`,梯形仍作为**网络输出之上的安全钳**生效 | 用,但形态是"学习网络 + 继承的梯形钳" |
| IsaacLab **G1_CFG / G1_MINIMAL_CFG**(出厂 G1 locomotion 任务真正用的那个) | 纯 `ImplicitActuatorCfg` | **无包络**——所以"IsaacLab 的人形机器人都有包络"是假的 |
| **unitree_rl_lab** `UnitreeActuator` | 分象限梯形 + 摩擦:同向用 Y1、反向用 Y2;`|ω| ≤ X1` 平台,之后 `k = −τ_max/(X2−X1)` 线性降到 X2 处为 0;`compute()` 再减 `Fs·tanh(ω/Va) + Fd·ω` | 声明了 8 套预设,**全仓只有 `UnitreeActuatorCfg_Go2HV`(X1=13.5, X2=30 rad/s, Y1=20.2, Y2=23.4 N·m)被实例化过**。Go2W / B2 / H1 / G1_23DOF / G1_29DOF 全是 IdealPD 或 Implicit 盒子——**而且字典键名是照预设起的**(B2 的 "M107-24-2"、H1 的 "GO2HV-1"、G1_23DOF 的 "N7520-14.3"),纯装饰。**别把键名当证据。** |
| **mjlab** `DcMotorActuator` | `dc_motor_clip()` 是 IsaacLab 梯形的逐行移植 | 3 个出厂机器人(unitree_g1 / unitree_go1 / i2rt_yam)**一个都没用**,全是 `BuiltinPositionActuatorCfg` |
| **mjlab** `BuiltinDcMotorActuatorCfg` | 包 MuJoCo 3.10 原生 `<dcmotor>`(`mjSpec.set_to_dcmotor`,mjGAIN/BIAS/DYN_DCMOTOR):**τ = Kt(V − Ke·ω)/R**,反电动势烘进 biasprm,是**连续物理模型不是分段线性** | 出厂 0 处使用 |
| **MuJoCo 原生**(除 `<dcmotor>` 外) | `forcerange`/`ctrlrange` 恒为固定 `[−lim, +lim]` 盒子,与转速无关 | ——**MJCF 没有其他速度相关力限** |
| legged_gym / humanoid-gym / IsaacGymEnvs | `clip(τ, ±torque_limits)`,取自 URDF `effort` 静态值(humanoid-gym 再乘 `cfg.safety.torque_limit`;IsaacGymEnvs anymal_terrain 直接硬写 ±80) | 全是静态盒子。唯一例外:legged_gym 的 ANYmal-C 默认 `use_actuator_network=True`,整条 PD+clip 换成预训练 LSTM,**没有显式钳位** |

**(C) 我们要采纳,缺什么**
按 IsaacLab DCMotor / mjlab DcMotor 口径,**每个执行器组要 3 个数,而我们只有 1 个**:
1. `saturation_effort` τ_sat = 零速堵转/峰值力矩;
2. `velocity_limit` ω₀ = 空载转速;
3. `effort_limit` τ_lim = S1 连续额定。
我们只有 `effort_limit_sim`(URDF 的峰值盒子)。`docs/gates/G07_mujoco_to_real.md:150` 已经白纸黑字承认我们的力矩映射"不是厂商连续力矩-转速-温度曲线的替代品"。这正是 §18.4 第 1/2/4/8 条要问智元的东西;**第 4 条(母线电压 V、Kt、Ke、相电阻 R)应该提到最前**——mjlab 的 `BuiltinDcMotorActuator` 已经能直接吃这四个数,这是**数据缺口不是能力缺口**。

**(D) 两条不要踩的坑**
1. 换成 DCMotor 就必须**离开 ImplicitActuator**(PhysX 求解器内的 PD 绕过 Python 的 `_clip_effort`),这是积分器换代不是加参数;2026-07-02 已经试过 explicit/IdealPD,200 Hz 离散化过冲把反手打坏了。**维持 §18.5 的判定:做成 reward barrier,不改 actuator 模型。**
2. 我们现在的线性包络判据 `ω/ω_max + τ_net/τ_max ≤ 1` 是**三角形**,比上面每一个框架实际使用的**梯形**(有 τ_sat 平台段)都更保守。所以"waist_roll 线性包络下 15/73 超限"是**上界**;拿到 τ_sat 之后必须重算,不要据此提前砍 clip。

### 19.3 Q3 为什么球越快旋转越多

**一句话:球的速度和旋转不是取舍关系(实测正相关 r=0.96),真正被瓜分的是"拍速预算"的法向/切向分配;而"球越快必须越转"是落台约束的严格结论,ω_req 与球速成正比。**

---
## (一) 落台约束:为什么 ω_req ∝ v

**解析形式**。要在距离 L 内把球从触球高度 z₀ 压到台面,需要的向下加速度 ≈ `a_req ≈ 2Δz·v²/L²`——**随球速平方增长**;但重力恒为 9.81 m/s²。Magnus 加速度 `a_M = k_m_eff·|ω × v| ∝ ω·v`(线性段)。令 a_M = a_req:

**ω_req ∝ v / R,等价地 SP = R·ω/v ≈ 常数**(SP = 无量纲旋转参数)。

这就是为什么文献里球速和旋转是**正相关而不是取舍**:Delumeau 等 2025(DOI 10.3390/app15116350,n=9 精英青少年)测得球速与旋转 **r=0.96, R²=0.93, p<0.001**;Sato 等 2025/26(n=10,60–100% 五档发力)两者随发力同步上升。

**用我们自己的 venue 拟合算的数**(`configs/ball_physics_optitrack_20260730.yaml`:k_d=0.1253 1/m、k_m=0.00404、饱和形式 C_L(SR)=0.87·SR/(1+0.87·SR/0.55) → SR_sat=0.632;触球点 0.24 m 高 = 我们 73 库中位;网在 1.77 m 处、留 2 cm 余量;落点窗口 1.37 m):

| 出球速度 | 能落台的最小上旋 | SP=Rω/v |
|---|---|---|
| 6–12 m/s | **0**(重力够用) | — |
| 15 m/s | 24 rad/s(3.9 rps) | 0.032 |
| 18 m/s | 135 rad/s(21.5 rps) | 0.150 |
| 20 m/s | 206 rad/s(32.7 rps) | 0.206 |
| 23 m/s | **330 rad/s(52.6 rps)** | 0.287 |

触球高度是**一阶因子**:同样 23 m/s,触球点抬到 0.30 m 只要 139 rad/s,抬到 0.45 m(在网高之上击球)**完全不需要上旋**。

**但比"能不能落台"更重要的是余量。** 可行发射角窗口宽度(度,z₀=0.24 m):

| 球速 \ 上旋 | ω=0 | 100 | 200 | 400 | 800 rad/s |
|---|---|---|---|---|---|
| 8 m/s | **7.22°** | 9.40 | 10.66 | 11.80 | 12.72 |
| 10 | 3.21 | 4.70 | 5.61 | 6.65 | 7.45 |
| 12 | 1.15 | 2.64 | 3.44 | 4.24 | 5.16 |
| 15 | **0** | 0.80 | 1.49 | 2.41 | 3.32 |
| 18 | 0 | 0 | 0.46 | 1.26 | 2.18 |
| 23 | 0 | 0 | 0 | **0.23** | 1.15 |

**读法**:我们现在的球速(实测出球中位 **5.79 m/s**)窗口 >7°,旋转买不到任何东西——这就是为什么我们那套"不管旋转"的反解到今天都没出过事。到 15 m/s 时零旋转窗口**恰好为 0**,400 rad/s 才买回 2.4°。**旋转的真实职能是把"穿针"变成"有容差"**,这与 Iino/Yoshioka/Fukashiro 2017 的 UCM 结论(高手赢在稳住触球拍面角)是同一件事的两端。战术回报另算:Kidokoro 2025 在**同初速**下,≥110 rps 比 ≤80 rps 少掉 1.4 m/s、早到 **27±5 ms**。

**一个必须标出的坑**:Miyazaki 等 2016(DOI 10.1088/1361-6404/aa51ea,官方球自由飞行高速摄影)发现 C_L **不是** SP 的单调函数——在 Re=9.0×10⁴、SP≈0.48–0.5 有"升力危机"谷,升力几乎消失。职业数(23 m/s、117 rps → SP = 0.02×735/23 = **0.64**)恰好在谷的**另一侧**。我们自己的饱和模型是单调的,**复现不出这个谷**,所以任何"优化旋转"的求解器都会兴高采烈地把球推进我们自己飞行模型算错的区域。先标注,不要盲修。

---
## (二) 给定拍速预算的法向/切向分配

接触模型(`spin_contact.py:56-99` / `virtual_ball.predict_paddle_contact`)把 v_r 拆成两路:
- **法向 → 球速**:`v_out ≈ v_r,n + e·u_n`,`u_n = v_r,n + v_in`;
- **切向 → 旋转**:`Δv_t = a_t·|u_t|`,上限 `μ(1+e)·u_n`;`Δω = Δv_t/(c·R)`,c=2/3(空心薄壳,yaml `inertia_coeff=0.6667`),R=0.020 m,即 **ωR = 1.5·Δv_t**。

关键补正:**回击上旋球必须把旋转反向**(来球在世界系里的上旋与出球上旋方向相反),所以 `Δω = ω_out + |ω_in|`,而且来球接触点的表面速度 `|ω_in|·R` 要先被抵消——来球 400 rad/s 就是 8.0 m/s 的表面速度,735 rad/s 是 **14.7 m/s**。

用 e=0.60(venue 中位)、a_t=0.637、μ=0.5 算出的分配表:

| 目标出球 v/ω | 来球 v/ω | v_r,n | v_r,t | \|v_r\| | 刷球角 |
|---|---|---|---|---|---|
| 9.7 / 0 | 4.0 / 0 | 4.56 | 0.00 | **4.56** | **0°** ← 这就是我们今天 |
| 15 / 24 | 10 / 100 | 5.62 | 2.60 | 6.19 | 24.8° |
| 18 / 135 | 12 / 200 | 6.75 | 7.01 | 9.73 | 46.1° |
| 20 / 206 | 14 / 300 | 7.25 | 10.59 | 12.84 | 55.6° |
| 23 / 330 | 15 / 400 | 8.75 | 15.28 | **17.61** | 60.2° |
| 23 / 735 | 15 / 735(真·职业对拉) | 8.75 | 30.77 | 31.99 | 74.1° |

(单位 m/s / rad/s / 度)

**刷球的代价**:`|v_r| = v_r,n / cos(刷球角)`。20° 多花 6%,30° 多花 15%,45° 多花 41%,60° 多花 **100%**。这才是真正的取舍——**不是"球的速度 vs 球的旋转",而是"拍的法向 vs 拍的切向"**。

**三条直接结论**
1. 职业 13–15 m/s 拍速的构成对上了:约 **8.75 m/s 法向**(在 e≈0.6、来球 15 m/s 下产出 23 m/s 球速)+ **10–12 m/s 切向**(产出 100+ rps),矢量和 ~14–17 m/s。**所以 23÷14=1.6 不是恢复系数,是几何投影**,我们文档里那句"拍速到球速 ~1.6 倍"必须改写。
2. 球越快要加越多切向速度,是因为 ω_req ∝ v(第一部分)**同时**来球旋转也随对拉水平上升,而反转来球旋转的成本 |ω_in|·R 独立增长。两者叠加,切向需求增长比法向**更陡**——表里刷球角从 0° 一路到 60°。
3. **最后一行是我们模型的失效证据,不是物理结论**:23 m/s + 735 rad/s 需要 Δv_t = 19.6 m/s,而库仑上限 μ(1+e)u_n = 19.0 m/s——**按我们现有常数,职业对拉这一板打不出来**。原因是 a_t=0.637 / μ=0.5 是在业余对打上拟合的(yaml 自己写明有效域:球速 1.7–6.4 m/s、u_n 2.2–8.8 m/s、旋转 p90 ~19 rps,**>23 rps 混叠**),而且模型永远处于"滑移"分支、没有真正的粘着(no-slip)分支,系统性低估大 u_t 下的造旋能力。外推到 u_n>10 m/s 必须打 WARN。

### 19.4 Q5 反解裁定(**待修项**)

**裁定:不完整,不是错。**每一行已发的题它都真能落在它说的点上(回路里跑的是全非线性 RK4 + 阻力 + Magnus),但这套**形式化表达不出**球速超过约 15 m/s 之后物理必须的那件事。而且它在 ≤12 m/s **静默正确**,一旦 §18 的提速路线成功就**静默错误**——所以要在提速之前改,不是之后。

**缺陷 1:出球旋转在任何地方都不是目标。**
全部入口(两条已接线的 LM + 未接线的解析解)只收 `aim_xy`/`target_xy`(2 个数)。全仓 grep `target_spin` / `desired_spin` / `spin_target` / `demanded_spin` / `w_plus_target` / `w_plus_desired`(tracking mdp 包 + hope_planner)= **0 命中**。出球旋转 w_plus 永远是"选中的拍状态"的**副产物**。

**缺陷 2:自由解用"最省力"约定挑答案,而它罚的正好是造旋的那两个分量。**
- `hope_training/.../tasks/tracking/mdp/strike_spec_torch.py:71` — `v_r = q[:,2:3]*n + q[:,3:4]*b1 + q[:,4:5]*b2`(法向 + 两个切向);
- `strike_spec_torch.py:148` — `residual = torch.cat([land_xy_ - target_xy, w_speed * q_[:, 2:5]], dim=-1)`,`w_speed=0.03`。**各向同性 Tikhonov 把 v_t1/v_t2 和 v_n 一视同仁地压向 0**,即偏向刷球角 = 0°。
- numpy oracle 自己写明:`hope_ws/src/hope_planner/strike_spec_planner.py:199-201` —— "picks the least-effort racket motion among the landing-equivalent solutions"。
- 补一刀:`strike_spec_analytic.py:96` 记录实测退出时**正则项是落点项的 912 倍**,即现役求解器连它自己设计要挑的"最省力解"都没可靠交付。约定本身也不是个良态约定。

**缺陷 3:训练侧适配器把拍速方向硬钉死。**
- `stroke_adapt_torch.py:218` — `v_r = q[:, 2:3] * d_hat_w`;
- docstring `stroke_adapt_torch.py:251` — "C1 direction: `v_r = s * d_hat_w` — exact by construction, **not a variable**"。
唯一自由量是有界标量 `s ∈ [speed_min, speed_max]`。**刷球角是每条 clip 继承来的常数**——而这批 clip 的实测出球中位只有 5.79 m/s,所以继承的是一个 5.8 m/s 级别的刷球角。

**缺陷 4:过网不是约束,是事后闸门。**
- `strike_spec_torch.py:136-138` docstring — "`net_z`/`net_valid` do NOT enter `ok`: the net test is the caller's";
- `continuous_questions.py:846-847` — `net_ok = out['net_valid'] & (out['net_z'] > net_top_z); good = good & net_ok`;
- `gen_stage1_questions.py:962-965` — "the solver ANNOTATES ... and the bank DECIDES" + `continue`。
**后果**:恰恰是"必须加上旋才能既过网又落台"的那些行(按 q3 的数,触球高 0.24 m 时 15 m/s 以上零旋转窗口 = 0°),被**丢弃重抽**而不是被求解。训练/考试分布被静默地从"需要旋转的球"上切掉了。

**缺陷 5:题库连来球旋转都没有。** `gen_stage1_questions.py:2` 表头 "fixed strike point, varying incoming speed, **no spin**, fixed landing",`incoming_spin = np.zeros_like(v_in)`。连续路径与 action_ball 采样确实把**来球**旋转当独立臂抽(manifest 实证:incoming_speed_max_mps 5.2377、spin_magnitude_max_radps 60.0、center 0.0),但**出球旋转从不是采样轴或目标**。

---
## 正确的表述应该是什么

**(a) 维数先说清楚。** 正向映射是 (拍面法向 n:2 DOF, 拍速 v_r:3 DOF) → (出球速度 v⁺:3, 出球旋转 ω⁺:3),**5 进 6 出**。所以任意 (v⁺, ω⁺) 组合一般**不可行**,只有落在 5 维可达流形上的才存在。不要对外承诺"任意速度 + 任意旋转"。

**(b) 同一出球速度对应一族拍速矢量——这条我们自己仓里已经证过,只是没接线。**
`strike_spec_analytic.py:395-427` 的 `answer_sphere()`:固定 v⁺ 时,合法 v_r 的集合是 v_r 空间里的一个**球面**(由 n 的 2 个角度自由度索引),再被摩擦锥限制成一个**球冠**,半角 = `2·atan(μ)` = **53.13°**(μ=0.5,该行在 :427)。冠上不同的点给出**相同落点、不同出球旋转**。该模块 docstring `:116` 明写 "NOT WIRED"。

**(c) 目标应该从二元组升成三元组。**
现在:`(落点 x, 落点 y)`。
应该:`(落点 xy, 到达速度或飞行时间 T, 出球旋转 ω⁺)`——或等价地 (v⁺ 矢量, ω⁺ 矢量) 加可行性检查。然后两段解:
- **第 1 段 飞行反解**:给定 aim + T + 期望 ω⁺,在阻力 + Magnus 下解 v⁺,并且**把过网当成不等式约束而不是事后闸门**。我们的解析模块已经做了自洽的含旋飞行反解(`strike_spec_analytic.py:288` 的 `Om = k_m*w_plus` 进 exp(Mt) 旋转),只差接线。
- **第 2 段 接触反解**:在 (b) 的球冠上,按**目标旋转**选点,而不是按最省力选点。现有三种 pin(`strike_spec_analytic.py:90-101` 的 normal / min_speed / clip_swing)全是几何或省力约定;**缺的第四种 pin 就是 "spin"**。默认的 'normal' pin(v_r ∥ n,切向严格为 0)的理由在 docstring 里写得很诚实:"the one that does not invalidate the shipped banks"——**工程连续性,不是物理**。

**(d) ω⁺ 该从哪来。** 不用拍脑袋:由落台约束本身给出。`ω⁺_req ≈ SP_req·v⁺/R`,其中 SP_req 由几何(触球高度、到落点距离、网高、要求的角度余量)决定,且**近似与球速无关**(见 q3 的推导与数表)。所以正确的设计是:在落台 + 过网 + 余量约束下**联合**解 (v⁺, ω⁺),再反解出同时产生这两者的拍状态。

**(e) 为什么至今没炸。** 整条流水线都活在球速 ≤12 m/s 的区间,那里所需上旋 = 0、可行发射角窗口 1.15–7.22° 宽。**这是一个潜伏缺陷,它会在提速成功的那一刻转正。**

### 19.5 Q4 重新动捕:采集设计

**核心问题的答案:泛化轴必须是"对不同来球种类的回应",不是"引拍距离/速度"。**理由按证据强度排:

1. **现有 73 库在来球轴上几乎没有方差——这是我这次直接量出来的。** 74 个 unit 的实测球:来球速度 **1.16–4.90 m/s(中位 3.88)**、触球高度 **0.11–0.35 m(中位 0.24)**、**来球旋转完全没有测量**(`ball_ext/README.md` 原文:"涂覆球无自旋数据";合成弧是无旋的)。也就是说,**这是同一种来球条件重复了 73 次**。无论新采集变什么,来球轴是当前唯一的"点质量"。
2. **引拍/速度这条轴的天花板可以解析算出来,不必花钱采。** §18.3 的 H2 已经给了:ω² = 2αΔθ,引拍路径加长 50% 只换 √1.5 = **1.225 倍**拍速;而纯时间缩放被 waist_roll 力矩卡在 **k=1.19**。一个回报 22%、上限已知的轴,不值得占一整个采集矩阵。
3. **文献说拍速不区分水平。** Iino & Kojima 2009:高水平与中等水平**触球拍速无显著差异**;Iino/Yoshioka/Fukashiro 2017(UCM):区分他们的是**触球瞬间稳住垂直拍面角**;Iino/Mori/Kojima 2008:对不同来球旋转的适应是靠**改变手臂构型**(对下旋时拍的向上速度显著更高,但肘伸/腕背屈的**角速度量级相同**)——**这正是"对来球种类的回应"这条轴**。而 Bańkosz & Winiarski 2018 自己的实验设计就是 FH1/FH2/FH3 = **无旋 / 下旋 / 强上旋**,领域公认的动作变化轴就是来球旋转。
4. **Ripoll & Latiri 1997**:恒速刺激下专家与新手**无差异**,减速/变轨迹刺激下**显著有差异**。专业性住在"变化"这条轴上。
5. **和 q5 挂钩**:我们现在连"给这一板要多少出球旋转"都问不出口。多采引拍长度只会给我们更多**瞄不准的速度**;多采来球种类才能给出求解器需要的 (拍状态 → 出球速度+旋转) 配对。

---
## 要不要偏离现有乒乓动作框架:不偏离,但会话设计整个换

**保持不变(硬约束)**
- **正反手双 clip 约定**(记忆:单 clip 臂是二等公民,三处代码硬编码成正手且无法自述)。新矩阵必须 FH/BH 对称。
- **帧合同**:120 Hz pkl → +6 帧头 pad → 50 Hz(`hit50 = round((hit_pkl+6)·50/120)`,z=0 地板,per-unit `yaw_norm_deg`),这样新 clip 直接掉进现有消费者。
- 重定向管线、npz 字段名、CLIP_ORDER 口径不动。

---
## 采集协议

### 变(设计矩阵)
| 轴 | 水平 | 理由 |
|---|---|---|
| **A. 来球旋转(主轴)** | 5 档:强上旋 / 弱上旋 / 无旋 / 弱下旋 / 强下旋 | 用**可控旋转的发球机**(三转子,与 Miyazaki 同类)发,旋转是**指令值**因而天然带标签,不靠事后反演 |
| **B. 来球速度** | 3 档:**4 / 8 / 12 m/s**(触球点处) | 现有全库(1.2–4.9)只等于第 1 档;12 m/s 是落台约束开始咬人的点(零旋转窗口只剩 1.15°) |
| **C. 触球高度/时相** | 2 档:上升期低球 z≈0.20–0.25 m(=现状)/ 高点期 z≈0.40–0.50 m | 一阶因子:23 m/s 所需上旋随触球高度 0.24→0.45 m 从 330 rad/s 掉到 0 |
| **D. 出手意图** | 2 档:**拉**(旋转优先)/ **打**(速度优先),口头指令 | 这是 Sato 的发力档操纵改成意图操纵,直接填满法向/切向分配的两端 |
| **E. 落点** | 直线 / 斜线(/ 中路) | 落点是我们求解器**唯一**在瞄的东西,而现有库里它也没变过 |

### 不变(刻意固定,写进协议)
- **引拍幅度/距离**:让选手自然发挥,**不设为因子**。回报上限已解析已知,设成因子会白吃格子。
- **步法/站位**:定脚,本次不采带步法的单元。步法是独立会话,混进来会让格子数×3 并污染触球帧统计。
- **球拍/胶皮/球**:全程一套。我们现有两次拟合已经在 e(u_n) 衰减上打架(venue g2=−0.0441 vs optitrack −0.0703),候选原因之一就是胶皮不同,别再加一个混杂。
- **场地/球台**:一张台,开场调平一次并记录倾角(optitrack 那次中途重新调平 0.744°→0.561°,现在是该拟合里的一个混杂)。

### 条数
- 主矩阵 A(5)×B(3)×C(2)×D(2) = **60 格/手**,E 作格内随机不作因子;**每格 8 条** → 480 条/手,FH+BH **960 条**。10 s/球的多球节奏 ≈ 3 小时,一个会话拿得下。
- **一天版(最小可用)**:A(5)×B(2: 4 与 10 m/s)×C(2) = 20 格 × 8 = **160 条/手,共 320 条**——仍然是现有 73 条变化量的 4 倍以上。

### 标注(这一节才是采集成不成立的关键,现有库正是死在这里)
1. **来球旋转**:发球机**指令值**(大小 + 轴)+ **实测值**。测量路线已经被验证可行:OptiTrack 那次证明标记球星座能通过逐帧点云配准恢复真实转动(`ball_orientation.py: solve_orientation_chained`,逐帧 ω 时间相关 0.26–0.82/轴,纯弹跳 1.6 rev/s vs 对打 5.1 rev/s 排序正确)。**阻断项**:该布局在 360 Hz 下混叠上限约 **22.9 rev/s**,而我们要覆盖 50–120 rps。**必须在开会话之前解决**(提帧率,或改用标记间距更大的布局),这是新采集唯一的技术卡口。
2. **出球速度与旋转**:同一通道,触球后。今天已有 ball sidecar 球位反演出的出球速度，
   但没有出球旋转，也没有与球同钟的拍状态——因此仍给不出一条 (拍状态 → ω⁺) 训练对，q5 的修法
   也没有足够数据标定。
3. **拍面法向 + 拍心速度**(触球时刻),来自拍上标记,**与球同一时钟**。chingmu sidecar 完全没有球拍通道;optitrack 那次把两块拍和球放在同一条 C3D 流里(`capture.timing.same_clock: true`)——照抄这一条。
4. **落点 (x,y)、过网高度、界内/界外**,逐板。
5. **逐板意图标签**(拉/打)、发力档、选手自评质量。
6. **触球时刻**:球侧与拍侧**两个**估计都记,并记录二者之差。optitrack 拟合发现 t_c 早约 4 ms 会给 e 带来 +0.044 的偏置,这个偏置**现在正在往我们的接触模型里渗**。

---
## 与现有 73 库的关系:**补充,但同时降级**

- **保留**:73 条作为模仿/风格先验和帧合同参照——它们已经重定向、已 bless、被在跑的臂消费(CLIP_ORDER sha256 已钉)。**不替换。**
- **降级**:不能再当"速度/质量参照"。实测出球中位 5.79 m/s(最大 7.96),来球轴是个点,而且线性包络下已有 **15/73** 违反 waist_roll。**任何 >10 m/s 的东西都不能拿它当基线。**
- **补失衡**:73 库是 FH 14 / BH 59;新矩阵按构造对称,同时正好补上 §17.3 说的"反手弱是示范数据自带的天花板"。
- **两个必须先修的数据缺陷**(否则同一管线会静默污染新会话):(1) 全部 73 条每一帧 `head_yaw_joint` / `head_pitch_joint` 恒为 0.000,几乎肯定是 GMR retarget 把头部 DOF pin 住了,找 Jiayi 核;(2) 6/73 条存在第二个 >90% 高度、距全局峰值 >10 帧的次峰,它们的"触球帧"代理可能选到了引拍。
- **排序**:**先跑 §18.5 的同步性诊断**。如果重定时真的把 5.50 → 7.85 m/s 兑现了,新采集的速度档要相对**重定时后的基线**设定,而不是原始基线。

### 19.6 行动项

**[P0][动作库 / 提速(§18 已定,保持并扩口径)]** 同步性诊断脚本:把 73 条 clip 的触球帧同步效率排序,低效率优先重定时。扩一条:同一脚本再输出每条 clip 的『法向利用率』——用 ball sidecar 实测出球速度(中位 5.79 m/s)与来球(3.88 m/s)反推法向拍速约 2.2–2.8 m/s,对比 Σ|ω_j|·r_j = 5.50 m/s,即只有 40–50% 拍速落在法向。53.4% 的时序损失和这层方向损失可能是同一件事的两个投影,必须先分开量再谈提速。

**[P0][动作库 / 重定时闸门(§18 已定,保持并加限定)]** 任何时间缩放脚本加 waist_roll 净力矩闸门(分母用 minimum_effort_margin_nm 31.87 N·m 而非 46.0,判据用 ω/ω_max+τ/τ_max≤1),并给 15 个超限 clip 挂标记。限定:该线性包络是三角形,比 IsaacLab/mjlab/unitree_rl_lab 实际使用的梯形(带 τ_sat 平台)更保守,所以 15/73 是上界——拿到厂商 τ_sat 前不要据此砍 clip。

**[P0][反解形式化(新,必须排在提速之前)]** 把反解目标从『落点 2 数』升级成三元组『落点 xy + 到达速度/飞行时间 T + 出球旋转 ω⁺』;把过网从事后闸门(strike_spec_torch.py:136-138、continuous_questions.py:846-847、gen_stage1_questions.py:962-965)提升成求解约束;接线 strike_spec_analytic.py 的 answer_sphere(:395-427,摩擦锥球冠半角 2·atan(μ)=53.13°)与含旋飞行反解(:288),并新增第 4 种 pin =『按目标旋转选点』,取代 strike_spec_torch.py:148 的各向同性 w_speed=0.03 最省力约定与 stroke_adapt_torch.py:218 的固定方向 v_r=s·d_hat_w。理由:现形式在 ≤12 m/s 静默正确,提速成功当天转为静默错误。

**[P1][reward 设计(§18 已定,保持并加护栏)]** 加触球帧同步奖励(各链关节 |ω_j| 相对自身前摆峰值的比值),不加提速奖励;躯干引导到 waist_yaw(力臂 0.736 m、220 N·m、余量 k=2.84)不到 shoulder_roll,更不到 waist_roll。新增护栏:同步项必须与『法向利用率』分开计权,否则策略可以靠加大刷球角虚假抬高 Σ|ω_j|·r_j。可并入第二项『稳住触球拍面角』(Iino 2017 UCM:高水平者该角的关节组合方差显著更小)。

**[P1][动捕采集(新)]** 先解决球旋转测量的混叠卡口(标记球星座在 360 Hz 下上限约 22.9 rev/s,目标要覆盖 50–120 rps),再定稿采集矩阵:来球旋转 5 档 × 来球速度 3 档(4/8/12 m/s)× 触球高度 2 档 × 出手意图 2 档 = 60 格/手 × 8 条;一天版降为 20 格 × 8 = 160 条/手。固定引拍幅度、步法、拍/胶/球、球台。必测标签:来球与出球的旋转、拍面法向与拍心速度(与球同时钟)、落点与过网高度、意图标签、球侧/拍侧两个触球时刻。

**[P1][厂商数据(§18.4 已定,调整次序)]** 向智元索取 12 项电机数据,把第 4 条(母线电压 V、力矩常数 Kt、反电动势常数 Ke、相电阻 R)提到最前——mjlab 的 BuiltinDcMotorActuator(τ=Kt(V−Ke·ω)/R,包 MuJoCo 3.10 原生 <dcmotor>)已经能直接吃这四个数,这是数据缺口不是能力缺口。其次是第 1 条 τ-ω 包络(需要 τ_sat/ω₀/τ_lim 三个数,我们只有 effort_limit_sim 一个)与第 8 条 velocity_limit_sim 的物理来源。

**[P2][球物理模型有效域(新)]** 给接触/飞行模型加有效域告警并入摘要(符合『WARN 必进摘要』):venue 拟合有效域为球速 1.7–6.4 m/s、u_n 2.2–8.8 m/s、旋转 p90 约 19 rps(>23 rps 混叠)。直接后果两条:(a) 按现有 a_t=0.637/μ=0.5,23 m/s+735 rad/s 的职业对拉需要 Δv_t 19.6 m/s 而库仑上限只有 19.0 m/s,即我们自己的模型说这板打不出来——这是模型缺陷不是物理结论;(b) 我们的饱和 Magnus 形式是单调的,复现不出 Miyazaki 2016 在 SP≈0.48–0.5、Re=9×10⁴ 的升力危机谷,而职业工况 SP≈0.64 就在谷的另一侧。

**[P2][文档口径修正(新)]** 修三处:(1) dr_reward_external_diligence_20260731.md:1626 骨盆『321–393°/s』应为 312–392°/s(Bańkosz 2018 Table 2 实值 320.63/312.21/392.45,转写错位);(2) 反手拍速『11.6–14.1 m/s』是 VRcont 与 VRmax 混口径(BH1 VRcont 11.63 vs VRmax 12.55),统一改用 VRmax;(3) 停止把 23 m/s 球速 ÷ 13–15 m/s 拍速当作 1.6× 转换系数——那是『法向约 8.75 m/s + 切向约 10–12 m/s』的几何投影,不是恢复系数。

**[P2][数据质量 / 上游(§18 已定,保持)]** 找 Jiayi 核两个缺陷,并在新采集复用同一管线前修掉:(1) 全部 73 clip 每一帧 head_yaw_joint / head_pitch_joint 恒为 0.000,几乎肯定是 GMR retarget pin 住了头部 DOF;(2) 6/73 clip 存在第二个 >90% 高度、距全局峰值 >10 帧的次峰,其『触球帧』代理可能选到了引拍,所有击球窗统计对它们不可信。

**[P3][仿真口径裂缝(§18 已定,原判不动)]** PhysX 的 velocity_limit_sim 是真刹车而 MuJoCo 无等价机制(mujoco_eval_onnx.py 只有积分后 np.clip 代理、自标 proxy_is_post_integration_nonexact),在 reward/惩罚层加与 PhysX 一致的速度 barrier,不改 actuator 模型;明确不要回 explicit/IdealPD(2026-07-02 已试过,200 Hz 离散化过冲把反手打坏)。


---

## 二十、三阶段架构提案裁决:拍状态入目标 + 结果条件化(08-02;Franco 提案)

**方法**:2 抽取(Ace/SMASH 奖励架构逐条读原文 / 我方球接触模型与标记球混叠)+ 2 对抗核查(3 修正)
+ 1 裁决。Ace 与 SMASH 均以本地 PDF 全文为准(pdftotext,两种抽取方式交叉);我方以代码 file:line 为准。
**口径纠错先行**:SMASH 的 86.38% 是**仿真里拍状态跟踪成功率**(判据 Ep<0.04 m、Eo<0.05、Ev<0.5),
**不是回球率**;真实回球率 MoCap 18/20=90%、自我中心相机 12/20=60%,642 板连续测试触球 93.7%、
**成功回球 59.7%(正手 66.7% / 反手 38.9%)**。本 doc 早前引用的 86% 一律按此更正。

### 20.1 裁决

【裁决:有条件成立(通过),但必须改一个关键点 + 不能替代 §16 的剂量修复】

一句话:方向对,而且对得有外部实弹先例(SMASH 整套就是这么搭的);但"把球拍状态放进 mimic 目标"这个提法要改成"把球拍状态做成独立的 task 通道,目标来源随阶段切换(第一阶段=clip,第二/三阶段=球条件化的规划目标)"。SMASH 恰恰是把两者拆开的:拍状态对齐**规划器反解出来的目标**(任务通道),全身对齐 mocap(风格通道,而且**把腕关节从模仿项里剔除**,理由原文写着"防止参考动作在击球时过度约束球拍")。塞进 mimic 在第一阶段与提案等价(此时 clip 就是目标),但到第二阶段会自己跟自己打架。

**它治好了什么(逐条对账)**

1) **§7 死核 —— 真治,但只治第一阶段。** 现役病灶是 racket 三核的 σ 选的是"验收容差"(pos 0.075 m / normal 0.262 rad / vel 0.5 m/s,`HOPEPingPongActionBall.yaml:113-118`),而目标是球条件化采样点,窗口开启时刻误差 ~0.70 m,e/σ=9.3 → exp(-87),float32 下是精确 no-op。换成"拍状态对齐同一条 clip"以后,目标变成**相位连续、由策略正在模仿的那条轨迹自己生成**,运行误差/饱和半径回到外部三家 RSI 下的 ~0.08 量级,梯度处处存在。代价:这个疗效**不会跟着进第二阶段**——目标一旦球条件化,死核原样回来。所以 §7 的 P0(σ 阶梯)/P1(粗细双核)**不作废,只是延后到第二阶段开球时必须同时上线**。

2) **§16"不挥拍是最优解" —— 部分治,治的是正确的那一半。** 现役 69% 盈亏平衡命中率来自"挥拍的唯一收入是稀有的落点 +32.98,而失败要吃 -72 死亡尖峰"。新架构把挥拍的收入从**稀有事件**搬到**每步 dense 通道**(拍位/拍面/拍速跟踪),挥拍本身就有钱,不再需要赌命中率——这是结构性修复,不是调参。**但 -72 不会因此消失**:它等于 1200 步(24 s)完美 dense 收入、约 10 倍整段 episode 全部正收入(update250 平均 episode 只有 124.11 步 = 2.48 s),而新增 dense 拍收入也就是 +0.0x/步量级,救不回来。**必须按 §16 独立降尖峰,否则新架构照样跑成"站着不动地模仿"**。

3) **§13 `strike_opportunity_count=0` —— 不治,但被绕开(这是好事也是陷阱)。** 第一阶段根本不需要击球机会,所以它必然跑得通。陷阱是:**第一阶段跑通不能当第二阶段的放行证据**。必须给第一阶段定一个不含球的验收(拍心位置/拍面/拍速三通道的窗内误差分布 + 同步效率),否则等于拿一个必然通过的指标发车。

4) 顺带治好的:①合同上不再需要"球拍目标从题库来"这条链,第一阶段可以完全脱离题库、脱离反解跑起来(降低耦合);②收入分层"模仿 < 击中 < 质量"第一次在结构上分得开——阶段一只有模仿层,阶段二加击中层,阶段三加质量层,每层单独定价、单独验收。

**它自身的新风险(六条,都有具体缓解)**

R1 **腕过载**。加大拍面/拍位权重 = 加大对腕的约束,而腕 pitch/yaw 是全机最小执行器(6 N·m,净预算 4.82 N·m),`bh_loop_c` 这条肩优先改造 clip 的 wrist_yaw 已是自然值的 5.5×(14.9% vs 2.7%)。**缓解:高权重拍状态项只允许挂 73 库自然 clip,禁止挂 bh_loop_c 族——"73 库优先"在这里不是偏好,是安全前提。**

R2 **把平庸质量钉成先验**。73 库实测拍速中位 5.50 m/s、同步效率 53.4%、法向利用率只有 40–50%、出球中位 5.79 m/s(职业 23 m/s 的 1/4)。高权重会把这套平庸拍状态学成硬先验,阶段三要提质量就得先对抗它。**缓解:阶段二/三必须显式衰减 racket-vs-clip 权重(或切目标源);并且现在就把同步效率/法向利用率做成 clip 准入排序,低效 clip 先重定时再当目标。**

R3 **控制点错位**。mimic 通道现在的"拍"其实是 `right_wrist_yaw_Link` 的 link 原点(拍的所有 mesh 被 PhysX 并进这个 body),而策略的官方控制点是 `official_racket_site`;两者差一个固定刚性偏移,姿态误差会被杠杆臂放大成拍心位置误差。**缓解:新项必须显式绑 site,不能复用 link 原点。**

R4 **分层倒挂**。"更大权重"若把模仿层顶到击中层之上,分层塌回"只模仿"。**缓解:一律按每步 dense 口径(×dt 0.02)定价,阶段二上线当天同步下调阶段一权重,A/B 用同一 seed。**

R5 **5 进 6 出的信息倒挂(最深的一条)**。拍状态是 5 自由度输入 → 出球 6 自由度输出,给定拍状态出球唯一确定;反过来同一落点对应一族拍状态(摩擦锥球冠,半角 2·atan(μ)=53.13°)。所以"拍状态目标"比"落点目标"**信息更强、约束更死**:阶段一用 clip 的拍状态没问题(那就是我们想要的风格),但阶段二/三若继续用单点拍状态目标,等于替策略把球冠上的点**选死**了,而选的那个点(clip 继承的刷球角)对应的是 5.8 m/s 级出球——直接与阶段三的旋转条件化冲突。**这正是反解不能扔的根因。**

R6 **观测合同漂移**。阶段切换若换目标源,actor 观测的语义变了(clip 派生 vs 球派生),必须走一次合同修订 + 部署 parity 复核,不能靠改 params 悄悄换。

**一句话给 Franco**:提案成立,按 SMASH 的拆法落地(拍状态=任务通道对齐规划目标、全身=风格通道且腕放松),第一阶段目标源用 clip 是完全合法的特例;但 §16 的 -72 死亡尖峰必须同期独立修,否则新架构的 dense 收入被尖峰吃掉,结论会和现在一样。

### 20.2 "只有上台奖励会不会引导不够"——Ace 与 SMASH 的实证

【"只有上台才有奖励,引导够不够?"—— 答:不够,而且 Ace 之所以能行是靠我们**结构上没有**的四件东西。建议抄 SMASH,不抄 Ace 的稀疏。】

**一、Ace(Sony,Nature)的真相:确实 100% 稀疏,但不是"只有上台才有钱"**

- 奖励全部是终局量:原文"The reward function used during training consists of several terms, all of which are calculated after the episode has finished, that is, as a function of the terminal state."全文 grep `shaping` / `dense` / `closest approach` / `time-to-contact` / `distance to` / `racket-to-ball` = **0 命中**。**没有任何稠密接近引导**,这条如实。
- 但它是**三级阶梯**(Eq. 10):`R_miss`(没碰到球)< `R_hit`(碰到但没回过去)< `R_hit_return`(碰到且回台)。**"击到球"本身就单独发钱**——这正是我们"模仿 < 击中 < 质量"分层里的中间层。所以 Ace 也不是"只有上台才有奖励"。
- 只有**一部分**策略用带参数的 `R_hit_return`:按期望落点 `y_desired` 与权重 `w_reward=[w_p, w_s]`(w_p∈[0,1] 位置权重,w_s∈[-1,1] 落台时球绕 y 轴角速度即旋转权重)条件化;`y_desired` 在对方半台均匀采样,`w_reward` 偏向稀疏与边界值,**训练与比赛用同一套采样分布**。Franco 提案的第三阶段(落点 + 旋转权重目标条件化)与这条**逐字对上**。
- Ace 用什么代替稠密引导(这才是关键):①**off-policy SAC + 回放**,稀有成功可以被反复重用;②**HER**:把 `y_desired` 事后改写成实际落点并令 w_p=1,于是"碰到球但落点不对"直接变成满分样本——这是主文里最接近课程/引导的装置;③**event tables 分层采样**(near miss / ball hit / ball returned / 高速高旋回球分桶),稀有事件被过采;④**起点分布很富**:来球状态从真人/合成数据的 KDE 采样(发球:回球 = 3:7),机器人起点在"静态中立位"与"**从此前训练回合存下来的 reset plan 采样**"之间二选一(自举式 RSI);⑤**每回合只打一板**(single shot),信用分配窗口极短;⑥非对称 critic 直接吃仿真真值球态 + 重建球态的辅助损失。
- **诚实标注**:Ace 主文**从未解释**稀疏奖励如何从零起步 bootstrap;若有说明,应在 Supplementary Information 1.4.1–1.4.8,而本地只有 11 页正文 PDF,**SI 不在本地,无法核对**。

**二、SMASH(humanoid 乒乓)的真相:完全相反,没有任何球结果奖励**

- 任务奖励是**每步的窗门控指数核**(Eq. 12):`r_j(t) = exp(-e_j(t)/σ_j(t)) · I[τ_t ∈ W_j]`,j ∈ {位置, 姿态, 速度};窗宽:位置 **0.02 s**、姿态与速度 **0.1 s**(围绕触球时刻)。总任务奖励 Eq. 13 = `w_pos·r_pos + w_ori·r_ori + w_vel·r_vel + r_succ`,`r_succ` 是三通道阈值同时满足时的稀疏 bonus。(`w_pos/w_ori/w_vel` 的数值**全文未公布**,已用 plain 与 -layout 两种抽取复核确认。)
- 误差是对着**规划器反解出来的拍状态目标** `p_hit, v_hit` 算的(第 VI.F 节 Eqs. 27-32:先在线性阻力模型下反解出球速度,再反解无摩擦法向碰撞得到拍面法向与法向速度),**不是**对着 mocap/VAE 参考 clip 算的。mocap 那条是**另一条**通道(风格/正则),且**明确把腕关节排除在模仿项之外**。
- **SMASH 没有任何球结果奖励项**:无落点、无过网、无旋转;论文自陈"不显式建模旋转"是已知局限。也就是说,**"只靠拍状态跟踪当 RL 目标"这件事,SMASH 就是实弹证据。**
- **σ 课程是它的命门,有消融数字**:σ_j 随在线跟踪误差自适应收紧(误差降 → σ 降 → 精度要求变严)。去掉自适应 σ:SR 从 **86.38% 崩到 22.60%**,位置误差 4.42 cm → **11.94 cm**,姿态误差 4.17 → **35.49**;论文归因是"初始 σ 太严 → 任务奖励拿不到 → 根本学不起来"。另一件:自适应区域采样(把台面击球工作空间分区,成功率最低的区过采),去掉后 SR 86.38% → 82.72%。
- **数字口径纠错(Franco 记的 86% 要改)**:86.38% 是**仿真里的拍状态跟踪成功率**(判据 Ep<0.04 m 且 Eo<0.05 且 Ev<0.5),**不是回球率**。真实回球率是另一个指标 `SR_return`,且更低、依感知条件:MoCap 18/20=**90%**,自我中心相机 12/20=**60%**;642 板 50 分钟连续测试里触球率 93.7%、**成功回球率只有 59.7%**(正手 66.7%,反手 **38.9%**)。反手这条与我们 §17.3"反手弱是示范数据自带的天花板"同向。
- 架构上 SMASH **明确拒绝分层**:"Instead of adopting a hierarchical controller … we use a simple motion matching scheme",单个 PPO 全身策略 + 非学习的最近邻动作检索(用规划目标去库里匹配 clip)。Ace 也不是端到端分层:它是**一堆各自独立训练的扁平 SAC 策略** + 一个**非 RL 训练**的外部选择器(固定/随机/规则/在精英对局数据上监督训练的分类器)。

**三、建议:抄 SMASH 的骨架,只抄 Ace 的两件小东西**

- **抄 SMASH**:①拍状态三通道窗门控指数核作为任务主通道(我们已有同形核 `hope_rewards.py:118/145/158`,连 `racket_strike_success` 三阈值 bonus 都已存在 `hope_rewards.py:3421`,等于 Eq.13 的 `r_succ`);②**σ 课程**——这是外部唯一一个"去掉就崩"的量化证据(86.38%→22.60%),而且和 Franco 钦定的"静态权重 + σ 课程"逐字一致,现役 `racket_*_std` 键已全程接线,**零代码**;③自适应区域采样(成功率最低的击球区过采);④窗宽分层照抄:位置窗窄、姿态/速度窗宽(我们现在三通道同窗)。
- **抄 Ace**:①**三级 outcome 阶梯**(没碰到 < 碰到 < 碰到且回台)——这直接对应第二阶段"能不能击到球"要单独发钱,不要等上台;②第三阶段的**目标条件化**(落点 `y_desired` + 旋转权重 `w_s`),并且**训练与评测用同一套采样分布**。
- **不抄 Ace 的稀疏**,理由是结构性的、不是口味:我们是 **on-policy PPO,没有回放、没有 HER、没有 event table 分层采样**;RSI 被 `HOPEPingPongActionBall.yaml:63` 的 `stand_start_prob: 1.0` 按裁定关掉;episode 是 10 s 多板而非单板。Ace 那四件支柱我们一件都没有。**照抄"只有上台才有钱",结果就是现役 §13 已经发生过的 `strike_opportunity_count=0`。**
- 一条可选的中间件:HER 式重标注在我们这里**不需要反解**,只需要**正向解**(拿实际拍状态正向算落点+旋转,回填成目标)。这条是 Ace 唯一一件能低成本移植到 PPO 之外的装置,建议记账,不排产。

### 20.3 反解还需不需要

【还需要。反解不是被替代,是被**降级 + 改岗**:从"在线唯一目标生成器"变成"离线出题器 + 阶段三目标生成器 + 可行性/课程分区器 + 评测拆账器"。】

**按阶段说清楚**

- **第一阶段(拍状态进目标、无球):完全不需要反解。** 目标 = clip 自己的拍状态,`answer_sphere` / LM 求解器 / 题库整条链都可以不上电。这是提案最大的工程红利:第一阶段可以脱离题库跑起来,而题库正是 §13 `strike_opportunity_count=0` 的病灶所在。
- **第二阶段(能不能击到球):需要的是"正向球飞行预测 + 交汇点求解",不是接触反解。** 判"能不能击到"只要球的可达交汇点与拍心距离;接触反解此时只做**可行性过滤**(这一板在拍状态包络内到底解不解得出来),防止出一堆无解题再复现"窗口开了但机会为零"。
- **第三阶段(上台 + 旋转条件化):必须要,而且是主角。** 这就是 SMASH 的 `p_hit, v_hit`——它的任务奖励对着的目标**就是**一个解析反解(Eqs. 27-32:反解阻力飞行 + 反解碰撞)出来的拍状态。我们要做同一件事,只不过要多解一维:出球旋转。

**反解的四个新角色(+ 一个不是反解的新活)**

1. **出题 / 可解性保证**:每一行题必须先证"在当前来球 + 机器人包络下存在拍状态解",否则不入库。这是防 §13 复发的唯一机制性手段。
2. **目标生成(给 σ 核一个稠密可跟踪的落点)**:阶段三的拍位/拍面/拍速目标由反解给,替掉阶段一的 clip 目标。σ 课程挂在这个目标上。
3. **可行性 / 课程分区**:SMASH 的自适应区域采样按工作空间分区;我们可以按**可解性余量**分区(球冠张角剩多少、摩擦锥余量多少),成功率低的区过采。这是我们比 SMASH 多的一手,因为我们有解析球冠。
4. **评测 / 拆账**:失败拆成"拍状态没到位"(策略问题)vs"拍状态到了但球没上台"(接触模型/标定问题)。没有反解,这两类失败在指标上不可分,§19.4 的静默错误就永远发现不了。
5. (**不是反解的新活**)HER 式重标注要的是**正向解**:由实际拍状态正算落点与出球旋转,回填成"其实你打成了这个目标"。建议记账。

**§19.4 五个缺陷的去留裁定**

| 缺陷 | 裁定 | 理由 |
|---|---|---|
| **1. 出球旋转在任何地方都不是目标**(全仓 `target_spin`/`desired_spin`/`w_plus_target` grep = 0 命中,ω⁺ 永远是副产物) | **必修,而且是阶段三的核心** | Franco 的第三阶段定义就是"落点 + 旋转权重条件化"(= Ace 的 `y_desired` + `w_s`)。这一条不修,第三阶段无法定义。 |
| **2. 自由解用"最省力"约定挑答案,各向同性 Tikhonov 偏向刷球角 0°**(`strike_spec_torch.py:148`,`w_speed=0.03`;numpy oracle `strike_spec_planner.py:199-201` 自陈 least-effort;`strike_spec_analytic.py:96` 记录实测退出时正则项是落点项的 **912 倍**) | **阶段一/二自动消失,阶段三必修——但换法** | 阶段一目标来自 clip,与 pin 约定无关。阶段三不要去"调"各向同性正则,而是**换 pin**:新增(a)"按目标旋转选点",(b)"按 73 库该 clip 的拍状态选点"(motion-matching pin)。(b) 是本次新提的:它同时解决缺陷 2 与"规划目标和风格库互相打架"这个新架构的固有张力——**让反解在球冠上挑离示范最近的那个点**。 |
| **3. 训练侧适配器把拍速方向硬钉死**(`stroke_adapt_torch.py:218` `v_r = q[:,2:3]·d_hat_w`;docstring `:251` 自陈"not a variable",唯一自由量是标量 s) | **阶段一自动消失(甚至是优点),阶段三必修** | 阶段一我们**就是要**继承 clip 的方向。阶段三若不放开,刷球角被冻死在示范继承的、对应 5.79 m/s 出球的那个值上,旋转指令无处落地。 |
| **4. 过网是事后闸门不是约束**(`strike_spec_torch.py:136-138` docstring 明说 net 不进 `ok`;`continuous_questions.py:846-847` 与 `gen_stage1_questions.py:962-965` 把不过网的行 `continue` 丢弃重抽) | **必修,优先级仅次于缺陷 1** | 被丢弃的恰恰是"必须加上旋才能既过网又落台"的那些行——阶段三的目标分布正是这些行。不修 = 阶段三的题库被静默切掉最有价值的一段。 |
| **5. 题库连来球旋转都没有**(`gen_stage1_questions.py:2` 表头 "no spin",`incoming_spin = np.zeros_like(v_in)`) | **一半自动消失,一半是数据问题不是反解问题** | 仿真里来球旋转可以直接当采样轴开出来,不需要反解。真正卡住的是两件与反解无关的事:(a) 接触模型在高旋区**没有标定**(venue 拟合有效域球速 1.7–6.4 m/s、旋转 p90 约 19 rps),(b) 采集端标记球在 360 Hz 下混叠上限约 22.9 rev/s 而目标覆盖 50–120 rps。**所以这一条要拆成"仿真轴现在就开"与"标定 blocked on 采集"两张单。** |

**还必须做但不在这五条里的**:`answer_sphere` 与含旋飞行反解至今 **NOT WIRED**(`strike_spec_analytic.py:116` docstring 自陈"Nothing in this repo calls this module yet";球冠在 `:395-427`,半角 `2·atan(μ)=53.13°` 在 `:403`/`:427`;含旋飞行反解在 `:288`)。阶段三开工前接线,是硬前置。

**排产结论**:反解的修复**不必挡住第一阶段发车**(这是新架构最大的好处),但**必须在第二阶段开球之前完成缺陷 4 + 接线,在第三阶段之前完成缺陷 1/2/3**。§19.4 原判"在 ≤12 m/s 静默正确、提速成功当天转为静默错误"不变,新架构只是把这个 deadline 从"提速那天"改成"阶段三那天",谁先到算谁。

### 20.4 落地方案(架构、证据、风险、迁移步骤、动作库规则)

## 二十、球拍状态进目标 + 三阶段重划:架构裁决(08-02;应 Franco 提案)

**方法**:1 提案解剖 + 2 外部证据抽取(Ace/Nature 与 SMASH 全文 pdftotext,17/17 逐条对抗核查)
+ 1 本仓代码取证(奖励栈 / 反解链 / 球接触模型 / 标记球定姿)+ 1 综合裁决。
外部两篇的**Supplementary Information 不在本地**(Ace 只有 11 页正文),凡涉及 SI 的地方本节都明确标注"无法核对"。

---

### 20.1 一句话裁决

**提案成立,按 SMASH 的拆法落地,但要改一个提法、并且不能替代 §16 的剂量修复。**

要改的提法:不是"把球拍状态放进 mimic 目标里",而是**"把球拍状态做成一条独立的任务通道,目标来源随阶段切换"**。
SMASH 的实弹配方正是把这两件事**拆开**:拍状态对着**规划器反解出来的目标**算误差(任务通道),
全身对着 mocap 库算误差(风格通道),而且**把腕关节从模仿项里剔除**,原文理由是"防止参考动作在击球时过度约束球拍"。
在第一阶段(无球)这两种提法**完全等价**,因为 clip 本身就是目标;
但到第二阶段球一泛化,"拍状态 = clip 的拍状态"这个等式就不成立了,塞在 mimic 里会自己跟自己打架。

不能替代的那件事:**死亡尖峰 −72.0 不会因为这套架构消失**。它等于 1200 步(24 s)完美 dense 收入,
约等于 update250 平均 episode(124.11 步 = 2.48 s)全部正收入的 10 倍。新架构给的 dense 拍收入是 +0.0x/步量级,救不回来。
**§16 的降尖峰必须同期独立做,否则新架构会跑成"站着不动地模仿"。**

---

### 20.2 三阶段职责重划(提案的正式表述)

| 阶段 | 任务 | 拍状态目标从哪来 | 主要收入 | 反解是否上电 | 通过判据(不含下阶段指标) |
|---|---|---|---|---|---|
| **一** | 学会人的挥拍(无球) | **73 库该 clip 自身的拍状态**(相位连续) | 全身模仿(dense)+ 拍位/拍面/拍速跟踪(dense,**权重高于普通身体节点**) | **不上电** | 窗内三通道误差分布 + 同步效率;**明确不看击球率** |
| **二** | 能不能击到球(球开始泛化) | **球条件化的交汇点**(正向球飞行预测求交)+ 反解只做可行性过滤 | 上一层保留并**降权** + **"碰到球"单独发钱**(Ace 的 R_hit 层) | **可行性过滤 + 出题** | 触球率;题库每行必须**先证可解**再入库 |
| **三** | 上台质量(落点 + 旋转) | **反解生成的拍状态目标**(落点 xy + 飞行时间/到达速度 + 出球旋转 ω⁺ 三元组) | 再加"上台 + 旋转"目标条件化层(Ace 的 `y_desired` + `w_s`) | **主角** | 上台率 + 旋转达成度;训练与评测**用同一套采样分布**(Ace 做法) |

三条硬规则贯穿三阶段:
1. **收入分层不许倒挂**:模仿 < 击中 < 质量。一律按每步 dense 口径(权重 ×dt 0.02)定价,阶段切换当天同步下调上一层权重。
2. **静态权重 + σ 课程**,不动态调权重。
3. **一个奖励项,两个目标源**。阶段切换切的是目标来源,不是新增/删除项——避免策略首次进入有效带时出现收入悬崖。

---

### 20.3 它治好了什么(逐条对账,不含推测)

**(1) §7 死核 —— 真治,但只治第一阶段。**
现役病灶:racket 三核的 σ 选的是**验收容差**而不是运行误差尺度
(`hope_training/whole_body_tracking/cfg/task/HOPEPingPongActionBall.yaml:113-118`:
`racket_position_std 0.075 m` / `racket_normal_std 0.262 rad` / `racket_velocity_std 0.5 m/s`),
而目标是球条件化采样点,窗口开启时刻拍到目标距离约 **0.70 m**,e/σ = 9.3 → exp(−87) ≈ 1.5e-38,
在 1–10 量级的每步收入里差 30 个数量级,**float32 下是精确 no-op**:"差 20 cm 的近失"和"差 70 cm 的挥空"的 advantage 完全一样。

新架构把目标换成**同一条 clip 自己产生的、相位连续的拍状态**,运行误差/饱和半径回到外部三家 RSI 下的 ~0.08 量级,
窗内失败之间**重新有排序信号**。**但疗效不跟着进第二阶段**:目标一旦球条件化,死核原样回来。
**因此 §7 的 P0(σ 阶梯)与 P1(粗细双核)不作废,只是改为"第二阶段开球当天必须同时上线"。**

**(2) §16"不挥拍是最优解" —— 治的是正确的那一半。**
现役盈亏平衡命中率 = 72/(72+33) = **69%**,一个 from-scratch 策略永远够不到,所以"不挥拍"在结构上就是最优解
(与 update250 实况自洽:exact 命中 0.23%、window 5.01%、capture/return = 0/0)。
新架构把挥拍的收入从**稀有事件**(落点 +32.98,而且 §13 已核实一次都没发放过)搬到**每步 dense 通道**,
**挥拍本身就有钱,不再需要赌命中率**——这是结构性修复,不是调参。
不治的:`death_penalty_weight: -3600.0`(`HOPEPingPongActionBall.yaml:125`,post-dt = −72.0/次)照旧压制一切冒险动作。**独立修。**

**(3) §13 `strike_opportunity_count=0` —— 不治,但被绕开,这既是好事也是陷阱。**
第一阶段不需要击球机会,所以必然跑得通。**陷阱:第一阶段跑通不能当第二阶段的放行证据。**
必须给第一阶段一个不含球的验收(见 20.2 表),否则等于拿一个必然通过的指标发车。

**(4) 顺带的结构收益**:第一阶段可以**完全脱离题库与反解**跑起来,而题库正是 §13 病灶所在;
"模仿 < 击中 < 质量"三层第一次在结构上分得开,每层单独定价、单独验收。

---

### 20.4 它自身的新风险(六条,每条配缓解)

| # | 风险 | 证据 | 缓解 |
|---|---|---|---|
| R1 | **腕过载**:加大拍面/拍位权重 = 加大对腕的约束,而腕是全机最小执行器(wrist_pitch/yaw **6 N·m**,扣重力摩擦后净 **4.82 N·m**) | §17.4:`bh_loop_c` 的 wrist_yaw 占比 14.9%,是 73 库自然中位 2.7% 的 **5.5×**;shoulder_yaw 被压到自然值的 1/6 | **高权重拍状态项只允许挂 73 库自然 clip,禁止挂 bh_loop_c 族。**"73 库优先"在这里不是偏好,是安全前提 |
| R2 | **把平庸质量钉成硬先验** | 73 库拍速中位 **5.50 m/s**(职业 13.13–14.69 m/s);同步效率中位 **53.4%**;法向利用率只有 **40–50%**;实测出球中位 **5.79 m/s**(职业 23 m/s 的 1/4) | 阶段二/三显式衰减 racket-vs-clip 权重或切目标源;**现在就**把同步效率与法向利用率做成 clip 准入排序,低效 clip 先重定时再当目标 |
| R3 | **控制点错位** | mimic 通道里的"拍"其实是 `right_wrist_yaw_Link` 的 link 原点(拍的所有 mesh 被 PhysX 并进该 body,见 `.../config/agibot_a3/hope_env_cfg.py:299`;该 body 在 `robots/agibot_a3.py:75-83` 的 `A3_UPPER_TRACKED` 里),而策略官方控制点是 `official_racket_site`(`.../mdp/action_ball_manifest.py:2166`) | 新项**必须显式绑 site**,不能复用 link 原点——两者差一个固定刚性偏移,姿态误差会被杠杆臂放大成拍心位置误差 |
| R4 | **分层倒挂**:"更大权重"顶穿击中层,分层塌回"只模仿" | §16 已证现役三层已塌成两层(模仿 + 不死) | 一律按每步 dense 口径(×dt 0.02)定价;阶段二上线当天同步下调阶段一权重;A/B 同 seed |
| R5 | **5 进 6 出的信息倒挂(最深)**:拍状态 5 自由度 → 出球 6 自由度,给定拍状态出球唯一;反过来同一落点对应**一族**拍状态(摩擦锥球冠,半角 `2·atan(μ) = 53.13°`,`.../mdp/strike_spec_analytic.py:395-427`) | 单点拍状态目标 = 替策略把球冠上的点**选死**;而选中的那个点是 clip 继承的刷球角,对应 5.79 m/s 级出球 | 这正是**反解不能扔**的根因(见 20.6);阶段三必须由反解在球冠上按目标旋转/按最近示范选点 |
| R6 | **观测合同漂移** | 阶段切换换目标源 = actor 观测语义变了(clip 派生 vs 球派生) | 走一次合同修订 + 部署 parity 复核,**不允许**靠改 params 悄悄换 |

**零新数据的好消息**:动作 npz 已逐帧存了**每个刚体**的世界位姿与速度
(`.../mdp/commands.py:135-137` 的 `body_pos_w/body_quat_w/body_lin_vel_w`,加载在 `:340-342`),
所以"拍状态对齐 clip"这条项**不需要重新采集、不需要重新 retarget**,只差把 site 偏移做进去。

---

### 20.5 外部佐证:Ace 与 SMASH 逐条(直接回答"只有上台奖励会不会引导不够")

**答:不够。而且 Ace 之所以能只用稀疏奖励,靠的是我们结构上没有的四件东西。**

**Ace(Sony,Nature)**
- 奖励**全部**是终局量,原文:"…all of which are calculated after the episode has finished, that is, as a function of the terminal state."
  全文 grep `shaping` / `dense` / `closest approach` / `time-to-contact` / `distance to` / `racket-to-ball` = **0 命中**。**确实没有任何稠密接近引导。**
- **但不是"只有上台才有钱"**:Eq. 10 是三级阶梯 `R_miss` < `R_hit`(碰到但没回过去)< `R_hit_return`。**"击到球"单独发钱**——正是我们分层里的中间层。
- 目标条件化只用在**一部分**策略上:`y_desired`(对方半台均匀采样)+ `w_reward=[w_p, w_s]`,`w_p∈[0,1]` 权位置、`w_s∈[-1,1]` 权落台时球绕 y 轴角速度(即旋转);`w_reward` 采样偏向稀疏与边界值,**比赛时沿用训练时同一分布**。Franco 的第三阶段与这条逐字对上。
- 它拿什么代替稠密引导:① **off-policy SAC + 回放**(稀有成功反复重用);② **HER**——把 `y_desired` 事后改写成实际落点、令 `w_p=1`,于是"碰到球但落点不对"直接变满分样本;③ **event tables 分层采样**(near miss / hit / returned / 高速高旋分桶);④ 起点分布很富(来球 KDE,发球:回球 = 3:7;机器人起点在静态中立位与**此前训练回合存下的 reset plan**之间二选一,自举式 RSI);⑤ **每回合只打一板**;⑥ 非对称 critic 吃真值球态 + 球态重建辅助损失。
- 架构上它**不是**端到端分层:一堆各自独立训练的**扁平** SAC 策略 + 一个**非 RL 训练**的外部选择器(固定/随机/规则/在精英对局上监督训练的分类器)。
- **诚实标注**:主文**从未解释**稀疏奖励如何从零 bootstrap;若有说明应在 SI 1.4.1–1.4.8,**本地无 SI,无法核对**。

**SMASH(humanoid 乒乓)**
- 任务奖励是**每步的窗门控指数核**:`r_j(t) = exp(−e_j(t)/σ_j(t)) · I[τ_t ∈ W_j]`,j ∈ {位置, 姿态, 速度};
  窗宽**位置 0.02 s、姿态/速度 0.1 s**;总任务奖励 = `w_pos·r_pos + w_ori·r_ori + w_vel·r_vel + r_succ`,`r_succ` 是三通道阈值同时满足的稀疏 bonus。(三个 w 的**数值全文未公布**,plain 与 -layout 两种抽取都确认。)
- 误差对着的是**反解出来的规划目标** `p_hit, v_hit`(§VI.F Eqs. 27-32:先反解线性阻力飞行得到所需出球速度,再反解无摩擦法向碰撞得到拍面法向与法向速度),**不是** mocap clip。
- **没有任何球结果奖励**:无落点、无过网、无旋转;并自陈"不显式建模球旋转"是已知局限。
  → **"只靠拍状态跟踪当 RL 目标"这件事,SMASH 就是实弹证据。**
- **σ 课程是它的命门**:σ_j 随在线跟踪误差自适应收紧。去掉:SR **86.38% → 22.60%**,位置误差 4.42 cm → **11.94 cm**,姿态误差 4.17 → **35.49**;论文归因"初始 σ 太严 → 任务奖励拿不到 → 根本学不起来"。
  另一件:自适应区域采样(按台面击球区分区、成功率最低的区过采),去掉后 86.38% → 82.72%。
- **口径纠错(重要,以后引用照此)**:86.38% 是**仿真里的拍状态跟踪成功率**(Ep<0.04 m 且 Eo<0.05 且 Ev<0.5),**不是回球率**。
  真实回球率 `SR_return`:MoCap **90%**(18/20)、自我中心相机 **60%**(12/20);642 板 50 分钟连续测试触球率 93.7%、**成功回球率 59.7%**(正手 66.7%、反手 **38.9%**)。反手这条与 §17.3"反手弱是示范数据自带的天花板"同向。
- 架构上**明确拒绝分层**:"Instead of adopting a hierarchical controller … we use a simple motion matching scheme"——单个 PPO 全身策略 + 非学习的最近邻 clip 检索(用规划目标去库里匹配)。**并把腕关节排除在模仿项外。**

**结论与抄法**
- **抄 SMASH 的骨架**:①拍状态三通道窗门控指数核当任务主通道(我们已有同形核:`.../mdp/hope_rewards.py:118`(位置)/`:145`(速度)/`:158`(法向),连三阈值 bonus 都已存在 `hope_rewards.py:3421` `racket_strike_success`,等价于 Eq.13 的 `r_succ`);②**σ 课程**——外部唯一"去掉就崩"的量化证据,且与钦定的"静态权重 + σ 课程"逐字一致,`racket_*_std` 键已全程接线,**零代码**;③自适应区域采样;④**窗宽分层照抄**:位置窗窄、姿态/速度窗宽(我们现在三通道同窗)。
- **抄 Ace 两件**:①**三级 outcome 阶梯**(没碰到 < 碰到 < 碰到且回台)→ 第二阶段"碰到球"必须单独发钱,不等上台;②第三阶段**目标条件化**(落点 + 旋转权重),训练与评测同分布。
- **不抄 Ace 的稀疏**,理由是结构性的:我们是 **on-policy PPO,无回放、无 HER、无 event table**;RSI 按裁定关闭(`HOPEPingPongActionBall.yaml:63` `stand_start_prob: 1.0`);episode 是 10 s 多板而非单板。Ace 的四根支柱我们一根都没有。**照抄"只有上台才有钱" = 复现现役 `strike_opportunity_count=0`。**
- 记账不排产:HER 式重标注在我们这里**不需要反解,只需要正向解**(由实际拍状态正算落点+旋转回填目标)——是 Ace 唯一低成本可移植的装置。

---

### 20.6 反解的新定位:降级 + 改岗,不是退休

**四个新角色(+ 一个不是反解的新活)**
1. **出题 / 可解性保证**:每行题先证"在当前来球 + 机器人包络下存在拍状态解"才入库。这是防 §13 复发的唯一机制性手段。
2. **目标生成**:阶段三的拍位/拍面/拍速目标由反解给(= SMASH 的 `p_hit, v_hit`),σ 课程挂在它上面。
3. **可行性 / 课程分区**:按**可解性余量**(球冠张角剩多少、摩擦锥余量多少)分区,成功率低的区过采。这是我们比 SMASH 多的一手,因为我们有解析球冠。
4. **评测 / 拆账**:把失败拆成"拍状态没到位"(策略问题)vs"拍状态到了球没上台"(模型/标定问题)。没有它,§19.4 的静默错误永远发现不了。
5. (**不是反解**)HER 式重标注用**正向解**。

**§19.4 五缺陷去留**

| 缺陷 | 裁定 | 说明 |
|---|---|---|
| 1. 出球旋转在任何地方都不是目标(`target_spin`/`desired_spin`/`w_plus_target` 全仓 grep **0 命中**,ω⁺ 永远是副产物) | **必修,阶段三的核心** | 第三阶段的定义就是落点 + 旋转条件化,不修则第三阶段无法定义 |
| 2. 自由解用"最省力"约定挑答案,各向同性 Tikhonov 偏向刷球角 0°(`.../mdp/strike_spec_torch.py:148`,`w_speed=0.03`;numpy oracle `hope_ws/src/hope_planner/strike_spec_planner.py:199-201` 自陈 least-effort;`.../mdp/strike_spec_analytic.py:96` 记录实测退出时正则项是落点项的 **912 倍**) | **阶段一/二自动消失;阶段三必修,但换法** | 不要去"调"正则,而是**换 pin**:(a) 按目标旋转选点;(b) **按 73 库该 clip 的拍状态选点(motion-matching pin,本节新提)**。(b) 同时化解"规划目标 vs 风格库互相打架"这个新架构的固有张力——让反解在球冠上挑离示范最近的那个点 |
| 3. 训练侧适配器把拍速方向硬钉死(`.../mdp/stroke_adapt_torch.py:218` `v_r = q[:,2:3]·d_hat_w`;docstring `:251` 自陈 "not a variable") | **阶段一自动消失(甚至是优点),阶段三必修** | 阶段一我们就是要继承 clip 方向;阶段三不放开则刷球角冻死在 5.79 m/s 级示范上,旋转指令无处落地 |
| 4. 过网是事后闸门不是约束(`strike_spec_torch.py:136-138` docstring 明说 net 不进 `ok`;`.../mdp/continuous_questions.py:846-847` 与 `gen_stage1_questions.py:962-965` 把不过网的行 `continue` 丢弃重抽) | **必修,优先级仅次于缺陷 1** | 被丢弃的恰恰是"必须加上旋才能既过网又落台"的行,而那正是阶段三的目标分布 |
| 5. 题库连来球旋转都没有(`gen_stage1_questions.py:2` 表头 "no spin",`incoming_spin = np.zeros_like(v_in)`) | **一半自动消失;剩下一半是数据问题,不是反解问题** | 仿真里来球旋转直接开成采样轴即可;真正卡住的是 (a) 接触模型高旋区**未标定**、(b) 采集端 22.9 rev/s 混叠。拆成"仿真轴现在开"与"标定 blocked on 采集"两张单 |

**不在五条里但同样是硬前置**:`answer_sphere` 与含旋飞行反解至今 **NOT WIRED**
(`.../mdp/strike_spec_analytic.py:116` docstring 自陈 "Nothing in this repo calls this module yet";
球冠在 `:395-427`,半角 `2·atan(μ) = 53.13°` 在 `:403`/`:427`;含旋飞行反解在 `:288`)。

**排产结论**:反解**不挡第一阶段发车**(这是新架构最大的工程红利),
但**第二阶段开球前**必须完成缺陷 4 + 接线,**第三阶段前**完成缺陷 1/2/3。
§19.4 原判"≤12 m/s 静默正确、提速成功当天转静默错误"不变,新架构只是把 deadline 从"提速那天"改成"阶段三那天",谁先到算谁。

---

### 20.7 球接触模型:切向(造旋)标定的现状与缺口

**先说结论:切向分支**存在**、四处实现一致,不是"只有法向恢复系数"。**
公式(四处逐字相同):`u = v⁻ + ω⁻×(−Rn) − v_r`;`u_n=(u·n)`;`u_t = u − u_n·n`;`cosθ = |u_n|/hypot(|u_t|,u_n)`;
`s = clip((a_t + b_t·cosθ)|u_t|, 0, μ(1+e)|u_n|)`;`Δv_t = −s·unit(u_t)`;`Δω = −(1/(cR))(n×Δv_t)`。
实现处:`hope_training/ball_physics_fit/contact_model.py:19-45`(numpy oracle)、
`.../tasks/table_tennis/physics/spin_contact.py:56-99`(torch,台+拍通用)、
`.../tasks/tracking/mdp/virtual_ball.py:524-568`(训练侧拍接触,喂 Tier-1 奖励)、
`.../tasks/tracking/mdp/physical_ball.py:335-376`(代码驱动的台面反弹,因为 PhysX 的 restitution 表达不了拟合模型)。

**逐系数标定状态**

| 系数 | venue(07-03,**现役默认**) | OptiTrack(07-30,含真旋转) | 状态 |
|---|---|---|---|
| 法向恢复 `e` | 台 `e_eff=0.9215`;拍 `e = 0.759·exp(−0.0441·u_n)` | 台 0.9215(采纳 venue);拍 `e = 0.8636·exp(−0.07033·u_n)` | **真拟合**,不需要旋转即可定 |
| 台面切向 `a_t` | **0.369 —— 不是 venue 自己的拟合**,注释写明 "RETAINED FROM v0"(更早一次 101 次弹跳的 OptiTrack grip 拟合),venue 自己的切向重拟合在 9 mm 噪声下退化 | **0.275**,60 次"旋转完整"的弹跳,**真正含旋** | venue 现役值**既非本管线拟合、也非含旋** |
| 球拍切向 `a_t` | **0.52** | **0.637**,24 次含旋击球联合拟合;把 `w_in` 强制置零则 `dv_rms` 由 1.11 → 1.80 m/s(**+62%**,即含真旋转使残差降 38%),配置注释称这是"恢复出来的旋转是物理的最干净的单一验证" | **口径要精确**:venue 的 0.52 是**"来球旋转有用、出球旋转(旋转传递)未验证"**,不是"完全无旋拟合"——来球旋转由触球前 35 帧飞行弧算出并**强制入模**(`ball_physics_fit/stage1_segments.py:164-170`),行级若 `spin_in_ok=False` 直接丢弃;不可信的是**触球处的旋转变化通道**(`configs/ball_physics_venue.yaml:169` `strike_dw_scale: 0.22`,即只读到真值的 ~22%),所以联合拟合的 Δω 残差被置零(`ball_physics_fit/stage2_fits.py:170-206`)。venue 自己的措辞"Spin-transfer prediction remains UNVALIDATED at strikes"是精确的 |
| `b_t` | 0.0 | 0.0 | **不可辨识**:近法向入射下只有 `a_t + b_t·cosθ` 这个组合可辨,拆分任意,全部折进 `a_t` |
| `μ_safety` | 台 2.0 / 拍 0.5 | 同 | **假设值**,库仑上限在观测数据里**从未起作用(0/130 次击球)**,选它只为"让擦击外推别发疯" |
| `inertia_coeff c` | 2/3 | 2/3 | 教科书薄壳假设,**从未针对这颗球验证** |
| `ball.mass` | 0.0034 kg | 0.0034 kg(**本次未称重**) | 文档自陈"本文件最大的无支撑假设",建议称重(约 10 秒) |
| `air.rho` | 1.20 | 1.20 | 默认值,未记温压 |
| 飞行 `k_d` / `k_m` | 真拟合(用测得旋转) | 真拟合 | **全流水线最强的含旋结果**:两套独立动捕互校 `k_d` 差 0.6%、`k_m` 差约 9% |
| Magnus `cl_max=0.55` | 文献值(Ito & Kamijima 2025) | 同 | 超出拟合覆盖;`cl_slope` 是为匹配低旋比下的 `k_m` 代数反推,非独立拟合 |

**两个必须记档的事故**

1. **现役默认加载的是 venue 那份,不是含旋重拟合那份。**
   `.../mdp/virtual_ball.py:75-93` 的 `default_venue_yaml_path()` 与 `.../physics/params.py:91-108` 的 `default_yaml_path()`
   在 `$HOPE_BALL_PHYSICS_YAML` 未设时都解析到 `configs/ball_physics_venue.yaml`,
   而**全仓 grep 没有任何 shell/launch/config 设过这个变量**。
   于是训练的 Tier-1 拍接触奖励与 `physical_ball` 的台面反弹,跑的都是
   **旋转传递未验证的拍 `a_t=0.52` + 非本管线拟合的台 `a_t=0.369`**,
   尽管 `docs/ball_physics_optitrack_20260730.md` 写着新常数"ACTIVE for the simulator, via the config-path switch"——**那个开关在仓里从没被拨过。**
2. **`configs/ball_physics_optitrack_20260730.yaml` 自相矛盾**:文件尾部 `#spin — RETRACTED AND CORRECTED` 段(`:252-255`)
   声称"`k_m`、两个 `a_t/b_t` 块与旋转传递模型仍是继承的 venue 值,因为重拟合尚未完成",
   但同一文件前面的 `contact.table`(`:153`)与 `contact.paddle`(`:178`)已经写着 "MEASURED HERE / MEASURED WITH REAL SPIN" 的重拟合结果。
   git blame 显示两者是**同一次提交**引入的,不是后来漏改;配套文档 `docs/ball_physics_optitrack_20260730.md` §6.4 复述了同一段过时说法。**数值以块内为准,这段话作废。**

**对新架构的直接含义**:阶段三要按旋转条件化发奖,而**旋转传递这条通道在现役配置里是未验证的**,
且 venue 拟合有效域只有球速 1.7–6.4 m/s、`u_n` 2.2–8.8 m/s、旋转 p90 约 19 rps。
**阶段三开工前两件事必须做**:(a) 把配置开关拨到含旋那份并跑一次逐字节 A/B;(b) 给接触/飞行模型加**有效域告警并进摘要**(符合"WARN 必进摘要")。

---

### 20.8 标记球混叠上限 22.9 rev/s:修正结论

**它是什么**:这颗 8 标记球上**实测到的最小成对角间距 45.7°**(实测范围 45.7–152.5°)的**一半** = 22.85°/帧;
在 **360 Hz** 下 °/帧 与 rev/s 数值恰好相等,故 ≈ **22.9 rev/s**。
出处只有两处原始:`configs/ball_physics_optitrack_20260730.yaml:242-250` 与 `docs/ball_physics_optitrack_20260730.md` §6.2-6.3;
下游文档只是转引。

**它不是什么(三条修正)**
1. **不是对称图案造成的信息论级二义**。这颗球的标记布局是**不对称的**(28 个成对距离基本互异),
   真正对称的星座才会有"转过对称角后任何算法在任何帧率下都永远分不出"的硬二义。**这个数是在真实不对称几何上算的。**
2. **它是"逐帧对应连续性"的界**:生产用的是 `hope_training/ball_physics_fit/optitrack/ball_orientation.py:169-230`
   的 `solve_orientation_chained` —— 用上一步角速度**预测**下一帧姿态、再做匈牙利最近邻配对的**局部顺序**匹配。
   一旦单帧转角超过最近两标记角间距的一半,最近邻就可能把两个标记认反,而且它是**链式**的,一步错可能污染后续。
   理论上一个全局暴力匹配器可以利用不对称性突破这个界,**但我们部署的算法不是那种。**
3. **代码/文档三处不一致,一并记档**:
   (a) 生产函数 `solve_orientation_chained` 的返回里**根本没有任何混叠字段**(只有 `quat/step_rms/omega/run_id/radius_m`);
   计算 `ang_limit`/`alias_risk`/`alias_spin_limit_rev_s` 的是**明确标注"仅供参考"、生产不用**的旧 `solve_orientation`(`:233-289`,公式在 `:256`),
   而模块顶部 docstring 却宣称"`spin_alias_risk` 会标记超界帧"——**描述的是没在跑的代码路径**。
   (b) 同一 docstring 的示例算式用的是 ~11.6 mm 弦长 → ~33.6° → **~17 rev/s**,比最终定稿的 22.9 rev/s **更紧**,是未更新的早期估计。
   (c) 全仓找不到任何**签入的**脚本能从标记数据重算出 "45.7–152.5°" 或 22.9 —— 这个数**在仓内不可复现**,只有 yaml 注释与文档在陈述它。

**结论与修法**:22.9 rev/s 是**当前部署算法的实用上限**,而观测到的每 take 旋转峰值 20.4–22.7 rev/s **正贴着这条线**,
docs 判定"更快的旋转已经在往下折叠",不是巧合。
目标要覆盖 **50–120 rev/s**,可行修法只有两条,且都与算法无关:**提高帧率**,或**加大最小标记间距**(上限与最小间距成正比)。
**这是新一轮采集唯一的技术卡口,必须在开会话之前解决。**

---

### 20.9 动作库规则(本节钦定,进准入门)

1. **73 库优先。** 它是真人打球、自带真实时序与真实发力结构。凡是要给"高权重拍状态目标"的臂,目标 clip **必须**取自 73 库自然 clip。
2. **不编辑人类的发力结构。** 文献与 73 库实测双否"人靠小臂/前臂发力":73 库触球帧贡献分解为
   肩 pitch 21.3% + 肘 18.6% + 肩 yaw 18.4%(合计 60–71%),腕三轴合计约 20% 主要管拍面控制;
   而 `bh_loop_c` 这种肩优先人工改造把 shoulder_yaw 峰值压到 **54°/s**(自然反手中位 465°/s 的 1/9),
   靠 wrist_yaw **344°/s**(自然 1.5×)硬凑拍速,把负载挤进**全机最小的 6 N·m 执行器**。**这类改造不得再做,已有的记档为已知副作用。**
3. **只允许两种加工**:
   (a) **选路径** —— 一个 style 保留多条路径,在线按**剩余时间与 torque margin** 选;
   (b) **TOPP 定时序** —— 只重排时间律,**造不出新几何**。
   V2.2 采集脚本(2026-07-22)已裁定并继续有效:**不给演员数值 target**;**`v_racket_hit` 不是 target**(human 侧是测量值,A3 侧是 TOPP 派生值)。
4. **5 个动作不用加速版。** 纯时间缩放下 ω∝k、α∝k²、τ∝k²,**全身第一堵墙是 waist_roll 力矩,k_max = 1.19(净)/1.43(盒子)**,
   连 ×1.5 都到不了;换成线性力矩-转速包络(`ω/ω_max + τ/τ_max ≤ 1`)且分母用净预算(waist_roll 46.0 → **31.87 N·m**)后,
   **73 条里已有 15 条超限(20.5%)**。提速的正解是**同步性改造**:实测同步效率中位 **53.4%**,
   把各关节在同一次前摆里**已经达到过的**角速度对齐到触球帧,拍速 **5.50 → 7.85 m/s(+44%)**,**任何关节的峰值速度与力矩都不增加一分**。
5. **任何重定时脚本必须挂 waist_roll 净力矩闸门**(分母 31.87 N·m,判据用线性包络而非两个独立 ≤1);
   限定:线性包络是三角形,比外部三库实际用的梯形更保守,**拿到厂商 τ_sat 之前不要据此砍 clip**。
6. **高权重拍状态项的 clip 准入门(新增)**:先按同步效率与法向利用率排序,低效 clip 先重定时再当目标;
   两个已知数据缺陷必须先修,否则触球帧代理不可信、拍状态目标的时间锚就是错的:
   (a) 全部 73 条每帧 `head_yaw_joint`/`head_pitch_joint` 恒为 0.000(几乎肯定是 GMR retarget pin 住了头部 DOF,找 Jiayi 核);
   (b) 6/73 条存在第二个 >90% 高度、距全局峰值 >10 帧的次峰,其"触球帧"可能选到了引拍。

---

### 20.10 迁移步骤(按依赖排序,全部新臂键、默认逐字节不变)

| # | 步骤 | 内容 | 前置 | 验收 |
|---|---|---|---|---|
| **M0** | **降尖峰(与本架构并行,不可省)** | 按 §16 处理 `death_penalty_weight: -3600.0`(`HOPEPingPongActionBall.yaml:125`,post-dt −72.0)与 barrier 剂量 | 无 | 回报分布不再双峰;单步最大罚项 / 折扣视野收入回到外部三库的 2% 量级(现在是 1200%) |
| **M1** | **拍状态项落地(阶段一)** | 新增绑 `official_racket_site` 的拍位/拍面/拍速**对 clip** 跟踪项;默认权重 0.0(逐字节不变),新臂 yaml 开启 | M0;20.9 第 6 条准入门 | 三通道窗内误差分布有连续下降趋势;**不看击球率** |
| **M2** | **σ 课程 + 窗宽分层** | 静态阶梯收紧(拍面 0.60→0.40→0.262 rad,拍位 0.30→0.15→0.075 m);窗宽改为位置窄(0.02 s)/姿态速度宽(0.1 s),对齐 SMASH | M1 | 与固定 σ 的同 seed A/B;外部先例:去掉自适应 σ 会让 SR 86.38%→22.60% |
| **M3** | **反解补齐(挡在开球之前)** | 缺陷 4(过网提升为约束)+ 接线 `answer_sphere`(`strike_spec_analytic.py:395-427`)与含旋飞行反解(`:288`);题库每行先证可解 | M1 | 题库不再 `continue` 丢弃需要旋转的行;`strike_opportunity_count > 0` |
| **M4** | **第二阶段开球** | 目标源从 clip 切到球条件化交汇点;**同时**上 §7 的 P0/P1(σ 阶梯 + 粗细双核);加 Ace 的 `R_hit` 层("碰到球"单独发钱);拍状态-对-clip 权重降档;腕从模仿项放松(SMASH 做法) | M2 + M3;观测合同修订 | 触球率;**不允许**用第一阶段指标放行 |
| **M5** | **第三阶段上台 + 旋转** | 目标升成三元组(落点 xy + 飞行时间/到达速度 + ω⁺);新增第 4 种 pin(按目标旋转)与第 5 种 pin(按 73 库该 clip 拍状态,motion-matching);缺陷 1/2/3 修完;接触模型配置开关拨到含旋那份 + 有效域告警 | M4;20.7 两件事;20.8 混叠卡口(若要真机标定) | 上台率 + 旋转达成度;训练与评测同分布(Ace 做法) |

**风险登记(排产用)**

| 风险 | 触发条件 | 早期信号 | 应对 |
|---|---|---|---|
| 第一阶段"假通过" | 拿必然通过的指标发车 | 击球率被写进阶段一验收 | 阶段一验收表里**显式禁列**击球率 |
| 死核在 M4 原样复发 | 只切目标源、没上 σ 课程 | 窗内三通道 reward 回到 1e-30 量级 | P0/P1 与 M4 **同臂上线**,不分两次 |
| 腕过载 | 高权重拍面项挂在 bh_loop_c 上 | wrist_yaw 贴限/豁免请求 | 20.9 第 1 条准入门 |
| 平庸质量固化 | 阶段二/三不降 racket-vs-clip 权重 | 拍速停在 5.5 m/s、法向利用率不涨 | M4 权重降档写进配方,不靠人记得 |
| 旋转奖励跑在未标定域 | M5 用现役 venue 配置发奖 | 有效域告警(上线后) | 20.7 的 (a)(b) 两件是 M5 的硬前置 |
| 采集端卡口 | 新一轮动捕未解 22.9 rev/s | 每 take 峰值贴 22.7 rev/s | 提帧率或加大标记间距,**开会话前**解决 |

---

### 20.11 本节顺带修正的既有说法(以本节为准)

1. **"venue 的拍 `a_t=0.52` 是无旋拟合"** → 应为 **"来球旋转已入模且必需,未验证的是触球处的旋转传递"**(`strike_dw_scale: 0.22`)。venue yaml 自己的英文措辞是精确的,中文转述过松。
2. **"SMASH 成功率 86%"** → 那是**仿真里的拍状态跟踪 SR**;真实**回球率**是 90%(MoCap)/60%(自我中心相机)/59.7%(642 板连续测试,正手 66.7%、反手 38.9%)。
3. **"Ace 只有上台才有奖励"** → Ace 是**三级阶梯**,"碰到球"单独发钱;它没有的是**稠密接近引导**(这条如实),代替品是 off-policy 回放 + HER + 事件分层采样 + 富起点 + 单板 episode。
4. **"22.9 rev/s 是对称图案的信息论极限"** → 是**部署用链式最近邻匹配的逐帧对应连续性界**,几何是**不对称**的;并且生产函数根本不输出混叠标记,模块 docstring 里的 ~17 rev/s 是过时估计,这个数在仓内**不可复现**。
5. **`configs/ball_physics_optitrack_20260730.yaml:252-255`** 的"重拟合尚未完成"整段作废(同一文件同一提交内自相矛盾),`docs/ball_physics_optitrack_20260730.md` §6.4 同步作废。


---

## 二十一、Policy 容量尽调:225-D 输入要不要扩网(08-02)

**方法**:3 抽取(我方容量与约束 / 13 个外部 repo 网络普查 / 四篇论文 + RL 扩容文献)+ 3 对抗核查
(22 修正)+ 1 裁决。参数量一律按 `Σ(in×out+out)` 手算复核。

### 21.1 裁决

不要扩容。现在扩网络是把钱花在没有故障的零件上。

一句话结论：我们学不好的原因，证据几乎全部指向"信号不足 / 优化被压制"，不是"网络太小"。我给"网络容量是当前主要瓶颈"的概率 ≤10%。

三条硬证据（都来自我们自己的尽调 doc 和代码，不是外部类比）：
1. 击中层收入恒为 0（§16：strike_opportunity_count=0）。网络再大也拟合不出一个从来没有出现过的目标。这是容量假说的直接证伪：容量只有在"有梯度但拟合不上"时才是瓶颈。
2. 回报分布被 -72 死亡尖峰支配，advantage 归一化后 dense 项的有效梯度约为死亡方向的 1/500（§16）。这是梯度方向被压缩到 1/500，不是表达能力不够。扩容会把这个被压缩的方向复制到更多参数上，压缩比例不变。
3. 窗内 exp 核在大误差处梯度精确为零（§7）。零梯度乘以任何参数量还是零。

两条外部对照证据说明 [512,256,128] @ 225-D 在同类任务上绰绰有余：
- HITTER（arXiv 2508.21043v2 §V.B.3，Unitree G1 29 自由度打乒乓，我们这条线的直接母本）actor obs 只有 104 维、网络就是 [512,256,128]、约 22.2 万参数、无历史堆叠，零样本上真机成功。我们 A225 是 225 维、28.4 万参数——比母本还大 28%。
- 生态普查：从 obs=42 到 obs=705（约 17 倍跨度），[512,256,128] 是近乎唯一的默认值，没有任何一个 locomotion/tracking 仓库把宽度随 obs 维度往上调（beyondmimic 160-D、mjlab 160-D、unitree_rl_lab 480-D、legged_gym 235-D、ASAP 576-D、XBot-L 705-D 全是这一个网络）。

关于"225-D 是否要求更大网络"：不要求，而且这个提问方式本身就错了。194→225 只多 31 列，第一层只多 31×512 = 15,872 个参数，占 actor 总量的 5.6%；而那个 512 宽的第一隐层本身就占了 actor 参数的 40.7%（115,712 / 283,935），actor+critic 合起来第一层占 45.6%（279,040 / 611,616）。换句话说：参数量几乎不看输入宽度，只看隐层宽度。而且 A225 的 225 列里有 154 列（68.4%）是 achieved-vs-teacher 成对镜像（base 15+15、joint pos 31+31、joint vel 31+31），这是刻意的残差/模仿结构——要学的函数更接近"恒等 + 修正"，比从零编码一个任务更容易表示，不是更难。

关于扩容文献的适用性（这条最容易踩坑）：SimBa（ICLR 2025）、BRO（NeurIPS 2024 spotlight，注意不是 ICLR 2025）、Dormant-Neuron/ReDo（ICML 2023）这三篇"扩容有效"的核心证据全部是 off-policy（SAC / TD-MPC2 / DQN / DrQ，带 replay buffer、高 UTD）。SimBa 唯一的 on-policy PPO 实验跑在 Craftax（离散生存游戏），不是连续控制。真正对口的是 Andrychowicz et al.（arXiv 2006.05990，ICLR 2021）：在 5 个 MuJoCo 连续控制环境上扫了宽度 {16,32,64,128,256,512} × 深度 {1,2,4,8}，结论是 actor 宽度最优值依环境而定且非单调——太窄和太宽都掉性能（HalfCheetah 最优 policy 宽度只有 16-32 单元），而 critic 加宽没有下行风险；深度 2 层在所有环境都够用，4/8 层无收益。这篇唯一对口的 on-policy 证据是反对盲目加宽 actor 的。

另外两个我们仓库特有的、必须写进裁决的约束：
- critic 在部署侧是免费的（exporter.py 只 trace self.actor），但在优化侧不免费：rsl_rl 用单个 Adam 覆盖 actor+critic 全部参数、单个全局 max_grad_norm=1.0、单个 adaptive-KL 驱动的学习率（desired_kl=0.01, lr=0.001）。加大 critic 会稀释/重塑 actor 也在其中的那份梯度范数预算和 KL 学习率。"critic 随便扩"在别的框架成立，在我们这里不成立。
- 任何 hidden_dims 改动都会让 checkpoint 形状不匹配（strict load 报错），等于放弃全部 warm start，从零重跑 2 万-2.5 万 iter。这是一次 GPU-周级别的代价，只应该在诊断指标明确指向欠拟合之后才付。

顺带纠正/确认前提：actor 末层 weight 零初始化是真的，但不在 rsl_rl 的 ActorCritic 里，而在 scripts/train.py 的 _apply_action_ball_fresh_policy_bootstrap（定义 7010 行、置零 7062 行、断言 7065 行、调用点 16237 行）——不是给定的 5004 行，请把 doc 里的行号改过来。init_noise_std=0.02 是 CLI 强制覆盖（ppo.yaml 自身默认仍是 1.0）并在 receipt 里硬校验。

### 21.2 全景对照表

## 观测维度 × 网络宽度 × 参数量 × 历史/RNN 全景对照

参数量口径统一为「actor（含输出层）」，按 `Σ(in×out+out)` 手算；标 ~ 的为估算或论文未给全。

| 来源 | 系统 / 任务 | actor obs 维 | 动作维 | actor 隐层 | 激活 | actor 参数量 | 历史 / RNN | 备注 |
|---|---|---|---|---|---|---|---|---|
| **我们** | A225 actor（现役诊断对） | **225** | 31 | [512,256,128] | ELU | **283,935** | 无（仅 prev action 31-D） | 154/225=68.4% 是 achieved-vs-teacher 成对镜像 |
| **我们** | A225 critic（318-D 特权） | 318 | 1 | [512,256,128] | ELU | **327,681** | 无 | A225 自有 ABI；与 L194 的 318 宽度只是巧合，不是共享合同 |
| **我们** | L194 actor | 194 | 31 | [512,256,128] | ELU | 268,063 | 无 | 与 A225 差 31 列 → 第一层仅差 15,872 参数（5.6%） |
| 我们（备选） | actor [512,512,256] | 225 | 31 | [512,512,256] | ELU | 517,663 | 无 | 合计 actor+critic 1,075,232 = 1.76× |
| 我们（备选） | actor [1024,512,256] | 225 | 31 | [1024,512,256] | ELU | 895,519 | 无 | 合计 1,878,560 = 3.07× |
| 论文 | **HITTER**（arXiv 2508.21043v2，Unitree G1 29-DoF 乒乓） | **104** | 29 | **[512,256,128]** | 未述 | 221,725 | **无历史** | 与我们同任务同机型；零样本上真机；论文明写三隐层 512/256/128 |
| 论文 | **PACE**（arXiv 2509.21690v3，Booster T1 乒乓） | **410**（82×H=5） | 21 | [512,512,128] | 未述 | 541,461 | 历史 H=5 | 另有 [64,64] 球轨迹预测小网络；4096 并行环境 |
| 论文 | **SMASH**（arXiv 2604.01158v1，G1 乒乓 + 自视觉） | 未给 | 未给 | **论文未给** | 未给 | — | actor 用带噪历史 | 15 页全文无隐层尺寸/激活/环境数；视觉是 YOLO+AprilTag+EKF 经典管线，非 CNN 进策略 |
| 论文 | **Sony Ace**（Nature s41586-026-10338-5，8-DoF 机械臂） | N 步带噪测量历史（N 未给） | 16 | **Supp. Table 8，正文无** | — | — | 历史 + 球态嵌入 | 唯一 off-policy（SAC，异步并行采集）；带球态重建辅助损失 |
| repo | beyondmimic G1 tracking | 160 | 29 | [512,256,128] | ELU | 250,397 | 无 | 我们 critic 的历史祖先 |
| repo | mjlab G1 tracking | 160 | 29 | [512,256,128] | ELU | 250,397 | 无 | 显式声明是 beyondmimic 重实现 |
| repo | unitree_rl_lab G1-29 locomotion | 480（96×5） | 29 | [512,256,128] | ELU | 414,237 | 历史 5（扁平堆叠） | |
| repo | unitree_rl_lab G1-29 mimic | 154 | 29 | [512,256,128] | ELU | 247,325 | 无 | |
| repo | legged_gym anymal_c rough / A1 | 235 | 12 | [512,256,128] | ELU | 286,604 | 无 | 含 187 维高度扫描 |
| repo | legged_gym anymal_c **flat** | 48 | 12 | **[128,64,32]** | ELU | **17,004** | 无 | ⚠️ 同仓库内主动**缩小**网络的实例（obs 变小则网络变小） |
| repo | humanoid-gym XBot-L | **705**（47×15） | 12 | [512,256,128]（critic [768,256,128]） | ELU | 527,244 | 历史 15 | 全普查中 obs 最宽仍用同一网络；actor/critic 非对称宽度 |
| repo | IsaacGymEnvs AnymalTerrain (MLP) | 188 | 12 | [512,256,128] | ELU | 262,540 | 无 | yaml 里有被注释掉的 rnn 块 |
| repo | IsaacGymEnvs AnymalTerrain (LSTM) | 188 | 12 | [512] + LSTM(256,1层) | ELU | ~887,308 | **RNN** | 同任务同 obs，加 RNN 后参数 ×3.4 |
| repo | IsaacGymEnvs ShadowHand full_state | 211 | 20 | [512,512,256,128]（共享 trunk） | ELU | ~538,004 | 无 | 灵巧手，非 locomotion |
| repo | IsaacGymEnvs ShadowHandOpenAI_FF（真实默认链） | 42 | 20 | **[400,400,200,100]** | ELU | 279,920 | 无 | ⚠️ 第三种独立架构；其 central_value critic 用 [512,512,256,128] 吃 211-D |
| repo | IsaacGymEnvs ShadowHandOpenAI_LSTM | 42 | 20 | [512] + LSTM(1024,1层) | ReLU | ~6,338,068 | **RNN** | 全普查最大 actor；**RNN 只出现在灵巧手，locomotion/tracking 无一例** |
| repo | booster_gym T1（真机人形） | 47 | 12 | **[256,128,128]** | ELU | **63,244** | 无 | 唯一主动缩小的真机人形；logstd=-2.0（std≈0.135） |
| repo | ASAP / HumanoidVerse G1-29 loco | 576（96 + 480 历史） | 29 | [512,256,128] | ELU | 463,389 | 历史 5 | init_noise_std=0.8 |
| repo | FALCON（HumanoidVerse 派生） | 575（两 actor 共享同一输入） | 15+14 | [512,256,128] ×2 个独立 actor | ELU | 922,013（合计） | 历史 5 | 上下身分离双 actor，宽度按身体复制而非共享 |
| repo | mujoco_playground G1 joystick | 103 | 29 | (512,256,128) | swish | ~224,954 | 无 | Brax，输出层 2×动作维（均值+logstd） |
| repo | PBHC（G1-23，HumanoidVerse fork） | 416（224 原始 + Conv 编码 128+64） | 23 | [768,512,256] + LayerNorm | **SiLU** | ~851,223 | Conv1D 压缩历史 10 步 + 未来 20 步 | 唯一用 LayerNorm + 编码器压缩历史的 locomotion 系 |
| repo | PHC（SMPL 全身模仿） | ~700-720（估） | 69 | [1024,512] | ReLU | ~1.3M / 每 primitive | 无；PMCP num_prim=3 | im_pnn_big 扩到 [2048,1536,1024,1024,512,512]，注释 `# comparable paramter to z_big_task`（原文有拼写错） |
| repo | ProtoMotions Mimic | 与机型相关 | 与机型相关 | [1024]×6 | ReLU | — | 无 | 全普查最深的纯 MLP actor；logstd 固定 -2.9 |
| repo | ProtoMotions MaskedMimic | 与机型相关 | 与机型相关 | Transformer(512) prior → 64-D VAE latent → trunk [1024]×3 | ReLU | — | 5 步历史 token + 5 步未来 token | 全普查唯一策略路径里有真 Transformer 的 |

### 这张表告诉我们三件事

1. **宽度不随 obs 维度走。** obs 从 42 到 705（17 倍），[512,256,128] 一路不变。两个例外都是**往下缩**（legged_gym flat [128,64,32]、booster_gym T1 [256,128,128]），没有一个是因为 obs 变宽而往上扩的。
2. **我们已经在同类任务的上半区。** 我们 283,935 参数 vs 直接母本 HITTER 221,725（+28%）、vs beyondmimic 250,397（+13%）。真正比我们大的（PBHC 851K、PHC 1.3M、ProtoMotions 6×1024）全都同时换掉了观测处理范式（Conv/Transformer/VAE 编码器），不是单纯把 MLP 拉宽。
3. **RNN 在这个生态里不属于 locomotion。** LSTM 只出现在 IsaacGymEnvs 的灵巧手任务；所有人形/四足 locomotion 与 tracking 仓库、以及四篇乒乓论文的 on-policy 三篇，用的都是前馈 MLP（有的加扁平历史堆叠）。

### 21.3 建议(排序)

**C1(P0):不扩容。冻结 hidden_dims=[512,256,128]，把这一轮资源全部投到 §16 死亡尖峰、§7 零梯度窗、strike_opportunity_count=0 这三个信号侧故障上。**
- 证据:§16：advantage 归一化后 dense 项有效梯度约为 -72 死亡方向的 1/500，且击中层收入恒为 0；§7：窗内 exp 核在大误差处梯度精确为零——三者都是「梯度不存在或被压到 1/500」，不是「表达能力不足」。外部对照：HITTER（arXiv 2508.21043v2 §V.B.3）用 104-D obs + [512,256,128] + 221,725 参数在同机型同任务上零样本成功；我们 225-D 下是 283,935 参数，比它大 28%。生态普查中 obs 从 42 到 705（17 倍）宽度不变，无一例因 obs 变宽而扩网。
- 落点:hope_training/whole_body_tracking/cfg/algo/ppo.yaml:39-40 保持不动；本轮改动全部落在 reward/termination 侧。
- 风险:若诊断后确证是欠拟合，会晚一轮才扩——但因为扩容必然要从零重训 2 万-2.5 万 iter（checkpoint 形状不匹配、strict load 报错），先诊断再扩本来就是省时的顺序。

**C2(P0):上一套可证伪的诊断仪表盘，把「容量不足 vs 信号不足」变成一个能被数据判定的问题，而不是一次架构赌博。五个指标：(a) explained_variance，且必须分拆成「含死亡终止 episode」与「不含」两条曲线；(b) 训练回报 vs 固定种子确定性评估回报的 gap；(c) clip 之前的梯度范数分布（对照 max_grad_norm=1.0）；(d) dormant neuron 比例（ReDo 定义，τ=0 与 0.025 两档）；(e) 倒数第二层激活的有效秩。**
- 证据:ppo.yaml:69 max_grad_norm=1.0、:54 desired_kl=0.01、:52 lr=0.001 全是单一全局量，rsl_rl/algorithms/ppo.py 用一个 Adam 覆盖 actor+critic 全部参数——所以「梯度预算是否已经被 clip 吃满」是可直接观测的。判定规则：EV 低但只在含死亡 episode 上低 → 是 §16 尖峰，不是 critic 容量；train/eval gap 大 → 已过拟合，扩容有害；dormant 比例 <5% 且有效秩接近层宽 → 现有容量尚未用满，扩容无收益。
- 落点:my_on_policy_runner.py 的日志端新增这五个标量；EV 分拆需要在 rollout 里带上 termination 标志。
- 风险:仪表本身要几天工时，且有效秩/dormant 的计算会给每次 update 加一点开销（建议每 N iter 采样一次而非每步算）。

**C3(P0):跑一次监督拟合能力探针——这是最便宜、最直接证伪「容量不足」的实验。冻结环境，dump 一个 98,304 样本的 rollout buffer，用完全相同的 [512,256,128] 架构去监督回归 teacher/reference 动作（或已算好的 advantage 目标），看训练损失能否压到接近零。**
- 证据:容量瓶颈的定义就是「目标存在但拟合不上」。若 [512,256,128] 能在 98,304 样本上把监督损失打到近零，容量假说被直接证伪，剩下的一切归因于 RL 信号侧。当前参数/样本比已经是 611,616/98,304 ≈ 6.2 倍（每 update 只有 5 epoch×4 minibatch=20 个梯度步），[1024,512,256] 下变成 19.1 倍——容量不是稀缺资源，样本的信息含量才是。
- 落点:单卡离线脚本即可，几分钟到几十分钟；不需要动训练主干。
- 风险:监督拟合能力是 RL 拟合能力的上界而非等价物——探针通过不能 100% 排除「on-policy 分布漂移下的有效容量不足」。但探针不通过就能 100% 确认容量确实是瓶颈，所以它作为证伪工具是有效的。

**C4(P1):无论扩不扩，先把 G07 的推理延迟实测数补上（--probe 工具已存在，输出从未入库）。**
- 证据:docs/operations/run_deploy_dryrun.md 的「Required Documentation Before Hardware」明确要求 G07 列出 latency result，但 docs/gates/G07_mujoco_to_real.md 全文没有任何延迟数字。部署侧硬约束：period_ns=20,000,000（a3_based_task.hpp:30）、policy_hz: 50.0（a3_runtime_config.yaml:201-203）、ORT 单线程 SetIntraOpNumThreads(1)（ort_session.hpp:50-51），a3_based_task.cpp:142 会统计 overrun。估算上 actor 前向 [512,256,128] 约 28.4 万次乘加、[1024,512,256] 约 89.4 万次，对 20,000 µs 的预算都可以忽略（差 4-5 个数量级，瓶颈是 ORT 会话/张量拷贝开销，与隐层宽度基本无关）——但「估算可忽略」不等于「实测过」。
- 落点:跑一次 --probe，把数字写进 docs/gates/G07_mujoco_to_real.md 的 Outputs。
- 风险:几乎无风险，纯补数据；顺带解掉 G07 的一个未填字段。

**C5(P1):如果（且仅如果）诊断明确指向欠拟合，第一步只扩 critic，不动 actor：critic 从 [512,256,128] 改到 [512,512,256]（327,681 → 557,569，合计 841,504 ≈ 1.38×），actor 保持不变。但必须同时监控共享的梯度范数与 KL 学习率。**
- 证据:Andrychowicz et al.（arXiv 2006.05990，ICLR 2021，>25 万 agent 的超参扫描，宽度网格 {16,...,512} × 深度 {1,2,4,8}）原文：value 网络加宽「没有下行风险」，而 policy 宽度最优值依环境而定、过宽会显著掉性能。部署侧 critic 完全免费：exporter.py:86-120 的 export() 只 trace self.actor(self.normalizer(x))，critic 从不进 ONNX。观测合同也不动，225-D deploy parity 不受影响。
- 落点:cfg/algo/ppo.yaml:40 critic_hidden_dims 单行改动 + 新建一条对照臂。
- 风险:关键风险（我们仓库特有，与外部文献不同）：rsl_rl/algorithms/ppo.py:96,385 是单个 Adam 覆盖 self.policy.parameters()（actor+critic 合并）、单个全局 clip_grad_norm_(max_grad_norm=1.0)、单个 adaptive-KL 学习率。加大 critic 会改变被 clip 的总梯度向量和 KL 驱动的有效学习率，actor 的优化动力学不是隔离的——所以「critic 免费」在别的框架成立，在我们这里只在部署侧成立，在优化侧不成立。另外这一改动仍然作废全部 checkpoint（strict load 形状不匹配），是从零重训。

**C6(P2):如果 actor 最终也要扩，只加宽中层到 [512,512,256]（283,935 → 517,663），不要加深，不要直接跳 [1024,512,256]。**
- 证据:Andrychowicz（arXiv 2006.05990）：2 个隐层在所有测试环境都够用，深度 4/8 无收益——我们已经是 3 层，已在甜点之上，加深是纯损失。宽度非单调（HalfCheetah 最优 policy 宽度仅 16-32 单元），所以要小步走。参数/样本比：[512,512,256] 下 actor+critic 合计 1,075,232 / 98,304 ≈ 10.9 倍；[1024,512,256] 下 1,878,560 / 98,304 ≈ 19.1 倍，且每 update 只有 20 个梯度步、同一批 on-policy 样本被复用 5 个 epoch——批内过拟合风险随容量线性上升，而 PPO 侧只有 clip_param=0.2 与 adaptive KL 这两道弱护栏。
- 落点:cfg/algo/ppo.yaml:39；必须与 P1 的 critic 臂分开跑，否则无法归因。
- 风险:零初始化末层的安全性会变：weight=0 时初始动作严格等于 bias（与宽度无关，这一点安全），但第一次反向传播后末层的有效步长正比于倒数第二层激活范数 ‖h‖，而 ‖h‖ 大致随宽度 √ 增长——把倒数第二层从 128 加宽到 256 会让策略脱离零初始点的速度变快约 √2 倍。在 init_noise_std=0.02 这种极紧探索下，早期动力学对这个变化敏感，需要在扩容臂上专门看头 200 iter 的动作幅度曲线。

**C7(P3):不要把 SimBa/BRO 式的 residual + LayerNorm 作为默认架构引入；如果要试，必须以独立消融臂的形式，且明确标注为「文献空白区的探索」而非「已验证配方」。**
- 证据:SimBa（ICLR 2025）、BRO（NeurIPS 2024 spotlight，注意不是 ICLR 2025）、ReDo（ICML 2023）的核心证据全部是 off-policy（SAC / TD-MPC2 / DQN / DrQ，带 replay buffer 与高 UTD）。SimBa 唯一的 on-policy PPO 实验在 Craftax（离散生存游戏），BRO 全文零 PPO 实验，ReDo 只测 DQN 与 SAC/DrQ。唯一专门研究 on-policy 可塑性的论文（Juliani & Ash 2024, arXiv 2405.19153）只跑 gridworld / Montezuma / ProcGen，不是连续控制；它的结论也需要精确复述：末层 reset 在三种条件下全部失效，而 LayerNorm 能解决训练侧可塑性丢失（§4.5 标题原意）、并在 CoinRun 上属于三个最有效方法之一，但在泛化/测试性能上不稳定。没有任何一篇论文在 100-300 维观测的腿式/人形连续控制任务上、用 PPO 测过 SimBa/BRO 式缩放架构——这是真实的文献空白。
- 落点:若做，作为单独 arm，与基线同种子同步跑，不进默认 cfg。
- 风险:在空白区上押注默认配置，等于用一次 2 万 iter 的从零重训去赌一个跨范式外推；即使有效也难归因（架构变了、初始化变了、warm start 没了）。

**C8(P3):不要上 RNN。若确认需要时序信息，history stacking 排在 RNN 之前，但必须走完整的观测合同流程（新命名合同 + 新 critic ABI + 部署侧历史缓冲）。**
- 证据:生态普查里 LSTM 只出现在 IsaacGymEnvs 的灵巧手任务（ShadowHandOpenAI_LSTM ~634 万参数、AllegroHandDextremeADR、AnymalTerrain_LSTM 变体），所有人形/四足 locomotion 与 tracking 仓库、以及三篇 on-policy 乒乓论文，无一使用 RNN。代码侧 rsl_rl 的 ActorCritic 硬编码 is_recurrent=False 且 __init__ 根本没有 rnn_type/rnn_hidden_size 参数，ppo_cfg.py 的 RslRlPpoActorCriticCfg 也没有 RNN 配置面——引入 RNN 是框架级改造，且 ONNX 侧要在 50 Hz 循环里跨 tick 携带隐状态，直接威胁部署 parity。history 侧的先例充分：PACE H=5（82×5=410 维）、ASAP 5（576）、FALCON 5（575）、unitree_rl_lab 5（480）、XBot-L 15（705）。但反证同样有力：HITTER（同机型同任务、我们的直接母本）不用任何历史堆叠。
- 落点:若做，作为新的命名合同（如 A225H5）进 actor_observation_contract.py，不得就地改 A225。
- 风险:history 会打断 194/225-D 部署 parity 这条神圣约束；且我们的 A1 延迟旋钮据前期结论全为零（本轮未重新核实，按「继承结论」处理），在零延迟下单帧观测的马尔可夫性本来就够——历史带来的收益可能很小而合同代价很大。

**C9(P2):修正尽调 doc 里的两处事实：(a) actor 末层零初始化的位置是 scripts/train.py 的 _apply_action_ball_fresh_policy_bootstrap（定义 7010 行、weight.zero_() 在 7062 行、断言在 7065 行、调用点 16237 行），不是 5004 行；(b) A225 的 318-D critic 是 A225 自有 ABI，与 L194 那条线的 318 维只是标量宽度巧合，不是共享合同。**
- 证据:grep 实测：train.py:7010 / 7062 / 7065 / 16237。A225 侧代码自己的 docstring（action_ball_225_trainability.py:41-43）明确警告：这是 A225-owned ABI，即使标量宽度等于历史 Stage-1 critic，其 normalizer/checkpoint 血统是新的。两条线的 critic 词条结构其实不同（A225 用 task_desired_contact_* 系列，另一条用 racket_pos_b / racket_lin_vel_w / racket_normal_w / episode_time_left）。
- 落点:docs 里对应段落的行号与措辞。
- 风险:不修的风险是后续有人按「critic ABI 跨 actor 宽度稳定」去做 warm start 或 checkpoint 复用，会踩到静默的语义不匹配（宽度对得上、含义对不上，strict load 不会报错）。


### 21.4 补充

## §21 网络容量裁决：225-D 输入要不要扩容？

**裁决：不扩。** 现在扩网络是给没坏的零件花钱。我给"网络容量是当前主要瓶颈"的概率 **≤10%**。

### 21.1 两种假说必须先分开

| | 容量不足 | 信号不足 / 优化被压制 |
|---|---|---|
| 长什么样 | 有明确的学习目标、有非零梯度，但网络拟合不上 | 目标从未出现，或梯度被压到近零 |
| 怎么救 | 加宽/加深/换架构 | 修 reward、修终止、修核函数 |
| 加参数有用吗 | 有 | **没有**——零乘以任何参数量还是零 |

我们的三条硬证据全部落在右列：

1. **击中层收入恒为 0**（§16，`strike_opportunity_count=0`）。网络再大也拟合不出一个从未出现过的目标。
2. **回报被 -72 死亡尖峰支配**（§16）。advantage 归一化后，dense 项的有效梯度约为死亡方向的 **1/500**。这是方向被压缩 500 倍，不是表达力不够；扩容只会把这个被压缩的方向复制到更多参数上，压缩比例一点不变。
3. **窗内 exp 核在大误差处梯度精确为零**（§7）。

### 21.2 外部对照：[512,256,128] @ 225-D 已经绰绰有余

- **直接母本 HITTER**（arXiv 2508.21043v2 §V.B.3）：Unitree G1、29 自由度、打乒乓、actor obs 仅 **104 维**、网络就是 **[512,256,128]**、**221,725 参数**、**无历史堆叠**，零样本上真机。我们 A225 是 225 维、**283,935 参数**，比母本大 **28%**。
- **PACE**（arXiv 2509.21690v3）：Booster T1 打乒乓，obs 82×H=5=**410 维**，网络 [512,512,128]（541,461 参数）。比我们宽的输入、只比我们大 1.9 倍的网络。
- **生态普查**：obs 从 42 维到 705 维（**约 17 倍跨度**），`[512,256,128]` 是近乎唯一的默认值，**没有任何一个 locomotion/tracking 仓库把宽度随 obs 维度往上调**（beyondmimic 160-D、mjlab 160-D、unitree_rl_lab 480-D、legged_gym 235-D、ASAP 576-D、XBot-L 705-D，全是这一个网络）。仅有的两个主动偏离都是**往下缩**：legged_gym 的 anymal_c flat 变体缩到 [128,64,32]（17,004 参数），booster_gym T1 缩到 [256,128,128]（63,244 参数）。

### 21.3 直接回答"225-D 是不是本身就要求更大网络"

**不要求，而且这个提法本身就是错的。**

- 194 → 225 只多 **31 列**，第一层只多 `31×512 = 15,872` 参数 = actor 总量的 **5.6%**。
- 而那个 512 宽的第一隐层本身就占 actor 参数的 **40.7%**（115,712 / 283,935）；actor+critic 合起来第一层占 **45.6%**（279,040 / 611,616）。
- 极端一点：把输入从 225 换成 318（critic 那边的宽度）也只多 47,616 参数，占合并总量的 7.8%。

**第一层参数占比说明的就是这件事：参数量几乎不看输入宽度，只看隐层宽度。** 用输入维度去推导"需要更大网络"是把两个不相关的量绑在一起了。

更进一步，A225 的 225 列里有 **154 列（68.4%）** 是 achieved-vs-teacher 成对镜像（base 15+15、joint_pos 31+31、joint_vel 31+31）。这是刻意的残差/模仿结构——要学的函数更接近「恒等 + 修正」，**比从零编码一个任务更容易表示，不是更难**。

### 21.4 扩容文献不能直接搬（最容易踩的坑）

| 论文 | 结论 | 算法基座 | 对我们适用吗 |
|---|---|---|---|
| SimBa（ICLR 2025） | RSNorm + 残差 + LayerNorm 让参数量可安全放大 | **SAC / TD-MPC2**（off-policy）。唯一的 PPO 实验在 Craftax（离散生存游戏） | ✗ 范式不同 |
| BRO（**NeurIPS 2024 spotlight**，不是 ICLR 2025） | 强正则化解锁 critic 缩放到 2630 万参数 | **SAC**，全文零 PPO 实验 | ✗ |
| Dormant Neuron / ReDo（ICML 2023） | 休眠神经元吃掉有效容量 | **DQN / SAC / DrQ**，全 replay-based | ✗（诊断指标本身可借） |
| **Andrychowicz et al.（arXiv 2006.05990，ICLR 2021）** | 5 个 MuJoCo 连续控制环境、宽度 {16…512} × 深度 {1,2,4,8} 大规模扫描 | **PPO / on-policy** | ✓ **唯一对口** |

唯一对口的那篇说的是：**actor 宽度最优值依环境而定且非单调，太窄和太宽都掉性能**（HalfCheetah 最优 policy 宽度只有 16-32 单元）；**critic 加宽没有下行风险**；**2 个隐层在所有环境都够用，4/8 层无收益**（我们已经是 3 层，已在甜点之上）。

另外要精确复述 on-policy 可塑性那篇（Juliani & Ash 2024, arXiv 2405.19153，PPO 但跑在 gridworld / Montezuma / ProcGen，非连续控制）：**末层 reset 在三种条件下全部失效**；**LayerNorm 能解决训练侧可塑性丢失、在 CoinRun 上属最有效的三个方法之一，但在泛化/测试性能上不稳定**（不要把 LayerNorm 和末层 reset 一起打成"无效"）。

**文献空白（重要）**：没有任何一篇论文在 100-300 维观测的腿式/人形连续控制任务上、用 PPO 测过 SimBa/BRO 式缩放架构。这正是我们所处的位置。

### 21.5 可证伪的诊断指标（先诊断，再决定扩不扩）

| 指标 | 怎么读 | 指向 |
|---|---|---|
| `explained_variance`，**分拆成含死亡终止 / 不含两条曲线** | 只在含死亡的那条上低 | §16 尖峰问题，**不是** critic 容量 |
| 训练回报 vs 固定种子确定性评估回报的 gap | gap 大 | 已过拟合，**扩容有害** |
| clip 之前的梯度范数分布（对照 `max_grad_norm=1.0`） | 长期远大于 1 | 预算已被 clip 吃满，加参数只是分蛋糕 |
| dormant neuron 比例（ReDo 定义，τ=0 与 0.025） | < 5% | 现有容量尚未用满，**扩容无收益** |
| 倒数第二层激活的有效秩 | 接近层宽 = 饱和；很低 = 表征坍缩 | 区分容量 vs 可塑性 |
| **监督拟合探针（最便宜、最直接）** | dump 98,304 样本 buffer，用同一架构监督回归 teacher 动作 | 若损失能压到近零 → **容量假说被直接证伪** |

判定规则：只有当"§16/§7 修完 → `strike_opportunity_count > 0` → EV 高、train/eval gap ≈ 0、dormant 低、监督探针也压不下去"这一整条链都成立时，才认定容量是瓶颈。

参数/样本比作为背景：`4096 env × 24 步 = 98,304 样本/update`，每 update 只有 `5 epoch × 4 minibatch = 20 个梯度步`。当前合并参数 611,616 已经是单次 update 样本量的 **6.2 倍**；改到 [1024,512,256]（1,878,560）会变成 **19.1 倍**。整轮 2 万-2.5 万 iter 累计约 20-25 亿样本，总量不缺；但**同一批 on-policy 样本被复用 5 个 epoch 的批内过拟合风险随容量线性上升**，而 PPO 侧只有 `clip_param=0.2` 与 adaptive KL 这两道弱护栏。

### 21.6 如果最终要扩，形状是什么

**第一步（且仅在诊断指向欠拟合后）：只扩 critic。**
`critic_hidden_dims: [512,256,128] → [512,512,256]`（327,681 → 557,569，合计 841,504 ≈ **1.38×**）。

- 部署侧完全免费：`exporter.py:86-120` 的 `export()` 只 trace `self.actor(self.normalizer(x))`，**critic 从不进 ONNX**。
- 观测合同不动，225-D deploy parity 不受影响。
- Andrychowicz 明确说 value 网络加宽无下行风险。

⚠️ **但"critic 免费"在我们这里只在部署侧成立，优化侧不成立**：`rsl_rl/algorithms/ppo.py:96,385` 是**单个 Adam** 覆盖 `self.policy.parameters()`（actor+critic 合并）、**单个全局** `clip_grad_norm_(max_grad_norm=1.0)`、**单个 adaptive-KL 学习率**（`desired_kl=0.01`, `lr=0.001`）。加大 critic 会重塑 actor 也在其中的梯度范数预算与有效学习率。两个网络在这套配置下**不是**可独立调的。

**第二步（若 actor 也确需扩）：只加宽中层到 [512,512,256]**（283,935 → 517,663）。**不加深**（Andrychowicz：深度 2 够用，4/8 无收益，我们已是 3 层）。**不要直接跳 [1024,512,256]**（3.07×，参数/样本比 19.1 倍）。

**不做的事：**
- ✗ **不上 residual + LayerNorm 作为默认**。全部证据是 off-policy 外推；若要试，只能作为独立消融臂并标注为"文献空白区探索"。
- ✗ **不上 RNN**。全普查中 LSTM 只出现在 IsaacGymEnvs 的灵巧手任务，所有人形/四足 locomotion、tracking 仓库与三篇 on-policy 乒乓论文无一使用。代码侧 rsl_rl 的 `ActorCritic` 硬编码 `is_recurrent=False`、`__init__` 根本没有 RNN 参数，`ppo_cfg.py` 也无 RNN 配置面——引入是框架级改造，且 ONNX 要在 50 Hz 循环里跨 tick 携带隐状态，直接威胁部署 parity。
- △ **history stacking 排在 RNN 之前但不是现在**。先例充分（PACE H=5、ASAP 5、FALCON 5、unitree_rl_lab 5、XBot-L 15），但反证同样有力：HITTER 同机型同任务不用任何历史。且它会打断 194/225-D 部署 parity 这条神圣约束，必须走新命名合同（如 A225H5），不得就地改 A225。A1 延迟旋钮据前期结论全为零（本轮未重新核实，按继承结论处理），零延迟下单帧观测的马尔可夫性本来就够。

### 21.7 与既定工程约束的相容性

| 约束 | 扩容后是否安全 | 依据 |
|---|---|---|
| **ONNX / 50 Hz 延迟** | ✓ 不是瓶颈，但**从未实测过** | 周期硬定 20 ms（`a3_based_task.hpp:30` period_ns=20,000,000；`a3_runtime_config.yaml:201-203` policy_hz: 50.0），ORT 强制单线程（`ort_session.hpp:50-51`）。估算 actor 前向 [512,256,128] ≈ 28.4 万次乘加、[1024,512,256] ≈ 89.4 万次，对 20,000 µs 预算差 4-5 个数量级；真正的开销是 ORT 会话/张量拷贝，与隐层宽度基本无关。**但仓库里没有任何实测延迟数字**——`docs/operations/run_deploy_dryrun.md` 要求 G07 列出 latency result，`docs/gates/G07_mujoco_to_real.md` 该字段至今空白，`--probe` 工具存在但输出从未入库。**无论扩不扩都该先补这个数。** |
| **194 / 225-D 观测合同** | ✓ 只改 hidden_dims 不动合同 | 观测合同与网络宽度正交 |
| **`init_noise_std=0.02` + 末层 weight=0** | △ 基本安全，但早期动力学会变 | weight=0 时初始动作严格等于 bias（与宽度无关，安全）。但首次反向后末层的有效步长正比于倒数第二层激活范数 ‖h‖，而 ‖h‖ 大致随宽度 √ 增长——倒数第二层 128→256 会让脱离零初始点的速度快约 √2 倍。在 0.02 这种极紧探索下需专门观察头 200 iter 的动作幅度曲线。 |
| **warm start / checkpoint** | ✗ **任何 hidden_dims 改动都作废全部 checkpoint** | 形状不匹配触发 strict load 报错；等于放弃 warm start，从零重跑 2 万-2.5 万 iter。这是 GPU-周级别代价，只应在诊断确证欠拟合之后付。 |
| **§16 尖峰是否让扩容变成浪费** | ✗ **会** | 在死亡方向支配、dense 梯度只有 1/500、击中层收入恒为 0 的前提下扩容，等于**用一次从零重训去放大一个被压缩 500 倍的信号**。修 §16/§7 之前扩容，是把钱烧在错误的自由度上。 |

### 21.8 需要修正的既有记录

1. actor 末层零初始化的位置是 `scripts/train.py` 的 `_apply_action_ball_fresh_policy_bootstrap`：**定义 7010 行、`output.weight.zero_()` 在 7062 行、`count_nonzero` 断言在 7065 行、调用点 16237 行**——不是 doc 里写的 5004 行。（`init_noise_std=0.02` 确认为 CLI 强制覆盖，`ppo.yaml` 自身默认仍是 1.0，并在 receipt 里硬校验 `required_realized_init_noise_std`。）
2. **A225 的 318-D critic 是 A225 自有 ABI**，与另一条线的 318 维只是标量宽度巧合，不是共享合同。A225 代码自己的 docstring（`action_ball_225_trainability.py:41-43`）明确警告过这点；两条线的 critic 词条结构不同（A225 用 `task_desired_contact_*` 系列，另一条用 `racket_pos_b` / `racket_lin_vel_w` / `racket_normal_w` / `episode_time_left`）。**不修的风险是：有人按"critic ABI 跨 actor 宽度稳定"去复用 checkpoint，宽度对得上、含义对不上，strict load 不会报错，静默出错。**
3. A225 中 teacher 镜像列的占比是 **68.4%（154/225）**，此前 doc 中出现过的 "~55%" 与 "大约一半" 两种说法都不对。


---

## 二十二、随机性/噪声的"开局加满 vs 逐步加"裁决(08-02)

**方法**:2 抽取(我方全轴盘点含可调度性 / 外部 9 库 + ADR 与论文的分档证据)+ 2 对抗核查(6 修正)
+ 1 裁决。**两处对本 doc 前文的重要修正已并入**:①PD 增益已于 07-31 commit `567dbbe25`
改为 **startup 级 + 厂商非对称 Kp(0.8,1.2)/Kd(0.7,1.3)**(§1/§11 记的"reset 级 ±15%"作废;
`HOPEPingPongHitter.yaml:519-520` 那句注释是陈旧残留);②执行器延迟**不再是死代码**——
`hope_actions.py` 新建了 `_EpisodeSharedPolicyActionDelay`(整条 31-D q_des 按控制步延迟、
每 env 每集抽一次),厂商叶子设 [0,2] 步、**现役 N1 叶子显式清零**,开启**不需要写代码**。

### 22.1 裁决:三闸判据

## 一句话裁决

**"开局就加满"的资格是三闸全过:零均值且只扰动机器人自身参数(不改任务几何)× 抽签发生在 startup/reset 级(不注入 episode 内动力学方差)× 实测终止率增量 ≤ 0.5 个百分点。三条缺一,就必须分档进;若同时踩中"改任务几何"和"抬终止率"两条,则在 -72 死亡尖峰与 σ=0.075 m 死核修好之前一律不开。**

---

### 闸 1(支撑集闸):这条随机性改不改"哪些动作能击中球"?

- **不改** → 只在标称值附近扰动机器人自身的物理参数(体摩擦、连杆质量、CoM、关节零点、Kp/Kd)。判定口诀:**同一条参考轨迹在扰动后是否仍近似可行**?是 → 过闸,day-1 开满。
- **改** → 动的是球的来路/落点、期望接触 p-v-face 元组、起点分布、目标可观测性。这些直接重塑"能击中"的集合。出处:32-arm 的 16 条轴全部是来球/落点/出生位几何(action_ball_sampling.py ARM_KEYS);A1 八旋钮直接给 racket target 加噪;canonical_ready 的零噪声 reset 是把起点塌成每 action 一个 delta 函数(§9)。
- 这条闸解释了一个容易搞混的不对称:**本体感受观测噪声(joint_pos/joint_vel)不改转移、不改支撑集**,可以 day-1 开满;而**给 desired-contact / racket-heading / time_to_contact 这些任务通道加噪 = 直接降低任务目标的可观测性 = 改支撑集**,不能 day-1。二者都叫"观测噪声",分档完全相反。

### 闸 2(终止闸,-72 放大器):这条随机性把终止率抬高多少?

- **硬阈值:一格 200-iter smoke,终止占比的绝对增量 Δp_term ≤ 0.5 pp 才准 day-1。**
- 依据(全部是我们自己的数字,§16.1):死亡一次 = -3600 × dt 0.02 = **-72.0**(生效点 scripts/train.py:8896 的 v2 DIRECT 表 + yaml 键 `death_penalty_weight`,train.py:11171-11184)。γ=0.99 的折扣视野 100 步 > 平均 episode 124.11 步(=2.48 s),所以**以死亡结尾的 episode 里几乎每个状态的回报都是 -72×0.99^k ∈ [-72,-21]**,而全段 dense 收入只有 ~7.4。全 batch advantage 归一化统一的是整体尺度、不改项间比例,于是 dense 项的归一化 advantage ≈ **0.007**、死亡方向 ≈ **3**,**相差约 500 倍**;PPO 的 clip=0.2 与 desired_kl=0.01 这两个"策略移动预算"几乎全花在死亡方向上。
- **在 strike_opportunity_count=0 的当下,这条闸比平时严厉得多**:§16 已核实盈亏平衡命中率 = 72/(72+32.98) = **69%**,一个 from-scratch 策略永远够不到,**"不挥拍"在结构上就是当前 reward 下的最优解**。此时任何 Δp_term > 0,都不是"练抗扰",而是**给"别动"这个已经最优的退化解再补一笔贴**。终止占比的基线本身就在 3.1%/15.7%/32.0% 之间跳一个数量级(configs/n1_contact_20260729/n1_live_wave_4ff48b21.v1.json),再往上抬只会让 grad-norm 裁剪因子的批间抖动更大(§16.1 C4)。
- 推论:**推撞、reset 状态噪声、摩擦低端外扩、terrain 凹凸、执行器延迟,全部在闸 2 前面排队**,直到死亡尖峰降到三库量级(三库最大单步罚项 post-dt ≈ -0.2,只占各自视野收入 9.5 的 2%;我们现在是 1200%,差约 600 倍)。

### 闸 3(cadence 闸):抽签频率决定它注入的是"环境异质"还是"episode 内方差"

- **startup 级**(env 构造抽一次,永不重抽)→ 只造跨 env 的异质性,不给单条轨迹注入额外方差 → **最安全,默认 day-1**。我们现役的摩擦/质量/CoM/关节零点/PD 五项全在这一级。
- **reset 级**(每集抽一次、集内固定)→ 次之,注入的是 episode 间方差,仍可接受。
- **每步级** → 只允许零均值**观测**噪声(不改转移);**不允许每步级动力学扰动**。

### 配套判据:"分几档"怎么定(不是拍脑袋)

**档位数 = ceil(终态幅度 ÷ 该轴的一个任务容差单位)**,容差单位取自我们自己的窗口常数:

- **延迟**的容差单位 = `strike_window_pos_s` = 0.02 s = **1 个控制步 @50 Hz**(HOPEPingPongActionBallA3VendorV2.yaml:15)。终态厂商值 2 步 = 40 ms = **位置窗的 2 倍** → 分 **2 档**(0→1→2),绝不允许一次上满 [0,2]。
- **推撞**的容差单位 = "期望撞进击球窗的次数 ≤ 0.1 次/episode"。击球窗 0.10 s、平均 episode 2.48 s:节奏 [1,3] s(均值 2 s)→ 每 episode 期望 1.24 次推、撞窗概率 ≈ 5%;节奏 [10,30] s(均值 20 s)→ 期望 0.124 次、撞窗概率 ≈ 0.5%。**外部的 [1,3] s 节奏不能抄**。
- **reset 噪声**:终态取厂商 A3 值(位姿 ±0.1、速度六维 ±0.2、关节 ±0.15 rad),按 1/3、2/3、满分 **3 档**。

### 元裁决:现在不该写 in-loop DR scheduler(这条同样可操作)

三条论证,全部有证据:

1. **ADR 家族的扩张条件是 success rate 过阈,我们 success = 0 → 扩张永远不触发,写了也是 no-op。** OpenAI 自己写明 ADR 的初始分布"concentrated on a single environment",并且"additional environments are only added when a minimum level of performance is achieved"(arXiv 1910.07113)。我们连一个环境都还没解开(exact 命中 0.23%、window 5.01%、capture/return = 0/0)。**ADR 的前提不满足,顺序是反的。**
2. **外部 9 个库里 8 个是 day-1 固定 DR**;唯一自带 in-loop DR 幅度课程的是 IsaacLab dexsuite(manipulation,performance-gated,difficulty 0..10);PBHC 的 obs 噪声课程和出生偏移课程**默认都是 False**。所有 locomotion/tracking 库 ramp 的是**任务难度**(terrain level / command 速度区间),**不是 DR 幅度**。
3. **机械上也不该在 loop 里改**:我们全部 5 条现役物理 DR 都是 startup 级(env 构造时抽),in-loop 改它需要新写 CurriculumTermCfg hook,而全仓 `CurriculumCfg` 至今还是 `pass`(tracking_env_cfg.py:286-290),零脚手架可复用。

→ **我们的"分档"落在发射/续训边界(换一次 argv = 升一档),不是 in-loop。** 里程碑判据本来就要人判(§13 的 NO_OPPORTUNITY_CONTINUE 契约、frozen-policy canary→heldout 双窗口也是离线 sidecar 判的),不存在"必须自动"的需求。

### 一句更难听的总结

现在讨论"DR 加多少"这件事本身,优先级低于 §16/§7 两个 critical。**在 -72 尖峰和 σ 死核修好之前,唯一正确的 DR 动作是"一格都不动"**——保持现役 day-1 集合原封不动,好让 reward 侧的改动是单变量可比的。任何在此之前加的 DR,都会被记进"为什么还是学不会"的混淆项里。

### 22.2 全轴分档表

## 全轴分档表(§1-21 盘点出的每一条)

档位含义:**D1** = day-1 开满/保持现状;**S** = 分阶段进;**X** = 暂不开(含"需新代码,不排产")。
触发信号定义见 schedule 一节:**M0** = 死亡尖峰与死核 σ 修复验收;**M1** = strike_opportunity_count > 0 且 window 命中率连续 3 个 ≥100-update 评测窗 ≥ 20%;**M2** = exact 命中率 ≥ 修复后的盈亏平衡率(死亡改 -7.2 时 = 7.2/(7.2+32.98) = **17.9%**,取整 20%)。

### A. 物理/引擎域随机化(全部 startup 级)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| 机器人体摩擦 physics_material | **ON**,static (0.3,1.6)/dynamic (0.3,1.2)/restitution (0,0.5),64 桶,startup | **D1**(保持)+ 低端外扩另算 **S** | 维持 (0.3,1.6)/(0.3,1.2) | 厂商 (0.2,1.8)/(0.2,1.5) | 外扩 1 档 | **M2** 之后单格消融 | 三闸全过:零均值乘性、只改握地容差、startup 级。但**低端从 0.3 降到 0.2 会增打滑→抬终止率**,踩闸 2 → 外扩必须等 M2。出处:tracking_env_cfg.py:163-173;CLI `task.plant.robot_material_static_friction_range` / `..._dynamic_friction_range`(train.py:11796-11891,含 `robot_material_make_consistent` 保证 dynamic≤static);厂商值见 §11.2 |
| 地面 terrain 摩擦 | 固定点值 1.0/1.0(**不是分布**) | **X** | — | — | — | — | 今天它是个点不是轴。CLI(`task.plant.ground_static_friction/ground_dynamic_friction`)只能移动这个点,造分布要新代码。机器人体那一侧已经在随机了,重复投入无收益。出处:tracking_env_cfg.py:49-54;train.py:11835-11871 |
| 连杆质量 link_mass | **ON**,scale (0.85,1.15) = ±15%,全身,recompute_inertia=True,startup | **D1**(保持) | ±15% | ±15%(不动) | — | — | 三闸全过。厂商是"选择性末端质量 ±20%",哲学不同但我们全身 ±15% 已覆盖。出处:randomization_base.yaml:5;hope_env_cfg.py:1112-1123;CLI `task.domain_rand.link_mass_range` |
| **腕 + 拍**选择性质量 ±20% | 不存在(link_mass 是全身单一 range) | **S**(低优先) | 关 | 腕组 + 拍 scale ±20% | 1 档 | **M2** | 拍子是真实的质量未知源(§11.3 第 6 条)。但**需要新键**(按 body 分组的 mass range),属"要写代码"的一类,§21 裁定资源投信号侧 → 排在最后 |
| 躯干 CoM 抖动 base_com | **ON**,torso_link x±0.025 / y±0.05 / z±0.05 m,startup | **D1**(保持) | 现值 | 现值 | — | — | 三闸全过。**注意:幅度没有 CLI 出口**,只有 `stable_ready_plant` 的捆绑式 on/off;要调幅度得写新键。既然判定是"保持",这个缺口不构成阻塞。出处:tracking_env_cfg.py:185-192 |
| 关节零点标定偏移 add_joint_default_pos | **ON**,±0.01 rad,全关节,**双写 sim default 与 action offset** | **D1**(保持,且不可关) | ±0.01 rad | ±0.01 rad | — | — | 这不只是 DR,是 train==deploy parity 的机制本体。厂商同值 ✅。完全无 CLI 出口(train.py 只读不写)。出处:tracking_env_cfg.py:175-183 |
| PD 增益 Kp/Kd | **ON**,startup 级 log_uniform,**Kp (0.8,1.2) / Kd (0.7,1.3)**(07-31 commit 567dbbe25 已对齐厂商) | **D1**(保持) | 现值 | 现值 | — | — | 三闸全过,且已经是厂商同底盘实跑值。**修正入档**:该 commit 之前是**对称 ±20%**(不是文档里写的 ±15%);HOPEPingPongHitter.yaml:519-520 那句 "+/-15% reset-level" 是仓库内的陈旧注释,顺手清掉。出处:randomization_base.yaml:9-11;train.py:10583-10721(硬性要求 mode=='startup') |
| armature / 关节机械摩擦 / 力矩上限随机化 | **不存在**(mdp 包里只有 randomize_joint_default_pos 与 randomize_rigid_body_com 两个本地函数) | **X** | — | — | — | — | 需要写新 EventTerm + 接线。§21 已裁定不扩工程、资源投信号侧。记台账,不排产 |

### B. 时序 / 扰动类(全部踩闸 2,必须排队)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| **执行器命令延迟** | **OFF**,N1 叶子显式写死 min=max=0(byte-identical no-op);父本 A3VendorV1 是厂商 [0,2] 控制步 | **S** | 0 步 | 2 控制步(= 40 ms @50 Hz) | **2 档**(0→1→2) | 1 步:**M1**;2 步:**M2** | 决定性数字:`strike_window_pos_s = 0.02 s` = **正好 1 个控制步**。一次上满 2 步 = 40 ms = **把位置窗整个吃掉**。外部双重先例都说"从零长进来":ADR 的 `action_latency` init_range 是 **[0,0]** 零宽起步;ManualDR 用 `actionLatencyScheduledSteps = 10,000,000`(**修正:不是 2,000,000;2M 那个值在 ADR yaml 里,而 ADR 子类完全 override 了那段代码、根本不读它**)做纯时间线性斜坡。落点已接线,无需改码:`task.actions.control_step_action_delay_min/max`(hope_actions.py:684-830 每 env 每集抽一次;发射器 launch_action_ball_a225_four_arm_diagnostic.py:1391-1392) |
| **外部推撞**(Wave-P 速度 / F 轴力 / 合并互斥) | **OFF**,N1 叶子 `push.enable=false` 且每个字段显式 null(train.py 拒绝 enable=false 携带休眠幅度,所以是干净全关);发射器另传 `task.push.enable=false` | **S** | 厂商半幅:vel_xy ±0.125 m/s、vz ±0.05、roll/pitch ±0.13、yaw ±0.195 rad/s,节奏 **[10,30] s** | 厂商全幅 ±0.25/±0.1/±0.26/±0.39,节奏 **[5,15] s**(**不是外部的 [1,3] s**) | **2 档** | 半幅:**M1** 且 Δp_term ≤ 0.5 pp;全幅:**M2** | 三家外部同配方 [1,3] s + ±0.5 m/s,但**他们是连续行走、没有击球窗**;厂商 ±0.25 已经是 BeyondMimic 的半幅。撞窗概率:[1,3] s → 每 episode 期望 1.24 次推、≈5% 撞进 0.10 s 窗;[10,30] s → 0.124 次、≈0.5%。§3.1 的 v2.3 裁定([10,30] s)方向正确,只是发射器断链没继承。落点已接线:`task.push.enable/recipe=axis_box_6d_v2/velocity_range/interval_range_s`(train.py:451-489, 12228-12238) |
| 推撞的**相位门控**(替代节奏放疏) | 不存在于 push,但 `lateral_perturbation` 已有 `recovery_hold` 相位门控的完整实现 | **S**(与推撞捆绑,优先于"放疏节奏") | 关 | 只在 recovery_hold 窗内触发 | 1 档 | 与推撞全幅同批(**M2**) | **这是仓库里已有的答案**:lateral_perturbation.py 已经实现了"只在 recovery 相位开火"的资格窗,恰好把冲量赶出挥拍/接触窗。把这个门控思路搬给 push,比单纯把 interval 拉长更精准。需要少量新代码,但结构现成 |
| lateral_perturbation L1 处理(冻结冲量实验) | **OFF**(缺 `task.lateral_perturbation` 键 = 历史无 hook 路径,byte-identical) | **X** | — | — | — | — | 它是一个**冻结的两格 CRN 实验单元**(L0 零对照 / L1 处理),不是训练用 DR:幅度/时序是模块内硬编码常数 + 不可变硬安全上限(0.15 m/s 冲量 / 2.0 m/s² / 200 N),**故意不参数化**以防蔓延成"任意时刻任意力"。保持 L0/不挂钩;只借它的相位门控设计 |
| 地形凹凸 terrain rough patch | **OFF**(默认无键 = 平地) | **X** → 最早 **M2** 之后 | 关 | (band 0.01-0.15 m 内,且须为高度场分辨率整数倍) | 1 档 | **M2** 之后再议 | 抬终止率(闸 2),且踩已知的雷:generator 把 env origins 与克隆桌子拆散(07-29 记忆)。任务上我们是室内平地,与厂商 parkour 的地形课程不对齐(§11.2 明确"不对齐")。收益最低、风险最高,排最后 |

### C. 观测噪声(关键在于分两类,不是一个轴)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| **本体感受** obs 噪声(A225 合同下只有这两条) | **ON**,`joint_pos ±0.01 rad`、`joint_vel ±0.5`,Unoise 每步 | **D1 开满** | 现值 | 现值 | — | — | 三闸全过:不改转移、不改支撑集、不造终止事件。外部 **9/9 库 day-1 固定**;厂商甚至**评测(play)时 DR 全关但 obs 噪声保留**。噪声区间与厂商/BeyondMimic 一致 ✅。出处:hope_env_cfg.py:2833-2954 |
| A225 合同里的 **14 条干净通道**(实际/教师底座状态、教师关节 p/v、actions、3 条拍朝向、3 条任务期望接触、desired_base_xy、time_to_contact、time_to_teacher_start) | **零噪声**(有意设计) | **X**(至少到 M2) | 关 | 待定,不早于 M2 | — | **M2** 之后单独立项 | **这是本次盘点最重要的一条修正**:07-31 尽调的 obs 噪声清单描述的是 legacy 177/194-D 合同,**现役 N1 走的是 `actor_obs_contract: action_ball_a225`,那个 ABI 里根本没有 base_ang_vel / projected_gravity / motion_anchor / racket_target 通道**。给任务/教师通道加噪 = 降低任务目标可观测性 = 踩闸 1(改支撑集),而击中层收入本来就恒为 0(§7 死核)。**顺带记一笔洞**:合同里没有 base_ang_vel/projected_gravity,真机 IMU 噪声将来无处对应 |
| 厂商式 obs `scale` + `history=8` 帧堆叠 | 无 | **X** | — | — | — | — | 厂商同底盘实跑(ang_vel scale 0.25、joint_vel scale 0.05、全组 history=8),但**破坏 110/177/194-D 部署契约**,必须独立臂 + 同批改部署侧。§11.3 已排在低优先第 8 条。与延迟臂捆绑才有意义(堆叠是策略自估延迟的标准配方) |
| 评测口径:play 带不带 obs 噪声 | 我们 `eval_deterministic` 把 noise_scales 归零;厂商 play 保留 obs 噪声 | **D1**(只做记录) | — | — | — | — | 不是训练轴,是**跨栈比较的陷阱**。判读文档里必须明示两种口径,否则拿我们的 deterministic 数去比厂商的带噪数会系统性偏乐观。§11.2 末行 |

### D. 任务目标 / 拍状态噪声(A1 八旋钮 —— 必须拆开看)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| `target_noise_white` | **0.0** | **S**(可早开) | 0.0 | 场地实测 **σ = 0.0019 m** | 1 档 | **M0**(死核修好即可) | 关键量纲比较:1.9 mm 相对 `sigma_pos_min = 0.075 m` 只有 **2.5%**,几乎不动核的输出。**它其实不踩闸 1**(零均值、幅度远小于容差),真正的门槛只是"别在死核期加任何东西"。实测值躺在注释里没用 |
| `target_noise_ar1_sigma` | **0.0** | **S**(可早开) | 0.0 | 场地实测 **σ = 0.0052 m** | 1 档 | **M0** | 5.2 mm = 核宽的 **6.9%**,同上。AR1 相关噪声比白噪更像真实跟踪误差 |
| `target_delay_steps` | 0 | **X** → **M2** | 关 | 待标定 | 1 档 | **M2** | 改接触相位,与执行器延迟叠加会双倍吃掉 0.02 s 位置窗 |
| `target_dropout_prob` / `target_post_strike_dropout_s` | 0.0 / 0.0 | **X** | 关 | 待定 | — | **M2** 后再议 | dropout 会**制造"目标突然消失"的状态**,在击中层收入恒为 0 时纯粹是加噪不加信 |
| `target_bias_per_swing` | 0.0 | **X** | 关 | 待定 | — | **M2** 后 | **非零均值**,直接违反闸 1 的核心条件:它系统性地移动目标,等于每挥拍换一个任务 |
| `target_jitter_pos_per_s` / `target_jitter_vel_per_s` | 0.0 / 0.0 | **X** | 关 | 待定 | — | **M2** 后 | 同 dropout,先修死核再谈 |
| `midswing_resample_prob` | 0.0 | **X** | — | — | — | — | 改支撑集最狠的一条(挥到一半换目标),且与 planner revisions / post_strike_t1 互斥。不排产 |
| `achieved_target_mix_prob`(HER 式混合) | 0.0 | **X** | — | — | — | — | 与 question_bank/CQ solver 互斥;本臂走 immutable_tape,路径上根本没有 bank。不排产 |
| **非对称 critic 的免疫性**(记录项) | `HOPECriticCfg.racket_target_vel_w_live` 永远看真值 | — | — | — | — | — | 好消息:即便把 actor 侧目标搞脏,**value estimation 不会跟着坏**。这降低了 A1 旋钮的风险等级,但不改变"死核期不加噪"的裁定 |

### E. Reset / 起点分布

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| reset 位姿 / 速度 / 关节噪声 | **全零**,`canonical_ready_mode=true` 把四组区间强制成 (0,0),`stand_start_prob=1.0`,`hold_steps_range=(0,0)`,`post_swing_start_prob=0`,`clip_switch_prob=0` | **S** | 全零(保持) | 厂商 A3 值:位姿 x/y/yaw **±0.1**、速度六维 **±0.2**、关节 **±0.15 rad**(关节速度 0) | **3 档**(0 → 1/3 → 2/3 → 满) | 1/3 档:**M1** 且 Δp_term ≤ 0.5 pp;后续每档需再测一次 Δp_term | 双重踩闸:改起点分布 = 改支撑集(闸 1),且抬终止率(闸 2)。§9 已把起点塌成"每 action 一个 delta 函数"记为病灶(自我收窄循环),所以终态**必须**离开全零 —— 但不是现在。**这是唯一值得写的新代码**:`canonical_ready_mode` 是验证器强制的 on/off,不是幅度旋钮,需要新增分级键。注意:**这不是重开 RSI**(§9.4 反 RSI 裁定不动摇),只是放宽 ready 球 |
| 失败加权自适应 RSI 采样器 | 建好、checkpoint-safe(α=0.001,uniform floor 0.1,λ=0.8),但**每个已注册任务的控制流都绕过它**,零臂可达 | **X** | — | — | — | — | 结构不可达。要用它得改任务控制流。§21 裁定不排产,记台账 |

### F. 任务难度 / 题目采样(定义上就是课程,但现在被冻结)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| **32-arm 球/题难度课程**(16 物理轴 × 双侧) | **冻结**在 manifest-initial:`task.racket.action_ball_diagnostic_unauthorized=true` 让 frozen_evaluation_boundary 直接短路(hope_commands.py:9060-9071),canary/heldout 永不请求 | **S**(但有三条硬前置) | manifest initial | manifest maximum(L=1.0 且 rho=1.0) | 机制自带 **5 档** `LEVELS=(0,0.25,0.5,0.75,1.0)` + 全局 `JOINT_RHOS` 同 5 档(action_ball_curriculum.py:61-62) | **M1** + §13 R1(可逆化)落地 + 154 行样本量修复 | 这是**唯一一条按定义就该分阶段**的轴,也是仓库里最强的现成 in-loop 调度器(Wilson 95% CI、冻结策略 canary→heldout 双窗口、≥100 update 节拍、state_dict + sha256 交叉校验)。三条硬前置缺一不可:(a) §13 裁定单臂**不可逆锁定**;(b) marginal 只用 154 行新带样本 → 有效扩张阈值 **3.25%**(设计目标 10%),真实失败率 5% 的臂有 **79%** 概率第一轮就被永久锁死;(c) **ADR 的前提"至少解开一个环境"在 strike_opportunity_count=0 时不满足**,现在解冻只会得到一串永久锁 |
| question_bank / CQ 求解器 | 旁路(本臂 `action_ball_target_source=immutable_tape`,三个 bank 键全空,继承自 HOPEPingPongActionBall.yaml:209-212 而非叶子) | **X**(本阶段) | — | — | — | — | §20 新架构第一阶段目标源用 clip,此时"不需要题库"。与 32-arm 是**两条独立轴**(即便走 tape,32-arm 仍管来球物理参数)。记忆里"题库是真卡点"针对的是别的臂 |
| `balanced_clip_sampling` | ON,seed=0,确定性轮转(计数差 ≤1) | **D1**(保持) | — | — | — | — | 不是噪声,是采样公平层;N1 单 action UID 下退化成平凡单 clip 情形。仓库独有的强项,零成本保留 |

### G. 策略侧随机性(容易被漏掉,但 §16 说它是 binding 的)

| 轴 | 现役状态 | 档 | 开局幅度 | 终态幅度 | 分档数 | 升档触发 | 理由与出处 |
|---|---|---|---|---|---|---|---|
| **PPO 探索噪声 `init_noise_std`** | **0.02**,且被 train.py:5689 / 6993 **硬性验证器钉死** | **S**(方向是**调大**,不是加满) | 0.02(现值) | 待定,建议先试 0.05-0.10 | 2 档 | **M0**(死亡尖峰修好之后,一步都不能提前) | §16.1 D:31 关节 × 每维熵 -2.494 nat = **-77.3 nat**;entropy_coef=0.01 对每个 log σ 的梯度是恒定 **-0.01**(往上推),而 surrogate 在 dense 侧只有 **≈0.007**(推不过熵)、死亡侧 **≈3**(往下狠压)。**结论:现在 σ 是被"死亡"和"熵"共同决定的,和击球质量无关。** 在 -72 修好之前调大 σ = 更快摔死 = 反向。这是全表**唯一一条"现役值本身就是病灶"**的随机性轴 |
| `adaptive_sigma` 核宽课程 | **ON**(从 A3VendorV2 继承),三通道单调收缩,`sigma_update_every=500`,pos [0.075,0.50] / vel [0.50,3.0] / normal [0.262,2.10] | **D1**(保持,但方向要复核) | — | — | 连续 | 误差 EMA 自驱 | 它是**误差自适应**的(符合"σ课程钦定"),且是全仓最干净的 in-loop hook 骨架。但注意:它只**单调收缩**、永不回宽,而 §7 的死核问题恰恰是"太窄"。**它不会造成死核(下限 0.075 就是死核值本身),但也永远救不了死核** —— 修死核要动的是远场核形状/Cauchy 尾/racket_progress,不是这个课程 |
| 死亡尖峰 -72.0(不是随机性,是全表的总闸门) | ON | **必修** | — | 目标:post-dt |罚| ≲ 视野收入 2%(即 **≈ -0.2/次**,或改用 termination bootstrap) | — | 立即 | 三库最大单步罚项 post-dt ≈ -0.2 = 视野收入 9.5 的 2%;我们是 1200%,**差约 600 倍**。修完后盈亏平衡命中率从 69% 降到(若改 -7.2)**17.9%** —— 这个数字直接变成 M2 的门槛。落点:train.py:8896 DIRECT 表 + yaml 键 `death_penalty_weight`(train.py:11171-11184) |

### H. 发射器口径不一致(必须先统一,否则上面整张表都失真)

| 项 | 现状 | 处置 |
|---|---|---|
| `task.domain_rand.stable_ready_plant` | 一个布尔键**同时关掉 base_com + link_mass + PD 增益三条轴**。`launch_n1_reward_screen_diagnostic.py` 和 `launch_n1_vendor_baseline_diagnostic.py` **设了**;现役的 `launch_action_ball_a225_four_arm_diagnostic.py` **没设** | **D1 动作:把 DR 姿态写进每个发射器的 claim 摘要**。今天"哪些物理 DR 开着"取决于跑的是哪个脚本,不是仓库统一默认 —— 读者必须查具体 argv。这直接违反"统一队列表 + 依赖核对单"的发射工序教训 |

### 22.3 与里程碑绑定的日程

## 与谱系里程碑绑定的分档日程

**总原则:每一档 = 一次新发射或一次续训边界上的 argv 变更,不是 in-loop。** 理由见 verdict 的元裁决(ADR 前提不满足 + startup 级 DR 只能在 env 构造抽 + CurriculumCfg 全仓为空)。

---

### 阶段零 — 信号侧修复(现在;**DR 一格不动**)

**这是全表的总前置。在它验收之前,任何 DR 变更都是在给"为什么还是学不会"添混淆项。**

| 动作 | 目标值 | 落点 |
|---|---|---|
| 压平死亡尖峰 | -72.0/次 → 量级 **-0.2/次**(= 三库口径:视野收入 9.5 的 2%),或改用 termination bootstrap | `scripts/train.py:8896` 的 v2 DIRECT 表 + yaml 键 `death_penalty_weight`(train.py:11171-11184)。**顺手修**:train.py:~11172 中文注释写 "-1800" 与 DIRECT 表实际 -3600 不符 |
| 修死核 | 让 e ∈ [0.15, 0.50] m 区间有非零梯度(现在 e=0.30 m 时 exp(-16) = 1.1e-7,**梯度精确为零**) | §7.4 的核形状菜单;`racket_progress` 现在的远场上限只有 +0.03/步(clamp ±0.15 m/步 × 权重 10 × dt) |
| 钉住 rsl_rl 版本 | 写进 training receipt | §16.1 C4 的版本警告:全仓 grep 不到任何 rsl-rl 版本 pin。若实际是 5.x,actor/critic 分开裁剪,C4 直接降级为不成立 |
| **DR 姿态** | **原封不动**:摩擦 (0.3,1.6)/(0.3,1.2)、link_mass ±15%、CoM ±0.025/0.05/0.05、关节零点 ±0.01、Kp(0.8,1.2)/Kd(0.7,1.3)、本体感受 obs 噪声开、延迟 0、推撞关、reset 全零、32-arm 冻结 | 保证 reward 侧改动是单变量可比的 |
| **发射器口径统一** | 把 `stable_ready_plant` 的取值和五条物理 DR 的实际幅度写进每个发射器的 claim 摘要 | 三个 N1 发射器现在 DR 姿态不一致;不统一的话下面所有 A/B 都不可比 |

**M0 验收:** 死亡尖峰量级达标 + 死核修复后在 e=0.3 m 处梯度可测非零 + 终止占比基线重测(建立 Δp_term 的比较基准)。

---

### 阶段一 — 拍状态 mimic(§20:任务通道 dense 化,第一阶段目标源用 clip)

**这一阶段任务已被大幅简化(不需要题库),是**验证"三闸判据本身"**的最好窗口。**

| 轴 | 动作 | 幅度 |
|---|---|---|
| 全部 D1 物理 DR | 继续保持 | 现值 |
| 本体感受 obs 噪声 | 继续开满 | joint_pos ±0.01 / joint_vel ±0.5 |
| `target_noise_white` / `target_noise_ar1_sigma` | **M0 之后可开**(唯一在这一阶段新开的轴) | 场地实测 **0.0019 m / 0.0052 m** —— 相对 `sigma_pos_min=0.075 m` 只有 2.5% / 6.9%,是真正的"零成本诚实" |
| `init_noise_std` | **M0 之后**试 0.02 → 0.05 一档 A/B | 硬验证器在 train.py:5689/6993,提值要同步改验证器。**M0 之前一步都不能提**(更大探索 = 更快摔死) |
| 其余全部 S/X 轴 | 不动 | — |

**M1 门槛(阶段一 → 二):`strike_opportunity_count > 0` 且 window 命中率连续 3 个 ≥100-update 评测窗 ≥ 20%**(现役 update250 是 5.01%,即要求 4 倍)。

---

### 阶段二 — 学会击中(strike_opportunity_count 首次非零之后)

**这一阶段第一次允许踩闸 2,但每开一轴都要配一格 200-iter smoke 测 Δp_term。**

| 轴 | 动作 | 幅度 | 前置 |
|---|---|---|---|
| **执行器延迟** | 进第 1 档 | `control_step_action_delay_min=0, max=1`(= 20 ms = 正好 1 个 `strike_window_pos_s`) | M0 + M1;Δp_term ≤ 0.5 pp |
| **推撞** | 进第 1 档 | `recipe=axis_box_6d_v2`、vel_xy **±0.125** m/s、vz ±0.05、roll/pitch ±0.13、yaw ±0.195 rad/s、`interval_range_s=[10,30]`(每 episode 期望 0.124 次推,≈0.5% 撞进 0.10 s 击球窗) | **必须等 M0**:推撞的全部风险都在闸 2。同时给 Wave-P 6 臂补收口(判读或作废,§3.1) |
| **reset 噪声** | 进第 1 档(1/3 幅) | 位姿 x/y/yaw ±0.033、速度六维 ±0.067、关节 ±0.05 rad。**需要新增分级键**(`canonical_ready_mode` 是 on/off 不是旋钮) | M0 + M1;这是全表唯一值得写的新代码 |
| 摩擦 / 质量 / CoM / PD / 零点 | 继续不动 | 现值 | — |
| A1 其余六旋钮、32-arm、terrain、腕拍质量 | 继续关 | — | — |

**M2 门槛(阶段二 → 三):exact 命中率 ≥ 修复后的盈亏平衡率。** 若死亡改到 -7.2,盈亏平衡 = 7.2/(7.2+32.98) = **17.9%** → 取 **20%** 作为门槛(现役 update250 是 0.23%)。这个门槛不是拍脑袋:它就是"挥拍在期望上不亏"的那条线。

---

### 阶段三 — 上台质量(capture / return 首次非零之后)

| 轴 | 动作 | 幅度 |
|---|---|---|
| **执行器延迟** | 进第 2 档(终态) | `max=2` 控制步 = 40 ms = 厂商 [0,2] 全值 |
| **推撞** | 进第 2 档(终态) | 厂商全幅 ±0.25/±0.1/±0.26/±0.39,节奏 **[5,15] s**;**优先改用相位门控**(照搬 lateral_perturbation 的 `recovery_hold` 资格窗),门控落地后节奏可回到 [1,3] s 而不撞窗 |
| **reset 噪声** | 进 2/3 档 → 满档(厂商 A3 值) | 位姿 ±0.1、速度六维 ±0.2、关节 ±0.15 rad |
| **32-arm 课程解冻** | 关掉 `action_ball_diagnostic_unauthorized` | **三条硬前置全部落地才准解冻**:(a) §13 R1 可逆化(`arm_reopen_after_epochs`);(b) 154 行样本量 / 3.25% 有效阈值修复;(c) M2 已达成(= ADR 的"至少解开一个环境"前提满足) |
| **摩擦外扩** | 单格消融 | (0.2,1.8)/(0.2,1.5)(厂商值) |
| **腕 + 拍质量 ±20%** | 若还有余力 | 需新键(按 body 分组的 mass range) |
| A1 delay/dropout/jitter | 逐个单格消融 | 待标定 |
| terrain 凹凸 | 最后,若确有必要 | band ∈ [0.01,0.15] m |

---

### 必须等信号侧修好才能开的轴(明确点名)

| 阻塞源 | 被阻塞的轴 | 为什么 |
|---|---|---|
| **-72 死亡尖峰**(§16) | 推撞(全部)、reset 状态噪声(全部)、摩擦低端外扩、terrain 凹凸、执行器延迟(全部)、`init_noise_std` 上调 | 这六条**全部会抬高终止率**。dense 归一化 advantage ≈0.007 vs 死亡方向 ≈3,**约 500 倍**;盈亏平衡 69% 已让"不挥拍"成为结构最优解。每抬一个百分点的终止率,就是给这个退化解再补一笔贴 |
| **σ=0.075 m 死核**(§7) | A1 八旋钮全部(包括本来很小的 white/AR1)、A225 任务通道加噪、32-arm 解冻 | 击中层收入恒等于 0 时,给任务目标加噪是在**零梯度带上加抖动** —— 纯粹损失,没有任何学习收益可换。white/AR1 因为幅度只有核宽的 2.5%/6.9%,可以在 M0(而不是 M1/M2)之后就开 |
| **§13 的不可逆锁定 + 154 行样本量** | 32-arm 课程解冻 | 现在解冻只会在训练最早期(~3k iter)用最弱的策略把 26-28 条臂逐个永久锁死。**用坏的调度器不如不用调度器** —— HITTER 完全没有课程也拿到 92.3% 回球率 |
| **发射器 DR 姿态不统一** | 所有 A/B | 三个 N1 发射器的 `stable_ready_plant` 取值不一致 → 跨臂比较全部失真。这条是阶段零的 D1 动作,零成本 |

### 22.4 建议(排序)

**N1(P0):【P0】DR 一格不动,先修 -72 死亡尖峰与 σ 死核;把所有 DR 变更冻结到 M0 验收之后**
- 证据:§16.1:死亡 -72.0/次(-3600 × dt 0.02),γ=0.99 视野 100 步 > 平均 episode 124.11 步(2.48 s),以死亡结尾的 episode 里每个状态回报 -72×0.99^k ∈[-72,-21],dense 收入 7.4 只是零头;dense 归一化 advantage ≈0.007 vs 死亡方向 ≈3,相差约 500 倍;三库最大单步罚项 post-dt ≈ -0.2 = 各自视野收入 9.5 的 2%,我们是 1200%,差约 600 倍。§7:σ_pos_min=0.075 m 在 e=0.30 m 处 exp(-16)=1.1e-7,梯度精确为零;唯一远场梯度 racket_progress 上限只有 +0.03/步。盈亏平衡命中率 72/(72+32.98)=69%,'不挥拍'是当前 reward 下的结构最优解。
- 落点:scripts/train.py:8896(v2 DIRECT 表)+ yaml 键 death_penalty_weight(train.py:11171-11184);§7.4 核形状菜单。顺手清两处注释漂移:train.py:~11172 写 '-1800' 与 DIRECT 表 -3600 不符;HOPEPingPongHitter.yaml:519-520 的 '+/-15% reset-level' 与现役 startup 级 Kp(0.8,1.2)/Kd(0.7,1.3) 不符。另需把 rsl_rl 版本写进 training receipt(全仓 grep 不到任何 pin,§16.1 C4 的成立与否完全依赖它)。
- 风险:改 reward 会打断与在跑臂的单变量可比性 → 必须走新臂而不是原地改。若不改 reward 而先加 DR,则所有后续 A/B 都被 -72 的噪声淹没,等于白跑。

**N2(P0):【P0】观测噪声分两类处理:本体感受 day-1 开满并永久保持;任务/教师通道保持零噪至少到 M2。答'能不能一直开满'——本体感受能,任务通道不能。**
- 证据:外部 9/9 库的 obs 噪声都是 day-1 固定;厂商 instinct_mj 连评测(play)时 DR 全关都保留 obs 噪声;PBHC 虽有 add_noise_currculum 但默认 False;唯一 ramp obs 噪声的 IsaacLab dexsuite 是 manipulation + ADR 家族(前提不满足)。我们现役 A225 合同下只有 joint_pos ±0.01 rad / joint_vel ±0.5 带噪,区间与厂商/BeyondMimic 一致。反面:该合同的另外 14 条通道(实际/教师底座状态、教师关节 p/v、actions、3 条拍朝向、3 条任务期望接触、desired_base_xy、time_to_contact、time_to_teacher_start)全部零噪 —— 给它们加噪 = 降低任务目标可观测性 = 改支撑集,而击中层收入本来就恒为 0。
- 落点:hope_env_cfg.py:2833-2954(HOPEActionBallA225ObservationsCfg / TrainableObservationsCfg)。同时必须入档一条重要修正:07-31 尽调 §2/§4/§8 的 obs 噪声清单描述的是 legacy 177/194-D 合同,现役 N1 走 actor_obs_contract: action_ball_a225(叶子 yaml:12),那个 ABI 根本没有 base_ang_vel / projected_gravity / motion_anchor / racket_target 通道 —— 现役噪声覆盖比文档暗示的窄得多。
- 风险:两个洞要记账:(a) A225 合同没有 base_ang_vel/projected_gravity,真机 IMU 噪声将来无处对应,是 sim2real 的结构性缺口;(b) 评测口径差异 —— 我们 eval_deterministic 归零噪声、厂商 play 保留噪声,跨栈比数会系统性偏乐观,判读文档必须明示。

**N3(P0):【P0,零成本】统一三个 N1 发射器的 DR 姿态口径,并把实际幅度写进每个发射器的 claim 摘要**
- 证据:task.domain_rand.stable_ready_plant=true 一个布尔键同时关掉 base_com + randomize_link_mass + randomize_pd_gains 三条轴。launch_n1_reward_screen_diagnostic.py 与 launch_n1_vendor_baseline_diagnostic.py 设了它;现役的 launch_action_ball_a225_four_arm_diagnostic.py 没设。也就是说'哪些物理 DR 开着'取决于跑的是哪个脚本,不是仓库统一默认。
- 落点:scripts/train.py:15277-15347(捆绑关闭逻辑与 restricted-launch 守卫);三个 launch_*.py 的 claim 组装处。
- 风险:不修的话,上面整张分档表和所有跨臂 A/B 都失真。这直接违反'统一队列表 + 依赖核对单'的发射工序教训,且属于'说没有先查三层'的典型:机制码、实验史裁定、现役 argv 三层不一致。

**N4(P1):【P1】执行器延迟必须分两档进(0→1→2 控制步),不能一次上满厂商 [0,2]**
- 证据:决定性数字:strike_window_pos_s = 0.02 s = 正好 1 个控制步 @50 Hz(HOPEPingPongActionBallA3VendorV2.yaml:15)。2 步延迟 = 40 ms = 位置窗的 2 倍,一次上满等于把位置窗整个吃掉。外部双重先例都指向'从零长进来':IsaacGymEnvs AllegroHandDextremeADR 的 action_latency init_range 是字面 [0,0] 零宽起步(27 个 ADR 参数里有 8 个是零宽起步);AllegroHandDextremeManualDR 用 actionLatencyScheduledSteps 做纯时间线性斜坡(action_latency_min=1 → actionLatencyMax=15)。修正一处外部数据:该值是 10,000,000 不是 2,000,000 —— 2M 那个值写在 ADR 的 yaml 里,而 ADR 子类完全 override 了 apply_action_noise_latency、根本不读它。
- 落点:task.actions.control_step_action_delay_min / control_step_action_delay_max —— 已完全接线,无需改码。机制在 hope_actions.py:684-830(_EpisodeSharedPolicyActionDelay,整 31-D q_des 延迟,每 env 每集抽一次,min=max=0 是 byte-identical no-op);现役 N1 叶子 yaml:21-25 与发射器 launch_action_ball_a225_four_arm_diagnostic.py:1391-1392 双重清零。顺带记录:robots/actuator.py:11-75 的 DelayedImplicitActuator 仍是真死代码(agibot_a3.py 五个 actuator 组全是 plain ImplicitActuatorCfg),尽调说的'延迟是死代码'只对了一半。
- 风险:延迟改接触相位,对毫秒/厘米级的击球是直接税(尽调点名它是最可能砸掉接触时序的一条)。阶段一/二早期绝不能开。若同时开 target_delay_steps,两者叠加会双倍吃掉位置窗。

**N5(P1):【P1】推撞:起步用厂商半幅 + [10,30] s 节奏;终态优先做相位门控而不是单纯放疏节奏**
- 证据:厂商 instinct_mj 的 ±0.25/±0.1/±0.26/±0.39 m·s⁻¹ @[1,3]s 恰好已是 BeyondMimic(±0.5/±0.2/±0.52/±0.78 @[1,3]s)的半幅 —— 但他们是连续行走、没有击球窗。撞窗概率算给你:平均 episode 2.48 s、击球窗 0.10 s,节奏均值 2 s → 每 episode 期望 1.24 次推、≈5% 撞进窗;节奏均值 20 s → 0.124 次、≈0.5%。所以外部的 [1,3] s 不能抄,§3.1 的 v2.3 裁定 [10,30] s 方向正确。仓库里已有现成的相位门控实现:lateral_perturbation.py 把冲量门控到 recovery_hold 资格窗,恰好把扰动赶出挥拍/接触相位 —— 这比放疏节奏精准得多。
- 落点:task.push.enable / recipe=axis_box_6d_v2 / velocity_range / interval_range_s(train.py:451-489, 12228-12238,已完全接线)。现役 N1 叶子 yaml:27-37 把每个字段显式 null(train.py 拒绝 enable=false 携带休眠幅度,所以是干净全关),发射器 :1393 另传一次。相位门控可借 lateral_perturbation.py:271-424 的资格窗结构。同时补 §3.1 的断链:Wave-P 6 臂(p1push_{w,v}_{p02,p035,p05},07-20/21 发射,science rc=0)至今无判读记录,EXP-P1 文档底部'运行表/决定'仍停在 preregistered、与头部运行态自相矛盾 —— 要么判读要么正式作废。
- 风险:推撞是全表最直接的闸 2 违反者,撞进击球窗 = 直接税击中收入层(红线)。合并抽签 CLI 未接线是已知 TODO,v2.3 用双独立事件近似时两种推可能同帧叠加。另:大踢恢复可能要求真机给不出的加速度(包络仅剩 ~3.5% 余量)。

**N6(P1):【P1】不要新写 scheduler —— 但也不要现在复用 32-arm。答'有没有现成机制可复用':有两个,一个语义不匹配、一个前提不满足,所以答案是'现阶段用分档发射,不用 in-loop'**
- 证据:三条论证:(1) ADR 家族的扩张条件是 success rate 过阈,我们 exact 0.23%/window 5.01%/capture=return=0 → 扩张永不触发,写了是 no-op;OpenAI 原文写明 ADR 从'concentrated on a single environment'起步、'additional environments are only added when a minimum level of performance is achieved'(arXiv 1910.07113)。(2) 外部 9 库里 8 个 day-1 固定 DR,唯一自带 in-loop DR 幅度课程的是 IsaacLab dexsuite(manipulation, performance-gated, difficulty 0..10, difficulty_frac<0.1 时冻结);所有 locomotion/tracking 库 ramp 的是任务难度(terrain level / command 速度)而非 DR 幅度;PBHC 的 add_noise_currculum 与 born_offset_curriculum 默认都是 False。(3) 我们 5 条现役物理 DR 全是 startup 级(env 构造时抽一次),in-loop 改需要新 CurriculumTermCfg hook,而全仓 CurriculumCfg 至今是 pass(tracking_env_cfg.py:286-290),零脚手架。两个现成 hook:32-arm 课程(LEVELS=(0,0.25,0.5,0.75,1.0) 5 档阶梯 + 同 5 档全局 JOINT_RHOS、Wilson 95% CI、冻结策略 canary→heldout 双窗口、≥100 update 节拍、state_dict+sha256 交叉校验 —— action_ball_curriculum.py:61-62, 502)语义是题目难度不是 DR 幅度;adaptive_sigma(sigma_update_every=500、误差 EMA 驱动)方向是单调收缩而非扩张。
- 落点:现阶段:把档位写进发射器 argv(每档一次新 launch 或 resume)。将来若真要 in-loop DR 幅度课程,复用 32-arm 的 5 档阶梯 + Wilson 判据 + checkpoint 骨架,按 IsaacLab dexsuite 的 initial_final_interpolate_fn 结构组织(任意嵌套 int/float/tuple 按 difficulty_frac 线性插值,frac<0.1 时返回 NO_CHANGE 冻结)—— 那是最贴切的现成模板,不必从零设计。
- 风险:最大的风险是'觉得有现成机器就该用' —— §13 已裁定 32-arm 现在通电只会在 ~3k iter 用最弱策略把 26-28 条臂逐个永久锁死。HITTER 完全没有课程也拿到 92.3% 回球率,固定难度本身不丢人。用坏的调度器不如不用调度器。

**N7(P2):【P2】32-arm 解冻的三条硬前置(缺一不可),在此之前 action_ball_diagnostic_unauthorized=true 保持**
- 证据:§13.3 三条 critical:(a) 单臂晋级不可逆 —— marginal 分支 quality_bad 或 too_hard 一律 statuses='decided'(lock_marginal),代码中无任何路径改回可探;所有外部先例(ADR 队列清空后同一边界无限重测、legged_gym/IsaacLab/PBHC 每次 reset 重判)都是可逆的。(b) 判决样本 n=154(HELDOUT_NEW_BAND_MIN),在 z=1.96、带 [0.075,0.125] 下 'too_easy 继续扩'实际要求 F≤5/154 = 3.25%,而设计目标是 10% —— 真实失败率 5% 的臂有 79% 概率第一轮永久锁死,7% 的是 96%。更糟:新带比例 = 1/5,768×0.2 = 153.6 < 154,即代码底线窗口下期望新带行数就低于门限。(c) ADR 前提'至少解开一个环境'在 strike_opportunity_count=0 时不满足。
- 落点:§13.4 R1:新增 arm_reopen_after_epochs(默认 0 = byte-identical),把 'lock/bound' 改成'休眠 + 到期重开',落在 action_ball_curriculum.py 的 _apply_formal_evidence marginal 分支(:4683-4700)与 _reselect_arm;_Progress 增 arm_decided_epoch 字段。同时修 :4640-4643 的 'NB<154 → quality_bad → 等价 too_hard → 永久锁'这条把'样本不够'当'太难'的路径(ADR 的做法是 deque 未满 256 时什么决定都不做,只等)。
- 风险:§13 已裁定 R1 可逆化是 R9 并行扩张的硬前置。另注意 high 级问题:整窗零容忍安全 blocker 门控的是单臂新带判决 —— 一次来自 center/interior 行(占 80%)的撞台会永久锁死一条完全无关的轴,残余不安全率高于 ~1e-5 晋级流水线就会被无关事件随机掐断。

**N8(P2):【P2】reset 状态噪声分 3 档进(0 → 1/3 → 2/3 → 厂商 A3 满值)—— 这是全表唯一值得写的新代码**
- 证据:现役 canonical_ready_mode=true 把位姿/速度/关节四组区间全部强制成 (0,0),stand_start_prob=1.0、hold_steps_range=(0,0)、post_swing_start_prob=0、clip_switch_prob=0 —— §9.2 已把这个'起点分布塌成每 action 一个 delta 函数'诊断为自我收窄循环。终态目标用厂商 A3 值(§11.3 第 5 条):位姿 x/y/yaw ±0.1、速度六维 ±0.2、关节 ±0.15 rad、关节速度 0 —— 比 BeyondMimic 的速度噪声温和 2.5×、关节宽 1.5×。
- 落点:需要新增分级键:canonical_ready_mode 今天是验证器强制的单一 on/off 开关,不是幅度旋钮;把它改成 false 会退回材质上不同的 legacy reset 架构,不是想要的。所以新键应该是'canonical_ready + 分级噪声球半径'的组合,落在 HOPEPingPongActionBall.yaml 的 motion 块 + 对应验证器。
- 风险:双重踩闸(改支撑集 + 抬终止率),必须 M0 + M1 之后,且每档配一格 200-iter smoke 测 Δp_term ≤ 0.5 pp。明确一句:这不是重开 RSI —— §9.4 的反 RSI 裁定不动摇,只是放宽 ready 球。另:仓库里那个失败加权自适应 RSI 采样器(α=0.001、uniform floor 0.1、λ=0.8、checkpoint-safe)对每个已注册任务都结构不可达,不排产。

**N9(P2):【P2】A1 八旋钮拆开处理:white/AR1 用场地实测值可以在 M0 之后就开;delay/dropout/jitter/bias/resample/HER-mix 一律等 M2**
- 证据:量纲比较是关键:场地实测 white σ=0.0019 m、AR1 σ=0.0052 m,相对 sigma_pos_min=0.075 m 只有 2.5% 和 6.9% —— 这两条其实不踩闸 1(零均值、幅度远小于任务容差),真正的门槛只是'别在死核期加任何东西'。而 target_bias_per_swing 是非零均值(系统性移动目标 = 每挥拍换一个任务)、dropout 制造'目标突然消失'的新状态、midswing_resample 是改支撑集最狠的一条(挥到一半换目标)。八个旋钮现役全零,实测值躺在注释里没用。
- 落点:task.racket.target_noise_white / target_noise_ar1_sigma(现役 N1 叶子 yaml:53-54 显式 0.0);其余六条同在 task.racket.* 下,均为普通 float/int CLI 键,无需改码。互斥守卫:achieved_target_mix_prob>0 与 question_bank/CQ solver 互斥;midswing_resample_prob>0 与 planner revisions / post_strike_t1 互斥(hope_commands.py:2361-2546, 3538-3616, 15077-15099)。
- 风险:好消息降低风险等级:非对称 critic(HOPECriticCfg.racket_target_vel_w_live)永远看真值,所以把 actor 侧目标搞脏不会连带坏掉 value estimation。坏消息:在击中层收入恒为 0 时,给任务目标加噪是在零梯度带上加抖动 —— 纯损失无收益。

**N10(P3):【P3】init_noise_std=0.02 是'现役值本身就是病灶'的唯一一条随机性轴,但方向是调大、且必须排在 -72 修复之后**
- 证据:§16.1 D/C5:31 关节 × 每维熵 0.5·ln(2πe·0.02²) = -2.494 nat,合计 -77.3 nat;entropy_coef=0.01 对每个 log σ 的梯度是恒定 -0.01(往上推 σ),而 surrogate 对 log σ 的梯度在 dense 侧只有 ≈0.007(推不过熵)、死亡侧 ≈3(往下狠压)。结论原文:现在 σ 是被'死亡'和'熵'共同决定的,和击球质量无关。它与 σ=0.075 m 死核、racket_progress 上限 +0.03/步 三者共同锁死了 strike_opportunity_count=0。
- 落点:cfg/algo/ppo.yaml:42 是 1.0,但 ActionBall 路径被 scripts/train.py:5689 与 :6993 两处硬验证器钉死为 0.02 —— 提值要同步改验证器(这是刻意的守卫,不是疏漏,改动要走新臂 + 明确记录)。建议 0.02 → 0.05 一档 A/B。
- 风险:在 -72 修好之前调大 σ = 探索更大 = 更快摔死 = 反向。这条排 P3 不是因为不重要,是因为它的前置比 DR 各轴还硬。

**N11(P3):【P3】阶段三消融清单:摩擦外扩、腕+拍选择性质量、terrain 凹凸 —— 收益递减,排最后**
- 证据:摩擦外扩到厂商 (0.2,1.8)/(0.2,1.5) 两端各宽 0.1-0.3,低端 0.3→0.2 会增打滑抬终止率;腕+拍 ±20% 是厂商'选择性末端质量'哲学 × 我们拍子质量未知的组合(§11.3 第 6 条),但需要新的按 body 分组 mass range 键;terrain 与我们室内平地任务不对齐(§11.2 明确写'不对齐'),且踩已知的雷(generator 把 env origins 与克隆桌子拆散,07-29 记忆),band 还有 [0.01,0.15] m 且须为高度场分辨率整数倍的约束。
- 落点:摩擦:task.plant.robot_material_static_friction_range / robot_material_dynamic_friction_range(已接线,含 robot_material_make_consistent 保证 dynamic≤static);terrain:task.plant.terrain_rough_height_range(已接线,要求 scene.terrain.terrain_type=='plane');腕拍质量需新键。
- 风险:三条都是'厂商有我们没有'的对齐项,容易被当成必做;实际上在击球都还没学会时,它们的边际收益接近零,而 terrain 的实现风险最高。§21 已裁定不扩网络/不扩工程,资源投信号侧。

**N12(记录):【记录,不排产】armature / 关节机械摩擦 / 力矩上限随机化;lateral_perturbation L1;失败加权 RSI 采样器;厂商 obs scale + history=8**
- 证据:前三条:mdp 包里本地只定义了 randomize_joint_default_pos 与 randomize_rigid_body_com 两个函数,armature/关节摩擦/力矩上限三条随机化全部不存在,要写新 EventTerm + 接线(task.plant.zero_joint_friction 是把静摩擦系数设成精确 0.0 的跨引擎单值覆写,不是随机化)。lateral_perturbation 是冻结的两格 CRN 实验单元(L0 零对照 / L1 处理),幅度/时序是模块内硬编码常数 + 不可变硬安全上限(0.15 m/s 冲量 / 2.0 m/s² / 200 N),故意不参数化以防蔓延成'任意时刻任意力'。失败加权 RSI 采样器每个已注册任务的控制流都绕过它。obs scale+history=8 是厂商同底盘实跑,但破坏 110/177/194-D 部署契约。
- 落点:台账项,不动代码。lateral_perturbation 唯一该借的是它的 recovery_hold 相位门控设计(见 P1 推撞条)。obs scale+history 若将来做,必须独立臂 + 部署侧同批改,并与延迟臂捆绑(堆叠是策略自估延迟的标准配方)。
- 风险:把这四条留在'可做清单'里会持续消耗审查注意力。明确写成'不排产'比留白更省事。



---

## 二十三、两处配置核查:immutable_tape 语义 与 base_ang_vel/projected_gravity 缺口(08-03)

**方法**:2 抽取(tape 语义与 curriculum 交互 / 观测合同 git 考古与部署侧可得性)+ 2 对抗核查
(6 修正)+ 1 裁决。判据用 Franco 的原话原意,而非"设计得对不对"。

### 23.1 immutable_tape:配错了(选错工具,不是填错参数)

配错了——但不是参数填错,是**选错工具**:现状与本意不一致,而且不是程度差异,是方向相反。

**一句话**:Franco 要的是"缓存"(档位不变就复用、档位一升就重解),仓库里给的是"永久冻结 + 课程停权"。immutable_tape 的设计岗位从来不是省算力,而是**目标信息消融夹具**(5 个 recipe 里 4 个只能走 tape,online_solver 被合同限死只能配 current_lm 全掩码,hope_commands.py:205-209)。

**(a) tape 之下 32-arm curriculum 还起不起作用:完全不起作用,而且是双重的。**
1. 结构性停权:`immutable_tape` 强制要求 `action_ball_diagnostic_unauthorized=true`(hope_commands.py:5349-5357,"immutable_tape is diagnostic-only";n_actions 必须 =1)。这个开关让 ActionBallCurriculum 构造时 `evaluator_authority=None`、`drain_reset_authority=None`(hope_commands.py:5307-5315,代码注释自陈"A diagnostic authority is deliberately NOT bound")。
2. 运行期短路:真正的把门人是 runner 侧的 `action_ball_frozen_evaluation_boundary`(hope_commands.py:10799-10823)——只要 diagnostic_unauthorized 为真,phase='poll' 直接返回 `{"diagnostic_unauthorized": True}`,**根本不调用** `_action_ball_eval_consume_ready`(hope_commands.py:10845),而后者是 `curriculum.observe_scheduler`(action_ball_curriculum.py:3625)/`stage_selected`(:3974)的唯一调用方。所以课程不是"被调用时报错",是**从头到尾没人调用**;曲线永远停在初始 phase='center'、32 臂全 0.0 档(LEVELS=(0,.25,.5,.75,1.0),action_ball_curriculum.py:61-62)。
3. 反向确认 tape 也不读课程:`FixedQuestionTapeSolver._assert_birth_matches_question`(action_ball_fixed_question_tape.py:838-870)对 domain_levels 做**硬断言**——课程若真升了档,进程是崩,不是换题。也就是说 tape 不是"跟着档位走的缓存",是"逼着课程每次 reset 复读同一个档位的常量"。artifact 实测:`row_count=1`、`selection="constant_row_zero"`、`online_lm_calls=0`(action_ball_fixed_question_tape.py:582-584;活体 JSON 见 configs/action_ball_n1_measured_20260803/fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/immutable_n1_tape.v1.22052606032f.json)。冻的不只是来球:base_spawn、落点、期望接触 p-v-face 元组全是字面浮点数。

**(b) 有没有"档位不变复用、档位升重解"的缓存:没有,一行都没有。** 搜遍 continuous_questions.py / action_ball_sampling.py / strike_spec_torch.py,唯一的 cache 是审计回执的 sha/digest 记账(action_ball_sampling.py:4579-4937),不是解算结果缓存。值得注意的是:**这个正确设计已经以字符串形式写在仓库里但零实现**——未提交的 hope_commands.py:6120-6134 往诊断 payload 里塞了 `"final_curriculum_question_source": "pregenerated_or_cached_band_question_bank"`、`"final_curriculum_reset_operation": "index_precomputed_question_row"`、`"does_not_freeze_final_curriculum_to_one_question": True`。没有任何代码读它。

**(c) 省算力动机今天还成不成立:成立,但只剩当初的一半,而且要修正一处跨文档口径。**
- **口径修正(重要)**:dr_reward_external_diligence_20260731.md §10.2 把 23.48 s/update 里的 13.5-17.8 s(60-75%)记成"reset 仪式"= 逐 env Python 记账残差,那是**估算**;同日晚些的 Pod 分段 profiler(design_audit_and_speedup_20260729.md §8.10)把它拆开了:profiled reset 40.732 s 里 `solver_solve_many` = 33.432 s = **82.1%**,占五轮总 collection 的 **64.7%**;而 `Racket install` 只有 0.202 s。**reset 仪式的大头本来就是求解器本身,不是 Python 记账。**§10.2 的归因应以 §8.10 为准。
- **今天的值**:§8.11(host-only solver result 优化后,同 seed 4096×5)五轮 solver=16.367 s / collection=32.924 s ≈ **49.7%**;均值 6.700 s/update,reset-free update ≈ 2.7 s。也就是 reset 增量 ≈ 4.0 s/update 里 solver 约 3.27 s ≈ **82%**。§8.11 自己的结论原话:reset-heavy update 的主差额仍在 fixed-direction solver/LM。
- 结论:**动机成立且是当前第一杠杆**,这恰恰说明不该用"永久冻结"去换它——应该用真缓存换。同时 §8.11 已排产一条不改语义就能拿走一截的路:固定题带预注册 `cq_n_iters=4/6/8/12`,若 8 次过数值门估省 ≈1.1 s/update(**估算,未验收**)。

**(d) 最小修复(三步,按"不改现役字节"顺序)**
1. **零代码,立刻**:承认 immutable_tape = N=1 消融夹具而非缓存,把这句人话写进 launch_action_ball_a225_four_arm_diagnostic.py 的 claim 摘要,并给它一个显式到期条件(M1 / strike_opportunity_count 首次非零)。现役 argv 在 launch_...py:1382-1385 一次性同时设了 target_source=immutable_tape + diagnostic_unauthorized=true,读者今天看不出这等于把课程停权。
2. **拿掉一半动机、零语义风险**:先跑 §8.11 已排产的 `cq_n_iters` 预注册,把求解器单价压下去。这条和 tape 完全独立。
3. **真缓存(才是本意的落点)**:新增**第三个** source 值 `banded_question_bank`(不动 immutable_tape 现语义,`_ACTION_BALL_TARGET_SOURCES` 在 hope_commands.py:121),reset 时按当前 domain_levels **索引预生成表行**;表按 32-arm 的 5 档 × 臂键分块离线生成。关键:**失效机制不用新写**——课程每次 reset 已经通过 `_action_ball_claim_domain`(hope_commands.py:6994-7042)报出 domain_levels,直接拿它做 key,升档=换块=天然重解;`_assert_birth_matches_question` 的硬断言逻辑按行保留即可。且这个 source **不绑定 diagnostic_unauthorized**,课程保留权威、可升档、可导出。风险:题带规模 = 5 档 × 32 臂 × 每档行数,离线生成与 sha 钉带来新的谱系管理成本;这是它比 tape 贵的地方,也是它唯一的成本。

### 23.2 base_ang_vel / projected_gravity:该加,但理由要换

**该加,Franco 的判断成立——但理由要换一个,原来的理由(缺信息)站不住。**

**(a) 有无表(actor 侧;base_lin_vel 全家族缺席,HOPEPolicyCfg 一律 `base_lin_vel = None`,hope_env_cfg.py:928)**

| 合同(总维) | base_ang_vel | projected_gravity | motion_anchor_pos_b | motion_anchor_ori_b |
|---|---|---|---|---|
| hitter_pure(110) | 有 | 有 | 无 | 无(用 base_forward_xy) |
| stage1_natural_clip_site_v1(170) | 有 | 有 | 有 | 有 |
| deploy_parity(175) | 有 | 有 | 无 | 有 |
| hitter_footwork(177) | 有 | 有 | 无 | 有 |
| deploy_parity_face179(179) | 有 | 有 | 无 | 有 |
| full(180) | 有 | 有 | 有 | 有 |
| deploy_parity_station181(181) | 有 | 有 | 无 | 有 |
| L194 | 有 `[68:71]` | 有 `[164:167]` | 无 | 有 `[62:68]` |
| **stage1_..._paddle_world_v2 / A225 / C225(225)** | **无独立列**(世界系角速度埋在两个 15-D 块的 `[12:15]`) | **无独立列**(可从 6-D 姿态子块代数反解) | 无 | 无 |

**(b) 刻意还是抄前缀漏的:刻意,且留了理由,不是静默遗漏。** 全 git 历史里**没有任何一次 diff 删掉过**这两个词(`git log --all -p` 对两个词名的 `-` 行零命中)——225 家族是**从零写的新合同**,不是砍别人。7e5907a6(08-02 06:43)建 170-D 时两条还都在;同日 d361d1bd(08-02 17:56)建 225-D V2 时同一个 commit 就把理由写进了 docs/interfaces/policy_observation_action.md:428(`projected_gravity` is omitted because actual base orientation already determines it)和 :466-471(legacy `base_ang_vel ±0.2` / `projected_gravity ±0.05` 这套旧噪声旋钮"do not define physically valid noise"给一个位置/姿态/线速度/角速度混装的 15-D 块)。A225/C225(10e3ab14,08-03)是直接切 `STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[:10]` 复用(actor_observation_contract.py:286-293、320-324),所以它们连"重新加回来"的机会都没有——但源头是设计决定,不是手滑。

**(c) 两个 15-D 块装了什么:确实已隐含角速度,重力方向也能代数反解——所以"补信息"这个理由不成立。** 两块都由同一个 `_stage1_pack_base_state_world`(hope_observations.py:310-349)按固定顺序拼:`[0:3]` 世界系位置、`[3:9]` 世界系旋转矩阵前两列(连续 6-D 表示)、`[9:12]` 世界系线速度、`[12:15]` **世界系角速度**。`actual_base_now_world` 取 `robot.data.root_lin_vel_w/root_ang_vel_w`;`teacher_base_now_world` 取对齐后的参考 clip 同构四段。所以:角速度**有**(世界系,不是骨盆系);重力方向**没有显式列**,但 `R^T·[0,0,-1]` 可由 `[3:9]` 反解。**信息意义上是冗余的。**

**真正该拿来支持"加回来"的三条理由(与信息量无关):**
1. **来源不同**:15-D 块的底座位姿真机侧要靠 OptiTrack mocap,而 mocap 按团队自己的"Deploy-Available Signal Set"(policy_observation_action.md:1100-1120)**只在 PLAY 期间可得**;base_ang_vel/projected_gravity 走骨盆 IMU,是**永远可得**的机器人侧信号。加回来不是加信息,是加一条**不依赖 mocap 的本体姿态通路**。
2. **噪声挂载点**:这正是 §22.2 C 表点名的那个"洞"——"合同里没有 base_ang_vel/projected_gravity,真机 IMU 噪声将来无处对应"。今天 A225 的这两个量藏在一个官方判定"无法定义物理有效噪声"的混装块里,等于**真机 IMU 噪声在训练侧没有任何可挂的钩子**。
3. **参考栈一致性**:四库 + 厂商 + 我们自己 8 个历史合同全都保留这两条骨盆系通道;C++ 部署侧 `build_obs_175/177/179/180` 每一个都在逐段拷 `state.base_ang_vel_b` 和 `projected_gravity_body(state.base_quat_w)`(pp_obs_builder.hpp)。

**(d) 真机给不给得出:给,而且已经在跑。** RobotState 携带骨盆 IMU 的 `imu_quat_wxyz` / `imu_gyro` / `imu_accel`(agi/a3_deploy_example/README_robot_io_backend.md:121-128、207-208)。硬证据不在那张已被标注为历史的 180-D 勾选表,而在 C++:pp_policy.hpp:1124-1125 直接 `st.base_quat_w = state.imu_quat_wxyz`(注释"real pelvis IMU orientation")、`st.base_ang_vel_b = state.imu_gyro`(注释"real pelvis gyro (body frame)"),`build_obs_175` 就吃这两个字段。**一处必须记的实况**:pp_policy.hpp 下游约 :1135/:1159 会按定位模式用 yaw 对齐或 oracle/mocap 骨盆姿态**覆盖** `base_quat_w`,所以"IMU 直供"是基线而非唯一路径。另注:部署侧**根本没有** build_obs_181/194/225(整个 deploy 树对 181/194/225 零命中),225 家族今天是纯 sim。

**(e) 加回来的完整机械成本清单(全部是"新增合同",不是原地改——225 家族被显式冻结)**
1. `actor_observation_contract.py`:新 `ActorObservationContract`(新 name、total_dim=228 或 231),在 `CONTRACTS` 字典按 `.name` 和 `.obs_mode` 双注册,并加进 `infer_actor_observation_contract()` 的元组(约 :1093-1113),否则按形状自动识别会静默失败。
2. `hope_env_cfg.py`:新 ObsGroup 子类,插 `base_ang_vel` / `projected_gravity` 两个 ObsTerm(mdp 函数已存在,零新数学),外加新 EnvCfg + 新 cfg/task yaml(不能改钉死的 `HOPEPingPongActionBallA225VendorV2N1Learnability.yaml`)。
3. **critic ABI:内容不用动**(318-D critic 经 PrivilegedCfg 已带 `base_lin_vel3`+`base_ang_vel3`,tracking_env_cfg.py:140-155),但 schema-3 要求**新注册一个 critic 身份串**。**口径纠正**:A225/C225 的 critic 不是 hope_env_cfg.py 的 `Stage1CriticCfg`,而是各自独立注册的 `action_ball_a225_critic_v1` / `action_ball_c225_critic_v1`(布局定义在 action_ball_225_trainability.py / action_ball_c225_trainability.py),doc :459-462 明令三者虽同为 318 不得互相复用谱系。
4. **checkpoint:必然不兼容**,现役 in_features=225,228/231 是硬形状失配,无 warm start;ONNX 元数据钉合同名+宽度,等于全新导出件。
5. `training_contract.py`(schema-3):`_STAGE1_..._ACTOR_OBS_CONTRACT` 一族冻结常量按合同名钉有序项名与总维(约 :155-194、:2351-2450、:5659-5845),不加新常量+校验分支,新合同一律 fail closed。**这就是问的那个"钉维度的校验器"。**
6. MuJoCo-native fixed-tape parity:**任何** 225-D 合同今天都没有(mujoco_native 全树不引用 `actual_base_now_world`),加不加这两列都是从零开始。
7. C++ 部署 builder:今天没有 `build_obs_225`,也是从零;但加回这两列反而让未来那个函数的这 6 个标量**退化成 175/177/179/180 已验证的成熟写法**,难点仍在 `actual_base_now_world`/`teacher_base_now_world` 需要的 mocap+因果估计器。
8. 下游把 225 当魔数的消费者要跟或另立:tests/test_action_ball_a225_trainability.py、scripts/materialize_action_ball_a225_lineage.py、上述 yaml。

**交叉问题裁决(DR 噪声同步开不开)——分成两件事,答案相反:**
- **观测噪声 ±0.2 / ±0.05:随合同一起开,day-1,不走 §22 的排队。** §22.1 三闸对它全过:闸 1 支撑集——骨盆 IMU 是**本体感受**,不改转移不改"哪些动作能击中球"(§22.1 原文点名了这个不对称:本体感受噪声可 day-1,任务通道噪声不行);闸 2 终止——零均值观测噪声不造终止事件,Δp_term 预期 0;闸 3 cadence——每步级**只允许**零均值观测噪声,正好落在允许区。§22.2 A 行现役 `joint_pos ±0.01` / `joint_vel ±0.5` 已判"D1 开满",外部 9/9 库 day-1 固定,厂商连 play 都保留 obs 噪声。而 doc :466-471 那句"旧噪声旋钮定义不出物理有效噪声"针对的是**混装 15-D 块**,不针对重新独立出来的骨盆系单列——单列的物理含义和 175/177/179/180/194 完全一致,旧旋钮原样适用。**更关键的是**:加通道却不加噪声,等于把 §22 记的那个洞从"无处对应"改写成"有处对应但故意空着",sim2real 缺口反而更隐蔽。
- **厂商 `scale=0.25` + `history=8`:不开,维持 §11.3 低优先第 8 条。** 这是另一根轴(§22.2 C 第三行),破坏 110/177/194-D 部署契约,必须独立臂 + 同批改部署侧,且只有与延迟臂捆绑才有意义。
- **唯一让步**:若 Franco 要与现役 A225 臂做严格单变量对比,则第一发新合同臂噪声关闭、第二发开——代价是多一格。默认建议**不这么做**,因为噪声正是本次修改的目的之一;真要控变量,变量应该控在"有没有这两列",而不是"这两列有没有噪声"。
- **不动的**:两个 15-D 块继续保持零噪声,按 doc :469-471 的要求,底座位姿/twist 噪声必须按 mocap/IMU 分量各自定义,那是另一个独立立项。

### 23.3 行动项

**[P0][tape / 发射器交代]** 零代码即办:在 launch_action_ball_a225_four_arm_diagnostic.py:1382-1385 的 claim 摘要里用人话写明『target_source=immutable_tape + diagnostic_unauthorized=true 这一对 argv 等于把 32-arm 课程整跑停权』,并写死一个到期条件(M1 或 strike_opportunity_count 首次非零)。今天读 argv 看不出这层后果,已经踩了『发射工序:依赖核对单 + WARN 必进摘要』的教训。风险:无,纯文档;不做的风险是下一个人以为 tape 只是省算力。

**[P0][tape / 认知纠偏]** 把『immutable_tape = 目标信息消融夹具(5 recipe 里 4 个只能走它,hope_commands.py:205-209),不是缓存』这条写进 docs/research/dr_reward_external_diligence_20260731.md §22.2 F 行与本次章节,避免后续再把它当成本优化手段引用。风险:无。

**[P0][求解器成本 / 口径修正]** 修正跨文档归因:dr_reward §10.2 把 reset 的 60-75% 记为逐 env Python 仪式是估算,已被同日 Pod 分段 profiler(design_audit §8.10)推翻——profiled reset 40.732 s 里 solver_solve_many 占 82.1%,Racket install 只有 0.202 s。以 §8.10 为准。风险:不改会导致提速工单排错顺序(去磨 Python 记账而不是求解器)。

**[P1][求解器成本 / 真提速]** 跑 design_audit §8.11 已排产的固定题带预注册 cq_n_iters=4/6/8/12,按残差 / admit mask+reason / racket task / replay 稳定性选最小充分迭代数;若 8 次过门,估省约 1.1 s/update(估算,未验收)。这条与 tape 完全解耦,是『不改任何语义就拿走一半省算力动机』的路。风险:降迭代可能动数值门,必须同 seed 4096x5 三组 JSON parity + finite checkpoint 验收。

**[P1][obs / 新合同]** 新建 228-D(或 231-D)合同:actor_observation_contract.py 加新 ActorObservationContract 并在 CONTRACTS 双注册 + 进 infer_actor_observation_contract() 元组(~:1093-1113);hope_env_cfg.py 新 ObsGroup 插 base_ang_vel / projected_gravity 两个 ObsTerm;新 EnvCfg + 新 cfg/task yaml。绝不原地改 A225(该家族被显式冻结)。风险:漏注册 infer 元组会让形状自动识别静默失败,是最容易踩的一处。

**[P1][obs / 校验器与身份]** training_contract.py schema-3 加新的冻结常量与校验分支(~:155-194、:2351-2450、:5659-5845),并为新合同注册独立的 318-D critic 身份(仿 action_ball_a225_critic_v1 / action_ball_c225_critic_v1,分别定义在 action_ball_225_trainability.py / action_ball_c225_trainability.py);critic 项表内容不变。风险:不加分支则新合同 fail closed 跑不起来;复用 A225 critic 身份会污染 normalizer/checkpoint 谱系(doc :459-462 明令禁止)。

**[P1][obs / DR 噪声]** 新合同的这两列 day-1 直接开厂商级观测噪声 base_ang_vel ±0.2、projected_gravity ±0.05(§22.1 三闸全过:本体感受、零均值、不造终止),不进 §22 的排队。厂商 obs scale=0.25 + history=8 不开(破坏 110/177/194-D 部署契约,维持 §11.3 第 8 条)。两个 15-D 块继续零噪声。风险:与现役 A225 臂不再严格单变量;若 Franco 坚持控变量,变量应控在『有没有这两列』而非『这两列有没有噪声』。

**[P2][tape / 真缓存]** 新增第三个 target source 值 banded_question_bank(不动 immutable_tape 语义;枚举在 hope_commands.py:121):reset 按当前 domain_levels 索引预生成题带行,题带按 32-arm 的 5 档(action_ball_curriculum.py:61-62)x 臂键离线分块生成。失效机制不用新写——课程每次 reset 已经通过 _action_ball_claim_domain(hope_commands.py:6994-7042)报出 domain_levels,直接做 key,升档=换块=天然重解;_assert_birth_matches_question(action_ball_fixed_question_tape.py:838-870)的硬断言按行保留。该 source 不绑定 diagnostic_unauthorized,课程保留权威。风险:题带规模 = 5 档 x 32 臂 x 每档行数,离线生成 + sha 钉 + 谱系管理是它比 tape 贵的唯一地方;必须排在 §13 R1(课程可逆化)与 154 行样本量修复之后,否则解冻课程只会拿到一串永久锁。

**[P2][tape / 前向意图落地]** 未提交的 hope_commands.py:6120-6134 已经把正确设计写成了 payload 字符串(final_curriculum_question_source=pregenerated_or_cached_band_question_bank、final_curriculum_reset_operation=index_precomputed_question_row),但零代码读它。要么随上条一起实现,要么明确标注为 aspirational 注解,别让它以后被当成已实现的机制引用。风险:今天它就是一条会被误读为『已经有缓存了』的字符串。

**[P3][obs / 部署与 parity 缺口(台账,不排产)]** 记账两条今天无论加不加这两列都不存在的东西:(1) 任何 225 家族合同都没有 MuJoCo-native fixed-tape parity(mujoco_native 全树不引用 actual_base_now_world);(2) 部署侧没有 build_obs_225(整个 deploy 树对 181/194/225 零命中)。同时记 pp_policy.hpp ~:1135/:1159 会按定位模式用 yaw 对齐或 oracle 骨盆姿态覆盖 base_quat_w,『IMU 直供 projected_gravity』是基线而非唯一路径。风险:不记会在未来上真机时被当成新发现的阻断。


### 23.4 补充

## 二十三、两项裁决:immutable_tape 的真实身份,与 base_ang_vel / projected_gravity 的回归(2026-08-04)

### 23.1 一句话

- **tape**:配错了,不是参数填错,是选错工具。要的是"缓存",拿到的是"永久冻结 + 课程停权"。
- **obs**:该加,Franco 判断成立;但支持它的理由必须换掉——不是"缺信息"(信息其实冗余),而是"缺一条不依赖 mocap 的本体姿态通路"和"缺一个能挂真机 IMU 噪声的钩子"。

---

### 23.2 immutable_tape 不是缓存,是消融夹具

**本意 vs 现状**

| | Franco 的本意 | 仓库现状 |
|---|---|---|
| 触发 | 档位不变就复用 | 与档位无关,永远一行 |
| 失效 | 档位升了就重解 | 档位**不可能**升;真升了会崩(硬断言) |
| 多样性 | 不变(只是省重复计算) | 塌成 1 题 |
| curriculum | 照常运转 | 整跑停权 |

**为什么它天生不是缓存**:`action_ball_target_source` 只有两个合法值(`online_solver` / `immutable_tape`,hope_commands.py:121),而合同校验器把 `online_solver` 限死只能配 `current_lm` + 全掩码(hope_commands.py:205-209)——5 个 target recipe 里另外 4 个(`analytic_full` / `analytic_no_velocity` / `teacher_pos_face_no_velocity` / `outcome_dense_only`)**根本没有 online 路径,只能走 tape**。它的设计岗位就是目标信息消融的载具。模块自述也是实验设计口径,不是缓存口径。

**32-arm curriculum 在 tape 下起不起作用:完全不起,而且是双重的**

1. **构造期停权**:`immutable_tape` 强制 `diagnostic_unauthorized=true`(hope_commands.py:5349-5357;并强制 n_actions=1),该开关让 ActionBallCurriculum 拿到 `evaluator_authority=None` / `drain_reset_authority=None`(hope_commands.py:5307-5315,注释自陈"A diagnostic authority is deliberately NOT bound")。
2. **运行期短路(这才是主机制)**:真正的把门人是 runner 侧的 `action_ball_frozen_evaluation_boundary`(hope_commands.py:10799-10823)——diagnostic_unauthorized 为真时,phase='poll' 直接返回 `{"diagnostic_unauthorized": True}`,**根本不调用** `_action_ball_eval_consume_ready`(:10845),而后者是 `curriculum.observe_scheduler`(action_ball_curriculum.py:3625)/ `stage_selected`(:3974)的唯一调用方。所以课程不是"被调用时报错",是**从头到尾没人调用**;曲线永远停在初始 `phase='center'`、32 臂全 0.0 档(`LEVELS=(0,.25,.5,.75,1.0)`,action_ball_curriculum.py:61-62)。课程内部的 `evaluator_authority is None` fail-loud 只是纵深防御,真跑里从不触发。
3. **tape 也不读课程**:`FixedQuestionTapeSolver._assert_birth_matches_question`(action_ball_fixed_question_tape.py:838-870)对 domain_levels 做**硬断言**——课程若真升了档,进程崩,不是换题。tape 不是"跟着档位走",是"逼课程每次 reset 复读同一常量"。

**冻的到底是什么**:不只是来球。实测 artifact `row_count=1`、`selection="constant_row_zero"`、`online_lm_calls=0`、`physical_rng_draws=0`(action_ball_fixed_question_tape.py:582-584;活体件 `configs/action_ball_n1_measured_20260803/fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/immutable_n1_tape.v1.22052606032f.json`)。`base_spawn` / `ball_contact` / `incoming_velocity` / `incoming_spin=[0,0,0]` / `landing_aim` 全是字面浮点数,5 个 recipe 的期望接触 p-v-face 元组也各自预算死。**这是一道题,不是一个分布。**

**有没有真缓存:一行都没有。** 搜遍 continuous_questions.py / action_ball_sampling.py / strike_spec_torch.py,唯一的 cache 是审计回执的 sha/digest 记账(action_ball_sampling.py:4579-4937)。有意思的是**正确设计已经以字符串形式躺在仓库里但零实现**:未提交的 hope_commands.py:6120-6134 往诊断 payload 塞了 `final_curriculum_question_source: "pregenerated_or_cached_band_question_bank"`、`final_curriculum_reset_operation: "index_precomputed_question_row"`、`does_not_freeze_final_curriculum_to_one_question: True`——没有任何代码读它,**不能当成"已经有缓存了"引用**。

---

### 23.3 省算力这个动机今天还成不成立:成立,但只剩一半,且要改一处归因

**跨文档口径修正(以本节为准)**:§10.2 把 23.48 s/update 里的 13.5-17.8 s(60-75%)记成"reset 仪式 = 逐 env Python 记账残差",那是**估算**。同日晚些的 Pod 分段 profiler(design_audit_and_speedup_20260729.md §8.10)把它拆开了:

- profiled reset = 40.732 s,其中 `pool_request_many` = 34.724 s、**`solver_solve_many` = 33.432 s = reset 的 82.1%**、= 五轮总 collection(51.654 s)的 **64.7%**;
- 对照:Motion true reset 4.979 s、provider 4.247 s、broker reserve 4.624 s、**Racket install 仅 0.202 s**(静态审计里"先做 install packet"的排序被实测推翻)。

**reset 仪式的大头本来就是求解器本身,不是 Python 记账。**

**优化后的今天(§8.11,host-only solver result,同 seed 4096×5)**:五轮 solver = 16.367 s / collection = 32.924 s ≈ **49.7%**;均值 **6.700 s/update**(≈14.67k env-steps/s),**reset-free update ≈ 2.7 s**。即 reset 增量 ≈ 4.0 s/update 里 solver 约 3.27 s ≈ **82%**。§8.11 原话:reset-heavy update 的主差额仍在 fixed-direction solver/LM。

→ **动机成立,而且求解器仍是第一杠杆**;这恰恰说明不该用"永久冻结"去换它。另有一条不改语义的路已排产:固定题带预注册 `cq_n_iters=4/6/8/12`,若 8 次过数值门估省 ≈1.1 s/update(**估算,未验收**)。

---

### 23.4 最小修复(三步,按"不改现役字节"顺序)

1. **零代码,即办**:承认 immutable_tape 是 N=1 消融夹具,把这句人话 + 显式到期条件(M1 或 strike_opportunity_count 首次非零)写进 `launch_action_ball_a225_four_arm_diagnostic.py` 的 claim 摘要。现役 argv(:1382-1385)一次性同时设了 `target_source=immutable_tape` 与 `diagnostic_unauthorized=true`,今天读 argv 看不出这等于把课程停权——这正是"发射工序:依赖核对单 + WARN 必进摘要"要防的。
2. **先拿掉一半动机**:跑 §8.11 的 `cq_n_iters` 预注册。与 tape 完全解耦,零语义风险,验收沿用同 seed 4096×5 三组 JSON parity + finite checkpoint。
3. **真缓存(本意的落点)**:新增**第三个** source 值 `banded_question_bank`(不动 immutable_tape 现语义),reset 按当前 domain_levels **索引预生成表行**,表按 32-arm 的 5 档 × 臂键离线分块生成。**失效机制不用新写**——课程每次 reset 已通过 `_action_ball_claim_domain`(hope_commands.py:6994-7042)报出 domain_levels,直接做 key,升档=换块=天然重解;`_assert_birth_matches_question` 的硬断言按行保留即可。该 source **不绑定** diagnostic_unauthorized,课程保留权威、可升档、可导出。**依赖**:必须排在 §13 R1(单臂可逆化)与 154 行样本量修复之后,否则解冻课程只会拿到一串永久锁。**成本**:题带规模 = 5 档 × 32 臂 × 每档行数,离线生成 + sha 钉 + 谱系管理,这是它比 tape 贵的唯一地方。

---

### 23.5 base_ang_vel / projected_gravity:有无表

actor 侧。`base_lin_vel` 全家族缺席(`HOPEPolicyCfg` 一律 `base_lin_vel = None`,hope_env_cfg.py:928),是 HOPE 家族级的 critic-only。

| 合同(总维) | base_ang_vel | projected_gravity | motion_anchor_pos_b | motion_anchor_ori_b |
|---|---|---|---|---|
| hitter_pure(110) | 有 | 有 | 无 | 无(用 base_forward_xy) |
| stage1_natural_clip_site_v1(170) | 有 | 有 | 有 | 有 |
| deploy_parity(175) | 有 | 有 | 无 | 有 |
| hitter_footwork(177) | 有 | 有 | 无 | 有 |
| deploy_parity_face179(179) | 有 | 有 | 无 | 有 |
| full(180) | 有 | 有 | 有 | 有 |
| deploy_parity_station181(181) | 有 | 有 | 无 | 有 |
| L194 | 有 `[68:71]` | 有 `[164:167]` | 无 | 有 `[62:68]` |
| **paddle_world_v2 / A225 / C225(225)** | **无独立列** | **无独立列** | 无 | 无 |

---

### 23.6 是刻意设计,不是抄前缀漏掉

`git log --all -p` 对这两个词名的删除行(`-` 开头)在 `actor_observation_contract.py` 全历史**零命中**:没有任何合同曾经有过再被删掉。225 家族是**从零写的新合同**。

- `7e5907a6`(08-02 06:43)建 170-D 时两条都在;
- `d361d1bd`(08-02 17:56,同日)建 225-D V2 时**同一个 commit** 写下了理由,见 `docs/interfaces/policy_observation_action.md:428`(`projected_gravity` is omitted because actual base orientation already determines it)与 `:466-471`(旧的 `base_ang_vel ±0.2` / `projected_gravity ±0.05` 旋钮 "do not define physically valid noise",因为 15-D 是位置/姿态/线速度/角速度混装块,底座位姿噪声必须按 mocap/IMU 分量各自定义);
- `10e3ab14`(08-03)建 A225/C225 时直接切 `STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[:10]` 复用(actor_observation_contract.py:286-293、320-324),所以它们连"重新加"的机会都没有——但源头是设计决定。

---

### 23.7 两个 15-D 块装了什么(会不会重复:信息上会)

两块由同一个 `_stage1_pack_base_state_world`(hope_observations.py:310-349)按固定顺序拼接并断言有限:

| 列 | 内容 |
|---|---|
| `[0:3]` | 世界系底座位置(Isaac per-env 帧换算到 HOPE 场地帧) |
| `[3:9]` | 世界系旋转矩阵前两列(连续 6-D 表示) |
| `[9:12]` | 世界系线速度 |
| `[12:15]` | **世界系角速度** |

`actual_base_now_world` 取 `root_lin_vel_w` / `root_ang_vel_w`;`teacher_base_now_world` 取对齐后参考 clip 的同构四段(速度经 `_stage1_reference_vector_in_aligned_world` 转到同一世界基)。

→ **角速度有**(世界系,不是骨盆系);**重力方向无显式列但可反解**(`R^T·[0,0,-1]`,由 `[3:9]` Gram-Schmidt 补全)。**所以"补信息"这个理由不成立。**

**真正支持加回来的三条**:

1. **来源不同**。15-D 块的底座位姿真机侧靠 OptiTrack mocap,而团队自己的 "Deploy-Available Signal Set"(policy_observation_action.md:1100-1120)写明 mocap **只在 PLAY 期间可得**;`base_ang_vel` / `projected_gravity` 走骨盆 IMU,**永远可得**。加回来是加一条不依赖 mocap 的本体姿态通路。
2. **噪声挂载点**。这正是 §22.2 C 表点名的洞:"合同里没有 base_ang_vel/projected_gravity,真机 IMU 噪声将来无处对应"。今天这两个量藏在一个官方判定"定义不出物理有效噪声"的混装块里 = IMU 噪声在训练侧**没有任何钩子**。
3. **参考栈一致性**。四库 + 厂商 + 我们 8 个历史合同全保留这两条骨盆系通道;C++ 部署侧 `build_obs_175/177/179/180` 每个都在逐段拷 `state.base_ang_vel_b` 与 `projected_gravity_body(state.base_quat_w)`。

---

### 23.8 真机给不给得出:给,而且已经在跑

- `RobotState` 携带骨盆 IMU 的 `imu_quat_wxyz` / `imu_gyro` / `imu_accel`(`agi/a3_deploy_example/README_robot_io_backend.md:121-128、207-208)。
- **硬证据在 C++,不在那张 180-D 勾选表**(那张表所在的 PINGPONG_DEPLOY_ALIGNMENT.md 第 3 节已被其自身第 0 节标注为历史;且现役是 175-D,偏移应为 `base_ang_vel[68:71]` / `projected_gravity[164:167]`,与 180-D 表差 3)。真正该引的是 `pp_policy.hpp:1124-1125`:`st.base_quat_w = state.imu_quat_wxyz`(注释 "real pelvis IMU orientation")、`st.base_ang_vel_b = state.imu_gyro`(注释 "real pelvis gyro (body frame)"),`build_obs_175` 直接吃这两个字段。
- **实况提醒**:`pp_policy.hpp` 下游约 `:1135` / `:1159` 会按定位模式用 yaw 对齐或 oracle/mocap 骨盆姿态**覆盖** `base_quat_w`,所以"IMU 直供"是基线路径而非唯一路径。
- **缺口**:部署侧只有 `build_obs_110/175/177/179/180`,**没有 181/194/225**(整个 deploy 树对这三个数字零命中)。225 家族今天是纯 sim。

---

### 23.9 加回来的机械成本清单

225 家族被显式冻结("must never be silently reinterpreted by a checkpoint or rollout receipt"),所以这是**新增合同**,不是原地改。

1. `actor_observation_contract.py`:新 `ActorObservationContract`(新 name、`total_dim`=228 或 231),在 `CONTRACTS` 按 `.name` 与 `.obs_mode` **双注册**,并加进 `infer_actor_observation_contract()` 的元组(约 :1093-1113)——**漏这一步会让按形状自动识别静默失败**。
2. `hope_env_cfg.py`:新 ObsGroup 子类插两个 ObsTerm(mdp 函数已存在,零新数学)+ 新 EnvCfg + 新 cfg/task yaml(不能改钉死的 `HOPEPingPongActionBallA225VendorV2N1Learnability.yaml`)。
3. **critic ABI:内容不动**(318-D critic 经 `PrivilegedCfg` 已带 `base_lin_vel3` + `base_ang_vel3`,tracking_env_cfg.py:140-155),但 schema-3 要求**新注册一个 critic 身份串**。**口径纠正**:A225/C225 的 critic 不是 `Stage1CriticCfg`,而是各自独立注册的 `action_ball_a225_critic_v1` / `action_ball_c225_critic_v1`(布局在 `action_ball_225_trainability.py` / `action_ball_c225_trainability.py`);doc `:459-462` 明令三者虽同为 318 不得互相复用 checkpoint 谱系。
4. **checkpoint 必然不兼容**:现役 `in_features=225`,228/231 是硬形状失配,无 warm start;ONNX 元数据钉合同名 + 宽度,等于全新导出件。
5. `training_contract.py`(schema-3):`_STAGE1_..._ACTOR_OBS_CONTRACT` 一族冻结常量按合同名钉有序项名与总维(约 :155-194、:2351-2450、:5659-5845)。不加新常量 + 校验分支,新合同一律 fail closed。**这就是那个"钉维度的校验器"。**
6. **MuJoCo-native fixed-tape parity**:任何 225-D 合同今天都没有(mujoco_native 全树不引用 `actual_base_now_world`),加不加这两列都是从零。
7. **C++ 部署 builder**:今天没有 `build_obs_225`,也是从零;但加回这两列反而让未来那个函数的这 6 个标量退化成 175/177/179/180 已验证的成熟写法,难点仍在 `actual_base_now_world` / `teacher_base_now_world` 需要的 mocap + 因果估计器。
8. **把 225 当魔数的下游**要跟或另立:`tests/test_action_ball_a225_trainability.py`、`scripts/materialize_action_ball_a225_lineage.py`、上述 yaml。

---

### 23.10 交叉裁决:DR 侧同步开噪声吗(应 §22 之问)

**分成两件事,答案相反。**

**(一)观测噪声 `base_ang_vel ±0.2` / `projected_gravity ±0.05` —— 随合同一起开,day-1,不进 §22 排队。**

对 §22.1 三闸逐条:

- **闸 1(支撑集)过**:骨盆 IMU 是**本体感受**通道,不改转移、不改"哪些动作能击中球"。§22.1 原文已点名这个不对称——本体感受噪声可 day-1,任务通道(desired-contact / racket-heading / time_to_contact)噪声不行。这两条属前者。
- **闸 2(终止)过**:零均值观测噪声不改动力学、不造终止事件,Δp_term 预期为 0,远在 0.5 pp 阈内。
- **闸 3(cadence)过**:每步级**只允许**零均值观测噪声,这正落在允许区。

旁证:§22.2 A 行已判现役 `joint_pos ±0.01` / `joint_vel ±0.5` 为 "D1 开满";外部 9/9 库 day-1 固定;厂商连 play 都保留 obs 噪声。

至于 doc `:466-471` 那句"旧旋钮定义不出物理有效噪声"——它针对的是**混装 15-D 块**,不针对重新独立出来的骨盆系单列:单列的物理含义、帧、单位与 175/177/179/180/194 完全一致,旧旋钮原样适用。

**更关键的一句**:加了通道却不给噪声,等于把 §22 记的那个洞从"无处对应"改写成"有处对应但故意空着",sim2real 缺口反而更隐蔽。

**(二)厂商 `scale=0.25` + `history=8` —— 不开。** 这是另一根轴(§22.2 C 第三行):破坏 110/177/194-D 部署契约,必须独立臂 + 同批改部署侧,且只有与延迟臂捆绑才有意义。维持 §11.3 低优先第 8 条。

**(三)唯一让步**:若要与现役 A225 臂做严格单变量对比,可第一发新合同臂噪声关闭、第二发开,代价多一格。**默认不建议**——噪声本身就是这次修改的目的之一;真要控变量,变量应控在"有没有这两列",不是"这两列有没有噪声"。

**(四)不动的**:两个 15-D 块继续零噪声,按 doc `:469-471`,底座位姿/twist 噪声必须按 mocap / IMU 分量各自定义,那是独立立项。

---

### 23.11 本节顺带修正的既有说法(以本节为准)

1. §10.2 的"reset 仪式 = 60-75% Python 记账"是估算,已被 §8.10 分段 profiler 细化:reset 的 82.1% 就是求解器本身。
2. §22.2 F 行把 immutable_tape 记作"旁路 / DR 轴一行",本节补足其真实身份:**它同时是课程停权开关**,不是中性的目标源选择。
3. 把 A225 的 critic 称为 `Stage1CriticCfg` 是口径错误;A225/C225 各有独立注册的 318-D critic 身份。
4. 引用真机 IMU 可用性时,不要引 `PINGPONG_DEPLOY_ALIGNMENT.md` 第 3 节的 180-D 勾选表(已被其自身第 0 节标为历史,且偏移与现役 175-D 差 3),改引 `pp_policy.hpp:1124-1125`。
