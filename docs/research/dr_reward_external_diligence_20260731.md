# 外部尽调:BeyondMimic / mjlab / unitree_rl_lab 的 DR 与 reward 设置对照(2026-07-31)

**触发**:有同事建议"训练时给 policy 加的基础泛化不够(比如随机 kp/kd)"。本文对照三个外部库
(BeyondMimic 上游 `HybridRobotics/whole_body_tracking`、`mujocolab/mjlab`、`unitreerobotics/unitree_rl_lab`,
均 2026-07-31 浅克隆最新 main)逐项核实:他们随机化什么、罚什么、用什么数值;我们已有什么、缺什么、
值得借什么。

**方法与可信度**:5 个抽取 agent(我方 reward / 我方 DR / 三个外部库各一)+ 5 个对抗核查 agent
(逐条 file:line 复核、反向搜漏项)+ 主线人工抽查(kp/kd yaml 链、joint_acc 数值、push 区间、v2 展开
机制)。核查共揪出 24 处修正与 37 处漏项,下文均为**修正后**数值。逐库原始 JSON 存
scratchpad(会话级,不入库);本文是归档结论。

**阅读规则（2026-07-31 采纳后更新）**：§1–12 保留外部尽调与采纳前代码快照，
其中“我们现役/当前”若未另标日期，指旧 stable-ready N1 或采纳前通用 task。
§13 是后续追加的 curriculum 升级机制专项尽调，它的 R1–R9 是设计证据
与候选修复，不是已采用的运行态；
执行顺序与阻塞只看
[分阶段准备账本](../experiments/2026-07/EXP-ACTION-BALL-PHASED-READINESS-20260730.md#0-当前执行看板本文唯一活跃-todo)，
本文不维护第二份 TODO。

---

## 一、结论先行(人话)

1. **"我们没随机 kp/kd"这个前提是错的，但现役 N1 是例外。**
   采纳前的 `randomize_pd_gains`(每次 reset 重抽、log_uniform、scale、31 关节全覆盖、kp/kd 独立抽样)
   在 ActionBall/Hitter 通用 task 谱系上以 ±15% 开启(经 `defaults:` 链继承 Hitter 的
   `pd_gain_range: [0.85, 1.15]`,[HOPEPingPongHitter.yaml:527](../../hope_training/whole_body_tracking/cfg/task/HOPEPingPongHitter.yaml));
   **现役 N1 reward-screen launcher 强制 `stable_ready_plant=true`，会把 PD/mass/CoM DR 全部摘掉**，
   因此不能把通用 task 默认误写成 N1 运行真值。
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
3. **采纳前的真差距在三条现役配方全关/全零的轴；其中 push 与
   actuator delay 的 host 实现已在§13收口:**
   - **外部推撞**:三家在模仿任务上全都开(BeyondMimic/mjlab/unitree-mimic 同配方:每 1–3 s、
     线速 ±0.5/±0.5/±0.2 m/s + 角速 ±0.52/±0.52/±0.78 rad/s)。我们这边**不是没推过,
     是推的裁定断在了传导上**(修正说明见 §3.1,时间线:Wave-P 14 臂 07-20/21 真推过，但未达预注册终点;
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
| kp/kd 随机化 | 采纳前通用 task 为 ±15% reset 级；旧 stable-ready N1 全关；当前实现见 §13 | 三家全无(mjlab 有原语未接线) |
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

1. **Wave-P(EXP-P1-PUSH-ROBUSTNESS-20260721)实际发射过 14 条推撞臂**:
   `{W,V} × {p02,p035,p05,yaw,p08,f035,f08}`；`{W,V} × {ang,fast}` 四臂从未发射。
   14 臂都保有 `model_8700`，但最高只到 9200–13900，**无一达到预注册 `model_16700`**。
   只有 `w_p02` 留有逐臂 full-scene probe，其余 13 臂跳过 admission；续训还跨越多个源码
   commit，且既有 5 份 judge 是 `K=371` / `evaluation_contract_exact=false` 的 diagnostic escape。
   因此这波正式状态为 **superseded**，closure 为 **incomplete**：保留机制和方向性历史证据，
   `no dose winner`，不续训、不补旧卷，四条未发臂取消。
2. **07-26 v2 冻结审计(§0.10,Franco 触发,原话"随机推没开,是不是还漏了什么")已把 push
   裁进 v2.3 模板**:`++task.push.enable=true ++task.push.vel_xy_mps=0.35 interval [10,30]s`
   + `++task.force_push.enable=true force_n=68 duration 0.3s`(双事件近似合并,合成期望
   ≈每 ~10 s 一次),并注明"pod2 新动作臂发射前必须采纳"。
3. **历史 N1 reward-screen 发射器没有继承 v2.3 的 push 键**:
   `launch_n1_reward_screen_diagnostic.py` 组装的
   argv 实测无任何 `task.push.*`/`task.force_push.*`。若 N1 作为 reward 筛查波刻意保持与
   在跑 v2 臂单变量可比(§0.10 原文"全队一致,不影响臂间比较"),这是合理的;但"下一代
   基线必须带 push"的裁定当时没有机械保证。现在 vendor task leaf 与独立
   diagnostic launcher 已把它变为可验证的任务身份，见 §13。

- **外部证据的作用**:三家外部库独立佐证 v2.3 裁定的方向——模仿任务全部开推
  (BeyondMimic `tracking_env_cfg.py:190-195`、mjlab `tasks/tracking/tracking_env_cfg.py:165-171`、
  unitree `tasks/mimic/.../tracking_env_cfg.py:210-215`,同配方 interval (1,3) s,线速
  x/y ±0.5、z ±0.2 m/s,角速 ±0.52/±0.52/±0.78 rad/s;unitree 步行臂温和版:5 s 一次、仅 xy ±0.5)。
  外部 1–3 s 节奏**不要抄**(会砸进击球窗);v2.3 的 [10,30] s 与 Wave-P 的 [5,15] s 都比
  外部保守,方向正确。
- **落点(两件事,都不是"要不要推"而是"把已裁定的推落实")**:
  (a) 按上述 `status=superseded, closure=incomplete` 结论统一 EXP-P1/NOW/PROGRESS 的 6/12/14/0 臂矛盾口径;
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

### 高:执行器延迟(数值与语义采用智元同底盘设定)
- **人话**:四家都没真开延迟,但我们要上真机,这是最值钱的没人做的轴。mjlab 的参数化最好:
  除了 lag 区间,还有 `delay_hold_prob`(丢包保持上一条命令)——那才像真丢包,不是干净恒延。
- **证据**:mjlab `actuator/actuator.py:67-112,151-224`(delay_min/max_lag、hold_prob、update_period、
  per_env_phase,全默认 0);我们 `robots/actuator.py` 的 DelayedImplicitActuator 是死代码,
  A3 五组执行器全是裸 ImplicitActuatorCfg;A1 球目标延迟/噪声旋钮七个 yaml 全零
  (标定过的场馆噪声 σ_white=0.0019 m、AR(1) σ=0.0052 m 就躺在注释里没启用)。
- **落点**:采用智元已运行的 **[0,2] 控制步、每 episode 抽一次且集内固定**。
  我们现有 `DelayedImplicitActuatorCfg` 的单位是 physics step，直接填 `[0,2]` 只是 0–10 ms，
  不等于智元的 0–40 ms；必须在 policy/control-step 命令边界实现，31 关节共享同一
  episode lag，并纳入 partial-reset、exact-resume 和 receipt 合同。
- **风险/门**:这是训练物理配方，不需要学习 A/B 来决定是否采用；但启用前必须过
  0-step 等价、2-step impulse、partial reset、exact-resume 和 nominal-hold 机械门。

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
| kp/kd 随机化 | **采纳前**通用 task ±15% reset 级；旧 stable-ready N1 全关；现行见 §13 | NONE | 原语未用 | NONE |
| 力矩上限/motor strength | NONE(静态常数) | NONE | 原语未用 | NONE(29dof 实值 88/139/25/5 Nm) |
| armature/关节摩擦 | NONE(手抄 MJCF 常数) | NONE | 原语未用 | 电机摩擦模型是死代码(仅 Go2HV 实例化且 Fs=Fd=0) |
| 摩擦/restitution | 全身 (0.3,1.6)/(0.3,1.2)/(0,0.5) | 同我们 | 仅足底切向 abs (0.3,1.2);无 restitution 原语 | 人形 (0.3,1.0)² + restitution 钉 0;mimic 同 BeyondMimic |
| 连杆质量 | 全身 scale ±15% | NONE | 原语未用 | 仅躯干 add (−1,+3) kg;mimic 无 |
| CoM 偏移 | 躯干 x±0.025/y±0.05/z±0.05 | 同 | 同(velocity 臂更紧) | 仅 mimic 有,同值 |
| 关节零位误差 | ±0.01 rad 双写(obs+动作偏置) | 同 | encoder_bias 仅偏 actor 观测(±0.01/±0.015) | 仅 mimic,双写 |
| 外部推撞 | 旧 stable-ready N1 无；Wave-P 14 臂未达终点；新 vendor profile 已开六轴 `5–15 s` 无 gate（§13） | **开**:(1,3) s 6-DOF ±0.5/±0.2/±0.52/±0.78 | **开**:同左(play 时摘除) | 步行:5 s 仅 xy ±0.5;**mimic 开**:同 BeyondMimic |
| 执行器延迟 | 采纳前死代码；新 vendor profile 已在 policy-step 边界开 `[0,2]`（§13） | 死代码(同款) | 字段全默认 0 | 仅 Go2 有字段且未设;H1/G1 无字段 |
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

该零成本诊断现已实现：每拍第一个 strike-window tick 记一次
`racket_target_distance`，用 `0.075/0.15/0.20/0.30/0.50/0.70/1.00 m` 分八个 finite
bin，另记 nonfinite/总数/有限和，并把 armed latch 纳入 exact resume。当前只缺 vendor
Pod 运行分布；多数 `>0.20 m` 就停 long 转粗+细核，多数 `<=0.20 m` 才继续查
termination/控制层。

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
   weight=zeros / bias=ready 姿态,[train.py:5004](../../hope_training/whole_body_tracking/scripts/train.py) 强制 0.02 精确等值,
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

**全篇最高杠杆的因果链**:真 reset 的 ~7 ms/env Python 成本(canonical_sha256 八处调用点
无缓存重算、逐 env 证明转录 ~9 次 `.item()`、O(N²) 区间扫描;实测 25-116 s/iter,健康带
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
| R0 | **给真 reset 降本**(memoize canonical_sha256 八调用点、证明转录懒化/GPU 化、O(N²) 扫描修掉;assert_contract 原样保留) | 外部先例:legged_gym reset O(kernel) 不 O(env);**这是 R1/R3/R4/R5 的 blocker** | 无 |
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
| reset 级 DR | 旧 stable-ready N1 的 PD/mass/CoM 全关；新 vendor PD 是 startup，delay 才是 episode-reset 抽样（§13） | 推撞三套 | 速度踢+interval push | 同 | 同(5 s 固定) |
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
`a3_ultra` 29dof)DR/PD 配置摘要,由 Franco 提供;**无法克隆核验,以摘要为准**。
它与我们是**同底盘**(A3;我们是 31 自由度含头的乒乓变体),因此其数值的迁移效力高于
BeyondMimic/mjlab/unitree 三家。**Franco 07-31 进一步定谳：这份新训练设定比仓内旧
URDF/MJCF/deploy 常数更权威；新训练 baseline 以它为真源，旧常数降为 legacy 部署对照。**

### 11.1 Kp/Kd/effort/armature 逐关节 diff(我方 [agibot_a3.py:222-364] vs 厂商表)

**结论:绝大部分已经对齐**(两边同源自 deploy 常数;armature 前 4-5 位有效数字一致,交叉
确认我们的 MJCF 转录基本正确)。真实差异只有三处:

| 关节 | 厂商 parkour | 我们 | 判定 |
|---|---|---|---|
| 髋(pitch/yaw 80/3,roll 120/4)、膝 250/8、踝 50/2、肩 40/3、肩yaw/肘 30/2、腕roll 30/2 | — | 同值 | ✅ 全对齐(effort 220/320/118.2/54.75/60/24 亦同) |
| **waist_yaw** | Kp **80**/Kd 3(与髋同组) | Kp **85**/Kd 3 | **新训练 baseline 采用 80**；旧 deploy 85 作为 legacy 对照留档 |
| **waist_pitch effort** | **115** | **118** | **新训练 baseline 采用 115**，action scale 随之从 0.590 变为 **0.575** |
| **wrist_pitch/yaw 整组** | Kp **30**/effort **24**/armature **0.004968** | Kp **20**/effort **6**/armature **0.00081** | **新训练 baseline 采用 30/24/0.004968**，action scale 从 0.075 变为 **0.2**；旧乒乓 URDF/MJCF/deploy 值保留为 legacy 硬件合同差异警告 |
| head_yaw/pitch | (无头,29dof) | 40/2, effort 6 | 不适用 |

**legacy 对照证据链(全在本仓)**:`agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf` 与
`agi/URDF/a3_t2d5/urdf/model.urdf`(两版 URDF 同值,支持"电机无变体差异")、
`agi/A3_MuJoCo_Sim/.../a3_pingpong.xml:160-171`、
`agi/a3_deploy_example/.../a3_policy_parameters.hpp:94-100`。
**推论**:新常数不能被原地塞进旧 N1 身份或用旧 checkpoint 续训；它们改变了腰/腕控制和
action decoder，必须重新物化 dynamic-ready、nominal-hold、policy/bundle/launch SHA，然后 fresh 发车。

### 11.2 DR 全面对照(厂商 vs 我们现役)

| 轴 | 厂商 instinct_mj A3 | 我们(N1) | 判定 |
|---|---|---|---|
| PD 随机化 | **startup**,Kp scale log_u **(0.8,1.2)**,**Kd (0.7,1.3) 不对称更宽**;play 关 | ActionBall 通用 task 为 reset 级 Kp/Kd 同 (0.85,1.15)；**现役 N1 stable-ready 全关** | 新 baseline 拆 Kp/Kd 两键并采用厂商范围；不倒灌旧 cohort |
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

### 11.3 对齐动作清单(按优先级;新物理身份 fresh 发车)

1. **[host 已实现，Pod pending]** 新 baseline 改为 waist-yaw Kp 80、waist-pitch effort 115、wrist-pitch/yaw
   Kp 30 / effort 24 / armature 0.004968，并重物化 action scale、dynamic-ready、nominal-hold 与全部 SHA。
2. **[host 已实现，Pod pending]** 执行器延迟 **[0,2] 控制步、每集抽一次**；机械等价/安全门代替学习 A/B。
3. **[host 已实现，Pod pending]** `pd_gain_range` 拆成 Kp **(0.8,1.2)** / Kd **(0.7,1.3)** 两键；
   legacy 单键只作向后兼容，混用必须 fail-loud。
4. **[host 已实现，Pod pending]** 新 baseline 推撞数值为厂商 6-DoF 组(vx/vy ±0.25、vz ±0.1、
   roll/pitch ±0.26、yaw ±0.39)；乒乓任务首版保留 5–15 s 无 gate，不照抄 parkour 1–3 s。
   recovery-hold gate 必须有独立 eligibility 与 skip/applied counters 才能声称接通。

这里的“host 已实现”只说明源码和定向机械测试存在；tracked authority runtime
materialization、实际 probe receipt 与 vendor plant Pod 运行边界以 §13.2 为准，不能据此
直接发 `long`。
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
  uniform,ADR 初始 [0.9,1.1]、上限可自动放宽到 [0.4,10.0]（行 278–281）——直接先例;
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
