# Reward 新一代设计(论文对照 + scale 推导)— 2026-07-25【部分归档】

> **现行权威文档 = [reward_v2_explained_20260725.md](reward_v2_explained_20260725.md)(v2.2)。**
> 本文档保留仍然有效的 §0 设计准绳、§1 五源论文对照、§2 scale 推导与 §4 验证记录;
> §3/§3.5 的数值蓝图是 v2.0 草案历史,已被 v2.1/v2.2 裁决迭代取代,只作演化追溯,勿照抄。

依据:Franco 2026-07-25 的 23 点裁决 + 五源论文/代码逐项提取(BeyondMimic arXiv:2508.08241 + 上游 repo、mjlab velocity 任务、HITTER arXiv:2508.21043、SMASH arXiv:2604.01158、PACE arXiv:2509.21690 + 开源代码)+ scale 课程文献(DeepMimic/AMP/KungfuBot/ASAP/羽毛球多阶段/PopArt)。
配套:[reward_audit_20260725.md](reward_audit_20260725.md)(现状 74 项盘点)。

## 0. Franco 核心设计思想(后续判断的准绳)

1. **收入分层**:模仿(每帧)< 击中(一次性)< 拍面/速度/位置质量(更大)——在**收敛达成度**下定序;罚项(抖动/平衡)是约束,**永远不许淹没收入**。
2. **不要 hardcode 的相位折扣**(motion_scale_in_window 之类):让 value 估计迷茫,不是大道。
3. 聚合一律 **SUM**,靠权重调;模仿要**全身**(flag 开关);平衡组向 local motion(mjlab)对齐、防杂;terminationss 防"到处学"即可。
4. 随挥必须学完整(否则击球段学减速);恢复到 ready 靠模仿老师,不靠专项 reward(foot_orientation/hold_ready 属"过了"的做法)。

## 1. 五源对照表(按目的组)

### 1.1 模仿组

| | 我们(现状) | BeyondMimic 原版 | HITTER | SMASH | PACE |
|---|---|---|---|---|---|
| 结构 | 6 项(锚 pos/ori + body pos/ori/linv/angv),上身 7 件,窗内 ×0.25 | **同 6 项,weights/stds 逐字节 = 我们的 base**(0.5/0.3, 0.5/0.4, 1.0/0.3, 1.0/0.4, 1.0/1.0, 1.0/3.14),全程 dense 无门控 | BeyondMimic 式,**pelvis 以上**,全程 dense,无窗内打折 | BeyondMimic 式全身,**拍腕除外**(R16 有论文背书),全程 dense | **无模仿**(残差动作 + 关节偏离 L1 罚撑姿态) |
| 课程 | adaptive sigma(只 pos/vel) | **无**(权重与 σ 全静态;课程在采样分布:RSI+难段优先) | 无 | task 核 adaptive sigma(mimic 核不动) | 无 |
| 备注 | anchor_pos 打球谱系删除 | 论文不发数字,repo 即真相 | 击前另有 base station 项,击后关 | 动作库最近邻选参考 | — |

**判读**:我们的 base 六项与 BeyondMimic 完全同分布(好);偏离在三处——窗内 ×0.25(HITTER/SMASH 都不打折,靠窗内高权任务项**加法**盖过)、上身-only(Franco 已裁全身+flag,SMASH 也是全身)、hold 归零(BeyondMimic/HITTER 无此手术)。**v2 方向:模仿全程全额、全身(flag),窗口只用于任务项激活(加法),废除打折(减法)。**

### 1.2 击球组

| | 我们 | HITTER | SMASH | PACE |
|---|---|---|---|---|
| 位置 | 17(W)/7(V),exp(-e²/σ²),紧窗 | 稀疏高权窗内(数字未发表) | exp(-e/σ) **一次方**,窗 0.02s(紧) | 接触奖 +150 一次性(proximity 连续核) |
| 速度 | 7/17,宽窗 | 同上窗内 | exp(-e/σ),窗 0.1s | 出球质量项吸收 |
| 拍面 | 5,固定 σ,窗内 | 同上窗内(拍面⊥拍速假设) | exp(-e/σ),窗 0.1s,\|n·v̂\| 对称双面 | 无显式拍面 |
| 全中奖金 | strike_success 5(乘法) | — | r_succ 稀疏(pos<0.04m ∧ ori<0.05 ∧ vel<0.5) | table_success +100 |
| 接近引导 | racket_progress 10(伸缩差分) | base station(击前) | — | EE/base/vel 三条 flat-top 引导(2/5/5)+ mask_invalid 硬零 |
| 课程 | adaptive sigma pos/vel | 无 | **adaptive sigma pos/ori/vel 三通道**(去掉后 SR 86.4→22.6,策略塌缩回纯模仿) | 无(论文明说 binary 太稀疏训不动,靠解析预测造 dense) |

**判读**:结构上我们与 HITTER/SMASH 同型(dense 模仿 + 窗内高权任务 + 全中奖金)。三处可校:①SMASH 的 σ 适配是 pos/ori/vel **三通道**——我们漏了 normal(已核实,face 相对降权最多 7×);②SMASH 窗口不对称(位置紧 0.02s、面/速宽 0.1s)——我们已有 1c 分窗机制未启用;③PACE 的 flat-top 核(容差内零梯度)与 mask_invalid(引导项在击球窗硬零,**tracking 永不打架 strike**)值得抄进引导项。

### 1.3 上台组(出球质量)

| | 我们 | PACE(唯一有完整数字的) |
|---|---|---|
| 过网 | +20,exp 核 @net_top+0.12,触球帧一次性 | +100,exp @z_target 1.11m,std 0.4,触球帧一次性 |
| 落点 | +30,exp 核 σ 1.0(爬坡放宽),全合法 +1 | +60,**线性有符号距离**(threshold 3.0)+ 真实弹跳 +100 |
| 旋转 | +5 minimize(保守) | 无(spin 硬零假设) |
| 结算方式 | 虚拟球 rollout(解析),capture 门(9.5cm+进拍速) | 触球帧解析预测(无 Magnus,阻力对数式) |

**判读**:我们已经就是"按出球质量"的双层结构——题库反解的拍速/拍面 = 落点质量的**密集代理**(每步),virtual landing/net = **稀疏验证**(每拍)。与 PACE 的实质差异只有 capture 门造成的早期零梯度(PACE 用 contact proximity 连续核,没有硬门)。保守策略(spin minimize)已是现状。

### 1.4 平衡/平滑组(mjlab local motion 对齐参照)

| mjlab velocity(参照集) | 我们对应 | 判读 |
|---|---|---|
| upright **+1.0 正向 exp 核**(std √0.2) | upright **-1.0 罚**(L2) | mjlab 用收入型,我们用税型;v2 建议改正核(收入型天然是 alive bonus,顺带解 fresh 臂早期净负问题) |
| pose +1.0 正核,std 按速度档分层(站立最严) | 无对应(姿态靠模仿) | 全身模仿开启后由模仿覆盖,不需要 |
| dof_pos_limits -1.0(线性) | joint_limit -10 + qbar -0.65 | 我们更重且双层(q 与 q_des);qbar 有站姿豁免后合理 |
| action_rate_l2 **-0.1**(唯一平滑项) | -0.1 base(波 -0.2)+ action_acc -0.05 | mjlab 外部先验:> -0.1 未必有益;我们 -0.2 是 jiayi V14 chatter 证据驱动,保留但记住这是偏离 |
| foot_clearance -2.0 @0.1m(**带地形高度传感器**) | -0.01 @0.15(世界 z,已单侧化) | mjlab 用 terrain sensor 做净空;我们粗糙地形臂长期方案应加 raycast |
| foot_swing_height -0.25 / foot_slip -0.1 / soft_landing -1e-5(力量纲) | slip_sq -1.0 / drag -0.5 / soft_landing -0.003(无量纲) | 我们 slip 税明显更重(×10);v2 可回调对齐 |
| 无 stand_still、无 alive bonus | 同 | 一致;站立质量靠采样分布(10% 零指令)不靠 reward |
| **没有任何窗内站稳包** | strike 四件套(-2/-0.5/-0.5/-1) | HITTER/SMASH 也没有;PACE 只有一条"单脚击球罚"(hit_unstable_support -10,触球事件门)。v2:四件套删,换 PACE 单条 |

### 1.5 恢复组

HITTER 的答案最干净:**没有恢复专项 reward**。恢复=①参考 clip 含完整随挥回 ready;②base 项击后关闭(放开去下一球);③10s episode 内连续挥拍采样。SMASH 同。PACE 也无(靠常开平衡项)。→ 支持 Franco 第 14 点:hold_ready/foot_orientation 删,恢复靠全程全额模仿(C1 修复后随挥段老师已恢复全权重)+连拍任务结构。

## 2. Scale 记账(每帧 vs 一次性的换算)

记号:dt=0.02s,T=每回合步数(16s→800),K=每拍数,T_c=T/K(每拍周期步数),模仿每帧上界 I=Σw_i(核≤1),命中率 p,目标命中率 p*。

**每帧等效收入(PSE)**:全程 dense = w·ρ;窗口 dense = w·ρ·(W/T_c);一次性 = **B·p/T_c**。
(折扣稳健性:T_c(1−γ)≪1 时折扣视角同结论。)

**定权公式**(收敛态定序 模仿 < 击中 < 质量,边际系数 m₁,m₂∈[1.5,3]):
- 击中一次性 B = m₁·I·T_c/p*
- 质量项(窗口 dense,窗宽 W 步,n 项均分):F_j = m₂·m₁·I·T_c/(W·ρ*·n)
- **自动课程**:早期 p≈0 → 击中 PSE≈0 < 模仿,先学模仿;p 上升后定序自动翻转为击中/质量主导。**静态权重自带课程,不需要改权重的 schedule。**

**罚项预算**:P ≤ f·当前阶段收入,f≈0.1–0.2;**早期分母必须用模仿收入**(ρ_I,min·I,取 0.3–0.5),不是击球收入——现 IU 的 1/3 守卫用窗内击球收入当分母,在 fresh 臂早期失效(净收入为负 → 死了最优)。另对 action_rate/torque 逐项 clamp,保单帧最坏罚 ≤ I(防单帧梯度反转)。

**方差论证**:同 PSE 下 one-shot 方差 ∝ B²p(1−p)/T_c,窗口 dense 小 1–2 个量级 → **命中判据用 one-shot(语义不可分),质量项尽量窗口 dense/连续核**。PACE("binary 太稀疏训不动")与羽毛球论文(B=4000 但配宽 σ 起步)是这个权衡的两端。

**课程放哪**:文献一边倒支持"**静态权重 + σ 课程**"——KungfuBot σ←min(σ, EMA误差) 使每帧实际收入近似恒定(≈e⁻¹)→ value 全程平稳(正中 Franco 第 9 点);改权重的 schedule = critic 追移动目标,要么 PopArt 要么像羽毛球论文分阶段重训。SMASH 消融:固定紧 σ 会让任务奖早期不可达 → 策略塌缩回纯模仿(SR 86.4→22.6,face 4°→35° 而 mimic 指标反而变好)。

## 3. v2 蓝图(草案,数字按 §2 公式代入后另行 prereg)

- **模仿**:6 项全程全额,全身(full_body_mimic=true,今日已实现 flag);放人开关保留(第 15 点);**废除 motion_scale_in_window**;hold 不再归零(hold 场景本就被 planner 禁,留待 rally 课程回归时直接全额)。I 归一为 1。
- **击球**:窗内(±0.12s,C1 修复后窗诚实)pos/vel/normal 三通道 + strike_success 乘法奖金;σ 适配扩到 **normal**(SMASH 三通道);分窗启用(位置紧/面速宽);racket_progress 保留但加 PACE 式 mask(窗内硬零,tracking 不打架 strike)+ flat-top 容差。B、F 按 §2 公式定。
- **上台**:现结构保留(密集代理+稀疏验证);spin minimize 维持;capture 门早期零梯度交给击球组解决,不动。
- **平衡**:mjlab 对齐版——upright 改**正向 exp 核**(+1.0),qbar(-0.65,站姿豁免版)+ joint_limit,action_rate -0.1~-0.2 + action_acc -0.05,foot slip/drag 减到 mjlab 档位,soft_landing -0.003;**删**:strike 四件套(换 PACE 单条 hit_unstable_support)、foot_orientation、hold_ready、prestrike_upright/waist_twist(反作弊交给全身模仿+地形)、settle/slew 探针系转纯指标。
- **SUM 一律**(今日已改两个 hinge;settle 五类是异类目录的 mean,维持)。
- **推事件**:速度推与力推合并单一事件,每次 fire 二选一(等冲量口径),防同时叠加(第 20 点)。
- **终止**:保留 time_out 16s / base_fell_tilt 0.7 / base_too_low 0.5 / 参考包络 z-only(全身模仿方向下语义更对,z-only 留步法自由);**不加 alive bonus**——upright 正核化后收入侧天然为正,fresh 臂早期净负问题由 §2 预算规则解决。
- **DR**:白噪+AR1 已按场馆标定开着;**延迟是缺口**(见验证:revision 谱系被 NO-LAUNCH 守卫钉在 delay=0,需要"governor+actor 同吃一份延迟 transport tuple"的代码活);base_com DR 保持开(不学 jiayi 全关)。

## 3.5 v2 数值表(草案 v0——公式代入名义假设;发射前按 probe 协议校准后冻结 prereg)

**名义假设**(全部要被 probe 实测替换):T=800 步(16s),一拍周期 T_c=100 步(~2s,v7 表 t_cycle),窗 W=13 步;达成度目标态 ρ_I*=0.6(模仿核均值)、ρ_Q*=0.5(窗内质量核)、p*=0.7(命中率);边际 m₁=2、m₂=1.5。单位=weight-units/步(dt 全局一致,比例即真自由度)。

| 层 | 项 | 权重(草案) | 每步等效收入 PSE(目标态) | 依据 |
|---|---|---|---|---|
| L1 模仿 | 六项 motion_*(全身,全程全额) | 0.5/0.5/1/1/1/1(=BeyondMimic 原版) | 5.0×0.6 = **3.0** | 基准层 |
| ~~L2 击中~~ | ~~strike_capture_bonus~~ **v2.1 删除**(Franco 07-25:capture 指示器=人造代理的二值版;capture 门保留=上台组闸门) | 0 | — | 落点核才是物理正确的联合成绩单 |
| L3 质量 | racket_position / velocity / normal(窗口 dense,σ 三通道自适应) | **60/45/35**(W 偏位;V 臂 45/60/35) | Σ138×0.13×0.5 = **9.0** = m₂×L2 | Σw = m₂m₁·PSE₁/(duty·ρ_Q*) |
| ~~L3.5~~ | ~~strike_success~~ **v2.1 删除**(三核乘积=结果的人造 AND 代理,落点核物理正确) | 0 | — | — |
| L4 上台(**v2.2:landing 独扛**) | 只留 virtual_landing(legal=过网∧落台=先决条件 gate,门内底薪 0.6+中心核梯度;pass_net/spin 删,遥测保留) | **0/1750/0**(名义;probe 按 legal 率校准 ~1736) | PSE_table = 1.2×质量层 ≈ 9.7;单拍事件额 ~1400 | 落点=拍位×拍速×拍面经球物理(RK4 含 Magnus,k_m 场馆拟合)的联合成绩单 |
| 平衡 | upright_exp(+,新) / hit_unstable_support(−,新) | **+1.0 / −10** | +0.9 站立收入(兼 alive bonus)/ 触发扣 | mjlab / PACE |
| 安全 | qbar / joint_limit / soft_landing | −0.65 / −10 / −0.003 | 常态 0(站姿豁免后) | 不动 |
| 平滑 | action_rate / action_acc | −0.2 / −0.05 + **值 clamp**(rate@10、acc 分位数待测) | 单帧最坏 PSE ≤ 2.0 | clamp 防 fresh 臂"自杀区间"(§2 B4),静态、value 平稳 |
| 清零 | strike 四件套 / foot_orientation / prestrike 两件 / arm_overreach / hold_ready / foot_slip_sq(→−0.1) / foot_drag(→0) | 0 | — | v2 蓝图 §3 |

**预算核对**:早期分母 = ρ_I,min·I = 0.4×5 = 2.0/步,f=0.15 → 罚项 P ≤ 0.3(clamp 后满足);收敛期收入 3+6+9=18/步,罚占比自动 ≪ f。

**probe 校准协议**(发射前一次,200-iter probe 读 tensorboard):实测 ①模仿核均值 ρ_I(早/晚);②窗内三核均值 ρ_Q;③E‖Δa‖² 与二阶差分的 p95(定两个 clamp);④capture 率 p 轨迹。代回公式重算 B/Σw/clamp → 冻结 prereg。任何臂发射前 `strike_window_hit_rate` 必须读 ~0.05–0.08(C1 修复生效的免费探针)。

**实现缺口**:~~`strike_capture_bonus`(L2 one-shot)尚未实现~~ 已落地(2026-07-25 本批):hope_rewards.strike_capture_bonus 读现成 vb_fired capture 门,HOPEVirtualBallRewardsCfg 以 weight=0 声明,reward_pack=v2 翻译层直写名义 850。同批 Franco 裁定默认翻转:train.py 的 reward_pack 缺席 = 按 v2 展开(§3.5 名义值全套),v1 变成显式兜底 flag(`task.rewards.reward_pack=v1`);legacy 配方(motion_scale_in_window / adaptive_sigma=false)在默认路径上响亮失败,必须显式声明 v1 才能渲染。

## 4. 三项验证结论(2026-07-25)

1. **planner 同任务修正**:开 planner_revision 时**每 20ms 一次原子修正**(pos+vel+face+tts 四元组),从 commit 到触球前一步,无 windup 冻结,当步进 obs。但 truth 在 commit 冻结,修正是有界收敛估计(±0.1m/±0.5m/s/0.2rad/±0.25s);**真移动的目标(超包络)当前任务内表达不了**,只能等挥拍 wrap 换新任务;mid-swing abort 不存在。planner_revision 默认 OFF 按波开。
2. **延迟/白噪**:白噪+AR1 **已经全波开着**(0.0019/0.0052/ρ0.717,场馆标定);Franco 记忆正确——delay=2 在 07-11..16 谱系 live 过,现 revision 谱系因 transport tuple 原子性守卫强制 0(hope_commands.py:763-768)。非 revision 臂恢复=纯配置;revision 臂=代码活。实机依据:传输 <10ms、端到端 ≤20ms,delay=2(40ms)是保守上界;jiayi 8dbb91e8 同方向(白噪 0.004 未标定 vs 我们 0.0019/0.0052 标定)。
3. **球台高度**:虚拟球台/网/落点**全部 env-origin 锚定**(rollout 显式减 3D env_origins),env_origins.z 骑地形 max → 球台不会显得矮;残差反向:台面比实际脚下高 0~4cm(=凹凸扰动本身)。全正原因:Isaac noise_range 是绝对高度带 [lo,hi],非 ± 抖动;想零均值用 (0.0,0.04)。无需修。
