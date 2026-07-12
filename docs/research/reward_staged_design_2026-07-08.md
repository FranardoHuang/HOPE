# 分阶段 reward 评估与设计(提案 v0,2026-07-08)

**要回答的问题(franco 原话重述)**:能不能让策略在"打不到球"的一开始,一边学动作一边把拍
挥到指定位置;等"打得到"了,再学目标拍面和拍速。窗口和门槛怎么设计?拍速是不是不必全程
模仿——也许触球前变向,也许一以贯之自己会调(franco 自认可能是错的,要认真论证)。

**一句话结论**:结构上按 franco 说的分两阶段,但**"阶段"属于每个球、不属于训练日历**——
用回合内接近度门控 + 分通道时间窗实现,不做"先学动作、后通任务"的日历式硬切换(四篇论文
零支持串行课程;SMASH 去掉容差退火 SR 86→23 是最近的反例)。拍速不全程硬模仿:窗外老师
引导、触球窗(宽,±0.1s)内答案接管,手腕速度的"双主人打架"(下详,中位 34°)用消融臂裁决。

---

## ① 诊断:现状奖励在两个阶段各给什么梯度、哪里断档

口径:主攻臂 s1w3_main(v5 标定对 + 题库 v2 + face_command + adaptive_sigma,热启 model_9600),
奖励收入=Episode_Reward 分项(≈奖励/秒),取证数据 `s1w3_curves.json` income_table。
live 权重来自 launch log 的 applied 清单(不是解析估计)。

| 时段 | 在发钱的通道(live 权重) | 实测收入/秒(后期) | 断档 |
| --- | --- | --- | --- |
| 等球段(63% 步数) | hold_ready(2.0) | +0.66 | 模仿 pos/ori 挥拍段限定不发;任务 0 |
| 挥拍段·窗外 | 模仿五项(各 1.0;anchor_pos 已移除;pos/ori 挥拍段限定,lin/ang_vel 全时段) | +2.24(pos .26/ori .21/lin_vel .74/ang_vel .62/anchor_ori .41) | **断档一:窗外没有任何"挥拍到指定位置"的持续梯度**——racket_progress(权重 10)是 telescoping 项,净额=起点距−入窗距,热启后实测仅 +0.037/s ≈ 死通道;franco 要的"常引导位置项"现在不存在 |
| 击球窗(±0.12s,三通道同窗) | racket_position 14.0(std 0.2)/ racket_velocity 10.0(std 1.0)/ racket_normal 5.0(std 0.30) + strike_success 5.0(乘法) | +1.02 / +0.43 / +0.37 / +0.15 | **断档二:无"触点紧、面速宽"分层**——位置、拍面、拍速同一个 0.12s 窗同时要求;窗内答案速度与老师手腕速度模仿打架(题库答案在老师挥速锥外中位差 34°) |
| 触球帧(一次性) | virtual_landing 30 / pass_net 20 / spin 5(vb 捕获门 9.5cm) | +0.29 / +0.09 / ~0 | **断档三:打不回时全是死钱**——热启转轨早期 normal 5e-5、success 0、landing 0.001(exp 核饿死型断电);"阶段二通道"其实早就自然断电,但断电≠没代价:窗内梯度噪声照吃 |
| 常年负项 | action_rate_l2(−0.10) | **−2.41 独大**(负项合计 −3.5) | **断档四:临触球变向的机械对抗项**——最后一刻改拍速方向的每一步都是 action_rate 的纳税大户,与"允许触球前变向"的目标直接冲突 |

终止侧(自杀取证):训练里的"摔"99.9% 是跟踪包络终止(ee_body_pos/anchor_pos z>0.25m),
不是倾倒;反手抽中≈必死没收全部后续球 → 平均回合 1.9-2.5s。**窗口/门控/常引导都治不了这个**
(fixE 掐头臂 0.99→0.26 已证病根在 v5 反手参考),反手修复走修A/B/C 梯子,另案。

**⚠ 比值读数与比值臂更正(本次审计新发现,launch log 为证)**:NOW.md「比值现状读数」记的
"跟踪 4.0+0.5+0.5、拍面单项 0.5 权重"与实跑不符——VirtualBall 类内的 0.5/0.5 重平衡被 yaml
组合(DeployParity ARM A pin)覆盖,**所有 S1 臂 live = pos 14 / vel 10 / normal 5**
(`rewards.racket_normal.weight=5.0` 见 s1w3_main.log applied 清单)。连带后果:比值臂按
"基线 0.5"设计,**face3x 实跑 1.5 = 主攻×0.3(是降权臂!),face10x 实跑 5.0 ≡ 主攻(纯复刻臂)**。
"×1/×3/×10 正手打平"的正确读法是"0.3×/1×/1× 打平":拍面权重 1.5→5.0 在固定点正手上不敏感,
且白送一对复刻臂 = run 噪声尺(正手上台率 0.78 vs 0.77 → **噪声 ≈0.01,判读门槛取 ≥0.03**)。
拍面单项:模仿的实测收入比 = 0.37:2.24 ≈ **1:6**(不是 1:9)。需要 franco 裁决:0.5(类注释
的设计本意)vs 5.0(实跑)哪个算基线;若真要 ×3/×10,应发 15/50。

---

## ② 设计:两阶段结构、连续门控实现

**原则(论文取证的裁决)**:HITTER/SMASH 模仿+任务同炉,PACE/Ace 零模仿,无一串行阶段;
文献支持的"分阶段"= 容差退火(SMASH adaptive σ)+ 自动统计门槛 + 难度自适应采样。所以:
franco 的"两阶段"落成**四通道 + 一个回合内的门**,让"早期主要吃模仿+常引导、后期吃窗内任务"
作为误差收敛的**涌现进程**,而不是排程。

| 通道 | 内容 | 状态 |
| --- | --- | --- |
| A 全程·风格 | 上半身模仿五项(现状不动,挥拍段限定)+ 平衡正则 | 已有 |
| B 常引导·"挥拍到指定位置"(=franco 阶段一) | B1 racket_progress 保留不动(近死但无害);**B2 新增常引导惩罚**(NOW.md 已立项"常引导惩罚×2"):`-w · min(‖p_racket−p_target‖, d_max)`,pre_strike+窗内全程付,挥不到也有梯度(exp 核远处饿死的解药);两档 w 消融,**铁律:每秒罚金 ≤ 模仿收入 10-20% ≈ 0.22-0.45/s**(按记账实测校,不按解析);B3 位置窗收紧到 ±0.02~0.04s(50Hz 下 ±1~2 帧,SMASH 位置窗 0.02s 同源) | B2/B3 新代码 |
| C 门控任务·"拍面+拍速"(=franco 阶段二) | C1 **分通道宽窗**:normal/vel 窗 ±0.10s(位置窗的 5 倍,SMASH 式12 出处,防手腕加速度尖峰、减 sim2real 差距)= 1c 击球窗分通道奖励,已立项;C2 **通电门**(下表两案);C3 退火:adaptive_sigma 已实装且 S1 在跑(pos 0.075-0.20 / vel 0.5-1.0,每 500 步),把 normal std 纳入(0.262→0.30 区间) | C1 已立项,C2 新代码 |
| D 结果奖金·稀疏大额 | landing 30 / net 20 / spin 5 + success 5(乘法)不动——它们是验证奖金不是引导;vb 捕获门 9.5cm 本来就是"接近度门控"的先例 | 已有 |

**C2 通电门的两案(franco 要求两种都议)**:

| 方案 | 机制 | 优点 | 缺点 | 推荐 |
| --- | --- | --- | --- | --- |
| a. 回合内接近度门 | normal/vel 项 × `sigmoid((r_gate − pos_err)/0.05)`;r_gate 候选 **0.15m**(=2×strike_success_pos_thresh 0.075)或 **0.095m**(vb 捕获门,与奖金同门) | ①"打得到"物理上是回合内事件,分球自动判;②无死锁——单次够到就通电;③够不着时不付拍面钱也不受拍面梯度噪声(blind 臂反手摔率独降 0.62 vs 0.93 = "够不着还硬喂指令加重摔"的旁证);④与 D 通道捕获门语义一致 | 多一个门参数;门附近有二阶激励(见⑤风险2) | **主用** |
| b. 训练进度课程门 | 分侧 virtual_hit_rate EMA>0.5 半通电、>0.8 全通电、跌回<0.3 减半(滞回防振荡);门槛=运行统计自动定(SMASH/文献口径),不人工过线 | 实现最简;全局稳 | ①粗糙:正手击球率 0.998 vs 反手 0(全局门无意义,必须分侧);②**死锁**:反手 hit 恒 0 → 拍面通道永不通电 → 反手永远学不了拍面——正是文献不用串行阶段的原因 | 仅作反手修复期的保险丝,不作主门 |

**课程维度归位**:难度课程作用在"出什么题"(题库 difficulty_deg 字段已存,每题都有,
metrics 里 question_difficulty_deg 已在记录;loader 按最易 30% 开窗滚动扩,门槛=滚动上台率),
**不**作用在"哪个奖励通道通电"。两件事解耦,各自消融。

---

## ③ 速度问题专节:全程模仿拍速 vs 触球窗约束+允许变向

**现状代码事实——拍速有两个主人**:
- 主人一(老师):`motion_body_lin_vel`(权重 1.0,std 1.0,**含 right_wrist_yaw_Link,整个
  挥拍段**)——老师的手腕线速度全程被模仿;
- 主人二(题目):`racket_velocity`(权重 10,std 1.0,**只在 ±0.12s 窗**,锚=题库答案速度);
- 两个主人打架的角度:题库答案速度在老师挥速锥外**中位差 34°**(阶段 0 遗产)——老师没示范
  过答案,窗外把手腕往老师方向拉,窗内往答案方向拉。`free_wrist_ori_mimic` 只解决了**朝向**
  的双主人(手腕已从 ori/ang_vel 模仿剔除),**速度的双主人原样健在**。

**论文证据(四篇全是正面证据,无反证)**:HITTER 拍 pos/vel/ori 只在触球短窗激活、模仿只管
上半身风格,且参考关节态是 critic-only、actor 根本看不见;SMASH 手腕整个剔出动作跟踪项
+ vel/ori 窗 0.1s + 任务噪声随 time-to-strike 递减(显式训练"边挥边修正到最后目标");PACE
容差内 clamp + 临触球主动关引导("保留自由度、不被强迫在预测点触球"原文);Ace 每 32ms
重决策、触网后 49ms 变向救回。统一结构=**慢时标模仿风格 + 触球窗约束任务量(触点紧、拍速拍面宽)**。

**我方证据(两条,方向一致)**:
- v5 反手 = 全程硬模仿不可平衡参考的反面教材(15.5 rad/s²,摔率 0.93-0.97,从零对照 0.97,
  换掐头资产 0.99→0.26)——"参考说什么就全程跟什么"的代价上限已经付过了;
- softteacher(六项模仿整体 ×0.5)全面恶化(反手 0.99、正手摔率也升到 0.44)——**解法不是
  全局压模仿价,而是把打架的通道做外科手术解耦**。

**franco 两个假说的可检验化**(判读噪声尺=复刻臂差 0.01,信号门槛 ≥0.03):

| 假说 | 预测 | 判读指标 |
| --- | --- | --- |
| H1 触球前变向(文献方向) | 把手腕速度从模仿剔除(V1)或窗内模仿让位(V2)后:击球帧拍速-vs-需求角差下降、正手上台率升 ≥0.03 | 上台率(分侧)、strike 帧拍速角差、手腕加速度尖峰(sim2real 代理) |
| H2 一以贯之、策略自己调(franco 自疑的方向) | V0≈V1≈V2,角差都能自然收敛——PPO 自己在两通道间加权成功 | 同上;若成立则**保留**全程速度模仿(白拿 sim2real 平滑红利,不动) |
| H3 折中(引导有值、末段让位) | V2(窗内让位、窗外保留)> V1(全程剔除)> V0 | 若 V1 反而最差=老师速度在挥拍早期仍是有用引导 |

**消融臂(全部默认关、同批对照、信号档 2000-4000 步先行)**:
- V0 对照 = 主攻配方;
- V1 手腕速度剔除臂:right_wrist_yaw_Link 从 motion_body_lin_vel(及 motion_body_pos 可选档)
  剔除——SMASH"手腕剔出动作跟踪"的我方版;新旗标 `rewards.free_wrist_vel_mimic`
  (照 `free_wrist_ori_mimic` 抄,~20 行);
- V2 窗内模仿让位臂:strike_window 内模仿项权重 × k(k=0 或 0.25),窗外照常——"平时跟老师,
  触球前后听题目";新旗标 `rewards.motion_scale_in_window`(默认 1.0,~30-50 行);
- V3 = 分通道窗臂(1c,已立项):vel/normal ±0.10s、pos ±0.02s,与 V1/V2 正交可同批。

**附:action_rate 的角色要点名**——它是变向的机械对抗项(−2.41/s 独大),但也是部署平滑
保险,不建议动。判读窗臂/速度臂时把**窗内 action_rate 收入单列**进记账;若它把 V1/V2 的
变向红利吃光(角差降了、上台率不动、窗内 action_rate 罚暴涨),再议"窗内豁免档"这张牌。

---

## ④ 落地:旗标映射、代码量、每臂一句人话

**现成可直接用**:`task.racket.adaptive_sigma`(在跑)、`task.racket.question_bank` /
`face_command` / `face_command_obs`(在跑)、`task.rewards.free_wrist_ori_mimic`(在跑)、
`task.rewards.motion_scale`、`task.racket.strike_window_s`、`task.rewards.racket_normal_weight`
等每项权重/std 覆盖、bank 每题 `difficulty_deg`(难度课程的数据面已就位)、
`virtual_return_rate(_rally)(_forehand/backhand)` 判读指标(已实装)。
注意:HER(`achieved_target_mix_prob`)在 question_bank 下被 fail-loud 强制 0(代码如此),
门控结构不改变此约束;solver-verified HER 变体属 S2b+。

**新代码(全部旗标默认关,估计量)**:

| 件 | 旗标(建议名) | 代码量 | 人话 |
| --- | --- | --- | --- |
| 1c 分通道窗 | `racket.strike_window_pos_s` / `strike_window_wide_s` | 半天(已立项,W2 门票) | 触点要准(±0.02s),挥向挥速给余量(±0.1s) |
| 常引导惩罚 | `rewards.racket_guidance_weight`(两档) | 半天(已立项) | 挥不到球也天天有"往哪挥"的工资单,小而恒 |
| 接近度通电门 | `rewards.face_gate_by_pos` + `face_gate_radius` | ~30 行 | 拍子够得着球才开始付拍面/拍速的钱 |
| V1 手腕速度剔除 | `rewards.free_wrist_vel_mimic` | ~20 行 | 老师的手腕速度不再抄,答案速度独占手腕 |
| V2 窗内模仿让位 | `rewards.motion_scale_in_window` | ~30-50 行 | 触球窗内老师闭嘴,听题目的 |
| 分侧 hit 课程门(备选) | `rewards.face_gate_by_hitrate` | ~50 行 | 反手修复期的保险丝,防够不着还被拍面拉扯 |
| 难度课程 loader 窗 | (1d；人类责任人：yikang；执行者：Claude) | 半天 | 先做最容易 30% 的题,学动了滚动扩窗 |
| R-a actor 腿参考遮蔽(⑥) | `task.actor_leg_ref_mask` | ~30-40 行 | actor 眼里腿参考=站姿常数,critic 照旧全看(HITTER 结构) |
| R-b 包络终止软化(⑥) | `rewards.envelope_soft` + `envelope_soft_weight`(两档) | ~80-120 行(含记账迁移) | 跟丢参考不再判死,改成站在违规区里每秒扣钱 |
| R-c(i) RSI 跳过安定帧(⑥) | `motion.rsi_skip_settle_frames`(默认 0,先用 6) | ~15 行 | 出生别传送到 IK 瞬态帧上,从第 6 帧起手 |
| R-c(ii) held-RSI 站立高度(⑥) | `motion.rsi_hold_root_stand_z` | ~15 行 | 站姿关节配站姿身高,脚不再穿地被物理引擎弹飞 |

**批次与铁律**:挂 W2 奖励结构波同批(对照=基准臂);未消融改动一律旗标默认关;信号档
2000-4000 步 → 赢家跑到底 12k;**反手修好(修A/B/C)之前,所有臂只按正手判读,反手格写
"未测/待复验"**;报数 2×2×2×2 十六格;比值判读一律用奖励收入记账实测(1b),不用解析估计。
优先顺序(按排序法则:奖励结构=尺子,最先):1c+常引导(已立项,不变)→ 接近度门 →
V1/V2 速度双臂 → R-a 腿参考遮蔽 + R-c(ii) held-RSI 修复(与 V1/V2 同波,见⑥)→
分侧课程门(仅当反手修复期需要)→ R-b 包络软化 + R-c(i)(反手资产修好或 trim6 之后,见⑥)。

**顺手修正两条(不占臂)**:① NOW.md「比值现状读数」与比值臂标签按①的审计更正,face10x
重定位为"主攻复刻臂"(免费噪声尺);② 裁决 racket vel/normal 基线权重的本意(类 0.5 vs
yaml 5/10),消掉 VirtualBall 类重平衡被 yaml 组合静默覆盖的坑(加一行 fail-loud 或把意图
钉进 VirtualBall yaml)。

---

## ⑤ 风险:自杀/刷分漏洞在新结构下会不会更糟

| # | 风险 | 裁决/防护 |
| --- | --- | --- |
| 1 | 常引导是负项 → 加重"死了少交税"? | 自杀取证:每步净收入 +1.76/s、死亡机会成本 ≈ +3.5 回报单位,罚 ≤0.45/s(铁律上限)不会翻转符号;且 `min(err, d_max)` 封顶防远距离深负。**前置条款:挣扎段(swing 中、终止前)按 clip/pre_strike 分桶记账先落地**(取证报告 open item),确认反手必死段净收入符号后再定罚档 |
| 2 | 接近度门的蹭分:把拍伸进 gate 挂着不挥,吃 normal/vel 核 | vel 核锚答案速度(2-3 m/s),静止时 exp(-(v_demand/1.0)²)≈0 吃不到;normal 单项可蹭 → 防护:normal 项乘 vel-gate 或只信 strike_success(乘法,已有)发大钱;门用平滑 sigmoid 免得门沿抖动 |
| 3 | 课程门死锁:反手 hit 恒 0 → 拍面永不通电 | b 案固有缺陷,所以只作保险丝;a 案(回合内)单次够到即通电,无死锁 |
| 4 | 时间套利:加罚后挥拍段净收入变负、hold 为正 → "死快点"微压(取证已证:影响摔的时机不影响事实) | 查账口径改为**挥拍段内净收入>0**(不只全回合);hold 时长由 command 时钟驱动、策略拖不长,结构上刷不了 hold_ready |
| 5 | 站桩不挥拍靠 hold_ready+正手刷分(取证 open item 预警过) | hold_ready 有 reach 门(0.65m)+ in_hold 门;新增常引导罚在 pre_strike 持续扣钱,站桩更亏;监控 in_hold 步数占比(现 63%)不升 |
| 6 | 把新结构当反手解药 | 明示:包络终止与 v5 反手参考病不归本提案管;反手读数在修 C 合格前不进本波判读 |

**尺子**:主尺=训练内上台率(分侧)+ MuJoCo bank 考卷(入账);辅尺=击球帧拍速角差、拍面
误差、手腕加速度尖峰、窗内 task:模仿收入比(记账实测)、窗内 action_rate 收入(新列)。

---

## ⑥ claude 补充消融(franco concern 以外,三臂;超参排期由 franco 定)

铁律同④:全部旗标默认关、同批对照、信号档 2000-4000 步先行、判读门槛 ≥0.03(噪声尺=复刻
臂差 0.01)、反手修好前反手格写"未测/待复验"。三臂共同点:都不动奖励**权重**,动的是
"策略看什么(R-a)/跟丢了怎么罚(R-b)/出生在哪(R-c)"——和①-⑤的窗口/门控正交,可叠加。

### 总表

| 臂 | 改什么 | 超参(钉到代码) | 预注册判读 | 风险 | 代码量 | 波次 |
| --- | --- | --- | --- | --- | --- | --- |
| **R-a actor 腿参考遮蔽** | actor 的 62 维 command 里 24 个腿维(12 关节 pos+vel)喂"默认站姿常数+零速度";critic 的 command 不动(HITTER 的 critic-only 结构)。obs **维数不变**=零契约成本 | 遮蔽维索引见下;填充值=`default_joint_pos` 腿关节(站姿)+0.0;旗标 `task.actor_leg_ref_mask`(顶层 bool,照 `task.physical_ball` 先例) | 正手上台率差 <0.03 → 腿参考无用 → **契约日直接去维(62→38)**;掉 ≥0.03 → 有用,保留;辅看反手摔率与 clip_switch 场景摔率(等球换代事故通道) | 遮蔽维索引算错=毁性实验 → 实现必须运行时 `find_joints` 派生+启动打印核对,禁止硬编码;若胜出转正,部署 C++ 需同步一行(镜像 hold 先例) | ~30-40 行 | **W2**(franco 点名提前;与 V1/V2 同批) |
| **R-b 包络终止软化** | `anchor_pos`/`ee_body_pos` 两个 z>0.25m 包络项不再终止,改"违规状态每步惩罚";`base_fell_tilt`(0.7rad)/`base_too_low`(0.5m)绝对终止保留;`anchor_ori`(0.8,实测率 0.0)保留终止不混淆 | 罚金两档:温和 **weight=-1.0**(=-1.0/s,每步 -0.02@50Hz;违规仍净正:实测收入 +1.76/s)、狠 **weight=-3.0**(=-3.0/s,每步 -0.06;违规净负 -1.24/s);阈值 0.25 不动;旗标 `rewards.envelope_soft` + `envelope_soft_weight` | 反手:被抽中后不再没收余下全部球 → 每回合击球数、rally 连续上台率应升;正手上台率不掉 ≥0.03;**必须配指标迁移**(下详),新旧臂对比用 fall+loss 合计 | ①挂机/摆烂容忍(违规里躺平);②**狠档复活自杀**:净负流下"绝对倾倒"变逃生门 → 监控 base_fell_tilt 率(现 ~1e-5/步)>1e-4 即弃狠档;③跟丢后乱走出题区 | ~80-120 行(罚项 ~25+挂钩 ~15+记账迁移 ~60) | **W2.5/W3**:反手资产修好后或直接 trim6 资产,避免与参考病混淆 |
| **R-c(i) RSI 跳过安定帧** | RSI 相位采样窗跳过首 N settling 帧:multiseg 出生帧 `seg_start`→`seg_start+N`;单 clip 路径采样结果 clamp min=N(失败自适应"越摔越采"的止血) | **N=6**(全局常数;登记表:v5 首帧瞬态 3-4 帧收敛 + 余量;后续接管线内 URDF 限位自动检测器 per-clip 化);旗标 `motion.rsi_skip_settle_frames`(int,默认 0) | 反手 pre_strike 摔率、anchor_pos/ee_body_pos 终止率、出生后 20 步内终止占比下降;正手不掉 ≥0.03 | N 吃掉真 windup(6 帧=0.12s,方向性风险低但要看 swing_completion);wrap 也走同路径=参考从第 6 帧起播(=管线内活体 trim,franco 备选案的训练侧实现);**GMR warm-up 源头修完 N→0 退役** | ~15 行 | 反手修复梯子(修A/B/C)配套臂,或 trim6 对照波 |
| **R-c(ii) held-RSI 站立高度** | hold>0 的 RSI 出生:root z 改写为默认站立高度(关节已是站姿,root 却钉在参考首帧 0.78m → 脚穿地 0.288m 被 PhysX 弹出);xy+yaw 保留参考首帧 | root z:0.78 → **1.0684**(`default_root_state`,agibot_a3.py L174);旗标 `motion.rsi_hold_root_stand_z`(bool,默认关) | held-RSI 出生后 10 步内接触力/root 竖直加速度尖峰(depenetration 事件计数,新 metric)归零;anchor_pos 终止率(现 0.0041)降;正手不掉 | 几乎无(这是让出生状态自洽,不是改激励);唯一坑=z 改了 xy 没改导致悬空/穿桌——用参考首帧 xy 即可 | ~15 行(+depenetration 计数 ~20 行) | **W2 同批默认关跑一臂;验无害即转默认开**(定位=正确性修复,不是消融超参,同 free_wrist 入 base yaml 的路径) |

### R-a 细则:遮蔽维索引与实现挂点

**command 的解剖**(代码事实):`MotionCommand.command` = `cat([joint_pos, joint_vel])`
(commands.py L211-213),62 维,经 `generated_commands` 同时喂 actor(tracking_env_cfg.py
L123)和 critic(L142)。关节序=**Isaac articulation 交错序**(npz 由 csv_to_npz_mujoco 按
ONNX metadata `joint_names` 排列),**不是** GMR/SDK 的"腰-头-臂-腿"序。由官方 29 维映射
`a3_isaaclab_to_mujoco`(a3_policy_parameters.hpp L269-274)+ 头关节 BFS 插位反推:

| articulation idx | 关节 | | articulation idx | 关节 |
| --- | --- | --- | --- | --- |
| 0/1 | left/right_hip_pitch | | 9/10 | left/right_knee |
| 3/4 | left/right_hip_roll | | 14/15 | left/right_ankle_pitch |
| 6/7 | left/right_hip_yaw | | 19/20 | left/right_ankle_roll |

**遮蔽 24 维** = pos:{0,1,3,4,6,7,9,10,14,15,19,20} ∪ vel:{+31}={31,32,34,35,37,38,40,41,
45,46,50,51}。⚠ 头两关节(推导插位 11/16)不在 29 维官方映射里,上表由 BFS 规则推导——
**实现以运行时 `robot.find_joints([".*_hip_.*",".*_knee_joint",".*_ankle_.*"])` 派生为准**,
assert 数量==12 并启动打印全名单入 launch log(预注册以打印为准,不以本表为准)。

**为什么零 OOD**:hold 期间(in_hold 实测 0.61 步数占比)command 本来就整体=站姿+零速度
(commands.py L221-246 的 hold 门),腿维遮蔽值是策略天天见的分布;且奖励侧腿模仿早已去掉
(imitation 五项 body_names=A3_UPPER_TRACKED,只含躯干+双臂)——R-a 只是把奖励侧已做的
"腿解耦"补到观测侧,HITTER 则是三步全做(obs 也不看)。**实现**:新 obs func
`generated_commands_actor_leg_masked`(hope_observations.py),train.py 里仅 swap
`observations.policy.command.func`(照 `racket_position_static` 的 func-swap 先例),critic
不动。**判读边界**:blind 臂(反手摔率独降 0.62 vs 0.91)遮的是 racket target,不是腿参考,
只作方向旁证,不可直接推包。

### R-b 细则:实测锚点与记账迁移方案

**终止解剖(main 臂后期 50 点均值,s1w3_curves)**:terminated 0.0094/步 = anchor_pos
0.0041 + ee_body_pos 0.0058;base_fell_tilt 1e-5、base_too_low 0、anchor_ori 0 →
"99.9% 的摔是包络、真倾倒≈0"就是这两个数。收入锚:income_table late 求和=+1.763/s
(正项 5.28 / 负项 -3.52),两档罚金即按它定(温和=收入的 57%,违规贵但活;狠=170%,违规
净负)。换算口径:收入/秒=weight×term 均值(RewardManager 的 dt 已折进"每秒"读数)。

**实现**:train.py `rewards.envelope_soft=true` → `terminations.anchor_pos=None`、
`terminations.ee_body_pos=None`(configclass 置 None=移除,照 footwork cfg 先例)+ 新增
RewTerm `tracking_envelope`(hope_rewards.py 新 func,直接复用 terminations.py L23-25 /
L51-58 的 z-only 误差表达式,返回 float 违规 indicator;body 列表同 ee_body_pos:双踝+双腕)。

**记账迁移(不做=新旧读数不可比,一票否决项)**:摔倒指标现锚在终止事件上
(hope_commands.py L1467-1509 `_count_swing_starts` 读 `termination_manager.terminated`)。
软化后 terminated 只剩绝对项,pre_strike_fall_rate 语义自动变窄。方案:
1. **"摔"只认绝对倾倒**:pre_strike_fall_rate 保名、语义=base_fell_tilt/base_too_low/
   anchor_ori,wandb 面板加注"envelope_soft 臂语义已变";
2. **违规单独计数**:新 metric `tracking_loss_rate`(分侧)=每 swing 内包络违规 rising edge
   / swing_starts,挂在现有 EMA 累加器旁(定义区 L300-312、读出 L1734+、decay L1894-1903,
   同一 EMA 时标);外加 `envelope_violated_frac`(违规步占比,挂机监控用);
3. **跨臂对比口径**:旧臂 pre_strike_fall ≈ 新臂 (fall + tracking_loss),报表按合计列对齐。

**监控包(预注册)**:base_fell_tilt 率(自杀复活探测,>1e-4 弃狠档)、in_hold 占比(现
0.61,升=挂机)、swing_completion、envelope_violated_frac、每回合击球数。与⑤风险 1/4/5
共用查账口径(挥拍段净收入分桶)。

### R-c 细则:代码挂点与取证钉子

**取证钉子(全部代码复核)**:S1 现役 VirtualBall yaml `hold_steps_range [0,100]`、
`post_swing_start_prob 0.0`,`stand_start_prob` 走 cfg 默认 0.25(MotionCommandCfg L741)
→ **真重置 75% 走 RSI**;multiseg 下 RSI 出生帧恒=`seg_start`(commands.py L393)=v5 首帧
IK 瞬态(7.4-15.9 rad/s);hold_counter 在 RSI 传送**之前**抽取(L507-508,>0 概率 100/101)
→ held-RSI:关节=站姿(joint_pos property 的 hold 门,L229-231)、root=参考首帧 0.78m
(body_pos_w **无** hold 门,L249-250)、站姿需 root 1.0684m → 脚穿地 0.288m(=取证的
~0.29m)被 PhysX 弹出;hold=0(1/101)时:蹲姿+首帧原始瞬态速度=出生即超限速下坠。

**挂点**:(i) commands.py L393(multiseg:`seg_start[new_clip] + N`,clamp ≤seg_len-1)
与 L433-437(单 clip:采样结果 clamp min=N);(ii) commands.py L560-591 RSI 分支内,对
`hold_counter[rsi_ids]>0` 的子集改写 root_pos z。两旗标加进 `_MOTION_KEYS` 白名单
(train.py L275 起)+ MotionCommandCfg 字段(L741 区)。

**交互条款**:R-b 若先于 R-c 上线,出生病(75% RSI 出生即违规)会变成"出生即扣钱"拉低
R-b 读数——**R-b 臂必须与 R-c(i)(ii) 同开,或跑在修好/trim6 资产上**(排臂时机已写入总表)。
R-c(i) 与源头修复(GMR warm-up+逐关节限位,franco 决策①)是同病两级保险:源头修完,
(i) 的 N 归零退役,(ii) 永久保留(它修的是 hold 语义不一致,与 IK 瞬态无关)。
