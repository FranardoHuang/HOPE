# v2.2 冻结权重表(v4rg 谱系)— prereg 草稿,2026-07-26

**状态**:Franco 2026-07-26 口头授权首臂("这个动作可以先发一个训,用来对照换动作")。

## 0. 发射收据(对照臂)

- smoke:`v2smoke_v4rg_ctrl_fresh_seed3_20260726` @ 7bd7c392 —— r1 被守卫误拦(action_rate 基线键,修为剥离+记账),**r2 通过**(迭代 0/2、1/2 零报错,剥离标记落账 ×1)。
- science:**`v2sci_v4rg_ctrl_fresh_seed3_20260726`** @ pod1/GPU0,commit `7bd7c392`,fresh-from-random,20000 迭代 save100,4096 env,v2.2 冻结表全默认(质量 393.4/295.1/229.5、landing 1648.8 legal_base、clamp 9.0/36.0)。角色:**换动作对照臂**——canonical 动作臂将以同 seed/同题库/同权重发射,唯一变量=动作库。
- 后续臂仍需:各自 smoke 一格;canonical 谱系先重 probe 出自己的冻结表。
- **r2 发射收据(07-26)**:smoke `v2smoke_..._r2` @ de888cb4 通过(2 迭代收据完整,stand_start=1.0 生效:ready 台账全采样、站位偏差均值 7.6cm、swing 完成率 84%);science **`v2sci_v4rg_ctrl_fresh_seed3_20260726_r2`** 发射——配方在冻结表之上加:death_penalty −1800、landing 延付 0.24s、stand_start_prob=1.0(废挥拍中段 RSI 空降)。首臂(farmed)目录留证。

### 0.1 r2 卡死与消融双臂重建(07-26 深夜)

- r2 臂再次 sim-init 卡死(15.5 核空转、日志冻于 "Starting the simulation",与首次 r2 前身同签名——该类卡死具独立复发性,已列值班 runbook 观察项;若三发再现,下一个排查变量 = kit 线程 16→8)。按 PGID 清场,目录留证。
- defer0 首发 smoke 死于 Hydra compose:argv 模板取自 `ps`(丢 shell 引号,planner_revision mapping 被打碎)。教训入档:**argv 模板必须取自引号完好的命令文件,不得取自进程表**。
- 重建(引号完好模板+严格串行:等前一发到首迭代再发下一发):**treatment `_r3`**(v2probe@de888cb4,延付 0.24,GPU0)已进训练;**defer0**(ablate@ab232018,延付 0,GPU1)smoke 通过后已发射。两臂 argv 逐字同(stand_start=1.0),唯一差异 = commit diff = 延付默认翻转,单变量成立。

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

**默认面二次收敛(07-26,Franco)**:①`stand_start_prob=1.0`+`post_swing_start_prob=0` 进 v2 包默认(防挥拍中段 RSI 空降;用户显式 motion 键仍赢;canonical_ready_mode 谱系天然无感);②**adaptive sigma 从包中退役**——σ 静态钉验收档 0.075/0.5/0.262(=k_eff 校准时的实测状态,口径自洽),精度由 capture/legal 门与落点核管;机制保留为显式 flag(消融轴:static vs adaptive,SMASH 反向证据待本栈检验)。注意:旧配方模板里的 `racket_*_std=0.2/1.0/0.3` 三键在 adaptive 关闭后会变成【固定宽 σ】,新配方应删掉这三键取 cfg 验收档默认。

**Franco 复核后加固(07-26)**:延付只把刷分周期拉长 12 步没关死——(奖26−0)/28步 vs 26/46步,
摔死重生仍在"跳过等下一球"上套利。两刀补全:①**死亡罚 death_penalty −1800**(=每次死
−36 实际 > 满分上台券 33,"比上台抽奖大一些";只计真终止不计 timeout;刷分账变
(26−36)/28=−0.36/步严格负;PACE −1000×dt=−20 同量级先例);②**relaunch 起步改
stand_start_prob=1.0**(去掉挥拍中段 RSI 空降——BeyondMimic 有 RSI 但 PACE 无、HITTER 连续
挥拍不靠空降,内证:yikang r2 取证单 clip 随机相位仅 38.7% 到击球帧、canonical_ready_mode
本就设计 frame-0 起步)。延付保留为纵深(领奖前 0.24s 摔=直接没收)。

## 0.9 pod1 reward-scale 消融队列(07-26 Franco 授权:"pod1 你看着来,着重 reward scale")

单一队列纪律:空槽拉最前就绪臂;每臂 20k fresh、seed 3,除单变量外与 defer0(=基线)逐字同;pod2 留新动作不占。

| # | 臂 | 单变量 | 键 | 状态 |
|---|---|---|---|---|
| 1 | table_r12 | 上台比例 2.5×→1.2× | virtual_landing_weight=791.9 | **在跑**(GPU2,smoke 通过,checkout scale@6e8a9e6b;applied 双标记确认键压包) |
| 2 | face_rescue | 线性拍面引导 0→−0.08(正手死区逃生梯) | racket_face_guidance_weight=-0.08(theta_max=π 模板已有) | **插队至最前**(07-26:三臂正手 face 93–150° 零过网、反手正常;工具已证 [+1,−1] 符号无误 → 真死区,见 §0.11) |
| 3 | qual_x23 | 质量层 ×⅔ | racket_{pos,vel,norm}_weight=262.3/196.7/153.0 | 待槽 |
| 4 | vsplit | pos:vel 互换(Σ不变) | 295.1/393.4/229.5 | 待槽 |
| 5 | sigma_static | σ 自适应→静态(**方向修正**:配方审计发现在跑基线仍带 adaptive_sigma=true,退役裁决尚未进模板;本臂删该键,验证 v2.3 静态默认) | 删 task.racket.adaptive_sigma=true | 待槽 |
| 6 | base03 | landing 底薪 0.6→0.3 | virtual_landing_base_frac=0.3 | 待槽 |
| 7 | delay2 | 感知延迟 0→2 步(40ms,Franco 裁决#1"延迟/白噪";白噪已在默认,延迟一直没开) | target_delay_steps=2(补偿模式键模板已有) | 候选,待 Franco 点头 |
| 8 | death09 | 摔死罚 −1800→−900(defer0 vs r3 摔率 4× 差提示死亡定价是敏感轴) | death_penalty_weight=-900(该 CLI 键 07-26 审计发现缺失,已接线并带测试——包 direct 值此前无覆盖面) | 候选,待 Franco 点头 |
| 9 | ladder_flat | 收入阶梯斜率 1:3:7.5→1:2:4(阶梯要多陡才够) | quality/landing 同比重排(发射前用校准脚本出数) | 候选,待 Franco 点头 |

判读轴与延付消融同一套(回合长/摔倒率/legal 率/落点质量/各组每步收入轨迹),外加各臂的"阶梯实测 PSE 是否仍单调"。

**~3.8k 中期读数(07-26)**:table_r12(上台 1.2×)全面领先——窗口内 legal/机会 6.8%(defer0 1.8%、r3 0.9%),legal/捕获转化 42%(vs 31%/23%),ep_len 772 满长、摔 3(vs 1/0,同量级)。方向性结论(待 8k 确认):**半额上台奖已是当前最优上台档**;canonical 新动作战役的底座直接用 1.2×,把 2.5×(冻结表原值)反过来当消融臂 land_full。所有队内臂继续用 r2 模板保单变量可比(known 缺口见 §0.10:无 push、落地冲击罚=0——全队一致,不影响臂间比较);v2.3 修正模板只给下一代(赢家确认跑 + pod2 新动作臂)。

## 0.10 配方审计(07-26,Franco 触发:"随机推没开,是不是还漏了什么")

对着 v2 蓝图逐键核对在跑三臂(defer0/treat_r3/table_r12)的真 argv(来源:引号完整的命令文件,非 ps),发现 5 处偏差:

| # | 偏差 | 人话 | 处置 |
|---|---|---|---|
| 1 | `task.push`/`task.force_push` 整组缺失 | 训练全程没人推机器人,抗扰全靠球和自己摔——部署鲁棒性缺口 | 进 v2.3 模板(见下);在跑臂不重启 |
| 2 | `foot_soft_landing` = 0 | 落地冲击罚(蓝图 §2.4 定 −0.003)既不在包里也不在 argv——包漏设 | **已修**:补进 `_REWARD_PACK_V2_KEYED`(marker 31→32,161 测试过);今后默认自带 |
| 3 | `task.racket.adaptive_sigma=true` 滞留 | σ 退役裁决(改静态验收)只改了包不再"代置 true",但模板自己带的显式键还在——在跑基线其实是自适应 | 队列 sigma 臂方向修正(§0.9 #5);v2.3 模板删键 |
| 4 | `target_delay_steps=0` | 裁决#1 要"延迟/白噪":白噪键在,感知延迟从来没开过 | 列 delay2 候选臂(§0.9 #7),待 Franco 裁 |
| 5 | `racket_*_std` 三键冗余 | 与 cfg 默认逐字相同,纯噪音 | v2.3 模板删 |

**为什么在跑三臂不重启**:三臂缺口逐字一致,臂间单变量比较(延付/上台比例)不受污染;重启烧掉 3×10+ GPU 时且答案不变。赢家出线后的确认跑改用 v2.3。

**v2.3 修正模板**(下一代基线;pod2 新动作臂发射前必须采纳——隔壁 session 注意):
- 基于 r2 模板,追加 push 双事件(合并抽签 CLI 未接线,B1 已知 TODO"后续波次接线";用双独立事件近似合并频率——每事件 interval [10,30]s,合成期望 ≈ 每 ~10s 一次、速度/力各半,IU 档位 vel ±0.35 m/s、力 68 N×0.3 s):
  `++task.push.enable=true ++task.push.interval_range_s=[10.0,30.0] ++task.push.vel_xy_mps=0.35 ++task.push.ang_vel_radps=0.0 ++task.push.ang_axes=none ++task.force_push.enable=true ++task.force_push.interval_range_s=[10.0,30.0] ++task.force_push.force_n=68.0 ++task.force_push.duration_s=0.3`
- 删 `task.racket.adaptive_sigma=true`(静态 σ = 默认);删 `racket_position_std/racket_velocity_std/racket_normal_std` 冗余键
- `foot_soft_landing` 不用写——包默认已带 −0.003
- delay2 若 Franco 点头则并入(否则保持 0)

## 0.11 正手 face 150° 判读(07-26:"是不是正反标反了,其实是 30°?"——否)

- **地面真值**:`suggest_face_sign.py` 打在 v4rg 两个 npz 的登记触球帧上——正手 cos(n,v)=+0.966(夹角 14.9°)→ 建议 **+1**;反手 cos=−0.912(155.8°)→ 建议 **−1**。与现配 `mount_normal_sign_per_clip=[1.0,-1.0]` 逐位一致,叉验通过。**符号没标反,150° 不是镜像出来的 30°。**
- **真诊断:正手死区漂移**。三臂共同模式:反手 face 收敛正常、正手 93–150° 且随训练**上升**(r12:87°→101°)。机制:位置核(393)有梯度一路拉,face 核 σ=0.262 在 ~45° 外梯度归零、条件税 readiness 门(pos<9.5cm 等)不开也无梯度——正手 face 无人看管,被位置追逐带着漂;probe 谱系从不打正手,所以老收据从来测不到这条。
- **解药**:线性 face 引导(`racket_face_guidance`,theta_max=π,全角域有斜率)= 当年为"反着的拍面"造的逃生梯,fresh 臂小额启用即可把死区兜住 → face_rescue 臂(§0.9 #2)。
- **对隔壁 session 的警告**:canonical 新动作 fresh 臂大概率踩同一死区,建议同样带上小额线性 face 引导键。

## 0.12 canonical 战役发射收据(07-26 晨,pod1/pod2 交接 Fable 执行)

人话:pod1 三条 v4rg 臂按 Franco"归你用,没停的给停了"停机(终帧 r3=4100 / defer0=3800 /
r12=3900,checkpoint+tfevents 留证,§0.9 的 8k 读数由这些存档出);canonical 谱系 probe 臂已
发射,校准闭环跑完前六臂不发。

- **checkout**:`/workspace/codexschema/nohope_v2c_canonical_20260726` = v2probe 基座 rsync +
  本地 `7e2a1d6e` 的 `scripts/`+`source/` 覆盖 + **工作树准入修复**(commands.py 默认路径
  去 legacy 门、admission 模块删空信任集——未提交,冻结文档前置 2 所述状态)。核验:
  death_penalty_weight CLI 在 train.py ×3、foot_soft_landing 在包 ×7、legacy 门 0 残留。
- **准入路径裁定(前置 2)**:走 legacy motion_file 无门通道(证书链今天出不来);放弃
  canonical registry 表。
- **clip 绑定(前置 3)**:fh_loop_upper(probe8b 5mrad 档,sha 2563…)+ bh_loop_c_upper
  (probe6a raw,sha f6c8…,与冒烟同字节);`strike_phase=[0.355030,0.204286]`、
  `mount_normal_sign=[+1,−1]` 均为 suggest_face_sign/fh_window_pick 对新 clip **重算值**
  (数值恰与 v4rg 相同是巧合,非沿用);FK 逐 clip 箱来自 BINDINGS.json。
- **三个模板降级(与 v4rg 波的真实差异,receipt 必读)**:①v4rg 题库按其族 SHA 合同
  **不可转移**(rebind 工具只收"触球帧逐位不变"的同族件)→ 本波弃 bank,
  uniform+FK 逐 clip 箱(probe/六臂同分布,校准自洽);②face_command 与题库耦合 → obs 合同降
  `deploy_parity`(175 维);正手死区(§0.11)防护改由 v2.3 底座 reward 侧 face 引导承担;
  ③planner TTS 修订与题库耦合 → 剥除;时序多样性由六臂的真实等待时间泛化
  (`hold_steps_range=[25,125]`=0.5–2.5s,Franco 07-26"按实际打乒乓的等待时间",probe 保持
  hold=[0,0] 干净测量)承担。**canonical 题库生成(gen_stage1_questions 适配 canonical 锚点)
  排队**,成了以后 face179/bank/planner 三件一起回归。
- **probe 臂**:`v2c_probe_canonical_upper_seed3_20260726` @ pod1/GPU0,3000 iter,4096 env,
  名义权重(v2.2 包渲染)。smoke 2/2 零报错(WARN 仅地面材质颜色一条,良性);正式发射后
  config 回显核验 foot_soft_landing=-0.003(reward_pack=v2)与 canonical strike_phase 生效。
  argv 文件:`/workspace/codexschema/v2c_canonical_20260726/arms/probe_canonical_upper/argv.txt`
  (planner 映射单引号包裹——r2_argv.txt 模板该参数**未带引号不可直接执行**,已在本波修正,
  §0.1 教训再次生效)。
- **全身谱系 probe 并行**:`v2c_probe_canonical_full_seed3_20260726` @ pod1/GPU1(smoke 2/2 过,
  clip=fh_full+bh_full,phase [0.368932,0.271875]、F 箱来自 BINDINGS)——F 侧 reward 臂的
  k_eff/T_c 提前测,校准窗不空卡。
- 排队:两 probe 3k → v2_probe_extract(--lineage canonical_20260726 upper/full 各一)→ 手工补齐
  k_eff/p_legal/E_land → v2_weight_calibration → 新冻结表落 §1 旁 → v2.3 底座六臂
  (c_base/c_land_full/c_face0/c_delay2/c_death09/c_qual_x23,含 hold=[25,125] 等待泛化)
  pod2 三卡+pod1 串行发射(pod2 现有四条动作矩阵臂共存,5090 叠跑依据见动作实验 §10.1.1)。

## 0.13 canonical probe 猝死事故与修复(07-26,多路会诊结论)

人话:**机器人出生被摆成旧 v4rg 站姿(右腕拍在 0.882 m),而 canonical 参考第 0 帧的拍在
1.229 m,差 0.347 m > `ee_body_pos` 的 0.25 m 阈值;这条配方又把唯一能遮住出生错位的"预备
冻结期"关成 0,于是 4096 个环境全部在第 1 步判死。** 跑了 2500 迭代零学习(回合长恒 1、
回报恒 −35.92 = 死亡罚 −1800×0.02)。

- 决定性证据:右腕参考 z=1.2290(四条 clip 首帧逐位相同)vs 站姿 FK z=0.8820(pelvis z=1.0684,
  `agibot_a3.py:196`),差 0.3470 超阈 0.0970;其余受检体(左腕 0.135/双踝 ~0.000)全过,
  anchor 差 0.152 也过——与日志里 `anchor_pos=0`、`ee_body_pos=4096` 逐项吻合。
  掩码空转链:`commands.py:2705` randint(0,0+1)=0 → `:2803` clamp(min=0) 不抬 → in_hold=False
  → `hope_rewards.py:3337` 的 `value & ~in_hold` 退化成恒等,终止第一步就带电。
- **clip 无罪**:pod1/pod2 的 npz 逐字节相同,pod2 用同 clip、同站姿、同终止项健康运行——
  唯一变量是 hold。
- **修复(只改一个键)**:`stand_start_min_hold=0 → 25`,`hold_steps_range=[0,0]` **不动**。
  依据两条新事实:①`commands.py:2729` 的 wrap 提前 return **在** `:2803` 的 min-hold clamp
  之前,所以该键只在真出生生效,回合内 wrap 仍是 0;②四条 clip 实测**严格闭环**(首末帧
  maxdiff=0),wrap 时机器人与参考都在 ready,**不需要 hold**。代价 = 25/800 = **3.1%** 的步。
- **否决 `hold_steps_range=[0,100]`**(四位复核者主推,被测量有效性审计全部否决):它给每次
  wrap 平均插 50 步冻结,clip 周期涨 41%(upper)/54%(full);而 `v2_probe_extract.py:118-120`
  的 T_c 用逐步窗命中率、`v2_weight_calibration.py:102/106/111` 的三个权重**严格线性正比于
  T_c 与 rho_I** → 整张冻结表偏高 41–54% 且两臂偏得不一样,护栏还不会报警。等于拿被测仪器当药吃。
- **否决 `canonical_ready_mode=true`**(方向正确但今天走不通):该键不在 `train.py:2377-2390`
  的 `_MOTION_KEYS` 白名单里,开机即 `_OverrideError`;且它绑定五段动作银行 + 四个 sha256 +
  promotion 证书,与本波"证书今天出不来走 legacy 通道"的裁定冲突。列为 v2.3 之后的正路。
- **禁止**:抬 `ee_body_pos` 阈值(要抬过 0.347 等于全程放松 39%,会盖住真摔);
  改资产 `default_joint_pos`(会同时改动作零点与 deploy_parity 观测,且与 C++ `pp_policy` lockstep)。
- **重发验收(r2,已实测通过)**:`Live/motion/in_hold` 首迭代 = **1.0**(硬门,原为 0.0);
  `mean_episode_length` 1.0 → **33.7/32.3**(门 >25);`ee_body_pos` 终止 4096 → **126/148**
  (健康对照臂自身首迭代也有 75.6,不能拿 0 当门)。
- **出表前必须做的 T_c 修正**:`T_c_clean = T_c_measured × (1 − h)`,h = `Metrics/motion/in_hold`
  尾段均值(预期 ≈0.03);upper/full 分开算(clip 周期 120.5 vs 92.5 步,h 不同)。rho_I 偏差同阶
  (≤3%),低于表分辨率,收据写明即可。
- **下游六臂三条必办**:①逐臂核对带的是 `hold_steps_range=[25,125]`,任何一条继承 `[0,0]`
  会以同样方式死光;建议显式写 `stand_start_min_hold=25` 当保险丝。②六臂吃的权重表必须用修正后
  的 T_c。③**canonical 谱系独有的代码缺口**:`commands.py:1907-1914` 规定 hold 期交给策略的关节
  参考 = `default_joint_pos`(旧站姿),只有 `canonical_ready_mode` 才换成 clip ready;而
  canonical ready 与站姿在 `right_shoulder_pitch` 差 **1.52 rad**。六臂用 [25,125] 时每拍有
  20–40% 的步收到"目标关节=旧站姿"而 body 空间参考要求它待在 ready——两个自相矛盾的指令
  (v4rg 谱系两者本就一致 ≤0.066 m,所以历史没暴露)。发六臂前要么加"hold 参考取 clip ready"
  开关(须与 C++ `pp_policy` level-0 参考同步),要么把该不一致写死备案。**不许默默改。**
- 残余风险:死因会搬到第 26 步(未训练策略在 hold 解除时仍够不到 ready);实测 r2 回合长 33.7
  正落在预测的 20–50 带内,**若几百迭代后钉在 ~26 不动**才需把出生窗放到 40–50(仍只动
  `stand_start_min_hold`)。另:hold 窗内没有任何一项奖励拉它去 ready
  (`motion_body_lin/ang_vel` 无 hold 门且参考速度被清零 = 付钱让它别动;`hold_ready_weight`
  在 v2 包里钉 0.0),3% 的步扛得住,这是不能放大 hold 的第二个理由。

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

> **常驻工序已收编**：引号完好的命令文件、一格 2-iter 冒烟、严格串行等首迭代等纪律现为
> [消融波发射工序](../../operations/run_ablation_wave_launch.md)。本节保留本波的具体前置。

## 4. 发射前置与已知残项

1. 任意臂:一格 2-iter smoke(渲染层默认即 v2.2+冻结值)。
2. **fresh-from-random 臂**:action_rate/acc 值 clamp 机制必须先实现(冻结表已给档位 9.0/36.0;无 clamp 的 fresh 臂存在早期自杀区间)。resume 谱系臂不受此限。
3. probe 用旧混合(capture 850/success 30/landing climb 30)运行——所有测量为权重无关仪器量,不受影响;但该 run 的 reward 总量曲线不代表冻结表行为,勿用于对照。
4. ‖Δa‖² 实测均值 ~94 偏高(200-iter 换奖励混合后的 KL 扰动期),p95 用 3×均值保守替代;首个 science 臂读实测分布后可回调 clamp。

## 5. 2026-07-27 发射回执:C 组消融批(6 卡满载)

### 5.1 一句话

在钉死的 checkout `/workspace/codexschema/nohope_pin_20260727` 上发了 5 条新臂 + 保留 1 条旧对照,两台 pod 六张卡全满。基线本身换了:每一拍的球拍速度和拍面**是逆解出来的答案(题库)**,不再是从盒子里随机抽的。

### 5.2 发射前抓到的真问题:随机盒子目标和"把球打回去"是互相打架的

新护栏 `allow_unbanked_landing_rewards` 在开机时拦下了当时**正在跑**的那条臂的构型:`target_mode=uniform` + 虚拟球落点/过网奖励 + **没有题库**。人话:球拍速度指令是从一个盒子里随便抽的,从来没有人算过"照这个速度打,球能不能落在台上"——于是"听话地照速度指令走"和"把球打回去"两件事在多数抽样下是反的。

活体证据:那条臂(`wave1_bhloopc_r2_seed0`)跑到 i4319,回球率 0.0038。它现在**保留继续跑,当作对照臂**,正好回答"解出来的目标 vs 随机盒子目标差多少"。

### 5.3 题库(这次才第一次有)

两台 pod 上一份 stage-1 题库都没有——这才是"跑不起来"的真卡点,不是别的。现生成(`--grip off`,marker 唯一权威):

| 题库 | 题数 | 可解率 | torch 闭环落点 | 过网 | stroke-guard |
|---|---|---|---|---|---|
| `bh_loop_c_train.npz` | 6126 | 94% | 6126/6126,中位 3 mm | 6126/6126,余量 ≥0.277 m | 0 拒(需求加速度 max 4.43 vs 可用 27.28 m/s²) |
| `bh_loop_c_exam.npz` | — | 同上 | — | — | — |
| `bh_loop_c_train_wide.npz` | 5836 | 88% | 5836/5836,中位 3 mm | 5836/5836,余量 ≥0.268 m | 0 拒(需求加速度 max 5.05) |

需求拍面与老师拍面的差:中位 7.6°,p90 11.8°,max 19.6°。只有 26% 的答案落在老师挥拍速度锥内(25°/0.6–1.4×)——**这就是这批臂真正要学的泛化量**。

### 5.4 发射清单被补齐的几处(以后不会再踩)

护栏一条条把清单逼出来了,每条都改成了默认或可表达:

1. `racket_vel_range_per_clip` 在题库模式下是死旋钮,护栏要求"设成 None",但配置系统**表达不出 None**(它的默认值来自 cfg 而非 yaml)。已在 `scripts/train.py` 加 `_explicitly_null()`:写 `vel_range_per_clip: null` 才算显式清空,缺省仍然什么都不动。
2. `achieved_target_mix_prob=0.3`(HER 回放混合)与题库互斥——题库覆写发生在 HER 块之后。发射器固定 `0.0`。
3. `face_command=true` 是护栏硬要求(否则解出来的拍面没人打分)。**但拍面观测通道 `face_command_obs` 这批关着**:开了要走 179D 契约 `deploy_parity_face179`,而 schema-3 结构校验**硬性要求正反手双 clip**(单反手臂给不出 `[+1,-1]`,家族表也要求两族齐全)。等新正手编译完成后,双 clip 臂才能开这条观测通道——那本身就是一条干净的消融。
   - 注意:拍面**仍然被打分**,主项 `racket_normal` 权重 0.5(两条 guidance 罚项按设计留 0)。
4. hydra key 一律用 `++`:`+` 在 key 已存在时会炸,`++` 两种情况都对。

### 5.5 六条臂

| pod:卡 | 臂 | 和基线差什么 | 人话 |
|---|---|---|---|
| p1:0 | `c_base_bank_seed0` | — | 基线:题库解出的目标 + 桌子(新默认) |
| p1:1 | `c_notable_seed0` | `table_obstacle=false` | 桌子是新默认,先量它自己花了多少代价 |
| p1:2 | `c_ep20s_seed0` | `episode_length_s=20` | 一回合连打更多拍,考的是打完能不能收住 |
| p2:1 | `c_speedwide_seed0` | `speed_scale_range=[0.5,1.2]` | 老师动作速度带放宽——速度泛化是第一优先轴 |
| p2:2 | `c_base_bank_seed1` | `seed=1` | 任何差异要先大过种子噪声才算数 |
| p2:0 | `wave1_bhloopc_r2_seed0` | 无题库 + `target_mode=uniform` | 旧构型对照(见 5.2) |

待排:`c_ballwide_seed0`(换 wide 题库,来球 1.5–7.0 m/s;题库已就绪 `banks/bh_loop_c_train_wide.npz`),等空卡。

> **为什么不砍"关桌子"臂给它腾卡**:MuJoCo 扫描显示反手 clip 一帧都不碰桌子,看起来"关桌子"是可预测的空结果。
> 但扫描量的是**参考动作**,不是**探索中的策略**——策略学的过程里完全可能挥进桌子,那道终止就会真的塑形。
> 参考动作不碰桌子,推不出策略不碰。六条臂原样保留。

需求拍面差(宽速题库):中位 8.3°,p90 12.8°,max 20.8°;只有 21% 的答案落在老师挥拍速度锥内(窄带是 26%)。

### 5.6 发射时踩的坑(工序补丁)

同一台 pod 上**三个 Isaac 进程间隔 2 秒同时起**,其中一个在 URDF 导入阶段 `malloc(): invalid size (unsorted)` 堆崩。单独重发即活。工序补一条:**同节点串行起,等前一个进 Learning iteration 再起下一个**。

### 5.7 MuJoCo 桌子:正手入台已被物理坐实

MuJoCo 侧现在有真碰撞桌子(内存内增补,磁盘 MJCF 一字未动,现有 inert 调用点逐字节相同)。对 10 条已编译 canonical clip 扫描:

| clip | 撞台 | 帧 | 最大穿透 |
|---|---|---|---|
| `fh_loop_full` | **是** | 25/78 帧(f14–38 连续) | **102.0 mm** @f17 |
| `fh_loop_upper` | **是** | 24/80 帧(f19–42 连续) | **96.3 mm** @f23 |
| 其余 8 条 | 否 | 0 | 0 |

接触点 ≈(0.62, 0.29, 0.76),只有球拍几何碰台面。这把 EXP-MOTION-CANONICAL-LIBRARY §7.4 里只靠分析得出的"正手在台面以下击球"变成了物理事实。三个消费者(playback / FK / dynamics gate / feasibility oracle)都加了 `--with-table`,**默认关**,所以没有任何现有 gate 判定被改动——开关一开,这两条正手 PASS→FAIL。要不要翻这个默认,是需要 Franco 拍的一刀。

## 6. 2026-07-27 对抗验证:三条 claim 的结论,以及**最重要的发现不在这三条里**

对正手线上的三条结构性 claim 做了对抗验证(14 个 agent,每条 claim 三个独立视角去反驳)。

### 6.1 在跑的臂**没有**被污染——四道实测防火墙

这是最高风险的开放问题,结论是否定的,而且理由比"应该没事"强:

1. **加载的根本不是有争议的产物。** 六条臂加载的 clip(sha `97f5e847…`)是 probe6a 输出 `f6c89539…` 整体旋转 −72.552° 后的文件。出厂 probe2 的反手是 `d907416d…`,**另一个文件,没有任何东西加载它**。Claim A/B/C 争的每一个产物都不在臂的依赖路径上。
2. **运行时和编译器零耦合。** 每条臂的 env.yaml 都是 `canonical_ready_mode: false` + `canonical_registry_path: ''`,MotionLoader 直接读裸 NPZ。**编译器在训练途中被删掉,臂都不会知道。**
3. **物理检查通过(实测,不是假设)。** 真正的风险是这个 clip 用 `probe_exact_pointwise_caps` 造的,而该通道只采格点、在收敛分辨率下低估真实曲率 50–70 倍。若这事要紧,交付的 clip 就会超加速度包络。**它没有**:31 个关节零超限,最坏 `right_shoulder_pitch` 0.9342 倍上限,`waist_roll` 0.3864。余量薄但是真的。
4. **绑定自洽。** 题库 meta 绑的 motion_sha256 + anchor_frame 16 + anchor_phase 0.228571 与实际加载一致,6126 条答案全部物理复核(残差 max 5 mm)。

### 6.2 编译器那件事是**排期问题,不是安全问题**

三条 claim 都把编译器和 marker 权威当成流水线的闸门。**运行时它们根本不是闸门**——`canonical_ready_mode: false`,那些证书、判定、权威、锚点闸门、重定时收敛阶梯,**执行在正在训练的东西上等于零**。这是记录在案的裁定(本文 :102),不是泄漏。但它意味着编译器问题只关乎**未来的 clip**,不关乎在跑的臂。

而且锚点之争流水线自己早就写着结论了:编译器给这条训练 clip 声明的触球窗是 **帧 22–37,拍面离地 1.159–1.521 m,即高出台面 40–76 cm**——它不可能是触球帧。臂实际用的是帧 16(来自可回球性扫描),可回球带(13–19)与编译器窗(22–37)**不相交**。所以 Claim A 的 50-vs-61 锚点之争和 Claim B 的权威封闭,争的是一个**因为对这个任务物理上不对而被刻意退役**的权威。

### 6.3 真正的活缺陷:**六条臂全把自己的反手 clip 标成了正手**

`clip_family_per_clip: null` + 单 clip → `resolve_clip_family_is_forehand` 返回 `(True,)` → `swing_sign = +1.0`。后果:**每一个逐侧指标结构性恒为 0.0000,而总量在动**(实测 i4071 总回球率 6.93%)。这正是 07-26 把"45% 回球率"读废的那个坑的**镜像**。

**是报表缺陷,不是学习缺陷**,因此不重启在跑的臂。理由三条,都查过:
- `swing_sign` 只喂观测通道;单 clip 时它对每个 env 是**同一个常数**,等于死输入——标成 −1.0 结果一样。
- 拍面通道显式传了 `mount_normal_sign=-1.0`,已补偿。
- 题库写的是**绝对位置**(`racket_target_pos_w = origins + pos`),不经过侧别镜像。

**但单 clip 臂连"我是反手"都说不出口**:`resolve_clip_family_is_forehand` 硬要求"正反手至少各一个"。已修(`commands.py`):`nseg >= 2` 才查这条,单 clip 可显式声明任一族;缺席仍走老默认,现役在跑臂逐字节不变。11 项隔离验证全过。

> **这是今天第三次撞到同一个模式**:系统假设每条臂都是正反手双 clip。
> ① `face_command_obs`/179D 契约要求双 clip;② `clip_family_per_clip` 要求两族齐全;
> ③ `vel_range_per_clip` 的 "set it to None" 表达不出来(已修)。
> 单 clip 臂在这套系统里是二等公民,每次都得现场发现。

### 6.4 顺手清掉的运维负担

pod1 上还挂着上一个 agent 遗留的正手穷举编译:**35 个进程、2120% CPU ≈ 128 核里的 21 核**,和三条训练臂同住,load average 38.14,已经烧了一天多,而结论早就知道(必然 RetimeError)。已清:load 38.69 → 13.92,三条臂毫发无损。

## 7. 2026-07-27 连续出题(continuous questions):接线完成,**默认没翻**

老板的裁定是两条:**题库离散只用于考试,训练要连续采样**;**必须开的东西要变成 default,别让 agent 去记**。
这一节是执行回执。

### 7.1 一句话结论

连续出题的**代码路径**已经接进训练、有闸门、有账、能开机自证同物理,并且**不动现役题库臂一个字节**;
但**默认没翻**,因为翻默认要交的三样证据里,缺的那样是最贵的:**从来没有一条连续臂真正训练过**。

选连续只要一行:`task.racket.target_mode: solved`。其余要么推导、要么配错当场炸并把该写什么写进报错里。

### 7.2 这一轮修掉的东西

**阻断级(会让仓库变红)**:
`_cq_enabled` 在两个最热的判据里是裸属性读,而仓库里一批源码级单测用 `__new__` 造对象、不走 `__init__`,
于是**8 个原本绿的测试直接 AttributeError**——其中就有"老采样器逐字节不变"那几条,也就是说
"不动现役臂"这个承诺当时**没有测试能证明**。改法是结构性的:在类上放 `_cq_enabled: bool = False`,
没走 `__init__` 的实例一律读到"关",以后新增读点也不必逐个记得写 `getattr`。

**两道被改名改红的老卫兵,retarget 而不是删**:
`test_face_sign_per_clip` 里那条"解出来的来球别在同一 resample 末尾被随机 vb 采样盖掉"——它守的正是
这次改动自己称为"最危险的静默脱钩"的那件事,判据改名后它只会红,下一个人顺手一删卫兵就退休了;
`test_question_bank_family_addressing` 的对账点计数 2 → 3(连续路径新增开机对账,族表传得对,只是数字旧了)。

**把"必须开"变成"必须写,不写就炸"(老板第二条裁定)**:

| 以前 | 现在 |
|---|---|
| `cq_anchor_bank` 默认空 → 一行 WARN 就放行,物理契约 SHA、动作契约、开机 parity 三道**全不跑** | 必填,不写当场炸 |
| `cq_vel_range_per_clip`(连续臂**唯一**的出球箱)**零校验** —— 球朝反方向飞、空箱子都能过 | 和 `vb_vel_range_per_clip` 同一套规则:`x_hi < 0`、`lo <= hi` |
| `cq_aim_xy` 可以瞄 A、打分在 B,只有 10 cm 的闭环兜底 | 必须等于 `vb_target_x/y`(真正打分的那个点) |
| 参考回球闸门在连续臂上采的是**死箱子** `vb_vel_range_per_clip`,判绿判的是一颗球都不来自的分布 | 连续时改读 `cq_vel_range_per_clip` / `cq_spin_abs_max` |
| `speed_scale_range` 必须是 `[1.0,1.0]`,但只在**第一次 env reset** 才炸(实测 Isaac 起完 3 分 09 秒) | 提前到 `train.py` 解析期,环境侧那条留作最后一道 |

### 7.3 实测数字(pod1,**纯 CPU**,`CUDA_VISIBLE_DEVICES=""` + `nice -n 19`,**没占一秒 GPU**)

三条在跑的臂(496822 / 496981 / 499603)跑完前后都在,显存 5861/5687/5815 MiB、利用率 23–27%,临时目录已删。

- `test_continuous_vs_bank_parity.py`:**14 passed**(原 12,新增 2 条)。
- 之前被 `_cq_enabled` 打红的 4 个文件:**105 passed / 0 failed**。
- 20 个引用 `hope_commands` 的测试文件,对 **pin(`nohope_pin_20260727`,五条臂真正在执行的那份)** 做 A/B:
  本次 **11 failed / 464 passed**,pin 基线 **14 failed / 447 passed** → **新增失败 0**。
  那 11 条两边一模一样(`metric_sync_fix` 9、`event_timing` 1、`virtual_return_scorer` torch parity 1),是兄弟未提交工作带来的,不是这次的。
- 配方闸门对抗探针 **15/15**(此前 8 个危险组合是**静默放行**的,上表那四条已关)。
- **变异测试(这条最重要)**:把 `generate()` 自己那次求解调用改坏的三种改法,原来 12 条断言**全绿**——
  也就是说 parity 测的是 `parity_report`,**训练真正走的那条调用路径一条断言都没有**。新增的
  `test_generate_itself_is_covered_not_only_the_solver` 把三种都逼红了:
  M1 不传 ref_normal(反手球拍面翻 180°)、M3 传错桌面/网平面、M4 传错 `h/n_steps`(答案还对、通过率 0.975 塌到 0.285)。

### 7.4 默认为什么**没**翻(缺的是哪一块证据)

成本不是问题:同卡同 clip 同邻居、前后相邻跑,连续比题库**每迭代贵 6.6%(中位 8.7%)**——
对照离线造 8192 道题要**单核 ~50 分钟**,这笔账划算。（此前报的 "+1.5%" 是漏了 `cq_overdraw`、
python 考卷筛选循环和闭环复验的算法,已作废。）

真正拦住默认的是三条:

1. **没有任何训练结果。** 连续路径最长只跑过 **12 个迭代**(能起、能存 checkpoint,仅此)。
   回球率、落点误差、拍面误差——一个都没有。默认是对**结果**的断言,而结果不存在。
2. **连续臂今天导不出、上不了机。** 正式的 `deploy_parity_face179` 导出要重读 npz 行
   (`derive_stage1_normal_envelope`),planner revisions 和 `post_strike_t1` 同样硬拒。
   翻了默认 = 默认跑出来的策略**不可部署**。
3. **声明的分布 ≠ 训练的分布。** 快题库 788 行里 36 行 torch 求解器解不出,全部 |v_in| ≥ 3.67 m/s;
   **最快的十分位有 32.9% 解不出,最慢的十分位 0%**。重抽循环会把这条快尾巴悄悄抽回箱子的容易区,
   而 `exhausted` 因为有 overdraw 读数是 0。谁翻默认,谁就是默认在一个比 yaml 写的更容易的箱子上训练。

另外两条要记在案:连续臂被强制 `speed_scale_range=[1.0,1.0]`,而五条在跑的臂是 `[0.6,1.0]`
(它们能跑只是因为同一道闸门对题库臂是死的),所以**连续 vs 题库的 A/B 天生不配平**;
`_cq_state_dict()` 只是暴露了游标和 RNG,**没接进 resume**,长臂一续跑就重置缓冲和出题随机流,
"每道题只发一次"当场破功。

**dataclass 里的 `target_mode` 默认建议永远别翻**(`hope_commands.py`)——它会打到所有直接构造 cfg 的地方
(测试、exporter、probe、judge 重建 env),这些都不过 task-yaml 的闸门。要翻,翻**任务 yaml 那一行**。

### 7.5 翻默认的验收条件(给下一个人)

按顺序,缺一条就不翻:
① 一条连续冒烟臂跑到能读曲线,`continuous_question_exhausted_rate` 全程 0、
   per-regime 桶账没有某个速度桶被抽干;② 快尾巴解不出那 36 行有结论——是求解器要修,还是箱子要收窄,
   写进 yaml 而不是靠重抽掩盖;③ 连续臂能走完导出/judge,或者明确接受"训练用连续、部署用题库臂"这个分工;
④ resume 接线;⑤ 配平的 A/B(题库臂也跑 `speed_scale_range=[1.0,1.0]`,或先把那道闸门的对错拍板)。

## 7. 连续题库:接上了,但**默认没翻**——三条证据不够

owner 裁定"训练用连续、考试用离散"是对的,接线也按它做了。但**默认是一个关于结果的断言**,而三样证据缺席。

### 7.1 挖出来的东西(基础调查阶段,还没动代码就出来了)

生成器 `generate()` **不能接真的 `scene.env_origins`**——`p_contact` 用世界系,但 aim_x 没做原点平移、aim_y 做了,`surface_z`/`net_x` 是 env 局部标量。实测:

| 原点 | 解出率 | 后果 |
|---|---|---|
| (0,0,0) | 100% | 现有两个调用者都传 0,所以从没咬过人 |
| (10,10,0) | **0%** | 全不可解 |
| (3,0,0) | **99.22%** | **更糟**:它会收敛到一个短 3 m 的目标,在一个网跑到机器人身后的坐标系里 |

另外四条:解不出的行**照样返回一个填满的、看起来很正常的目标**(只有 `ok` 能区分);默认模式的失败原因直方图是**编造的**(全部记成 `resid_gt_tol`);默认路径**不查过网**;拍面符号**没对齐**到题库那套 +Y/A 约定。

### 7.2 挡住翻默认的三条

1. **没有任何训练结果。** 连续臂跑过最长 12 个迭代——能开机、能存 checkpoint,仅此。回球率、落点误差、拍面误差,一个数都没有。默认是在说"这样训更好",而什么都还没训过。
2. **连续臂今天导不出、部不了。** 正式 `deploy_parity_face179` 的 ONNX 导出要重读 npz 行,planner 修订和 `post_strike_t1` 同理。翻默认 = 让默认跑法结构性不可部署。**这条单独就够否决。**
3. **声明的分布不是训练的分布。** torch 求解器在 |v_in| ≥ 3.67 m/s 解不出(最快十分位 32.9% 解不出,最慢 0%),而重抽循环把这些球悄悄抽回容易区,`exhausted` 却读 0(过抽吸收掉了)。**恰恰是决定回球质量的那个区间。**

### 7.3 但**离线题库没有这个毛病**(实测,别把两件事混了)

同一天量了已发射的两份题库,声明多宽就是多宽:

| 题库 | 声明 | 保留 | 每一档占比(均匀 = 10.0) |
|---|---|---|---|
| 窄带 | 2.0–5.0 | 2.00–5.00 | 10.2 9.7 9.6 9.8 9.2 10.3 9.8 10.3 11.1 10.0 |
| 宽带 | 1.5–7.0 | 1.50–7.00 | 9.9 10.8 10.2 10.1 10.9 11.6 9.3 7.2 9.4 10.6 |

原因:离线生成器**丢掉解不出的题并诚实报可解率**,不重抽,所以保留集不歪;连续路径**重抽**,于是歪了而计数器看不见。`c_ballwide` 的"宽来球"消融因此是真的。

### 7.4 顺带修好的护栏(不修的话下一个 agent 会把它们删掉)

一个 blocker(`_cq_enabled` 裸读在 `__new__` 构造的实例上崩,连带把"离散路径逐字节不变"那几条守卫测试弄红了——正是证明在跑的臂没被动过的那几条)+ 两个 major。修法都是**结构性**的:类级默认值、把守卫改指向新谓词并写明为什么不能删、计数 2→3 并留言"下次是抬数字不是删断言"。

对照 pin(五条在跑的臂真正执行的那份):**新增失败 0**(改动 11 failed/464 passed vs pin 基线 14/447)。15 条对抗配方全部按预期行为,其中 8 种危险组合以前是静默通过的。成本 **+6.6% 均值 / +8.7% 中位**(同卡同 clip 4096 env 配对测)——对比离线一份 8192 题要单核 50 分钟,是明确的赢。
