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
