# Reward 全量盘点(按目的分组)— 2026-07-25【历史快照:v1】

> 本文是 v1(reward_pack=v1)时代的现状盘点,用于追溯与对照。**v2.2 已是默认包**,
> 现行说明见 [reward_v2_explained_20260725.md](reward_v2_explained_20260725.md)。

**范围**:main @6b43aa6b,现役训练谱系 = `HOPEPingPongVirtualBall`(0720–0723 全部波次都跑它)。74 项逐条读码盘点(3 个独立 agent 分域穷举 + 1 个交叉批判),行号证据在各项注明的源文件里。
**记账口径**:IsaacLab RewardManager 每步计 `value × weight × dt`(dt=0.02 s @50 Hz);weight=0 的项被整个跳过(字节等价基线)。
**目的标签**:模仿(mimic)/ 击球(hit)/ 上台(table)/ 平衡与恢复(balance/recovery)/ 平滑(smooth)/ 安全(safety)。一项可多标签;下面按主标签归组。

---

## 0. 一张图看懂收入结构

- **模仿组**(六项 motion_*)基础上限 ≈ **5.0/步**;触球窗内被 `motion_scale_in_window=0.25` 整体打折 → 窗内模仿上限 ≈ 1.25。
- **击球组**(racket_position 17 或 7 + velocity 7 或 17 + normal 5 + strike_success 5 + progress 10)在触球窗内**决定性压过模仿**——这是设计出来的:窗外老师说了算,窗内题目说了算。
- **上台组**(virtual_pass_net 20 / landing 30 / spin 5)是**每挥拍一次性**结算(exact-strike 一步),不参与逐步梯度竞争,但主导每拍记账。
- **软惩罚组**(action_rate -0.2、action_acc -0.05、face 条件税 -0.4、脚部若干、qbar -0.65)全部有预算表管着(IU 波 1/3 比值守卫)。
- **W/V 配方轴**:W = pos 17/vel 7(稳),V = pos 7/vel 17(快)。对应记忆里"W稳但不回球/V摔但回球好"。

---

## 1. 模仿组(mimic)——"老师"

| 项 | 人话 | 权重(现役) | 何时生效 | 阶段备注 |
|---|---|---|---|---|
| motion_global_anchor_pos | 躯干钉在参考世界位置 | **已删**(打球谱系全删) | — | 钉底座打架步法,删除而非调权——模仿vs击球冲突的最早解法 |
| motion_global_anchor_ori | 躯干朝向贴参考 | 0.5 | 全程;窗内×0.25 | std 0.4 rad,歪超 ~1.2 rad 无梯度(但那时早被终止) |
| motion_body_pos | 上半身(躯干+双臂)相对位置贴老师 | 1.0 | **挥拍段专属**(hold 归零);窗内×0.25 | 早期主收入;fresh 臂起步姿态在 0.3 m 带外时梯度弱 |
| motion_body_ori | 上半身朝向贴老师 | 1.0 | 同上 | **拍腕已被摘出**(free_wrist_ori_mimic,全波)——拍面归 racket_normal 管,消"二主"冲突 |
| motion_body_lin_vel | 身体件线速度贴老师 | 1.0 | 现役谱系**hold 里也付**(参考速度=0,变相奖静止) | 拍腕已摘(free_wrist_vel_mimic);题库答案速度与老师锥中位差 34°,不摘必打架 |
| motion_body_ang_vel | 身体件角速度贴老师 | 1.0 | 同上 | 核最宽(std π),早期不死、晚期弱信息 |
| lower_body_pose_imitation | 12 腿关节贴参考(B1 假设) | **0**(探针常开) | 支撑窗(击前0.3s/击后0.4s) | 与 B2(参考无关稳)互斥假设,只 A/B 不默认 |
| foot_orientation | 髋yaw/roll+踝roll 贴参考脚法(L1 罚) | -0.3(W 配方) | 全程(hold 用站立参考) | 模仿家族唯一罚项;L1 无死区不饱和;治"怪脚"(hip_yaw p95 ±0.94 vs 参考 ±0.41) |
| hold_ready | hold 里站稳+踩实+够得着→发钱 | +2.0(名义) | in_hold + reach 门 | **现役波 hold_steps_range=[0,0] → 实际哑火**;racket 模式 reach 门可被手臂姿态操纵(已知未修) |
| 三把"放人"开关 | 拍腕朝向/拍腕速度/非持拍左臂退出模仿名单 | 全波开前两把;第三把=W配方 | 配置期 | 每一把都是拆一个"第二主人";W/V 之别就是左臂放不放 |
| adaptive_sigma(P2.3) | 拍位/拍速核宽每500步按实测误差自动收紧 | 全波开 | exact-strike 样本够了才动 | **只收 pos/vel/strike_success,不收 normal——face 相对降权最多 ~7×(已核实,见 §5)** |
| motion_scale_in_window | 触球窗内模仿整体×0.25 | 全波 | 宽窗 ±0.12 s | 窗口时钟归 clip 管,不可被策略操纵 |

**组内要点**:模仿的塑形哲学是"窗外教风格、窗内让位给击球",实现靠三层:摘人(免拍腕/免左臂)、打折(窗内0.25)、删项(锚点位置)。恢复段(hold)没有老师(挥拍段专属包装器把 hold 归零),设计上由 hold_ready/settle 系补——但现役波 hold 被关,恢复塑形实际全靠 settle-debt/slew 探针系(见 §4)。

## 2. 击球组(hit)——"把拍子按时按点按面送到球上"

| 项 | 人话 | 权重 | 何时生效 | 阶段备注 |
|---|---|---|---|---|
| racket_position | 拍心贴"穿击线"(目标点−目标速×tts) | **17(W)/7(V)**,std 0.2→0.075 自适应 | 紧窗 ±0.12 s | 窗外零梯度——接近全靠 progress;swing-through 形式暗付时机分(static 消融项存在但没波开过) |
| racket_velocity | 拍速矢量贴目标出拍速度 | **7(W)/17(V)**,std 1.0→0.5 自适应 | 宽窗 | 早期核宽容忍半速挥;R3b 曾发生"刷速度面子不接触"病,靠 strike_success 乘法形式治 |
| racket_normal | 拍面法向贴题库要求(shared_plus_y 口径) | 5(全配方一致),std 0.3 **固定** | 宽窗 | **3σ≈52° 外零梯度**(M3c 死区);唯一不进自适应 sigma 的核 |
| racket_strike_success | 位置×速度×拍面三核**相乘**,真打上才有 | 5(冻结,无 CLI 键) | 传统窗 | 早期≈0,纯精修钱;乘法防偏科刷分 |
| racket_progress | 击前拍到目标每靠近1cm发1cm的钱 | 10(冻结) | pre_strike | **窗外唯一稠密击球梯度**,从第0步就活;伸缩式记账不可刷震荡 |
| racket_face_conditional_guidance | 窗内固定预算的拍面税:没准备好恒扣、准备好了按面误差返还 | **-0.4 全波**(penlight 臂 -0.13) | 宽窗 | face 死区的现役解法;设计上不可"躺平逃税";Franco 关注的软罚压模仿风险由 penlight 剂量臂检验 |
| racket_guidance / racket_face_guidance | 线性距离/角度"往哪挥工资单" | **0(全波显式关)** | — | 历史死区解药,被 progress+条件税取代 |
| base_position | 击前底座贴指令站位 | **谱系已删**(base-free) | — | 只活在 jiayi 侧 Hitter 谱系;删因:钉底座致滑步侧倾 |

**组内要点**:击球采集链 = progress(窗外稠密)→ pos/vel 宽核(窗内,自适应变窄)→ strike_success(乘法精修)→ virtual_*(上台结算)。**唯一没有课程的通道是拍面**:normal 核固定 std、不进 sigma 白名单、死区外只有条件税撑着(而条件税"没准备好"时对 face 也无梯度——这是故意的,but 见 §5 缺口)。

## 3. 上台组(table)——"球真的过网落对面"

| 项 | 人话 | 权重 | 何时生效 | 阶段备注 |
|---|---|---|---|---|
| virtual_pass_net | 过网瞬间贴"网上12cm"给钱 + 合法过网半分 | 20 | vb_fired(捕获门)一拍一次 | 核故意**不**用合法落台门控(07-03 饿死事故:0.2% 合法率下颗粒无收) |
| virtual_landing | 落点贴对面半台中心 + 全合法(过网+落台+进深0.3m)满分 | 30(**全栈最大单笔**) | 同上 | σ 放宽 0.3→1.0 爬坡期;"吸血小球"由进深门挡 |
| virtual_spin | minimize 档:合法球转速越小越给钱 | 5 | 三重门(捕获+合法) | 最晚激活的项;spin 轴公认低验证 |

**捕获门 vb_fired** = exact-strike 一步 & 拍面正确半球 & 位置误差<9.5cm & 沿法向进拍速>0.3m/s——所以上台组的瓶颈**就是**击球组训练的那几个量,链路自洽。physical_ball/shadow_ball 是纯度量仪器,**没有任何 reward 读它们**(真球落台奖励在 main 上不存在)。

## 4. 平衡/恢复/平滑/安全组

**常开平衡税**(小额、全程):upright -1.0、base_ang_vel_xy -0.05、base_lin_vel_z -0.5、joint_vel -1e-4、joint_torques -3e-5。全是二次型小票,晚期在 17 权重收入面前只是 tie-breaker;设计注释明说角速度类罚与挥拍功率反相关,故意压小。

**击球窗内站稳包**(窗内集中):strike_upright -2.0、strike_ang_vel -0.5、strike_foot_vel -0.5、strike_vbob -1.0。约 12/800 步生效,打的瞬间不许歪/晃/挪/弹。

**反作弊塑形**:prestrike_upright -1.0(W)/-2.0(V)(线性无死区)、prestrike_waist_twist -1.0(不许拧腰代步)、arm_overreach -0.5(不许撑臂极限够球,有界 [0,1])。

**脚部家族**:foot_slip_sq -1.0(踩着不许滑,平方)、foot_drag -0.5(线性孪生)、foot_velocity -0.05(甩腿税)、pre_strike_foot_slip -0.4(击前踩实)、foot_soft_landing(-0.003@300N,IU)落地冲击、foot_clearance(-0.01@0.15m,仅 combo_fresh)抬脚不足——**07-25 起单侧化**(只罚抬不够,多抬免费;旧双侧在凸包顶把脚往下按,见 §5)。

**限位安全**:joint_limit -10(真实角越软限才罚,有 qdes_clamp 后近乎哑)、**qdes_limit_barrier(qbar)-0.65**(目标角贴限 8% 带即罚、SUM 不平均、全程常开——**07-25 起站姿豁免**,见 §5)、joint_velocity_limit_hinge(现役全关,被 qbar+slew hinge 取代)、arm_torque_saturation -0.5(implicit 执行器下疑似哑项,靠 arm_torque_sat_frac 指标核实)。

**恢复机制系(S/H 机制,现役多为探针)**:post_swing_settle_debt(击后 0.2–1.55s 五项"没站稳的债",有免费额+有界尾巴;S1 臂 -0.25)、processed_qdes_slew_hinge(恢复窗 15 关节部署空间 q_des 限速 85% 免费带;H 臂 -0.25/-1.0)、lower_body_stability_bundle(B2:站宽塌陷+腿速尾巴;S2 臂 -0.25)、base_decel(PACE 减速剖面,现役全关)。**有界尾巴 + same-attempt 时钟**是这组的两大护栏:深度失败饱和不炸梯度,击球前死掉的样本恒零不污染。

**事件(不是 reward)**:push_robot ±0.35 m/s @5-15s + force_push 68N×0.3s(IU 三臂并存)——制造平衡组要打分的状态。

**终止(也是塑形)**:time_out 16s(截断,bootstrap,无逃债扭曲);anchor_pos/anchor_ori/ee_body_pos(0.25m z-only 参考包络,hold 豁免——豁免是防"出生即死",不是漏洞);base_fell_tilt 0.7rad(硬摔线);base_too_low 0.5m(塌陷线,和 settle 根高债形成 [0.5,1.0184] 有界带)。**逃债分析**:所有负债项有界且小(≤0.013/步),挥拍收入大一个量级,"死掉躲债"永不划算——这是负债类项没被死亡截断玩坏的结构性原因(对照 yikang r 系 settle_debt=-0.25 被截断的教训:他们的债不带 same-attempt 时钟)。

---

## 5. 已证实的缺口与本日修复

1. **qbar 站姿在罚带内(已修)**:双肩 shoulder_roll 硬限位不对称(内收硬停 -5°/外展 +150°),0.9 软限系数+0.08 罚带按全跨度等比,把 ready 外展 0.12 rad 圈进带内(d=0.0296):站姿常驻罚 0.656、梯度持续外推双肩、above_margin_joint_count 垫 2 地板、IU"常态 0"守卫假设破产。**修复**:罚带按关节收窄到默认站姿之外(eps 0.005,带宽下限 0.005 fail-loud);29 关节数学逐字节不变;合同 formula 串同步更换(旧 sidecar 不可静默续训);队列脚本两处"常态 0"表述更正。
2. **foot_clearance 双侧且世界 z(已修单侧)**:文档自述"最低离地要求",实现却是 |·| 双侧——凸包顶主动压脚,恰与 combo_fresh"逼抬腿"目的相反。**修复**:单侧 clamp(target−foot_z, 0);粗糙地形 target 需按 hi+0.07+期望净空调高(写进 docstring;冻结的 IU prereg YAML 未动)。
3. **adaptive sigma 白名单缺 normal(未修,记账)**:pos/vel 核收紧至多 7×/4×,normal 固定——训练越到后期 face 通道相对越弱,而 face 恰是历史事故重灾区(M3c 33° 平台、反手 53°)。可选解:把 normal 纳入自适应(需自己的误差 EMA)或给条件税加期末加严档。**yikang r8 的教训指向另一个方向:face 卡平台的根因可能是模仿合同冲突而非 reward 弱**——main 已摘拍腕模仿,但若 canonical 臂再现 face 平台,先查合同再调权。
4. **hold_ready 实际哑火(记账)**:现役波 hold 全关,恢复正信号缺位,靠 S/H 探针系撑;若未来开 rally/连拍,记得 hold_ready 的 racket-模式 reach 门可被手臂操纵(station 模式是修法)。
5. **arm_torque_saturation 疑似哑项(记账)**:implicit 执行器不填 computed_torque;看 arm_torque_sat_frac 是否恒 0,恒 0 即无效项。
6. **D5 静音指标全生态缺失(待做)**:无任何逐关节 Δq_des 高频功率指标进评测 JSON;CGF/IU 的 chatter 臂跑完无法证明"更安静"。

## 6. 交叉批判(critic agent,已独立复核头条)

### C1【头条|已亲核代码】planner 波的"触球窗"覆盖整个随挥段

机制(全部 HEAD 行号,已人工复核):planner revision 开启时(0716 起**所有**现役波),`time_to_strike = _planner_truth_tts.clamp(min=0.0)`,而 truth tts 递减后 `clamp(min=0.0)` **钉死在 0 直到 clip 收尾**(commands.py:1252-1254;hope_commands.py:3086-3089 注释原文:"reaches zero at contact and remains zero through follow-through");`strike_window = |tts| <= 0.12`(hope_commands.py:3102)→ 触球后窗恒开。设计上 ±0.12 s = 13 步的窗,实际 = 随挥段全程 ≈50–100 步(正手 53%/反手 66% clip 时长)。

**后果**(每个现役臂都中):
- **L1 停拍薅钱**:触球后把拍停在击球点可持续赚 racket_position 17 + normal 5,上限 ≈ +31/拍,是诚实 13 步窗收入(≈5.5)的 ~6 倍——梯度直接奖励"不挥穿"。strike_success(速度因子归零)和 virtual_*(一拍一次锁存)自保,只有窗控稠密项被吹开。
- 击球窗站稳包(-2/-0.5/-0.5/-1)整个随挥段全程计费;strike_foot_vel 恰好在 S1 settle 窗(0.2–1.55 s)视为免费的恢复步伐上收税——同一臂内 reward 打 reward。
- face 条件税实际花 ≈0.56/拍,是预算表"0.4×窗步数"的 ~5 倍。
- 模仿×0.25 静音贯穿整个随挥段——恢复段唯一的老师恰在恢复段被捂嘴。

**免费检测(不用新跑)**:看现有 tensorboard 的 `strike_window_hit_rate`——窗诚实应读 ~0.05–0.08,C1 下会读 ~0.3–0.5。
**修法候选**(reward 侧,不动观测):exact_strike 后锁存关窗(锁存机制已存在,hope_commands.py:4512-4519),或窗掩码改走 legacy 有符号 clip tts(代码明确还在算,hope_commands.py:3064-3077)。
**可比性**:0716 后所有波内部 A/B 仍自洽(同病);绝对语义、跨 planner 采用日的对比、以及预算表全部失真。**修不修、何时修是 lineage 决策,挂 Franco。**

### 早期阶段:combo_fresh 从零起步的"自杀区间"(S3+F7)

从零初始化时逐步收入算术:模仿上限 ≈ +0.10/步;action_rate -0.2 在随机策略下 E‖Δa‖²≈62 → 约 **-0.25/步**,再叠 action_acc/qbar 地板/脚税——**净收入起步为负,且无 alive bonus、无终止罚 → V(死)=0 > 继续,早期梯度指向摔倒**,而所有教恢复的负项恰在它该教的状态里被死亡截断。IU 的 1/3 预算守卫用"窗内击球收入"当分母,但 fresh 臂早期根本没有这份收入——**预算规则在它最该拦的区间失效**。修法方向:fresh 臂的负项预算改按 0.10/步模仿上限折算,或给 fresh 臂加 alive bonus/降起步剂量。

### 其余批判(按可信度排)

- **F5**:模仿核是 body-均值,`free_non_striking_arm` 摘掉左臂三件**不是**减收入,而是把同权重核变容易(均值里少了误差贡献)——W vs V 实际差了两个轴(pos/vel 权重 + 模仿难度),不是配方名义上的单轴。
- **F4**:安全家族聚合口径不一致——qbar 按 Franco 裁定用 SUM,但 qdot hinge mean/31、slew hinge mean/15,单关节尖叫被稀释 31×/15×;若按历史剂量(-5.0)复活 qdot hinge,实际每关节剂量与 qbar 语言差 31 倍。
- **S1+F8**:现役波恢复组**零正梯度**(planner 硬禁 hold → hold_ready/heading 永不付;post_strike_brake 不在现役谱系;S/H 全 0)。C1 修掉后恢复组从"被错误塑形"变成"空"——届时 settle 类正项或恢复 hold 课程是**承重件**,不是可选项。
- **L5**:<10 N 贴地滑行逃掉 slip/drag 两税(都是接触门控);combo_fresh 的旧双侧 clearance 下"贴 0.15 m 世界高度滑"曾是最便宜合规解(单侧化后此项已减轻)。
- **L6**:HER 自出题 30%(achieved_target_mix)是有界但真实的自我放水通道,预算表无对应反指标。
- **S2**:face 通道对"还没 ready 的策略"全阶段零梯度(条件税的 ready 门 + exp 核死区 + 线性引导全关)——是故意的时序设计,但要知道它是真死区。
- **S7(纠正盘点)**:adaptive sigma 会在退步时重新变宽(每500步 clamp(EMA), 不是只收不放)——盘点里"回退冻结"的担忧不成立;真实性质是"挥空期间 sigma 钉在最宽",行为良性。
- **L7(验证过的非漏洞)**:racket_progress 不可震荡/传送薅;landing 核最大值在目标点(钻网必亏);undesired_contacts 的 A3 大小写 regex 已正确重钉。
