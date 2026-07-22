# mjlab velocity 任务 reward 全量取舍讨论(2026-07-22)

结论先说:mjlab(mujocolab/mjlab,MuJoCo-Warp 上的 Isaac-Lab 风格训练库)velocity 任务全部
13 条 reward + 通用库 11 条 + 3 条终止 + 6 条事件逐项对完,按我们的任务(阶段1=固定站位单拍
回球+保平衡,阶段2=变到达+脚步移动,阶段3=物理球)裁成四档:**现在就要 3 项、阶段2再要
4 项、已有等价 11 项、不要 5 项**。最大的坑:mjlab 的脚部罚(落地冲击/滑动/抬脚)全部挂着
"速度指令门"——速度指令低于阈值(=站立)时罚项自动清零。我们全程都是"站立击球",照抄等于
全部失效,**必须拆门常开**;并行作业正在实现的 `task.rewards.foot_soft_landing_weight` /
`task.rewards.foot_clearance_weight` 就是拆门版,本文档与这两个键名对齐。

来源(2026-07-22 抓取 github mujocolab/mjlab main 分支):

- `src/mjlab/tasks/velocity/velocity_env_cfg.py`(任务装配:权重、门参数、DR、终止、curriculum)
- `src/mjlab/tasks/velocity/mdp/rewards.py`(velocity 专属 reward 函数)
- `src/mjlab/envs/mdp/rewards.py`(通用 reward 函数库)

我们侧对照(main 现役):

- `hope_training/.../tasks/tracking/config/agibot_a3/hope_env_cfg.py`
  (`HOPERewardsCfg` 350 行起、`HOPEDeployParityRewardsCfg` 690 行起、`HOPEVirtualBallRewardsCfg` 965 行起)
- `hope_training/.../tasks/tracking/tracking_env_cfg.py:204`(基类 `RewardsCfg` 通用正则项、`EventCfg` DR)
- `configs/phase1_intel_wave_20260721.yaml`(现役权重覆盖:racket 17/7 或 7/17、action_rate -0.1、
  racket_face_conditional_guidance -0.4、三稳定机制 weight 0 等)

---

## 1. mjlab velocity 任务全清单(逐项:人话 → 公式要点 → 默认权重)

### 1.1 挂在任务上的 reward(13 条)

| mjlab 项 | 人话 | 公式要点 | 默认权重 |
|---|---|---|---|
| `track_linear_velocity` | 底座线速度跟指令,主任务奖 | exp(−(xy 误差²+z 速度²)/0.25),body 系 | **+2.0** |
| `track_angular_velocity` | 底座角速度跟指令,主任务奖 | exp(−(yaw 误差²+rp 角速度²)/0.5) | **+2.0** |
| `upright` | 站得正,有奖 | exp(−‖投影重力 xy‖²/0.2);可选按地形法线算 | **+1.0** |
| `pose`(`variable_posture`) | 姿态贴默认位,**按速度分三档容忍度**:站立最严、走路中等、跑步最松 | exp(−mean(Δq²/std²)),std 按指令速度选 std_standing/walking/running,阈值 0.05 / 1.5 m/s | **+1.0**(std 逐机器人配) |
| `body_ang_vel` | 躯干别乱晃(roll/pitch 角速度罚) | Σ ω_xy²(世界系,指定 body) | 0.0(逐机器人开) |
| `angular_momentum` | 全身角动量罚,逼出自然摆臂 | ‖L‖²(whole-body angmom 传感器) | 0.0(逐机器人开) |
| `dof_pos_limits` | 关节别顶限位 | Σ 超出软限位的量(线性,不平方) | **−1.0** |
| `action_rate_l2` | 动作别抖(一阶差分) | Σ(aₜ−aₜ₋₁)²,罚的是**原始策略输出** | **−0.1** |
| `air_time` | 摆动腿滞空时间落在合理区间给奖 | Σ 1[0.05<t_air<0.5];**指令<0.5 时清零** | 0.0(逐机器人开) |
| `foot_clearance` | 抬脚别贴地蹭:离目标抬脚高度越远、脚移得越快罚越重 | Σ \|h−0.1\|·‖v_foot_xy‖(地形高度传感器);**指令<0.05 时清零** | **−2.0** |
| `foot_swing_height` | 每步落地时结算"这步摆到的峰值高度"离 0.1 m 目标差多少 | Σ(h_peak/0.1−1)²·1[首次触地];**指令门同上** | **−0.25** |
| `foot_slip` | 触地的脚别打滑 | Σ 接触·‖v_foot_xy‖²;**指令<0.05 时清零** | **−0.1** |
| `soft_landing` | 落地要轻:罚首次触地那一步的接触力 | Σ ‖F_contact‖·1[首次触地];**指令<0.05 时清零** | **−1e-5**(力是几百 N 量纲,故权重极小) |

**注意 1(速度指令门)**:表中加粗"指令<阈值时清零"的五条,门的写法都是
`(‖v_cmd_xy‖+|ω_cmd|) > command_threshold` 才生效。mjlab 的意图是"站立环境脚不动,罚了没意义";
我们的任务里**站立正是击球工况**,落地冲击、脚滑恰恰发生在挥拍反作用力下,所以拆门是采纳前提,
不是可选项。

**注意 2(没有 stand_still 项)**:mjlab velocity **没有**独立的 stand_still 奖。"站好"由两件事
承担:指令采样里 10% 环境速度指令为零(`rel_standing_envs=0.1`)+ `variable_posture` 的
std_standing 档(站立时姿态核最严)。"站立=姿态核+采样分布的产物,不设专项罚"是它的设计哲学。

**注意 3(velocity 任务没挂 torque/qacc 罚)**:通用库里有 `joint_torques_l2`/`joint_acc_l2`,
但 velocity 任务默认一条都没挂——平滑几乎全靠 action_rate −0.1(与我们同值)撑着。

### 1.2 通用库里有、velocity 任务没挂的(11 条)

| mjlab 库函数 | 人话 | 公式要点 |
|---|---|---|
| `is_alive` / `is_terminated` | 活着给钱 / 非超时死亡扣钱 | 0/1 指示 |
| `joint_torques_l2` | 力矩能耗罚 | Σ τ² |
| `joint_vel_l2` | 关节速度罚 | Σ q̇² |
| `joint_acc_l2` | 关节加速度罚 | Σ q̈² |
| `action_acc_l2` | **动作二阶差分罚(抖动的"加速度")** | Σ(aₜ−2aₜ₋₁+aₜ₋₂)² |
| `joint_pos_limits` | 同 1.1 `dof_pos_limits` | 线性超限量 |
| `posture` | 固定 std 版姿态核 | exp(−mean(Δq²/std²)) |
| `electrical_power_cost` | **只罚正机械功**(τ·q̇ 截负),贴电机发热 | Σ max(τ·q̇, 0) |
| `flat_orientation_l2` | 站不正罚(L2 版,upright 的负罚孪生) | Σ 投影重力 xy² |
| `self_collision_cost`(velocity mdp 内) | 自碰计数罚 | 力史>10 N 的子步计数 |

### 1.3 终止(3 条)

| mjlab 项 | 人话 | 参数 |
|---|---|---|
| `time_out` | 到点收工 | episode 20 s |
| `fell_over` | 倒了判死 | 倾角>70° |
| `out_of_terrain_bounds` | 走出地形算超时 | time_out=True(不当失败罚) |

### 1.4 事件 / 域随机(6 条)+ curriculum(2 条)

| mjlab 项 | 人话 | 参数 |
|---|---|---|
| `reset_base` | 出生点/朝向打散 | xy±0.5 m,yaw±π,z 0.01–0.05 |
| `reset_robot_joints` | 关节从默认位起 | 偏移 (0,0) |
| `push_robot` | 每 1–3 s 推一把(六轴) | v_xy±0.5、v_z±0.4 m/s,rp±0.52、yaw±0.78 rad/s |
| `foot_friction` | **只随机脚的摩擦**,双脚共享同一抽样 | 0.3–1.2,startup |
| `encoder_bias` | 编码器零位偏置 | ±0.015 rad,startup |
| `base_com` | 底座质心偏移 | xy±0.025、z±0.03 m |
| curriculum `terrain_levels` | 走得好就上难地形 | — |
| curriculum `command_vel` | 分阶段提速度指令上限 | 5k/10k 步两次提档 |

---

## 2. 逐项对应:mjlab 的 X ≈ 我们的 Y

| mjlab | 我们(main 现役) | 关系 |
|---|---|---|
| `track_linear/angular_velocity` +2.0 | 无速度指令;移动由 `racket_progress` +10 与击球任务驱动 | **语义冲突**,见档④ |
| `upright` +1.0(exp 正奖) | `upright`(flat_orientation_l2)−1.0 + `prestrike_upright` −1.0/−2.0 + `strike_upright` −2.0 | 等价,我们是负罚+分相位,更重 |
| `pose`(variable_posture) | `lower_body_pose_imitation`(0→fullbody 臂 +2.0)+ 上肢模仿 4 件套 | 我们参考驱动,更强;分速度档是它独有 |
| `body_ang_vel`(默认 0) | `base_ang_vel_xy` −0.05 + `strike_ang_vel` −0.5 | 等价 |
| `angular_momentum`(默认 0) | 无 | 我们没有;常开版与挥拍冲突,见档②/④ |
| `dof_pos_limits` −1.0 | `joint_limit` −10.0(同函数);另有 q_des 侧 `qdes_limit_barrier` −0.65(qbar 臂在测) | 等价,我们 10 倍严,且双侧(实测 q + 目标 q_des) |
| `action_rate_l2` −0.1 | `action_rate_l2` −0.1(**权重逐字相同**) | 等价;chatter 波 ar02/ar05/ar10 正在扫剂量 |
| `action_acc_l2`(库) | 无 | **我们没有**,见档① |
| `air_time`(默认 0) | 无(阶段1不迈步) | 见档② |
| `foot_clearance` −2.0 | `foot_drag` −0.5(近地高速蹭地罚,无高度传感器、无目标抬脚高) | 近亲不同义;mjlab 版=并行作业 `foot_clearance_weight`,见档① |
| `foot_swing_height` −0.25 | 无 | 见档② |
| `foot_slip` −0.1(带指令门) | `foot_slip_sq` −1.0(**常开**)+ `pre_strike_foot_slip` −0.2 + `strike_foot_vel` −0.5 | 等价且我们更重更全——mjlab 反而验证了"我们拆门常开"是对的 |
| `soft_landing` −1e-5(带指令门) | 无(落地冲击完全没罚) | **我们没有**,=并行作业 `foot_soft_landing_weight`,见档① |
| `joint_torques_l2`(库,任务没挂) | `joint_torques` −1e-5 | 等价;`electrical_power_cost`(正功版)可当能耗轴后备 |
| `joint_vel_l2`(库) | `joint_vel` −1e-4;另有 `joint_velocity_limit_hinge`(0,D6 门) | 等价 |
| `joint_acc_l2`(库) | 无(由 action_rate 家族间接覆盖) | 与平滑轴重叠,优先级低 |
| `self_collision_cost`(库) | `undesired_contacts` −0.1(>1 N 非脚/腕接触) | 等价;它的"力史子步计数"更细,暂不必换 |
| `is_alive`/`is_terminated` | 无(终止本身+短窗结算承担) | 见档④ |
| `fell_over` 70° | `base_fell_tilt` 0.7 rad≈40° + `base_too_low` 0.5 m | 等价,我们更严(站立击球该更严) |
| `push_robot` 1–3 s 六轴 | Wave-P `push`/F 轴 `force_push` 开关组(默认关,push 波 5–15 s 扫过档) | 等价;mjlab 频率高 3–5 倍、带 z/rp 轴,幅度表可当加压参考 |
| `foot_friction` 0.3–1.2(只脚,共享抽样) | `physics_material` 全身 0.3–1.6/0.3–1.2 + 恢复系数 0–0.5 | 等价;"只随机脚+双脚共享"的细节与今天 grip/rough 臂同方向 |
| `encoder_bias` ±0.015 | `add_joint_default_pos` ±0.01 | 等价,量级相近 |
| `base_com` xy±0.025/z±0.03 | `base_com` torso x±0.025/y±0.05/z±0.05 | 等价 |
| (无 link mass DR) | `randomize_link_mass` ±15%、`randomize_pd_gains` ±20% | 我们更全 |
| terrain 扫描/`terrain_levels`/`out_of_terrain_bounds` | 无(平地固定场) | 见档④ |
| `command_vel` curriculum | 无 | 见档④ |

---

## 3. 四档裁决(理由全部基于我们的任务)

### 档①【现在就要】——站立击球直接受益(3 项)

1. **`soft_landing` 落地冲击罚 → `task.rewards.foot_soft_landing_weight`(并行作业实现中)。**
   人话:挥拍反作用力+回位那一下,脚砸地越狠实机越疼,现在这条完全没人管
   (`undesired_contacts` 只管"不该碰的部位",不管脚落地力)。采纳要点:(a) **拆掉速度指令门**
   ——mjlab 原版指令<0.05 就清零,我们全程"站立",照抄=永久失效;(b) **权重按力量纲定**:
   mjlab 用 −1e-5 是因为接触力几百 N 量纲,直接抄 −0.1 这类惯用值会一步天崩,起步该在
   −1e-5 ~ −1e-4 量级再看 income 记账;(c) 只结算"首次触地"那步,不罚静态支撑力,这个语义
   保留(它保证站着不动不出血)。
2. **`feet_clearance` 贴地蹭步罚 → `task.rewards.foot_clearance_weight`(并行作业实现中)。**
   人话:脚离地又没离干净、还横着蹭,是脚滑/绊倒前兆;与现役 `foot_drag`(−0.5)近亲但
   不同义——drag 罚"近地高速",clearance 罚"|抬脚高−目标|×脚速",有明确目标高度,对阶段1
   的小步调整(receiving 微调站位)同样生效。同样**必须拆门**。与 foot_drag 的重叠留给
   footrw 臂读数说话,先不动 foot_drag。
3. **`action_acc_l2` 动作二阶平滑(库函数,建议键 `task.rewards.action_acc_weight`)。**
   人话:action_rate 罚"步子迈多大",action_acc 罚"方向掉头多猛"——高频抖(chatter)恰恰是
   一阶小、二阶大的信号,对我们正在打的抖动问题是正交新轴,且不依赖移动、阶段1直接受益。
   注意排程纪律:今天 ar02/ar05/ar10 在扫一阶剂量,二阶臂**等剂量读数出来后**加在最优
   ar 档上,不与今天的波重复(见 §4)。

### 档②【阶段2再要】——跨步/移动才有意义(4 项)

1. **`air_time` 滞空区间奖**:迈步才有摆动腿;阶段1开了只会诱导原地倒腾脚。到阶段2,
   mjlab 的"门"我们反而要留——但门信号得换:我们没有速度指令,用"移动意图"
   (如 racket 目标水平距离>阈值,或底座指令速度出现后)替代 twist 门。
2. **`foot_swing_height` 摆动峰值高度结算**:同上,治拖步/绊障碍,配 air_time 一对。
3. **`variable_posture` 分速度档姿态核**:站/走/跑三档容忍度的机制本身是好东西——映射到
   我们=站位保持/移动中/急停三种工况各给一套 std。阶段1由 lower_body_pose_imitation
   (fullbody 臂)覆盖;阶段2"站→移→站"切换时它是现成模板,门信号同样换成移动意图。
4. **`angular_momentum` 罚(只要恢复窗变体)**:挥拍本身就是甩角动量,常开必与任务打架
   (见档④)。但**恢复窗**(0.2–1.55 s post-contact)里角动量残留=没卸干净的旋转动能,
   可当 `post_swing_settle_debt` 五债之外的第六债候选,阶段2步伐+急停后价值更大。

### 档③【已有等价】——列对应与权重差,标注是否值得对齐(11 项)

| mjlab(权重) | 我们(权重) | 差多少 | 值得对齐吗 |
|---|---|---|---|
| `action_rate_l2` −0.1 | 同名 −0.1 | **0** | 不动;两家独立收敛到同值,反而给 ar02/ar05(更轻)臂当先验:比 −0.1 更重大概率没益 |
| `dof_pos_limits` −1.0 | `joint_limit` −10.0 | 我们 10× | 不对齐;我们还有挥拍大幅动作,顶限位代价高,且 qbar 臂正在 q_des 侧加 barrier,等它读数 |
| `foot_slip` −0.1(带门) | `foot_slip_sq` −1.0 常开 (+pre_strike −0.2, strike −0.5) | 我们 10× 且常开 | 不对齐;mjlab 的门恰证明我们拆门是对的,权重差是任务差(击球时打滑=毁拍) |
| `upright` +1.0 exp | `upright` −1.0 L2 (+两个相位版) | 正核 vs 负罚 | 不动;可留一句:正核有界 [0,1] 不会爆,若未来出现倾斜爆罚再考虑换核 |
| `body_ang_vel`(默认 0) | `base_ang_vel_xy` −0.05 | 我们常开 | 不动 |
| `joint_vel_l2`(没挂) | `joint_vel` −1e-4 | — | 不动 |
| `joint_torques_l2`(没挂) | `joint_torques` −1e-5 (+`arm_torque_saturation` −0.5) | — | 不动;备注:库里 `electrical_power_cost`(只罚正功)更贴电机热,能耗轴后备消融 |
| `self_collision_cost`(没挂) | `undesired_contacts` −0.1 | — | 不动;它的力史阈值 10 N 版本若我们自碰误报多可借鉴 |
| `push_robot` 1–3 s, v±0.5+z±0.4+rp/yaw | Wave-P p02–p08 档 5–15 s | mjlab 频 3–5×、多 z/rp 轴 | 半对齐:push 波胜档定型后,可加一臂对表 mjlab 频率(1–3 s)看抗扰上限,别动幅度 |
| DR 三件(foot_friction/encoder_bias/base_com) | physics_material/add_joint_default_pos/base_com | 量级相近 | 不动;"摩擦只随机脚+双脚共享抽样"这个细节与 grip/rough 臂假设同向,读数后可借 |
| `fell_over` 70° | `base_fell_tilt` ≈40° | 我们严 30° | 不动;站立击球不需要 70° 的宽容 |

### 档④【不要】——与任务冲突或已被机制取代(5 项)

1. **`track_linear/angular_velocity` 速度指令主奖**:我们的移动是"目标驱动"
   (`racket_progress` 望远镜奖+击球成功),不是"指令驱动"。引入 twist 指令=换任务;阶段2
   的脚步也应由到达目标驱动,不走速度指令路线(这是 HOPE 与 velocity 任务的根本分岔)。
2. **terrain 高度扫描观测 + `terrain_levels` curriculum + `out_of_terrain_bounds` 终止**:
   乒乓场地=平地固定场,阶段3也不变;高度扫描还会改 obs 契约(175/179-D 冻结)。
3. **`command_vel` curriculum**:皮之不存(无速度指令)。
4. **`is_alive`/`is_terminated`**:活着奖是常数底薪,稀释击球 income 记账
   (我们的 income 审计纪律靠各项收入可归因);死亡罚已由终止+episode 截断隐式承担。
5. **`angular_momentum` 常开版**:挥拍=角动量任务,常开直接抽任务的血
   (同一教训:base-ang-vel 类罚"anti-swing、gameable"已写在 prestrike_upright 的设计注释里)。
   恢复窗变体归档②。

---

## 4. 建议消融顺序(与 phase1_chatter_ground_foot_wave_20260722 衔接,不重复已排臂)

今天新波 8 臂(ar02/ar05/ar10=action_rate 剂量、grip/rough=摩擦与地面、footrw=脚部奖励、
penlight/kdpassive=罚项轻量化与被动阻尼;臂定义以该波预注册为准——本文档冻结时其 yaml
尚未合入本工作树)已经覆盖:一阶平滑剂量、摩擦 DR、foot_soft_landing/foot_clearance 首证。
以下按价值排序,全部是**今天波读数出来之后**的下一步,不与 8 臂重复:

1. **footrw 拆单项归因**(若 footrw 臂把 soft_landing+clearance 捆绑且信号显著):
   soft_landing-only vs clearance-only 两臂,权重沿用 footrw 胜档;若 footrw 无信号,
   先查门是否真拆干净(mjlab 门语义残留=罚项全程为零,收入记账应有非零 income,
   WARN 必进摘要)再谈加减。
2. **`action_acc_weight` 二阶平滑臂**:在 ar02/ar05/ar10 的最优档上叠加,起步权重取
   该档 action_rate 的 1/5~1/2(二阶量纲更大,先小);与 kdpassive 不同机制
   (一个罚策略输出的掉头,一个改执行器阻尼),读数可交叉验证"抖动来自策略还是执行器"。
3. **`electrical_power_cost` 能耗臂**:替代或叠加 `joint_torques`(−1e-5),只罚正功、
   不罚回收,是 τ² 的更物理版本;排在平滑轴收敛后,给 sim2real 电机热预算铺路。
4. **push 频率对表臂**:push 波胜档幅度不动,把间隔从 5–15 s 压到 mjlab 的 1–3 s
   加一臂,试抗扰上限(mjlab 还推 z/rp 轴,可分两步)。
5. **阶段2解锁包**(变到达+脚步开跑后):`variable_posture` 三档 std +
   `air_time`/`foot_swing_height` 一对,门信号统一换"移动意图"(目标距离或阶段2指令),
   预注册时把门语义写进合同,别再踩速度指令门的坑。
6. **恢复窗角动量债**:`post_swing_settle_debt` 第六债候选,只在 0.2–1.55 s 窗内结算,
   等 S1 债 bundle 有正读数后再加。

---

## 5. 附:本次考证的三个新知(汇报用)

1. **mjlab 脚部罚全挂速度指令门,站立自动关零**——它的"站立"是没事干,我们的"站立"是
   击球工况,拆门是采纳的前提;并行作业 `foot_soft_landing_weight`/`foot_clearance_weight`
   即拆门版键名。
2. **`soft_landing` 权重 −1e-5 是量纲信号**:罚的是首次触地接触力(几百 N),权重必须
   极小;接我们的键时按力量纲起步(−1e-5 ~ −1e-4),不能抄惯用的 −0.1/−1 档。
3. **mjlab 没有 stand_still 专项,velocity 任务也没挂 torque/qacc 罚**:站好=姿态三档核+
   10% 零指令采样的产物,平滑几乎全押 action_rate −0.1(与我们逐字同值)——佐证我们
   "少而真的护栏"路线,也给 ar 剂量臂一个外部先验:比 −0.1 更重大概率无益。
