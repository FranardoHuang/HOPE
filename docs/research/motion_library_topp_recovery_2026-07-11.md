# 新动作库、TOPP 与任意时刻下一拍

日期：2026-07-11
状态：设计已预注册；原视频 intake 已验证；Pod1 GVHMR 结构重建与 diagnostic GMR 均 10/10 完成；ground/schema-2、正式体型、仿真消融和安全门禁尚未完成
范围：仅离线处理与仿真。本文不授权任何真机动作。

## 结论先行

1. **TOPP 适合解决“完整 clip 太长”**，但只能改时间律，不能修坏的空间路径。先把视频变成
   A3 路径并过关节、自碰撞、桌/网余隙和动力学门禁，再用同一预算求每条路径的最短可行时间。
   现成 `topp_mintime.py` 已能锁住触球窗、逐位保留触球行和拍速；它是 oracle-in-loop 的
   TOPP 启发式，输出是搜索族内最短时间，不是带严格全局最优证书的 TOPP。
2. **四动作不能简单等同于“把四个 NPZ 填进 motion_file”**。当前训练、题库、指标、MuJoCo
   考卷和 C++ runner 多处仍写死正手/反手两类，而且当前采样顺序是“先随机动作，再从该动作
   的题库抽题”。要做“这颗球选最稳动作”，必须先建立共同题目轴和动态动作目录。
3. **两拍之间不应只学回到一个精确第 0 帧**。更稳的目标是一个有容差的准备状态集合：站位、
   朝向、关节姿态、全身速度、拍/身体余隙均合格。它分担吸收上一拍、恢复平衡、等待未知来球、
   启动下一动作四项职责。
4. **恢复奖励必须和击球信用隔离**。仓库有强警告但不是干净单变量：Rally v1 同时加了
   长 hold/episode、`post_strike_brake=1` 和 `hold_ready=1`，击球 composite 从 0.994 降到
   0.866。“正的减速收入经 PPO/GAE 反向削弱挥拍”的机制可信，但旧实验不能把降幅
   全归给 brake 单项。第一版仍应关闭正的 hold/brake 收入，使用外部固定窗、恢复终点
   损失/有界负 debt 或独立 value 边界，再做小步联合微调。
5. 当前 24 条 Phase-1 是冻结合同的另一条轴，不把新视频、TOPP 或 recovery 偷换进去。新轴优先
   使用自然终档释放的 GPU 槽；若确需让路，顺序是先暂停非目标 plant 的 LZ/LP 诊断臂，绝不先停
   formal SZ，也不把中途改配方伪装成同一臂。

## 原始素材与可追溯入口

源视频不进 git。内容寻址清单为
[`configs/motion_video_intake_20260711.json`](../../configs/motion_video_intake_20260711.json)，
校验命令为：

```bash
python3 scripts/audit_motion_video_intake.py
```

本机和 Pod1 私有 staging 均已验证 10/10 的字节数、SHA-256、编码、帧数和时长。Pod1 路径：

```text
/workspace/codexschema/motion_video_intake_20260711/raw/
```

GVHMR 队列脚本为 `scripts/run_motion_video_gvhmr_queue.py`。Pod1 PID/PGID
`1383735` 在 GPU1 自然低于 `19000 MiB` 后启动，从 09:27:50Z 到 09:37:16Z
完成 10/10，GVHMR commit 为
`6ec3ca39336c50492c0fae65fba2fb831fc7d866`。队列先做 Franco 正手挡，再做余下
Franco 和 v6/v7；它全程在 Phase-1 训练 checkout 之外运行。
每段绑定 manifest/队列工具/结构审计器/GVHMR commit/干净工作树/7.4 GB 权重与人体模型树/
Python 环境/input/output SHA。输出还必须通过预期帧数、SMPL 参数 shape 和全 finite 结构审计；
任一失败立即停队并保留日志。这个 pass 只允许继续批量预处理，不代替 pilot 视觉质量、镜像/
深度、A3 安全或击球可行性判定。完整的 queue-state、输入/输出、工具、模型及环境 SHA 在
`configs/motion_video_gvhmr_results_20260711.json`。`19000 MiB` 是开跑前采样门而不是 GPU 资源预留，
所以 worker/judge 仍需持续监控，不把它写成绝对不会 OOM 的保证。

随后使用 repo-owned `scripts/run_motion_video_gmr_queue.py` 在 CPU 串行完成
10/10 GMR。队列从 10:33:20Z 到 10:34:12Z 运行，要求 GMR worktree clean
HEAD `aabea2eee4be4bc16d4be17dac5ffa85e5a31539`，并验证包含该 commit 的
source bundle；不允许 `--no-warmup` 或跳过速度约束。十条输出均为预期帧数、30 Hz、31 DoF、
全 finite，frame-0 warm-up 为 17--28 轮且 final `max|dq| < 1e-4`。完整
bindings 在 `configs/motion_video_gmr_results_20260711.json`。这一批沿用逐视频 GVHMR
betas，故合同固定为 `diagnostic_video_betas`、`formal_eligible=false`；它证明管线和结构，
不代替 canonical-betas 正式资产。

对 Franco 正手挡做的更深只读 pilot 表明：GMR 与 canonical MJCF 的 31 关节顺序相同，
关节位置均在限位内，30 Hz 差分最大速度 `8.45185 rad/s`（右肘）低于对应 URDF
`15.70796 rad/s`，641 个离散/子步采样姿态中 robot self-contact 为 0。但末段仍有
`0.23452 rad/s` 关节速度，不是严格静止 ready；MJCF 又不含 table/net，零自碰只是一张
有限采样诊断。硬阻塞是 65/65 帧都有地面穿透，最低 collision geom 约
`-0.0773..-0.0841 m`。必须先用单文件、内容寻址的固定 root-z 校准落地，再重跑关节/
速度/加速度、连续碰撞、动力学和桌网门，禁止直接把当前 PKL 转成训练资产。

| 组 | 语义动作 | 文件 | 时长 / 帧数 | 当前角色 |
| --- | --- | --- | --- | --- |
| Franco | 正手挡 | `forehand_dang.mp4` | 2.167 s / 65 | 四动作候选 |
| Franco | 反手挡 | `backhand_dang.mp4` | 2.100 s / 63 | 四动作候选 |
| Franco | 正手拉 | `forehand_la.mp4` | 2.333 s / 70 | 四动作候选 |
| Franco | 反手拉 A | `backhand_la.mp4` | 2.300 s / 69 | 同一动作位候选 1 |
| Franco | 反手拉 B | `backhand_la2.mp4` | 3.033 s / 91 | 同一动作位候选 2 |
| Franco | 反手拉 C | `backhand_la3.mp4` | 3.267 s / 98 | 同一动作位候选 3 |
| v6 | 正/反手挡 | `forehand_v6.mp4`, `backhand_v6.mp4` | 各 1.167 s / 35 | 短动作双臂候选 |
| v7 | 正/反手挡 | `forehand_v7.mp4`, `backhand_v7.mp4` | 各 2.133 s / 64 | 中等时长双臂候选 |

只读接触图初审：

- Franco 四类动作大体共享同一宽站准备姿态，首尾比跨组拼接更有希望；三条反手拉都经过胸腹前方并有
  高收拍，自碰撞风险最高。
- v6/v7 的准备姿态与 Franco 不同，双臂更外展。跨 family 不能宣称共享同一个第 0 帧，除非桥接
  门禁实测通过。
- v6 只有 35 帧，时间短但更容易受 GVHMR/GMR 冷启动和有限差分端点噪声影响；不能因“短”直接晋级。
- 所有素材都是正面单目空挥，没有球、球桌和真实接触。视频本身不能证明击球点、落台率、桌边余隙，
  也不能可靠观测前后景深。手腕和球拍还有运动模糊，拍面只能以 A3 FK 为准。
- `backhand_la`, `la2`, `la3` 是**同一个反手拉动作位**的三个候选，不作为三个额外动作增加样本数。
- 镜像状态尚未得到独立证据。管线必须显式钉右手、镜像状态、HOPE +X、胶皮面符号，禁止按文件名猜。

## 从视频到可训练动作的门禁

每个候选必须单独走完整链，不能先拼动作库再找问题：

1. 使用同一 pinned GVHMR/GMR 和 canonical body betas，生成 SMPL-X 与 A3 retarget。
   当前 per-video-betas GMR 只作管线诊断，不能冒充该正式步骤。
2. 对单条 GMR 轨迹做内容寻址的 root/ground 校准并保存 before/after clearance 报告；
   禁止目录扫描、原地覆盖或用共享最小值污染别的动作。随后生成 50 Hz、31 关节、runtime
   body order、kinematics schema-2 NPZ，并重落地到 HOPE +X。
3. 跑 `audit_motion_npz.py`：finite、关节位置/速度、加速度、首帧冷启动、foot skate、触球窗。
4. 跑 vendor MJCF `audit_self_collision.py`：要求零自碰撞；另外记录球拍/手柄到头颈、胸腹、
   对侧手臂的最小余隙，不能只看碰撞布尔值。
5. 扫全轨迹的球拍 swept volume：球拍/手臂不能打到机器人自己；低引拍和高收拍还要查桌边与网。
6. 空挥没有真实 contact truth。保留 `contact_phase=null`，用
   `gen_stage1_questions.py --phase-scan` 得到 `train_phase_candidates`。这些是“在哪一帧假设触球最容易
   合法回台”的训练候选，不冒充视频接触真值。
7. 对每个安全候选，在同一来球 train split 上建立逐帧回球率/可解率曲线；人工检查前三个峰值的
   球拍运动方向、拍面、引拍余量和触球后余量。若多个峰接近，预注册小型 phase 消融。
8. 原始路径过门后才跑 TOPP v3；输出重新登记 `phase_out`，再完整重复第 3–7 项。

任何一个候选有非 finite、硬限位、机器人自碰撞、桌面穿透或触球窗不可约动力学失败，就先淘汰，
不进入 RL。TOPP 不能把碰撞空间路径“放慢成安全路径”。

## TOPP 的使用方式

### 已有实现能做什么

`hope_training/whole_body_tracking/scripts/topp_mintime.py` 已实现：

- 固定空间路径 `q(s)`，在相同 URDF 速度、实证加速度、CoP、摩擦锥和关节力矩预算下搜索最快时间律；
- 健康路径可压缩，病路径会被预算推回较慢时间；
- 触球附近默认 `±0.1 s` 锁窗，触球路径行逐位 pin，默认拍速取源动作 clean blade speed；
- 每个输出带新 `phase_out`、触球行/拍速/拍面保真报告。

这和 TOPP 文献的正确抽象一致：几何路径固定，只优化沿路径的速度/加速度。TOPP-RA 论文用可达/
可控集在离散路径位置求受约束的时间参数化，并明确覆盖速度、加速度、力矩和接触稳定性约束
([Pham & Pham, 2018](https://arxiv.org/abs/1707.07239))。本仓库因 CoP/摩擦/`mj_inverse`
是黑盒 oracle，采用外层缩放加局部 `rho(s)` 修复，不能把“TOPP”写成严格全局最优证书。

### 新动作的配对合同

每条通过空间门禁的动作只生成两个主资产：

- `native`：同一 A3 空间路径和原始 50 Hz 时间律；
- `topp_v3`：同一空间路径、相同触球行/拍速、相同统一预算的最短可行时间律。

配对验收：

- 触球 joint row bitwise equal；拍速偏差不超过 2%；拍面偏差在报告阈值内；
- 起点/终点速度近零，并落进相应 family 的 ready set；
- 原始和 TOPP 输出各自 L0/L1/桌网余隙零硬失败；
- 训练按**击球机会数**和每动作完成 swing 数对齐，不能只按 iteration 对齐。短 clip 天然每小时看到
  更多球；这正是产品收益，但另需一张等 exposure 表分离“节拍收益”和“更好学”。

## 两动作和四动作的公平消融

### 为什么当前代码还不能直接正式跑四动作

底层 `MotionLoader` 已支持任意 clip 数，但其上游/下游仍有二分类假设：

- `MotionCommand` 当前先均匀随机 `clip_id`，题库再按已选 `clip_id` 抽一题；方向与目标相反。
- `RacketTargetCommand._clip_names`、分侧指标、若干 buffer 固定为 forehand/backhand。
- `swing_sign` 把 clip 0 当正手、所有非 0 clip 当反手，第二个正手动作会被误标。
- train YAML per-clip 映射、exam materializer、MuJoCo 累计器和 C++ reference clock 仍有长度 2 假设。

第一步引入单一有序 `clip_catalog`：

```text
{name, motion_sha, family, stroke, side_sign, strike_phase,
 mount_sign, pos_box, vel_box, sampling_weight, enabled}
```

它统一生成 motion 顺序、phase/sign/range、指标名称、hard contract 和评测目录。第一版不增加 actor
维度：62-D 参考 command 已隐式携带动作差异；`side_sign` 只继续表达正/反手，动作身份由内部
`clip_id` 明确表达。若以后加动作 one-hot，二动作/四动作臂都使用同一个固定宽度，避免网络容量混杂。

### 共同问题轴与最稳 selector

题库改成一次生成全局 `question_id`，然后为每个 `(question, clip)` 保存：

```text
valid, demanded_pos/vel/normal, solver_residual, net/table margin,
stroke_accel_margin, TOPP torque/CoP/friction margin, self_clearance,
policy_return_lcb, recovery_success_lcb
```

当前阶段只追求“最稳”，冻结 selector 的顺序是：

1. 硬过滤：不可解、碰撞/余隙失败、动力学预算失败、到点时间不足一律无资格。
2. 在 **train split** 上按合法回台率的保守下置信界排序；同分时选余隙更大、恢复失败更少的动作。
3. selector 和阈值内容寻址冻结后，才看 immutable exam split；禁止用 exam 成绩挑动作。
4. 一次性下发 `{question_id, clip_id, target, strike_time, reveal_time}`，不能先异步切 clip 再换题。

评测同时报两张表：

- **共同支持集**：二动作和四动作都能作答的同一批题，量非退化和纯策略质量；
- **覆盖包络**：同一全局题纸上至少有一个动作可作答的比例，量增加动作是否真的扩大可接球种类。

### 预注册矩阵

先做便宜的离线筛选，再训练：

| 轴 | 配对 | 回答的问题 |
| --- | --- | --- |
| 反手拉候选 | A / B / C | 哪条安全、可回、可 TOPP；候选不是额外动作 |
| 挡动作 family | Franco / v6 / v7 | 肩主导、极短、较完整哪种 A3 路径更好 |
| 时间律 | native / TOPP v3 | 同一空间路径缩时是否增大每小时机会且不降回台率 |
| 动作数 | Franco 2 挡 / Franco 4 动作 | 同一 family 内增加拉球是否提高覆盖/稳定性 |
| 算力口径 | 等总 transitions / 等每动作 exposure | 产品成本收益 / 去掉样本稀释后的动作库价值 |

每个训练配对共用 plant、题库 family、episode/来球时序、网络、seed、checkpoint cadence 和判卷。
四动作臂只能多两个动作，不能同时换 TOPP、phase、reward 或 plant。每个 checkpoint 同时报总
transitions、每动作 swing 数、训练内全分母回台率与 immutable Isaac/MuJoCo exam；保留峰值，
不等终档才判断。

## 任意时刻下一拍：状态机而不是长 clip

当前完整 clip-wrap 的语义是“整条录像播完才出下一题”。场馆保守 A-B-A 接触间隔中位数为
1.903 s，而现役理论同侧间隔中位约 3.75 s；因此“同一 episode、不传送、carry state”仍不等于
“下一球随时来”。

推荐把一拍拆成四段：

```text
strike:  引拍 -> 触球 -> 最小安全随挥
absorb:  吸收触球/随挥动量，避免碰撞和跌倒
recover: 从实际终态回到准备状态集合
ready:   在未知下一球到来前保持可向多动作启动
```

下一任务可以在 `absorb/recover/ready` 任意时刻出现。调度器用当前状态、下一动作和 deadline 判断：

- 已在 ready set：直接启动新动作；
- 正在 recover 且仍有足够时间：把 recover 目标改成下一动作的可接受入口集合；
- 时间不足：selector 必须换成当前状态可达的动作，或把该题标成不可执行；不能硬跳参考帧。

### 准备状态集合

不是单点 `q=q0`，而是以下合取：

- base XY/偏航在允许范围，脚接触稳定；
- 关节在各动作起点的交集或经验证的 union 区域；
- base/关节/球拍速度低于阈值；
- 球拍到头颈、躯干和对侧手臂有余隙；
- 从该状态到每个启用动作的安全启动时间有上界。

Franco 组可先用四段第 0 帧的鲁棒中心构造 ready set；v6/v7 各自单独构造。若 family 间起点
不重叠，就需要显式 transition，而不是把两个名字都叫 ready。

### 从论文能直接拿什么

- **Ace** 并未让击球策略自己慢慢“学会收回来”。它的 rally policy 在仿真只按单球 episode
  训练；每个 32 ms segment 同时计算一条从 segment 终态回到静止 reset pose 的近时间最优 MPC
  轨迹。击球后执行最新 reset trajectory 直到下一球。reset pose 可以固定，也可以由 prepare
  network 根据下一球状态选择，以最大化下一拍 dexterity；训练初态还会采样历史 reset plan 的动态
  状态。这直接支持“击球技能 + 显式安全 recovery/prepare”分层，而不是把所有职责塞进一个长
  clip ([Dürr et al., Nature 2026](https://www.nature.com/articles/s41586-026-10338-5))。
- **HITTER** 直接训练 10 s 多拍 episode：每个 swing 完成后重采样正/反手、球拍和 base 目标；
  base-position reward 在触球前生效，论文解释为让策略在击球后准备转向下一目标。它证明多拍 carry
  state 可以得到长 rally，但公开描述仍是“完成 swing 后换题”，没有证明任意 mid-clip 来球
  ([Su et al., 2025](https://arxiv.org/abs/2508.21043))。
- **Leave No Trace** 联合学习 forward policy 和 reset policy，并用 reset value 识别即将进入不可逆
  状态，支持把 recovery 可达性同时当安全判据，而不是只给一个姿态奖励
  ([Eysenbach et al., ICLR 2018](https://arxiv.org/abs/1711.06782))。
- **Reset-Free RL via Multi-Task Learning** 的核心观察是：合适的多任务可以互相充当 reset，通过显式
  排程一起学而不依赖人工重置。这里对应“击球、吸收、恢复、准备”是相互衔接的任务，而不是一个
  无限长 reward soup ([Gupta et al., ICRA 2021](https://arxiv.org/abs/2104.11203))。
- **Adversarial Skill Chaining** 指出朴素拼技能失败的根因是前一技能终态分布和后一技能初态分布不匹配，
  并通过终态分布正则化改善串联。这支持优化 ready **分布/集合**，不只优化平均第 0 帧误差
  ([Lee et al., CoRL 2022](https://proceedings.mlr.press/v164/lee22a.html))。

上述论文没有直接证明 A3 上哪一种恢复 reward 最优；下面是结合本仓库失败证据提出的工程假设。

### 本仓 Hitter 分支恢复实验再审计

真正继续 recovery 试验的谱系是 `origin/hitter`，不是停在 merge-base `c951d9` 的
`origin/jiayi`。这些旧试验提供了很多模块和失败机制，但多数 YAML 同时改了数个因素，
不能直接当成干净消融结论。

- HitterPure (`baf6215`) 是 10 s、no-teleport、多 clip-wrap carry-state 基线，没有 hold/
  recovery reward。它证明“物理状态不重置”，不证明会从任意收拍状态回到 ready。
- A8 own-reset (`287efc4`) 把策略自己的 wrap 尾态存进 ring buffer 再做 true reset；
  2k 筛选 control=0.42、post-swing=0.40，方向中性略负。它没按“上一动作×下一动作”分层，
  而且只收录能活到 wrap 的状态，有 selection bias；可复用接口，不复用旧结论。
- Rally v1 (`5e97504`) 就是上述 0.994->0.866 的混合失败包；还同时受过 runner
  idle-station bug 影响。禁止把它写成 brake-only 因果实验。
- Rally V4 是旧线中最好的候选 bundle：world-x settle、vx quiet、leg quiet，记录为
  12/12 零跌倒、x drift 0.23 m、idle angular velocity 0.11 rad/s。它相对 V3 同时加
  三项，底座还已折入 passive damping，所以只用作“定向/腿部 debt”候选，不宣称三项
  任意一项独立有效。
- RallyFinal 暴露的核心不是 reward 剂量，而是 state coverage：strike composite 已有
  0.998，但训练 held-state 只占约 1.24%，带 20--35 cm station command 的仅 0.54%；严格
  Gate3 在首次约 24.5 cm frozen-ready 移站时 engage 前跌倒，ankle-roll qdes clamp 约
  83%/97%。这是“必须显式覆盖等球/转换状态”的强证据。
- RallyFinalV2 把等待改为外部固定 40--60 tick clamp，策略不能延长来刷分；
  mean-policy probe 改善了 release/far-step，但尚无完整 PPO 闭环，只是下一消融的更干净起点。

因此第一轮不复制 ARM-A/Rally v1/V5/V6/FinalV2 整包。复用的是固定外部 timer、
canonical ready 参考、按方向的 station settle、触球后 0.2--0.35 s 自然随挥 grace、以及有界
负 debt。`hold_until_settled` 与正的 hold income 均先关闭，防止策略控制等待长度刷分。

### 恢复消融

| 臂 | 设计 | 目的 |
| --- | --- | --- |
| R0 | 当前完整 clip-wrap + carry state | 现役基线，不声称任意时刻 |
| R1a/R1b | 固定外部随机 ready clamp，hold income 全关；default stand / canonical ready 两臂 | 先分离 timer coverage 和 ready 资产 |
| R2 | 触球后保留 0.2--0.35 s 自然随挥，随后走离线安全 bridge 到 ready set | 先验证不用 RL 能否完成 |
| R3 | 独立 recovery option；post-swing buffer 按前一/下一动作分层，过滤跌倒/碰撞态，只给终点成功/有界负 debt | 隔离击球信用并修 A8 selection bias |
| R4 | recovery 条件化下一动作与 deadline | 下一题在恢复中出现时仍可接 |
| R5 | ready-set 随机化 + family 间 transition graph | 抗偏差并支持多 family |

R3 应先单独训练/判卷，再小学习率接回统一 actor。若必须同一次 PPO rollout 联训，至少设置 option
边界的独立 value/GAE mask 或 recovery critic，且每个 checkpoint 先跑冻结单拍考试。禁止复活已知会
污染击球的 dense 正 `post_strike_brake`；`hold_ready` 也不能同时奖励零角速度并要求转身。
heading 若单独开，必须同时给 yawed/post-swing 起始状态覆盖；仅有 reward 而 94% hold 本来就是正的，
只会学“保持正”而不是“恢复正”。

### 任意时序测试

建立与真实 A-B-A 间隔分布绑定的 event schedule，而不是均匀长 hold：

1. T0：完整 clip 结束才 reveal，保持现役可比；
2. T1：触球后从实测间隔抽下一任务 reveal time，允许落在随挥/恢复/ready；
3. 同一 schedule 对所有 recovery 臂，至少 30 s / 12 次机会；miss 后继续，不能 reset 掩盖债务；
4. 报每侧物理不摔、击球率、回台率、deadline miss、ready-set 到达时间、动作切换矩阵；
5. 单拍 frozen exam 必须非退化，连续成绩不能靠牺牲击球速度换取。

## 实施顺序

1. 已完成：10 段视频内容寻址 intake、本机与 Pod1 staging 双重验证。
2. 已完成：自然释放 GPU 槽后，按预定顺序串行跑完十段 GVHMR；
   全部通过帧数/shape/finite 结构审计。视觉质量和安全仍未被这个 pass 代替。
3. 已完成：CPU-only diagnostic GMR 10/10；结构和 warm-up 均通过，但 body betas 非正式，
   且 pilot 暴露约 8 cm 穿地，未授权 schema-2 或 RL。
4. 先完成单文件 ground/root 校准、canonical-betas 重跑和 schema-2，再做 L0/L1/逐帧
   回球率，淘汰空间路径，不启动 RL。
5. 对幸存路径做 native/TOPP 双资产和重复门禁。
6. 泛化动态 clip catalog 与共同问题轴；先在 CPU/Isaac smoke 证明四动作身份、side sign、题目绑定。
7. 先跑 family/候选/时间律小消融，再跑 2-vs-4；不把所有轴一次打包。
8. 最后才进入 event-driven recovery R0–R4，并以 immutable Isaac/MuJoCo 连续门禁裁决。

尚未完成的关键事实应始终写明：空挥素材没有真实接触真值；四动作 selector 未实现；任意时刻下一拍
未通过；任何新动作均未获准真机执行。

## 资源优先级与不空转规则

GPU 槽按以下顺序消费，且只在前置门禁有机器可验证输入时入队：

1. 保留已预注册 fresh `SZ` formal target 与必需 paired controls；
2. 保留已存在的 immutable checkpoint 曲线判卷，不等 terminal 才看增长；
3. 自然释放的第一个安全显存槽交给新视频 GVHMR/GMR 处理和离线门禁；
4. 新动作只在 schema-2/L0/L1/击球点扫描/TOPP 门全通过后才占 RL 槽；
5. 若必须主动让路，先停未目标 plant 的 LZ/LP 诊断臂，不先停 SZ 或同卷对照。

“不空转”不意味着跳过 pilot 质量查验，也不意味着在没有动态 clip catalog 时偷发一个
伪四动作训练。能安全自动的预处理/静态门禁连续排队；需要视觉质量、击球点或安全判断的边界则
停在有完整输入和日志的检查点。
