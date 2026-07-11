# 新动作库、TOPP 与任意时刻下一拍

日期：2026-07-11
状态：原视频、canonical-beta GMR、grounding 和 240 Hz 安全屏均 10/10；canonical counterfactual HOPE frame/mirror 已验并消费 64 题，但 exact coverage 全 0；已预注册全十动作的原子 SE(2) spatial-retarget proposal 屏，真正晋级仍卡 schema-2/L0/L1/桌网，2-vs-4 与 TOPP 暂停
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

`scripts/ground_gmr_pkl.py` 已把这个校准步骤实现成 fail-loud 工具：一个显式 input、一个全新
output、一个全新 report，input/MJCF 都要求 expected SHA；canonical MJCF 必须只有一个 floating
root 且 31 hinge 顺序逐名相同。工具只纳入 robot subtree 中启用的 collision geom，primitive 用
解析 support、mesh 用 MuJoCo compiled vertices，逐源帧求最低 world-z，再只给
`root_pos[:,2]` 加一个固定量。输出回读后必须保持 root XY/root quaternion/31 DoF 逐位不变、
关节仍在 MJCF range、最差帧接近地面而不穿透/过度悬空；report 绑定 tool/input/output/MJCF 和
compiled collision digest。真实 canonical `a3_pingpong.xml` 的 native MuJoCo 测试已通过。
Pod1 随后对十条 diagnostic GMR 输出逐条运行了同一内容寻址工具，全部 no-clobber 完成；结果账本为
`configs/motion_video_gmr_ground_results_20260711.json`。原始离散帧最深穿地范围为
`8.072--8.716 cm`，逐条固定 root-z shift 后的全局最小余量均约 `10 um`；每条 output/report
SHA、输入 GMR SHA、MJCF、tool 与 compiled-collision digest 都已交叉绑定。它们仍使用
per-video betas，且这仍只审 30 Hz 离散帧，不是 inter-frame 连续 clearance 证明，所以后续
门禁顺序没有缩短。

体型归一化的上游 PT 也已完成，但仍是 diagnostic。工具
`scripts/materialize_canonical_gvhmr_betas.py` 先对每个视频的 10 维 beta 逐维取中位数，再让十个
视频等权取中位数，避免 35--98 帧的长度差给长片更多票。Pod1 CPU-only validate 和 no-clobber
materialize 均通过，统一向量 SHA 为 `a03f1642...9cc6`；十份 save/reload 输出除
`smpl_params_global.betas` 外的逐叶 semantic digest 均逐位相同。完整输入/output/tool/plan SHA 在
`configs/motion_video_canonical_betas_result_20260711.json`。GMR 的
`1.66 + 0.1*beta[0]` 只给出 `1.73066 m` 的内部 heuristic，不是实测身高，故
`a3_calibrated=false/formal_eligible=false`。随后 source audit 作废了旧 prereg 中未绑定的
“补六个零”猜测：clean `aabea2e` loader SHA `2737f472...5de2` 实际只取
`betas[0].detach().cpu().numpy()[:10]`，zero padding=false。独立 body-shape-aware CPU queue 已
10/10 重跑完成：全部 30Hz/31DoF/finite，warm-up 16--29 轮后 max|dq|<`1e-4`，且没有
占用 GPU 或改 GMR/train checkout。账本为
`configs/motion_video_canonical_gmr_results_20260711.json`。下一步是对这十条新输出重做
no-clobber grounding 与稠密安全门，不是直接进 RL。

这个前置现已有两层独立证据。第一层
`configs/motion_video_canonical_gmr_ground_results_20260711.json` 把十条 canonical-beta GMR
各自加一个 root-z 常量，离散源帧的最小地面余量均约 `10 um`；这只证
30 Hz 离散落地。第二层 `scripts/screen_motion_gmr_phase_safety.py` 对显式清单中每条轨迹做
root quaternion shortest-arc SLERP 和关节线性插值，以 8 个子步/源帧区间（240 Hz）复查地面、
全 robot self-contact，以及拍面/拍柄对头颈、躯干、对侧手臂和下肢的距离。十条共
654 个源帧/5162 个稠密样本：地面危险 0、自碰 0、余隙 `<5 mm` 危险 0、`<20 mm`
warning 0；全局最薄身体余隙是 Franco 反手拉 A 的 `40.2466 mm`。这仍是有限采样，
不是数学上的连续时间证书，也没有 table/net geom 或动力学。

屏全工具从 vendor MJCF 官方 `right_racket` site 逐帧取真实拍心/拍面，并用
`mj_differentiatePos + mj_objectVelocity` 取拍心速度。但 canonical grounding 只改 root z，
尚未证明 GMR world 到 HOPE +X/虚拟球桌的变换，intake 的 mirror status 也仍是 unverified。
因此首次 v2 中生成的 virtual-return/覆盖列已全部作废，只保留其与 v3/v4 逐资产相等的
safety subtree。接受的 v4 虽冻结了 64 题（semantic SHA `4dfa0548...`），但写明
`consumed_for_returnability=false`；十条的 `top_training_phase_candidates/question_coverage`
和全部 2-vs-4 selector 都是 `null/blocked`。完整证据链在
`configs/motion_video_gmr_phase_safety_results_20260711.json`。必须先完成 schema-2 + HOPE +X
reground（或给出独立验证的显式 proper-rigid 4x4 transform）和镜像语义，才能用同一题纸
重跑相位与覆盖；v4 已把这个未来开关的验证和对位应用写进工具，但 v4 合同当时仍关闭。

后续证据已经把这个开关**只对 canonical counterfactual table**打开。十条 final MP4 的 midpoint
crop 里，洗衣机中文标签均为正常、未反射方向；GMR 的右/左臂相邻帧关节增量能量比最小仍约
`9.98x`（预注册门 `5x`），所以 `verified_not_mirrored/no side swap` 可证。每条 proper-rigid
变换只由 frame-0 pelvis heading/XY 与已审地面生成：root XY→原点、heading→+X、Z 不动；
矩阵在看题前冻结，禁止按覆盖率再调 yaw/XY。标准桌仍是 near edge `0.5m`、surface `0.76m`，
不是录制现场桌外参。frame result SHA 为 `e70492be...`。

v5 随后实际消费同一张 64 题纸（full result SHA `c299b7a0...`），并复现 v4 十条 safety subtree。
结果是所有 motion/library 的 exact zero-retarget coverage 都为 `0/64`，因此 common support=0，
2-vs-4 没有可判分母。这个 0 表示固定题的位置、拍位与可回球状态没有在同一帧重合，**不等于动作
无效**。把球仅诊断性搬到拍心后，Franco 反手拉 B 为 `32/32 @ phase 0.5444`，C 为
`27/32 @ 0.5155`，A 仅 `1/32`；但 B/C 峰距最近 immutable question position 仍约
`0.165/0.237m`。因此只保留 B/C 为显式 spatial-retarget 候选，不把 intrinsic 当命中率，
也不从结果反调 frame。`TOPP=paused_until_spatial_retarget`；之后还要 schema-2/L0/L1、桌网、
动力学与 TOPP 后复审，最终动作与 2-vs-4 先过智元 vendor MuJoCo Gate3 runtime/stability，
再由共用 runtime 的 Gate3B no-reset behavior 卷主判。
小账本为 `configs/motion_video_gmr_phase_counterfactual_results_20260711.json`，完整坐标合同见
`docs/interfaces/motion_gmr_hope_frame_contract.md`。

### spatial retarget 不是反推录制桌外参

v5 的 `0/64` 把下一问限定得很精确：不改人体/机器人动作的空间路径，只问“整条动作放在
哪个安全站位、取哪个源帧，能否服务这道 immutable question”。新的
`configs/motion_video_spatial_retarget_prereg_20260712.json` 因此只允许一个对全轨迹原子应用的
保地 SE(2) 变换：R0 只平移，R1 只在预冻结 `[-10,-5,0,5,10] deg` 上旋转再平移；
`z=0`、scale=1、`det(R)=+1`，禁止镜像、逐帧变换和关节修补。站位上界为平移范数
`0.30 m`、`|x|<=0.20 m`、`|y|<=0.30 m`。它表示 planner 在 HOPE 标准虚拟桌下的站位请求，
绝不表示从空挥视频恢复出了房间相机/球桌外参。

搜索必须遍历十条 motion 与全 64 题；B/C 的 intrinsic 证据只能改报告排序，不能从卷中
删掉其他动作。每个安全/拍速合格源帧先把整轨迹绕 HOPE root 旋转，再求使拍心与题目
XY 重合的最小平移；Z 不能改，仍用原 capture radius 严格判定。反事实飞行从 immutable
球位置开始，不从空挥房间的拍心点开始。工具
`scripts/screen_motion_spatial_retarget.py` 已用 7 个纯 CPU 回归锁住十动作不可跳过、保地 proper-rigid、
站位上界、side/安全帧和 fail-closed certificate。`candidate_id` 还绑源 motion SHA 与 full-v5
result SHA，不允许同名/同帧跨资产复用证书；virtual scorer 实现、venue physics、`9.5 cm`
capture、`0.3 m/s` approach 与 `10 ms x 100` rollout 也全部显式内容绑定。

当前只允许生成 `proposal`，不允许“搜到就晋级”。准候选必须对精确 `candidate_id` 内容绑定：

1. 整轨迹使用该原子变换后的 runtime-order schema-2 物化；
2. L0 `audit_motion_npz.py` PASS；
3. L1 vendor-MJCF `audit_self_collision.py` PASS；
4. 整轨迹桌/网 swept-clearance 零硬失败且最小余隙 `>=5 mm`。

当前 prereg 明确 `certificate_bundle_preregistered=false`，临时塞一个“通过证书”会被拒绝；必须另立
新的内容寻址 prereg 才能消费证书。原子平面变换在数学上保持 z 和机器人内部距离，但
schema-2 改变了 body order/速度语义，所以 L0/L1 仍不得继承旧 GMR 结果而必须重跑。随后的动力学/
平衡仍是 TOPP/RL 之前的独立门。运行说明见
`docs/operations/run_motion_spatial_retarget_screen.md`。

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
   body order、kinematics schema-2 NPZ，并重落地到 HOPE +X。现有 240 Hz GMR 安全屏可以提前
   淘汰明显穿模，但不能代替这一坐标契约，也不能代替 schema-2 后的重审。
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

### 2026-07-12 命令 tuple 审计：R2/R3 之前先拒绝当前 hybrid

现有 T1 训练代码会在 reveal 前保留整套上一拍题目，reveal 时再原子安装整套新的
`position/velocity/normal/rho/side/incoming/base-anchor`。普通 wrap 也是整套换题，不会只换位置。
但当前 vendor C++ 179-D idle 变成了：

```text
position = new live-base-anchored hold position
velocity = previous strike velocity
normal/rho = previous strike face tuple
```

而且 position 随 live base 每 tick 移动。这不是“上一拍 tuple”，也不是“canonical ready tuple”；是训练
从未看到的混代命令。后续不在这条路上猜 anchor 和 reward，而是用
`configs/phase1_recovery_tuple_abc_prereg_20260712.json` 固定三臂：

- A：安全、可打断、deadline-aware 的外部 bridge 到 ready set；
- B：actor 内完整 canonical tuple，含 ready-set 选位、零拍速、neutral normal、rho0 和 phase 语义；
- C：整套 previous tuple 保留到下一题原子切换。

现有 179 checkpoint 可以给 A 做冻结 swing diagnostic，可以给 C 做 zero-shot 同代 tuple diagnostic，但两者
都不能改名为“已学会随机来球恢复”。B 必须 fresh；A/B/C 的因果比较也必须 fresh exact paired。
A 还有一个特别的交接问题：bridge 执行时 actor 不控制，但 actor history 下一刻仍需要 action。合同现在
固定为“实际 executed bridge action 到 actor-action 坐标的精确内容绑定投影”；shadow-policy action 只诊断，
直接清零/保留 stale action 都不属于 no-reset 正式卷。逐 tick `actor_control_mask=1` 仅当 actor sample 真正
执行，其 logprob 必须对应该执行 action；bridge tick 不伪造 logprob，policy/entropy/value loss mask 均为 0。
真实 bridge reward 作为 option return 折叠到前一 actor transition，非真终止在下一 actor state bootstrap。三臂共用
预绑定 env-step/机会/update/actor-sample/minibatch/epoch；B/C 多余样本按不读取结果的固定索引下采样，
A 不足则整对 update fail，不补样或多跑；评测机会分母不缩。

ready 也不能用一个名字代替数值。静态审计已发现 Isaac reset pelvis 为 `(0,0,1.0684)`，vendor
MJCF `stand` 为 `(-0.0416378,0.000359049,1.06839)`；31 关节 L2 差 `0.171845 rad`，去掉头仍
`0.028789 rad`。Stage-1 拍点是 env-origin 绝对坐标，179 位置观测是 target 减当前拍子 FK，所以
`4.16 cm` root-x 差不会自动消失。这是待验的因果假设，不是已证根因；正式卷前要绑定两引擎
完整 ready/base/joint/racket/target/obs 数值 SHA。

与此相对，恢复 reward 仍留在结构证据之后。如果需要，平衡债、ready-set potential 和 random-arrival
readiness 先归一化，再做配对 `2^3` 交互，最后才在固定总预算上混合。自碰/跌倒/桌网/执行器安全
始终是不可补偿硬门。最终 MuJoCo 又分两层：Gate3 先硬验同 C++/MJCF/plant/model 的 first-tick 和连续稳定；
Gate3B 才在共用 runtime 上消费 random-arrival q50，主判 first-strike non-regression 和 return quality。

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
4. 已完成：十条 diagnostic GMR 的单文件 ground/root 校准与内容寻址账本；只证明离散源帧
   近地不穿透，未改变 `diagnostic_video_betas`/formal-ineligible 状态。
5. 已完成：十条 GVHMR PT 的同人等权 canonical-betas diagnostic materialize、canonical
   GMR 10/10、独立离散 grounding 10/10，以及 240 Hz 有限插值地面/自碰/拍柄-身体安全屏。
   没有实测身高，仍不是 A3 标定；桌网/动力学/schema-2 仍开。GMR-world→HOPE 桌球坐标和
   mirror 未证，所以逐帧回球率、击球相位和 2-vs-4 覆盖已 fail-closed，不启动 RL。
6. 已完成设计/工具验证：全十动作×64 题的原子 SE(2) spatial-retarget proposal 合同；
   实际 proposal 运行待恢复 full v5 结果，晋级待 schema-2/L0/L1/桌网 candidate certificate。
7. 对通过上述证书和动力学的幸存路径做 native/TOPP 双资产和重复门禁。
8. 泛化动态 clip catalog 与共同问题轴；先在 CPU/Isaac smoke 证明四动作身份、side sign、题目绑定。
9. 先跑 family/候选/时间律小消融，再跑 2-vs-4；不把所有轴一次打包。
10. 最后才进入 event-driven recovery R0–R4，并以 immutable Isaac/MuJoCo 连续门禁裁决。

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
