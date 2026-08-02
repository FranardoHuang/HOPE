# 标准工序：编译好的动作片 → 可训练绑定（16 步）

**每步 = 做什么 → 用哪个工具 → 通过条件。** 不过就停，不许用相邻动作或另一个 scope 补证。
工具细节看它自己的 docstring（[工具目录](tool_catalogue.md)给出一行索引）。

## A. 这条片子能不能用

| # | 做什么 | 工具 | 通过条件 |
| --- | --- | --- | --- |
| 1 | 落地 + 转向：机器人面向球台、脚在地上 | `reground_hope_frame.py`、`scripts/ground_gmr_pkl.py` | 第 0 帧骨盆偏航 ≈ 0°；局部关节数据原封不动；保留 before/after clearance 报告。**⚠ 这只保证站姿，不保证瞄向——见第 7 步** |
| 2 | schema-2 契约 | `csv_to_npz{,_mujoco}.py`；老件 `migrate_motion_kinematics.py` | 50 Hz / 31 关节 / runtime body order；`body_lin_vel_w` 是**质心**速度（故意的）；迁移必须显式 `--source-point`，**绝不从文件名猜** |
| 3 | **L0 判炸器** | `audit_motion_npz.py` | **退出码 0**。WARN 逐条看过并记录。修复顺序写死：先改源头（GMR 预热 + 逐关节速度限），trim 只在 FAIL 局限头尾时才允许，**永不建议全局速度钳制**（合法挥拍峰值 11.88–12.63 rad/s 必须活） |
| 4 | **L1 自碰撞** | `audit_self_collision.py` | **退出码 0**，零自碰。另记球拍/手柄到头颈、胸腹、对侧手臂的**最小余隙**，不能只看布尔值 |
| 5 | **L2 真动力学** | `motion_dynamic_replay.py` | **退出码 0**。⚠ 先读它 docstring 的"已测事实"段：纯厂商 PD 连静态站立都撑不满 2 s，在真机 MJCF 上当**相对分级器**用（比 `fall.time_s`）。PASS 只证明参考轨迹本身动力学可执行，不替代 policy replay 或厂商 Gate3 |

## B. 哪一帧触球、朝哪个方向打（**2026-07-26 事故的正面修复**）

空挥片没有 contact truth：`contact_phase` 保持 `null`，触球帧必须由扫描算出，
**不能用"窗内前向速度最大帧"这类启发式**。

| # | 做什么 | 工具 | 通过条件 |
| --- | --- | --- | --- |
| 6 | **可回球性相位扫描** | `gen_stage1_questions.py --phase-scan --clip <键>:<npz> --grip off` | 输出里有 `>50%` 可回球带且带内有明确峰值。打印 `NONE — no frame ... can return` ⇒ **这条片在该球速档下没有可用触球帧**，回第 7 步或换片，**不要硬登记 0 分相位**。峰值贴在窗边界上 ⇒ **当红灯**（argmax 撞边界正是事故根因）。它从不回写 registry，只打印 `train_phase_candidates` 给**人**抄进 `cfg/strike_annotations.yaml`。校验 harness：在 v4rg 已注册 clip 上应逐位复现注册值 |
| 7 | 瞄向修正 | 同上 + 偏航扫掠 | 第 6 步全片 0.0000 ⇒ 先做偏航扫掠找窗口。实测 `fh_loop +90°`→1.00、`bh_block −45°`→1.00、`s0_highpress −60/−45°`→1.00 —— **不是一个统一常数能修的，每片按自己的击球方向重旋**。病因识别：世界系 `\|n_x\|` 应 ≥0.8（v4rg 0.86–0.94），病态只有 0.36–0.68 ⇒ 落点 `\|y\|` 0.85–1.86 m > 半台宽 0.7625 m。**重旋后接触位置也会动，第 6 步必须重跑** |
| 8 | 物理击球帧选择 | `rebind_physical_strike.py`（⚠ 仅在 pod2，未收编，见开放问题） | 前向拍速最大，受三约束：(a) 落在接触窗内；(b) 拍高 ≥ `0.76+0.02+0.10 = 0.88 m`（整个 ±0.10 目标框都要离台）；(c) 排除下落 >0.3 m/s 的回收段。**并恢复被误删的前向主导判据 `\|vx\| ≥ \|vy\|`**。选出的帧在第 6 步曲线上不能是 0 |
| 9 | 拍面符号 | `suggest_face_sign.py` | `sign = sign(n·v)`（哪面超前就是哪面），人审定后写进 `mount_normal_sign_per_clip`。**铁律：只能离线算成每 clip 常量**，运行时动态定符号会把反面击球合法化 |

## C. 绑成训练配置

| # | 做什么 | 工具 | 通过条件 |
| --- | --- | --- | --- |
| 10 | 编译产物验收 | `canonical_motion_bank_gate.py` | [`run_motion_face_shift.md` §5](run_motion_face_shift.md) 的 5 项。必须从 exact snapshot **重算**——复用 compiler 内部对象、信 producer JSON 或只比文件名**都不算独立验收**。即使写 `PASS_COMPILER_CANDIDATE_ONLY`，也必须同时看到 `training_authorized=false` |
| 11 | Registry / consumer 验收 | `canonical_motion_registry.py` + `_admission.py` | [`run_motion_face_shift.md` §7](run_motion_face_shift.md) 的 11 项。**第 11 项（`compiler_candidate`→`training_adopted`）发生前不提供训练启动命令**。走 legacy `motion_file` 无门通道是一个**必须写进实验记录的显式裁定**，不能默认发生 |
| 12 | 绑定收据 | 一份 `BINDINGS.json` | 记每条 clip 的 SHA-256、相位来源（第 6/8 步，**不是沿用别的谱系**）、符号来源（第 9 步重算值）、框来源。⚠ **数值巧合不是沿用**——收据原话："数值恰与 v4rg 相同是巧合，非沿用"。⚠ **题库不可跨谱系转移**（rebind 工具只收"触球帧逐位不变"的同族件） |
| 13 | 出题（若用题库） | `gen_stage1_questions.py --anchor train-candidate` | `--check` **默认开不许关**（每道答案经 torch 物理重放，落点在容差内且断言过网）；`--split train\|exam` 必填（按来球速度确定性哈希切 ~80/20，任何种子下都不相交）；**可答率低 = 击球点选坏了**，回第 6/8 步换点；分母报表入账连它一起抄 |

## D. 发射前的物理与冒烟门

| # | 做什么 | 工具 | 通过条件 |
| --- | --- | --- | --- |
| 14 | **目标框物理有效性** | `_assert_contact_clears_table`（构造期自动）+ `check_perclip_{pos,vel}_sampling.py` | 构造期不抛 `ValueError`；两个 check 脚本输出 `below table surface` **~0%**。规则与常量真源见[球拍目标物理有效性](../interfaces/racket_target_physical_validity.md)。⚠ **两个 check 脚本只 print 没有退出码门**（打印 83% 也 `exit 0`），补上之前这一步**必须人工读**，不能挂 CI |
| 15 | 对齐快门 | `check_motion_target_alignment.py` | 打印 `PASSED`（FAILED 退出码 1）：yaw≈0 / 配置相位处**控制点**（`pingpang_red_Link` 原点，不是裸手腕）速度 +X 主导 / `reference_perturbed` 精确从教师片击球状态起步 |
| 16 | 一格 2-iter 冒烟 | `train.py`（波级工序见[发射工序](run_ablation_wave_launch.md)） | 2/2 零报错；WARN 全进摘要、Error/Traceback 零条；含 `q_des CLAMP ACTIVE` 行；config 回显与第 12 步收据一致。**⚠ `mean_episode_length` 恒为 1 = 出生位对不上，不是"学得慢"**——canonical probe 出生在旧站姿（拍 0.882 m）而参考第 0 帧在 1.229 m，差 0.347 m > `ee_body_pos` 阈值 0.25 m，**跑了 2500 迭代零学习** |

<a id="task-first-addendum"></a>
<a id="action-ball-addendum"></a>

## Action-conditioned Ball-first 动作 bank 附加门

上述 16 步仍是每个 exact motion 的基础。把动作接进
[按动作条件化 Ball-first](../interfaces/action_conditioned_ball_first_contract.md) 还要补以下
bank 级约束：

1. manifest 逐行绑定 stable action UID、motion SHA、strike phase、family/face sign，以及完整
   incoming contact/speed/spin、base spawn/travel、landing-aim profile；action order 必须与
   loader/control-plane slot 相同；UID/slot 不进入 fresh actor observation；
2. level 0 使用 manifest 的 non-zero initial std；各轴先独立找 marginal frontier，再调 joint
   `rho`。10% 只指 safe closed policy non-return，solver reject 与 unsafe 分账；
3. 旧 `fh_loop` 不进入新 training view；旧 bytes 留档，不删除。`fh_loop_high` 必须产生新的
   upper/full 两件和各自证书，不能把 source NPZ 或旧正手证书直接塞进 manifest；
4. 新正手的 `[0,0] / [-0.05,0] / [-0.10,0] m` 是整动作/contact/base profile 对照。负 X 远离桌，
   取 upper/full 共同通过中最小后移；station 选择完成后才冻结 manifest；
5. source anchor 只作 compiler diagnostic。正式 post-retime `t_hit` 用动作特定 behavior/contact
   authority；另报 `t_cycle`、physical `right_racket` site speed 和 ready→recovery 全轨无撞桌；
6. `scripts/certify_task_first_action.py` 的 `template`、`scan-collisions`、`certify` 只组织
   内容寻址 reference checks，不替代 compiler/bank/dynamics/reference-return authority。当前输出
   `diagnostic_smoke_authorized=false` 与 `training_authorized=false`，不能据此启动 simulator 或
   trainer；
7. 全 bank 的 manifest/file SHA、motion bytes、balanced sampling、fresh N1 actor
   `action_ball_table_pose_twist_heading_task_teacher_start_v2`（固定 194-D；老师开始倒计时
   替换旧 N1 常量 one-hot 槽，不包含 UID/slot）、
   fixed-action solver/physics canonical payload、single-use birth/task receipt、curriculum Gate 与
   effective Reward receipt 一起进入 checkpoint hard contract；manifest metadata 不得自授权。
   旧 N-dependent `...heading_task_n<N>`（`193+N`）和
   `...teacher_start_n<N>`（`194+N`）只作历史 checkpoint/receipt 兼容，不得续成 fresh 合同。
   N5/N73 必须先冻结固定宽 teacher-trajectory/ball/task/validity/history ABI，并过
   N2/N3 共享策略容量/串扰验证；teacher trajectory 已表达动作，不另加 intent/ID。
   此前不得用 N1 v2 发射。

新正手缺 upper/full、grounded collocation trace 或任一时序/速度/撞桌 Gate 时，到此停止；
不执行第 16 步 trainer smoke。完整前置见
[action-ball 实验](../experiments/2026-07/EXP-ACTION-CONDITIONED-BALL-FIRST-20260727.md)。

## 发射之后

逐族累计击球机会到 **500** 次仍零合法回球 ⇒ 报警；**5000** 次 ⇒ 中止训练
（`utils/my_on_policy_runner.py:41-46`）。逐侧比率样本不足时是 `None` 不是 `0.0`。
报数规则见[结果判读](read_and_report_results.md)：**只报总数 = 漏病**。

## 这一页为什么存在

2026-07-26 一天内同一条链炸两个洞：①相位/瞄向没扫（`--phase-scan` 2026-07-05 就建成，
canonical **从未跑过** ⇒ 11 片里 9 片一帧都回不了球）；②目标框没查物理
（正手击球帧 `z=0.694 m` < 桌面 `0.76 m`，四条臂正手回球率恒 `0.0000` 跑了几千迭代，
汇报的"45% 回球率"是和健康反手平均出来的，**该读数作废**）。
知识此前散在四处：`research/motion_library_topp_recovery_2026-07-11.md`（唯一写了 phase-scan 是门）、
`run_motion_face_shift.md` §5+§7（编译后 16 项）、
[canonical 实验 §12](../experiments/2026-07/EXP-MOTION-CANONICAL-LIBRARY-20260723.md)（排期）、
`research/PREREG_canonical_formalization_draft_20260723.md` §4（逐格 10 门）。没人把它们串起来。

## 开放问题（留给 Franco）

1. `rebind_physical_strike.py`（第 8 步）只在 pod2 `newmotion_matrix_20260726/`，**未收编进仓库**，
   无版本管控无单测。收不收？收到哪？
2. `check_perclip_*_sampling.py` 补不补退出码门 + 改从 `geometry.py` 读常量？
   （见[应当变成闸门的规则](rules_that_should_be_gates.md) P0 #1/#2）
3. `rally_yaw_deg`（每片击球方向）要不要升为 registry 一等字段？
   v4rg 有、canonical 没有，这个不对称就是第 7 步事故的根因。
4. canonical 锚点的题库生成"排队"中；在它落地前 canonical 谱系只能用 uniform + FK 逐 clip 箱，
   第 13 步对该谱系不适用。这个状态维持多久？
