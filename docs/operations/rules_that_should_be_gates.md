# 应当变成代码闸门而不是文字的规则

**没读文档的 session 绕不过一个 fail-closed 检查，但可以绕过任何一段话。**
下面每条给出：规则 → 确切检查 → 落在哪个文件 → 现在为什么拦不住。
按"违反代价 × 静默程度"排序：**排在前面的，违反了不会报错，只会让某个数字悄悄变假。**

按惯例：构造期违规 `raise ValueError`（不是 warning），产出物门用退出码 `2`。

---

## P0 — 违反了完全静默

### 1. 逐片目标框采样检查必须有退出码

- **规则**：目标框在桌面以下 = 命令机器人往台子里打球，该侧回球率永远 `0.0000`。
- **确切检查**：`check_perclip_pos_sampling.py` 与 `check_perclip_vel_sampling.py` 的
  `main()` 末尾统计 `below table surface` 占比与 `x_hi > vb_table_near_x` 的框；
  任一片 `> 0%` ⇒ `sys.exit(2)`。速度侧：前向下界 `≤ 0` ⇒ `sys.exit(2)`。
- **落点**：`hope_training/whole_body_tracking/scripts/check_perclip_{pos,vel}_sampling.py`
- **现在为什么拦不住**：两个脚本**只 print，`main()` 里没有任何 `sys.exit`**——
  打印 `below table surface: 83%` 之后照样返回 0。挂 CI 上等于没挂。

### 2. 物理常量必须只有一份

- **规则**：桌面高度和球半径是一等公民代码，**读它的人不许自己抄一份**。
- **确切检查**：删掉 `check_perclip_pos_sampling.py:8-9` 的
  `TABLE_H = 0.76` / `PELVIS_Z = 0.93`，改为从
  `tasks/table_tennis/geometry.py`（`TABLE_HEIGHT = 0.76`，第 39 行）与
  `tasks/table_tennis/physics/params.py`（`ball_radius = 0.020`，第 50 行）读取；
  加一个单测断言"任何脚本里不出现字面量 `0.76` 作为桌面高度"。
- **落点**：同上 + `tests/`
- **现在为什么拦不住**：常量若改，硬编码副本不会跟着改，检查会静默用旧值放行。
  这正是原始缺陷 2 的形状——**常量只被球仿真读，检查侧各抄各的**。

### 3. 相位登记必须先有扫描证据

- **规则**：空挥片没有真实 contact truth，触球相位只能来自可回球性扫描，
  不能来自"窗内前向速度最大帧"这类启发式。
- **确切检查**：registry / `strike_annotations.yaml` 加载器要求每条 clip 的
  `train_phase_candidates` 必须存在**且**带一条扫描收据（工具版本 + clip SHA-256 + 该帧扫描得分）；
  得分 `< 0.5` 或缺收据 ⇒ 构造期 `raise ValueError`。
  额外：选中帧若落在扫描窗**边界**上 ⇒ 红灯（`argmax` 撞边界正是 §7.4 的病根）。
- **落点**：`gen_stage1_questions.py` 的 registry 读取路径 +
  `tasks/tracking/mdp/hope_commands.py` 的 clip 绑定构造
- **现在为什么拦不住**：`--anchor train-candidate` 缺字段时会 `SystemExit`，
  但**默认 `--anchor annotated` 不要求任何扫描证据**，所以手搓相位一路畅通。
  代价：11 片里 9 片一帧都回不了球，无人发现。

### 4. 击球方向（`rally_yaw_deg`）必须是 registry 一等字段

- **规则**：把**骨盆首帧偏航**归零 ≠ 把**击球方向**摆正。乒乓是侧身打的。
- **确切检查**：registry schema 要求每条 clip 有显式 `rally_yaw_deg`；
  缺失 ⇒ 拒绝加载。附带断言：触球帧世界系拍面法向 `|n_x| ≥ 0.8`
  （参考 v4rg 实测 0.86–0.94；病态 canonical 只有 0.36–0.68）。
- **落点**：`canonical_motion_registry.py` + `cfg/strike_annotations.yaml` schema
- **现在为什么拦不住**：v4rg 谱系有这个字段，canonical 五动作**没有**，
  于是被统一给了一个骨盆推出来的 −72.552°。这个不对称就是根因，schema 不查它。

---

## P1 — 违反了要等很久才发现

### 5. 出生位必须和参考第 0 帧对得上

- **规则**：出生位与参考第 0 帧的末端距离超过终止阈值 ⇒ 每个 env 第 1 步就判死。
- **确切检查**：env 构造期算 `‖ee_pos(出生位) − ee_pos(参考 frame 0)‖`，
  与 `ee_body_pos` 终止阈值（0.25 m）比较；超阈值**且**预备冻结期为 0 ⇒ `raise ValueError`。
- **落点**：`tasks/tracking/mdp/hope_commands.py` 或 env cfg `__post_init__`
- **现在为什么拦不住**：2026-07-26 canonical probe 出生在旧 v4rg 站姿（拍 0.882 m），
  参考第 0 帧的拍在 1.229 m，差 0.347 m > 0.25 m，**跑了 2500 迭代零学习**
  （回合长恒 1、回报恒 −35.92）才被人看出来。

### 6. 导出必须烤进和训练同一批片子

- **规则**：导出 ONNX 里烤的动作片必须与该 checkpoint 训练时用的逐字节相同。
- **确切检查**：导出脚本从 run 的 `params/env.yaml` 读 `motion_file` 并比对
  实际传入的片子 SHA-256；不符 ⇒ `exit 2`。
- **落点**：`export_onnx_*.sh` / `standalone_onnx_export.py`
- **现在为什么拦不住**：现在是文字铁律 + 一条"请自己 grep wandb debug.log"的建议。
  一次 v4 导出 hopex 谱系的 checkpoint，deploy-faithful 门速度误差 2.6 m/s。

### 7. `motion_file` 路径写错不许静默回退

- **规则**：路径写错时训练**静默 fallback 到 WandB registry**，于是训的不是你以为的片子。
- **确切检查**：显式传了 `motion_file` 时，路径不存在 ⇒ `raise FileNotFoundError`，
  绝不回退 registry。回退只允许在完全没传 `motion_file` 时发生。
- **落点**：`train.py` 的动作解析路径
- **现在为什么拦不住**：只是 [runbook 第 20 条](../runbook.md)的一句话。

### 8. 冒烟摘要必须包含 WARN 行

- **规则**：只抓预期信号 = 确认偏误。落地警告就是这么漏的。
- **确切检查**：probe 收口器（`full_scene_probe_runtime.py`）自己跑
  `grep -cE 'WARN|Error|Traceback'` 与 `grep -Fc 'q_des CLAMP ACTIVE'`，
  把两个计数写进 `probe_receipt.json`；`Error|Traceback > 0` 或 `CLAMP ACTIVE == 0`
  ⇒ `status != passed`。
- **落点**：`hope_training/whole_body_tracking/scripts/full_scene_probe_runtime.py`
- **现在为什么拦不住**：现在是每份 ops 页里一条要人手敲的 `grep`。

### 9. argv 只能来自引号完好的命令文件

- **规则**：从 `ps` 抄 argv 会丢 shell 引号，Hydra mapping 被打碎。
- **确切检查**：队列渲染器把完整 argv 写进 `arms/<job>/argv.txt` 并记其 SHA-256 到
  `run_binding.json`；发射器只接受 `--argv-file`，不接受行内长命令串。
- **落点**：`run_lean_training_queue.py` / 各 `run_phase1_*_queue.py` 渲染器
- **现在为什么拦不住**：这条规则 2026-07-26 才立，**只存在于
  [`EXP-V2-REWARD-FREEZE-20260726.md`](../experiments/2026-07/EXP-V2-REWARD-FREEZE-20260726.md) 一份实验记录里**，
  没有传播到 runbook 或任何 ops 页；同一波内**又犯了一次**（`r2_argv.txt` 模板参数未带引号）。

---

## P2 — 已经是闸门，记在这里防止被拆

| 规则 | 闸门在哪 |
| --- | --- |
| 命令的击球点必须在桌面 + 球半径之上 | `hope_commands.py:1389-1415` `_assert_contact_clears_table`（构造期 `ValueError`） |
| 目标速度必须朝对手方向 | `hope_commands.py` `_assert_target_velocity_points_forward` |
| 逐侧零回球报警/中止 | `utils/my_on_policy_runner.py:41-46`（500 次机会报警，5000 次中止） |
| 逐侧比率样本不足时是 `None` 不是 `0.0` | `my_on_policy_runner.py:49-51` `_ratio_or_none` |
| canonical 消费必须有代码根信任集证书 | `canonical_motion_admission.py`（出厂为空 = fail-closed） |
| 出题答案必须过 torch 闭环自检 | `gen_stage1_questions.py --check`（默认开，不许关） |
| 击球窗不能跨件拼接 | `canonical_protected_window.py` 内容摘要 |
| 进程组精确停止 | `exact_process_group.py`（PID 复用/starttime 变化 ⇒ KILL 前 fail-closed） |

---

## 怎么用这一页

1. 做工程 pass 时从 P0 往下清，每关掉一条就把它移进 P2 表并注明闸门位置。
2. 新写规则时先问："这能不能是个闸门？"能 ⇒ 写进代码并登记 P2；
   暂时不能 ⇒ 写进这一页的 P0/P1，**不要只写进散文**。
3. 本页只列**能机器判定**的规则。需要人判断的（该不该买第二个 seed、这个结论保质期多长）
   留在 [runbook](../runbook.md) 和 [结果判读](read_and_report_results.md)。
