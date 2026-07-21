# spdmix v2 多 clip 烤入列表可行性（2026-07-22）

结论：**多 clip 列表本身受支持，但 6-clip 变速列表被两道"只有 2 个 clip"的硬绑定挡死——不能纯配置上，需按最小改造清单动代码+题库。**
人话：装载器吃得下 6 个 clip，采样/相位/planner governor 都按 N 泛化；但"哪个 clip 是正手"和题库对账写死了 2 clip，正手 1.0/1.2 变体会被当反手，题库当场 fail-closed。

## 源码证据（逐条，行号为 2026-07-22 main 同步工作树）

1. 列表语法：`motion_file`/`motion_file_2` 由 `_configured_items` 拼成一个平铺列表（scripts/train.py:856-869、4477-4504），`MotionLoader` 逐个装载并记录 `seg_start`/`seg_len`，任意 N 个 clip、要求各 clip fps 完全相等（mdp/commands.py:259、319-338）。烤入片段 fps=50/帧数不变，满足。
2. clip 选择：每次挥拍（wrap/reset）对 `num_segments` **均匀随机**采 clip、从该 clip 首帧起播（commands.py:1440）；`clip_switch_prob` 只是中途弃拍重采的概率（commands.py:2386-2395），本波恒 0。6 clip = 正反手各 1/2、手内三档速度均匀——采样语义天然成立，无需新开关。
3. `strike_phase_per_clip`/`mount_normal_sign_per_clip`：长度必须 == 加载 clip 数，否则当场报错（hope_commands.py:1256-1272、1273-1290）。6 clip 配 `[0.471,0.471,0.471,0.338,0.338,0.338]`、`[1,1,1,-1,-1,-1]` 即可，纯配置。
4. planner task-revision governor 与多 clip **兼容**：strike_step 逐 env 按其 clip 的 `seg_start + phase*(seg_len-1)` 求出（hope_commands.py:2355-2380）再交 `begin_planner_task`（commands.py:940-1009），governor 内无 clip 数假设；烤入保持帧数/fps/触球帧不变、每 clip 时间律自洽，`speed_scale_range` 保持 `[1.0,1.0]`、不设 `speed_scale_per_clip`，不触发 commands.py:499-503 的单时钟守卫。
5. **硬绑定一：swing_sign 写死 clip0=正手、其余全反手**（hope_commands.py:1837、1974、2207 均为 `clips == 0 ? +1 : -1`；uniform 目标 y 侧符号同病 1527）。6-clip 列表里正手 1.0/1.2 变体（clip 1、2）会被判成反手：obs `swing_type`、拍面配对、目标侧全错——不崩但训错，最危险的一类。
6. **硬绑定二：题库按 2-clip 内容寻址、SHA 对账 fail-closed**。`load_question_bank` 要求 bank `clip_order == ("forehand","backhand")`（调用处不传自定义名，hope_commands.py:243；loader stage1_question_bank.py:187）；bank 张量 (C=2,Q,3)，`select_questions`（stage1_question_bank.py:579）直接拿 clip_id 当下标（clip 2-5 越界）；face_command 首次应用即跑 `validate_runtime_motion_contract`（stage1_question_bank.py:121-170，自 hope_commands.py:2029 触发）：clip 数 2≠6 直接 ValueError，且逐 clip 对账 **motion 文件 SHA-256**——烤入 npz（含 1.0 档）重序列化后 SHA ≠ bank 记录的 cal 原件，当场拒绝。本波 `face_command=true` 必须带题库，绕不开。
7. 次要：per-clip 指标账本键写死 `{0:forehand,1:backhand}`（hope_commands.py:423、453-471），clip 2-5 的分手指标会静默缺席（不崩、判读盲）。

## 最小改造方案（若要真上 6-clip 混合；须另立代码波预注册，不塞进 Wave Q）

1. `MotionCommandCfg` 加 `clip_family_per_clip`（长度=N、值 0/1、空=恒等旧行为），hope_commands.py 四处 `clips == 0` 换成按 family 查表；per-clip 账本按 family 聚合。
2. 题库改按 family 寻址（`select_questions(bank, family[clip], …)`），`validate_runtime_motion_contract` 加"运行时 clip→bank clip"映射：烤入 manifest 已证触球行逐位相同（`row_bitwise: true`、帧数/锚帧不变），答案可按 family 原样复用——照 `scripts/rebind_stage1_question_bank_physics_contract.py` 先例新写 motion 重绑脚本，把 6 个烤入 SHA 绑进 bank 元数据。
3. 配置层（改造合入后）：`motion_file=[fh0.8,fh1.0,fh1.2]`、`motion_file_2=[bh0.8,bh1.0,bh1.1]`——反手 1.2 物理不可行（envelope `max_feasible_ratio=1.1931`，offending left_knee），已烤最近可行档是 **1.10**；strike/mount 表扩 6 位。
4. 改动面：commands.py、hope_commands.py、stage1_question_bank.py、新重绑脚本、单测——代码波量级。

## 裁定

按预注册纪律**不硬上**：spdmix v2 维持 blocked，资产不上传、config/runner/tests 不改；在线 v1 被 governor 拒绝（commands.py:499-503）的记录保留为历史，见 EXP-P1-INTEL-WAVE-20260721.md spdmix 节。
