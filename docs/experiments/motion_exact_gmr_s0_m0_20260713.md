# S0/M0 exact GMR 与横移末态脚距

- 状态：`preregistered_not_executed`（source/static gate 通过；runtime `inspect/consume` 未运行）
- 人类负责人：Franco
- 执行者：Codex
- 证据等级：E1（exact runtime closure、两份 static plan 与 mutation/no-clobber 测试）；尚无 GMR 输出

共享缩写见[术语与人话对照](../DEFINITIONS.md)。本卷只回答：已经完成 exact donor canonical-beta 的
[`S0`](../DEFINITIONS.md) 高点拍压与 [`M0`](../DEFINITIONS.md) 四条横移老师，能否在不混批、不覆盖旧制品
的条件下进入同一个 exact A3 GMR 源码/runtime；M0 能否同时生成一份预先冻结、不会把“收脚变窄”判成
成功的机器人坐标脚距报告。它不回答动作安全、击球效果、动力学、训练或真机。

## 两批为什么独立

S0 与 M0 使用同一份只读 runtime closure，但有不同的 input manifest、output root 和 completion manifest。
任一批失败只在自己的新 root 留下日志/失败 binding；另一批不被占名，也不能用 `--asset-id` 跳过失败项。
两份批次计划是：

- `configs/motion_exact_gmr_s0_prereg_20260713.json`：一条反手高点拍压；
- `configs/motion_exact_gmr_m0_prereg_20260713.json`：左移 1/2、右移 1/2 四条候选；
- `configs/motion_s0_m0_exact_gmr_runtime_20260713.json`：共享、内容寻址的 GMR/Python/A3 合同。

consumer 只有 `static`、只读 `inspect` 和一次性 `consume`。`consume` 要求 output root 原先不存在，逐条
保存 converter output、log、structural audit 和 binding；所有文件 fsync 后，才最后发布
`completion_manifest.json`。进程中断或某条失败时 completion 不存在，因此 partial 不能被误认成完成，也
不能删除后用同一计划重跑。

## exact source/runtime 闭包

共享合同要求 clean ignored GMR checkout 同时匹配 commit 与 tree OID，并验证 recovery bundle。converter、
SMPL loader、neutral SMPL-X NPZ、A3 retarget MJCF 和 `smplx_to_a3` mapping 都必须逐文件匹配 bytes/SHA；
Python `3.10.20`、规范化 `pip freeze --all` 以及 `numpy/torch/mujoco/smplx/scipy` import 一起绑定。GMR 这条
路径是优化式 converter，没有 checkpoint CLI 或 checkpoint runtime read；机器合同因此明确冻结空
checkpoint 集，而不是伪造一个模型权重。

2026-07-14 的第二轮低频只读复核把 16 项缺口全部变成 exact 字段：clean GMR tree 仍为
`dc32626643e1a35820ec3ccf00e4d20b590d77cf`；package init、motion retarget、params、kinematics、viewer、
data loader 与 neck retarget 七个 import module 都绑定 realpath/bytes/SHA；`smplx_to_a3.json` 的 exact path
为 `general_motion_retargeting/ik_configs/smplx_to_a3.json`。Python 是非 symlink regular file
`/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10`，`17,331,920` bytes、SHA
`dd9eb336...aa55`；完整 pip origin string 与规范化 `pip freeze --all` SHA `97c66009...18ff` 已绑定，
`importlib.util.find_spec("xrobot_utils")` 明确为 absent。

两份 batch plan 与 shared runtime 均为 `preregistered_not_executed`；blocked-only 本地 receipt 和
`required_unresolved_evidence` 已删除。shared runtime SHA 为 `cb9b01b9...0d45`，S0/M0 plan SHA 分别为
`a5c65e9e...8917` / `137b38c9...a4bc`。两次 host `static` 都输出 `PASS`；这只解除 source/static blocker，
不是 ignored runtime、私有 PT 或 converter 行为已经通过。

## 31 关节、两套 body order 与足点

GMR `dof_pos[:,31]` 到 tracked canonical A3 `qpos[7:38]` 使用逐 index、逐 name 的显式 bijection。direct
retarget XML 解析得到 31 个 hinge 和 32 个 body；本次名字/顺序恰与 canonical 列表相同，但两份证据仍各自
绑定，不能因为结果相同就互相替代。`inspect` 将重读 exact XML 并逐项复核；canonical A3 FK 还绑定整个
76 文件 vendor model tree，而不只绑定顶层 XML。

一个重要负结果是：direct `a3_mocap.xml` 的完整 site inventory 精确为空，`left_foot/right_foot` 都 absent。
旧 consumer 强迫 retarget XML 拥有 canonical 足点，会造成假阻塞；现在 source 合同明确绑定空 inventory 与
左右 absent，任何后来出现的 site 都算 runtime drift。下面的末态站距只在绑定的 canonical vendor model
上做 FK；mutation test 明确禁止把 vendor 足点抄进 retarget 字段。

M0 足点固定为 canonical vendor MJCF：

- 左足：site `left_foot`，parent `left_ankle_roll_Link`，local position `[0.04,0,-0.067] m`；
- 右足：site `right_foot`，parent `right_ankle_roll_Link`，local position `[0.04,0,-0.067] m`。

每个 GMR sample 的时间严格为 `i/30 s`；人工 `ready_before/ready_after` 闭区间在看结果前已经展开成 exact
sample index。以 frame-0 pelvis local `+X` 的地面投影消掉公共 heading，并减去共同 frame-0 root XY；然后
在两个窗口分别求
`d_xy = right_foot_xy - left_foot_xy` 的逐坐标 median。报告同时保留 `+X` 前后错位和 `+Y` 横向分离，
不会只报一个欧氏距离。

## 冻结判定

运行前所有 initial/terminal vector、分量值和 `stance_passed` 都是 `null`。阈值先冻结为：

- 前后分量绝对变化不超过 `0.03 m`；
- 横向 signed 分量绝对变化不超过 `0.03 m`；
- 初始横向绝对分离至少 `0.05 m` 且左右符号不翻转；
- 末态横向绝对分离最多只允许 `0.005 m` 的数值性变窄。

最后一条是独立硬门：即使“变窄 2 cm”仍落在宽松的 3 cm ready-set 分量带内，也必须失败。这样 3 cm
描述恢复姿态的整体近似，5 mm 单独阻止双脚并拢/明显收窄钻空子。四条都只是候选；通过这张结构卷也不
等于选出横移老师。

S0 的 `observed_ball_contact` 与 `strike_effectiveness` 在 GMR 后仍必须是 `null`，且不得消费拉球题。后续需
另建高球拍压题族，再依次通过 schema-2、L0/L1、桌网整轨余隙和动力学。

## 验证与下一步

专项测试覆盖 closed-window sample 映射、前后/横向分量、独立防变窄门、no-clobber、失败无 completion、
成功 report-last：

```bash
python3 -m pytest -q tests/test_run_motion_s0_m0_exact_gmr.py
```

当前专项为 `13 passed`，基于最新 main 的仓内 `tests/` 回归为 `851 passed, 10 skipped`；两份真实 batch plan 的 `static`
都为 `PASS`。新增负测覆盖 canonical site 冒充
retarget、retarget site 突然出现、左右 absent 被改写、joint-order runtime drift；原有 no-clobber、失败无
completion、成功 report-last 继续通过。

下一步只允许在 code review 后用绑定的 CPU runtime 分别运行只读 `inspect`；本分支不连 Pod、不读取私有
PT、也不执行 `consume`。`inspect` 通过后才可另行授权两个 no-clobber batch 的一次性 consume。完整命令见
[操作文档](../operations/run_motion_s0_m0_exact_gmr.md)。
