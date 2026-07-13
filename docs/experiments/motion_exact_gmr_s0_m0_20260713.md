# S0/M0 exact GMR 与横移末态脚距

- 状态：`blocked`
- 人类负责人：Franco
- 执行者：Codex
- 证据等级：E1（consumer、blocked-runtime schema 与 mutation/no-clobber 测试）；尚无 GMR 输出

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

2026-07-14 的低频只读网络复核已取回 clean GMR tree
`dc32626643e1a35820ec3ccf00e4d20b590d77cf`、retarget MJCF `19741` bytes/SHA、mapping
`3581` bytes/SHA、四个关键 import 文件的 bytes/SHA、Python binary SHA 与按逐字节排序规范化的
`pip freeze --all` SHA。原始回执为 `5621` bytes、SHA
`32c90a8882be02e5bd7260a8531f1cc0c5b212e88663c3f5e3a7a8aec13c8236`。

同一次 XML parser 的 body/site 段落落入传输截断区，因此不能把 tracked canonical 32-body order 或历史
31-joint order抄进 retarget XML 字段。回执也没有返回四个已知 import 文件的绝对路径、三个 eager-import
模块的文件 binding、mapping 绝对路径、Python executable path/bytes、完整 `pip --version` origin string
和 `xrobot_utils` resolution。两份 batch plan 已冻结为 `preregistered_not_executed`，共享 runtime 仍明确为
`blocked_pending_exact_ignored_gmr_source_closure`；其 `required_unresolved_evidence` 逐 JSON pointer 列出
16 项只读补证。`static` 先验证回执/schema/tool binding，再以 rc=2 列出缺口，不能称为 ready 或启动。
当前 shared runtime SHA 是 `25e21b3c...c9563`；S0/M0 batch plan SHA 分别为
`cce033f6...6e3a9` / `58b48f48...1c64a`。

## 31 关节、两套 body order 与足点

GMR `dof_pos[:,31]` 到 tracked canonical A3 `qpos[7:38]` 最终必须使用逐 index、逐 name 的显式
bijection。retarget MJCF 与 canonical vendor MJCF 的 body order 必须分别冻结，不能假定名字相同；当前
retarget order/bijection 仍为空值，正是阻塞项。`inspect` 将各自解析 XML 后再验 31 关节 bijection。
canonical A3 FK 还绑定整个 76 文件 model tree，而不只绑定顶层 XML。

retarget XML 的足点 name/parent/local position 也必须单独直读并绑定；它们不能从 canonical 值反推，也不
要求与 canonical 足点逐字相等。下面的末态站距只在绑定的 canonical vendor model 上做 FK。

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

当前专项为 `12 passed`，全仓为 `645 passed, 9 skipped`。两份真实 batch plan 的 `static` 都按预期 rc=2，
并逐项打印同一份 unresolved 列表。

下一次网络窗口只允许补齐机器清单里的字段、移除 blocked-only receipt/list、把共享 runtime 切为
`preregistered_not_executed`、重算 tool/runtime/plan SHA 并重新运行 `static`；在 source review 之前不得运行
`inspect/consume`。完整命令见[操作文档](../operations/run_motion_s0_m0_exact_gmr.md)。
