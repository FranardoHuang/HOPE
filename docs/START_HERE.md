# HOPE 开发者与 Agent 起始页

这是改代码、移动资产或分配工作前必须先读的文件。仓库在策略侧已经走通过一次
Isaac → MuJoCo → 真实 A3 链路：统一挥拍策略于 2026-07-02 首次完成仿真到真实（sim-to-real）迁移，
仅覆盖正手，且真实行为与 MuJoCo 一致。但闭环仍未完成：感知链路
（动捕 mocap → 规划器 planner → 部署运行器 deploy runner）尚未接通，还没有被接受的质量基线，
数据采集/物理标定阶段也尚未运行。

读完本文件后，用 [`INDEX.md`](INDEX.md) 作为一站式路由。它列出了训练、实验、MuJoCo、Gate3、
动作、规划器（planner）、部署、接口和本地资产任务各自的最小阅读集。

## 项目目标

在 Agibot A3 上构建与 HITTER 接口兼容的人形机器人乒乓球系统：

1. 复现 HITTER 中有用的系统约定：基于模型的来球规划，加上 RL 全身控制。
2. 让实现能够配合真实 A3、ChingMu/VRPN 动捕、MuJoCo、Isaac 和 Agibot 部署材料运行。
3. 以该基线识别并解决论文的盲点，例如旋转、二跳、短球/深球、发球和对手适应。

目标不是盲目复制 HITTER 的每个细节，而是在适配本硬件的同时保留其接口与评测纪律，
并改进论文中的薄弱之处。

## 当前 Gate 状态

| Gate | 状态 | 主文档 |
| --- | --- | --- |
| G00 Materials and harness | Partial | [G00](gates/G00_materials_and_harness.md) |
| G01 Real preparation | Partial | [G01](gates/G01_real_preparation.md) |
| G02 Data acquisition | Partial | [G02](gates/G02_data_acquisition.md) |
| G03 Data processing and physics calibration | Partial | [G03](gates/G03_data_processing_and_physics_calibration.md) |
| G04 Sim modeling in MuJoCo and Isaac | Partial | [G04](gates/G04_sim_modeling_mujoco_isaac.md) |
| G05 Isaac training first loop | Partial | [G05](gates/G05_isaac_training_first_loop.md) |
| G06 Isaac-to-MuJoCo parity | Partial | [G06](gates/G06_isaac_to_mujoco.md) |
| G07 MuJoCo-to-real deployment | Partial | [G07](gates/G07_mujoco_to_real.md) |
| G08 Blind-spot improvements | Research track | [G08](gates/G08_blind_spot_improvements.md) |

状态标签：

- `Done`：本仓库中已有通过验证的证据。
- `Partial`：已有材料或代码，但 Gate 尚未得到完整验证。
- `Not complete`：已有脚手架，但尚未演示被接受的闭环。
- `Not started`：尚无有意义的实现路径。
- `Research track`：基线完成后仍有意保持开放的改进研究线。

## 如何开始一项具体任务

做具体工作前不必读完所有配置文档。按任务优先的路径进入：

1. 从上表选择相关 Gate。
2. 只读该 Gate 文档，重点看 `Current State`、`Acceptance Criteria` 和 `Next Steps`。
3. 打开该 Gate 链接的操作（operation）文档。
4. 按操作文档中的 `Task Setup` 或配置（setup）章节执行。
5. 仅当操作文档链接到接口/资产文档，或你正要修改对应约定时，再打开这些文档。

全面入门时，阅读 [PROJECT_MAP.md](PROJECT_MAP.md) 和 [DEFINITIONS.md](DEFINITIONS.md)。
文件夹或资产策略有变化时，阅读 [ASSET_POLICY.md](ASSET_POLICY.md)。

### 当前文档分工

| 问题 | 阅读/更新位置 |
| --- | --- |
| 当前完整训练怎样连起来、处于哪个课程阶段、各主题的问题/解法/效果/差距是什么？ | [NOW.md](NOW.md) |
| 具体假设、实验运行（run）、失效结论和决定是什么？ | [experiments/](experiments/README.md) |
| 哪项重要能力或根因修复已经进入 `main`？ | [TIMELINE.md](TIMELINE.md) |
| 哪个可复现 Gate 已通过或仍被阻塞？ | 对应的 `gates/G*.md` |
| 应该运行什么命令？ | 对应的 `operations/*.md` |

完整路由表见 [`INDEX.md`](INDEX.md)；贡献者不应再依赖聊天消息才能知道要读哪些文件。
正常入门时不要阅读 `docs/experiments/archive/` 下的旧流水归档。

### 可读性、`NOW` 权威与排程

- 每个 `run_name`、命令行标志（flag）或缩写在一份文档/记录中首次出现时，必须同时给出人话解释，
  并链接到 [DEFINITIONS.md](DEFINITIONS.md) 中的定义；如果定义尚不存在，先补定义。
- `origin/main` 上的 [NOW.md](NOW.md) 是唯一运行态权威。功能分支（feature branch）里的 `NOW`
  只能作为提案，
  不能改变当前优先级、认领状态或已采用配置；合入前必须基于最新 `origin/main` 逐项对账并解决冲突，
  只有进入 `main` 后才生效。
- [NOW.md](NOW.md) 的统一工作队列是全项目唯一的优先级账本。实验记录、Gate 文档、个人笔记和
  Agent 任务可以记录局部步骤与证据，但不得维护与其竞争的影子优先级队列。

**开始工作项前**，先核对 `origin/main` 上 [NOW.md](NOW.md) 的最新队列条目，再用
`Human owner + Executor + branch` 认领；认领只有进入 `main` 后才生效。`Human owner` 必须是人，
Claude/Codex 只表示执行来源。实验点火前，先在 `docs/experiments/` 新建或更新对应记录。

## 重新实现节奏

`reimplement.md` 是长篇补充运行手册，不是首要配置入口，也不是另一份项目计划。应通过 Gate 文档使用它：

1. 找到与当前工作匹配的 Gate。
2. 先读 Gate 文档和操作（operation）文档。
3. 使用其中引用的 `reimplement.md` 步骤查命令细节。
4. 验证 Gate 的验收标准。
5. 在同一分支更新 Gate 文档及按路由命中的现状/实验/接口/操作文档；
   `docs/PROGRESS.md` 只追加带链接的简短条目。对于 `docs/NOW.md`，功能分支内容始终只是提案，
   必须按上面的 main 权威规则对账后才可生效。

如果 `reimplement.md` 的某个阶段/步骤（phase/step）与 Gate 文档冲突，以 Gate 文档为当前标准；
继续之前必须同时
更新这两个文件。

## 实现原则

使用基于第一性原理的 Gate，不要盲目复制论文或供应商示例：

1. 先定义接口约定，再优化实现。
2. 坐标系、关节顺序、策略观测、动作和资产路径各自只保留一个真源。
3. 先测量或验证，再调参。
4. 分开管理源代码、整理后的数据、运行时资产和外部参考。
5. 优先使用小型、可复现的 Gate 检查，不做大型但无文档的演示。
6. 把 HITTER 作为基线系统约定，而不是限制改进的边界。
7. 真实硬件风险的推进速度不得超过安全文档和空跑（dry-run）的验证进度。

## 文档更新规则

每个人和每个 Agent 都必须让文档与实际工作同步。

发生以下任一情况时，必须在同一分支更新文档；`docs/NOW.md` 仍遵守上文的 main 权威规则：

1. Gate 目标变化。
2. Gate 状态变化。
3. 发现或解除新的阻塞。
4. 增加新的文件夹或制品类型。
5. 接口发生变化：坐标系（frame）、topic、message、关节顺序、观测、动作、模型格式或运行时资产路径。
6. 配置、构建、测试、训练或部署命令变化。
7. 某个仅存在于本地的资产变成复现结果的必要条件。

最低更新要求：

- 更新相关的 `docs/gates/G*.md`。
- 实验设计、实验运行（run）、结果或决定变化时，更新 [experiments/](experiments/README.md)。
- 当前训练配置（setting）、成绩表、课程阶段、组合小目标、采用/拒绝决定、阻塞或人类责任人变化时，
  提议更新
  [NOW.md](NOW.md)，并在合入前按最新 `origin/main` 对账。
- 只有值得记录的逻辑变化已经真正进入 `main` 时，才更新 [TIMELINE.md](TIMELINE.md)。
- 文件夹职责变化时，更新 [PROJECT_MAP.md](PROJECT_MAP.md)。
- Git、忽略规则、Git LFS 或 submodule 策略变化时，更新 [ASSET_POLICY.md](ASSET_POLICY.md)。
- 坐标系、ROS topic、message、关节顺序、观测、动作或运行时接口约定变化时，更新
  `docs/interfaces/` 下的相应文件。
- 配置、构建、测试、训练或部署命令变化时，更新 `docs/operations/` 下的相应文件。
- 在 [PROGRESS.md](PROGRESS.md) 添加简短的带日期条目。

不要把项目状态藏在聊天记录里。未来贡献者或 Agent 需要知道的内容，必须写入仓库。

## 快速入口

- 一站式任务/文件路由：[INDEX.md](INDEX.md)。
- 当前训练闭环、现行课程阶段、分动作单拍/连续成绩卡及唯一工作队列：[NOW.md](NOW.md)。
- 实验登记与模板：[experiments/README.md](experiments/README.md)。
- 经过筛选的重要主线（mainline）变化：[TIMELINE.md](TIMELINE.md)。
- 新电脑或新 Agent 启动：先用本文件作为索引，再进入任务对应的操作（operation）文档。Isaac 训练使用
  [operations/run_training.md](operations/run_training.md)；全新的 `git clone` 缺少被忽略/私有资产时，使用
  [operations/setup_local_sync.md](operations/setup_local_sync.md)。只有 Gate 或操作文档指向
  `reimplement.md` 的具体步骤时，才把 [reimplement.md](../reimplement.md) 当作长篇运行手册使用。
- 端到端历史运行手册（环境创建、GMR/GVHMR 动作流水线、A3 资产准备、训练、部署）：
  [reimplement.md](../reimplement.md)。`operations/*` 文档是按任务整理后的真源；只有操作文档或
  Gate 文档引用了具体步骤（例如 `Step 12.7`）时，才用 `reimplement.md` 补充命令细节。
- 环境配置按任务区分。先读任务对应的操作文档；
  [operations/setup_environments.md](operations/setup_environments.md) 只作为参考矩阵。
- 必须手动复制的忽略/本地资产汇总在
  [operations/setup_local_sync.md](operations/setup_local_sync.md)，但每份操作文档仍应列出本任务
  所需的本地资产。
- 使用被忽略的外部参考前自动同步：`scripts/sync_external_repos.sh`。
- 规划器（Planner）测试：[operations/build_and_test.md](operations/build_and_test.md)。
- 动捕（Mocap）启动：[operations/run_mocap.md](operations/run_mocap.md)。
- 规划器（Planner）运行：[operations/run_planner.md](operations/run_planner.md)。
- Isaac 训练：[operations/run_training.md](operations/run_training.md)。
- 共享 RunPod GPU 训练（团队 pod、每人独立目录、冒烟测试套件 smoke suite）：
  [operations/run_on_runpod.md](operations/run_on_runpod.md)。
- 训练用的视频转动作（video-to-motion）参考：先通过
  [operations/setup_local_sync.md](operations/setup_local_sync.md) 恢复被忽略的 GVHMR/GMR 资产，再按
  [reimplement.md](../reimplement.md) 的 Step 9-12 生成本地
  `hope_training/motions/preprocessed/*.npz`；只有需要共享登记库（registry）制品时才上传 WandB。
- A3 部署空跑（dry-run）：[operations/run_deploy_dryrun.md](operations/run_deploy_dryrun.md)。
- 本地资产与同步：[operations/setup_local_sync.md](operations/setup_local_sync.md)。

## 当前已知环境限制

- 项目运行于 Linux 和 ROS 2 Jazzy。`hope_ws` 的 `colcon` 构建尚未在本测试 shell（harness shell）中
  独立验证，
  因此必须在 ROS 环境内运行并验证（见 [operations/build_and_test.md](operations/build_and_test.md)）。
- 全新的 `git clone` 有意设计为**非自包含**：检出（checkout）后不存在 `external_repos/` 和
  `vendor_assets/`，`hope_training/GMR` / `hope_training/GVHMR` 仓库副本（clone）、参考动作和
  二进制模型制品也都被 Git 忽略。
  按需重建：TTRL 使用 `scripts/sync_external_repos.sh`，其余内容按
  [operations/setup_local_sync.md](operations/setup_local_sync.md) 的手动恢复步骤处理。
- `vendor_assets/` 下的 Agibot 运行时资产仅存在于本地且被 Git 忽略；完整 A3 部署载荷是供应商私下
  交付的约 1.7 GB 文件。
- TTRL 是 `external_repos/` 下被忽略、自动同步的参考仓库，通过
  `scripts/sync_external_repos.sh` 更新；除非未来某个 Gate 将它提升为正式依赖，否则有意不固定版本。
