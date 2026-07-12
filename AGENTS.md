# 本仓库的 Agent 规则

改动前先读 `docs/START_HERE.md`，再用 `docs/INDEX.md` 作为一站式任务路由。

## 文档是工作的一部分

只要 Agent 改了代码、移动了文件、增加了资产、改变了目标、发现了阻塞，或验证了某个 Gate，
就必须在同一分支同步更新文档。`docs/NOW.md` 的运行态权威规则见下文；功能分支（feature branch）中的
`NOW` 更新只能作为待合入提案。

必须完成的更新：

- 更新受影响的 `docs/gates/G*.md`。
- 在 `docs/PROGRESS.md` 添加简短的带日期条目。
- 当假设、实验运行（run）、结果、失效结论或采用/拒绝决定变化时，更新或新建对应的
  `docs/experiments/` 记录。
- 只有在已采用的训练配置（setting）、阶段/小目标状态、当前成绩表或当前人类责任人变化时，
  才更新 `docs/NOW.md`。
- 只有重要逻辑能力、根因修复、结论翻转或接口约定（contract）已经合入 `main` 时，
  才更新 `docs/TIMELINE.md`；不要抄录每个提交（commit）或实验运行（run）事件。
- 文件夹职责变化时，更新 `docs/PROJECT_MAP.md`。
- Git、忽略资产、Git LFS 或外部仓库策略变化时，更新 `docs/ASSET_POLICY.md`。
- 坐标系（frame）、ROS 主题（topic）、消息（message）、关节顺序、观测、动作或运行时接口约定变化时，
  更新 `docs/interfaces/` 下的相应文件。
- 配置、构建、测试、训练或部署命令变化时，更新 `docs/operations/` 下的相应文件。

不要把聊天记录当作项目状态的唯一记录位置。

## 可读性与术语

- 每个 `run_name`、命令行标志（flag）或缩写在一份文档/记录中首次出现时，必须同时给出人话解释，
  并链接到 [`docs/DEFINITIONS.md`](docs/DEFINITIONS.md) 中的定义；如果定义尚不存在，先补定义。
- 不要只写内部代号。实验表可以保留机器可读字段，但旁边必须有人能直接理解的名称与目的。
- 每个事实只保留一个详细真源；其他文档只写简短摘要和链接。

## `NOW` 主板与统一工作队列

- `origin/main` 上的 `docs/NOW.md` 是唯一运行态权威。
- 功能分支（feature branch）中的 `docs/NOW.md` 只能提出候选改动，不能改变当前优先级、认领状态或已采用配置；
  合入前必须基于最新 `origin/main` 逐项对账并解决冲突，只有进入 `main` 后才生效。
- `docs/NOW.md` 中的统一工作队列是全项目唯一的优先级账本。实验记录、Gate 文档、个人笔记和
  Agent 任务不得维护与之竞争的影子优先级队列；它们只记录局部步骤、证据或链接。

## 人的责任归属与 Agent 参与来源

- `Owner`、`DRI` 和 `Responsible` 字段只能写人名。
- Claude/Codex 只记录在 `Executor` 或 `Assisted by` 字段，不能作为责任人。
- 如果不知道由谁负责，写 `UNASSIGNED`；不要根据 Git 作者（author）猜责任人，也不要用 Agent 替代人名。

## Gate 纪律

除非 Gate 文档已经包含可复现命令、验证结果、已知限制和当前输入/输出，否则不得把 Gate 标为
`Done`。

已有材料或代码、但验证尚未完成时，使用 `Partial`。

## 资产纪律

源代码和小型配置保存在 Git 中。除非团队明确采用 Git LFS 或其他制品系统，否则大型运行时制品
应放在被忽略的本地根目录（例如 `vendor_assets/`）下。

除非本地专有资产已经迁移、备份或被明确宣布废弃，否则不得删除。

如果任务依赖被忽略的文件或文件夹，必须在 `docs/operations/setup_local_sync.md` 中补充手动恢复路径，
并在相关 Gate 文档中写明该依赖。不得假设另一台机器上已经存在这些被忽略的文件。

被忽略的外部参考仓库不是固定版本依赖。任务需要参考 TTRL 时，先运行
`scripts/sync_external_repos.sh`；如果从中提取了任何想法、代码、配置或结果，再把所用 TTRL 源码
提交（commit）记录到相关 Gate 文档中。

## 安全

只有相关 Gate 文档和操作（operation）文档已经明确说明空跑（dry-run）、关节顺序、命令缩放和
安全停机检查均已
通过，才允许执行真实机器人命令测试。
