# HOPE 起始页（读这一页，然后只读你那一行）

在 Agibot A3 上做与 HITTER 接口兼容的人形乒乓系统。策略侧 2026-07-02 走通过一次
Isaac → MuJoCo → 真机（仅正手）；感知闭环（mocap → planner → deploy）未接通，无质量基线。

## 三条只看这里的事

| 问题 | 唯一权威 |
| --- | --- |
| 现在在训什么、优先级、谁认领 | [`NOW.md`](NOW.md)（**只认 `origin/main` 上的**；分支副本只是提案） |
| 该跑哪个命令 / 有没有现成工具 | [`INDEX.md`](INDEX.md) → [工具目录](operations/tool_catalogue.md) |
| 这个缩写是什么意思 | [`DEFINITIONS.md`](DEFINITIONS.md) |

## 动手前必须知道的五条

1. **先查工具再造工具。**[工具目录](operations/tool_catalogue.md)一页列全部脚本。
   2026-07-26 的三个缺陷全部是"工具早就有、没人知道"。
2. **分侧报数。** 正手/反手绝不能只报平均——一次 45% 是"反手 0.85 + 正手 0.0000"平均出来的。
   见[结果判读](operations/read_and_report_results.md)。
3. **桌面以下没有球。** 目标框 `z_lo ≥ 0.76 + 0.02 = 0.78 m`，
   见[目标物理有效性](interfaces/racket_target_physical_validity.md)。
4. **摘要抓异常不抓预期。** `grep -nE 'WARN|Error|Traceback'`，WARN 必进摘要。
5. **每个 `run_name`/flag 首次出现必须带一句人话**，并在 `DEFINITIONS.md` 有定义。

## Gate 状态

G00–G07 全部 `Partial`，G08 `Research track`。逐条现状看 [`gates/`](gates/)——
各 Gate 文件头部的 `Status:` 行是权威，本表只说明"没有一个 Gate 已 `Done`"。

## 规矩在哪

- 贡献者与 Agent 的完整规则（文档更新义务、责任归属、Gate 纪律、资产纪律、安全）：
  [`../AGENTS.md`](../AGENTS.md)。
- 目录职责：[`PROJECT_MAP.md`](PROJECT_MAP.md)；Git/LFS/忽略策略：[`ASSET_POLICY.md`](ASSET_POLICY.md)。
- `reimplement.md` 是长篇命令附录，**不是入口**；只在 Gate 或操作文档点名某个 `Step N` 时才打开。
- `docs/archive/` 与 `docs/experiments/archive/` 是历史，**正常入门不读**。

## 环境的三个坑

- 全新 `git clone` **有意非自包含**：`external_repos/`、`vendor_assets/`、GMR/GVHMR 副本、
  参考动作和二进制模型都被忽略。恢复看
  [`operations/setup_local_sync.md`](operations/setup_local_sync.md)；TTRL 用 `scripts/sync_external_repos.sh`。
- 项目跑在 Linux + ROS 2 Jazzy；`hope_ws` 的 `colcon` 必须在 ROS 环境内验证。
- host 是 py3.8、pod 是 py3.10+；新代码别裸用 `zip(strict=)`／`math.ulp`。
