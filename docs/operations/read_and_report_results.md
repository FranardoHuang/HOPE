# 标准工序：读结果与报数

## 一、报数格式（违反这条会让一整侧的瘫痪隐形）

**只报总数 = 漏病。** 任何臂的读数一律按格报：

> **正手/反手 × 击球率/上台率 × 训练内/考卷 × 单球/连续** —— 十六格齐。

- 单元格没有数字就写"未测/不适用"，**不许合并平均后只报一个数**。
- 两次真实代价：
  - 2026-07-26：汇报的 **45% 回球率**是"反手 0.78–0.87 + 正手 **0.0000**"平均出来的，**该读数作废**。
  - 更早：全局击球率 **0.998** 盖住了反手全零。
- 逐侧比率在样本不足时是 `None` **不是 `0.0`**——"从来没资格"和"有时候失败"必须能分开
  （`utils/my_on_policy_runner.py:49-51`）。
- 多 seed 同理：**平均数掩盖最差初始化**。四个初始化 `83、100、100、20` 报四个数，不报 75.75。

## 二、哪个是主曲线

| 角色 | 指标 | 规矩 |
| --- | --- | --- |
| **北极星** | **MuJoCo 考卷回球率** | 正式入账只认 MuJoCo 版；**连分母一起记**（可解率/锥内率） |
| 训练时主看 | `virtual_return_rate`（上台率，分正反手） | 按**机会数**算：到点击球算一次机会，打到+过网+落进对面台内才计 1。打不到就是 0 |
| 辅助 | `virtual_hit_rate`（击球率，分正反手） | 到点的机会里真碰到球的比例 |
| **只作诊断** | 跟踪三合格 composite | **不用它判死臂**（拍面 25° 误差照样 79% 上台） |

- **报人一律用 rally 全分母口径** `virtual_return_rate_rally_*`（分母 = 挥拍起手数，摔倒计败）。
  `virtual_return_rate` 是**幸存者分母**（只算活到击球帧的挥拍），两者可差近一倍
  （实测 0.8415 vs 0.5255），幸存者口径只作诊断辅尺。
- Isaac 训练内的虚拟球版可以并列报，但**必须标注"训练内虚拟球"**；裁决以 MuJoCo 为准。
- 选存档 = **训练内上台率**的峰值附近，入账前 MuJoCo 复核。

## 三、门槛数字

- **研究推进最低线**：有效同卷中每个动作解析回球率 ≥ 50%。只用来判断管线值不值得继续扩题，
  **不是部署质量线**。
- **正式候选质量目标**：跨独立初始化稳定达到每动作约 80%，同时拍面误差 **p90 < 15°**（不用中位数）、
  落点误差 < 0.3 m、零摔，且判分尺可信。
- 没有分母的百分比不作数。分母报表（kept/asked/锥内比例/难度中位）判卷时自动打印，
  **入账连它一起抄**；`qdes_clamp=ON/OFF` 状态同理。

### 三点一、入击球窗拍距决策

[入击球窗拍距](../DEFINITIONS.md#strike-window-entry-distance)只在每拍第一个 strike-window tick
记一次。报数必须一起抄：

- `strike_window_entry_racket_target_distance_count`；
- 以 `0.075/0.15/0.20/0.30/0.50/0.70/1.00 m` 为边界的八个互斥 finite bin；
- `strike_window_entry_racket_target_distance_nonfinite_count`与
  `strike_window_entry_racket_target_distance_m_sum`。

守恒必须满足：`count = 八个 finite bin 之和 + nonfinite_count`，而 finite sum 不包含
nonfinite。分母不守恒或出现 nonfinite 先停在证据链，不判学习。若有效 entry 的多数
`>0.20 m`，说明策略进窗时仍在细 exp 核死区：下一动作是粗+细核，不继续烧
vendor long。若多数 `<=0.20 m`，才可把主卡点继续归因于 termination/控制或更细的
击球误差。

## 四、证据等级

`E0` 设计 · `E1` 源码/单测/静态 · `E2` 运行时冒烟或模型加载 · `E3` 受控训练 ·
`E4` 留出仿真器/Gate3 考卷 · `E5` 真机。
**只记录实际达到的最高等级；不得用大量低等级测试推断高等级证据。**

证据的通货是 `run_name` + PID/PGID + `run_binding.json` + 内容寻址收据 SHA + checkpoint SHA，
**不是** WandB run id。判活/判死**只看日志签名**——Isaac 异常后仍 `exit=0`，退出码不可信。

## 五、写法

- **先人话后代号**：`M3`、`R9c`、`SZ` 这类裸代号后面必须跟一句人话；
  首次出现的 `run_name`/flag 必须在 [`DEFINITIONS.md`](../DEFINITIONS.md) 有定义。
- 摘要**抓异常不抓预期**：WARN 行全部进摘要。
- 一项事实只在一处详细记录，其他地方一句摘要 + 链接。

## 六、一条命令出报告

```
bash hope_training/whole_body_tracking/scripts/judge.sh <run_dir> [checkpoint.pt]
```

自动解析该臂的动作对/相位/题库 → 原生导出 + sidecar → 双侧 × 双噪声档 → md 报告落 `run_dir/judge/`。
**解析不到会 fail-loud 要求手传，绝不静默用默认值。** `--dry-run` 只打印命令链。
手动排障步骤与已知坑（sidecar 缺失、`--qdes-clamp`、`--hold-ref`、`torso_z +0.11` 常量）
见 [runbook 判卷链](../runbook.md#判卷链北极星数字怎么产2026-07-06-全链踩通)。
