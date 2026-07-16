# EXP-MOTION-READY-TO-STRIKE-0P5：从静止准备态压缩到触球

- 状态：`ready`（host 候选生成器完成；真实动作候选和后续门禁尚未运行）
- 阶段/轴：动作空间路径 × 时间律 × 模仿约束
- 集成小目标：让动作从所选 clip 的第 0 帧静止准备态出发，在 0.5 秒内进入保真的触球窗
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E1 source gate
- 创建日期/最后复核日期：2026-07-17

这里的 [schema-2 motion](../../DEFINITIONS.md) 是机器人动作资产合同；
[`TOPP`](../../DEFINITIONS.md) 是在固定空间路径上找受速度、加速度、力矩和接触稳定约束的时间律。
“0.5 秒候选”只表示路径的触球行被放在第 25 个 50 Hz tick，不表示该路径已经动力学可行或能回球。

## 问题与假设

现役 `v4rg`（第四版正反手参考挥拍）只重定时固定长路径时，production-FK TOPP 找到的最佳可行
run-up 上界是正手 `0.98 s`、反手 `0.78 s`。因此单纯把完整旧 clip 加速不能闭合 0.5 秒。

新假设是：从动作自己的第 0 帧姿态、全零速度出发，用一条 quintic 位置桥接跳过与击球无关的长引拍，
并逐字节保留触球前 `0.1 s` 和触球行，可能得到更短且仍可由 TOPP/动力学认证的空间路径。

证伪条件：预注册的桥接族没有任一候选同时达到 production FK、TOPP run-up `<=0.5 s`、L0、vendor L1
自碰/自打、桌网整轨余隙 `>=5 mm` 和动力学稳定；或实际 K100/厂商 MuJoCo 表明缩短路径不能回球。

## 固定的构造合同

- 等待态只取显式 `ready-source` 的 **第 0 帧姿态**；输入文件里的速度不继承，输出从 bitwise zero
  速度开始，并至少保留三行 producer-gradient 为零的 50 Hz 样本。
- `joint_pos` 和 link-position 使用解析 quintic endpoint 条件构造；这是 host 候选，不是运行时 C2
  或动力学证书。join 处离散速度误差必须进 contract，不能藏掉。
- 触球前 `0.1 s` 到触球行的六个 schema-2 时序通道逐字节等于原动作。
- 四元数导数工作区围绕原始 join 行做 hemisphere alignment；发布的 join 行必须保持原始 `q/-q`
  字节符号，避免等价旋转在分量空间跳变。
- 输出 NPZ 与 JSON 使用 no-clobber；下游必须两份同时存在并核 SHA。双文件发布不是 crash-atomic，
  若进程被 `SIGKILL` 留下孤儿文件，保全现场并人工判 invalid，禁止自动删除后重放。
- 输出 contract 固定 `training/deployment/hardware_authorized=false`。host 生成成功不得越过后续门。
- 输入允许两种精确 schema-2 形态：没有迁移溯源的原生核心字段，或完整携带
  `kinematics_migration_source_sha256/source_point/tool` 三元组。三项只允许 canonical migration v2
  的 scalar unicode 形态并必须全有；未知字段、残缺三元组、坏 SHA/point/tool 均拒绝。输出逐位继承
  击球 source 的三元组，不把 ready-source 的三元组冒充输出血缘；JSON 分别绑定两份当前输入 SHA。
  历史 legacy ancestor bytes 尚未重读复算，因此这只证明 canonical syntax 与逐位保留，不证明祖先
  制品仍可取或其 SHA 已重新验证。

## 完整而不混因果的四格

| 格 | 空间路径 | 击球窗模仿 | 回答的问题 |
| --- | --- | --- | --- |
| A | 原完整路径 | 现役模仿 | 基线 |
| B | 新 ready-to-strike 路径 | 现役模仿 | 只缩短路径是否已足够 |
| C | 原完整路径 | 放松击球窗模仿 | 只释放 policy 控制预算是否足够 |
| D | 新 ready-to-strike 路径 | 放松击球窗模仿 | 两者是否必须联合 |

先离线筛路径；只有 B 的动作证书全过才允许 B/D 进入单 seed 训练。C 使用与现役相同安全门。
若 B、C 都无单独改善而 D 改善，结论是交互项，不能把收益单记到 TOPP 或 Reward。

## 决策与当前边界

- 决定：`inconclusive`；采用 host-only candidate builder，不采用任何生成动作。
- 是否已纳入当前 setting：`no`。
- 真实 attempt 1：Pod2 CPU-only、namespace
  `/workspace/codexschema/ready_to_strike_0p5_20260717/attempt_2137b82b`。正/反手都在写候选前
  fail closed，原因是 v1 生成器把正式 v4rg 的完整 canonical migration 三元组误判为 unexpected；
  没有候选 NPZ/contract、没有启动 TOPP、没有占 GPU，也不是动作/动力学失败。该 namespace 永久保留，
  不重放。合同修复后的专项为 `21 passed`；下一次只能使用新源码和新 namespace。
- 真实 attempt 2：新源码 `66f93559`、namespace `attempt_2_66f93559`。两侧 host 候选均生成成功并把
  触球放在第25 tick；正/反手 candidate SHA 分别为 `b0350f4d…53ac4d` / `0f017dc5…dfc89`。
  production-FK TOPP 两侧 hard acceptance 都通过，但可行 run-up 上界分别为 `0.64 s` / `0.94 s`，
  因而 **没有** 0.5秒动作证书，也不送后续安全门。反手使用共同的正手 frame0 ready，结果还不能区分
  “join 位置不佳”和“共同 ready 距离过远”。
- 后续搜索已在
  [`ready_to_strike_join_ladder_20260717.yaml`](../../../configs/ready_to_strike_join_ladder_20260717.yaml)
  预注册：保持 frame0 zero、hold=4、触球第25 tick和保护窗不变。先做 `delta=contact-join` 的合法
  两端 `6/17` × 正反手 × forehand/backhand frame0 ready 的小因子阵（两个既有格不重跑），再按冻结
  规则跑中点 `12`，只在主点附近细化 `9/14`。shared-ready 按两侧 minimax 选，单侧胜者只作诊断。
  若全失败，停止这个 two-ready/hold/join family，再单独改变一个结构轴，绝不放宽 TOPP 动力学限值。
- 下一门：在 ignored runtime 资产上按预注册 join ladder 生成正/反手候选，production FK 重建后逐个跑
  TOPP；只把 `<=0.5 s` 的候选送 L0/L1/桌网/动力学，再用绑定该 motion SHA 的 0.5 秒 K100 和
vendor MuJoCo 判行为。

复现入口见[操作文档](../../operations/run_ready_to_strike_motion.md)。
