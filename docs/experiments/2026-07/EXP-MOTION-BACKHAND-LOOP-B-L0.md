# EXP-MOTION-BACKHAND-LOOP-B-L0 — 反手拉 B 的静态动作证书

- 状态：portable source gate pass，Pod2 full dry-run not run（首次 runtime 调用在上游谱系校验阶段 fail closed；未生成证书）
- 阶段/轴：新动作库 / runtime-order schema-2 后的纯 CPU 静态审计
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E1（源码、合同与合成反例；尚无本门 runtime 结果）

本文的 [`L0`](../../DEFINITIONS.md#motion-l0-static) 指“完全不推进物理仿真的静态动作可行性审计”，
不是训练层级；source/static pass 也不等于真实资产已有 runtime certificate。

## 要回答的问题

反手拉候选 B 已有一次性 runner 认可的 exact runtime-order schema-2/FK 输出。本门只问：这一个
NPZ 是否在 exact A3 模型上满足最基础的结构与离散逐帧运动学条件，因而有资格进入下一张
vendor L1 自碰/自打证书？它不回答动作能不能打球，也不回答动态平衡。

## 冻结输入

预注册
[`motion_backhand_loop_b_l0_static_prereg_20260714.json`](../../../configs/motion_backhand_loop_b_l0_static_prereg_20260714.json)
（SHA-256 `7e155c894dae7b37771487aeb4051bb72cf146088c78bffa7dac437576c97bc0`）只接受以下四份已由
Pod1 产生、但可按内容身份在 Pod2 复核的运行制品：

| 制品 | SHA-256 |
| --- | --- |
| 151-frame schema-2 NPZ | `e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc` |
| schema-2/FK report | `4f5245937956290b3f623acbb588d99b346e5a1d874e55ee9caf010f2d75bc38` |
| irreversible consume claim | `76e7ff88fea39c13b45096edaad504b2570b3ce079acc96366b820a9c1295fb0` |
| completion-last success ledger | `c0a25f2cba0e61bf0df7f63e6493948e16c5a3d3074f65091430f29e417f4f8b` |

validator 会先调用原一次性 runner 的 `validate_formal_result`，重新验证
claim → activation/receipt/runtime → child → NPZ/report → success 的完整谱系；随后本门再次独立做
duplicate-key、regular-file、symlink 与 SHA 检查。直接 materializer 输出、换字节、缺 claim 或缺 success
都不能进入 L0。

首次 runtime 调用在 `validate_upstream_result` 内、任何运动学检查和 certificate 写入前 fail closed：历史
runner 把 consume checkout 的绝对 activation 路径重复写进 claim/preflight/success，并误要求新的 clean
L0 checkout 使用同一绝对路径。claim、activation、receipt、NPZ/report 与 success 内容没有因此失效；
失败只创建了预注册 certificate 父目录，certificate 仍不存在，也没有重跑 audit。修正版保留历史
runner/activation/claim 原字节，用 claim 绑定的 activation exact bytes/SHA 与 source tuple 保存历史来源，
再用当前 clean detached checkout 的 commit、runner、source validator 与 `runtime_body_order` 内容绑定建立
portable context，复用原 runner 的完整 result/NPZ 校验。旧 Pod1 checkout 路径不再是读取 body order 的
必要条件，也没有从历史绝对路径 fallback；旧 attempt ID、receipt、runner、claim、success 与 NPZ/report
绑定全部保持严格。原生 consume loader 仍要求 activation 绑定的旧 runner 就在当前 checkout，因而当前
runner 不会借 portability 修复接管未消费的 C。

模型侧绑定 exact vendor MJCF `2ab1cd31...feb97`、`1 XML + 74 mesh` closure
`e0381752...962de`、compiled collision contract `18e7f6ff...386e5`、31-joint runtime order、32-body
runtime order和 donor metadata。运行环境沿用已记录的 CPU venv：Python `3.12.3`、NumPy `2.5.0`、
MuJoCo `3.10.0`，并强制 `CUDA_VISIBLE_DEVICES=''`。

## 最小通过条件

[`audit_motion_schema2_l0_static.py`](../../../scripts/audit_motion_schema2_l0_static.py)
（SHA-256 `5970f82bf5cde11ff448e056469f2a7276f2a743f97ed8ed9a287a70dc91b411`）只做：

1. 11 个 NPZ 字段、151 帧、50 Hz、31 关节、32 body、float32 time series、finite、schema-2 与
   link-origin/COM-velocity 点语义必须 exact；四元数单位范数沿用 producer 已冻结的 `1e-5` 容差。
2. `joint_vel` 必须逐字节等于 `gradient(joint_pos, 1/50)`；runtime joint/body order 必须分别与
   donor metadata 和 vendor MJCF 构成完整双射。
3. 关节位置逐帧对 exact compiled MJCF range，唯一容差沿用已执行 grounding 合同的 `1e-5 rad`，
   不新增 soft-limit、速度、加速度或 reward 阈值。
4. 用存储的 pelvis pose + runtime-order `joint_pos` 做 151 次 `mj_forward`，从不调用 `mj_step`；重新
   计算的 link pose、COM linear velocity 与 body angular velocity 转成 float32 后必须与 NPZ 逐字节相等。
5. 用原 grounding 的 exact collision-surface 算法复算 151 个离散帧。全局最小余隙必须落在
   `1e-5 ± 5e-7 m` 到 `1e-3 + 5e-7 m` 的冻结区间内，且每帧最低 collision body 必须属于左右
   ankle-roll 支撑脚子树。这只证明离散帧地面一致性，不证明连续时间余隙。

`dry-run` 会执行同一完整上游谱系、NPZ 与运动学审计但不发布证书，并明确输出
`certificate_written=false`、`l0_static_complete=false`；正式 `audit` 才会在所有检查通过后以 `O_EXCL`
写入预注册绝对路径。目标已存在、父目录是 symlink、JSON duplicate key、任一输入漂移都会 fail
closed。当前尚未在 Pod2 运行私有 NPZ，因此没有 L0 pass certificate。

## 明确不在本门内

本门没有执行或声称：vendor 自碰、球拍打到机器人、桌/网整轨 `>=5 mm`、连续时间地面余隙、
动力学/平衡、TOPP、击球/上台率、RL、Gate3 或真机安全。即便未来 L0 通过，也只解锁独立 vendor L1
审计，不解锁训练。

## 源码验证

```bash
python3 -m pytest -q tests/test_motion_backhand_loop_b_l0_static.py
```

本分支两个 dependency-light 专项合跑为 `51 passed`。攻击面覆盖 same-byte activation 跨 clean
checkout、历史 activation 不可被当前 consume runner 接管、portable loader 不改 activation 字节、错误
历史 commit、当前 checkout/body-order 漂移与 symlink、claim source commit 漂移、duplicate JSON key、
`NaN`、symlink 输入/输出父目录、NPZ
unexpected/duplicate member、NaN/Inf、非单位四元数、伪造 velocity、关节范围越界、地面余隙上下界与
certificate no-clobber，以及 full `dry-run` 不写 certificate。本修复按范围没有跑全仓
回归。`origin/main@b609c0d` 的 `1018 passed, 10 skipped` 只保留为修复前历史结果。这些结果只证明源码和合成反例，
不将缺少私有 runtime 执行的 source gate 冒充真实 L0 结果。
