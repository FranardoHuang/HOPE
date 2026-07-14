# EXP-MOTION-BACKHAND-LOOP-B-L0 — 反手拉 B 的静态动作证书

- 状态：V1 portable dry-run 因非自洽的 float32 byte-equality 门 fail closed；V2 的 Pod2
  dry-run 与一次 no-clobber formal audit 均通过，L0 证书已生成，只解锁 vendor L1
- 阶段/轴：新动作库 / runtime-order schema-2 后的纯 CPU 静态审计
- 人类负责人：Franco
- 执行者：Codex
- 最高证据等级：E2（exact Pod2 V2 runtime certificate；仍无 vendor L1/动力学/击球结果）

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

portable 修复后，Pod2 用同一份 Pod1 产出的 exact NPZ 执行了 V1 full `dry-run`，并在证书写入前得到
第二个、不同层次的 fail-closed 结果：link position `537` 个 component 不逐字节相等，最大差
`1.1920929e-7 m`；quaternion `917` 个，最大差 `5.9604645e-8`；COM linear velocity `1261` 个，
最大差 `2.9802322e-6 m/s`；angular velocity `2320` 个，最大差 `5.9679151e-6 rad/s`。certificate
仍不存在。该次只有操作人保存的失败摘要，没有独立 result artifact，因此这里按负结果登记，不能冒充
可重新签名的正式证书。

## V1 byte-equality 为什么不自洽

源码链已经足以证明根因，不需要把差值反向拟合成阈值：producer 的 `resample_payload()` 先得到原始
float32 `root_pos/root_rot`，`fk_series_with_com()` 把它们写入 MuJoCo free-joint `qpos`，
`mj_forward` 后再把归一化的 `xpos/xquat/xipos` 投影为 float32。schema-2 只保存这份**后 FK 的 root
body pose**，不保存原始 free-joint qpos。V1 audit 再把保存的 root body pose 当 qpos 注回 MuJoCo，做
第二次 quaternion normalization/FK/float32 投影，然后要求所有结果 byte equal。这个 lossy round trip
一般不幂等；Pod1 生产、Pod2 审计只是让问题暴露，并不是根因成立所必需。

量级也与该机制一致：position 最大差正好是一份 unit-scale float32 相邻格宽；quaternion 是半格；
50 Hz central difference 的系数 `1/(2*0.02)=25` 把 `1.1920929e-7` 放大为
`2.9802322e-6`。因此不能把 V1 负结果解释成动作、关节范围、接地或支撑脚失败。

## V2 预注册数值合同

V1 prereg/validator 原字节和失败账全部冻结。新 V2 计划
[`motion_backhand_loop_b_l0_static_prereg_20260715_v2.json`](../../../configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json)
（SHA-256 `185612a99d5dd1e0aba0d04d50467103ea9b3967b917c58371bd409d10fc6ccb`）绑定 V1 完整
input/runtime/MJCF/lineage/safety 合同；V2 validator SHA-256 为
`d025586b1d505432978ea772462a6d90ad3e83a566d8818586d9339ceccfab25`，只替换不可重构的数值比较：

1. link position 与 quaternion 用 component-wise 两个 float32 相邻可表示数格宽（unit in the last
   place，简称 [`ULP`](../../DEFINITIONS.md#float32-ulp)）的绝对包络；近零 component 以
   `spacing(float32(1))` 为 floor，避免全局 FK cancellation 把合法舍入误差压成 subnormal 阈值。
   position 与 quaternion 的物理硬上限都为 `5e-7`；quaternion 还必须保持同 hemisphere。
2. COM position 不从第二次 FK 的 `xipos` 猜 producer，而由已存 link pose 与 exact MJCF
   `body_ipos` 重构。linear velocity 容差按 float32 pose 投影、四元数到旋转矩阵的解析上界
   `8q+4q²` 和 frozen 50 Hz finite-difference endpoint 最坏系数推导；另有 `2e-4 m/s` 物理硬上限。
3. angular velocity producer 本来就是直接对**已存** `body_quat_w` 做 `so3_derivative`，所以 V2
   继续要求 byte equality，不给容差。`joint_vel=gradient(joint_pos)` 也继续 byte exact。
4. exact lineage/SHA、shape/order/dtype/finite、`1e-5 rad` joint range、原地面区间、左右
   ankle-roll support ancestry、no-clobber 和所有 downstream false authorization 完全继承 V1；没有
   放宽关节、ground/support 或 safety 门。

更严格的长期方案是 schema-3 motion 另存 producer 原始 free-joint qpos，再决定 exact-runtime 内的
bit replay；不能从现有 schema-2 body pose 唯一恢复那些 pre-normalization bytes。V2 只解决冻结 B
资产的 L0 数值可复现性，不改变 schema，也不宣称跨硬件 bit determinism。

## V2 Pod2 runtime 结果

`main@cc1a2b101431f42ad2e1ddd94816605781404f51` 以 clean detached checkout 在 Pod2 的 exact
Python `3.12.3` / NumPy `2.5.0` / MuJoCo `3.10.0` CPU venv 上先运行一次 full `dry-run`，
输出 `runtime_audit=true certificate_written=false l0_static_complete=false`。独立只读复核随后
确认 source detached/clean、plan/validator、四份输入 SHA 和 V1/V2 certificate absence 均 exact。

操作人再显式授权只创建 exact V2 父目录并执行同计划唯一一次 `audit`；它以 `O_EXCL`
写入：

`/workspace/codexschema/motion_video_intake_20260711/l0_static_primary_v2/franco_backhand_loop_b_98e7b883b29d.l0_static_certificate.json`

certificate SHA-256 为
`60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6`。证书 `schema_version=2`、
`asset_id=franco_backhand_loop_b`，`l0_static_complete=true`、`vendor_l1_authorized=true`；同时
`table_net/dynamics/simulator/training/formal_motion/hardware_authorized` 仍全为 `false`。这证明 151 帧
离散静态运动学、关节范围、grounding 与 support ancestry 过门；不证明自碰、球拍打自身、
桌网扫掠、动力学平衡或能打球。

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
4. V1 用存储的 pelvis pose + runtime-order `joint_pos` 做 151 次 `mj_forward`，从不调用 `mj_step`；
   其 pose/velocity byte-equality 已由上述真实负结果证明不自洽。V2 保留同样 FK、joint/ground/support
   计算，但改用上节预注册的 field-specific 数值合同；angular 与 joint velocity 仍逐字节 exact。
5. 用原 grounding 的 exact collision-surface 算法复算 151 个离散帧。全局最小余隙必须落在
   `1e-5 ± 5e-7 m` 到 `1e-3 + 5e-7 m` 的冻结区间内，且每帧最低 collision body 必须属于左右
   ankle-roll 支撑脚子树。这只证明离散帧地面一致性，不证明连续时间余隙。

`dry-run` 会执行同一完整上游谱系、NPZ 与运动学审计但不发布证书，并明确输出
`certificate_written=false`、`l0_static_complete=false`；正式 `audit` 才会在所有检查通过后以 `O_EXCL`
写入预注册绝对路径。目标已存在、父目录是 symlink、JSON duplicate key、任一输入漂移都会 fail
closed。V1 已在 Pod2 运行并 fail closed；V2 已按上述 exact runtime 合同通过并发布 L0
certificate。

## 明确不在本门内

本门没有执行或声称：vendor 自碰、球拍打到机器人、桌/网整轨 `>=5 mm`、连续时间地面余隙、
动力学/平衡、TOPP、击球/上台率、RL、Gate3 或真机安全。已通过的 L0 也只解锁独立 vendor L1
审计，不解锁训练。

## 源码验证

```bash
python3 -m pytest -q \
  tests/test_motion_backhand_loop_b_l0_static.py \
  tests/test_motion_backhand_loop_b_l0_static_v2.py
```

V2 分支两份 L0 dependency-light 专项合跑为 `29 passed`。攻击面覆盖 same-byte activation 跨 clean
checkout、历史 activation 不可被当前 consume runner 接管、portable loader 不改 activation 字节、错误
历史 commit、当前 checkout/body-order 漂移与 symlink、claim source commit 漂移、duplicate JSON key、
`NaN`、symlink 输入/输出父目录、NPZ
unexpected/duplicate member、NaN/Inf、非单位四元数、伪造 velocity、关节范围越界、地面余隙上下界与
certificate no-clobber，以及 V2 two-bin/物理上限、quaternion hemisphere、50 Hz derivative bound、
angular exact、硬 safety 继承和 full `dry-run` 不写 certificate。全仓 `python3 -m pytest -q` 尝试在
collection 阶段因本机缺 `zmq`、`torch`、`hydra` 出现 `15 errors`，没有进入测试执行；这不是 V2
断言失败，也不能记成全仓通过。dependency-light `tests/` 另跑为 `1116 passed, 10 skipped, 94 failed`；
94 项是 latest-main 已存在的 frozen `training_contract` SHA 与 manifest 漂移（以及同类旧 causal fixture），
没有一项来自两份 B L0 测试，故不在本分支改写这些无关合同，也不记成整合绿灯。
`origin/main@b609c0d` 的 `1018 passed, 10 skipped` 只保留为 V1
portability 修复前历史结果。这些结果
只证明源码和合成反例；真实 L0 结果只由上节 Pod2 certificate 与其 SHA 提供，不用本地
测试冒充 runtime 证据。
