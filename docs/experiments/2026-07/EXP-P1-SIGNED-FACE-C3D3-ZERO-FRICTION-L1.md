# EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1 — 显式零摩擦 L1 配对

- 状态：`L1 provenance complete / behavior pending`
- 阶段/轴：阶段 1 / 有符号拍面引导
- 集成小目标：得到一对配方、命令、实际 PhysX plant 与 checkpoint 谱系一致的 fresh L1 证据
- 人类负责人：Franco
- 执行者：Codex
- 复核/决策负责人：Franco
- 最高证据等级：`E1`
- 创建日期/最后复核日期：2026-07-14 / 2026-07-14

共享的 [`L1`](../../DEFINITIONS.md) 指 `512 env × 25 update` 的小规模发射与合同冒烟，不是
行为判卷。本文的 `C3` 是关闭有符号拍面引导的 fresh 显式零摩擦对照；`D3` 是只把该引导权重改成
`-0.4` 的匹配 fresh 实验。`3` 表示旧 C2/D2 合同失配后启用的全新 namespace，不是第三个 seed。
`run_name` 是不可复用的训练运行名；`claim` 是首次发射时创建的外层原子运行凭据。其余术语见
[术语与人话对照](../../DEFINITIONS.md)。

## 问题与假设

旧 C2 manifest 声明 `zero_joint_friction=true`，但实际 trainer argv 和外层优化配方均没有
`task.plant.zero_joint_friction=true`。它写出的 hard contract 中 31 个 PhysX 关节摩擦系数全部非零；
因此 C2 不符合自己声明的 setting，未启动的 D2 也不能再续接。C2 checkpoint 只保留为失效证据，
不得被采用或改名重跑。

本实验先回答一个更窄、可证伪的问题：新 pair 能否让以下四层对同一个零摩擦事实完全一致？

1. manifest 的 declarative setting；
2. exact Hydra argv 与 outer optimization recipe/claim；
3. trainer 在实例化 PhysX 后打印的 `ZERO_FRICTION_RUNTIME_OK` marker；
4. 相邻 schema-3 hard contract 的 31 个关节摩擦系数。

任一层缺失、重复、为 false、为非有限值或出现任意非零系数，就证伪这次发射并永久停止该 namespace；
不得自动 retry。即使两条 L1 都通过，也只证明 provenance 闭合，不证明 `-0.4` 引导有效。

## 冻结的 setting

| 字段 | 值/SHA |
| --- | --- |
| 训练 source | commit `4467d79f1ed425a4263f0caaad2f661e1ec737ad`；tree `497db1d8f2d7fb1b554337928f098a2951d4cf0d` |
| Manifest | [`phase1_signed_face_c3d3_l1_prereg_20260714.json`](../../../configs/phase1_signed_face_c3d3_l1_prereg_20260714.json)，SHA-256 `eefc8023786a3bc90f86bc0803dda69003c76cc3076bc089a15dec618f435dc2` |
| Launcher/finalizer | [`run_phase1_signed_face_c3d3_l1.py`](../../../scripts/run_phase1_signed_face_c3d3_l1.py)，SHA-256 `192148906a893e8b55e9a9cf666c3557e14ce8d0b53f2337b2e311e2430aa628` |
| 动作 | 正手 `f2cb2d9f...41687`；反手 `17225533...7534`；顺序固定 |
| Train bank | schema-3 exact bank `3a9d8851...85b71`；physics `09dfe899...fb95`；family `9603a178...9db` |
| 观测/action | `deploy_parity_face179` / 179；31-D action |
| Plant | 两格命令中恰好一个 `task.plant.zero_joint_friction=true`；hard contract 必须 31/31 finite zero |
| Seed/预算 | 两格都 fresh seed3，`512 env × 25 update`，终档 `model_24.pt` |
| Runtime lane | Pod1 GPU1=C3；Pod1 GPU2=D3；每格首次 claim 前本卡须无 compute PID |
| 禁止边界 | activation、judge、L2、第二 seed、stop/promote、自动 retry、部署与真机全部 false |

## 实验差异

- 对照 C3：位置引导 `0.0`；有符号拍面引导 `0.0`。
- 改变的唯一变量 D3：有符号拍面引导权重 `-0.4`。
- 其余固定项：source、plant、动作、题库、seed、预算、初始化、显式环境、Kit thread cap 和
  `PYTHONPATH` 均相同；physical GPU lane 只属于运营 provenance。
- 决策规则：只有两格自然结束、`model_24.pt` finite/iter24/lineage1、各自绑定 outer claim 与相邻
  hard contract，且两份 hard contract 删除唯一 signed-face weight 后逐值相同，才发布 paired L1 receipt。
- 停止/无效规则：任一 claim/failure 已存在、GPU 非空、源码/环境/input 漂移、marker 不唯一、hard
  contract 非 31/31 zero、NaN/Inf/Traceback/OOM 或提前退出，均保留现场并停止；不自动重发。

## 组成与接口

- Launcher 把 `zero_joint_friction=true` 和 exact argv 同时写入 optimization recipe；claim 再绑定该配方、
  source、GPU lane、seed、终档迭代和 claim directory identity。
- Trainer source 在 `gym.make` 前把所有 actuator friction 设为 `0.0`，实例化后从真实运行时 hard
  contract 再检查 31/31 exact zero，检查通过后才会打印 marker 和 hard-contract path。
- Finalizer 重新构造命令与 claim，读取相邻 hard contract，检查 finite checkpoint、iter、lineage、
  contract SHA 和 claim SHA；它没有判卷入口。
- compact `question_bank` 仍严格为 trainer 实际发出的五键结构；physics SHA 独立从 exact NPZ metadata
  与 source-family contract 复算，不能伪造第六键。

## 运行表

| 运行（人话名 + `run_name`） | 状态 | Checkpoint/seed | 证据 | 结果产物 | 有效性说明 |
| --- | --- | --- | --- | --- | --- |
| 显式零摩擦 fresh 对照 `phase1_signed_face_l1_c3d3_v1_C3_fresh_control_seed3` | 自然终档 | seed3 / `model_24.pt` SHA `6b3e2cb1...70e7` | E2 runtime provenance | hard `d76dc944...ef2c`；terminal `8c579386...e8ef` | 31/31 零摩擦、finite/iter24/lineage1/claim binding 通过；尚未判 K100 |
| 显式零摩擦 fresh 引导 `phase1_signed_face_l1_c3d3_v1_D3_fresh_guidance_seed3` | 自然终档 | seed3 / `model_24.pt` SHA `44c6117c...85b8` | E2 runtime provenance | hard `98f6468f...34f4`；terminal `ccb9933c...7f0e` | 同门通过；只与 C3 相差 signed-face weight，尚未判 K100 |

## 决定

- 决定：`inconclusive`
- 理由：Pod1 GPU1/GPU2 上两条都只 claim 一次、各自 hard/zero-friction marker 唯一并自然到 iter24；paired
  receipt SHA `bb3cd749477861b1cd55f059ed3b23307784030dcad758db3a819c3c8a37bbde` 证明 provenance 完整，
  但还没有同一 immutable K100 行为结果。
- 是否已纳入当前 setting：`no`
- 局限/下一个 gate：不得重跑 C3/D3；下一步只允许先由 generic checkpoint attestor 绑定两份终档，再用
  同一 immutable signed-face K100 execution consumer 做 paired 行为判读。K100 前不得从 L1 reward 曲线
  晋级 L2、第二 seed 或 stop/promote。

### 同卷 K100 execution consumer source gate

新的 [`C3/D3 K100 paired execution consumer`](../../DEFINITIONS.md) 已把 paired receipt、两份
checkpoint/hard/producer-claim/terminal SHA、generic attestor、现有 K100 schedule/activation 和 exact float
`[1.0,-1.0]` 合同冻结到一个 one-shot manifest。它要求两份独立 attestation claim 都存在且可完整 replay，
再把 checkpoint-adjacent `env.yaml` 复制到独立 eval root，顺序运行同一卷；不写训练 run、不发 signal。

当前仍是 source gate：focused `28 passed`，`py_compile`、`static-validate`、`source-plan` rc0；没有 SSH、
attest 或 judge。执行步骤见[同卷 K100 操作](../../operations/run_phase1_signed_face_c3d3_k100.md)。consumer
产出的 paired count 只供后续人类 L2 decision contract 使用，本身仍把 L2、第二 seed、stop/promote、采用
setting、Gate3、部署和真机全部固定为 false。

### v1 运行失败与 ignored Isaac asset v2 修复

Pod1 v1 paired execution 已消费并冻结；C3 的 `judge.sh` 在 ONNX 导出阶段自然退出 `rc=1`。
exact traceback 是独立 eval checkout 缺少
`hope_training/.../assets/agibot_a3/urdf/model.urdf`。训练 checkout 中该文件为 43,240 bytes、
SHA-256 `79655f05d204c24f028778425aa971410773d1f8bbbd214de6fdb8f8ae75d1cc`，且同目录还有导出所需
meshes/config；整个目录被上游 `assets/.gitignore:*` 忽略，所以原 tracked-clean/source-SHA gate 没有
覆盖它。失败发生在 ONNX、MuJoCo 和 K100 attempt 之前，不是 C3 模型行为结论。

v1 output、attestation 与失败日志永久保留且不重放。v2 source gate 使用全新 attestor/pair namespace：
C3 attestation 先验证训练时实际 ignored `agibot_a3` 递归 canonical inventory、required URDF 和
`libGLU.so.1` 可加载性，再把闭包一次性 hydrate 到全新 eval checkout；D3 只能验证同一已存在闭包。
paired consumer 在 claim/judge 前重放两侧完整 asset evidence。focused `56 passed`，
`py_compile`、attestor static、pair static/source-plan 均 rc0；本 source gate 未连接 Pod、未 attest、
未 judge。运行真源见 [v2 操作](../../operations/run_phase1_signed_face_c3d3_k100_v2.md)。

随后用同一训练 A3 asset closure、exact checkpoint/bank/plant binding 做的 C3/D3 runtime diagnostic
两侧都成功导出 ONNX 并进入 MuJoCo，故 **asset packaging blocker 已关闭**。该方向筛显式使用
`--allow-inexact-contract`，日志诚实记录 `evaluation_contract_exact=false`；但该 flag 没有关闭
`formal_execution_contract_ok`，两侧仍在第 0 题前被同一 formal guard 拒绝：
`formal BankExam reached bound PhysX joint-velocity limit on articulation indices [8]; MuJoCo lacks same
braking constraint`。没有 attempt 或 score；这证明新的 blocker 是 articulation index `8` 的
velocity-limit braking parity，不是 C3/D3 guidance 行为差异。

只读复核路径为
`.../evaluations/diagnostic_asset_hydrated_inexact_v1/{C3,D3}/judge.runner.log`；原精确
PID `1873348/1873349` 均已退出。两份日志均为 `scheduled=50/side`、`asked=0`，因此 **C3/D3
方向分均不存在**，不能把进程成功导出或无结果解释成 `0/100`。

决定仍为 `inconclusive`，K100 behavior 状态为 `OPEN/BLOCKED`：v2 只修复运行打包合同；velocity-limit
parity 未闭合前不得执行正式 paired judge，也不授权 L2、第二 seed、stop/promote 或采用 setting。任何
明确 `allow-inexact` 的同卷方向筛都只能作为诊断，不能替代 formal K100。

后续 source 快审又关闭两项假绿：新 evaluator SHA 已逐级绑定到 attestor manifest 与 paired execution
manifest；hydrate 发布不再用可覆盖并发同名 child 的 `rename(2)`，而是 exclusive root/directory 加
`link(2)` 原子 no-replace。并发 sentinel 攻击保持 sentinel 与 stage 不变并 fail closed。focused
`57 passed`、两份 static 与 source-plan rc0；新的 velocity-proxy diagnostic 仍未运行，不能补写方向分。

## 复现与证据

操作真源：[运行 C3/D3 显式零摩擦 L1](../../operations/run_phase1_signed_face_c3d3_l1.md)。Pod1 runtime
使用文档中的 exact control/source 和一次性 root token；没有 judge、部署或真机命令。

```bash
python3 scripts/run_phase1_signed_face_c3d3_l1.py --mode static-validate
python3 scripts/run_phase1_signed_face_c3d3_l1.py --mode plan
python3 -m pytest -q tests/test_run_phase1_signed_face_c3d3_l1.py
python3 -m pytest -q tests
```

专项测试覆盖 declaration/argv 重复与反值、旧 namespace 环境 SHA 误复用、claim 配方漏绑、任意一个
非零/NaN/bool friction、compact-bank 第六键伪造、independent physics drift、no-clobber 和禁止隐藏
activation/judge/L2/retry/robot mode。结果：专项 `38 passed`；完整 `tests/` 回归
`972 passed, 10 skipped`；`static-validate` 与 plan 均 rc0。
