# Run S0/M0 exact GMR

本操作只处理 [`S0/M0`](../DEFINITIONS.md) canonical-beta PT 到 A3 [GMR](../DEFINITIONS.md) 的 CPU 离线
阶段。S0 是第五种反手高点拍压空挥；M0 是四条左右横移老师候选。它不运行 schema-2、仿真、
[TOPP](../DEFINITIONS.md)、RL、Gate3 或真机。设计与
当前结果与边界见[实验卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 0. 当前状态与边界

attempt v1 已永久 **NO-CONSUME**：它只保存了一个无法从保留证据复现的 pip-freeze hash。2026-07-13
22:53Z 的真实 S0 `inspect` 在 output root 创建前精确拒绝 `97c66009...18ff` vs 实际规范化
`56b0f8af...c694`；M0 未重复同一 shared blocker，两份 `exact_gmr_v1` root 均 absent。不得修改 v1
consumer/plan、重跑 v1 或复用 v1 root。

attempt v2 保留
原 16 项 exact GMR closure，并把 234 行、4,702 bytes 的规范化 pip snapshot 本身加入 Git；同时绑定
`numpy/torch/mujoco/smplx/scipy` 五个直接 import 的 version、origin 与 dist-info `METADATA/RECORD`。
实际复用的冻结 v1 base consumer 也有独立 bytes/SHA binding，plan/runtime JSON duplicate key 会 fail closed。
两份 host `static-v2` 已通过。2026-07-15 的特定 Pod2 checkout 两批 runtime `inspect-v2` 在 consumer 前共同遇到
rc127：合同绑定的 `/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10` 整棵环境不存在。
恢复审计还确认 exact GMR bundle/tree、SMPLX/model/mapping 和七份 S0/M0 canonical 输入也全部 absent；
现有 Isaac venv 只与冻结环境部分重合，不能替代。这只说明该 Pod2 路径失败；2026-07-20 回收的 live
evidence 已证明 S0/M0 v2 都在 Pod1 exact 路径完成，所以下面的 completions 为当前权威，不再写“两份
v2 root 都 absent / 未 consume”。

两份 completion 共同绑定 GMR commit `aabea2eee4be4bc16d4be17dac5ffa85e5a31539`、runtime
`a55c52cc...7db7b2`、S0 plan `0746291e...caf2f2` 与 M0 plan `a810ee01...2441f3`。

S0 completion 发布于 `2026-07-14T05:05:55.085040Z`，manifest 位于
`/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v2/completion_manifest.json`，SHA-256
`a762d6df22d4ffdcfc323425c234a0d3b910022d17a1541fa48ab7fe700d1a23`。唯一 `88`-frame 输出
`static_backhand_high_press...gmr.pkl` 的 SHA-256 为
`2dbe61e80af7187e9524b63095887287d2fd6aa615cbe9b712f68ea4dfc70edc`，finite/`30 Hz`/`31 DoF`
structural pass；但 ball contact/effectiveness 仍为 `null`，formal/schema2/training/hardware 全 false。
S0 不得重复 consume；下一门是独立高球拍压题族。

M0 completion 发布于 `2026-07-14T05:06:21.749762Z`，status=`complete_exact_gmr_diagnostic`，manifest
SHA-256=`fdd60fcfdc7290677aa51ec7804278568a267e239de548cdb623d0565dac396e`。四份输出全部通过
finite/`30 Hz`/`31 DoF` structural audit，SHA-256 分别为 left1 `3701837b...999895`、left2
`a21ab061...9b969`、right1 `e1cd0fca...a5f426`、right2 `4283cfe0...dfa7aa`；但 stance gate 为
`0/4`，且 formal/schema2/training/hardware 全 false。M0 不得重复 consume、进入 schema-2 或占 RL GPU。

direct retarget `a3_mocap.xml` 的完整 site inventory 是空列表，且 `left_foot/right_foot` 明确 absent；不得抄
canonical vendor MJCF 的足点去伪造 retarget site。M0 脚距只在 canonical vendor MJCF 的 `left_foot` 与
`right_foot` 上做 FK。

## 1. Host static

```bash
S0=configs/motion_exact_gmr_s0_prereg_20260714_v2.json
M0=configs/motion_exact_gmr_m0_prereg_20260714_v2.json
S0_SHA=$(shasum -a 256 "$S0" | awk '{print $1}')
M0_SHA=$(shasum -a 256 "$M0" | awk '{print $1}')

python3 scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" static
python3 scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" static
python3 -m pytest -q \
  tests/test_run_motion_s0_m0_exact_gmr.py \
  tests/test_run_motion_s0_m0_exact_gmr_v2.py
```

`static` 只读 Git 中的 prereg/tool/A3 tracked model tree，不访问私有 PT 或 ignored GMR。当前预期是两次
分别输出 `PASS static-v2 s0_static_high_press` 与 `PASS static-v2 m0_lateral_teachers`；任何 snapshot bytes、
direct-import binding、site absence、31-index bijection 或 tool SHA 漂移都必须 fail closed。

## 2. Runtime inspect（不写）

只在 exact source review 通过后，使用计划绑定的 CPU runtime：

```bash
export CUDA_VISIBLE_DEVICES=
PY=/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10

"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" inspect
"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" inspect
```

`inspect-v2` 验 commit/tree/clean status、bundle、converter/import/model closure、direct retarget 的空 site
inventory、完整 pip snapshot bytes、五个 direct import 的 origin/version/METADATA/RECORD、canonical-beta
completion 与五条 exact PT，并确认各自 `exact_gmr_v2` root 不存在。它设置
`PYTHONDONTWRITEBYTECODE=1`，不创建 output/state/report。S0/M0 分开 inspect，一批失败不改变另一批。
若 shared consume lock 已存在，inspect 只读验证它是 single-link regular file 且 marker 精确；lock absent 也
合法，inspect 绝不创建 lock。
这两条命令曾在 Pod2 真实尝试，但都在解释器启动前按上节 rc127 结束，未进入 consumer；这是保留的
失败 location。Pod1 completions 后来回收并闭合了两批结果，因此现在不得用这些命令重新 inspect 已完成
namespace。新动作版本必须使用新的 prereg/root，并重新绑定 exact runtime。

## 3. 一次性 consume

以下命令是冻结的历史 v2 consume 形状，**现在不得执行**：Pod1 上 S0/M0 两个 namespace 都已有 exact
completion，no-clobber 纪律禁止重跑。它们只用于解释原计划，不能作为新版本动作的 launch recipe。

```bash
"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" consume
"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" consume
```

输出 root 分别为：

- `/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v2`；
- `/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v2`。

首次 consume 时 root 必须原先不存在；现在两个 root 都已有 authoritative completion。每个 root 内先出现
`outputs/logs/audits/bindings`，所有内容完整并 fsync 后才最后出现
`completion_manifest.json`。失败/超时或中断留下的是不可覆盖证据；不得删除后用同一 plan 重跑，需另建
版本化 prereg/root。v2 在所有 converter child 退出后、completion 发布前再次验证 GMR/input/model 与
Python snapshot/direct-import closure；post-check 失败必须保留 partial 且没有 completion。

冻结 consumer 要求 S0/M0 两条 `consume` 串行运行；它用
`/workspace/codexschema/motion_s0_m0_exact_gmr_v2.consume.lock` 的 exclusive flock 做代码级互斥。首次
consume 在持锁时写固定 marker；后续只接受 exact marker。这个互斥不要求 S0 成功后 M0 才能运行，也不把
一批的失败写进另一批 root。两批现已完成，此段只解释证据生成合同，不授权重跑。

## 4. 结果边界

S0 completion 已写 `observed_ball_contact=null`、`strike_effectiveness=null`，不能拿拉球题判拍压；下一门
必须是独立高球拍压题族。M0 每条都含 exact foot-site mapping、ready sample indices、initial/terminal
二维 `d_xy`、前后/横向分量、四项子检查与 `stance_passed`，结果为 `0/4`：

- left1 lateral `+0.095425 m`，narrowing `0.095425 m`；
- left2 lateral `-0.200557 m`；
- right1 lateral `+0.076532 m`，narrowing `0.076532 m`；
- right2 lateral `+0.024300 m` 在 3 cm band 内，但 narrowing `0.024300 m > 0.005 m`。

这四条证明 moving reference 存在，但都不能在移动后回到自身初始 stance 且保持脚距。因此 M0 input gate
为 reject/no-launch；修复版必须保留横移、修复末态 stance，再走新的 exact GMR。

completion 只表示 exact GMR diagnostic 完成；`formal/schema2/training/hardware` 仍全 false。S0 完成高球
题族、M0 完成动作修复后，才分别另建 schema-2、L0/L1、桌网整轨和动力学 prereg。
