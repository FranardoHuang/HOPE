# Run S0/M0 exact GMR

本操作只处理 [`S0/M0`](../DEFINITIONS.md) canonical-beta PT 到 A3 [GMR](../DEFINITIONS.md) 的 CPU 离线
阶段。S0 是第五种反手高点拍压空挥；M0 是四条左右横移老师候选。它不运行 schema-2、仿真、
[TOPP](../DEFINITIONS.md)、RL、Gate3 或真机。设计与
当前 source/static 状态见[实验卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 0. 当前状态与边界

attempt v1 已永久 **NO-CONSUME**：它只保存了一个无法从保留证据复现的 pip-freeze hash。2026-07-13
22:53Z 的真实 S0 `inspect` 在 output root 创建前精确拒绝 `97c66009...18ff` vs 实际规范化
`56b0f8af...c694`；M0 未重复同一 shared blocker，两份 `exact_gmr_v1` root 均 absent。不得修改 v1
consumer/plan、重跑 v1 或复用 v1 root。

当前可审阅入口是 attempt v2：两份 batch plan 与共享 runtime 均为 `preregistered_not_executed`。它保留
原 16 项 exact GMR closure，并把 234 行、4,702 bytes 的规范化 pip snapshot 本身加入 Git；同时绑定
`numpy/torch/mujoco/smplx/scipy` 五个直接 import 的 version、origin 与 dist-info `METADATA/RECORD`。
实际复用的冻结 v1 base consumer 也有独立 bytes/SHA binding，plan/runtime JSON duplicate key 会 fail closed。
两份 host `static-v2` 已通过。2026-07-15 Pod2 的两批 runtime `inspect-v2` 在 consumer 前共同遇到
rc127：合同绑定的 `/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10` 整棵环境不存在。
恢复审计还确认 exact GMR bundle/tree、SMPLX/model/mapping 和七份 S0/M0 canonical 输入也全部 absent；
现有 Isaac venv 只与冻结环境部分重合，不能替代。两个 output root 与 shared lock 仍 absent；不得用别的
Python 或猜测重建 v2，也不得运行 `consume-v2`。先按
[本地同步操作](setup_local_sync.md)从权威备份恢复内容寻址资产，再另建隔离 v3 runtime/plan。

direct retarget `a3_mocap.xml` 的完整 site inventory 是空列表，且 `left_foot/right_foot` 明确 absent；不得抄
canonical vendor MJCF 的足点去伪造 retarget site。M0 脚距只在 canonical vendor MJCF 的 `left_foot` 与
`right_foot` 上做 FK。当前仍未用 v2 运行 ignored runtime `inspect`、私有 PT、converter 或 `consume`，
所以没有 GMR 输出，也未解锁 schema-2/训练/真机。

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
这两条命令已在 Pod2 真实尝试，但都在解释器启动前按上节 rc127 结束，未进入 consumer。只有 exact
runtime 恢复，或另建并审过一个完整绑定 interpreter/package origins 的 v3，才可重新获得 inspect 权限。

## 3. 一次性 consume

只有相应 batch 的只读 `inspect` 已通过、且输出 root 仍不存在时，才可授权以下命令。本次 source 闭环
**没有执行** `consume`。

```bash
"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" consume
"$PY" scripts/run_motion_s0_m0_exact_gmr_v2.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" consume
```

输出 root 分别为：

- `/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v2`；
- `/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v2`。

root 必须原先不存在。每个 root 内先出现 `outputs/logs/audits/bindings`，所有内容完整并 fsync 后才最后出现
`completion_manifest.json`。失败/超时或中断留下的是不可覆盖证据；不得删除后用同一 plan 重跑，需另建
版本化 prereg/root。v2 在所有 converter child 退出后、completion 发布前再次验证 GMR/input/model 与
Python snapshot/direct-import closure；post-check 失败必须保留 partial 且没有 completion。

S0/M0 两条 `consume` 必须按上面的命令串行运行；consumer 还用
`/workspace/codexschema/motion_s0_m0_exact_gmr_v2.consume.lock` 的 exclusive flock 做代码级互斥。首次
consume 在持锁时写固定 marker；后续只接受 exact marker。这个互斥不要求 S0 成功后 M0 才能运行，也不把
一批的失败写进另一批 root。

## 4. 结果边界

S0 completion 必须继续写 `observed_ball_contact=null`、`strike_effectiveness=null`，不能拿拉球题判拍压。
M0 每条必须含 exact foot-site mapping、ready sample indices、initial/terminal 二维 `d_xy`、前后/横向分量、
四项子检查与 `stance_passed`。任一末态横向分离比初态缩小超过 5 mm，即使仍在 3 cm component band 内，
也必须失败。

completion 只表示 exact GMR diagnostic 完成；`formal/schema2/training/hardware` 仍全 false。下一步必须另建
schema-2、L0/L1、桌网整轨和动力学 prereg。
