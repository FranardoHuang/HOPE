# Run S0/M0 exact GMR

本操作只处理 [`S0/M0`](../DEFINITIONS.md) canonical-beta PT 到 A3 [GMR](../DEFINITIONS.md) 的 CPU 离线
阶段。S0 是第五种反手高点拍压空挥；M0 是四条左右横移老师候选。它不运行 schema-2、仿真、
[TOPP](../DEFINITIONS.md)、RL、Gate3 或真机。设计与
当前 source/static 状态见[实验卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 0. 当前状态与边界

两份 batch plan 与共享 runtime 均为 `preregistered_not_executed`。16 项只读 closure 已绑定 exact GMR
commit/tree、十二个 runtime 文件、direct retarget XML 的 31-joint/32-body preorder、显式 qpos bijection、
Python/pip 与 `xrobot_utils=absent`；两份 host `static` 已通过。

direct retarget `a3_mocap.xml` 的完整 site inventory 是空列表，且 `left_foot/right_foot` 明确 absent；不得抄
canonical vendor MJCF 的足点去伪造 retarget site。M0 脚距只在 canonical vendor MJCF 的 `left_foot` 与
`right_foot` 上做 FK。当前仍未运行 ignored runtime `inspect`、私有 PT、converter 或 `consume`，所以没有
GMR 输出，也未解锁 schema-2/训练/真机。

## 1. Host static

```bash
S0=configs/motion_exact_gmr_s0_prereg_20260713.json
M0=configs/motion_exact_gmr_m0_prereg_20260713.json
S0_SHA=$(shasum -a 256 "$S0" | awk '{print $1}')
M0_SHA=$(shasum -a 256 "$M0" | awk '{print $1}')

python3 scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" static
python3 scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" static
python3 -m pytest -q tests/test_run_motion_s0_m0_exact_gmr.py
```

`static` 只读 Git 中的 prereg/tool/A3 tracked model tree，不访问私有 PT 或 ignored GMR。当前预期是两次
分别输出 `PASS static s0_static_high_press` 与 `PASS static m0_lateral_teachers`；任何 binding、site absence、
31-index bijection 或 tool SHA 漂移都必须 fail closed。

## 2. Runtime inspect（不写）

只在 exact source review 通过后，使用计划绑定的 CPU runtime：

```bash
export CUDA_VISIBLE_DEVICES=
PY=/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10

"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" inspect
"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" inspect
```

`inspect` 验 commit/tree/clean status、bundle、converter/import/model closure、direct retarget 的空 site
inventory、Python/pip、canonical-beta completion 与五条 exact PT，并确认各自 output root 不存在。它设置
`PYTHONDONTWRITEBYTECODE=1`，不创建 output/state/report。S0/M0 分开 inspect，一批失败不改变另一批。
本次 source 闭环没有执行这两条命令；必须在 code review 后另行运行并记录结果。

## 3. 一次性 consume

只有相应 batch 的只读 `inspect` 已通过、且输出 root 仍不存在时，才可授权以下命令。本次 source 闭环
**没有执行** `consume`。

```bash
"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" consume
"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" consume
```

输出 root 分别为：

- `/workspace/codexschema/motion_video_intake_20260713_s0/exact_gmr_v1`；
- `/workspace/codexschema/motion_video_intake_20260713_m0/exact_gmr_v1`。

root 必须原先不存在。每个 root 内先出现 `outputs/logs/audits/bindings`，所有内容完整并 fsync 后才最后出现
`completion_manifest.json`。失败/超时或中断留下的是不可覆盖证据；不得删除后用同一 plan 重跑，需另建
版本化 prereg/root。

## 4. 结果边界

S0 completion 必须继续写 `observed_ball_contact=null`、`strike_effectiveness=null`，不能拿拉球题判拍压。
M0 每条必须含 exact foot-site mapping、ready sample indices、initial/terminal 二维 `d_xy`、前后/横向分量、
四项子检查与 `stance_passed`。任一末态横向分离比初态缩小超过 5 mm，即使仍在 3 cm component band 内，
也必须失败。

completion 只表示 exact GMR diagnostic 完成；`formal/schema2/training/hardware` 仍全 false。下一步必须另建
schema-2、L0/L1、桌网整轨和动力学 prereg。
