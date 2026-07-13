# Run S0/M0 exact GMR

本操作只处理 [`S0/M0`](../DEFINITIONS.md) canonical-beta PT 到 A3 [GMR](../DEFINITIONS.md) 的 CPU 离线
阶段。S0 是第五种反手高点拍压空挥；M0 是四条左右横移老师候选。它不运行 schema-2、仿真、
[TOPP](../DEFINITIONS.md)、RL、Gate3 或真机。设计与
当前阻塞见[实验卷宗](../experiments/motion_exact_gmr_s0_m0_20260713.md)。

## 0. 当前阻塞

两份 batch plan 已是 `preregistered_not_executed`；共享 runtime 仍是
`blocked_pending_exact_ignored_gmr_source_closure`。2026-07-14 的只读回执已绑定 tree OID、retarget
MJCF/mapping bytes/SHA、关键 import SHA、Python binary SHA 和规范化 pip-freeze SHA，但 direct retarget XML
的 joint/body/site parser 段被传输截断。另缺 import/mapping 的完整绝对路径、三个 eager-import 文件 binding、
Python path/bytes、完整 pip origin string 与 `xrobot_utils` resolution。

机器真源 `configs/motion_s0_m0_exact_gmr_runtime_20260713.json` 的
`required_unresolved_evidence` 含 16 个 JSON pointer 和下一条只读 probe。不得抄 canonical 32-body 列表填
retarget XML，不得使用历史 31-joint 列表冒充 direct parser receipt。补齐并 code review 后，只把共享 runtime
切为 `preregistered_not_executed`，移除 blocked-only receipt/list，重算全部 binding，下面 runtime 命令才适用。

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

`static` 只读 Git 中的 prereg/tool/A3 tracked model tree，不访问私有 PT 或 ignored GMR。当前预期是两次均
以 rc=2 打印同一份 `required_unresolved_evidence`；若它输出 `PASS` 反而是错误。补齐 closure 后才应分别
输出 `PASS static`。

## 2. Runtime inspect（不写）

只在 exact source review 通过后，使用计划绑定的 CPU runtime：

```bash
export CUDA_VISIBLE_DEVICES=
PY=/exact/path/from-reviewed-runtime-contract/python

"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$S0" --expected-plan-sha256 "$S0_SHA" inspect
"$PY" scripts/run_motion_s0_m0_exact_gmr.py \
  --plan "$M0" --expected-plan-sha256 "$M0_SHA" inspect
```

`inspect` 验 commit/tree/clean status、bundle、converter/import/model closure、Python/pip、canonical-beta
completion 与五条 exact PT，并确认各自 output root 不存在。它设置 `PYTHONDONTWRITEBYTECODE=1`，不创建
output/state/report。S0/M0 分开 inspect，一批失败不改变另一批。

## 3. 一次性 consume

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
