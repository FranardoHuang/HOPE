# 运行 signed-face C3/D3 显式零摩擦 L1

状态：**source/static gate 已就绪；Pod runtime 尚未运行。** 本页只处理一对 fresh
[`L1`](../DEFINITIONS.md) 小臂：`C3` 是关闭有符号拍面引导的显式零摩擦对照，`D3` 只把引导权重改为
`-0.4`。两格都是 seed3、`512 env × 25 update`；`L1` 仅是发射与合同冒烟，不是行为判卷。

旧 C2 虽在 manifest 写了零摩擦，却没有在 argv 传
`task.plant.zero_joint_friction=true`，实际 hard contract 的 31 个 PhysX 关节摩擦系数全部非零。
因此旧 C2 checkpoint 不采用，D2 v1r2 永久 `NO-LAUNCH`；不得修改/复用旧 control、claim、run 或
checkpoint。本页使用全新的 source checkout、artifact root、run name、atomic claim 与 evidence。
详细因果记录见
[C3/D3 实验卷宗](../experiments/2026-07/EXP-P1-SIGNED-FACE-C3D3-ZERO-FRICTION-L1.md)。

## 冻结输入

- 训练 source：commit `4467d79f1ed425a4263f0caaad2f661e1ec737ad`，tree
  `497db1d8f2d7fb1b554337928f098a2951d4cf0d`。
- Manifest：[`phase1_signed_face_c3d3_l1_prereg_20260714.json`](../../configs/phase1_signed_face_c3d3_l1_prereg_20260714.json)，
  SHA-256 `eefc8023786a3bc90f86bc0803dda69003c76cc3076bc089a15dec618f435dc2`。
- Launcher/finalizer：[`run_phase1_signed_face_c3d3_l1.py`](../../scripts/run_phase1_signed_face_c3d3_l1.py)，
  SHA-256 `192148906a893e8b55e9a9cf666c3557e14ce8d0b53f2337b2e311e2430aa628`。
- Runtime source checkout：`/workspace/codexschema/nohope_signed_face_c3d3_l1_4467d79`，必须 clean detached
  exact source；不得修改归档训练 checkout。
- External control root：`/workspace/codexschema/phase1_signed_face_c3d3_l1_20260714/control/v1`；只允许
  本 manifest 和 launcher 两个 single-link read-only regular file。
- Artifact/run root：`/workspace/codexschema/phase1_signed_face_c3d3_l1_20260714/runs/l1`。
- C3 固定 Pod1 GPU1；D3 固定 Pod1 GPU2。每格 claim 前自己的 GPU 必须没有任何 compute PID。

## 为什么这版不能再“声明零摩擦但实际没传”

1. manifest 只接受一个 exact argv leaf：`task.plant.zero_joint_friction=true`；缺失、false、使用 `++`
   重复添加或出现第二个同 leaf 都 fail closed。
2. plan 中两条完整 command 都必须恰好含一次该 leaf。
3. outer optimization recipe 与 atomic claim 均显式记录 declarative bool 和 exact argv 字符串。
4. 训练 source 在 scene 实例化后检查实际 hard contract 必须是 31/31 finite zero，并打印一次
   `ZERO_FRICTION_RUNTIME_OK`；launcher 同时复核 marker 和合同值。
5. checkpoint 必须绑定该相邻 hard-contract SHA 和 outer claim SHA；pair finalizer 要求两份合同除
   signed-face weight 外完全相同。

## 本地 source gate

下面命令均不连接 Pod、不写 runtime artifact：

```bash
python3 scripts/run_phase1_signed_face_c3d3_l1.py --mode static-validate
python3 scripts/run_phase1_signed_face_c3d3_l1.py --mode plan
python3 -m pytest -q tests/test_run_phase1_signed_face_c3d3_l1.py
python3 -m pytest -q tests
```

预期：`static_valid_no_writes`；plan 的 `writes_or_launches_performed=false`，C3/D3 command 各含且仅含
一次 exact zero-friction leaf，所有 activation/judge/L2/second-seed/stop-promote/retry/robot 开关为 false。
本 source gate 的实际结果为：专项 `38 passed`，完整 `tests/` 回归 `972 passed, 10 skipped`，
`static-validate` 与 plan 均 rc0。

## Runtime 前置与安装边界

只有本 source gate 进入 main 后，root 才能从 exact main bytes 建立独立 detached source 和只读 control。
安装不是发射；先逐字节复核上面的两个 SHA。任何目标文件/目录已经存在都必须停止，不得覆盖。

运行 `validate-runtime` 前必须重新确认：

- source checkout、IsaacLab、Python/pip、critical files、ignored A3 asset、动作、bank 和 rebind report 均与
  manifest 完全一致；
- 新 artifact root 下 C3/D3 arm、claim、failure、result 与 exact training `run_name` 均 absent；
- Pod1 GPU1/GPU2 当前均无 compute PID，host available RAM 高于 manifest 下限；
- 网络 timeout 只能记为 `UNKNOWN`，不得据此重复安装或 claim；
- 不读取旧 snapshot 作为当前 absence/GPU 授权。

在 external control root 中执行：

```bash
./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode static-validate

./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode plan

./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode validate-runtime
```

任一步失败都停止；source gate 本身不授权 root 跳过 runtime preflight。

## 一次性发射

每次 invocation 只原子 claim 一格。第一次只可能创建 C3；C3 写出 `runtime_verified.json` 后，第二次才
可能在 distinct GPU 创建 D3。Kit boot 由 host-wide lock 串行，lock 释放后两格训练可并发。不得等到
terminal 后把同一 namespace 当 retry 使用。

```bash
./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode launch-next \
  --root-confirm ROOT_APPROVES_SIM_ONLY_SIGNED_FACE_C3_D3_EXPLICIT_ZERO_FRICTION_L1_V1
```

第一次成功后重新做只读进程/GPU/日志检查，再执行同一命令一次来 claim D3。只管理 launcher state 中
记录的 exact numeric PID/PGID；本工具没有 broad process discovery 或 signal path。禁止 `pkill`、
`killall`、`pgrep -f` 信号和任何真机命令。

## 监控与失败处理

每格检查 exact PID/PGID、GPU、RAM、最新 iteration/checkpoint、NaN/Inf/Traceback/OOM/malloc/Killed、
hard-contract marker、唯一 zero-friction marker 和 sidecar binding。L1 预算没有 +200/+500/+1000 checkpoint；
它只期望自然形成 `model_24.pt`，不能拿 reward 曲线 stop/promote。

若 claim 后任一步失败：保留 arm、launch contract、state、log 和 failure（若已物化）；不覆盖、不自动
retry、不迁移到旧 C2/D2 namespace。先诊断合同或环境；只有另行预注册的新科学问题才可购买新 namespace。

## 终档与 paired receipt

只有 exact trainer 已自然退出且 assigned GPU 为空，才逐格运行：

```bash
./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode finalize-cell --cell C3

./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode finalize-cell --cell D3

./run_phase1_signed_face_c3d3_l1.py \
  --manifest ./phase1_signed_face_c3d3_l1_prereg_20260714.json \
  --mode finalize-pair
```

Finalizer 必须证明 `model_24.pt` 文件名/embedded iter=24、全部浮点 tensor finite、lineage=1、checkpoint
的 hard-contract/claim SHA 与外层记录完全一致。paired receipt 仍固定 activation/judge/L2/第二 seed=false；
后续必须使用同一 immutable signed-face paper 的独立 execution consumer 才能做机制判断。
